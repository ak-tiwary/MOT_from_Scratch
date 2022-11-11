import copy
import numpy as np
from filterpy.kalman import KalmanFilter


#TO ADD:
#1) handle short term and long term recovery differently
#for example use IoU more in short term recovery and appearance more in long term?

#2) handle camera motion compensation frame to frame when performing observation
#centric recovery


class Track:
    """A single track, containing information about one box
    over multiple frames."""
    def __init__(self, bbox, id):
        """Initialize track with bounding box in xywh format. We track xywh itself in
        the Kalman Filter following BoT-Sort (2022) instead of the standard xysa.
        

        Args:
            bbox (np.array): an array of length 7, [x,y,w,h, obj conf, 
                             class conf, class pred]
            id (int) : ID number of this track
        """
        
        box_coordinates = bbox[:4]
        self.class_category = bbox[-1]
        
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.x = box_coordinates
        self.kf.P = np.diag([1., 1, 1, 1, 1000, 1000, 1000, 1000]) #unsure of velocities at start
        
        #state transition matrix is [[1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0],...] as usual
        np.put(self.kf.F, [x * 8 + x + 4 for x in range(4)], np.ones(4))

        #measurement function is just the first four out of the 8 state variables
        np.put(self.kf.H, [x * 8 + x for x in range(4)], np.ones(4))
        
        #process noise represents deviation of real motion from estimated motion in KF
        #since a linear motion model is a good approximation at low velocities,
        #we assign a low process noise
        self.kf.Q = np.diag([1., 1., 1., 1., .01, .01, .01, .01]) 
        
        #measurement noise.
        #center is easier to estimate than width and height
        self.kf.R = np.diag([1., 1., 10., 10.])
        
        
        self.time_since_last_detection = 0
        self.kf_at_last_detection = None 
        self.id = id
        
        
    def predict(self):
        """Predicts the location of the box using the kalman filter and returns box coordinates."""
        
        #just lost the track, so keep a copy of the KF before future predictions
        if self.time_since_last_detection == 1:
            self.kf_at_last_detection = copy.deepcopy(self.kf)
            
        self.time_since_last_detection += 1
            
        self.kf.predict()
            
        return self.get_box_coordinates()
    
    def update(self, box):
        """Updates the kalman filter with the observation in 1d np array box.
        Box is either xywh coordinates, or xywh+3 coordinates."""
        
        assert len(box) in [4,7], "box should have length 4 or 7"
        
        if self.time_since_last_detection > 1:
            self.observation_centric_recovery(box, self.time_since_last_detection)
            
        if len(box) == 7:
            box = box[:4]
         
            
        self.kf.update(box)
        
        self.time_since_last_detection = 0
        
    def observation_centric_recovery(self, box, num_steps):
        """Reverts the update steps of the kalman filter to the last observation and
        adds new "observations" that are linear interpolations from the last observation
        to the new observation."""
        
        kf = self.kf_at_last_detection
        
        old_box = self.get_box_coordinates(kf)
        diff_box = box - old_box
        step_size = 1./num_steps
        for i in range(num_steps-1):
            box_i = old_box + (i+1) *step_size * diff_box
            kf.predict()
            kf.update(box_i)
        kf.predict()
        kf.update(box)
        self.kf = kf
        self.kf_at_last_detection = None
        
        
    def get_box_coordinates(self, kf=None):
        """Returns box coordinates"""
        if kf is not None:
            return kf.x[:4]
        return self.kf.x[:4]

