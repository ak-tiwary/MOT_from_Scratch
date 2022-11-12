import copy
import numpy as np
from filterpy.kalman import KalmanFilter
from collections import defaultdict
from functools import partial




#TO ADD:
#1) handle short term and long term recovery differently
#for example use IoU more in short term recovery and appearance more in long term?

#2) handle camera motion compensation frame to frame when performing observation
#centric recovery


class Track:
    """A single track, containing information about one box
    over multiple frames."""
    def __init__(self, bbox, id, box_conf=None, class_category=None, dt=3):
        """Initialize track with bounding box in xywh format. We track xywh itself in
        the Kalman Filter following BoT-Sort (2022) instead of the standard xysr.
        

        Args:
            bbox (np.array): an array of length 4, [x,y,w,h]
            id (int) : ID number of this track
            box_conf (float | None): Optional box confidence
            class_category (float | None): Optional class category
            dt (int):  Frame interval to use for velocity calculations. Defaults to 3, as suggested by OC-Sort.
        """
        
        box_coordinates = bbox
        self.box_conf = box_conf
        self.class_category = class_category
        
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
        self.kf_at_last_detection = None #for OC-Smoothing, we revert KF to this state
        self.last_observation = bbox
        self.observation_before_last = None
        self.obs_streak = 1 #streak of matched boxes
        self.observation_history = defaultdict(partial(np.ndarray, 0))
        self.id = id
        self.time_lapsed = 0
        self.observation_history[self.time_lapsed] = bbox
        self.velocity_direction = np.array([0,0])
        
        
    def predict(self):
        """Predicts the location of the box using the kalman filter and returns box coordinates.
        Assumed that consecutive predictions will not be called without an update(None) in between."""
         
            
        self.kf.predict()
            
        return self.get_box_coordinates()
    
    def update(self, box):
        """Updates the kalman filter with the observation in 1d np array box.
        Box is either xywh coordinates, or xywh+3 coordinates. If the box is None, will
        update the kalman filter's posteriors only. It is assumed that update is called
        even when there is no matched box with None as parameter."""
        self.time_lapsed += 1
        
        if box is None:
            #there was no matching at this step.
            self.kf.update(None)
            self.obs_streak = 0
            
            #we just lost the track, so we keep a copy of the KF before future steps
            if self.time_since_last_detection == 0:
                self.kf_at_last_detection = copy.deepcopy(self.kf)
            
            self.time_since_last_detection += 1
            return
        
        
        if self.time_since_last_detection > 1:
            self.observation_centric_recovery(box, self.time_since_last_detection)
        
            
        if len(box) == 7:
            box = box[:4]
         
       
        self.kf.update(box)
        
        self.observation_history[self.time_lapsed] = box
        self.obs_streak += 1

        self.observation_before_last = self.last_observation
        self.last_observation = box
        self.time_since_last_detection = 0
        self._update_velocity()
        
    def get_obs_from_dt_frames_back(self, give_obs_before_last=False):
        """Returns the first observation that occurred in the last dt frames. If the flag is True,
        will return the observation before last when no observation was found between dt frames ago and 1 frame ago."""
        prev_box = None
        flag = True
        for i in range(self.dt, 0, -1):
            prev_box = self.observation_history[self.time_lapsed - i]
            if prev_box.size: #if we have this particular observation
                flag = False
                break
        if flag: #no observations in the last dt frames
            return self.observation_before_last if give_obs_before_last else self.last_observation
        return prev_box
    
    def _update_velocity(self):
        """Updates the velocity prediction"""
        curr_box = self.last_observation

        prev_box = self.get_obs_from_dt_frames_back(give_obs_before_last=True)
               
        self.velocity_direction = normalize((curr_box - prev_box)[:2])
        
        
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




def normalize(v, eps=1e-12):
    """Given a 1d numpy array v, normalizes it to have unit length (or zero)"""
    norm = np.linalg.norm(v)
    return v / (norm(v) + eps)