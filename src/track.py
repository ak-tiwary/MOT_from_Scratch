import copy
import numpy as np
from filterpy.kalman import KalmanFilter
from collections import defaultdict
from functools import partial
from loguru import logger




#TO ADD:
#1) handle short term and long term recovery differently
#for example use IoU more in short term recovery and appearance more in long term?

#2) handle camera motion compensation frame to frame when performing observation
#centric recovery

#3) Add variation in Q and R noise matrices for kalman filter to depend on confidence of the detection (NSA Kalman Filter)
#4) Replace IoU with GIoU
#5) Replace hungarian algorithm with Jonker-Volgenant for linear assignment 



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
        _,_,w,h = box_coordinates
        self.box_conf = box_conf
        self.class_category = class_category
        
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        self.kf.x[:4,0] = box_coordinates
        
        
        #initial values for covariance and noise matrices taken from BOT-SORT
        self._std_weight_position = 1./20
        self._std_weight_velocity = 1./160
        std = [
            2 * self._std_weight_position * w,
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * w,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * w,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * w,
            10 * self._std_weight_velocity * h]
        
        self.kf.P = np.diag(np.square(std))
        
        #state transition matrix is [[1,0,0,0,1,0,0,0], [0,1,0,0,0,1,0,0],...] as usual
        np.put(self.kf.F, [x * 8 + x + 4 for x in range(4)], np.ones(4))

        #measurement function is just the first four out of the 8 state variables
        np.put(self.kf.H, [x * 8 + x for x in range(4)], np.ones(4))
        
        #process noise represents deviation of real motion from estimated motion in KF
        std = [
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_velocity * w,
            self._std_weight_velocity * h,
            self._std_weight_velocity * w,
            self._std_weight_velocity * h]
        self.kf.Q = np.diag(np.square(std)) 
        
        #measurement noise.
        std = [
            self._std_weight_position * w,
            self._std_weight_position * h,
            self._std_weight_position * w,
            self._std_weight_position * h]
        self.kf.R = np.diag(np.square(std))
        self.dt = dt
        
        
        self.time_since_last_detection = 0
        self.kf_at_last_detection = None #for OC-Smoothing, we revert KF to this state
        self.last_observation = bbox
        self.observation_before_last = None
        self.obs_streak = 1 #streak of matched boxes. Currently we have 1 match streak.
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
            
            
            #we just lost the track, so we keep a copy of the KF before future steps
            if self.time_since_last_detection == 0:
                self.kf_at_last_detection = copy.deepcopy(self.kf)
                
            self.kf.update(None)
            self.obs_streak = 0
            
            self.time_since_last_detection += 1
            return
        
        assert box.size == 4, "box should have size 4"
        #we missed a detection somewhere and just recovered so perform recovery of Kalman filter
        if self.time_since_last_detection >= 1:
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
        for i in range(num_steps):
            box_i = old_box + (i+1) *step_size * diff_box
            kf.predict()
            kf.update(box_i)
        
        self.kf = kf
        self.kf_at_last_detection = None
        
        
    def get_box_coordinates(self, kf=None):
        """Returns box coordinates"""
        if kf is not None:
            return kf.x[:4].squeeze()
        return self.kf.x[:4].squeeze()




def normalize(v, eps=1e-12):
    """Given a 1d numpy array v, normalizes it to have unit length (or zero)"""
    norm = np.linalg.norm(v)
    return v / (norm + eps)