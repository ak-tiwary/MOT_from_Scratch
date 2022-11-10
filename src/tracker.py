import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian_algorithm
from detector import Detector
from track import Track


#TODO: Add appearance information
# class LostTrack:
#     """Wrapper class for Track also handling appearance information?"""
#     def __init__(self, track, appearance=None):
#         self.track = track #Track object
#         self.appearance = appearance #keep track of appearance at the step when lost.
        

class Tracker:
    def __init__(self, det, track_expiry_time=50, low_conf_threshold=0.5, min_conf_threshold=0.2):
        """The main tracker that handles a single step of tracking.
        
        Args:
            det (Detector): A detector object initialized already with desired settings.
            track_expiry_time (int): Number of frames to wait before expiring a track as lost
                                     and starting a new track.
            low_conf_threshold (float): The threshold below which a bbox is considered 
                                        "low" confidence.
            min_conf_threshold (float): The minimum confidence threshold below which boxes
                                        are not to be considered
        """
        
        
        self.detector = det
        self.detector.test_conf = min_conf_threshold
        self.alive_tracks = []
        self.tracks_on_hold = []
        self.id_ctr = 0
        self.appearance_model = None #TODO
        self.low_conf_threshold = low_conf_threshold
        

        
    def step(self, frame):
        """Track objects in new frame and update tracks.
        
        Args:
            frame (np.ndarray): The frame as read by opencv and meant to be passed to the detector."""
            
        #bboxes is Nx7
        bboxes, img_info = self.detector(frame)
        
        #box_coords = bboxes[:, :4]
        
        confidences = bboxes[:, 4] * bboxes[:, 5]
        mask = confidences > self.low_conf_threshold
        
        high_conf_boxes = bboxes[mask]
        low_conf_boxes = bboxes[~mask]
        
        high_conf_box_coords = high_conf_boxes[:4]
        low_conf_box_coords = low_conf_boxes[:4]
        
        self.alive_tracks
        
        