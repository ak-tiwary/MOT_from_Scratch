import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian_algorithm
from detector import Detector
from track import Track
from visualizer import TrackVisualizer
from tools import get_iou_matrix
import time


#TODO: Add appearance information
# class LostTrack:
#     """Wrapper class for Track also handling appearance information?"""
#     def __init__(self, track, appearance=None):
#         self.track = track #Track object
#         self.appearance = appearance #keep track of appearance at the step when lost.
#TODO: Camera Motion Compensation
#Make Q and R be time dependent following BoT-SORT


#BASIC ALGORITHM:
#
# 1. Take in an image -> detector -> detected boxes. Normalize the boxes.
# 2. Divide boxes into high and low confidence boxes.
# 3. The high confidence boxes are matched first with the predictions of the tracks.
# 4. Next the remaining tracks are matched with the low confidence boxes (maybe occluded objects)
# 5. Finally the *last observations* (OC-recovery) of the remaining tracks are matched with the
#    remaining high confidence boxes.
# 6. Any remaining high confidence boxes are used to create new tracks.
# 7. Tracks are updated with new boxes.
# 8. All the tracks that have a streak of detections > threshold (=3) are considered "proper"
#   tracks and are returned/added to the image.     
        
    

class Tracker:
    def __init__(self, det, track_expiry_time=50, low_conf_threshold=0.6, min_conf_threshold=0.2,
                 iou_threshold=0.2):
        """The main tracker that handles a single step of tracking. For MOT20 the low_conf_thres
        should be 0.6 according to OC-SORT.
        
        Args:
            det (Detector): A detector object initialized already with desired settings.
            track_expiry_time (int): Number of frames to wait before expiring a track as lost
                                     and starting a new track.
            low_conf_threshold (float): The threshold below which a bbox is considered 
                                        "low" confidence.
            min_conf_threshold (float): The minimum confidence threshold below which boxes
                                        are not to be considered
            iou_threshold (float): The minimum iou threshold below which a matching is rejected.
        """
        
        
        self.detector = det
        self.detector.test_conf = min_conf_threshold
        self.alive_tracks = None
        self.tracks_on_hold = []
        self.id_ctr = 0
        self.appearance_model = None #TODO
        self.low_conf_threshold = low_conf_threshold
        self.latest_frame = None
        self.visualizer = TrackVisualizer()
        self.iou_threshold = iou_threshold
        

        
    def step(self, frame):
        """Track objects in new frame and update tracks. Returns frame with bboxes along with IDs attached. Wrapper function around `update` and `draw`.
        
        Args:
            frame (np.ndarray): The frame as read by opencv and meant to be passed to the detector.
            
        Returns: Image with tracked boxes along with identity information to it, time taken by tracker."""
            
        #bboxes is Nx7
        
        self.latest_frame = frame
        bboxes, img_info = self.detector(frame)
        
        
        ratio = self.detector._get_ratio(img_info["raw_img"])
        bboxes[..., :4] /= ratio #normalize boxes
        scores = bboxes[..., 4] * bboxes[..., 5]
        
        t0 = time.time()
        
        tracked_boxes = self.update(bboxes[..., :4], scores)
        t1 = time.time()
        return self.draw(frame, tracked_boxes), t1-t0
        
        
        
        
    def update(detections, scores, frame=None):
        """Given rescaled detected bounding boxes and their confidence scores, updates the trackers by associating detections to appropriate trackers.

        Args:
            detections (np.ndarray): detected bboxes in xywh format.
            scores (np.ndarry): 1d array of confidence scores
            frame (np.ndarray | None): The frame currently being considered. If None, will use self.latest_frame.
        
        Returns:
            output bboxes with IDs of shape Nx5 [[x,y,w,h,ID], ...]
        """
        
        # 2. Divide boxes into high and low confidence boxes.
        
        # 3. The high confidence boxes are matched first with the predictions of the tracks.
        # 4. Next the remaining tracks are matched with the low confidence boxes (maybe occluded objects)
        # 5. Finally the *last observations* (OC-recovery) of the remaining tracks are matched with the
        #    remaining high confidence boxes.
        # 6. Any remaining high confidence boxes are used to create new tracks.
        # 7. Tracks are updated with new boxes.
        # 8. All the tracks that have a streak of detections > threshold (=3) are considered "proper"
        #   tracks and are returned/added to the image.     
            
    
    def _associate(self, track, box):
        """Given an existing track and a box, add that box to the track."""   
        
        
            

            
    def draw(self, frame, tracks):
        """Draws the bounding boxes in each track and returns an img in RGB order."""
         
        return self.visualizer(frame, tracks) if tracks else frame