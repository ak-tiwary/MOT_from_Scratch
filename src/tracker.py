import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian_algorithm
from detector import Detector
from track import Track
from visualizer import TrackVisualizer
from tools import get_iou_matrix


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
            
        Returns: Image with tracked boxes along with identity information to it."""
            
        #bboxes is Nx7
        
        self.latest_frame = frame
        bboxes, img_info = self.detector(frame)
        
        #box_coords = bboxes[:, :4]
        
        confidences = bboxes[:, 4] * bboxes[:, 5]
        mask = confidences > self.low_conf_threshold
        
        high_conf_boxes = bboxes[mask]
        low_conf_boxes = bboxes[~mask]
        
        high_conf_box_coords = high_conf_boxes[:4]
        low_conf_box_coords = low_conf_boxes[:4]
        
        if self.alive_tracks is None:
            #first frame with any objects, no matching required. Ignore low_conf boxes.
            
            self.alive_tracks = [Track(box, id=i+self.id_ctr) for i,box in enumerate(high_conf_boxes) ]
            id_ctr += len(self.alive_tracks)
            return self._draw(frame, self.alive_tracks)

        #We have some alive tracks, some tracks on hold for recovery
        #and we have low and high confidence detection boxes.
        
        
        #make the predictions for where the boxes will be in this step
        #we will use this for IoU and association calculations
        predicted_tracks = []
        for track in self.alive_tracks:
            predicted_tracks.append(track.predict())
        predicted_tracks = np.stack(predicted_tracks, axis=0)
        
        if high_conf_box_coords:
            iou_cost = get_iou_matrix(predicted_tracks, high_conf_box_coords)
            track_matches, box_matches = hungarian_algorithm(iou_cost)
            
            matches = np.stack([track_matches, box_matches], axis=1)
            matched_ious = iou_cost[track_matches, box_matches]
            mask = matched_ious > self.iou_threshold
            matches = matches[mask] #only take those matches that have sufficient IoU overlap
            
            for match in matches:
                track_idx, box_idx = match
                self._associate(self.alive_tracks[track_idx], high_conf_boxes[box_idx])
            matched_track_idxs, matched_box_idxs = matches.T  
                
            matched_mask = np.zeros(shape=len(self.alive_tracks), dtype=bool)
            matched_mask[matched_track_idxs] = True
            unmatched_mask = ~matched_mask #mask of those alive tracks unmatched
            
            matched_box_mask = np.zeros(shape=len(high_conf_boxes), dtype=bool)
            matched_box_mask[matched_box_idxs] = True
            unmatched_high_conf_mask = ~matched_box_idxs #mask of those high conf boxes unmatched   
            
            
            #do second level matching between remaining high confidence boxes
            #and low confidence boxes with unmatched tracks and lost tracks?
            #how to do track recovery? See OC-SORT/ByteTrack
            
            
            
            #remaining boxes can be thrown away,
            #remaining tracks are either expired, or put on hold
            
            
            
            
           

        #handle case where all boxes are low confidence. 
        
        
    def update(detections, scores, frame=None):
        """Given rescaled detected bounding boxes and their confidence scores, updates the trackers by associating detections to appropriate trackers.

        Args:
            detections (np.ndarray): detected bboxes in xywh format.
            scores (np.ndarry): 1d array of confidence scores
        
        Returns:
            output bboxes with IDs of shape Nx5 [[x,y,w,h,ID], ...]
        """
            
    
    def _associate(self, track, box):
        """Given an existing track and a box, add that box to the track."""   
        
        
            

            
    def draw(self, frame, tracks):
        """Draws the bounding boxes in each track and returns an img in RGB order."""
         
        return self.visualizer(frame, tracks) if tracks else frame