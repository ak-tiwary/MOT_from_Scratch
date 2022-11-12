import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian_algorithm
from detector import Detector
from track import Track
from visualizer import TrackVisualizer
from tools import get_iou_matrix, get_velocity_matrix
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
                 iou_threshold=0.2, lambda_vel=0.2, streak_threshold=3, MOT_IDs=True):
        """The main tracker that handles a single step of tracking. 
        
        Args:
            det (Detector): A detector object initialized already with desired settings.
            track_expiry_time (int): Number of frames to wait before expiring a track as lost
                                     and starting a new track.
            low_conf_threshold (float): The threshold below which a bbox is considered 
                                        "low" confidence.
            min_conf_threshold (float): The minimum confidence threshold below which boxes
                                        are not to be considered
            iou_threshold (float): The minimum iou threshold below which a matching is rejected.
            lambda_vel (float): Hyperparameter for the cost IoU + λ * velocity_cost
            steak_threshold (int): The minimum number of consecutive detections before a track is considered proper and displayed/returned.
            MOT_IDs (bool): MOT 20 etc. want IDs to be positive, so boolean flag for this.
        """
        
        
        self.detector = det
        self.detector.test_conf = min_conf_threshold
        self.tracks = []
        self.id_ctr = 1 if MOT_IDs else 0 
        self.appearance_model = None #TODO
        self.low_conf_threshold = low_conf_threshold
        self.latest_frame = None
        self.visualizer = TrackVisualizer()
        self.iou_threshold = iou_threshold
        self.lambda_vel = lambda_vel
        self.track_expiry_time = track_expiry_time
        self.streak_threshold = streak_threshold
        

        
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
          
        
    def update(self, detections, scores, frame=None):
        """Given rescaled detected bounding boxes and their confidence scores, updates the trackers by associating detections to appropriate trackers.

        Args:
            detections (np.ndarray): detected bboxes in xywh format.
            scores (np.ndarry): 1d array of confidence scores for each box
            frame (np.ndarray | None): The frame currently being considered. If None, will use self.latest_frame.
        
        Returns:
            output bboxes with IDs of shape Nx5 [[x,y,w,h,ID], ...]
        """
        if frame is None:
            frame = self.latest_frame
            
        
        
        # 2. Divide boxes into high and low confidence boxes.
        high_conf_boxes = detections[scores > self.low_conf_threshold]
        low_conf_boxes = detections[scores  <= self.low_conf_threshold]
        
        no_high_boxes = high_conf_boxes.size == 0 #flag for no boxes
        no_low_boxes = low_conf_boxes.size == 0
        
        if not self.tracks: #all high confidence boxes are new tracks.
            self.tracks = [Track(box, id=i+self.id_ctr) for i,box in enumerate(high_conf_boxes)]
            self.id_ctr += len(self.tracks)
            
            box_with_ids = np.stack([np.concatenate([t.box_coordinates, 
                                                     np.array([t.id])]) for t in self.tracks], 
                                    axis=0)
            
            return box_with_ids
        
        if no_high_boxes and no_low_boxes:
            for track in self.trackers:
                track.update(None)
            return np.empty((0,5))
        
        pred_tracks = np.stack([t.predict() for t in self.tracks], axis=0)
        last_obs_tracks = np.stack([t.last_observation for t in self.tracks], axis=0)
        track_indices = np.arange(len(self.tracks))
        
        # 3. The high confidence boxes are matched first with the predictions of the tracks
        if not no_high_boxes :#if there are high confidence boxes
            box_indices = np.arange(len(high_conf_boxes))
            #we want a cost matrix, so higher is worse
            iou_scores = get_iou_matrix(pred_tracks, high_conf_boxes)
            iou_cost = 1 - iou_cost
            
            last_observations = np.stack([t.last_observation for t in self.tracks], axis=0)
            #NxMx2
            
            velocity_matrix_from_tracks_to_detections = get_velocity_matrix(last_observations, high_conf_boxes) 
            track_velocities = np.stack([t.velocity_direction for t in self.tracks], axis=0).unsqueeze(1) #Nx1x2
            
            #take dot product between the two unit vectors. The more aligned they are, the higher the score.
            velocity_consistency_matrix = np.sum(velocity_matrix_from_tracks_to_detections *
                                                 track_velocities,
                                                 axis=-1) #NxM
            velocity_consistency_matrix *= scores.unsqueeze(0) #velocity score is weighted by object confidence
            
            #we want a cost matrix for which higher is worse. Also we normalize to get between 0 and 1
            velocity_cost = (1-velocity_consistency_matrix) / 2
            
            #the cost function is a combination of IoU, velocity consistency, and appearance (to be added)
            
            matches, unmatched_tracks, unmatched_boxes = self._associate(track_indices, box_indices,
                                                                         iou_cost + self.lambda_vel * velocity_cost)
            to_unmatch_trks = []
            to_unmatch_boxes = []
            for trk_idx, box_idx in matches:
                if iou_scores[trk_idx, box_idx] < self.iou_threshold: #low iou match, reject
                    to_unmatch_trks.append(trk_idx)
                    to_unmatch_boxes.append(box_idx)
                else:
                    #good match
                    self.tracks[trk_idx].update(high_conf_boxes[box_idx])
                    
            unmatched_tracks = np.concatenate([unmatched_tracks, np.array(to_unmatch_trks)])
            unmatched_boxes = np.concatenate([unmatched_boxes, np.array(to_unmatch_boxes)])

        else: #there are no high boxes
            unmatched_tracks = track_indices
            

    
        # 4. Next the remaining tracks are matched with the low confidence boxes (maybe occluded objects) using only IoU as a cost. This is the BYTE association of ByteTrack.
        
        if not no_low_boxes: #if there are low boxes to consider.
            low_box_indices = np.arange(len(low_conf_boxes)) 
            unmatched_track_boxes = pred_tracks[unmatched_tracks]
            iou_scores = get_iou_matrix(unmatched_track_boxes, low_conf_boxes)
            iou_cost = 1 - iou_scores
            
            #ignore unmatched low confidence boxes
            matches, _, _ = self._associate(unmatched_tracks, low_box_indices, iou_cost)
            
            #we may ignore any matches below IoU threshold
            matched_trks, matched_dets = matches.T #the rows and columns from 
            filtered_matches = matches[iou_scores[matched_trks,matched_dets] >= self.iou_threshold] 
            
            matched_idxs = []
            for trk_idx, box_idx in filtered_matches:
                self.tracks[trk_idx].update(low_conf_boxes[box_idx])
                matched_idxs.append(trk_idx)
                
             
            unmatched_tracks = np.setdiff1d(unmatched_tracks, np.array(matched_idxs))
                    
                    
        
        
        # 5. Finally the **last observations** (OC-recovery) of the remaining tracks are matched with the remaining high confidence boxes. Gives priority to observations over predictions.
        if not no_high_boxes:
            
            remaining_high_box_idxs = np.setdiff1d(box_indices, unmatched_boxes)
            
            if remaining_high_box_idxs.size: #if there are boxes remaining
                
                remaining_high_boxes = high_conf_boxes[remaining_high_box_idxs]
                remaining_trks = last_observations[unmatched_tracks]
                
                #Cost is a combination of IoU and appearance (todo).
                iou_scores = get_iou_matrix(remaining_high_boxes, remaining_trks)
                iou_cost = 1 - iou_scores
                
                matches, unmatched_tracks, unmatched_boxes = self._associate(remaining_high_box_idxs,
                                                                             unmatched_tracks, iou_cost)
                
                extra_unmatched_trks = []
                extra_unmatched_boxes = []
                for trk_idx, box_idx in matches:
                    if iou_scores[trk_idx, box_idx] >= self.iou_threshold:
                        #keep association
                        self.tracks[trk_idx].update(high_conf_boxes[box_idx])
                        
                    else:
                        #remove association
                        extra_unmatched_trks.append(trk_idx)
                        extra_unmatched_boxes.append(box_idx)
                unmatched_tracks = np.concatenate([unmatched_tracks, np.array(extra_unmatched_trks)])
                unmatched_boxes = np.concatenate([unmatched_boxes, np.array(extra_unmatched_boxes)])
        
            
        # 6. Unmatched Tracks are updated with None.
        for trk in unmatched_tracks:
            self.tracks[trk].update(None)
        
        # 7. Any remaining high confidence boxes are used to create new tracks.
        for box_idx in unmatched_boxes:
            self.tracks.append(Track(high_conf_boxes[box_idx], id=self.id_ctr))
            self.id_ctr += 1
        
        
        # 8. All the tracks that have a streak of detections > threshold (=3) are considered "proper" tracks and are returned/added to the image. All tracks that have been missed for longer than expiry date are removed.
        proper_detections = []
        
        #reverse iteration so we can pop tracks as we go 
        for i in range(len(self.tracks) - 1, -1, -1):    
            track = self.tracks[i]
            
            if track.obs_streak >= self.streak_threshold or track.time_lapsed < self.streak_threshold:
                proper_detections.append(np.concatenate(track.get_box_coordinates(),
                                                        np.array([track.id])))
            
            if track.time_since_last_detection > self.track_expiry_time:
                self.tracks.pop(i)
                
        return np.stack(proper_detections, axis=0)
    
    def _associate(self, tracks, boxes, cost_matrix):
        """Given track and box indices as 1d numpy arrays, and a cost matrix for associations between them, performs linear assignment.
        Returns matches as an Nx2 np array, unmatched_tracks as a 1d array, and unmatched_dets as a 1d array."""
        matched_tracks, matched_boxes = hungarian_algorithm(cost_matrix)
        track_mask = np.zeros_like(tracks, dtype=bool)
        track_mask[matched_tracks] = True
        box_mask = np.zeros(boxes, dtype=bool)
        box_mask[matched_boxes] = True
        matched_tracks = tracks[track_mask]
        matched_boxes = boxes[box_mask]
        
        unmatched_tracks = tracks[~track_mask]
        unmatched_boxes = boxes[~box_mask]
        
        return np.stack([matched_tracks, matched_boxes], axis=1), unmatched_tracks, unmatched_boxes
        
        
            

            
    def draw(self, frame, tracks):
        """Draws the bounding boxes with ID for each track and returns an img in RGB order."""
         
        return self.visualizer(frame, tracks) if tracks else frame