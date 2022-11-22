import numpy as np
from scipy.optimize import linear_sum_assignment as hungarian_algorithm
from .track import Track
from .visualizer import TrackVisualizer
from .tools import get_iou_matrix, get_velocity_matrix
import time


import torch
import torchvision.transforms as T
import torch.nn.functional as F

from loguru import logger


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
    def __init__(self, det, appearance_model = None, track_expiry_time=500000, 
                 low_conf_threshold=0.5, min_conf_threshold=0.2, 
                 iou_threshold=0.1, lambda_vel=0.2, lambda_app=1.5, streak_threshold=3, 
                 appearance_thresh_alive=0.5, appearance_thresh_dead=0.75, MOT_IDs=True,
                 embedding_dim=2048):
        """The main tracker that handles a single step of tracking. 
        
        Args:
            det (Detector): A detector object initialized already with desired settings.
            appearance_model (nn.Module): A similarity network that estimates how similar
                                          two images are.  
            track_expiry_time (int): Number of frames to wait before expiring a track as lost
                                     and starting a new track.
            low_conf_threshold (float): The threshold below which a bbox is considered 
                                        "low" confidence.
            min_conf_threshold (float): The minimum confidence threshold below which boxes
                                        are not to be considered
            iou_threshold (float): The minimum iou threshold below which a matching is rejected.
            lambda_vel (float): Hyperparameter for the cost IoU + λ * velocity_cost
            steak_threshold (int): The minimum number of consecutive detections before a track is considered proper and displayed/returned.
            apperance_thresholds (float): These are the maximum cost thresholds for a matching
            following "Simple Cues lead to a Strong Multi object Tracker (2022)".
            MOT_IDs (bool): MOT 20 etc. want IDs to be positive, so boolean flag for this.
            embedding_dim (int) : the embedding dimension from the backbone of the appearance
            model. For Resnet50 this is 2048.
        """
        
        
        self.detector = det
        self.detector.test_conf = min_conf_threshold
        self.tracks = []
        self.id_ctr = 1 if MOT_IDs else 0 
        self.appearance_model = appearance_model #TODO
        self.low_conf_threshold = low_conf_threshold
        self.low_conf_iou_threshold = 0.5
        self.latest_frame = None
        self.visualizer = TrackVisualizer()
        self.first_iou_threshold = 0.3
        self.second_iou_threshold = iou_threshold
        self.appearance_threshold_alive = appearance_thresh_alive
        self.appearance_threshold_dead = appearance_thresh_dead
        self.lambda_vel = lambda_vel
        self.lambda_app = lambda_app
        
        #track expiry is very high since we can perform
        #long term track handling
        self.track_expiry_time = track_expiry_time
        self.streak_threshold = streak_threshold
        self.num_frames = 0
        
        self.uses_appearance = self.appearance_model is not None
        if self.uses_appearance:
            self.appearance_model.eval().to("cpu")
            self.embedding_dim = embedding_dim
        

        
    def step(self, frame):
        """Track objects in new frame and update tracks. Returns frame with bboxes along with IDs attached. Wrapper function around `update` and `draw`.
        
        Args:
            frame (np.ndarray): The frame as read by opencv and meant to be passed to the detector.
            
        Returns: Image with tracked boxes along with identity information to it, time taken by tracker."""
            
        #bboxes is Nx7
        
        self.latest_frame = frame
        self.num_frames += 1

        bboxes, img_info = self.detector(frame)
        
        if self.num_frames < 2:
            #logger.log("INFO", f"Starting step and bboxes.shape is {bboxes.shape}") 
            pass      
        
        #ratio = self.detector._get_ratio(img_info["raw_img"]) 
        ratio = img_info["ratio"]
        #logger.info(f"bboxes shape = {bboxes.shape}")
        bboxes[..., :4] /= ratio #resize boxes to original image size
        scores = bboxes[..., 4] * bboxes[..., 5]
        
        t0 = time.time()
        
        tracked_boxes = self.update(bboxes[..., :4], scores)
        t1 = time.time()
        return self.draw(frame, tracked_boxes), t1-t0
          
        
    def update(self, detections, scores, frame=None):
        """Given rescaled detected bounding boxes and their confidence scores, updates the trackers by associating detections to appropriate trackers. NOTE: The ID is passed as a float. Convert it 
        to an int before proceeding.f

        Args:
            detections (np.ndarray): detected bboxes in xywh format.
            scores (np.ndarry): 1d array of confidence scores for each box
            frame (np.ndarray | None): The frame currently being considered. If None, will use self.latest_frame.
        
        Returns:
            output bboxes with IDs of shape Nx5 [[x,y,w,h,ID], ...]
        """
        if frame is None:
            #we need frame if we are using appearance, 
            if self.uses_appearance:
                frame = self.latest_frame
            
        
        
        # 2. Divide boxes into high and low confidence boxes.
        high_conf_boxes = detections[scores > self.low_conf_threshold]
        low_conf_boxes = detections[scores  <= self.low_conf_threshold]
        scores = scores[scores > self.low_conf_threshold]
        #logger.info(f"Num high conf boxes = {len(high_conf_boxes)}")
        #logger.info(f"Num low conf boxes = {len(low_conf_boxes)}")
        no_high_boxes = high_conf_boxes.size == 0 #flag for no boxes
        no_low_boxes = low_conf_boxes.size == 0
        
        if not self.tracks: #all high confidence boxes are new tracks.
            if no_high_boxes:
                return np.empty((0,5))
            if self.uses_appearance:
                appearance_features = self.get_apperance_features(high_conf_boxes)
            else:
                appearance_features = None
            
            self.tracks = [Track(box, id=i+self.id_ctr, appearance_features=features) 
                           for i,box,features in 
                           enumerate(zip(high_conf_boxes, appearance_features))]
            self.id_ctr += len(self.tracks)
            #logger.log("INFO", f"There are no tracks. There are {len(high_conf_boxes)} boxes.")
            box_with_ids = np.stack([np.concatenate([t.get_box_coordinates(), 
                                                     np.array([t.id])]) for t in self.tracks], 
                                    axis=0)
            
            return box_with_ids
        
        
        if no_high_boxes and no_low_boxes:
            for track in self.tracks:
                track.update(None)
            return np.empty((0,5))
        
        
        pred_tracks = np.stack([t.predict() for t in self.tracks], axis=0)
        last_observations = np.stack([t.last_observation for t in self.tracks], axis=0)
        track_indices = np.arange(len(self.tracks))
        # 3. The high confidence boxes are matched first with the predictions of the tracks
        if not no_high_boxes :#if there are high confidence boxes
            box_indices = np.arange(len(high_conf_boxes))
            #we want a cost matrix, so higher is worse
            iou_scores = get_iou_matrix(pred_tracks, high_conf_boxes)
            iou_cost = 1 - iou_scores
            
            #last_observations = np.stack([t.last_observation for t in self.tracks], axis=0)
            #NxMx2
            velocity_matrix_from_tracks_to_detections = get_velocity_matrix(last_observations[...,:2], high_conf_boxes[..., :2]) 
            track_velocities = np.expand_dims(np.stack([t.velocity_direction for t in self.tracks], axis=0)
                                              , 1) #Nx1x2
            #logger.info(f"track velocities shape = {track_velocities.shape}")
            #logger.info(f"velocity matrix shape = {velocity_matrix_from_tracks_to_detections.shape}")
            
            #take dot product between the two unit vectors. The more aligned they are, the higher the score.
            velocity_consistency_matrix = np.sum(velocity_matrix_from_tracks_to_detections *
                                                 track_velocities,
                                                 axis=-1) #NxM
            
            velocity_consistency_matrix *= scores[np.newaxis, ...] #velocity score is weighted by object confidence
            
            #we want a cost matrix for which higher is worse. Also we normalize to get between 0 and 1
            velocity_cost = (1-velocity_consistency_matrix) / 2
            
            #the cost function is a combination of IoU, velocity consistency, and appearance (to be added)
            
            #############################
            #add appearance model. Have different thresholds for lost and found tracks
            #
            ############################
            if self.uses_appearance:
                appearance_batch = self.get_appearance_features(high_conf_boxes)
                appearance_cost = self._get_appearance_cost_matrix(
                                        track_indices, 
                                        appearance_batch
                                    )
                
                
                #the weights should be tuned.
                cost_matrix = iou_cost \
                            + self.lambda_vel * velocity_cost \
                            + self.lambda_app * appearance_cost
            else:
                cost_matrix = iou_cost \
                            + self.lambda_vel * velocity_cost
                    
            ############################
            
            
            
            
            
            matches, unmatched_tracks, unmatched_boxes = self._associate(track_indices, box_indices, cost_matrix)
                                                                         
            to_unmatch_trks = []
            to_unmatch_boxes = []
            for row, column in matches:
                trk_idx = track_indices[row]
                box_idx = box_indices[column]
                trk = self.tracks[trk_idx]
                if self.uses_appearance:
                    app_cost = appearance_cost[row, column]
                    threshold = self.appearance_threshold_dead if trk.is_inactive \
                                else self.appearance_threshold_alive    

                iou_score = iou_scores[row, column]
                iou_thresh = self.first_iou_threshold
                
                #after 100 frames ignore iou cues
                if trk.time_since_last_detection >= 100:
                    iou_thresh=0.
                    
                condition =  iou_score < iou_thresh
                if self.uses_appearance:
                    condition = condition or (app_cost > threshold) 
                    
                
                if condition: #not a good match
                    to_unmatch_trks.append(trk_idx)
                    to_unmatch_boxes.append(box_idx)
                else:
                    #good match
                    # logger.info(f"Track ID = {self.tracks[trk_idx].id}")
                    # logger.info(f"Last observation was {self.tracks[trk_idx].last_observation}")
                    # logger.info(f"New matched box = {high_conf_boxes[box_idx]}")
                    # logger.info(f"iou score = {iou_scores[row, column]}")
                    # logger.info(f"predicted tracks = {pred_tracks[trk_idx]}\n\n\n\n")
                    features = None
                    if self.uses_appearance:
                        features = appearance_batch[box_idx]
                    
                    self.tracks[trk_idx].update(high_conf_boxes[box_idx],
                                                features)
            
            # logger.info(f"\n\n!!!!!! 1) track number 1 on screen: {self.tracks[0].get_box_coordinates()}\n\n") 
            unmatched_tracks = np.concatenate([unmatched_tracks, np.array(to_unmatch_trks,dtype=int)])
            unmatched_boxes = np.concatenate([unmatched_boxes, np.array(to_unmatch_boxes, dtype=int)])

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
            #logger.info(f"iou_scores has shape {iou_scores.shape}, {len(unmatched_track_boxes)}, \
             #   {len(low_conf_boxes)}, matches.shape {matches.shape} ")
            #logger.info(f"matches.shape {matches.shape}, {iou_scores[matched_trks, matched_dets]}")
            filtered_matches = matches[iou_scores[matched_trks,matched_dets] >= self.low_conf_iou_threshold] 
            
            matched_idxs = []
            for row, column in filtered_matches:
                self.tracks[unmatched_tracks[row]].update(low_conf_boxes[low_box_indices[column]])
                matched_idxs.append(unmatched_tracks[row])
                
             
            unmatched_tracks = np.setdiff1d(unmatched_tracks, np.array(matched_idxs, dtype=int))
            
        # 5. Finally the **last observations** (OC-recovery) of the remaining tracks are matched with the remaining high confidence boxes. Gives priority to observations over predictions.
        if not no_high_boxes:
            
            matched_box_idxs = np.setdiff1d(box_indices, unmatched_boxes)
            if matched_box_idxs.size: #if there are boxes remaining
                
                remaining_high_boxes = high_conf_boxes[unmatched_boxes]
                remaining_trks = last_observations[unmatched_tracks]
                #Cost is a combination of IoU and appearance (todo).
                iou_scores = get_iou_matrix(remaining_trks, remaining_high_boxes)
                iou_cost = 1 - iou_scores
                rem_tracks = unmatched_tracks.copy()
                rem_boxes = unmatched_boxes.copy()
                
                #################
                # Appearance cost
                if self.uses_appearance:
                    rem_box_features = appearance_batch[unmatched_boxes]
                    appearance_cost = self._get_appearance_cost_matrix(
                        unmatched_tracks, rem_box_features
                    )
                
                    #emphasize appearance when long-term recovering
                    cost_matrix = 0.5*iou_cost + appearance_cost
                else:
                    cost_matrix = iou_cost
                #################
                
                matches, unmatched_tracks, unmatched_boxes = self._associate(unmatched_tracks, unmatched_boxes, cost_matrix)
                extra_unmatched_trks = []
                extra_unmatched_boxes = []
                tmp, _ = matches.T
                for row, column in matches:
                    trk_idx = rem_tracks[row]
                    box_idx = rem_boxes[column]
                    iou_score = iou_scores[row, column]
                    flag = not self.uses_appearance 
                    if not flag: #uses appearance
                        app_cost = appearance_cost[row, column]
                        if self.tracks[trk_idx].is_inactive:
                            threshold = self.appearance_threshold_dead
                        else:
                            threshold = self.appearance_threshold_alive
                        flag =  app_cost < threshold
                    if iou_score >= self.second_iou_threshold:
                        if flag:
                            #keep association
                            if self.uses_association:
                                features = appearance_batch[box_idx]
                            else:
                                features = None
                            
                            self.tracks[trk_idx].update(
                                high_conf_boxes[box_idx],
                                features
                                )
                    else:
                        #remove association
                        extra_unmatched_trks.append(trk_idx)
                        extra_unmatched_boxes.append(box_idx)
                unmatched_tracks = np.concatenate([unmatched_tracks, np.array(extra_unmatched_trks, dtype=int)])
                unmatched_boxes = np.concatenate([unmatched_boxes, np.array(extra_unmatched_boxes, dtype=int)])
        
        # 6. Unmatched Tracks are updated with None.
        for trk in unmatched_tracks:
            self.tracks[trk].update(None)
        
        
        
        # 7. Any remaining high confidence boxes are used to create new tracks.
        for box_idx in unmatched_boxes:
            self.tracks.append(Track(high_conf_boxes[box_idx], id=self.id_ctr, appearance_features=appearance_batch[box_idx]))
            self.id_ctr += 1
        
        
        
        # 8. All the tracks that have a streak of detections > threshold (=3) are considered "proper" tracks and are returned/added to the image. All tracks that have been missed for longer than expiry date are removed.
        proper_detections = []
        
        #reverse iteration so we can pop tracks as we go 
        for i in range(len(self.tracks) - 1, -1, -1):    
            track = self.tracks[i]
            
            if track.obs_streak >= self.streak_threshold or track.time_lapsed < self.streak_threshold:
                proper_detections.append(np.concatenate([track.get_box_coordinates(),
                                                        np.array([track.id])]))
            
            if track.time_since_last_detection > self.track_expiry_time:
                self.tracks.pop(i)
                
        if not proper_detections:
            return np.empty((0,5))
        
        # logger.info(f"\n\n\n number of consec hits for each track = \n {[t.obs_streak for t in self.tracks]} \n\n\n")
        
        # logger.info(f"track number 1 with id {self.tracks[0].id} last observation: {self.tracks[0].observation_before_last}")
        # logger.info(f"\n\n!!!!!! track number 1 on screen: {self.tracks[0].get_box_coordinates()}\n\n")   
        return np.stack(proper_detections, axis=0)
    
    
    

    
    def _get_appearance_cost_matrix(self, track_indices, feature_matrix):
        """Feature matrix is an np array of shape NxD. track_indices are indices into self.tracks and of length M
        
        Will return an appearance cost matrix of length MxN. Assumes that
        the feature vectors are normalized.
        
        Uses the Proxy distance for inactive track from the "Simple Cues lead
        to a Strong Object Tracker" paper."""
        
        _, D = feature_matrix.shape
        track_features = np.zeros((len(track_indices, D))).astype(float)
        
        track_apperances = [    
                                trk.avg_apperance
                                if (trk := self.tracks[trk_idx]).is_inactive
                                else trk.get_last_appearance()
                            
                            for trk_idx in track_indices]
        track_features = np.stack(track_apperances, dim=0)
        
        return 1 - track_features @ feature_matrix.T
        
        
       
      
      
      
    
    def _associate(self, tracks, boxes, cost_matrix):
        """Given track and box indices as 1d numpy arrays, and a cost matrix for associations between them, performs linear assignment.
        Returns matches as an Nx2 np array, unmatched_tracks as a 1d array, and unmatched_dets as a 1d array."""
        #logger.info(f"cost_matrix = {cost_matrix}")
        matched_tracks, matched_boxes = hungarian_algorithm(cost_matrix)
        track_mask = np.zeros_like(tracks, dtype=bool)
        track_mask[matched_tracks] = True
        box_mask = np.zeros_like(boxes, dtype=bool)
        #logger.info(f"len(tracks)= {len(tracks)}, boxes = {boxes}, track_mask")
        box_mask[matched_boxes] = True
        #matched_tracks = tracks[track_mask]
        #matched_boxes = boxes[box_mask]
        
        unmatched_tracks = tracks[~track_mask]
        unmatched_boxes = boxes[~box_mask]
        
        return np.stack([matched_tracks, matched_boxes], axis=1), unmatched_tracks, unmatched_boxes
        
        
            

            
    def draw(self, frame, tracks):
        """Draws the bounding boxes with ID for each track and returns an img in RGB order."""
        if tracks.size == 0: #if there is nothing to draw, return image as is
            return frame 
        return self.visualizer(frame, tracks)
    
    def get_box_apperance_features(self, boxes):
        """
        Given an Nx4 np array of boxes in xywh format, returns NxD apperance features array.
        """
        #for each N we want to calculate the int of the xywh
        
        boxes_corner = np.zeros_like(boxes)
        boxes_corner[:, :2] = boxes[:, :2] - boxes[:, 2:] / 2
        boxes_corner[:, 2:] = boxes[:, :2] + boxes[:, 2:] / 2
        
        boxes_corner = boxes_corner.astype(int) #we want to use it as coordinates
        frame = self.latest_frame #Nx [x1,y1,x2, y2]
        
        if frame.shape[0] == 3: #C x H xW
            frame_torch = torch.from_numpy(frame).to(torch.float)
        elif frame.shape[2] == 3: #HxWxC
            frame_torch = torch.from_numpy(np.transpose(frame, (2,0,1))).to(torch.float)
        else:
            raise Exception(f"Frame has shape {frame.shape}")
        
        resize = T.Resize((256, 128))
        input = torch.stack([resize(frame_torch[:, y1:y2, x1:x2]) 
                             for y1,x1,y2,x2 in boxes_corner],
                            dim=0)
        features = self.appearance_model(input) #Nxd
        return features.numpy()
                            
        
        
        
        
        
        