import numpy as np
from loguru import logger

def xywh_to_xyxy(boxes):
    boxes_tmp = np.zeros_like(boxes)
    centers, widths = boxes[..., :2], boxes[..., 2:4]
    boxes_tmp[...,:2], boxes_tmp[...,2:4] = centers - (widths / 2), centers + (widths / 2)
    return boxes_tmp

def get_iou_matrix(boxes_1, boxes_2, eps=1e-8):
    """Given np arrays boxes_1 and boxes_2 of shapes Mx4 and Nx4 in xywh coordinates  and 
    returns a matrix of pairwise IoUs between each pair of boxes such that 
    M[i,j] = IoU(box1_i, box2_j)"""
    
    
    boxes_1_xyxy = np.expand_dims(xywh_to_xyxy(boxes_1), 1) #shape Nx1x4
    boxes_2_xyxy = np.expand_dims(xywh_to_xyxy(boxes_2), 0) #shape 1 x M x 4
    
    #print(f"boxes_1_xyxy = \n{boxes_1_xyxy}, \nboxes_2_xyxy = \n{boxes_2_xyxy}")
    #logger.log("INFO", "expanded box dimensions")
    #we desire shape NxM, so we only need to perform pairwise IoU calculations
    
    area = lambda boxes: np.maximum((boxes[..., 3] - boxes[..., 1]),0) * \
                         np.maximum((boxes[..., 2] - boxes[..., 0]), 0)
    
    #logger.log("INFO", f"boxes_1.shape {boxes_1.shape}, boxes_2.shape {boxes_2.shape}")
    top_left_intersection = np.maximum(boxes_1_xyxy[..., :2], boxes_2_xyxy[..., :2])
    #logger.log("INFO", "computed top left intersection")
    bottom_right_intersection = np.minimum(boxes_1_xyxy[..., 2:4], boxes_2_xyxy[..., 2:4])
    
    #print(f"top_left_intersection = \n{top_left_intersection}, bottom_right_intersection = \n{bottom_right_intersection}")
    #logger.log("INFO", "computed bottom right intersection")
    intersection_boxes = np.concatenate([top_left_intersection, 
                                         bottom_right_intersection], axis=-1)
    #logger.log("INFO", "computed intersection boxs")
    inter_areas = area(intersection_boxes) #shape NxM
    #logger.log("INFO", "computed inter areas")
    union_areas = area(boxes_1_xyxy) + area(boxes_2_xyxy) - inter_areas
    
    #print(f"inter_areas = {inter_areas}, union_areas = {union_areas}, area_b1 = {area(boxes_1_xyxy)}, area_b2 = {area(boxes_2_xyxy)}")
    return inter_areas / (union_areas + eps)

def get_velocity_matrix(sources, targets, eps=1e-12):
    """Given np arrays sources and targets of shapes Mx2 and Nx2 will 
    returns an MxNx2 matrix of pairwise velocity directions from each source to each target such that 
    M[i,j] = velocity_direction(sources[i], targets[j])"""
    sources = np.expand_dims(sources, 1)
    targets = np.expand_dims(targets, 0)
    
    displacement = targets-sources #NxMx2
    norms = np.linalg.norm(displacement, axis=2, keepdims=True)
    
    return displacement / (norms + eps) #shape is NxMx2
    