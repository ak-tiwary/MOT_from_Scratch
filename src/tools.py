import numpy as np

def xywh_to_xyxy(boxes):
    centers, widths = boxes[..., :2], boxes[..., 2:]
    boxes[...,:2], boxes[...,2:] = centers - (widths / 2), centers + (widths / 2)
    return boxes

def get_iou_matrix(boxes_1, boxes_2):
    """Given np arrays boxes_1 and boxes_2 of shapes Mx4 and Nx4 in xywh coordinates  and 
    returns a matrix of pairwise IoUs between each pair of boxes such that 
    M[i,j] = IoU(box1_i, box2_j)"""
    
    
    boxes_1 = xywh_to_xyxy(boxes_1).unsqueeze(1) #shape Nx1x4
    boxes_2 = xywh_to_xyxy(boxes_2).unsqueeze(0) #shape 1 x M x 4
    
    #we desire shape NxM, so we only need to perform pairwise IoU calculations
    
    area = lambda boxes: np.max((boxes[..., 3] - boxes[..., 1]),0) * \
                         np.max((boxes[..., 2] - boxes[..., 0]), 0)
    
    top_left_intersection = np.max(boxes_1[..., :2], boxes_2[..., :2])
    bottom_right_intersection = np.max(boxes_1[..., 2:], boxes_2[..., 2:])
    intersection_boxes = np.concatenate([top_left_intersection, 
                                         bottom_right_intersection], axis=-1)
    inter_areas = area(intersection_boxes) #shape NxM
    union_areas = area(boxes_1) + area(boxes_2) - inter_areas
    
    return inter_areas / union_areas