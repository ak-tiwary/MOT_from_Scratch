import numpy as np
import torch
import torchvision
import cv2


    
#borrowed from YOLOX's visualize function. Different color for different categories.
_COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
    
    

class Visualizer:
    
    def __init__(self, class_names, class_colors=(_COLORS * 255).astype(np.uint8)):
        """Takes in output from the object detector and adds bounding boxes with 
        appropriate color to the image. Can also write image/video to file if need be.

        Args:
            class_names (list): a list of N strings to translate from class id to class name.
            class_colors (numpy uint8 array): Nx3 array of RGB colors for each class
        """
        self.class_names = class_names
        self.class_colors = class_colors
        self.num_classes = len(self.class_names)
        
    def add_boxes_to_img(self, img, boxes, img_info, add_text=True):
        """Adds boxes to the image at the appropriate location. If box_info is provided,
        will use it to write text. Otherwise will write class label as text.

        Args:
            img (np array): raw image as passed to the detector of shape CxHxW.
            boxes (torch tensor): output of detector.__call__ of shape Mx7 (xywh format)
            img_info (dict): output of detector.__call__.
            add_text (bool): A flag to determine whether to add text or not.
            
            
        Returns:
            modified img with bounding box and possibly text added.
        """
        class_labels = [self.class_names[label_idx] for label_idx in boxes[:, -1].astype(int)]
        
        #opencv does inplace modifications, so just for safety we work with a copy
        img_copy = img.copy()  
            
            
        
        #cv2 wants xyxy format    
        boxes[..., :2], boxes[..., 2:] = boxes[..., :2] - boxes[...,2:] / 2, \
                                         boxes[..., :2] + boxes[..., 2:] / 2
                                        

            
        
        for i,box in enumerate(boxes):
            #write this box using cv2.rectangle
            top_left, bottom_right, object_conf, class_conf  = box[:2], box[2:4], box[4], box[5]
            conf = object_conf * class_conf
            
            text = f"{class_labels[i]} {conf * 100:.1f}%"
            
            box_color = self.class_colors[box[-1].astype(int)]
            
            text_color = (0,0,0) if np.mean(box_color) > (255 / 2) else (255,255,255)
            
            
            
        
    
    