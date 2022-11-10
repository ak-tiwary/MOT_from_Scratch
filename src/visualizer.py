import numpy as np
import torch
import torchvision
import cv2
from loguru import logger

    
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
 
 
COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
   
    

class Visualizer:
    
    def __init__(self, class_names=COCO_CLASSES, class_colors=_COLORS):
        """Takes in output from the object detector and adds bounding boxes with 
        appropriate color to the image. Can also write image/video to file if need be.

        Args:
            class_names (list): a list of N strings to translate from class id to class name.
            class_colors (numpy uint8 array): Nx3 array of RGB colors for each class
        """
        self.class_names = class_names
        self.class_colors = (class_colors * 255).astype(np.uint8)
        self.num_classes = len(self.class_names)
        
        
    def add_boxes_to_img(self, boxes, img_info, add_text=True):
        """Adds boxes to the image at the appropriate location. If box_info is provided,
        will use it to write text. Otherwise will write class label as text.

        Args:
            img (np array): raw image as passed to the detector of shape HxWxC. Also available from img_info["raw_img"]
            boxes (torch tensor): output of detector.__call__ of shape Mx7 (xywh format)
            img_info (dict): output of detector.__call__.
            add_text (bool): A flag to determine whether to add text or not.
            
            
        Returns:
            modified img with bounding box and possibly text added.
        """
        img = img_info['raw_img']
        ratio = img_info['ratio']
        boxes = boxes.to(torch.device("cpu")).numpy()
        boxes[..., :4] /= ratio
        logger.log("INFO", f"boxes shape {boxes.shape}")
        logger.log("INFO", f"boxes[:,-1] {boxes[:,-1]}")
        class_labels = [self.class_names[label_idx] for label_idx in boxes[:, -1].astype(int)]
        
        #opencv does inplace modifications, so just for safety we work with a copy
        img_copy = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
            
            
        
        #cv2 wants xyxy format    
        boxes[..., :2], boxes[..., 2:4] = boxes[..., :2] - boxes[...,2:4] / 2, \
                                         boxes[..., :2] + boxes[..., 2:4] / 2
                                        

            
        
        for i,box in enumerate(boxes):
            #write this box using cv2.rectangle
            top_left, bottom_right, object_conf, class_conf  = box[:2], box[2:4], box[4], box[5]
            x1,y1 = top_left
            #print(f"x1 = {x1}, y1 = {y1}")
            x2,y2 = bottom_right
            #opencv wants integers for coordinates
            x1,y1,x2,y2 = map(int, [x1,y1,x2,y2])
            conf = object_conf * class_conf
            
            text = f"{class_labels[i]} {conf * 100:.1f}%"
            
            box_color = tuple(self.class_colors[box[-1].astype(int)].tolist())
            print(f"box_color = {box_color}")
            text_color = (0,0,0) if np.mean(box_color) > (255 / 2) else (255,255,255)
            
            #drawn bounding box
            img_copy = cv2.rectangle(img_copy, (x1,y1), (x2,y2), color=box_color,thickness=3)
            
            (text_width,_), _ = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
                                                          fontScale=.8, thickness=1)
            img_copy = cv2.rectangle(img_copy, (x1,y1-30), (x1 + text_width, y1), color=box_color,thickness=-1)
            img_copy = cv2.putText(img_copy, text=text, org=(x1,y1-5),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                   fontScale=0.8, color=text_color, thickness=1)
            
        return cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR)
            
            
        
    
    