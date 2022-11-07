#This will create a detector class which will handle detections for a given batch of images/videos.
# Will output boxes in a form useful for the tracking pipeline, as well as a detection pipeline.
#The detector will have access to a yolox model, along with other necessary information
#like test_size (obtained from the desired exp) to resize input images before evaluating.
#It will also handle postprocessing of model outputs like batched nms, focusing on desired
#classes only, etc. For this purprose, it will maintain desired nms, test_conf, test_size values.
#It will work with normalized torch tensors primarily, and leave denormalization and convertions to and from numpy arrays 
#to a pre/postprocessing step.
#It will also load the model from a checkpoint if provided.
import numpy as np
import torch
import cv2

class Detector:
    def __init__(self, exp,  filter_classes=None, device=torch.device("cpu"),test_conf=None, test_size=None, nms_thres=None, class_agnostic=False, chkpt=None):
        """Handles detections using YOLOX. Input images will be numpy images in RGB order and shape
        (N)xCxHxW. Output boxes will be in xywh format (center, width, height).

        Args:
            exp (Exp): A YOLOX experiment object with settings already provided (function parameters will override 
                       these). 
            filter_classes (list | None) : If not None, will only keep those boxes with classes among those provided.
            device (torch.device): Defaults to cpu
            test_conf (float | None): A threshold value below which detections are rejected. Default is  0.01.
            test_size (tuple | None): The size to resize input images to before feeding into yolox. Defaults to 
                                      (640,640)
            nms_thres (float | None): Defaults to 0.7.
            class_agnostic (bool): Default false. Set to true if non-maximal suppression should ignore class.
            chkpt (string | None) : path to checkpoint for the model provided with exp.

        """
        self.exp = exp
        self.model = exp.get_model().to(device)
        self.model.eval() #inference mode
        
        self.filter_classes = filter_classes
        self.device = device
        
        if nms_thres is not None:
            self.nms_thres = nms_thres
        else:
            self.nms_thres = self.exp.nms_thre
            
        if test_size is not None:
            self.test_size = test_size
        else:
            self.test_size = self.exp.test_size
            
        if self.test_conf is not None:
            self.test_conf = test_conf
        else:
            self.test_conf = self.exp.test_conf
            
        self.class_agnostic = class_agnostic

        if chkpt is not None:
            checkpoint = torch.load(chkpt, map_location="cpu")
            self.model.load_state_dict(checkpoint["model"])
            
        
            
    def __call__(self, img):
        """Will run yolox on a single image of shape CxHxW.

        Args:
            img (numpy array):unnormalized image in RGB order.
            
        Returns:
            a tuple (bboxes, image_info)
        """
        assert len(img.shape) == 3, f"Shape of imgs is {img.shape} but it should be CxHxW"
        img_info = {}
        img_info["raw_img"] = img
        img_info["ratio"] = self._get_ratio(img)
        
        img_tensor = self._preprocess(img, img_info["ratio"]).to(self.device)
        with torch.no_grad():
            unprocessed_boxes = self.model(img_tensor)
            
        
        #shape of output is NxMx (4 + 1 + C) where M is the number of boxes per image and C is num_classes
  
    def postprocess(self, unprocessed_boxes):
        """Will convert the output of yolox and perform nms, filtering desired classes, converting class scores to class prediction,
        

        Args:
            unprocessed_boxes (torch tensor): Shape N x M x (4 + 1 + C), where N is number of images, M is objects per image,
                                              and the last dimension has box coordinates in XYWH format, object confidence, and 
                                              class probability scores 

        Returns:
            a list with the ith index having shape M_i x (4 + 1 + 1 + 1), where M_i is the number of objects in image i, and the 
            second dimension is box coordinates, object confidence, class confidence, and class prediction.
        """
  
    def _preprocess(self, img, ratio):
        """Given a numpy image and desired ratio, rescales the image, normalizes using ImageNet mean and std.,
        and converts to torch tensor."""
        
        img = self._resize(img, ratio, to_float=True)
        img /= 255 
        
        means = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
        stds=np.array([0.229,0.224,0.225]).reshape((3,1,1))
        
        normalized_img = (img - means)/stds
        
        return torch.from_numpy(normalized_img)
        
    def _resize(self, img, ratio, to_float=True):
        """Resizes an image to the given ratio and pads with (114,114,114) (gray) as in the demo from YOLOX. 
        If to_float, converts np.array from uint8 to float."""
        
        h, w = img.shape[1]
        new_h, new_w = int(h * ratio), int(w * ratio)
        
        padded_img = np.ones(self.test_size, dtype=np.uint8) * 114
        resized_img = cv2.resize(src=img, dsize=(new_w,new_h), interpolation=cv2.INTER_LINEAR)
        padded_img[:new_h, :new_w] = resized_img
        
        if to_float:
            padded_img = np.ascontiguousarray(padded_img,dtype=np.float32)
        return padded_img
        

    def _get_ratio(self, img):
        """Returns the resize ratio for the given image."""
        h, w = img.shape[1:]
        test_h, test_w = self.test_size

        ratio = min(test_h / h, test_w / w)
        return ratio