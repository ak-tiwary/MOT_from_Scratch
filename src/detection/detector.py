#This will create a detector class which will handle detections for one image.
#The detector will have access to a yolox model, along with other necessary information
#like test_size (obtained from the desired exp) to resize input images before evaluating.
#It will also handle postprocessing of model outputs like batched nms, focusing on desired
#classes only, etc.
#It will work with torch tensors primarily, and leave convertions to and from numpy arrays 
#to a pre/postprocessing step.