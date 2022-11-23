#TAKEN FROM my other ReID project on github



import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from loguru import logger

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
    
def get_backbone():
    """Can be modified to change backbone."""
    resnet50 = torchvision.models.resnet50(weights="DEFAULT")
    
    for i, layer in enumerate(resnet50.modules()):
        if isinstance(layer, nn.BatchNorm2d):
            #use batch stats during inference
            layer.track_running_stats = False 
            
    #dummy variable to transport info
    resnet50._infeatures_temp = resnet50.fc.in_features
    #discard fc layer
    resnet50.fc = Identity()
    
    return resnet50
    
    
class Net(nn.Module):
    """The full Siamese network with BNNeck."""
    def __init__(self, num_classes, backbone=get_backbone(), device=torch.device("cuda"),
                 normalize=False):
        """num_classes is the number of identities being classified.
        
        If the normalize flag is true, will normalize feature vectors during inference."""
        #gives features, used to calculate triplet and center loss
        super().__init__()
        self.backbone = backbone.to(device) 
        
        in_features = backbone._infeatures_temp
        
        #output of this is used during inference
        self.batch_norm = nn.BatchNorm1d(num_features=in_features,
                                         track_running_stats=False).to(device)
        
        self.fc = nn.Linear(in_features=in_features, 
                            out_features=num_classes,
                            bias=False).to(device)
        #self.register_buffer("imagenet_mean",torch.tensor([0.485, 0.456,0.406]).reshape((1,-1,1,1)))
        #self.register_buffer("imagenet_sd", torch.tensor([0.229, 0.224, 0.225]).reshape((1,-1,1,1)))
        
        # self.imagenet_mean = nn.Parameter(torch.tensor([0.485, 0.456,0.406]).reshape((1,-1,1,1)))
        # self.imagenet_sd = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).reshape((1,-1,1,1)))
        
        # self.imagenet_mean.requires_grad = False
        # self.imagenet_sd.requires_grad = False
        
        self.normalize = normalize
        
    def forward(self, x):
        """During test time returns features and output of fc layers.
        During inference time returns output after batch normalization.
        
        Potential minor issue: When in validation mode should the model
        act as if in inference mode and return batch normalized outputs?"""
        #normalize
        #x = (x - self.imagenet_mean) / self.imagenet_sd
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1,-1,1,1)).to(x.device)
        sd = torch.tensor([0.229, 0.224, 0.225]).reshape((1,-1,1,1)).to(x.device)
        
        x = (x - mean) / sd
       
        features = self.backbone(x)
        
        if self.training:
            class_probs = self.fc(self.batch_norm(features))
            return class_probs, features 
        else: #inference mode
            features = self.batch_norm(features)
            if self.normalize:
                features = F.normalize(features, dim=1)
            return features

