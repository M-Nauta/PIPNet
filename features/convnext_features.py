import torch
import torch.nn as nn
from torchvision import models

def replace_convlayers_convnext(model, threshold):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_convlayers_convnext(module, threshold)
        if isinstance(module, nn.Conv2d):
            if module.stride[0] == 2:
                if module.in_channels > threshold: #replace bigger strides to reduce receptive field, skip some 2x2 layers. >100 gives output size (26, 26). >300 gives (13, 13)
                    module.stride = tuple(s//2 for s in module.stride)
                    
    return model

def convnext_tiny_26_features(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()    
        model = replace_convlayers_convnext(model, 100) 
    
    return model

def convnext_tiny_13_features(pretrained=False, **kwargs):
    model = models.convnext_tiny(pretrained=pretrained, weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    with torch.no_grad():
        model.avgpool = nn.Identity()
        model.classifier = nn.Identity()    
        model = replace_convlayers_convnext(model, 300) 
    
    return model
