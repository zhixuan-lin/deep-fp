import torch
from torch import nn
from torch.nn import CrossEntropyLoss

class DFNetFrontEnd(nn.Module):
    def __init__(self, backbone):
        nn.Module.__init__(self)
        self.backbone = backbone
        self.criterion =  CrossEntropyLoss()
        
    def forward(self, x, targets):
        """
        :param x: shape (N, D)
        :param target: shape (N,)
        :return: loss
        """
        
        preds = self.backbone(x)
        return self.criterion(preds, targets)
        
        

