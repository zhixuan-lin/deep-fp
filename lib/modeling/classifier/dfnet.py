import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax

class DFNetFrontEnd(nn.Module):
    def __init__(self, backbone):
        nn.Module.__init__(self)
        self.backbone = backbone
        self.criterion =  CrossEntropyLoss()
        
    def forward(self, x, targets=None):
        """
        :param x: shape (B, D)
        :param target: shape (B)
        :return: loss
        """
        
        # (B, C)
        preds = self.backbone(x)
        
        if not targets is None:
            return self.criterion(preds, targets)
        
        # else, return predicted results:
        return softmax(preds, dim=1)
        
        
        

