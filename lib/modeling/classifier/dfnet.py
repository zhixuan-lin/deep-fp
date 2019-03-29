import torch
from ..build import make_backbone
from torch.nn import CrossEntropyLoss

class DFNetFrontEnd:
    def __init__(self, cfg):
        self.backbone = make_backbone(cfg)
        self.criterion =  CrossEntropyLoss()
        
    def forward(self, x, targets):
        """
        :param x: shape (N, D)
        :param target: shape (N,)
        :return: loss
        """
        
        preds = self.backbone(x)
        return self.criterion(preds, targets)
        
        

