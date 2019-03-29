from lib.modeling.backbone.dfnet import DFNet
from lib.modeling.classifier.dfnet import DFNetFrontEnd


def make_model(cfg):
    return DFNetFrontEnd(cfg)

def make_backbone(cfg):
    return DFNet(cfg.MODEL.CLASSES)
