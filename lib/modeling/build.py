from lib.modeling.backbone.dfnet import DFNet
from lib.modeling.classifier.dfnet import DFNetFrontEnd


def make_model(cfg):
    backbone = make_backbone(cfg)
    return DFNetFrontEnd(backbone)

def make_backbone(cfg):
    return DFNet(cfg.MODEL.CLASSES)
