from lib.modeling.backbone.dfnet import DFNet
from lib.modeling.classifier.dfnet import DFNetFrontEnd
from lib.modeling.evaluator.df_evaluator import DFEvaluator, PREvaluator


def make_model(cfg):
    backbone = make_backbone(cfg)
    return DFNetFrontEnd(backbone)

def make_backbone(cfg):
    return DFNet(cfg.MODEL.CLASSES)

def make_evaluator(cfg):
    if cfg.TEST.METRIC == 'ACC':
        return DFEvaluator()
    elif cfg.TEST.METRIC == 'PR':
        return PREvaluator(cfg.MODEL.CLASSES)
    
