from lib.modeling.backbone.dfnet import DFNet
from lib.modeling.backbone.lstm import LstmNet
from lib.modeling.autoencoder.df_autoencoder import DFAutoEncoderSimple
from lib.modeling.classifier.dfnet import DFNetFrontEnd
from lib.modeling.evaluator.df_evaluator import DFEvaluator, PREvaluator


def make_model(cfg):
    if cfg.MODEL.NAME == 'dfnet':
        backbone = make_backbone(cfg)
        return DFNetFrontEnd(backbone)
    elif cfg.MODEL.NAME == 'autoencoder_simple':
        return DFAutoEncoderSimple()
    elif cfg.MODEL.NAME == 'lstm':
        backbone = make_backbone(cfg)
        return DFNetFrontEnd(backbone)

def make_backbone(cfg):
    if cfg.MODEL.NAME == 'dfnet':
        return DFNet(cfg.MODEL.CLASSES)
    elif cfg.MODEL.NAME == 'lstm':
        args = dict(
            input_size=64,
            hidden_size=200,
            result_classes_count=cfg.MODEL.CLASSES,
            num_layer=1,
        )
        return LstmNet(**args)

def make_evaluator(cfg):
    if cfg.TEST.METRIC == 'ACC':
        return DFEvaluator()
    elif cfg.TEST.METRIC == 'PR':
        return PREvaluator(cfg.MODEL.CLASSES)
    
