import torch

def make_optimizer(cfg, model: torch.nn.Modele):
    lr = cfg.TRAIN.BASE_LR
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    
    return optimizer

def make_scheduler(cfg, optimizer):
    return None
