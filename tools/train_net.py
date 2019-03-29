import sys
if '.' not in sys.path:
    sys.path.insert(0, '.')
    
import torch
from lib.data.build import make_dataloader
from lib.engine.trainer import train
from lib.modeling.build import make_model
from lib.solver.build import make_optimizer, make_scheduler
from lib.utils.checkpoint import Checkpointer
from lib.config import cfg

def train_net(cfg):
    # get device
    device = torch.device(cfg.MODEL.DEVICE)
    
    # make model
    model = make_model(cfg).to(device)
    
    # make solvers
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)
    
    # use multiple GPUs
    if cfg.MODEL.PARALLEL:
        model = torch.nn.DataParallel(model)
    
    # make dataloader
    dataloader = make_dataloader(cfg)
    
    # make arguments that should be checkpointed
    arguments = {}
    arguments['iteration'] = 0

    # checkpoint directory
    output_dir = cfg.MODEL_DIR
    checkpointer = Checkpointer(model, optimizer, scheduler, output_dir)
    
    # if we intend to resume, this loads model, optimizer, scheduler, and arguments
    if cfg.TRAIN.RESUME:
        arguments_checkpoint = checkpointer.load()
        arguments.update(arguments_checkpoint)
    
    # checkpoint every
    checkpoint_period = cfg.TRAIN.CHECKPOINT_PERIOD
    
    # max epochs
    max_epochs = cfg.TRAIN.MAX_EPOCHS
    
    train(
        model,
        dataloader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        max_epochs,
        arguments
    )
    
    return model

def main():
    train_net(cfg)
    
if __name__ == '__main__':
    main()
