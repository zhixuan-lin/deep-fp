import argparse
import os
import sys
if '.' not in sys.path:
    sys.path.insert(0, '.')
    
import torch
from lib.data.build import make_dataloader
from lib.engine.trainer import train
from lib.modeling.build import make_model, make_evaluator
from lib.solver.build import make_optimizer, make_scheduler
from lib.utils.checkpoint import Checkpointer
from lib.config import cfg
from lib.utils.tensorboard import TensorBoard

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
    dataloader = make_dataloader(cfg, mode='train')
    
    # make arguments that should be checkpointed
    arguments = {}
    arguments['iteration'] = 0
    arguments['epoch'] = 0

    # checkpoint directory
    output_dir = os.path.join(cfg.MODEL_DIR, cfg.EXP.NAME)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # checkpoint every
    checkpointer = Checkpointer(model, optimizer, scheduler, cfg.TRAIN.NUM_CHECKPOINT, output_dir)
    
    # if we intend to resume, this loads model, optimizer, scheduler, and arguments
    if cfg.TRAIN.RESUME:
        arguments_checkpoint = checkpointer.load()
        arguments.update(arguments_checkpoint)
    
    
    # max epochs
    max_epochs = cfg.TRAIN.MAX_EPOCHS

    checkpoint_period = cfg.TRAIN.CHECKPOINT_PERIOD
    # validation settings
    dataloader_val = None
    evaluator = None
    if cfg.VAL.IS_ON:
        dataloader_val = make_dataloader(cfg, mode='val')
        evaluator = make_evaluator(cfg)
        
    tensorboard = None
    
    if cfg.TENSORBOARD.IS_ON:
        logdir = os.path.join(cfg.TENSORBOARD.LOG_DIR, cfg.EXP.NAME)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        tensorboard = TensorBoard(
            logdir=logdir,
            scalars=cfg.TENSORBOARD.TARGETS.SCALAR,
            images=cfg.TENSORBOARD.TARGETS.IMAGE,
            resume=cfg.TRAIN.RESUME
        )

    train(
        model,
        dataloader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        max_epochs,
        arguments,
        tensorboard,
        
        dataloader_val=dataloader_val,
        evaluator=evaluator
    )
    
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file',
        default='',
        help='Path to config file',
        type=str
    )
    parser.add_argument(
        'opts',
        help="Modify config options using the command line",
        default=None,
        nargs=argparse.REMAINDER
    )
    args = parser.parse_args()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    train_net(cfg)
    
if __name__ == '__main__':
    main()
