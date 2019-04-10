import argparse
import sys
if '.' not in sys.path:
    sys.path.insert(0, '.')
    
import torch
from lib.data.build import make_dataloader
from lib.engine.eval import test
from lib.modeling.build import make_model, make_evaluator
from lib.utils.checkpoint import Checkpointer
from lib.config import cfg

def test_net(cfg):
    print("Testing model")
    # get device
    device = torch.device(cfg.MODEL.DEVICE)
    
    # make model
    model = make_model(cfg).to(device)
    
    # use multiple GPUs
    if cfg.MODEL.PARALLEL:
        model = torch.nn.DataParallel(model)
    
    # make dataloader
    dataloader = make_dataloader(cfg, mode='test')
    

    # checkpoint directory
    output_dir = cfg.MODEL_DIR
    checkpointer = Checkpointer(model, save_dir=output_dir)
    
    # if we intend to resume, this loads model, optimizer, scheduler, and arguments
    checkpointer.load()
    
    # build evaluator
    evaluator = make_evaluator(cfg)
    
    results = test(model, device, dataloader, evaluator)
    
    print('Final accuracy: {:.3f}'.format(results))
    
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
    
    test_net(cfg)
    
if __name__ == '__main__':
    main()
