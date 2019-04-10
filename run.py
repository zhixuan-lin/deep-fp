import argparse

from tools.train_net import train_net
from tools.test_net import test_net
from lib.config import cfg
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-file',
        default='',
        help='Path to config file',
        type=str
    )
    parser.add_argument(
        '--test',
        help='Test or train model',
        action='store_true'
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
    
    if args.test:
        test_net(cfg)
    else:
        train_net(cfg)


if __name__ == '__main__':
    main()
