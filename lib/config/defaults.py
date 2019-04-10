"""
    Default setting for the network,
    Could be overwrite with training configs
"""

import os
from yacs.config import CfgNode as CN


_C = CN()

# -----------------------------------------------------------------------------
# EXP
# -----------------------------------------------------------------------------
_C.EXP = CN()
_C.EXP.NAME = 'OpenWorldNoDef'

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.NAME = "dfnet"
_C.MODEL.TEST = False
_C.MODEL.CLASSES = 96
_C.MODEL.DEVICE = "cpu"
_C.MODEL.PARALLEL = False


# -----------------------------------------------------------------------------
# PATH SETTING
# -----------------------------------------------------------------------------
_C.PATH = CN()
_C.PATH.CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_C.PATH.LIB_DIR = os.path.dirname(_C.PATH.CONFIG_DIR)
_C.PATH.ROOT_DIR = os.path.dirname(_C.PATH.LIB_DIR)
_C.PATH.DATA_DIR = os.path.join(_C.PATH.ROOT_DIR, 'data')


# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TRAIN = 'OpenWorldNoDef.train'
_C.DATASET.VAL = 'OpenWorldNoDef.val'
_C.DATASET.TEST = 'OpenWorldNoDef.test_mon'


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4


# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# resume training?
_C.TRAIN.RESUME = True

# number of epochs
_C.TRAIN.MAX_EPOCHS = 30

# batch size
_C.TRAIN.BATCH_SIZE = 128

# use Adam as default
_C.TRAIN.BASE_LR = 0.002
_C.TRAIN.WEIGHT_DECAY = 0.0005

# scheduler, use MultiStepLR as default
_C.TRAIN.MILESTONES = [10000, 40000, 80000]
_C.TRAIN.GAMMA = 0.1

_C.TRAIN.CHECKPOINT_PERIOD = 2500
_C.TRAIN.NUM_CHECKPOINT = 10

# ---------------------------------------------------------------------------- #
# Validation settings
# ---------------------------------------------------------------------------- #
_C.VAL = CN()

# resume training?
_C.VAL.IS_ON = True
_C.VAL.BATCH_SIZE = 1

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1


# ---------------------------------------------------------------------------- #
# Tensorboard
# ---------------------------------------------------------------------------- #
_C.TENSORBOARD = CN()
_C.TENSORBOARD.IS_ON = True
_C.TENSORBOARD.TARGETS = CN()
_C.TENSORBOARD.TARGETS.SCALAR = ["loss", "accuracy"]
_C.TENSORBOARD.TARGETS.IMAGE = []
_C.TENSORBOARD.LOG_DIR = os.path.join(_C.PATH.ROOT_DIR, "logs", _C.EXP.NAME)


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# default model saving directory
_C.MODEL_DIR = os.path.join(_C.PATH.DATA_DIR, "model", _C.EXP.NAME)


# ---------------------------------------------------------------------------- #
# Path setups
# ---------------------------------------------------------------------------- #
import sys
import os
if _C.PATH.ROOT_DIR not in sys.path:
    sys.path.append(_C.PATH.ROOT_DIR)
    
if not os.path.exists(_C.MODEL_DIR):
    os.makedirs(_C.MODEL_DIR)
    
