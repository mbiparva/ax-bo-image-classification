"""Config file setting hyperparameters

This file specifies default config options. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.
"""

from easydict import EasyDict as edict
import os
import datetime

__C = edict()
cfg = __C   # from config.py import cfg


# ================
# GENERAL
# ================

# Set modes
__C.TRAINING = True
__C.VALIDATING = True

# Root directory of project
__C.ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Dataset directory
__C.DATASET_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, 'dataset'))

# Model directory
__C.MODELS_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, 'lib', 'models'))

# Experiment directory
__C.EXPERIMENT_DIR = os.path.abspath(os.path.join(__C.ROOT_DIR, 'experiment'))

# Set meters to use for experimental evaluation
__C.METERS = ['loss', 'label_accuracy']

# Use GPU
__C.USE_GPU = True

# Default GPU device id
__C.GPU_ID = 0

# Number of epochs
__C.NUM_EPOCH = 20

# Dataset name
__C.DATASET_NAME = ('MNIST', 'CIFAR10', 'CIFAR100')[2]

if __C.DATASET_NAME in ('MNIST', 'CIFAR10'):
    # Number of categories
    __C.NUM_CLASSES = 10
elif __C.DATASET_NAME in ('CIFAR100', ):
    # Number of categories
    __C.NUM_CLASSES = 100
else:
    raise NotImplementedError

# Normalize database samples according to some mean and std values
__C.DATASET_NORM = True

# Input data size
__C.SPATIAL_INPUT_SIZE = (32, 32)
__C.CHANNEL_INPUT_SIZE = 3

# Set parameters for snapshot and verbose routines
__C.MODEL_ID = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
__C.SNAPSHOT = True
__C.SNAPSHOT_INTERVAL = 5
__C.VERBOSE = True
__C.VERBOSE_INTERVAL = 50
__C.VALID_INTERVAL = 1

# Network Architecture
__C.NET_ARCH = ('alexnet', 'vggnet', 'resnet')[0]

# Pre-trained network
__C.PRETRAINED_MODE = (None, 'Custom')[0]

# Path to the pre-segmentation network
__C.PT_PATH = os.path.join(__C.EXPERIMENT_DIR, 'snapshot', '20181010_124618_219443', '079.pt')

# ================
# Training options
# ================
if __C.TRAINING:
    __C.TRAIN = edict()

    # Images to use per minibatch
    __C.TRAIN.BATCH_SIZE = 128

    # Shuffle the dataset
    __C.TRAIN.SHUFFLE = True

    # Learning parameters are set below
    __C.TRAIN.LR = 1e-3
    __C.TRAIN.WEIGHT_DECAY = 1e-5
    __C.TRAIN.MOMENTUM = 0.90
    __C.TRAIN.NESTEROV = False
    __C.TRAIN.SCHEDULER_MODE = False
    __C.TRAIN.SCHEDULER_TYPE = ('step', 'step_restart', 'multi', 'lambda', 'plateau')[2]
    __C.TRAIN.SCHEDULER_MULTI_MILESTONE = [10]

# ================
# Validation options
# ================
if __C.VALIDATING:
    __C.VALID = edict()

    # Images to use per minibatch
    __C.VALID.BATCH_SIZE = __C.TRAIN.BATCH_SIZE

    # Shuffle the dataset
    __C.VALID.SHUFFLE = False
