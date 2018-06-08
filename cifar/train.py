from __future__ import print_function

import sys
import os
import argparse
import numpy as np
# if '/data/software/opencv-3.4.0/lib/python2.7/dist-packages' in sys.path:
#     sys.path.remove('/data/software/opencv-3.4.0/lib/python2.7/dist-packages')
# if '/data/software/opencv-3.3.1/lib/python2.7/dist-packages' in sys.path:
#     sys.path.remove('/data/software/opencv-3.3.1/lib/python2.7/dist-packages')
import cv2
from datetime import datetime

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from utils.config_parse import cfg_from_file, cfg
from multitrain import mtrain, single_train
from train_entry import SingleTrain, MultiTrain

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a ssds.pytorch network')
    parser.add_argument('--cfg', dest='config_file',
            help='optional config file', default=None, type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()
    if args.config_file is not None:
        cfg_from_file(args.config_file)

    assert cfg.STRATEGY != '', 'No Strategy Setted.'
    if cfg.STRATEGY == 'multi':
        m = MultiTrain()
        m.multi_train()
    elif cfg.STRATEGY == 'single':
        s = SingleTrain(1001)
        s.single_train()
