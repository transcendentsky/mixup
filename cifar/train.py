#coding:utf-8
"""
This script is the entry of the programme
"""
from __future__ import print_function
import sys
import argparse

from lib.utils.config_parse import cfg_from_file, cfg
# from lib.multitrain import mtrain, single_train
from lib.train_entry import SingleTrain, MultiTrain


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

  _args = parse_args()
  if _args.config_file is not None:
    cfg_from_file(_args.config_file)

  assert cfg.STRATEGY != '', 'No Strategy Setted.'

  if cfg.STRATEGY == 'single':
    s = SingleTrain(1002)
    s.single_train()
  elif cfg.STRATEGY == 'single_adamw':
    s = SingleTrain(1002)
    s.single_train_adamw()
  else:
    m = MultiTrain(cfg.STRATEGY)
    m.multi_train()
