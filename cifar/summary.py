from __future__ import print_function
from tensorboardX import SummaryWriter
import argparse
from LearningSchedule import *
import os

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--epoch', default=0, type=int)
parser.add_argument('--sess', default='mixup_default', type=str, help='session id')
parser.add_argument('--seed', default=0, type=int, help='rng seed')
parser.add_argument('--alpha', default=0., type=float, help='interpolation strength (uniform=1., ERM=0.)')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay (default=1e-4)')
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--wr', '-w', action='store_true')
parser.add_argument('--verbose', '-v', default=1, type=int)
parser.add_argument('--ga', '-g', action='store_true')
parser.add_argument('--redir', default=None, type=str)
parser.add_argument('--labelsm', action='store_true', default=False)
args = parser.parse_args()

_LOG_FOLDER = None
result_folder = _LOG_FOLDER
resume_folder = args.redir
resume_folder = r'results/DataParallel/CustomLearningRateScheduler_staging/mixup1.0/labelsm2/'
writer = None

if args.resume is True:
    assert resume_folder is not None, "Resume folder is NONE......"

if args.verbose:
    if args.alpha > 0.0:
        print("\n#####  Using mixup #####\n  alpha = ", args.alpha)
    else:
        print("\n#####  No mixup #####")
    if args.wr:
        print("\n#####  Using warm rstart  #####")
    else:
        print("\n#####  No warm rstart  #####")
    if args.labelsm:
        print("\n#####  Using label smootihing  #####")
    else:
        print("\n#####  No Label Smoothing  #####")

if args.wr:
    sche = CustomLearningRateScheduler_wr(ti=10)
    if args.resume:
        sche.ti = input("sche ti")
        sche.base = input("sche base")
else:
    sche = CustomLearningRateScheduler_staging()
    # sche = CustomLearningRateScheduler3()

# if not os.path.exists(result_folder):
#     os.makedirs(result_folder)

def initSummary(netname, lrschename, mixup, **kwargs):
    global _LOG_FOLDER
    global writer
    _LOG_FOLDER = 'results/' + netname + '/' + lrschename + '/' + str(mixup) + '/'
    for arg in kwargs:
        _LOG_FOLDER += kwargs[arg] + '/'
    assert _LOG_FOLDER is not None, 'Setting result folder ERROR......'

    # if not os.path.exists(result_folder):
    #     os.makedirs(result_folder)

    print("\nThe Saving Dir :  ", _LOG_FOLDER)
    writer = SummaryWriter(_LOG_FOLDER)
    print('type writer  ', type(writer))

def get_writer():
    return writer

def get_resfolder():
    return _LOG_FOLDER


