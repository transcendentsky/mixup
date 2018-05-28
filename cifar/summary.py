from __future__ import print_function
from tensorboardX import SummaryWriter
import argparse
import os
from utils.config_parse import cfg

writer = None
init_flag = False

def initSummary():
    global writer
    _LOG_FOLDER = cfg.EXP_DIR + '/logs'
    # print(_LOG_FOLDER)
    assert _LOG_FOLDER is not None
    print("\nThe Saving Dir :  ", _LOG_FOLDER)
    writer = SummaryWriter(_LOG_FOLDER)
    # print('type writer  ', type(writer))

def get_writer():
    global init_flag
    if init_flag is False:
        initSummary()
        init_flag = True
    return writer
