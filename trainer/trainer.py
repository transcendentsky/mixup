import yaml
import numpy as np
import sys
import os

# Out to log file
f_handler = open('out.log','w')
__console__ = sys.stdout
# import torch

"""
TO DO:
1. run complete running for one config
2. 
"""


class Trainer(object):
    def __init__(self):
        self.index = 0
        self.loss_detected = np.zeres(20)
        pass

    def import_yml(self):
        pass

    def create_yml(self):
        pass

    def detect_loss(self, latest_loss):
        self.loss_detected[self.index % 20] = latest_loss

        if tmp != 0:
            tmp_back = self.loss_detected[index:20]
            tmp_front = self.loss_detected[0:index]
            tmp = np.zeros(20)
            tmp[0:len(tmp_back)] = tmp_back
            tmp[len(tmp_back):20] = tmp_front
        else:
            tmp = self.loss_detected

        y = tmp[0:10]
        x = np.arange(10)
        y_mean = np.mean(y)
        top = np.sum(y_1 * x) - len(x) * x_mean * y_mean
        bottom = np.sum(x ** 2) - len(x) * x_mean ** 2
        k_1 = top / bottom

        y = tmp[10:]
        x = np.arange(10)
        y_mean = np.mean(y)
        top = np.sum(y_1 * x) - len(x) * x_mean * y_mean
        bottom = np.sum(x ** 2) - len(x) * x_mean ** 2
        k_2 = top / bottom

        y = tmp
        x = np.arange(20)
        y_mean = np.mean(y)
        top = np.sum(y_1 * x) - len(x) * x_mean * y_mean
        bottom = np.sum(x ** 2) - len(x) * x_mean ** 2
        k = top / bottom

        ###  update  ###
        self.index += 1

        if k >= 0:
            """  To Stop Training OR Adjust Learning rate  """
            pass
        else:
            if k_2 + k_1>= 0:
                # Adjust
                pass
            else:
                pass

    def adjust_lr(self):
        """Adjust Learning Rate"""
        pass

    ###########################################################
    ###    Hyper Params Adjusting                           ###
    ###

    def switch_initializer(self):
        pass

    def set_base_lr(self):
        pass

    #####################################################
    ###    Auto Running                               ###
    def run(self):
        dada


    def run_script(self):
        cfg_file = 'cfgs/' + 'ga_vgg.yml'
        cammand = 'python {} --cfg={}'.format("train.py", )
        try:
            os.system(cammand)
        except:
            pass

    def auto_log(self):
        pass

    def auto_test(self):
        pass


    def auto_restart(self):
        pass


    ######################################################
    ##      Auto Model Modified

    def add_bn(self):
        pass

    def set_activiation(self):
        pass