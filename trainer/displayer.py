import matplotlib.pyplot as plt
import numpy as np
import os
import sys

class Table(object):
    def __init__(self):
        self.dir = ''
        self.filenames = []
        self.tables = []

    def find_logfiles(self, dirs):
        if type(dirs) is list:
            for dir in dirs:
                filelist = os.listdir(dir)

        elif type(dirs) is str:
            pass
        else:
            raise ValueError("Type Error")