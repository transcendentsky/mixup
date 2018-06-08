from __future__ import print_function
import os
import sys

class DebugLogger(object):
    def __init__(self):
        self._DEBUG = False
        self.verbose = False
        pass

    def ll(self,*args):
        if self._DEBUG:
            print(args)

    def verbose_print(self, *args, **kwargs):
        if self.verbose:
            for x in args:
                print(x, end=' ')
            for key, value in kwargs.iteritems():
                print(value, end=' ')
            print(' ')


