from __future__ import print_function

import sys
import os
import argparse

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
        print(args.config_file)
        pass

    # sleep(100)
    print("#####  Finished  ##### ")
    # sys.stdout.flush()
    exit(0)