from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ast import literal_eval
import os
import os.path as osp
import numpy as np

"""config system.
This file specifies default config options. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use merge_cfg_from_file(yaml_file) to load it and override the default
options.
"""
class AttrDict(dict):

    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        elif name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            self[name] = value


__C = AttrDict()

cfg = __C

__C.SESS = 'mixup_default'
__C.NUM_RUNS = 10

__C.MODEL = AttrDict()
__C.MODEL.NET = 'VGG19'
__C.MODEL.NUM_CLASSES = 10

# ---------------------------------------------------------------------------- #
# Train options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

# batch_size
__C.TRAIN.BATCH_SIZE = 128

# start epoch
__C.TRAIN.START_EPOCH = 0
__C.TRAIN.MAX_EPOCHS = 300

__C.TRAIN.OPTIMIZER = AttrDict()
__C.TRAIN.OPTIMIZER.OPTIMIZER = 'sgd'
__C.TRAIN.OPTIMIZER.BASE_LR = 0.1
__C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
__C.TRAIN.OPTIMIZER.WEIGHT_DECAY = 0.0005

__C.TRAIN.LR_SCHEDULER = AttrDict()
__C.TRAIN.LR_SCHEDULER.SCHEDULER = 'WR'
__C.TRAIN.LR_SCHEDULER.WARM_UP_EPOCHS = 300
__C.TRAIN.LR_SCHEDULER.WR_TI = 10
__C.TRAIN.LR_SCHEDULER.GAMMA = 2.0

__C.TRAIN.SEED = 0

__C.TRAIN.MIXUP = 'mixup'  # 'mixup' or 'nomix'
__C.TRAIN.MIXUP_ALPHA = 0.0

__C.TRAIN.WEIGHT_DECAY = 1e-4
__C.TRAIN.NUM_WORKERS = 2

__C.TRAIN.VERBOSE = True
__C.TRAIN.GA = False
__C.TRAIN.LABEL_SM = False
__C.TRAIN.RESUME_SCOPE = [0, 1000]
__C.TRAIN.CHECKPOINTS_EPOCHS = 10
__C.TRAIN.TEST_STEP = 1
# ---------------------------------------------------------------------------- #
# Test options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()
__C.TEST.BATCH_SIZE = 128
__C.TEST.TEST_SCOPE = [0, 1000]

__C.DATASET = AttrDict()
__C.DATASET.DATASET = 'cifar10'

# ---------------------------------------------------------------------------- #
# Export options
# ---------------------------------------------------------------------------- #
# Place outputs model under an experiments directory
__C.EXP_DIR = ''
__C.LOG_DIR = __C.EXP_DIR
__C.RESUME_CHECKPOINT = ''
__C.CHECKPOINTS_PREFIX = '{}_{}'.format(__C.MODEL.NET, __C.DATASET.DATASET)
__C.PHASE = ['train', 'eval', 'test']


# def _merge_a_into_b(a, b):
#   """Merge config dictionary a into config dictionary b, clobbering the
#   options in b whenever they are also specified in a.
#   """
#   if type(a) is not AttrDict:
#     return

#   for k, v in a.items():
#     # a must specify keys that are in b
#     if k not in b:
#       raise KeyError('{} is not a valid config key'.format(k))

#     # the types must match, too
#     old_type = type(b[k])
#     if old_type is not type(v):
#       if isinstance(b[k], np.ndarray):
#         v = np.array(v, dtype=b[k].dtype)
#       else:
#         raise ValueError(('Type mismatch ({} vs. {}) '
#                           'for config key: {}').format(type(b[k]),
#                                                        type(v), k))
#     # recursively merge dicts
#     if type(v) is AttrDict:
#       try:
#         _merge_a_into_b(a[k], b[k])
#       except:
#         print(('Error under config key: {}'.format(k)))
#         raise
#     else:
#       b[k] = v

def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = _decode_cfg_value(v_)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def update_cfg():
    # global __C.EXP_DIR
    if __C.EXP_DIR is '':
        if __C.TRAIN.GA == True:
            __C.EXP_DIR = osp.join('results', 'with_GA', __C.MODEL.NET, __C.TRAIN.LR_SCHEDULER.SCHEDULER, __C.TRAIN.MIXUP)
        else:
            __C.EXP_DIR = osp.join('results',  __C.MODEL.NET, __C.TRAIN.LR_SCHEDULER.SCHEDULER, __C.TRAIN.MIXUP)
    print("Setting LOG dir: ", __C.EXP_DIR)
    __C.LOG_DIR = __C.EXP_DIR
    __C.CHECKPOINTS_PREFIX = '{}_{}'.format(__C.MODEL.NET, __C.DATASET.DATASET)

    assert __C.TRAIN.MIXUP is 'mixup' or 'nomix', 'cfg.MIXUP setting ERROR '
    if __C.TRAIN.MIXUP is 'mixup':
        print("######   Using Mixup   ######")
        if __C.TRAIN.MIXUP_ALPHA == 0.0:
            __C.TRAIN.MIXUP_ALPHA = 1.0
    else:
        __C.TRAIN.MIXUP_ALPHA = 0.0


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
    update_cfg()
    print("Updated Cfgs")


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, str):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    import sys
    if sys.version_info[0] == 2:
        if type(value_a) == unicode:
            value_a = value_a.encode('utf-8')
        if type(value_b) == unicode:
            value_b = value_b.encode('utf-8')

    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    # encode

    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, str):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
