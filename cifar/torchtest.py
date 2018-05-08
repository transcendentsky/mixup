import torch
from torch.autograd import Variable
import numpy as np


def test1():
    a = np.arange(10)
    b = np.arange(10)
    # choice = np.where(np.random.choice([0,1],(3,3,4)) > 0)
    # print(choice)

    va = Variable(torch.Tensor(a))
    vb = Variable(torch.Tensor(b))

    sum = torch.zeros_like(va.data)
    # sum = va.add(0)
    sum.add_(va.data)
    sum.zero_()
    va.data.add_(0.1,va.data)
    choice = np.random.choice([0, 1], va.data.shape)
    choice = torch.FloatTensor(choice)
    va.data.mul_(1-choice)

    print( 1 - np.random.normal(loc=0.0, scale=1.0, size=va.shape))
    print("sum:", sum)
    print("va:", va)
    print(np.random.normal(size=vb.shape) * vb.data)


def test2():
    t1 = torch.FloatTensor([1., 2.])
    v1 = Variable(t1)
    t2 = torch.FloatTensor([2., 3.])
    v2 = Variable(t2)
    v3 = v1 + v2
    vv3 = torch.zeros_like(v3.data)
    # print(vv3)
    # print(v3)
    vv3.add_(v3.data)

    # v3_detached = v3.detach()
    # vv3 = v3.data.copy_(v3)
    v3.data.add_(t1)
    print(v3, vv3)


def test3():
    t1 = torch.FloatTensor([1., 2.])
    v1 = Variable(t1)
    t2 = torch.FloatTensor([2., 3.])
    v2 = Variable(t2)
    v3 = v1 + v2
    vv3 = v3.data
    vv4 = vv3.add(0)
    vv3.add_(v1.data)
    vv4 = vv4
    print(vv3, vv4)

def test4():
    x1 = np.arange(3*3*4*10).reshape((4,10,3,3))
    x2 = np.arange(3*3*4*10).reshape((4,10,3,3))
    v1 = Variable(torch.FloatTensor(x1))
    v2 = Variable(torch.FloatTensor(x2))
    print(len(v1.data.shape))
    v1max = v1.data[0,0].max()
    # print(v1[0,:,:,:])
    v1.mul_(2.0)
    print(v1)

_LOG_FOLDER = None
def initSummary(netname, lrschename, mixup, **kwargs):
    _LOG_FOLDER = 'results/'+ netname + '/' + lrschename +'/' + str(mixup)+'/'
    for arg in kwargs:
        _LOG_FOLDER += kwargs[arg] + '/'
    assert _LOG_FOLDER is not None, 'Setting result folder ERROR......'
    print("\nThe Saving Dir :  ", _LOG_FOLDER)
    # writer = SummaryWriter(result_folder)

import torch.nn.functional as F
import torch.autograd as autograd
def test5():
    input = torch.randn((3, 2), requires_grad=True)
    print(input)
    target = torch.rand((3, 2), requires_grad=False)
    print(target)
    loss = F.binary_cross_entropy(F.sigmoid(input), target)
    print(loss)
    loss.backward()

def test6():
    input = autograd.Variable(torch.randn(3), requires_grad=True)
    target = autograd.Variable(torch.FloatTensor(3).random_(2))
    print(input, target)
    loss = F.binary_cross_entropy(F.sigmoid(input), target)
    print(loss)
    loss.backward()
# test3()
# test2()
# test1()
# test4()
# initSummary('vgg', 'sche1', 'labelsmooth')
test6()