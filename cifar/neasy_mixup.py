'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import os
import argparse
import csv

from models import *
from utils import progress_bar, mixup_data, mixup_criterion
from torch.autograd import Variable
from summary import *
import numpy as np

torch.manual_seed(args.seed)

use_cuda = torch.cuda.is_available()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
batch_size = 128
base_learning_rate = 0.1
if use_cuda:
    # data parallel
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu
    base_learning_rate *= n_gpu

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(resume_folder), 'Error: resume folder is Wrong......'
    # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if args.epoch == 0:
        epoch = input("epoch ? ")
    else:
        epoch = args.epoch
    checkpoint = torch.load(resume_folder + 'ckpt.t7.' + args.sess + '_' + str(args.seed) + '_epoch' + str(epoch))
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
else:
    from wrn import wrn

    print('==> Building model..')
    net = VGG('VGG19')
    # net = PreActResNet18()
    # net = ResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = wrn(depth=28, num_classes=10, widen_factor=10)

#####################  record some info  #################
logname = result_folder + net.__class__.__name__ + sche.__class__.__name__ + 'mixup-' + str(args.alpha)

if use_cuda:
    net.cuda()
    # net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)
import CustomOpm

optimizer = CustomOpm.CSGD(net.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)

#########  Parameters Initialization  #########
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight.data)
        nn.init.xavier_normal(m.bias.data)
if True:
    print("####  Using xavier Initializer  ####")
    net.apply(weights_init)

# optimizer = CustomOpm.Adam(net.parameters(), lr=0.001, momentum=0.9, weight_decay=args.decay)
# optimizer = CustomOpm.GSGD(net.parameters(), lr=base_learning_rate, weight_decay=0.0005)
#################################################################################################
# Training
def train(epoch, is_ga=False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        # generate mixed inputs, two one-hot label vectors and mixing coefficient
        inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, args.alpha, use_cuda)
        optimizer.zero_grad()
        inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
        outputs = net(inputs)

        loss_func = mixup_criterion(targets_a, targets_b, lam)
        loss = loss_func(criterion, outputs)
        loss.backward()
        if batch_idx % 39 == 5:
            _, check_grads, check_g2, check_b2 = optimizer.step(gstep=epoch, check=True)
            angle = check_grads / np.sqrt(check_g2 * check_b2)
            writer.add_scalar("grads_variance", check_grads, batch_idx / 39 + epoch * 391 / 39)
            writer.add_scalar("grads^2", check_g2, batch_idx / 39 + epoch * 391 / 39)
            writer.add_scalar("buf^2", check_b2, batch_idx / 39 + epoch * 391 / 39)
            writer.add_scalar("angel", angle, batch_idx / 39 + epoch * 391 / 39)
        else:
            optimizer.step(gstep=epoch, check=False)

        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return (train_loss / batch_idx, correct / total)


best_acc = 0.
acc = 0.


#################################################################################################
def test(epoch):
    global best_acc
    global acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    # Save checkpoint.
    correct = correct * 1.0
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch)
    return (test_loss / batch_idx, correct / total)


#################################################################################################
def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, result_folder + 'ckpt.t7.' + args.sess + '_' + str(args.seed) + '_epoch' + str(epoch))


####################################    Manual Early Stopping   ######################################
import signal

global_epoch = 0


def sigint_handler(signum, frame):
    global global_epoch
    global acc
    print("Early Stopping......")
    checkpoint(acc, global_epoch)
    print("Checkpoint Saved.  eopch: ", global_epoch)
    exit()


signal.signal(signal.SIGINT, sigint_handler)
#######################################################################################################

for epoch in range(start_epoch, 200):
    global_epoch = epoch
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
    sche.adjust_learning_rate(optimizer, epoch, train_acc, train_loss)

    writer.add_scalar('val_loss', test_loss, epoch)
    writer.add_scalar('val_acc', test_acc, epoch)
    writer.add_scalar('loss', train_loss, epoch)
    writer.add_scalar('acc', train_acc, epoch)

# writer.export_scalars_to_json("./all_scalars.json")
writer.close()
