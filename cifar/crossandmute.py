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

############################    Data Augmentation    ##################################
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

#############################   Resume First Model    ##################################
assert args.resume is True, "Should Resume First"
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    epoch = input("epoch ? ")
    # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    # checkpoint = torch.load('./checkpoint/ckpt.t7.' + args.sess + '_' + str(args.seed))
    ckfile = result_folder + 'ckpt.t7.' + args.sess + '_' + str(args.seed) + '_epoch' + str(epoch)
    checkpoint = torch.load(ckfile)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
    print("Best acc: ", best_acc)
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
    # net = wrn(depth=28, num_classes=10)

# result_folder = './results_g/'
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

logname = result_folder + net.__class__.__name__ + '_' + args.sess + '_' + str(args.seed) + '.csv'

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)

############################    Import Custom Optimizer    ##########################
import CustomOpm
# optimizer = CustomOpm.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9, weight_decay=args.decay)
# optimizer = CustomOpm.Adam(net.parameters(), lr=0.001, momentum=0.9, weight_decay=args.decay)
optimizer = CustomOpm.GSGD(net.parameters(), lr=base_learning_rate, weight_decay=0.0005)

###########################    Training Step    ###########################
def train(epoch, is_ga):
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
            _, check_grads, check_g2, check_b2 = optimizer.step(gstep=epoch, check=True, is_ga=is_ga)
            angle = check_grads / np.sqrt(check_g2 * check_b2)
            writer.add_scalar("grads_variance", check_grads, batch_idx / 39 + epoch * 391 / 39)
            writer.add_scalar("grads^2", check_g2, batch_idx / 39 + epoch * 391 / 39)
            writer.add_scalar("buf^2", check_b2, batch_idx / 39 + epoch * 391 / 39)
            writer.add_scalar("angel", angle, batch_idx / 39 + epoch * 391 / 39)
        else:
            optimizer.step(gstep=epoch,check=False, is_ga=is_ga)
        if is_ga:
            print("GA break")
            return 0, 0
        train_loss += loss.data[0]
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    return (train_loss / batch_idx, correct / total)

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
    if acc > best_acc and epoch > 50:
        best_acc = acc
        checkpoint(acc, epoch)
    return (test_loss / batch_idx, correct / total)

#####################    Import Custom Learning Schedule   ################
from LearningSchedule import *
sche = CustomLearningRateScheduler_staging(ti=10)
#####################    restore Pre Model    #####################
def restore_pre():
    net.train()
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
        print("Epoch ", start_epoch)
        optimizer.step(gstep=start_epoch, check=False, is_ga=False, restore_buf=True)
        break
restore_pre()

#####################    restore Latter Model    ####################
if True:
    # Load checkpoint.
    print('==> [Second] Resuming from checkpoint..')
    epoch = input("epoch ? ")
    ckfile = result_folder + 'ckpt.t7.' + args.sess + '_' + str(args.seed) + '_epoch' + str(epoch)
    checkpoint = torch.load(ckfile)
    net = checkpoint['net']
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])
    print("Best acc: ", best_acc)

    sche.base = 120
    sche.ti = 60
    print("sche.base,  sche.ti")
    print(sche.base, sche.ti)

def first_cross():
    net.train()
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
        print("Epoch ", start_epoch)
        optimizer.step(gstep=start_epoch, check=False, is_ga=True)
        break
print("###  first Cross  ###")
first_cross()

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

#################    Manual Early Stopping   ################
import signal
global global_epoch
def sigint_handler(signum, frame):
    global global_epoch
    global acc
    print("Early Stopping......")
    checkpoint(acc, global_epoch)
    print("Checkpoint Saved.  eopch: ",global_epoch)
    exit()
signal.signal(signal.SIGINT, sigint_handler)
#############################################################

#####################    Start Training    #######################
### Record Learning Schedule by make file
schename = result_folder + net.__class__.__name__ + '_' + sche.__class__.__name__ + '_mixup-' + str(args.alpha)
if not os.path.exists(schename):
    with open(schename, 'w') as schefile:
        pass

if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow(['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

for epoch in range(start_epoch, 300):
    global_epoch = epoch
    is_ga = sche.adjust_learning_rate(optimizer, epoch)
    # if is_ga == True:
    # is_ga = True
    print("[Debug] is_ga True")
    train_loss, train_acc = train(epoch, is_ga)
    test_loss, test_acc = test(epoch)
    # with open(logname, 'a') as logfile:
    #     logwriter = csv.writer(logfile, delimiter=',')
    #     logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])

    writer.add_scalar('val_loss', test_loss, epoch)
    writer.add_scalar('val_acc', test_acc, epoch)
    writer.add_scalar('loss', train_loss, epoch)
    writer.add_scalar('acc', train_acc, epoch)

# writer.export_scalars_to_json("./all_scalars.json")
writer.close()
