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
import numpy as np

from models import *
from utils.utils import progress_bar, mixup_data, mixup_criterion
from torch.autograd import Variable

from wrn import wrn
# Custom utils
from summary import get_writer
from utils.config_parse import cfg
from utils.saver import *
import CustomOpm

from LearningSchedule import *

best_acc = 0  # best test accuracy


def single_train(run):
    # print(cfg)
    writer = get_writer()
    torch.manual_seed(cfg.TRAIN.SEED)
    use_cuda = torch.cuda.is_available()
    start_epoch = cfg.TRAIN.START_EPOCH  # start from epoch 0 or last checkpoint epoch
    batch_size = cfg.TRAIN.BATCH_SIZE
    base_learning_rate = cfg.TRAIN.OPTIMIZER.BASE_LR

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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                              num_workers=cfg.TRAIN.NUM_WORKERS)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=cfg.TRAIN.NUM_WORKERS)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Networks
    print('==> Building model..')
    if cfg.MODEL.NET == 'VGG19':
        net = VGG('VGG19')
    elif cfg.MODEL.NET == 'PreRes':
        net = PreActResNet18()
    # net = ResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    elif cfg.MODEL.NET == 'WRN':
        net = wrn(depth=28, num_classes=10, widen_factor=10)
    else:
        raise ValueError('No Such Network: {}'.format(cfg.MODEL.NET))

    # Model
    previous = find_previous()
    if previous:
        start_epoch = previous[0][-1]
        resume_checkpoint(previous[1][-1], net)
    else:
        start_epoch = cfg.TRAIN.START_EPOCH

    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.benchmark = True
        print('Using CUDA..')

    criterion = nn.CrossEntropyLoss()

    # optimizer = CustomOpm.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9,
    # weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    optimizer = CustomOpm.CSGD(net.parameters(), lr=base_learning_rate, momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
                               weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    if cfg.TRAIN.LR_SCHEDULER.SCHEDULER == 'WR':
        sche = CustomLearningRateScheduler_wr(ti=cfg.TRAIN.LR_SCHEDULER.WR_TI, lr_min=0.0001, lr_max=base_learning_rate)
    elif cfg.TRAIN.LR_SCHEDULER.SCHEDULER == 'stage':
        sche = CustomLearningRateScheduler_staging(ti=cfg.TRAIN.LR_SCHEDULER.WR_TI, lr_min=0.0001,
                                                   lr_max=base_learning_rate)
    else:
        raise ValueError('Not Implemented {}'.format(cfg.TRAIN.LR_SCHEDULER))

    # Training
    def train(epoch, is_ga):
        print('\nEpoch: %d' % epoch)
        net.train()  # Sets the module in training mode.
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            # print('targets.size  ', targets.size)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # generate mixed inputs, two one-hot label vectors and mixing coefficient
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, cfg.TRAIN.MIXUP_ALPHA, use_cuda)
            optimizer.zero_grad()
            inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            outputs = net(inputs)

            loss_func = mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(criterion, outputs)
            loss.backward()
            if batch_idx % 39 == 5:
                _, _, _, _ = optimizer.step(gstep=epoch, check=False, is_ga=is_ga)
                # angle = check_grads / np.sqrt(check_g2 * check_b2)
                # writer.add_scalar("grads_variance", check_grads, batch_idx / 39 + epoch * 391 / 39)
                # writer.add_scalar("grads^2", check_g2, batch_idx / 39 + epoch * 391 / 39)
                # writer.add_scalar("buf^2", check_b2, batch_idx / 39 + epoch * 391 / 39)
                # writer.add_scalar("angel", angle, batch_idx / 39 + epoch * 391 / 39)
            else:
                optimizer.step(gstep=epoch, check=False, is_ga=is_ga)

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(
                targets_b.data).cpu().sum()

            if is_ga:
                print("GA break")
                return 0, 0
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        return (train_loss / batch_idx, correct / total)

    def test(epoch):
        global best_acc
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
            print("Saving Checkpoint, acc = {}".format(acc))
            save_checkpoints(epoch, net)
        elif epoch % cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
            print("Saving Checkpoint, acc = {}".format(acc))
            save_checkpoints(epoch, net, run)

        return (test_loss / batch_idx, correct / total)

    for epoch in range(start_epoch, cfg.TRAIN.SMALL_EPOCHS):
        global global_epoch
        global_epoch = epoch
        is_ga = sche.adjust_learning_rate(optimizer, epoch)
        is_ga = False
        # print("[Debug] is_ga True")
        train_loss, train_acc = train(epoch, is_ga)
        test_loss, test_acc = test(epoch)
        # with open(logname, 'a') as logfile:
        #     logwriter = csv.writer(logfile, delimiter=',')
        #     logwriter.writerow([epoch, train_loss, train_acc, test_loss, test_acc])
        prefix_train = 'Train_' + str(run)
        prefix_test = 'Test_' + str(run)
        writer.add_scalar(prefix_test + '/val_loss', test_loss, epoch)
        writer.add_scalar(prefix_test + '/val_acc', test_acc, epoch)
        writer.add_scalar(prefix_train + '/loss', train_loss, epoch)
        writer.add_scalar(prefix_train + '/acc', train_acc, epoch)

    return train_loss


def mtrain():
    num_runs = cfg.NUM_RUNS
    losses = np.zeros(num_runs)
    for i in range(cfg.TRAIN.MAX_EPOCHS // num_runs):
        for run in range(num_runs):
            loss = single_train(run)
            losses[run] = loss
            GA(losses)



    writer.close()