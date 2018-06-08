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
net = None
psaver = None


def single_train(run, mt=False, training=True, x_max_epoch=150, initializer=None, psa=False, load_bone=False):
    global net
    global psaver

    ### Extra Setting ###
    if x_max_epoch == 0: x_max_epoch = cfg.TRAIN.MAX_EPOCHS

    if net is None:
        print("[***] net None")
    else:
        print("[***] net Already.")
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
    dataset = cfg.DATASET.DATASET
    # if run % 1000 == 2: dataset = cfg.DATASET.DATASET
    # else: dataset = cfg.DATASET.DATASET_PRE

    if dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=cfg.TRAIN.NUM_WORKERS)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                                 num_workers=cfg.TRAIN.NUM_WORKERS)
    elif dataset == 'fashion-mnist':
        print("Fashion-MNIST")
        trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=cfg.TRAIN.NUM_WORKERS)
        testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                                 num_workers=cfg.TRAIN.NUM_WORKERS)
    elif dataset == 'cifar100':
        print("Cifar100")
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                     transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=cfg.TRAIN.NUM_WORKERS)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                                 num_workers=cfg.TRAIN.NUM_WORKERS)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Networks
    print('==> Building model..')
    if cfg.MODEL.NET == 'VGG19':
        net = VGG('VGG19',out_channel=100)
    elif cfg.MODEL.NET == 'PreRes':
        net = PreActResNet18()
    elif cfg.MODEL.NET == 'VGG16':
        net = VGG('VGG16')
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

    # print("Architectures:")
    # print(net)
    if use_cuda:
        net.cuda()
        net = torch.nn.DataParallel(net)
        print('Using', torch.cuda.device_count(), 'GPUs.')
        cudnn.benchmark = True
        print('Using CUDA..')

    # Model
    previous = find_previous(run)
    if previous:
        start_epoch = previous[0][-1] + 1
        if load_bone:
            resume_checkpoint(previous[1][-1], net, run, noclassifier=True)
        else:
            resume_checkpoint(previous[1][-1], net, run)
    else:
        start_epoch = cfg.TRAIN.START_EPOCH

        #########  Parameters Initialization  #########
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if initializer == None:
                    if run % 1000 == 1:
                        print("####  Using xavier Initializer  ####")
                        nn.init.xavier_normal(m.weight.data)
                    elif run % 1000 == 3:
                        print("####  Using kaiming normal Initializer  ####")
                        nn.init.kaiming_normal(m.weight.data)
                    elif run % 1000 == 2:
                        print("####  Using normal Initializer  ####")
                        nn.init.normal(m.weight.data)
                    else:
                        raise NotImplementedError("Only 2 initializer.")
                else:
                    raise NotimplementedError("???")
                # nn.init.xavier_normal(m.bias.data)

        if True:
            net.apply(weights_init)

    criterion = nn.CrossEntropyLoss()

    # optimizer = CustomOpm.SGD(net.parameters(), lr=base_learning_rate, momentum=0.9,
    # weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    optimizer = CustomOpm.CSGD(net.parameters(), lr=base_learning_rate, momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
                               weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    if cfg.TRAIN.LR_SCHEDULER.SCHEDULER == 'WR':
        sche = CustomLearningRateScheduler_wr(ti=cfg.TRAIN.LR_SCHEDULER.WR_TI, lr_min=0.0001, lr_max=base_learning_rate)
    elif cfg.TRAIN.LR_SCHEDULER.SCHEDULER == 'stage':
        sche = CustomLearningRateScheduler_staging(ti=cfg.TRAIN.LR_SCHEDULER.WR_TI, lr_min=0.00001,
                                                   lr_max=base_learning_rate)
    elif cfg.TRAIN.LR_SCHEDULER.SCHEDULER == 'SGDR':
        sche = CustomLearningRateScheduler_wr(ti=x_max_epoch, lr_min=0.00001,
                                                   lr_max=base_learning_rate)
    elif cfg.TRAIN.LR_SCHEDULER.SCHEDULER == 'FIXED':
        sche
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
            save_checkpoints(epoch, net, run)
        elif epoch % cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
            print("Saving Checkpoint, acc = {}".format(acc))
            save_checkpoints(epoch, net, run)
        elif (epoch+1) % x_max_epoch == 0:
            print("One Turn finished.  Saving Checkpoint, acc = {}".format(acc))
            save_checkpoints(epoch, net, run)

        return (test_loss / batch_idx, correct / total)

    if training:
        for epoch in range(start_epoch, x_max_epoch):
            global global_epoch
            global_epoch = epoch
            sche.adjust_learning_rate(optimizer, epoch)
            train_lr = optimizer.param_groups[0]['lr']
            is_ga = mt
            # print("[Debug] is_ga True")
            train_loss, train_acc = train(epoch, is_ga)
            if is_ga:
                print("GA break training.")
                return 0
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
            writer.add_scalar(prefix_train + '/lr', train_lr, epoch)

    if psa and psaver is None:
        print("\n[###] Save to Backups, run: {}\n".format( run ))
        psaver = CustomOpm.PSaver(net.parameters(), ga_prob=0.4, converse=False)
        psaver.step()

    return


def mutate(selected, epoch=1,save=True):
    id = selected[0] // 1000 * 1000 + 1000
    id_1 = selected[0]
    id_2 = selected[1]
    global net
    global psaver
    assert net is not None,'11111'
    assert psaver is not None,'22222'

    psaver.step(save=False, mute=True)

    save_checkpoints(epoch, net, id)
    return id

def simple_mutate(id, epoch=1, save=True):
    global net
    global psaver
    print("[###] Execute Common mutation....")
    psaver.step(save=False, mute=True)
    save_checkpoints(epoch, net, id)
    return id

def GA(checkpoints):
    # select two ckpts
    assert len(checkpoints) >= 2, 'Need at least 2 Checkpoints .'
    if len(checkpoints) == 2:
        selected = checkpoints
    else:
        select_idx = np.random.randint(len(checkpoints), 2)
        selected = [checkpoints[select_idx[0]], checkpoints[select_idx[1]]]
    id = mutate(selected, save=True)  # mutate and run

    single_train(id)



def mtrain():
    # num_runs = cfg.NUM_RUNS
    # losses = np.zeros(num_runs)
    # for i in range(cfg.TRAIN.MAX_EPOCHS // num_runs):
    #     for run in range(num_runs):
    #         loss = single_train(run)
    #         losses[run] = loss
    #         GA(losses)
    population = 2
    ids = [1001, 1002]
    # loss = single_train(ids[0], training=True)

    # loss = single_train(3001, training=False, load_bone=True , psa=True)
    for x in range(1):  # just run once
        x += 1
        # for i in range(population):
        #     id = x * 1000 + i + 1
        #     loss = single_train(id, training=True)
        #     ids.append(id)
        loss = single_train(1001, training=True, x_max_epoch=cfg.TRAIN.MAX_EPOCHS * x)
        # GA(ids)
        # simple_mutate(1001, epoch=50*x)

# def mute_train():
#     single_train(1000, training=True)
