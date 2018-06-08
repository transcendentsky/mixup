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


class SingleTrain(object):
    def __init__(self, run, mt=False, training=True, x_max_epoch=0, initializer='xavier', psa=False, load_bone=False,
                 net=None, out_channel=100):
        # Parameters from __init__
        self.run = run
        self.training = training
        self.x_max_epoch = x_max_epoch
        self.initializer = initializer
        self.psa = psa
        self.load_bone = load_bone
        self.out_channel = out_channel

        self.ga_prob = 0.4
        self.ga_converse = False

        # init from cfg
        self.dataset = cfg.DATASET.DATASET
        self.base_learning_rate = cfg.TRAIN.BASE_LR
        self.batch_size = cfg.TRAIN.BATCH_SIZE
        self.start_epoch = cfg.TRAIN.START_EPOCH
        self.model_class = cfg.MODEL.NET
        self.num_workers = cfg.TRAIN.NUM_WORKERS
        self.lr_sche_wr_ti = cfg.TRAIN.LR_SCHEDULER.WR_TI
        if self.x_max_epoch == 0:
            self.x_max_epoch = cfg.TRAIN.MAX_EPOCHS

        # self.using_mixup = None
        self.optimizer = None
        self.psaver = None
        self.learning_sche = None
        self.base_acc = 0
        self.use_cuda = True
        self.trainloader = None
        self.testloader = None
        self.transform_test = None
        self.transform_train = None
        self.writer = get_writer()

        # Not implemented other criterions
        self.criterion = nn.CrossEntropyLoss()

        ### Extra Setting ###
        torch.manual_seed(cfg.TRAIN.SEED)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        # Network
        if net == None:
            self.choose_net()
            self.init_net()
        else:
            self.net = net

        self.init_datasets()
        self.init_lr_sche()
        self.init_optimizer()

    def check_status(self):
        assert self.net != None, "Network Setting Error"
        assert self.writer != None, "Logfile Writer Setting Error"
        # assert

    def init_datasets(self):
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

        if self.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True,
                                                           num_workers=self.num_workers)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                                          num_workers=self.num_workers)
        elif self.dataset == 'fashion-mnist':
            print("Fashion-MNIST")
            trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True,
                                                         transform=transform_train)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True,
                                                           num_workers=self.num_workers)
            testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True,
                                                        transform=transform_test)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                                          num_workers=self.num_workers)
        elif self.dataset == 'cifar100':
            print("Cifar100")
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                     transform=transform_train)
            self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True,
                                                           num_workers=self.num_workers)
            testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
            self.testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False,
                                                          num_workers=self.num_workers)

    def choose_net(self):
        # Networks
        print('==> Building model..')
        if self.model_class == 'VGG19':
            self.net = VGG('VGG19', out_channel=self.out_channel)
        elif self.model_class == 'PreRes':
            self.net = PreActResNet18()
        elif self.model_class == 'VGG16':
            self.net = VGG('VGG16', out_channel=self.out_channel)
        elif self.model_class == 'ResNet18':
            self.net = ResNet18()
        elif self.model_class == 'GoogleNet':
            self.net = GoogLeNet()
        elif self.model_class == 'DenseNet121':
            self.net = DenseNet121()
        elif self.model_class == 'ResNeXt29_2x64d':
            self.net = ResNeXt29_2x64d()
        elif self.model_class == 'MobileNet':
            self.net = MobileNet()
        elif self.model_class == 'DPN92':
            self.net = DPN92()
        elif self.model_class == 'ShuffleNetG2':
            self.net = ShuffleNetG2()
        elif self.model_class == 'SENet18':
            self.net = SENet18()
        elif self.model_class == 'WRN':
            self.net = wrn(depth=28, num_classes=self.out_channel, widen_factor=10)
        else:
            raise ValueError('No Such Network: {}'.format(self.model_class))

        # print("Architectures:")
        # print(net)
        if self.use_cuda:
            self.net.cuda()
            self.net = torch.nn.DataParallel(self.net)
            print('Using', torch.cuda.device_count(), 'GPUs.')
            cudnn.benchmark = True
            print('Using CUDA..')

    def init_net(self):
        previous = find_previous(self.run)
        if previous:
            self.start_epoch = previous[0][-1] + 1
            resume_checkpoint(previous[1][-1], net, self.run, noclassifier=self.load_bone)
        else:
            #########  Parameters Initialization  #########
            def weights_init(m, initializer):
                classname = m.__class__.__name__
                if classname.find('Conv') != -1:
                    if initializer == 'xavier':
                        print("####  Using xavier Initializer  ####")
                        nn.init.xavier_normal(m.weight.data)
                    elif initializer == 'kaiming_normal':
                        print("####  Using kaiming normal Initializer  ####")
                        nn.init.kaiming_normal(m.weight.data)
                    elif initializer == 'normal':
                        print("####  Using normal Initializer  ####")
                        nn.init.normal(m.weight.data)
                    else:
                        raise NotImplementedError("Only 2 initializer.")

            if True:
                self.net.apply(weights_init, self.initializer)

    def init_lr_sche(self):
        if self.learning_sche == 'WR':
            self.sche = CustomLearningRateScheduler_wr(ti=self.lr_sche_wr_ti, lr_min=0.0001,
                                                       lr_max=self.base_learning_rate)
        elif self.learning_sche == 'stage':
            self.sche = CustomLearningRateScheduler_staging(ti=self.lr_sche_wr_ti, lr_min=0.00001,
                                                            lr_max=self.base_learning_rate)
        elif self.learning_sche == 'SGDR':
            self.sche = CustomLearningRateScheduler_wr(ti=self.x_max_epoch, lr_min=0.00001,
                                                       lr_max=self.base_learning_rate)
        elif self.learning_sche == 'FIXED':
            self.sche = None
        else:
            raise ValueError('Not Implemented {}'.format(self.learning_sche))

    # Training
    def train(self, epoch, is_ga):
        print('\nEpoch: %d' % epoch)
        self.net.train()  # Sets the module in training mode.
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.trainloader):
            # print('targets.size  ', targets.size)
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            # generate mixed inputs, two one-hot label vectors and mixing coefficient
            inputs, targets_a, targets_b, lam = mixup_data(inputs, targets, cfg.TRAIN.MIXUP_ALPHA, self.use_cuda)
            self.optimizer.zero_grad()
            inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            outputs = self.net(inputs)

            loss_func = mixup_criterion(targets_a, targets_b, lam)
            loss = loss_func(self.criterion, outputs)
            loss.backward()
            if batch_idx % 39 == 5:
                _, _, _, _ = self.optimizer.step(gstep=epoch, check=False, is_ga=is_ga)
                # angle = check_grads / np.sqrt(check_g2 * check_b2)
                # writer.add_scalar("grads_variance", check_grads, batch_idx / 39 + epoch * 391 / 39)
                # writer.add_scalar("grads^2", check_g2, batch_idx / 39 + epoch * 391 / 39)
                # writer.add_scalar("buf^2", check_b2, batch_idx / 39 + epoch * 391 / 39)
                # writer.add_scalar("angel", angle, batch_idx / 39 + epoch * 391 / 39)
            else:
                self.optimizer.step(gstep=epoch, check=False, is_ga=is_ga)

            train_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += lam * predicted.eq(targets_a.data).cpu().sum() + (1 - lam) * predicted.eq(
                targets_b.data).cpu().sum()

            if is_ga:
                print("GA break")
                return 0, 0
            progress_bar(batch_idx, len(self.trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        return (train_loss / batch_idx, correct / total)

    def test(self, epoch):
        self.net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(self.testloader):
            if self.use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs, volatile=True), Variable(targets)
            outputs = self.net(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.data[0]
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(self.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        # Save checkpoint.
        correct = correct * 1.0
        acc = 100. * correct / total
        if acc > self.best_acc and epoch > 50:
            self.best_acc = acc
            print("Saving Checkpoint, acc = {}".format(acc))
            save_checkpoints(epoch, self.net, self.run)
        elif epoch % cfg.TRAIN.CHECKPOINTS_EPOCHS == 0:
            print("Saving Checkpoint, acc = {}".format(acc))
            save_checkpoints(epoch, self.net, self.run)
        elif (epoch + 1) % self.x_max_epoch == 0:
            print("One Turn finished.  Saving Checkpoint, acc = {}".format(acc))
            save_checkpoints(epoch, self.net, self.run)

        return (test_loss / batch_idx, correct / total)

    def init_optimizer(self):
        optimizer = CustomOpm.CSGD(net.parameters(), lr=self.base_learning_rate, momentum=cfg.TRAIN.OPTIMIZER.MOMENTUM,
                                   weight_decay=cfg.TRAIN.WEIGHT_DECAY)

    def single_train(self):
        if self.training:
            for epoch in range(self.start_epoch, self.x_max_epoch):
                self.sche.adjust_learning_rate(self.optimizer, epoch)
                train_lr = self.optimizer.param_groups[0]['lr']
                train_loss, train_acc = self.train(epoch, False)
                test_loss, test_acc = self.test(epoch)
                prefix_train = 'Train_' + str(self.run)
                prefix_test = 'Test_' + str(self.run)
                self.writer.add_scalar(prefix_test + '/val_loss', test_loss, epoch)
                self.writer.add_scalar(prefix_test + '/val_acc', test_acc, epoch)
                self.writer.add_scalar(prefix_train + '/loss', train_loss, epoch)
                self.writer.add_scalar(prefix_train + '/acc', train_acc, epoch)
                self.writer.add_scalar(prefix_train + '/lr', train_lr, epoch)

    def mutate(self, selected, epoch=1, save=True):
        id = selected[0] // 1000 * 1000 + 1000
        id_1 = selected[0]
        id_2 = selected[1]
        psaver.step(save=False, mute=True)
        save_checkpoints(epoch, net, id)
        return id

    def simple_mutate(self, id, epoch=1, save=True):
        global net
        global psaver
        print("[###] Execute Common mutation....")
        psaver.step(save=False, mute=True)
        save_checkpoints(epoch, net, id)
        return id

    def GA(self, checkpoints):
        # select two ckpts
        assert len(checkpoints) >= 2, 'Need at least 2 Checkpoints .'
        if len(checkpoints) == 2:
            selected = checkpoints
        else:
            select_idx = np.random.randint(len(checkpoints), 2)
            selected = [checkpoints[select_idx[0]], checkpoints[select_idx[1]]]
        id = self.mutate(selected, save=True)  # mutate and run
        self.single_train(id)


class MultiTrain(object):
    def __init__(self):
        self.psa = True
        self.psaver = None
        self.strategy = None

        self.ptrain1 = None
        self.ptrain2 = None

    def simple_mutate(self, id, epoch=1, save=True):
        global net
        global psaver
        print("[###] Execute Common mutation....")
        psaver.step(save=False, mute=True)
        save_checkpoints(epoch, net, id)
        return id

    def psave(self):
        if self.psaver is None:
            print("\n[###] Save to Backups, run: {}\n".format(self.run))
            self.psaver = CustomOpm.PSaver(net.parameters(), ga_prob=self.ga_prob, converse=self.ga_converse)
        if self.psa:
            self.psaver.step()

    def multi_train(self):
        if self.strategy == 'mute_with_inited_model':
            self.ptrain1 = SingleTrain(1001, training=False)

            self.ptrain2
