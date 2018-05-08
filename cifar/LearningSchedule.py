import numpy as np


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = 0.1
    if epoch <= 9 and lr > 0.1:
        # warm-up training for large minibatch
        lr = 0.1 + (0.2 - 0.1) * epoch / 10.
    if epoch >= 80:
        lr /= 10
    if epoch >= 130:
        lr /= 10
    print("Learning rate = ", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CustomLearningRateScheduler_staging(object):
    def __init__(self, ti=10, lr_min=0.001, lr_max=0.1):
        self.ti = ti
        self._lr_min = lr_min
        self._lr_max = lr_max
        self.base = 0
        print("##########  Using CustomLearningRateScheduler1 : Staging lr Schedule ##########\n")

    def adjust_learning_rate(self, optimizer, epoch):
        """decrease the learning rate at 100 and 150 epoch"""
        lr = 0.1
        if epoch <= 9 and lr > 0.1:
            # warm-up training for large minibatch
            lr = 0.1 + (0.2 - 0.1) * epoch / 10.
        if epoch >= 100:
            lr /= 10
        if epoch >= 150:
            lr /= 10
        print("Learning rate = ", lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if epoch == 180:
            return True
        else:return False

class CustomLearningRateScheduler_wr(object):
    def __init__(self, ti=10, lr_min=0.001, lr_max=0.1):
        self.ti = ti
        self._lr_min = lr_min
        self._lr_max = lr_max
        self.base = 0
        print("##########  Using CustomLearningRateScheduler2 : Warm restart ##########\n")

    def adjust_learning_rate(self, optimizer, epoch):
        ga = False
        lr_max = self._lr_max
        lr_min = self._lr_min
        if epoch >= self.ti + self.base:
            self.base += self.ti
            ga = True  # evolutionary strategy
            if self.ti < 50:
                self.ti = self.ti * 2
            else:
                pass

        epoch_n = epoch - self.base
        lr = lr_min + (lr_max - lr_min) * (np.cos(epoch_n * np.pi / self.ti / 2))

        ## adjust learning rate
        print("Learning rate = ", lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        return ga


class CustomLearningRateScheduler_own(object):
    def __init__(self, ti=10, lr_min=0.001, lr_max=0.1):
        self.lr = lr_max
        self.acc_r = np.zeros(10)
        self.loss_r = np.zeros(10)
        self.record = 0
        print("##########  Using CustomLearningRateScheduler3 : A bad auto ##########\n")

    def adjust_learning_rate(self, optimizer, epoch, acc, loss):
        record = self.record
        self.acc_r[record%10] = acc
        self.loss_r[record%10] = loss

        if epoch <= 10:
            return self.lr
        else:
            if epoch % 5 == 0:
                # acc_mean = self.acc_r.mean()
                if epoch %10 == 0:
                    y = self.loss_r
                else:
                    y = np.zeros(10)
                    y[:5] = self.loss_r[5:]
                    y[5:] = self.loss_r[:5]
                x = np.arange(10)
                y_mean = self.loss_r.mean()
                x_mean = np.arange(10).mean()
                top = np.sum(y * x) - len(x) * x_mean * y_mean
                bottom = np.sum(x ** 2) - len(x) * x_mean ** 2
                k = top / bottom
                if k > 0.05:
                    self.lr = self.lr
                else:
                    self.lr = 0.6 * self.lr

                print("Adjusting Learning rate = ", self.lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.lr
        self.record += 1




