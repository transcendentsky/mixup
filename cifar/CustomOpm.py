import torch
from torch.optim.optimizer import Optimizer, required
# torch.optim.sgd
from summary import *
import math
import numpy as np


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None, check=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        check_grad = 0
        check_g2 = 0
        check_b2 = 0
        angle = 0
        # print("check_grad init: ", check_grad)
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                check_g = d_p.add(0)
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        check_b = buf.add(0)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        check_b = buf.add(0)
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                    if check == True:
                        dsum = check_g * check_b
                        dsum = dsum.sum()
                        check_grad += dsum

                        dsum2 = check_g.pow(2)
                        dsum2 = dsum2.sum()
                        check_g2 += dsum2

                        dsum3 = check_b.pow(2)
                        dsum3 = dsum3.sum()
                        check_b2 += dsum3

                p.data.add_(-group['lr'], d_p)
        return loss, check_grad, check_g2, check_b2


class Adam(Optimizer):
    """Implements Adam algorithm.

    It has been proposed in `Adam: A Method for Stochastic Optimization`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.9), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(Adam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg_sq.mean_()

                denom = exp_avg_sq.sqrt().add_(0)

                # bias_correction1 = 1 - beta1 ** state['step']
                # bias_correction2 = 1 - beta2 ** state['step']
                # step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(0, exp_avg, denom)

        return loss


class GSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, ga_prob=1.0):
        print("#####  Using GSGD  #####")
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, ga_prob=ga_prob)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(GSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(GSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, gstep, closure=None, check=False, is_ga=False, restore_buf=False, converse=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        check_grad = 0
        check_g2 = 0
        check_b2 = 0
        angle = 0
        # print("check_grad init: ", check_grad)
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            ga_prob = group['ga_prob']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                # print("p.data.shape ", p.data.shape)
                # print("d_p.shape ", d_p.shape)
                check_g = d_p.add(0)
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                param_state = self.state[p]

                if converse:
                    pass

                #################################   cross and mutation   ##############################
                if is_ga and gstep > 99:
                    # if is_ga and gstep >=1:
                    # print("[Debug] Test Crossing and mutating......")
                    if np.random.random() <= ga_prob:
                        if 'ga_buffer' not in param_state:
                            raise ValueError("Not Save ga_buffer......")
                        ga_buf = param_state['ga_buffer']
                        p.data.mul_(2.0).add_(-ga_buf)
                        continue



                # #################################   cross and mutation   ##############################
                # if is_ga and gstep > 99:
                #     # if is_ga and gstep >=1:
                #     # print("[Debug] Test Crossing and mutating......")
                #     if np.random.random() <= ga_prob:
                #         if 'ga_buffer' not in param_state:
                #             raise ValueError("Not Save ga_buffer......")
                #         ga_buf = param_state['ga_buffer']
                #         pre = p.data.clone()
                #         evol = np.random.normal(loc=0.0, scale=0.1, size=ga_buf.shape)
                #         evol = torch.cuda.FloatTensor(evol)
                #         choice = np.random.choice([0, 1], ga_buf.shape)
                #         choice = torch.cuda.FloatTensor(choice)
                #         namda = ga_buf.mul(1 - choice)
                #         p.data.mul_(choice).add_(namda)
                #         dif = p.data.add(ga_buf).mul(evol)
                #         p.data.mul_(dif)
                #         ### test
                #         if True:
                #             xdif = dif.sum()
                #             x = p.data.add(-pre).sum()
                #             print("Change :", x, xdif)
                #         continue


                # ################################  store params before warm restart   ##########################
                if restore_buf or (ga_prob != 0 and is_ga):
                    # print("Saving params......")
                    if 'ga_buffer' not in param_state:
                        ga_buf = param_state['ga_buffer'] = torch.zeros_like(p.data)
                        ga_buf.add_(p.data)
                    else:
                        ga_buf = param_state['ga_buffer']
                        ga_buf.zero_()
                        ga_buf.add_(p.data)
                    if restore_buf:
                        continue
                    # if ga_buf.pow(2).mean() == 0:
                    # raise ValueError('ga_buf ValueError')

                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        check_b = buf.add(0)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        check_b = buf.add(0)
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                    if check == True:
                        dsum = check_g * check_b
                        dsum = dsum.sum()
                        check_grad += dsum

                        dsum2 = check_g.pow(2)
                        dsum2 = dsum2.sum()
                        check_g2 += dsum2

                        dsum3 = check_b.pow(2)
                        dsum3 = dsum3.sum()
                        check_b2 += dsum3
                if is_ga == False:
                    p.data.add_(-group['lr'], d_p)
        return loss, check_grad, check_g2, check_b2


class CSGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, ga_prob=0.1):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, ga_prob=ga_prob)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(CSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, gstep, closure=None, check=False, is_ga=False):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        check_grad = 0
        check_g2 = 0
        check_b2 = 0
        angle = 0
        # print("check_grad init: ", check_grad)
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            ga_prob = group['ga_prob']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p = d_p.add(weight_decay, p.data)
                param_state = self.state[p]

                if is_ga and gstep > 99:
                    if len(p.data.shape) == 4 and np.random.random() < 0.5:
                        outdim = p.data.shape[0]
                        indim = p.data.shape[1]
                        for i in range(outdim):
                            for j in range(indim):
                                if np.random.random() <= ga_prob:
                                    pmax = p.data[i, j].max()
                                    pmin = p.data[i, j].min()
                                    if pmin > 0:
                                        ceil = pmax + pmin
                                    else:
                                        ceil = pmax
                                    p.data = ceil - p.data
                    continue

                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        check_b = buf.add(0)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        check_b = buf.add(0)
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if is_ga == False:
                    p.data.add_(-group['lr'], d_p)
        return loss, check_grad, check_g2, check_b2
