import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.hooks import RemovableHandle
from models.modules import CustomLinear, CustomConv2d, CustomBatchNorm2d
from typing import Tuple, List, NamedTuple, Mapping, Optional, Callable
from tqdm import tqdm, trange
from utils import icycle, MultiEpochsDataLoader
from dataclasses import dataclass
from collections import OrderedDict
from torch import optim


def compute_A(module: nn.Module, a: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        # a.shape == (B, Nin)
        b = a.size(0)
        if module.bias is not None: #TODO
            a = F.pad(a, (0,1), value=1) # (B, Nin+1)
        A = torch.einsum("bi, bj -> ij", a, a)
        A /= b
    elif isinstance(module, nn.Conv2d):
        # a.shape == (B, Cin, h, w)
        a = F.unfold(a, module.kernel_size, module.dilation, module.padding, module.stride) # (B, Cin*k*k, h*w)
        bhw = a.size(0)*a.size(2)
        if module.bias is not None:
            a = F.pad(a, (0,0,0,1,0,0), value=1) # (B, Cin*k*k+1, h*w)
        A = torch.einsum("bij, bkj -> ik", a, a) # (Cin*k*k, Cin*k*k)
        A /= bhw
    else:
        raise NotImplementedError(f'{type(module)}')
    return A


def compute_S(module: nn.Module, g: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        # g.shape == (B, Nout)
        b = g.size(0)
        S = torch.einsum("bi, bj -> ij", g, g)
        S /= b
    elif isinstance(module, nn.Conv2d):
        # g.shape == (B, Cout, h, w)
        b = g.size(0)
        S = torch.einsum("bijk, bljk -> il", g, g)
        S /= b
    else:
        raise NotImplementedError(f'{type(module)}')
    return S


class EKFAC(torch.optim.Optimizer):

    def __init__(self, modules, eps, sua=False, ra=False, update_freq=1, alpha=.75):
        self.eps = eps
        self.sua = sua
        self.ra = ra
        self.update_freq = update_freq
        self.alpha = alpha
        self.params = []
        self._fwd_handles = []
        self._bwd_handles = []
        self._iteration_counter = 0
        if not self.ra and self.alpha != 1.:
            raise NotImplementedError
        assert all(isinstance(m, (nn.Linear, nn.Conv2d)) for m in modules)
        for mod in modules:
            mod_class = mod.__class__.__name__
            # handle = mod.register_forward_pre_hook(self._save_input)
            handle = mod.register_forward_hook(self._forward_hook)
            self._fwd_handles.append(handle)
            # handle = mod.register_backward_hook(self._save_grad_output)
            # self._bwd_handles.append(handle)
            self.params.append({
                'params': list(mod.parameters()),
                'mod': mod,
                'layer_type': mod_class,
                'SUA': self.sua if isinstance(mod, nn.Conv2d) else False,
            })
            defaults = {
                'SUA': False,
                'lr': 1e-1,
            }
        super(EKFAC, self).__init__(self.params, defaults)



class EKFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 kl_clip=0.001,
                 weight_decay=0,
                 TCov=10,
                 TScal=10,
                 TInv=100,
                 batch_averaged=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)
        # TODO (CW): EKFAC optimizer now only support model as input
        super(EKFACOptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.MatGradHandler = ComputeMatGrad()
        self.batch_averaged = batch_averaged

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.S_l = {}
        self.A, self.DS = {}, {}
        self.stat_decay = stat_decay

        self.kl_clip = kl_clip
        self.TCov = TCov
        self.TScal = TScal
        self.TInv = TInv

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.CovAHandler(input[0].data, module)
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
            update_running_stat(aa, self.m_aa[module], self.stat_decay)
        if torch.is_grad_enabled() and self.steps % self.TScal == 0  and self.steps > 0:
            self.A[module] = input[0].data

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
            update_running_stat(gg, self.m_gg[module], self.stat_decay)

        # if self.steps % self.TInv == 0:
        #     self._update_inv(module)

        if self.acc_stats and self.steps % self.TScal == 0 and self.steps > 0:
            self.DS[module] = grad_output[0].data
            # self._update_scale(module)

    def _prepare_model(self):
        count = 0
        print(self.model)
        print("=> We keep following layers in EKFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                print('(%s): %s' % (count, module))
                count += 1

    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        eps = 1e-10  # for numerical stability
        self.d_a[m], self.Q_a[m] = torch.symeig(
            self.m_aa[m], eigenvectors=True)
        self.d_g[m], self.Q_g[m] = torch.symeig(
            self.m_gg[m], eigenvectors=True)

        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_g[m].mul_((self.d_g[m] > eps).float())
        # if self.steps != 0:
        self.S_l[m] = self.d_g[m].unsqueeze(1) @ self.d_a[m].unsqueeze(0)

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.S_l[m] + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _kl_clip_and_update_grad(self, updates, lr):
        # do kl clip
        vg_sum = 0
        for m in self.modules:
            v = updates[m]
            vg_sum += (v[0] * m.weight.grad.data * lr ** 2).sum().item()
            if m.bias is not None:
                vg_sum += (v[1] * m.bias.grad.data * lr ** 2).sum().item()
        nu = min(1.0, math.sqrt(self.kl_clip / vg_sum))

        for m in self.modules:
            v = updates[m]
            m.weight.grad.data.copy_(v[0])
            m.weight.grad.data.mul_(nu)
            if m.bias is not None:
                m.bias.grad.data.copy_(v[1])
                m.bias.grad.data.mul_(nu)

    def _step(self, closure):
        # FIXME (CW): Modified based on SGD (removed nestrov and dampening in momentum.)
        # FIXME (CW): 1. no nesterov, 2. buf.mul_(momentum).add_(1 <del> - dampening </del>, d_p)
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1, d_p)
                    d_p = buf

                p.data.add_(-group['lr'], d_p)

    def _update_scale(self, m):
        with torch.no_grad():
            A, S = self.A[m], self.DS[m]
            grad_mat = self.MatGradHandler(A, S, m)  # batch_size * out_dim * in_dim
            if self.batch_averaged:
                grad_mat *= S.size(0)

            s_l = (self.Q_g[m] @ grad_mat @ self.Q_a[m].t()) ** 2  # <- this consumes too much memory!
            s_l = s_l.mean(dim=0)
            if self.steps == 0:
                self.S_l[m] = s_l.new(s_l.size()).fill_(1)
            # s_ls = self.Q_g[m] @ grad_s
            # s_la = in_a @ self.Q_a[m].t()
            # s_l = 0
            # for i in range(0, s_ls.size(0), S.size(0)):    # tradeoff between time and memory
            #     start = i
            #     end = min(s_ls.size(0), i + S.size(0))
            #     s_l += (torch.bmm(s_ls[start:end,:], s_la[start:end,:]) ** 2).sum(0)
            # s_l /= s_ls.size(0)
            # if self.steps == 0:
            #     self.S_l[m] = s_l.new(s_l.size()).fill_(1)
            update_running_stat(s_l, self.S_l[m], self.stat_decay)
            # remove reference for reducing memory cost.
            self.A[m] = None
            self.DS[m] = None

    def step(self, closure=None):
        # FIXME(CW): temporal fix for compatibility with Official LR scheduler.
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        updates = {}
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                self._update_inv(m)

            if self.steps % self.TScal == 0 and self.steps > 0:
                self._update_scale(m)

            p_grad_mat = self._get_matrix_form_grad(m, classname)
            v = self._get_natural_grad(m, p_grad_mat, damping)
            updates[m] = v
        self._kl_clip_and_update_grad(updates, lr)

        self._step(closure)
        self.steps += 1


if __name__ == "main":
    for batch_idx, (inputs, targets) in prog_bar:
    inputs, targets = inputs.to(args.device), targets.to(args.device)
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, targets)
    if optim_name in ['kfac', 'ekfac'] and optimizer.steps % optimizer.TCov == 0:
        # compute true fisher
        optimizer.acc_stats = True
        with torch.no_grad():
            sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                            1).squeeze().cuda()
        loss_sample = criterion(outputs, sampled_y)
        loss_sample.backward(retain_graph=True)
        optimizer.acc_stats = False
        optimizer.zero_grad()  # clear the gradient for computing true-fisher.
    loss.backward()
    optimizer.step()

    train_loss += loss.item()
    _, predicted = outputs.max(1)
    total += targets.size(0)
    correct += predicted.eq(targets).sum().item()

    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (tag, lr_scheduler.get_lr()[0], train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    prog_bar.set_description(desc, refresh=True)
