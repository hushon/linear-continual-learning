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
import opt_einsum as oe


@dataclass
class KFACState:
    A: torch.Tensor = None
    S: torch.Tensor = None
    weight: torch.Tensor = None
    bias: torch.Tensor = None


@dataclass
class EKFACState:
    Q_A: torch.Tensor = None
    Q_S: torch.Tensor = None
    scale: torch.Tensor = None # Hessian eigenvalues
    weight: torch.Tensor = None
    bias: torch.Tensor = None


@dataclass
class EWCState:
    G: torch.Tensor = None
    parameter: torch.Tensor = None


@torch.no_grad()
def compute_A(module: nn.Module, a: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        # a.shape == (B, Nin)
        b = a.size(0)
        if module.bias is not None: #TODO
            a = F.pad(a, (0,1), value=1) # (B, Nin+1)
        a = a.t() # (Nin, B)
        A = a @ a.t() # (Nin, Nin)
        A /= b
    elif isinstance(module, nn.Conv2d):
        # a.shape == (B, Cin, h, w)
        b = a.size(0)
        a = F.unfold(a, module.kernel_size, module.dilation, module.padding, module.stride) # (B, Cin*k*k, h*w)
        # a = a.transpose(0,1).contiguous() # (Cin*k*k, B, h*w), might be slow due to memcopy
        # if module.bias is not None: #TODO
        #     a = F.pad(a, (0,0,0,0,0,1), value=1) # (Cin*k*k+1, B, h*w)
        # a = a.view(a.size(0), -1) # (Cin*k*k, B*h*w)
        # A = a @ a.t() # (Cin*k*k, B*h*w)@(B*h*w, Cin*k*k) = (Cin*k*k, Cin*k*k)

        if module.bias is not None:
            a = F.pad(a, (0,0,0,1,0,0), value=1) # (B, Cin*k*k+1, h*w)
        A = oe.contract("bij, bkj -> ik", a, a, backend='torch') # (Cin*k*k, Cin*k*k)
        A /= b
    elif isinstance(module, nn.BatchNorm2d):
        # a.shape == (B, Cin, h, w)
        b = a.size(0)
        a = a.view(a.size(0), a.size(1), -1) # (B, C, h*w)
        a = a.transpose(0,1).contiguous() # (C, B, h*w), might be slow due to memcopy
        if module.bias is not None: #TODO
            a = F.pad(a, (0,0,0,0,0,1), value=1) # (C, B, h*w)
        A = a.view(a.size(0), -1) @ a.view(a.size(0), -1).t() # (C, B*h*w)@(B*h*w, C) = (C, C)
        A /= b
    else:
        raise NotImplementedError(f'{type(module)}')
    return A


@torch.no_grad()
def compute_S(module: nn.Module, g: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        # g.shape == (B, Nout)
        b = g.size(0)
        g = g.t() # (Nout, B)
        S = g @ g.t() # (Nout, Nout)
        S /= b
    elif isinstance(module, nn.Conv2d):
        # g.shape == (B, Cout, h, w)
        bhw = g.size(0)*g.size(2)*g.size(3)
        # g = g.transpose(0,1) # (Cout, B, h, w)
        # g = g.reshape(c, b*h*w) # (Cout, B*h*w)
        # S = g @ g.t() # (Cout, B*h*w) @ (Cout, B*h*w) = (Cout, Cout)

        S = oe.contract("bijk, bljk -> il", g, g, backend='torch')
        S /= bhw
    elif isinstance(module, nn.BatchNorm2d):
        # g.shape == (B, C, h, w)
        b = g.size(0)
        g = g.transpose(0,1) # (C, B, h, w)
        g = g.view(g.size(0), g.size(1), -1).contiguous() # (C, B, h*w)
        S = g.view(g.size(0),-1) @ g.view(g.size(0),-1).t() # (C, B*h*w) @ (C, B*h*w) = (C, C)
        S /= b
    else:
        raise NotImplementedError(f'{type(module)}')
    return S


# @torch.no_grad()
def unfold_weight(weight: torch.Tensor, bias: torch.Tensor = None) -> Tuple[torch.Tensor, Callable]:
    """reshapes multidimensional weight tensor to 2D matrix, then augments bias to the weight matrix.

    Args:
        weight (torch.Tensor): weight tensor
        bias (torch.Tensor, optional): bias tensor. Defaults to None.

    Raises:
        ValueError: when weight tensor shape is not supported

    Returns:
        Tuple[torch.Tensor, Callable]: tuple of unfolded-augmented weight matrix and a function to revert the shape.
    """
    is_batchnorm = weight.ndim == 1
    weight_shape = weight.shape
    has_bias = bias is not None

    if weight.ndim == 1:
        weight = weight.diag() # (C, C)
    elif weight.ndim == 2:
        pass # (Nout, Nin)
    elif weight.ndim == 4:
        weight = weight.reshape(weight.size(0), -1) # (Cout, Cin*k*k)
    else:
        raise ValueError(f'{weight.ndim}')

    if has_bias:
        weight_aug = torch.cat((weight, bias[:, None]), dim=1)
    else:
        weight_aug = weight

    def fold_weight_fn(weight_aug: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if has_bias:
            weight, bias = weight_aug[:, :-1], weight_aug[:, -1]
        else:
            weight, bias = weight_aug, None
        if is_batchnorm:
            weight = weight.diagonal() # (C,)
        else:
            weight = weight.reshape(weight_shape)
        return weight, bias

    return weight_aug, fold_weight_fn


class KFACRegularizer:

    def __init__(self, model: nn.Module, criterion: nn.Module, modules: List[nn.Module]) -> None:
        self.model = model
        self.criterion = criterion
        self.modules : List[nn.Module] = modules
        self.a_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.g_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.kfac_state_dict : Mapping[nn.Module, KFACState] = OrderedDict()
        self.n_iter = 0
        self._init_kfac_states()

    def _init_kfac_states(self) -> None:
        for module in self.modules:
            if isinstance(module, nn.Linear):
                a_size = module.in_features
                g_size = module.out_features
            elif isinstance(module, nn.Conv2d):
                a_size = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                g_size = module.out_channels
            elif isinstance(module, nn.BatchNorm2d):
                a_size = g_size = module.num_features
            else:
                raise NotImplementedError()

            if module.bias is not None:
                a_size += 1

            self.kfac_state_dict[module] = KFACState(
                A=module.weight.new_zeros((a_size, a_size)),
                S=module.weight.new_zeros((g_size, g_size)),
                weight=None,
                bias=None
            )

    def _del_temp_states(self) -> None:
        del self.a_dict
        del self.g_dict

    def _register_hooks(self) -> List[RemovableHandle]:
        hook_handles = []
        for module in self.modules:
            handle = module.register_forward_hook(self._forward_hook)
            hook_handles.append(handle)
        return hook_handles

    def _remove_hooks(self, hook_handles: List[RemovableHandle]) -> None:
        for handle in hook_handles:
            handle.remove()

    @torch.no_grad()
    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
        if type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
            a, _ = input # primal input
            _, jvp = output # tangent output
            self.a_dict[module] = a.detach()
            def _tensor_backward_hook(grad: torch.Tensor) -> None:
                self.g_dict[module] = grad.detach()
            jvp.register_hook(_tensor_backward_hook)
        elif type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
            input, = input # primal input
            output = output # primal output
            self.a_dict[module] = input.detach()
            def _tensor_backward_hook(grad: torch.Tensor) -> None:
                self.g_dict[module] = grad.detach()
            output.register_hook(_tensor_backward_hook)
        else:
            raise NotImplementedError

    @torch.no_grad()
    def _accumulate_curvature_step(self) -> None:
        for module in self.modules:
            a = self.a_dict[module]
            g = self.g_dict[module]
            kfac_state = self.kfac_state_dict[module]
            kfac_state.S.add_(compute_S(module, g))
            kfac_state.A.add_(compute_A(module, a))
        self.n_iter += 1

    @torch.no_grad()
    def _divide_curvature(self) -> None:
        for module in self.modules:
            kfac_state = self.kfac_state_dict[module]
            kfac_state.A.div_(self.n_iter)
            kfac_state.S.div_(self.n_iter)

    @torch.no_grad()
    def _update_center(self):
        for module in self.modules:
            if type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
                self.kfac_state_dict[module].weight = module.weight_tangent.clone()
                self.kfac_state_dict[module].bias = module.bias_tangent.clone() if module.bias_tangent is not None else None
            elif type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
                self.kfac_state_dict[module].weight = module.weight.clone()
                self.kfac_state_dict[module].bias = module.bias.clone() if module.bias is not None else None
            else:
                raise NotImplementedError

    def compute_curvature(self, dataset: Dataset, n_steps: int, t: int = None, pseudo_target_fn = torch.normal) -> None:
        data_loader = MultiEpochsDataLoader(
                            dataset,
                            batch_size=64,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4,
                        )
        data_loader_cycle = icycle(data_loader)

        hook_handles = self._register_hooks()
        self.model.eval()
        for _ in trange(n_steps):
            input, _ = next(data_loader_cycle)
            input = input.cuda()
            self.model.zero_grad()
            if t is not None:
                output = self.model(input)[t]
            else:
                output = self.model(input)
            pseudo_target = pseudo_target_fn(output.detach())
            loss = self.criterion(output, pseudo_target).sum(-1).sum()
            loss.backward()
            self._accumulate_curvature_step()

        self._divide_curvature()
        self._update_center()
        self._remove_hooks(hook_handles)
        self._del_temp_states()

    def compute_loss(self) -> torch.Tensor:
        losses = []
        for module in self.modules:
            if type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
                losses.append(KFAC_penalty.apply(self.kfac_state_dict[module], module.weight_tangent, module.bias_tangent))
            elif type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
                losses.append(KFAC_penalty.apply(self.kfac_state_dict[module], module.weight, module.bias))
            else:
                raise NotImplementedError
        loss = sum(losses)
        return loss


def kfac_mvp(A: torch.Tensor, B: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Computes the matrix-vector product, where the matrix is factorized by a Kronecker product.
    Uses 'vec trick' to compute the product efficiently.

    Returns:
        torch.Tensor: Matrix-vector product (A⊗B)v
    """
    # (A⊗B)v = vec(BVA')
    m, n = A.shape
    p, q = B.shape
    if vec.ndim == 1:
        V = vec.view(q, n)
    elif vec.ndim == 2:
        assert vec.shape == (q, n)
        V = vec
    else:
        raise ValueError
    mvp = torch.chain_matmul(B, V, A.T) # (p, m)

    if vec.ndim == 1:
        return mvp.view(-1)
    else:
        return mvp


def kfac_loss_batchnorm(kfac_state: KFACState, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert weight.ndim == 1 # is_batchnorm
    S = kfac_state.S # (C, C)
    A = kfac_state.A # (C, C) or (C+1, C+1)

    if bias is not None:
        dw = weight - kfac_state.weight # (C,)
        db = bias - kfac_state.bias # (C,)
        Sdw = S * dw[None, :]
        Sdb = torch.mv(S, db)
        Sv = torch.cat((Sdw, Sdb[:, None]), dim=1) # (C, C+1)
        # Hv = Sv @ A # (C, C+1)
        # Hdw = Hv[:, :-1].diagonal() # (C,)
        # Hdb = Hv[:, -1] # (C,)
        Hdw = (Sv * A[:-1, :]).sum(1) # (C,)
        Hdb = torch.mv(Sv, A[:, -1]) # (C,)
        vHv = torch.dot(dw, Hdw) + torch.dot(db, Hdb)
        return 0.5 * vHv, Hdw, Hdb
    else:
        dw = weight - kfac_state.weight # (C,)
        Sdw = S * dw[None, :]
        Sv = Sdw # (C, C)
        # Hv = Sv @ A # (C, C)
        # Hdw = Hv.diagonal() # (C,)
        Hdw = (Sv * A).sum(1) # (C,)
        vHv = torch.dot(dw, Hdw)
        return 0.5 * vHv, Hdw, None


def kfac_loss_linear_conv(kfac_state: KFACState, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    S = kfac_state.S # (Nout, Nout)
    A = kfac_state.A # (Nin, Nin)
    param, fold_weight_fn = unfold_weight(weight, bias)
    center, fold_weight_fn = unfold_weight(kfac_state.weight, kfac_state.bias)
    v = param - center
    Hv = torch.chain_matmul(S, v, A) # (Nout, Nin)
    vHv = torch.dot(v.view(-1), Hv.view(-1))
    Hdw, Hdb = fold_weight_fn(Hv)
    return 0.5 * vHv, Hdw, Hdb


class KFAC_penalty(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kfac_state: KFACState, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight.ndim == 1:
            loss, Hdw, Hdb = kfac_loss_batchnorm(kfac_state, weight, bias)
        elif weight.ndim == 2 or weight.ndim == 4:
            loss, Hdw, Hdb = kfac_loss_linear_conv(kfac_state, weight, bias)
        else:
            raise ValueError

        ctx.save_for_backward(Hdw, Hdb)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # grad_output.shape == (,)
        Hdw, Hdb = ctx.saved_tensors
        grad_weight = grad_output * Hdw
        if Hdb is not None:
            grad_bias = grad_output * Hdb
        else:
            grad_bias = None
        return None, grad_weight, grad_bias


class EWCRegularizer:
    def __init__(self, model: nn.Module, criterion: nn.Module):
        self.model = model
        self.criterion = criterion
        self.ewc_state_dict : Mapping[nn.Parameter, EWCState] = OrderedDict()
        self.n_steps = 0
        self._reset_state()

    def _reset_state(self):
        for parameter in self.model.parameters():
            self.ewc_state_dict[parameter] = EWCState(
                G = torch.zeros_like(parameter),
                parameter = None
            )

    def _accumulate_curvature_step(self):
        for parameter in self.model.parameters():
            self.ewc_state_dict[parameter].G.add_(parameter.grad.data.square())
        self.n_steps += 1

    def _update_center(self):
        for parameter in self.model.parameters():
            self.ewc_state_dict[parameter].parameter = parameter.detach().clone()

    def _divide_curvature(self):
        for ewc_state in self.ewc_state_dict.values():
            ewc_state.G.div_(self.n_steps)

    def compute_curvature(self, dataset: torch.utils.data.Dataset, n_steps, t = None):
        self._reset_state()
        data_loader = MultiEpochsDataLoader(
                            dataset,
                            batch_size=1,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4,
                        )
        data_loader_cycle = icycle(data_loader)
        self.model.eval()
        for _ in trange(n_steps):
            input, _ = next(data_loader_cycle)
            input = input.cuda()
            self.model.zero_grad()
            if t is not None:
                output = self.model(input)[t]
            else:
                output = self.model(input)
            pseudo_target = torch.normal(output.detach()) #TODO
            loss = self.criterion(output, pseudo_target).sum(-1).squeeze() #TODO
            loss.backward()
            self._accumulate_curvature_step()
        self._divide_curvature()
        self._update_center()

    def compute_loss(self):
        loss = 0.
        for parameter in self.model.parameters():
            ewc_state = self.ewc_state_dict[parameter]
            loss += (ewc_state.G * torch.square(parameter - ewc_state.parameter)).sum()
        return 0.5*loss

    def merge_regularizer(self, old_state_dict):
        old_importance = old_state_dict['importance']
        self.importance = [x + y for x, y in zip(self.importance, old_importance)]


class EKFACRegularizer:

    def __init__(self, model: nn.Module, criterion: nn.Module, modules: List[nn.Module]) -> None:
        self.model = model
        self.criterion = criterion
        self.modules : List[nn.Module] = modules
        self.a_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.g_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.kfac_state_dict : Mapping[nn.Module, KFACState] = OrderedDict()
        self.n_iter = 0
        self._init_kfac_states()
        self.ekfac_state_dict: Mapping[nn.Module, EKFACState] = OrderedDict()

    def _init_kfac_states(self) -> None:
        for module in self.modules:
            if isinstance(module, nn.Linear):
                a_size = module.in_features
                g_size = module.out_features
            elif isinstance(module, nn.Conv2d):
                a_size = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
                g_size = module.out_channels
            elif isinstance(module, nn.BatchNorm2d):
                a_size = g_size = module.num_features
            else:
                raise NotImplementedError()

            if module.bias is not None:
                a_size += 1

            self.kfac_state_dict[module] = KFACState(
                S=module.weight.new_zeros((g_size, g_size)),
                A=module.weight.new_zeros((a_size, a_size)),
                weight=None,
                bias=None
            )

    def _init_ekfac_states(self) -> None:
        for module in self.modules:
            kfac_state = self.kfac_state_dict[module]
            self.ekfac_state_dict[module] = EKFACState(
                Q_A=torch.symeig(kfac_state.A, eigenvectors=True).eigenvectors,
                Q_S=torch.symeig(kfac_state.S, eigenvectors=True).eigenvectors,
                scale=kfac_state.A.new_zeros((kfac_state.S.size(0), kfac_state.A.size(0))),
                weight=kfac_state.weight,
                bias=kfac_state.bias
            )

    def _del_temp_states(self) -> None:
        del self.a_dict
        del self.g_dict

    def _register_hooks(self) -> List[RemovableHandle]:
        hook_handles = []
        for module in self.modules:
            handle = module.register_forward_hook(self._forward_hook)
            hook_handles.append(handle)
        return hook_handles

    def _remove_hooks(self, hook_handles: List[RemovableHandle]) -> None:
        for handle in hook_handles:
            handle.remove()

    @torch.no_grad()
    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
        if type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
            input, _ = input # primal input
            _, output = output # tangent output
        elif type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
            input, = input
            output = output
        else:
            raise NotImplementedError
        self.a_dict[module] = input.detach()
        def _tensor_backward_hook(grad: torch.Tensor) -> None:
            self.g_dict[module] = grad.detach()
        output.register_hook(_tensor_backward_hook)

    @torch.no_grad()
    def _accumulate_curvature_step(self) -> None:
        for module in self.modules:
            a = self.a_dict[module]
            g = self.g_dict[module]
            kfac_state = self.kfac_state_dict[module]
            kfac_state.A.add_(compute_A(module, a))
            kfac_state.S.add_(compute_S(module, g))
        self.n_iter += 1

    @torch.no_grad()
    def _accumulate_scale_step(self) -> None:
        '''scale = E[ (Q_A⊗Q_S)' @ grad)**2 ]'''
        for module in self.modules:
            ekfac_state = self.ekfac_state_dict[module]
            a = self.a_dict[module]
            g = self.g_dict[module]

            # compute per-sample gradient
            if isinstance(module, nn.Linear):
                # a.shape = (B, Nin)
                # g.shape = (B, Nout)
                if module.bias is not None:
                    a = F.pad(a, (0,1), value=1) # (B, Nin+1)
                grad = torch.bmm(g[:, :, None], a[:, None, :]) # (B, Nout, Nin)
            elif isinstance(module, nn.Conv2d):
                # a.shape = (B, Cin, h, w)
                # g.shape = (B, Cout, h, w)
                b = g.size(0)
                hw = g.size(2)*g.size(3)
                a = F.unfold(a, module.kernel_size, module.dilation, module.padding, module.stride) # (B, Cin*k*k, h*w)
                if module.bias is not None:
                    a = F.pad(a, (0,0,0,1,0,0), value=1) # (B, Cin*k*k+1, h*w)
                # a = a.transpose(1, 2).reshape(b*hw, -1) # (B*h*w, Cin*k*k)
                # g = g.permute(0, 2, 3, 1).reshape(b*hw, -1) # (B*h*w, Cout)
                # grad = torch.bmm(g[:, :, None], a[:, None, :]) # (B*h*w, Cout, Cin*k*k)
                # grad = grad.reshape(b, hw, grad.size(1), grad.size(2)).sum(1) # (B, Cout, Cin*k*k)
                
                g = g.reshape(g.size(0), g.size(1), -1) # (B, Cout, h*w)
                # grad = torch.einsum("bij, bkj -> bik", g, a)
                grad = oe.contract("bij, bkj -> bik", g, a, backend='torch')

            else:
                raise NotImplementedError
            # grad = torch.chain_matmul(ekfac_state.Q_S.T, grad, ekfac_state.Q_A)
            # (Q_A⊗Q_S).T @ grad
            # grad = torch.einsum("ij, bjk, kl -> bil", ekfac_state.Q_S.T, grad, ekfac_state.Q_A) # TODO: not sure about this
            grad = oe.contract("ij, bjk, kl -> bil", ekfac_state.Q_S.T, grad, ekfac_state.Q_A, backend='torch') # TODO: not sure about this
            scale = grad**2 # (B, Nout, Nin) or (B, Cout, Cin*k*k)
            ekfac_state.scale.add_(scale.mean(0))
        self.n_iter += 1

    @torch.no_grad()
    def _divide_curvature(self) -> None:
        for module in self.modules:
            kfac_state = self.kfac_state_dict[module]
            kfac_state.A.div_(self.n_iter)
            kfac_state.S.div_(self.n_iter)

    @torch.no_grad()
    def _divide_scale(self) -> None:
        for module in self.modules:
            ekfac_state = self.ekfac_state_dict[module]
            ekfac_state.scale.div_(self.n_iter)

    @torch.no_grad()
    def _update_center(self):
        for module in self.modules:
            if type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
                self.ekfac_state_dict[module].weight = module.weight_tangent.clone()
                self.ekfac_state_dict[module].bias = module.bias_tangent.clone() if module.bias_tangent is not None else None
            elif type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
                self.ekfac_state_dict[module].weight = module.weight.clone()
                self.ekfac_state_dict[module].bias = module.bias.clone() if module.bias is not None else None
            else:
                raise NotImplementedError

    def compute_curvature(self, dataset: Dataset, n_steps: int, t: int = None, pseudo_target_fn = torch.normal) -> None:
        data_loader = MultiEpochsDataLoader(
                            dataset,
                            batch_size=64,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4,
                        )
        data_loader_cycle = icycle(data_loader)

        hook_handles = self._register_hooks()
        self.model.eval()
        for _ in trange(n_steps):
            input, _ = next(data_loader_cycle)
            input = input.cuda()
            self.model.zero_grad()
            if t is not None:
                output = self.model(input)[t]
            else:
                output = self.model(input)
            pseudo_target = pseudo_target_fn(output.detach())
            loss = self.criterion(output, pseudo_target).sum(-1).sum()
            loss.backward()
            self._accumulate_curvature_step()

        self._divide_curvature()
        self._remove_hooks(hook_handles)

        self.n_iter = 0

        self._init_ekfac_states()
        hook_handles = self._register_hooks()
        self.model.eval()
        for _ in trange(n_steps):
            input, _ = next(data_loader_cycle)
            input = input.cuda()
            self.model.zero_grad()
            if t is not None:
                output = self.model(input)[t]
            else:
                output = self.model(input)
            pseudo_target = pseudo_target_fn(output.detach())
            loss = self.criterion(output, pseudo_target).sum(-1).sum()
            loss.backward()
            self._accumulate_scale_step()
        self._divide_scale()
        self._remove_hooks(hook_handles)
        self._del_temp_states()

        self._update_center()

        del self.kfac_state_dict



    def compute_loss(self) -> torch.Tensor:
        losses = []
        for module in self.modules:
            if type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
                losses.append(EKFAC_penalty.apply(self.ekfac_state_dict[module], module.weight_tangent, module.bias_tangent))
            elif type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
                losses.append(EKFAC_penalty.apply(self.ekfac_state_dict[module], module.weight, module.bias))
            else:
                raise NotImplementedError
        loss = sum(losses)
        return loss


class EKFAC_penalty(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ekfac_state: EKFACState, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        weight_aug, fold_weight_fn = unfold_weight(weight, bias)
        weight_aug_center, _ = unfold_weight(ekfac_state.weight, ekfac_state.bias)

        V = weight_aug - weight_aug_center
        # Hv = torch.chain_matmul(
        #     ekfac_state.Q_S,
        #     torch.chain_matmul(ekfac_state.Q_S.T, V, ekfac_state.Q_A) * ekfac_state.scale,
        #     ekfac_state.Q_A.T
        # )

        Hv = oe.contract("ea, ab, bc, cd, ad, df -> ef", ekfac_state.Q_S, ekfac_state.Q_S.T, V, ekfac_state.Q_A, ekfac_state.scale, ekfac_state.Q_A.T, backend='torch')

        Hdw, Hdb = fold_weight_fn(Hv)
        ctx.save_for_backward(Hdw, Hdb)
        # return torch.dot(V.view(-1), Hv.view(-1)) * 0.5
        return oe.contract("ab, ab -> ", V, Hv, backend='torch') * 0.5

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # grad_output.shape == (,)
        Hdw, Hdb = ctx.saved_tensors
        grad_weight = grad_output * Hdw
        if Hdb is not None:
            grad_bias = grad_output * Hdb
        else:
            grad_bias = None
        return None, grad_weight, grad_bias

