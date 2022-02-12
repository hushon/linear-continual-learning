import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.hooks import RemovableHandle
from models.modules import CustomLinear, CustomConv2d, CustomBatchNorm2d
from typing import Tuple, List, NamedTuple, Mapping, Optional, Callable
from tqdm.auto import tqdm, trange
from utils import icycle, MultiEpochsDataLoader
from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class KFACState:
    A: torch.Tensor = None
    S: torch.Tensor = None


@dataclass
class TKFACState:
    A: torch.Tensor = None
    S: torch.Tensor = None
    trace: torch.Tensor = None


@dataclass
class CenterState:
    weight: torch.Tensor = None
    bias: Optional[torch.Tensor] = None


def get_center_dict(modules: List[nn.Module]) -> Mapping[nn.Module, CenterState]:
    result = OrderedDict()
    for module in modules:
        if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
            weight = module.weight
            bias = module.bias
        elif type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
            weight = module.weight_tangent
            bias = module.bias_tangent
        else:
            raise NotImplementedError
        result[module] = CenterState(
            weight=weight.detach().clone(),
            bias=bias.detach().clone() if bias is not None else None
            )
    return result


@torch.no_grad()
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
    elif isinstance(module, nn.BatchNorm2d):
        # a.shape == (B, C, h, w)
        b = a.size(0)
        a = (a - module.running_mean[None, :, None, None]).div(torch.sqrt(module.running_var[None, :, None, None] + module.eps))
        if module.bias is not None:
            a = F.pad(a, (0,0,0,0,0,1,0,0), value=1) # (B, C+1, h, w)
        A = torch.einsum("bijk, bljk -> il", a, a) # (C, C)
        A /= b
    else:
        raise NotImplementedError(f'{type(module)}')
    return A


@torch.no_grad()
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
    elif isinstance(module, nn.BatchNorm2d):
        # g.shape == (B, C, h, w)
        bhw = g.size(0)*g.size(2)*g.size(3)
        S = torch.einsum("bijk, bljk -> il", g, g)
        S /= bhw
    else:
        raise NotImplementedError(f'{type(module)}')
    return S


@torch.no_grad()
def compute_trace(module: nn.Module, a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    b = a.size(0)
    if isinstance(module, nn.Linear):
        # a.shape == (B, Nin)
        # g.shape == (B, Nout)
        if module.bias is not None: #TODO
            a = F.pad(a, (0,1), value=1) # (B, Nin+1)
        trace = a.square().sum(1) * g.square().sum(1)
        trace = trace.sum(0).div_(b)
    elif isinstance(module, nn.Conv2d):
        # a.shape = (B, Cin, h, w)
        # g.shape = (B, Cout, h, w)
        a = F.unfold(a, module.kernel_size, module.dilation, module.padding, module.stride) # (B, Cin*k*k, h*w)
        g = g.reshape(g.size(0), g.size(1), -1) # (B, Cout, h*w)
        trace = torch.einsum("bij, bkj -> bik", g, a).square().view(b, -1).sum(1)
        if module.bias is not None:
            trace.add_(g.sum(2).square().sum(1))
        # trace.shape = (B, )
        trace = trace.sum(0).div_(b)
    elif isinstance(module, nn.BatchNorm2d):
        # a.shape = (B, C, h, w)
        # g.shape = (B, C, h, w)
        a = (a - module.running_mean[None, :, None, None]).div(torch.sqrt(module.running_var[None, :, None, None] + module.eps))
        trace = torch.einsum("bchw, bchw -> bc", a, g).square().sum(1)
        if module.bias is not None:
            trace.add_(torch.einsum("bchw -> bc", g).square().sum(1))
    else:
        raise NotImplementedError(f'{type(module)}')
    return trace


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
        return torch.chain_matmul(B, vec.view(q, n), A.T).view(-1)
    elif vec.ndim == 2:
        return torch.chain_matmul(B, vec, A.T) # (p, m)
    else:
        raise ValueError


def kfac_loss_batchnorm(kfac_state: KFACState, center_state: CenterState, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert weight.ndim == 1 # is_batchnorm
    S = kfac_state.S # (C, C)
    A = kfac_state.A # (C, C) or (C+1, C+1)

    if bias is not None:
        dw = weight - center_state.weight # (C,)
        db = bias - center_state.bias # (C,)
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
        dw = weight - center_state.weight # (C,)
        Sdw = S * dw[None, :]
        Sv = Sdw # (C, C)
        # Hv = Sv @ A # (C, C)
        # Hdw = Hv.diagonal() # (C,)
        Hdw = (Sv * A).sum(1) # (C,)
        vHv = torch.dot(dw, Hdw)
        return 0.5 * vHv, Hdw, None


def kfac_loss_linear_conv(kfac_state: KFACState, center_state: CenterState, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    A = kfac_state.A # (Nin, Nin)
    S = kfac_state.S # (Nout, Nout)
    param, fold_weight_fn = unfold_weight(weight, bias)
    center, fold_weight_fn = unfold_weight(center_state.weight, center_state.bias)
    v = param - center
    Hv = torch.chain_matmul(S, v, A) # (Nout, Nin)
    vHv = torch.dot(v.view(-1), Hv.view(-1))
    Hdw, Hdb = fold_weight_fn(Hv)
    return 0.5 * vHv, Hdw, Hdb


class KFAC_penalty(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kfac_state: KFACState, center_state: CenterState, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        if weight.ndim == 1:
            loss, Hdw, Hdb = kfac_loss_batchnorm(kfac_state, center_state, weight, bias)
        elif weight.ndim == 2 or weight.ndim == 4:
            loss, Hdw, Hdb = kfac_loss_linear_conv(kfac_state, center_state, weight, bias)
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
        return None, None, grad_weight, grad_bias


class TKFACRegularizer:

    def __init__(self, model: nn.Module, criterion: nn.Module, modules: List[nn.Module]) -> None:
        self.model = model
        self.criterion = criterion
        self.modules : List[nn.Module] = modules
        self.a_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.g_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.tkfac_state_dict : Mapping[nn.Module, TKFACState] = OrderedDict()
        self.n_iter = 0
        self._init_tkfac_states()

    def _init_tkfac_states(self) -> None:
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

            self.tkfac_state_dict[module] = TKFACState(
                A=module.weight.new_zeros((a_size, a_size)),
                S=module.weight.new_zeros((g_size, g_size)),
                trace=module.weight.new_tensor(0.)
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
        hook_handles.clear()

    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
        if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
            input, = input
            output = output
        elif type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
            input, _ = input # primal input
            _, output = output # tangent output (=jvp)
        else:
            raise NotImplementedError
        input: torch.Tensor
        output: torch.Tensor
        self.a_dict[module] = input.detach().clone()
        def _tensor_backward_hook(grad: torch.Tensor) -> None:
            self.g_dict[module] = grad.detach().clone()
        output.requires_grad_(True).register_hook(_tensor_backward_hook)

    @torch.no_grad()
    def _accumulate_curvature_step(self) -> None:
        for module in self.modules:
            a = self.a_dict[module]
            g = self.g_dict[module]
            tkfac_state = self.tkfac_state_dict[module]
            A = compute_A(module, a)
            S = compute_S(module, g)
            trace = compute_trace(module, a, g)
            tkfac_state.A.add_(A)
            tkfac_state.S.add_(S)
            tkfac_state.trace.add_(trace)
        self.n_iter += 1

    def _divide_curvature(self) -> None:
        for module in self.modules:
            tkfac_state = self.tkfac_state_dict[module]
            tkfac_state.A.div_(self.n_iter)
            tkfac_state.S.div_(self.n_iter)
            tkfac_state.trace.div_(self.n_iter)

    def compute_curvature(self, dataset: Dataset, n_steps: int, t: int = None, pseudo_target_fn = torch.normal, batch_size=64) -> None:
        data_loader = MultiEpochsDataLoader(
                            dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4,
                        )
        data_loader_cycle = icycle(data_loader)

        for m in self.modules:
            for p in m.parameters():
                p.requires_grad_(False)

        hook_handles = self._register_hooks()
        self.model.eval()
        for _ in trange(n_steps, desc="compute curvature", dynamic_ncols=True):
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
        self._del_temp_states()

        for m in self.modules:
            for p in m.parameters():
                p.requires_grad_(True)

    def compute_loss(self, center_state_dict: Mapping[nn.Module, CenterState]) -> torch.Tensor:
        loss = 0.
        for module in self.modules:
            tkfac_state = self.tkfac_state_dict[module]
            center_state = center_state_dict[module]
            kfac_state = KFACState(
                A=tkfac_state.A,
                S=tkfac_state.S,
            )
            scaling = tkfac_state.trace / (tkfac_state.A.trace() * tkfac_state.S.trace())
            if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
                loss += KFAC_penalty.apply(kfac_state, center_state, module.weight, module.bias) * scaling
            elif type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
                loss += KFAC_penalty.apply(kfac_state, center_state, module.weight_tangent, module.bias_tangent) * scaling
            else:
                raise NotImplementedError
        return loss

