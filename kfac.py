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

@dataclass
class KFACState:
    S: torch.Tensor
    A: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor


@dataclass
class EWCState:
    g: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor


@torch.no_grad()
def compute_A(module: nn.Module, a: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        # a.shape == (B, Nin)
        if module.bias_tangent is not None: #TODO
            a = F.pad(a, (0,1), value=1) # (B, Nin+1)
        a = a.t() # (Nin, B)
        A = a @ a.t() # (Nin, Nin)
    elif isinstance(module, nn.Conv2d):
        # a.shape == (B, Cin, h, w)
        a = F.unfold(a, module.kernel_size, module.dilation, module.padding, module.stride) # (B, Cin*k*k, h*w)
        a = a.transpose(0,1).contiguous() # (Cin*k*k, B, h*w), might be slow due to memcopy
        if module.bias_tangent is not None: #TODO
            a = F.pad(a, (0,0,0,0,0,1), value=1) # (Cin*k*k+1, B, h*w)
        A = a.view(a.size(0), -1) @ a.view(a.size(0), -1).t() # (Cin*k*k, B*h*w)@(B*h*w, Cin*k*k) = (Cin*k*k, Cin*k*k)
    elif isinstance(module, nn.BatchNorm2d):
        # a.shape == (B, Cin, h, w)
        a = a.view(a.size(0), a.size(1), -1) # (B, C, h*w)
        a = a.transpose(0,1).contiguous() # (C, B, h*w), might be slow due to memcopy
        if module.bias_tangent is not None: #TODO
            a = F.pad(a, (0,0,0,0,0,1), value=1) # (C, B, h*w)
        A = a.view(a.size(0), -1) @ a.view(a.size(0), -1).t() # (C, B*h*w)@(B*h*w, C) = (C, C)
    else:
        raise NotImplementedError(f'{type(module)}')
    return A


@torch.no_grad()
def compute_S(module: nn.Module, g: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        # g.shape == (B, Nout)
        g = g.t() # (Nout, B)
        S = g @ g.t() # (Nout, Nout)
    elif isinstance(module, nn.Conv2d):
        # g.shape == (B, Cout, h, w)
        g = g.transpose(0,1) # (Cout, B, h, w)
        g = g.view(g.size(0), g.size(1), -1).contiguous() # (Cout, B, h*w)
        S = g.view(g.size(0),-1) @ g.view(g.size(0),-1).t() # (Cout, B*h*w) @ (Cout, B*h*w) = (Cout, Cout)
    elif isinstance(module, nn.BatchNorm2d):
        # g.shape == (B, C, h, w)
        g = g.transpose(0,1) # (C, B, h, w)
        g = g.view(g.size(0), g.size(1), -1).contiguous() # (C, B, h*w)
        S = g.view(g.size(0),-1) @ g.view(g.size(0),-1).t() # (C, B*h*w) @ (C, B*h*w) = (C, C)
    else:
        raise NotImplementedError(f'{type(module)}')
    return S


@torch.no_grad()
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
    layer_types = (CustomLinear, CustomConv2d, CustomBatchNorm2d)

    def __init__(self, model: nn.Module, criterion: nn.Module) -> None:
        self.model = model
        self.criterion = criterion
        self.modules : List[nn.Module] = [m for m in self.model.modules() if isinstance(m, self.layer_types)]
        self.a_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.g_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.kfac_state_dict : Mapping[nn.Module, KFACState] = OrderedDict()
        self.n_samples = 0
        self.hook_handles : List[RemovableHandle] = []
        self._init_kfac_states()

    def _init_kfac_states(self) -> None:
        for module in self.modules:
            self.kfac_state_dict[module] = KFACState(
                                S = None,
                                A = None,
                                weight = None,
                                bias = None,
                            )

    def _del_temp_states(self) -> None:
        del self.a_dict
        del self.g_dict

    def _register_hooks(self) -> None:
        for module in self.modules:
            # handle = module.register_forward_pre_hook(self._forward_pre_hook)
            # self.hook_handles.append(handle)
            handle = module.register_forward_hook(self._forward_hook)
            self.hook_handles.append(handle)
            # handle = module.register_full_backward_hook(self._backward_hook)
            # self.hook_handles.append(handle)

    def _remove_hooks(self) -> None:
        for handle in self.hook_handles:
            handle.remove()

    # @torch.no_grad()
    # def _forward_pre_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...]) -> None:
    #     a, _ = input # primal input
    #     self.a_dict[module] = a.detach().clone()

    @torch.no_grad()
    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
        a, _ = input # primal input
        self.a_dict[module] = a.detach()

        _, jvp = output # primal output
        def _tensor_backward_hook(grad: torch.Tensor) -> None:
            self.g_dict[module] = grad.detach()
        jvp.register_hook(_tensor_backward_hook)

    # @torch.no_grad()
    # def _backward_hook(self, module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
    #     _, g_jvp = grad_output # grad of jvp output
    #     self.g_dict[module] = g_jvp.detach()

    @torch.no_grad()
    def _accumulate_curvature_step(self) -> None:
        for module in self.modules:
            a = self.a_dict[module]
            g = self.g_dict[module]
            kfac_state = self.kfac_state_dict[module]
            if kfac_state.S is None:
                kfac_state.S = compute_S(module, g)
            else:
                kfac_state.S.add_(compute_S(module, g))
            if kfac_state.A is None:
                kfac_state.A = compute_A(module, a)
            else:
                kfac_state.A.add_(compute_A(module, a))
        self.n_samples += a.size(0)

    @torch.no_grad()
    def _divide_curvature(self) -> None:
        for module in self.modules:
            kfac_state = self.kfac_state_dict[module]
            kfac_state.A.div_(self.n_samples)
            kfac_state.S.div_(self.n_samples)

    @torch.no_grad()
    def _update_center(self):
        for module in self.modules:
            self.kfac_state_dict[module].weight = module.weight_tangent.clone()
            self.kfac_state_dict[module].bias = module.bias_tangent.clone() if module.bias_tangent is not None else None

    def compute_curvature(self, dataset: Dataset, t: int, n_steps: int) -> None:
        data_loader = MultiEpochsDataLoader(
                            dataset,
                            batch_size=64,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4,
                        )
        data_loader_cycle = icycle(data_loader)

        self._register_hooks()
        self.model.eval()
        for _ in trange(n_steps):
            input, _ = next(data_loader_cycle)
            input = input.cuda()
            self.model.zero_grad()
            output = self.model(input)[t]
            pseudo_target = torch.normal(output.detach())
            # loss = self.criterion(output, pseudo_target).sum(-1).mean()
            loss = self.criterion(output, pseudo_target).sum(-1).sum()
            loss.backward()
            self._accumulate_curvature_step()

        self._divide_curvature()
        self._update_center()
        self._remove_hooks()
        self._del_temp_states()

    def compute_loss(self) -> torch.Tensor:
        loss = sum(KFAC_penalty.apply(self.kfac_state_dict[m], m.weight_tangent, m.bias_tangent) for m in self.modules)
        return loss


    def train_dataset(self, dataset: Dataset, t: int, n_steps: int) -> None:
        data_loader = MultiEpochsDataLoader(
                            dataset,
                            batch_size=64,
                            shuffle=True,
                            drop_last=True,
                            num_workers=4,
                        )
        optimizer = optim.Adam(self.model.parameters(), lr=1e-5)
        data_loader_cycle = icycle(data_loader)
        state_dict = self.model.state_dict().copy()
        self.model.eval()
        for _ in trange(n_steps):
            input, target = next(data_loader_cycle)
            input = input.cuda()
            self.model.zero_grad()
            output = self.model(input)[t]
            pseudo_target = torch.normal(output.detach())
            loss = self.criterion(output, 15.*F.one_hot(target, num_classes=10).float()).sum(-1).mean()
            loss.backward()


class KFAC_penalty(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kfac_state: KFACState, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):

        param, fold_weight_fn = unfold_weight(weight, bias)
        center, fold_weight_fn = unfold_weight(kfac_state.weight, kfac_state.bias)
        ctx.fold_weight_fn = fold_weight_fn

        S = kfac_state.S # (Nout, Nout)
        A = kfac_state.A # (Nin, Nin)

        dw = param - center
        Hdw = torch.chain_matmul(S, dw, A) # (Nout, Nin) TODO: use broadcasting for diagonal layer
        ctx.save_for_backward(Hdw)

        return dw.view(-1).dot(Hdw.view(-1))*0.5

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # grad_output.shape == (,)
        Hdw, = ctx.saved_tensors
        grad_param = grad_output*Hdw
        grad_weight, grad_bias = ctx.fold_weight_fn(grad_param)
        return None, grad_weight, grad_bias