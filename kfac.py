import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.hooks import RemovableHandle
from models.modules import CustomLinear, CustomConv2d, CustomBatchNorm2d
from typing import Tuple, List, NamedTuple, Mapping, Optional, Union, Literal
from tqdm import tqdm, trange
from utils import icycle, MultiEpochsDataLoader
from dataclasses import dataclass


@dataclass
class KFACState:
    S: torch.Tensor
    A: torch.Tensor
    center: torch.Tensor
    # layer_type: Union[Literal['linear'], Literal['conv'], Literal['bn']]


@dataclass
class EWCState:
    G: torch.Tensor
    center: torch.Tensor

# module.bias_tangent instead of module.bias

def compute_A(module: nn.Module, a: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        # a.shape == (B, Nin)
        if module.bias_tangent is not None: #TODO
            a = F.pad(a, (0,1), value=1) # (B, Nin+1)
        a = a.t() # (Nin, B)
        A = (a @ a.t()).div(a.size(1)) # (Nin, Nin)
    elif isinstance(module, nn.Conv2d):
        # a.shape == (B, Cin, h, w)
        a = F.unfold(a, module.kernel_size, module.dilation, module.padding, module.stride) # (B, Cin*k*k, h*w)
        a = a.transpose(0,1).contiguous() # (Cin*k*k, B, h*w), might be slow due to memcopy
        if module.bias_tangent is not None: #TODO
            a = F.pad(a, (0,0,0,0,0,1), value=1) # (Cin*k*k+1, B, h*w)
        A = a.view(a.size(0), -1) @ a.view(a.size(0), -1).t() # (Cin*k*k, B*h*w)@(B*h*w, Cin*k*k) = (Cin*k*k, Cin*k*k)
        A.div_(a.size(1))
    elif isinstance(module, nn.BatchNorm2d):
        # a.shape == (B, Cin, h, w)
        a = a.view(a.size(0), a.size(1), -1) # (B, C, h*w)
        a = a.transpose(0,1).contiguous() # (C, B, h*w), might be slow due to memcopy
        if module.bias_tangent is not None: #TODO
            a = F.pad(a, (0,0,0,0,0,1), value=1) # (C, B, h*w)
        A = a.view(a.size(0), -1) @ a.view(a.size(0), -1).t() # (C, B*h*w)@(B*h*w, C) = (C, C)
        A.div_(a.size(1))
    else:
        raise NotImplementedError(f'{type(module)}')
    return A


def compute_S(module: nn.Module, g: torch.Tensor) -> torch.Tensor:
    if isinstance(module, nn.Linear):
        # g.shape == (B, Nout)
        g = g.t() # (Nout, B)
        S = g @ g.t() # (Nout, Nout)
        S.div_(g.size(1))
    elif isinstance(module, nn.Conv2d):
        # g.shape == (B, Cout, h, w)
        g = g.transpose(0,1) # (Cout, B, h, w)
        g = g.view(g.size(0), g.size(1), -1).contiguous() # (Cout, B, h*w)
        S = g.view(g.size(0),-1) @ g.view(g.size(0),-1).t() # (Cout, B*h*w) @ (Cout, B*h*w) = (Cout, Cout)
        S.div_(g.size(2))
    elif isinstance(module, nn.BatchNorm2d):
        # g.shape == (B, C, h, w)
        g = g.transpose(0,1) # (C, B, h, w)
        g = g.view(g.size(0), g.size(1), -1).contiguous() # (C, B, h*w)
        S = g.view(g.size(0),-1) @ g.view(g.size(0),-1).t() # (C, B*h*w) @ (C, B*h*w) = (C, C)
        S.div_(g.size(2))
    else:
        raise NotImplementedError(f'{type(module)}')
    return S


class KFACRegularizer:
    layer_types = (CustomLinear, CustomConv2d, CustomBatchNorm2d)

    def __init__(self, model: nn.Module, criterion: nn.Module) -> None:
        self.model = model
        self.criterion = criterion
        self.modules : List[nn.Module] = [m for m in self.model.modules() if type(m) in self.layer_types]
        self.a_dict : Mapping[nn.Module, torch.Tensor] = dict()
        self.g_dict : Mapping[nn.Module, torch.Tensor] = dict()
        self.kfac_state_dict : Mapping[nn.Module, KFACState] = dict()
        self.hook_handles : List[RemovableHandle] = []
        self._init_kfac_states()

    def _init_kfac_states(self) -> None:
        for module in self.modules:
            kfac_state = KFACState(
                S = module.weight.new_tensor(0.),
                A = module.weight.new_tensor(0.),
                center = module.weight.new_tensor(0.),
            )
            self.kfac_state_dict[module] = kfac_state

    def _register_hooks(self) -> None:
        for module in self.modules:
            handle = module.register_forward_pre_hook(self._forward_pre_hook)
            self.hook_handles.append(handle)
            handle = module.register_forward_hook(self._forward_hook)
            self.hook_handles.append(handle)
            # handle = module.register_full_backward_hook(self._backward_hook)
            # self.hook_handles.append(handle)

    def _remove_hooks(self) -> None:
        for handle in self.hook_handles:
            handle.remove()

    @torch.no_grad()
    def _forward_pre_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...]) -> None:
        a, _ = input # primal input
        self.a_dict[module] = a.detach().clone()

    @torch.no_grad()
    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
        _, jvp = output # primal output

        def _tensor_backward_hook(grad: torch.Tensor) -> None:
            self.g_dict[module] = grad.data.clone()
        jvp.register_hook(_tensor_backward_hook)

    # @torch.no_grad()
    # def _backward_hook(self, module: nn.Module, grad_input: Tuple[torch.Tensor, ...], grad_output: Tuple[torch.Tensor, ...]) -> None:
    #     _, g_jvp = grad_output # grad of jvp output
    #     self.g_dict[module] = g_jvp.data.clone()

    @torch.no_grad()
    def _update_curvature_step(self, decay=0.99) -> None:
        for module in self.modules:
            a = self.a_dict[module]
            g = self.g_dict[module]
            kfac_state = self.kfac_state_dict[module]
            # kfac_state.A.data = kfac_state.A.mul(decay).add(compute_A(module, a), alpha=1.-decay)
            # kfac_state.S.data = kfac_state.S.mul(decay).add(compute_S(module, g), alpha=1.-decay)
            kfac_state.A.data = kfac_state.A*decay + compute_A(module, a)*(1.-decay)
            kfac_state.S.data = kfac_state.S*decay + compute_S(module, g)*(1.-decay)

    @torch.no_grad()
    def _update_center(self):
        for module in self.modules:
            weight = module.weight_tangent
            bias = module.bias_tangent
            kfac_state = self.kfac_state_dict[module]
            if isinstance(module, CustomLinear):
                if bias is not None:
                    center = torch.cat([weight, bias[:, None]], dim=1)
                else:
                    center = weight # (Nout, Nin)
            elif isinstance(module, CustomConv2d):
                center = weight.reshape(weight.size(0), -1) # (Cout, Cin*k*k)
                if bias is not None:
                    center = torch.cat([center, bias[:, None]], dim=1)
            elif isinstance(module, CustomBatchNorm2d): #TODO
                center = weight.diag() # (C,C)
                if bias is not None:
                    center = torch.cat([center, bias[:, None]], dim=1)
            else:
                raise NotImplementedError(f'{type(module)}')
            kfac_state.center.data = center.clone()

    def compute_curvature(self, dataset: Dataset, t: int, n_steps: int) -> None:
        self._register_hooks()

        data_loader = MultiEpochsDataLoader(
                            dataset,
                            batch_size=64,
                            shuffle=False,
                            drop_last=True,
                            num_workers=4,
                        )

        data_loader_cycle = icycle(data_loader)

        self.model.eval()
        for _ in trange(n_steps):
            input, _ = next(data_loader_cycle)
            input = input.cuda()
            self.model.zero_grad()
            output = self.model(input)[t]
            pseudo_target = torch.normal(output.detach())
            loss = self.criterion(output, pseudo_target).sum(-1).mean()
            loss.backward()
            self._update_curvature_step()
        self._update_center()

        self._remove_hooks()

    def compute_loss(self) -> torch.Tensor:
        loss = sum(KFAC_penalty.apply(self.kfac_state_dict[m], m.weight_tangent, m.bias_tangent) for m in self.modules)
        return loss


class KFAC_penalty(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kfac_state: KFACState, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        if weight.ndim == 2:
            new_center = weight # (Nout, Nin)
        elif weight.ndim == 4:
            new_center = weight.reshape(weight.size(0), -1) # (Cout, Cin*k*k)
        elif weight.ndim == 1:
            new_center = weight.diag() # (C, C) TODO: inefficient implementation
        else:
            raise ValueError(f'Cannot infer layer type: {weight.shape}')

        if bias is not None:
            new_center = torch.cat([new_center, bias[:, None]], dim=1)

        S = kfac_state.S # (Nout, Nout)
        A = kfac_state.A # (Nin, Nin)
        center = kfac_state.center # (Nout, Nin)

        dw = new_center - center
        Hdw = torch.chain_matmul(S, dw, A) # (Nout, Nin) TODO: how to compute for diagonal layer?
        loss = dw.view(-1).dot(Hdw.view(-1))*0.5

        if bias is not None:
            Hdw_weight = Hdw[:, :-1]
            if weight.ndim == 1:
                Hdw_weight = torch.diagonal(Hdw_weight)
            Hdw_bias = Hdw[:, -1]
        else:
            Hdw_weight = Hdw.reshape(weight.shape)
            if weight.ndim == 1:
                Hdw_weight = torch.diagonal(Hdw_weight)
            Hdw_bias = None

        ctx.save_for_backward(Hdw_weight, Hdw_bias)

        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # grad_output.shape == (,)
        Hdw_weight, Hdw_bias = ctx.saved_tensors
        if Hdw_bias is not None:
            grad_weight = grad_output*Hdw_weight
            grad_bias = grad_output*Hdw_bias
        else:
            grad_weight = grad_output*Hdw_weight
            grad_bias = None
        return None, grad_weight, grad_bias