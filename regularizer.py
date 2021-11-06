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
class MASState:
    O_weight: torch.Tensor = None
    O_bias: Optional[torch.Tensor] = None

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


class MASRegularizer:
    def __init__(self, model: nn.Module, modules: List[nn.Module]):
        self.model = model
        self.modules : List[nn.Module] = modules
        self.a_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.g_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.mas_state_dict : Mapping[nn.Module, MASState] = OrderedDict()
        self.n_iter = 0
        self._init_mas_states()

    def _init_mas_states(self):
        for module in self.modules:
            if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
                weight = module.weight
                bias = module.bias
            elif type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
                weight = module.weight_tangent
                bias = module.bias_tangent
            self.mas_state_dict[module] = MASState(
                O_weight=torch.zeros_like(weight),
                O_bias=None if bias is None else torch.zeros_like(bias)
            )

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
            _, output = output # jvp output
        else:
            raise NotImplementedError
        self.a_dict[module] = input.detach()
        def _tensor_backward_hook(grad: torch.Tensor) -> None:
            self.g_dict[module] = grad.detach()
        output.register_hook(_tensor_backward_hook)

    def _del_temp_states(self) -> None:
        del self.a_dict
        del self.g_dict

    @torch.no_grad()
    def _accumulate_importance_step(self):
        for module in self.modules:
            a = self.a_dict[module]
            g = self.g_dict[module]
            if isinstance(module, nn.Linear):
                # a.shape = (B, Nin)
                # g.shape = (B, Nout)
                grad_weight = torch.einsum("bi, bj -> bij", g, a)
                grad_bias = g if module.bias is not None else None
            elif isinstance(module, nn.Conv2d):
                # a.shape = (B, Cin, h, w)
                # g.shape = (B, Cout, h, w)
                a = F.unfold(a, module.kernel_size, module.dilation, module.padding, module.stride) # (B, Cin*k*k, h*w)
                g = g.reshape(g.size(0), g.size(1), -1) # (B, Cout, h*w)
                grad_weight = torch.einsum("bij, bkj -> bik", g, a).reshape(a.size(0), module.out_channels, module.in_channels, *module.kernel_size)
                grad_bias = g.sum(2) if module.bias is not None else None
            elif isinstance(module, nn.BatchNorm2d):
                # a.shape = (B, C, h, w)
                # g.shape = (B, C, h, w)
                a = (a - module.running_mean[None, :, None, None]).div(torch.sqrt(module.running_var[None, :, None, None] + module.eps))
                grad_weight = torch.einsum("bchw, bchw -> bc", a, g)
                grad_bias = torch.einsum("bchw -> bc", g) if module.bias is not None else None
            else:
                raise NotImplementedError

            mas_state = self.mas_state_dict[module]
            mas_state.O_weight.add_(grad_weight.abs().sum(0))
            if mas_state.O_bias is not None:
                mas_state.O_bias.add_(grad_bias.abs().sum(0))

        self.n_iter += 1

    def _divide_importance(self):
        for mas_state in self.mas_state_dict.values():
            mas_state.O_weight.div_(self.n_iter)
            if mas_state.O_bias is not None:
                mas_state.O_bias.div_(self.n_iter)

    def compute_importance(self, dataset: Dataset, n_steps: int, t: int = None) -> None:
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
        for _ in trange(n_steps, desc="compute importance"):
            input, _ = next(data_loader_cycle)
            input = input.cuda()
            self.model.zero_grad()
            if t is not None:
                output = self.model(input)[t]
            else:
                output = self.model(input)
            loss = output.square().sum()
            loss.backward()
            self._accumulate_importance_step()
        self._divide_importance()
        self._remove_hooks(hook_handles)
        self._del_temp_states()

    def compute_loss(self, center_dict: Mapping[nn.Module, CenterState]) -> torch.Tensor:
        losses = []
        for module in self.modules:
            mas_state = self.mas_state_dict[module]
            center_state = center_dict[module]
            if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
                weight = module.weight
                bias = module.bias
            elif type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
                weight = module.weight_tangent
                bias = module.bias_tangent
            else:
                raise NotImplementedError
            loss = torch.sum(mas_state.O_weight * torch.square(weight - center_state.weight))
            if bias is not None:
                loss += torch.sum(mas_state.O_bias * torch.square(bias - center_state.bias))
            losses.append(loss)
        return 0.5 * sum(losses)
