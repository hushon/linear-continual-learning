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
from torch import optim


@dataclass
class KFACState:
    A: torch.Tensor = None
    S: torch.Tensor = None


@dataclass
class EKFACState:
    Q_A: torch.Tensor = None
    Q_S: torch.Tensor = None
    scale: torch.Tensor = None # Hessian eigenvalues


@dataclass
class TKFACState:
    A: torch.Tensor = None
    S: torch.Tensor = None
    trace: torch.Tensor = None


@dataclass
class EWCState:
    G_weight: torch.Tensor = None
    G_bias: Optional[torch.Tensor] = None

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
        # A /= a.size(0)
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
                S=module.weight.new_zeros((g_size, g_size))
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
        self.a_dict[module] = input.detach().clone()
        def _tensor_backward_hook(grad: torch.Tensor) -> None:
            self.g_dict[module] = grad.detach().clone()
        output.requires_grad_(True).register_hook(_tensor_backward_hook)

    @torch.no_grad()
    def _accumulate_curvature_step(self) -> None:
        for module in self.modules:
            a = self.a_dict[module]
            g = self.g_dict[module]
            kfac_state = self.kfac_state_dict[module]
            A = compute_A(module, a)
            S = compute_S(module, g)
            kfac_state.A.add_(A)
            kfac_state.S.add_(S)
        self.n_iter += 1

    def _divide_curvature(self) -> None:
        for module in self.modules:
            kfac_state = self.kfac_state_dict[module]
            kfac_state.A.div_(self.n_iter)
            kfac_state.S.div_(self.n_iter)

    # @torch.no_grad()
    # def _update_center(self):
    #     for module in self.modules:
    #         if type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
    #             self.kfac_state_dict[module].weight = module.weight_tangent.clone()
    #             self.kfac_state_dict[module].bias = module.bias_tangent.clone() if module.bias_tangent is not None else None
    #         elif type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
    #             self.kfac_state_dict[module].weight = module.weight.clone()
    #             self.kfac_state_dict[module].bias = module.bias.clone() if module.bias is not None else None
    #         else:
    #             raise NotImplementedError

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
        for _ in trange(n_steps, desc="compute curvature"):
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

    def compute_loss(self, center_state_dict: Mapping[nn.Module, CenterState], damping: float = None) -> torch.Tensor:
        loss = 0.
        for module in self.modules:
            kfac_state = self.kfac_state_dict[module]
            center_state = center_state_dict[module]
            if damping is not None:
                dim_a = kfac_state.A.size(0)
                dim_g = kfac_state.S.size(0)
                pi = torch.sqrt((kfac_state.A.trace()*dim_g)/(kfac_state.S.trace()*dim_a))
                damping = pi.new_tensor(damping)
                kfac_state = KFACState(
                    A = kfac_state.A + torch.zeros_like(kfac_state.A).fill_diagonal_(torch.sqrt(damping) * pi),
                    S = kfac_state.S + torch.zeros_like(kfac_state.S).fill_diagonal_(torch.sqrt(damping) / pi)
                )
            if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
                loss += KFAC_penalty.apply(kfac_state, center_state, module.weight, module.bias)
            elif type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
                loss += KFAC_penalty.apply(kfac_state, center_state, module.weight_tangent, module.bias_tangent)
            else:
                raise NotImplementedError
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


class KroneckerBiliearProduct(torch.autograd.Function):
    """Computes the product v.T @ (A⊗B) @ v

    Returns:
        torch.Tensor: bilinear product (A⊗B)v
    """
    # (A⊗B)v = vec(BVA')
    pass


class EWCRegularizer:
    def __init__(self, model: nn.Module, criterion: nn.Module, modules: List[nn.Module]):
        self.model = model
        self.criterion = criterion
        self.modules : List[nn.Module] = modules
        self.a_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.g_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
        self.ewc_state_dict : Mapping[nn.Module, EWCState] = OrderedDict()
        self.n_iter = 0
        self._init_ewc_states()

    def _init_ewc_states(self):
        for module in self.modules:
            if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
                weight = module.weight
                bias = module.bias
            elif type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
                weight = module.weight_tangent
                bias = module.bias_tangent
            else:
                raise NotImplementedError
            self.ewc_state_dict[module] = EWCState(
                G_weight=torch.zeros_like(weight),
                G_bias=torch.zeros_like(bias) if bias is not None else None
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
            _, output = output # tangent output (=jvp)
        else:
            raise NotImplementedError
        self.a_dict[module] = input.detach().clone()
        def _tensor_backward_hook(grad: torch.Tensor) -> None:
            self.g_dict[module] = grad.detach().clone()
        output.requires_grad_(True).register_hook(_tensor_backward_hook)

    def _del_temp_states(self) -> None:
        del self.a_dict
        del self.g_dict

    @torch.no_grad()
    def _accumulate_curvature_step(self):
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

            ewc_state = self.ewc_state_dict[module]
            ewc_state.G_weight.add_(grad_weight.square().mean(0))
            if ewc_state.G_bias is not None:
                ewc_state.G_bias.add_(grad_bias.square().mean(0))

        self.n_iter += 1

    def _divide_curvature(self):
        for ewc_state in self.ewc_state_dict.values():
            ewc_state.G_weight.div_(self.n_iter)
            if ewc_state.G_bias is not None:
                ewc_state.G_bias.div_(self.n_iter)

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
        for _ in trange(n_steps, desc="compute curvature"):
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

    def compute_loss(self, center_dict: Mapping[nn.Module, CenterState]) -> torch.Tensor:
        loss = 0.
        for module in self.modules:
            ewc_state = self.ewc_state_dict[module]
            center_state = center_dict[module]
            if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
                weight = module.weight
                bias = module.bias
            elif type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
                weight = module.weight_tangent
                bias = module.bias_tangent
            else:
                raise NotImplementedError
            loss += torch.sum(ewc_state.G_weight * torch.square(weight - center_state.weight))
            if bias is not None:
                loss += torch.sum(ewc_state.G_bias * torch.square(bias - center_state.bias))
        return 0.5 * loss


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
                A=module.weight.new_zeros((a_size, a_size))
            )

    def _init_ekfac_states(self) -> None:
        for module in self.modules:
            kfac_state = self.kfac_state_dict[module]
            self.ekfac_state_dict[module] = EKFACState(
                Q_A=torch.symeig(kfac_state.A, eigenvectors=True).eigenvectors,
                Q_S=torch.symeig(kfac_state.S, eigenvectors=True).eigenvectors,
                scale=kfac_state.A.new_zeros((kfac_state.S.size(0), kfac_state.A.size(0))),
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

    @torch.no_grad()
    def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
        if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
            input, = input
            output = output
        elif type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
            input, _ = input # primal input
            _, output = output # tangent output (=jvp)
        else:
            raise NotImplementedError
        self.a_dict[module] = input.detach()
        def _tensor_backward_hook(grad: torch.Tensor) -> None:
            self.g_dict[module] = grad.detach()
        output.requires_grad_(True).register_hook(_tensor_backward_hook)

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
            if type(module) in (nn.Linear, CustomLinear):
                # a.shape = (B, Nin)
                # g.shape = (B, Nout)
                if module.bias is not None:
                    a = F.pad(a, (0,1), value=1) # (B, Nin+1)
                grad = torch.einsum("bi, bj -> bij", g, a)
            elif type(module) in (nn.Conv2d, CustomConv2d):
                # a.shape = (B, Cin, h, w)
                # g.shape = (B, Cout, h, w)
                a = F.unfold(a, module.kernel_size, module.dilation, module.padding, module.stride) # (B, Cin*k*k, h*w)
                if module.bias is not None:
                    a = F.pad(a, (0,0,0,1,0,0), value=1) # (B, Cin*k*k+1, h*w)
                g = g.reshape(g.size(0), g.size(1), -1) # (B, Cout, h*w)
                grad = torch.einsum("bij, bkj -> bik", g, a)
            elif type(module) in (nn.BatchNorm2d, CustomBatchNorm2d):
                # a.shape = (B, C, h, w)
                # g.shape = (B, C, h, w)
                a = (a - module.running_mean[None, :, None, None]).div(torch.sqrt(module.running_var[None, :, None, None] + module.eps))
                grad_weight = torch.einsum("bchw, bchw -> bc", a, g)
                grad_bias = torch.einsum("bchw -> bc", g)
                grad = torch.cat((torch.diag_embed(grad_weight), grad_bias[:, :, None]), dim=2) # (B, C, C+1)
                # grad = torch.stack((grad_weight, grad_bias), dim=2) # (B, C, 2)
            else:
                raise NotImplementedError
            # grad = torch.chain_matmul(ekfac_state.Q_S.T, grad, ekfac_state.Q_A)
            grad = torch.einsum("ij, bjk, kl -> bil", ekfac_state.Q_S.T, grad, ekfac_state.Q_A)
            ekfac_state.scale.add_(grad.pow(2).mean(0)) # (Nout, Nin) or (Cout, Cin*k*k)
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

    # @torch.no_grad()
    # def _update_center(self):
    #     for module in self.modules:
    #         if type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
    #             self.ekfac_state_dict[module].weight = module.weight_tangent.clone()
    #             self.ekfac_state_dict[module].bias = module.bias_tangent.clone() if module.bias_tangent is not None else None
    #         elif type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
    #             self.ekfac_state_dict[module].weight = module.weight.clone()
    #             self.ekfac_state_dict[module].bias = module.bias.clone() if module.bias is not None else None
    #         else:
    #             raise NotImplementedError

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
        for _ in trange(n_steps, desc="compute curvature"):
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
        # del self.kfac_state_dict

        hook_handles = self._register_hooks()
        self.model.eval()
        for _ in trange(n_steps, desc="compute scaling"):
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

        for m in self.modules:
            for p in m.parameters():
                p.requires_grad_(True)

    def compute_loss(self, center_dict: Mapping[nn.Module, CenterState]) -> torch.Tensor:
        losses = []
        for module in self.modules:
            if type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
                losses.append(EKFAC_penalty.apply(self.ekfac_state_dict[module], center_dict[module], module.weight_tangent, module.bias_tangent))
            elif type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
                losses.append(EKFAC_penalty.apply(self.ekfac_state_dict[module], center_dict[module], module.weight, module.bias))
            else:
                raise NotImplementedError
        loss = sum(losses)
        return loss


class EKFAC_penalty(torch.autograd.Function):
    """
    Hv = (Q_A⊗Q_S) @ Λ @ (Q_A⊗Q_S).T @ v
    out = 0.5 * vHv
    dout/dv = Hv
    """

    @staticmethod
    def forward(ctx, ekfac_state: EKFACState, center_state: CenterState, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # if weight.ndim == 1: # batchnorm
        #     # weight.shape = (C,)
        #     # bias.shape = (C,)
        #     pass
        weight_aug, fold_weight_fn = unfold_weight(weight, bias)
        weight_aug_center, _ = unfold_weight(center_state.weight, center_state.bias)

        V = weight_aug - weight_aug_center
        # Hv = torch.chain_matmul(
        #     ekfac_state.Q_S,
        #     torch.chain_matmul(ekfac_state.Q_S.T, V, ekfac_state.Q_A) * ekfac_state.scale,
        #     ekfac_state.Q_A.T
        # )

        Hv = torch.einsum("ea, ab, bc, cd, ad, df -> ef", ekfac_state.Q_S, ekfac_state.Q_S.T, V, ekfac_state.Q_A, ekfac_state.scale, ekfac_state.Q_A.T)

        Hdw, Hdb = fold_weight_fn(Hv)
        ctx.save_for_backward(Hdw, Hdb)
        # return torch.dot(V.view(-1), Hv.view(-1)) * 0.5
        return torch.einsum("ab, ab -> ", V, Hv) * 0.5

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


@torch.no_grad()
def compute_trace(module: nn.Module, a: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    b = a.size(0)
    if isinstance(module, nn.Linear):
        # a.shape == (B, Nin)
        # g.shape == (B, Nout)
        if module.bias is not None: #TODO
            a = F.pad(a, (0,1), value=1) # (B, Nin+1)
        trace = a.square().sum(1) * g.square().sum(1)
        trace = trace.mean()
    elif isinstance(module, nn.Conv2d):
        # a.shape == (B, Cin, h, w)
        # g.shape == (B, Cout, h, w)
        hw = a.size(2) * a.size(3)
        a = F.unfold(a, module.kernel_size, module.dilation, module.padding, module.stride) # (B, Cin*k*k, h*w)
        if module.bias is not None:
            a = F.pad(a, (0,0,0,1,0,0), value=1) # (B, Cin*k*k+1, h*w)
        trace = a.view(b, -1).square().sum(1) * g.view(b, -1).square().sum(1)
        trace = trace.mean().div_(hw)
    else:
        raise NotImplementedError(f'{type(module)}')
    return trace


# class TKFACRegularizer:

#     def __init__(self, model: nn.Module, criterion: nn.Module, modules: List[nn.Module]) -> None:
#         self.model = model
#         self.criterion = criterion
#         self.modules : List[nn.Module] = modules
#         self.a_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
#         self.g_dict : Mapping[nn.Module, torch.Tensor] = OrderedDict()
#         self.tkfac_state_dict : Mapping[nn.Module, TKFACState] = OrderedDict()
#         self.n_iter = 0
#         self._init_tkfac_states()

#     def _init_tkfac_states(self) -> None:
#         for module in self.modules:
#             if isinstance(module, nn.Linear):
#                 a_size = module.in_features
#                 g_size = module.out_features
#             elif isinstance(module, nn.Conv2d):
#                 a_size = module.in_channels * module.kernel_size[0] * module.kernel_size[1]
#                 g_size = module.out_channels
#             elif isinstance(module, nn.BatchNorm2d):
#                 a_size = g_size = module.num_features
#             else:
#                 raise NotImplementedError()

#             if module.bias is not None:
#                 a_size += 1

#             self.tkfac_state_dict[module] = KFACState(
#                 A=module.weight.new_zeros((a_size, a_size)),
#                 S=module.weight.new_zeros((g_size, g_size)),
#                 trace=module.weight.new_tensor(0.)
#             )

#     def _del_temp_states(self) -> None:
#         del self.a_dict
#         del self.g_dict

#     def _register_hooks(self) -> List[RemovableHandle]:
#         hook_handles = []
#         for module in self.modules:
#             handle = module.register_forward_hook(self._forward_hook)
#             hook_handles.append(handle)
#         return hook_handles

#     def _remove_hooks(self, hook_handles: List[RemovableHandle]) -> None:
#         for handle in hook_handles:
#             handle.remove()
#         hook_handles.clear()

#     def _forward_hook(self, module: nn.Module, input: Tuple[torch.Tensor, ...], output: Tuple[torch.Tensor, ...]) -> None:
#         if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
#             input, = input
#             output = output
#         elif type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
#             input, _ = input # primal input
#             _, output = output # tangent output (=jvp)
#         else:
#             raise NotImplementedError
#         self.a_dict[module] = input.detach().clone()
#         def _tensor_backward_hook(grad: torch.Tensor) -> None:
#             self.g_dict[module] = grad.detach().clone()
#         output.requires_grad_(True).register_hook(_tensor_backward_hook)

#     @torch.no_grad()
#     def _accumulate_curvature_step(self) -> None:
#         for module in self.modules:
#             a = self.a_dict[module]
#             g = self.g_dict[module]
#             tkfac_state = self.tkfac_state_dict[module]
#             A = compute_A(module, a)
#             S = compute_S(module, g)
#             trace = compute_trace(module, a, g)
#             tkfac_state.A.add_(A)
#             tkfac_state.S.add_(S)
#             tkfac_state.trace.add_(trace)
#         self.n_iter += 1

#     def _divide_curvature(self) -> None:
#         for module in self.modules:
#             tkfac_state = self.tkfac_state_dict[module]
#             tkfac_state.A.div_(self.n_iter)
#             tkfac_state.S.div_(self.n_iter)
#             tkfac_state.trace.div_(self.n_iter)

#     def compute_curvature(self, dataset: Dataset, n_steps: int, t: int = None, pseudo_target_fn = torch.normal, batch_size=64) -> None:
#         data_loader = MultiEpochsDataLoader(
#                             dataset,
#                             batch_size=batch_size,
#                             shuffle=True,
#                             drop_last=True,
#                             num_workers=4,
#                         )
#         data_loader_cycle = icycle(data_loader)

#         for m in self.modules:
#             for p in m.parameters():
#                 p.requires_grad_(False)

#         hook_handles = self._register_hooks()
#         self.model.eval()
#         for _ in trange(n_steps, desc="compute curvature"):
#             input, _ = next(data_loader_cycle)
#             input = input.cuda()
#             self.model.zero_grad()
#             if t is not None:
#                 output = self.model(input)[t]
#             else:
#                 output = self.model(input)
#             pseudo_target = pseudo_target_fn(output.detach())
#             loss = self.criterion(output, pseudo_target).sum(-1).sum()
#             loss.backward()
#             self._accumulate_curvature_step()

#         self._divide_curvature()
#         self._remove_hooks(hook_handles)
#         self._del_temp_states()

#         for m in self.modules:
#             for p in m.parameters():
#                 p.requires_grad_(True)

#     def add_damping(self, damping: float = None):
#         dim_a = tkfac_state.A.size(0)
#         dim_g = tkfac_state.S.size(0)
#         pi = torch.sqrt((tkfac_state.A.trace()*dim_g)/(tkfac_state.S.trace()*dim_a))
#         damping = pi.new_tensor(damping)
#         tkfac_state = KFACState(
#             A = tkfac_state.A + torch.zeros_like(tkfac_state.A).fill_diagonal_(torch.sqrt(damping) * pi),
#             S = tkfac_state.S + torch.zeros_like(tkfac_state.S).fill_diagonal_(torch.sqrt(damping) / pi)
#         )

#     def compute_loss(self, center_state_dict: Mapping[nn.Module, CenterState]) -> torch.Tensor:
#         loss = 0.
#         for module in self.modules:
#             tkfac_state = self.tkfac_state_dict[module]
#             center_state = center_state_dict[module]
#             kfac_state = KFACState(
#                 A=tkfac_state.A*tkfac_state.trace / tkfac_state.A.trace(),
#                 S=tkfac_state.S / tkfac_state.S.trace(),
#             )
#             if type(module) in (nn.Linear, nn.Conv2d, nn.BatchNorm2d):
#                 loss += KFAC_penalty.apply(kfac_state, center_state, module.weight, module.bias)
#             elif type(module) in (CustomLinear, CustomConv2d, CustomBatchNorm2d):
#                 loss += KFAC_penalty.apply(kfac_state, center_state, module.weight_tangent, module.bias_tangent)
#             else:
#                 raise NotImplementedError
#         return loss

