import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Tuple


class CustomSequential(nn.Sequential):
    def forward(self, input, jvp):
        for module in self:
            input, jvp = module(input, jvp)
        return input, jvp


class CustomLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        for k in ('weight', 'bias'):
            x = self._parameters.pop(k)
            if x is not None:
                self.register_buffer(k, x.data)
            else:
                self.register_buffer(k, None)
        self.weight_tangent = nn.Parameter(torch.zeros_like(self.weight))
        if self.bias is None:
            self.register_parameter('bias_tangent', None)
        else:
            self.bias_tangent = nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, input, jvp):
        output = F.linear(input, self.weight, self.bias)
        jvp = F.linear(input, self.weight_tangent, self.bias_tangent) + F.linear(jvp, self.weight, None)
        return output, jvp


class CustomReLU(nn.ReLU):
    def __init__(self, inplace: bool = False) -> None:
        super().__init__(inplace=inplace)

    def forward(self, input, jvp):
        mask = input > 0.
        output = F.relu(input, self.inplace)
        jvp = jvp * mask
        return output, jvp


class CustomLeakyReLU(nn.LeakyReLU):
    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super().__init__(negative_slope=negative_slope, inplace=inplace)

    def forward(self, input, jvp):
        mask = input > 0.
        output = F.leaky_relu(input, self.negative_slope, self.inplace)
        jvp = jvp * torch.where(mask, 1., self.negative_slope)
        # jvp = jvp.where(mask, jvp*self.negative_slope)
        return output, jvp


class CustomConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        super().__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias, padding_mode=padding_mode)
        for k in ('weight', 'bias'):
            x = self._parameters.pop(k)
            if x is not None:
                self.register_buffer(k, x.data)
            else:
                self.register_buffer(k, None)
        self.weight_tangent = nn.Parameter(torch.zeros_like(self.weight))
        if self.bias is None:
            self.register_parameter('bias_tangent', None)
        else:
            self.bias_tangent = nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, input, jvp):
        output = self._conv_forward(input, self.weight, self.bias)
        jvp = self._conv_forward(input, self.weight_tangent, self.bias_tangent) + self._conv_forward(jvp, self.weight, None)
        return output, jvp


class CustomBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        for k in ('weight', 'bias'):
            x = self._parameters.pop(k)
            if x is not None:
                self.register_buffer(k, x.data)
            else:
                self.register_buffer(k, None)
        self.weight_tangent = nn.Parameter(torch.zeros_like(self.weight))
        self.bias_tangent = nn.Parameter(torch.zeros_like(self.bias))

    def forward(self, input, jvp):
        # assert not self.training
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        output = F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
        jvp = F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight_tangent, self.bias_tangent, False, exponential_average_factor, self.eps) \
            + F.batch_norm(
            jvp,
            # If buffers are not to be tracked, ensure that they won't be updated
            torch.zeros_like(self.running_mean) if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight, None, False, exponential_average_factor, self.eps)
        return output, jvp


class CustomAdaptiveAvgPool2d(nn.AdaptiveAvgPool2d):
    def forward(self, input, jvp):
        output = F.adaptive_avg_pool2d(input, self.output_size)
        jvp = F.adaptive_avg_pool2d(jvp, self.output_size)
        return output, jvp


class CustomMaxPool2d(nn.MaxPool2d):
    def forward(self, input: torch.Tensor, jvp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output, indices = F.max_pool2d(input, self.kernel_size, self.stride, self.padding, self.dilation, self.ceil_mode, True)
        b, c, out_h, out_w = output.shape
        jvp = torch.gather(jvp.view(b, c, -1), 2, indices.view(b, c, -1)).reshape(b, c, out_h, out_w)
        return output, jvp
