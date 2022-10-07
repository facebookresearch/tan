#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


'''
Adapted from timm
'''

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn

from opacus.grad_sample import register_grad_sampler
import timm
from timm.models.fx_features import register_notrace_module
from timm.models.layers import make_divisible, DropPath
from timm.models.nfnet import DownsampleAvg
from timm.models.layers.padding import get_padding,  get_padding_value, pad_same
import torch.nn.functional as F


def unsqueeze_and_copy(tensor, batch_size):
    expand_size = [batch_size] + [-1] * tensor.ndim
    tensor_copy = torch.tensor(tensor.expand(expand_size), dtype=tensor.dtype, device=tensor.device, requires_grad=True)

    return tensor_copy

# def get_standardized_weight(weight,out_channels,eps,gain, scale):
#     weight = F.batch_norm(
#             weight.reshape(1, out_channels, -1), None, None,
#             weight=(gain * scale).view(-1),
#             training=True, momentum=0., eps=eps).reshape_as(weight)
#     return weight

import numpy as np
def get_standardized_weight(weight, gain=None, eps=1e-4):
    # Get Scaled WS weight OIHW;
    fan_in = np.prod(weight.shape[-3:])
    mean = torch.mean(weight, axis=[-3, -2, -1], keepdims=True)
    var = torch.var(weight, axis=[-3, -2, -1], keepdims=True)
    weight = (weight - mean) / (var * fan_in + eps) ** 0.5
    if gain is not None:
        weight = weight * gain
    return weight


class MyScaledStdConv2d(nn.Conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding=None,
            dilation=1, groups=1, bias=True, gamma=1.0, eps=1e-6, gain_init=1.0):
        if padding is None:
            padding = get_padding(kernel_size, stride, dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.eps = eps
        

    # def forward(self, x):
    #     weight = get_standardized_weight(self.weight, self.out_channels, self.eps,self.gain,self.scale)
    #     return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, x):
        std_weight = get_standardized_weight(self.weight, gain=self.gain, eps=self.eps)
        return F.conv2d(x, std_weight, self.bias,self.stride, self.padding, self.dilation, self.groups)

class MyScaledStdConv2dSame(nn.Conv2d):
    """Conv2d with Weight Standardization. Used for BiT ResNet-V2 models.

    Paper: `Micro-Batch Training with Batch-Channel Normalization and Weight Standardization` -
        https://arxiv.org/abs/1903.10520v2
    """
    def __init__(
            self, in_channels, out_channels, kernel_size, stride=1, padding='SAME',
            dilation=1, groups=1, bias=True, gamma=1.0, eps=1e-6, gain_init=1.0):
        padding, is_dynamic = get_padding_value(padding, kernel_size, stride=stride, dilation=dilation)
        super().__init__(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias)
        self.gain = nn.Parameter(torch.full((self.out_channels, 1, 1, 1), gain_init))
        self.scale = gamma * self.weight[0].numel() ** -0.5  # gamma * 1 / sqrt(fan-in)
        self.same_pad = is_dynamic
        self.eps = eps
        

    # def forward(self, x):
    #     weight = get_standardized_weight(self.weight, self.out_channels, self.eps,self.gain,self.scale)
    #     return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, x):
        std_weight = get_standardized_weight(self.weight, gain=self.gain, eps=self.eps)
        if self.same_pad:
            x = pad_same(x, self.kernel_size, self.stride, self.dilation)
        return F.conv2d(x, std_weight, self.bias,self.stride, self.padding, self.dilation, self.groups)

from opacus.utils.tensor_utils import unfold2d, unfold3d

def compute_conv_grad_sample(
    layer: nn.Conv1d,
    activations: torch.Tensor,
    backprops: torch.Tensor,
) -> Dict[nn.Parameter, torch.Tensor]:
    """
    Computes per sample gradients for convolutional layers

    Args:
        layer: Layer
        activations: Activations
        backprops: Backpropagations
    """
    n = activations.shape[0]

    # get activations and backprops in shape depending on the Conv layer
    activations = unfold2d(
        activations,
        kernel_size=layer.kernel_size,
        padding=layer.padding,
        stride=layer.stride,
        dilation=layer.dilation,
    )
    backprops = backprops.reshape(n, -1, activations.shape[-1])

    ret = {}
    if layer.weight.requires_grad:
        # n=batch_sz; o=num_out_channels; p=(num_in_channels/groups)*kernel_sz
        grad_sample = torch.einsum("noq,npq->nop", backprops, activations)
        # rearrange the above tensor and extract diagonals.
        grad_sample = grad_sample.view(
            n,
            layer.groups,
            -1,
            layer.groups,
            int(layer.in_channels / layer.groups),
            np.prod(layer.kernel_size),
        )
        grad_sample = torch.einsum("ngrg...->ngr...", grad_sample).contiguous()
        shape = [n] + list(layer.weight.shape)
        ret[layer.weight] = grad_sample.view(shape)

    if layer.bias is not None and layer.bias.requires_grad:
        ret[layer.bias] = torch.sum(backprops, dim=2)

    return ret

@register_grad_sampler([MyScaledStdConv2d])
def compute_wsconv_grad_sample(layer: MyScaledStdConv2d,activations: torch.Tensor,backprops: torch.Tensor,) -> Dict[nn.Parameter, torch.Tensor]:
    ret = compute_conv_grad_sample(layer, activations, backprops)
    batch_size = activations.shape[0]

    with torch.enable_grad():
        weight_expanded = unsqueeze_and_copy(layer.weight, batch_size)
        gain_expanded = unsqueeze_and_copy(layer.gain, batch_size)

        std_weight = get_standardized_weight(weight = weight_expanded,gain=gain_expanded,eps=layer.eps)
        std_weight.backward(ret[layer.weight])
    ret[layer.weight] = weight_expanded.grad.clone() #erased copy?
    ret[layer.gain] = gain_expanded.grad.clone()

    return ret

@register_grad_sampler([MyScaledStdConv2dSame])
def compute_wsconv_grad_sample(layer: MyScaledStdConv2dSame,activations: torch.Tensor,backprops: torch.Tensor,) -> Dict[nn.Parameter, torch.Tensor]:
    ret = compute_conv_grad_sample(layer, activations, backprops)
    batch_size = activations.shape[0]

    with torch.enable_grad():
        weight_expanded = unsqueeze_and_copy(layer.weight, batch_size)
        gain_expanded = unsqueeze_and_copy(layer.gain, batch_size)

        std_weight = get_standardized_weight(weight = weight_expanded,gain=gain_expanded,eps=layer.eps)
        std_weight.backward(ret[layer.weight])
    ret[layer.weight] = weight_expanded.grad.clone() #erased copy?
    ret[layer.gain] = gain_expanded.grad.clone()

    return ret

timm.models.layers.std_conv.ScaledStdConv2d = MyScaledStdConv2d


class Expand(nn.Module):

    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(tensor)

    def forward(self, x):
        expand_size = [x.size(0)] + [-1] * self.weight.ndim
        return self.weight.unsqueeze(0).expand(expand_size)


@register_grad_sampler([Expand])
def compute_expand_grad_sample(
    layer,
    activations,
    backprops
):
    """
    Computes per sample gradients for expand layers.
    """
    return {layer.weight: backprops}

@register_notrace_module  # reason: mul_ causes FX to drop a relevant node. https://github.com/pytorch/pytorch/issues/68301
class MyNormFreeBlock(nn.Module):
    """Normalization-Free pre-activation block.
    """
    
    def __init__(
            self, in_chs, out_chs=None, stride=1, dilation=1, first_dilation=None,
            alpha=0.2, beta=1.0, bottle_ratio=0.25, group_size=None, ch_div=1, reg=True, extra_conv=False,
            skipinit=True, attn_layer=None, attn_gain=2.0, act_layer=None, conv_layer=None, drop_path_rate=0.):
        conv_layer = MyScaledStdConv2d
        skipinit=True
        super().__init__()
        first_dilation = first_dilation or dilation
        out_chs = out_chs or in_chs
        # RegNet variants scale bottleneck from in_chs, otherwise scale from out_chs like ResNet
        mid_chs = make_divisible(in_chs * bottle_ratio if reg else out_chs * bottle_ratio, ch_div)
        groups = 1 if not group_size else mid_chs // group_size
        if group_size and group_size % ch_div == 0:
            mid_chs = group_size * groups  # correct mid_chs if group_size divisible by ch_div, otherwise error
        self.alpha = alpha
        self.beta = beta
        self.attn_gain = attn_gain

        if in_chs != out_chs or stride != 1 or dilation != first_dilation:
            self.downsample = DownsampleAvg(
                in_chs, out_chs, stride=stride, dilation=dilation, first_dilation=first_dilation, conv_layer=conv_layer)
        else:
            self.downsample = None

        self.act1 = act_layer()
        self.conv1 = conv_layer(in_chs, mid_chs, 1)
        self.act2 = act_layer(inplace=True)
        self.conv2 = conv_layer(mid_chs, mid_chs, 3, stride=stride, dilation=first_dilation, groups=groups)
        if extra_conv:
            self.act2b = act_layer(inplace=True)
            self.conv2b = conv_layer(mid_chs, mid_chs, 3, stride=1, dilation=dilation, groups=groups)
        else:
            self.act2b = None
            self.conv2b = None
        if reg and attn_layer is not None:
            self.attn = attn_layer(mid_chs)  # RegNet blocks apply attn btw conv2 & 3
        else:
            self.attn = None
        self.act3 = act_layer()
        self.conv3 = conv_layer(mid_chs, out_chs, 1, gain_init=1. if skipinit else 0.)
        if not reg and attn_layer is not None:
            self.attn_last = attn_layer(out_chs)  # ResNet blocks apply attn after conv3
        else:
            self.attn_last = None
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        #self.skipinit_gain = nn.Parameter(torch.tensor(0.)) if skipinit else None
        self.expand = Expand(torch.tensor(1.)) if skipinit else None


    def forward(self, x):
        out = self.act1(x) * self.beta

        # shortcut branch
        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(out)

        # residual branch
        out = self.conv1(out)

        out = self.conv2(self.act2(out))

        if self.conv2b is not None:
            out = self.conv2b(self.act2b(out))
        if self.attn is not None:
            out = self.attn_gain * self.attn(out)
        out = self.conv3(self.act3(out))
        if self.attn_last is not None:
            out = self.attn_gain * self.attn_last(out)
        out = self.drop_path(out)

        # if self.skipinit_gain is not None:
        #     out.mul_(self.skipinit_gain)  # this slows things down more than expected, TBD
        if self.expand is not None:
            out=out*self.expand(out).unsqueeze(1).unsqueeze(2).unsqueeze(3)#carefull

        out = out * self.alpha + shortcut
        return out

timm.models.nfnet.NormFreeBlock = MyNormFreeBlock





