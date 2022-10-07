#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, Union

import numpy as np
import torch
import torch.nn as nn
from opacus.utils.tensor_utils import unfold2d, unfold3d, sum_over_all_but_batch_and_last_n
import torch.nn.functional as F
from .NFnet import MyScaledStdConv2d, unsqueeze_and_copy, get_standardized_weight

class AugmentationMultiplicity:
    def __init__(self, K):
        self.K = K

    def augmented_compute_conv_grad_sample(
        self,
        layer: nn.Conv2d,
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
            activations = activations.reshape(
                (
                    -1,
                    self.K,
                )
                + (activations.shape[1:])
            )
            backprops = backprops.reshape(
                (
                    -1,
                    self.K,
                )
                + (backprops.shape[1:])
            )
            grad_sample = torch.einsum("nkoq,nkpq->nop", backprops, activations)
            # rearrange the above tensor and extract diagonals.
            n = activations.shape[0]
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
            ret[layer.bias] = torch.einsum("nkoq->no", backprops)

        return ret

    def augmented_compute_wsconv_grad_sample(self,layer: MyScaledStdConv2d,activations: torch.Tensor,backprops: torch.Tensor,) -> Dict[nn.Parameter, torch.Tensor]:
        ret = self.augmented_compute_conv_grad_sample(layer, activations, backprops)
        batch_size = activations.shape[0] // self.K

        with torch.enable_grad():
            weight_expanded = unsqueeze_and_copy(layer.weight, batch_size)
            gain_expanded = unsqueeze_and_copy(layer.gain, batch_size)

            std_weight = get_standardized_weight(weight = weight_expanded,gain=gain_expanded,eps=layer.eps)
            std_weight.backward(ret[layer.weight])
        ret[layer.weight] = weight_expanded.grad.clone() #erased copy?
        ret[layer.gain] = gain_expanded.grad.clone()

        return ret
    def augmented_compute_expand_grad_sample(self,
        layer,
        activations,
        backprops
    ):
        """
        Computes per sample gradients for expand layers.
        """
        return {layer.weight: backprops.reshape((-1,self.K)+(backprops.shape[1:])).sum(1)}

    def augmented_compute_linear_grad_sample(
        self, layer: nn.Linear, activations: torch.Tensor, backprops: torch.Tensor
    ) -> Dict[nn.Parameter, torch.Tensor]:
        """
        Computes per sample gradients for ``nn.Linear`` layer
        Args:
            layer: Layer
            activations: Activations
            backprops: Backpropagations
        """
        ret = {}
        activations = activations.reshape(
            (
                -1,
                self.K,
            )
            + (activations.shape[1:])
        )
        backprops = backprops.reshape(
            (
                -1,
                self.K,
            )
            + (backprops.shape[1:])
        )
        if layer.weight.requires_grad:
            gs = torch.einsum("n...i,n...j->nij", backprops, activations)
            ret[layer.weight] = gs
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.einsum("n...k->nk", backprops)
        return ret

    def augmented_compute_group_norm_grad_sample(
        self,
        layer: nn.GroupNorm,
        activations: torch.Tensor,
        backprops: torch.Tensor,
    ) -> Dict[nn.Parameter, torch.Tensor]:
        """
        Computes per sample gradients for GroupNorm
        Args:
            layer: Layer
            activations: Activations
            backprops: Backpropagations
        """
        ret = {}
        if layer.weight.requires_grad:
            normalize_activations = F.group_norm(
                activations, layer.num_groups, eps=layer.eps
            )
            normalize_activations = normalize_activations.reshape(
                (
                    -1,
                    self.K,
                )
                + (activations.shape[1:])
            )
            backprops = backprops.reshape(
                (
                    -1,
                    self.K,
                )
                + (backprops.shape[1:])
            )
            ret[layer.weight] = torch.einsum(
                "nki..., nki...->ni", normalize_activations, backprops
            )
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.einsum("nki...->ni", backprops)
        return ret


    def augmented_compute_layer_norm_grad_sample(self,
        layer: nn.LayerNorm,
        activations: torch.Tensor,
        backprops: torch.Tensor,
    ) -> Dict[nn.Parameter, torch.Tensor]:
        """
        Computes per sample gradients for LayerNorm
        Args:
            layer: Layer
            activations: Activations
            backprops: Backpropagations
        """
        ret = {}
        if layer.weight.requires_grad:
            normalize_activations = F.layer_norm(activations, layer.normalized_shape, eps=layer.eps)
            normalize_activations = normalize_activations.reshape((-1,self.K,)+ (activations.shape[1:]))
            backprops = backprops.reshape((-1,self.K,)+ (backprops.shape[1:]))
            ret[layer.weight] = sum_over_all_but_batch_and_last_n(
                normalize_activations
                * backprops,
                layer.weight.dim(),
            )
        if layer.bias.requires_grad:
            ret[layer.bias] = sum_over_all_but_batch_and_last_n(backprops, layer.bias.dim())
        return ret