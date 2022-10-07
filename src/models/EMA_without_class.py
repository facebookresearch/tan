#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch import nn

from copy import deepcopy
from collections import OrderedDict
from sys import stderr

# for type hint
from torch import Tensor


def create_ema(model):
    ema = deepcopy(model)
    for param in ema.parameters():
        param.detach_()
    return ema


def update(model, ema, t, decay=0.9999,change_ema_decay_end=0):
    t2 = t - change_ema_decay_end if t > change_ema_decay_end else t
    
    effective_decay = min(decay, (1 + t2) / (10 + t2))
    model_params = OrderedDict(model.named_parameters())
    ema_params = OrderedDict(ema.named_parameters())
    # check if both model contains the same set of keys
    assert model_params.keys() == ema_params.keys()

    for name, param in model_params.items():
        # see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
        # ema_variable -= (1 - decay) * (ema_variable - variable)
        ema_params[name].sub_((1.0 - effective_decay) * (ema_params[name] - param))

    model_buffers = OrderedDict(model.named_buffers())
    ema_buffers = OrderedDict(ema.named_buffers())

    # check if both model contains the same set of keys
    assert model_buffers.keys() == ema_buffers.keys()

    for name, buffer in model_buffers.items():
        # buffers are copied
        ema_buffers[name].copy_(buffer)
