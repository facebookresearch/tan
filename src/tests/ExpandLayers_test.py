#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.grad_sample import GradSampleModule
from opacus.grad_sample.utils import register_grad_sampler
from torch.testing import assert_allclose
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData
import math

B = 8
import sys
sys.path.append("../../src")
from models.NFnet import Expand
from models.NFnet import MyScaledStdConv2d

class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MyScaledStdConv2d(3, 16, 8, 2, padding=3, bias=False)
        self.conv2 = MyScaledStdConv2d(16, 32, 4,2)
        self.fc1 = nn.Linear(32 * 6 * 6, 32)
        self.fc2 = nn.Linear(32, 10)
        self.expand = Expand(torch.tensor(1.))
        self.gn = nn.GroupNorm(math.gcd(32, 16), 16)

    def forward(self, x):
        # x of shape [B, 3, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = self.gn(x)
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5] # -> [B, 32, 7, 7]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4] # -> [B, 32, 6, 6]
        if self.expand:
            x=x*self.expand(x).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        x = x.view(-1, 32 * 6 * 6)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"

class ExpandLayer(unittest.TestCase):
    def setUp(self):
        self.original_model = SampleConvNet()
        copy_of_original_model = SampleConvNet()
        copy_of_original_model.load_state_dict(
            self.original_model.state_dict(), strict=True
        )

        self.grad_sample_module_copy = GradSampleModule(
            copy_of_original_model, batch_first=True
        )
        self.DATA_SIZE = B
        self.setUp_data()
        self.criterion = nn.L1Loss()

    def setUp_data(self):
        self.ds = FakeData(
            size=self.DATA_SIZE,
            image_size=(3, 28, 28),
            num_classes=10,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
        self.dl = DataLoader(self.ds, batch_size=self.DATA_SIZE)

    def test_gradients(self):
        # with randomized transforms
        images, _ = next(iter(self.dl))

        # self.grad_sample_module_copy.max_batch_len = self.DATA_SIZE
        self.original_model = self.original_model.train()
        self.grad_sample_module_copy = self.grad_sample_module_copy.train()
        gs_out = self.original_model(images)
        gs_out_copy = self.grad_sample_module_copy(images)
        loss1 = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss2 = self.criterion(gs_out_copy, torch.zeros_like(gs_out_copy))

        loss1.backward()  # classic pytorch
        loss2.backward()  # opacus
        params_with_g = [p.grad for p in self.original_model.parameters() if p.grad is not None]
        params_with_gs_copy = [
            p.grad_sample
            for p in self.grad_sample_module_copy.parameters()
            if p.grad_sample is not None
        ]
        assert len(params_with_g) == len(
            params_with_gs_copy
        ), f"original:{len(params_with_g)} vs copy:{len(params_with_gs_copy)}"
        params_with_g_shapes = [p.shape for p in params_with_g]
        params_with_gs_copy_shapes = [p.shape for p in params_with_gs_copy]

        def g(shape1, shape2):
            return (B,) + shape1 == shape2

        check_shape = [
            g(shape1, shape2)
            for (shape1, shape2) in zip(
                params_with_g_shapes, params_with_gs_copy_shapes
            )
        ]
        assert all(check_shape)
        params_with_gs_mean = [p.mean(0) for p in params_with_gs_copy]
        check_mean = [
            torch.allclose(p, q) for (p, q) in zip(params_with_gs_mean, params_with_g)
        ]
        assert all(check_mean), "not good"


if __name__ == "__main__":
    unittest.main()