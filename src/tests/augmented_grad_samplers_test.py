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
from opacus.grad_sample.linear import compute_linear_grad_sample
from opacus.grad_sample.utils import register_grad_sampler
from torch.testing import assert_allclose
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FakeData
from torchvision.models import mobilenet_v3_small

from opacus.validators import ModuleValidator
from opacus import PrivacyEngine
import math

K =16
B = 2

import sys
sys.path.append("../../src")
from models.wideresnet import WideResNet
from models.NFnet import MyScaledStdConv2d, Expand
from models.augmented_grad_samplers import AugmentationMultiplicity
import torch.optim as optim

class SampleConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 8, 2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)
        self.gn = nn.GroupNorm(math.gcd(32, 16), 16)

    def forward(self, x):
        # x of shape [B, 3, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = self.gn(x)
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"

class AugmentationMultiplicityTest(unittest.TestCase):
    def setUp(self):
        self.original_model = ModuleValidator.fix(WideResNet(16, 10, 4))
        self.augmentation = AugmentationMultiplicity(K)
        copy_of_original_model = ModuleValidator.fix(WideResNet(16, 10, 4))
        # model = ModuleValidator.fix(WideResNet(16,10, 4))
        copy_of_original_model.load_state_dict(
            self.original_model.state_dict(), strict=True
        )

        def set_batch_len(module):
            module.max_batch_len = B

        # self.original_model.apply(set_batch_len)
        # copy_of_original_model.apply(set_batch_len)

        self.grad_sample_module_copy = GradSampleModule(
            copy_of_original_model, batch_first=True,K=K
        )

        self.grad_sample_module_copy.GRAD_SAMPLERS[
            torch.nn.modules.conv.Conv2d
        ] = self.augmentation.augmented_compute_conv_grad_sample
        self.grad_sample_module_copy.GRAD_SAMPLERS[
            torch.nn.modules.linear.Linear
        ] = self.augmentation.augmented_compute_linear_grad_sample
        self.grad_sample_module_copy.GRAD_SAMPLERS[
            nn.GroupNorm
        ] = self.augmentation.augmented_compute_group_norm_grad_sample
        self.DATA_SIZE = B
        self.setUp_data()
        self.criterion = nn.L1Loss()

    def setUp_data(self):
        self.ds = FakeData(
            size=self.DATA_SIZE,
            image_size=(3, 32, 32),
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

    def test_augmentation_multiplicity_level2(self):
        print("level2")
        # with randomized transforms
        images, _ = next(iter(self.dl))
        images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=(32, 32), padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        # images_duplicates = transform(images_duplicates) WOULD DO THE SAME TRANSFORM ON EACH IMAGE
        images_duplicates = transforms.Lambda(
            lambda x: torch.stack([transform(x_) for x_ in x])
        )(images_duplicates)

        # self.grad_sample_module_copy.max_batch_len = self.DATA_SIZE
        self.original_model = self.original_model.train()
        self.grad_sample_module_copy = self.grad_sample_module_copy.train()
        gs_out = self.original_model(images_duplicates)
        gs_out_copy = self.grad_sample_module_copy(images_duplicates)
        loss1 = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss2 = self.criterion(gs_out_copy, torch.zeros_like(gs_out_copy))

        loss1.backward()  # classic pytorch
        loss2.backward()  # opacus

        params_with_g = [
            p.grad for p in self.original_model.parameters() if p.grad is not None
        ]
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
            torch.allclose(p, q, atol=1e-07) for (p, q) in zip(params_with_gs_mean, params_with_g)
        ]
        assert all(check_mean), "not good"

    def test_augmentation_multiplicity(self):
        images, _ = next(iter(self.dl))
        images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
        transform = transforms.Compose(
            [transforms.CenterCrop(size=(32, 32)), transforms.RandomHorizontalFlip(p=1)]
        )
        # images_duplicates = transform(images_duplicates) WOULD DO THE SAME TRANSFORM ON EACH IMAGE
        images_duplicates = transforms.Lambda(
            lambda x: torch.stack([transform(x_) for x_ in x])
        )(images_duplicates)
        images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(
            images
        )

        # self.grad_sample_module_copy.max_batch_len = self.DATA_SIZE DOES NOT WORK
        self.original_model = self.original_model.train()
        self.grad_sample_module_copy = self.grad_sample_module_copy.train()
        gs_out = self.original_model(images)
        gs_out_copy = self.grad_sample_module_copy(images_duplicates)
        loss1 = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss2 = self.criterion(gs_out_copy, torch.zeros_like(gs_out_copy))

        loss1.backward()
        loss2.backward()

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
            torch.allclose(p, q, atol=1e-06) for (p, q) in zip(params_with_gs_mean, params_with_g)
        ]
        assert all(check_mean), "not good"

class SampleConvNet2(nn.Module):
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
        return "SampleConvNet2"

class AugmentationMultiplicityTest_expand_WSconv(unittest.TestCase):
    def setUp(self):
        self.original_model = SampleConvNet2()
        self.augmentation = AugmentationMultiplicity(K)
        copy_of_original_model = SampleConvNet2()
        copy_of_original_model.load_state_dict(
            self.original_model.state_dict(), strict=True
        )

        def set_batch_len(module):
            module.max_batch_len = B

        # self.original_model.apply(set_batch_len)
        # copy_of_original_model.apply(set_batch_len)

        self.grad_sample_module_copy = GradSampleModule(
            copy_of_original_model, batch_first=True,K=K
        )

        self.grad_sample_module_copy.GRAD_SAMPLERS[
            MyScaledStdConv2d
        ] = self.augmentation.augmented_compute_wsconv_grad_sample
        self.grad_sample_module_copy.GRAD_SAMPLERS[
            torch.nn.modules.linear.Linear
        ] = self.augmentation.augmented_compute_linear_grad_sample
        self.grad_sample_module_copy.GRAD_SAMPLERS[
            nn.GroupNorm
        ] = self.augmentation.augmented_compute_group_norm_grad_sample
        self.grad_sample_module_copy.GRAD_SAMPLERS[
            Expand
        ] = self.augmentation.augmented_compute_expand_grad_sample
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

    def test_augmentation_multiplicity_level2(self):
        # with randomized transforms
        images, _ = next(iter(self.dl))
        images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=(28, 28), padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        # images_duplicates = transform(images_duplicates) WOULD DO THE SAME TRANSFORM ON EACH IMAGE
        images_duplicates = transforms.Lambda(
            lambda x: torch.stack([transform(x_) for x_ in x])
        )(images_duplicates)

        # self.grad_sample_module_copy.max_batch_len = self.DATA_SIZE
        self.original_model = self.original_model.train()
        self.grad_sample_module_copy = self.grad_sample_module_copy.train()
        gs_out = self.original_model(images_duplicates)
        gs_out_copy = self.grad_sample_module_copy(images_duplicates)
        loss1 = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss2 = self.criterion(gs_out_copy, torch.zeros_like(gs_out_copy))

        loss1.backward()  # classic pytorch
        loss2.backward()  # opacus

        params_with_g = [
            p.grad for p in self.original_model.parameters() if p.grad is not None
        ]
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
            torch.allclose(p, q, atol=1e-07) for (p, q) in zip(params_with_gs_mean, params_with_g)
        ]
        assert all(check_mean), "not good"

    def test_augmentation_multiplicity(self):
        images, _ = next(iter(self.dl))
        images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
        transform = transforms.Compose(
            [transforms.CenterCrop(size=(28, 28)), transforms.RandomHorizontalFlip(p=1)]
        )
        # images_duplicates = transform(images_duplicates) WOULD DO THE SAME TRANSFORM ON EACH IMAGE
        images_duplicates = transforms.Lambda(
            lambda x: torch.stack([transform(x_) for x_ in x])
        )(images_duplicates)
        images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(
            images
        )

        # self.grad_sample_module_copy.max_batch_len = self.DATA_SIZE DOES NOT WORK
        self.original_model = self.original_model.train()
        self.grad_sample_module_copy = self.grad_sample_module_copy.train()
        gs_out = self.original_model(images)
        gs_out_copy = self.grad_sample_module_copy(images_duplicates)
        loss1 = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss2 = self.criterion(gs_out_copy, torch.zeros_like(gs_out_copy))

        loss1.backward()
        loss2.backward()

        params_with_g = [
            p.grad for p in self.original_model.parameters() if p.grad is not None
        ]
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
        # import pdb;pdb.set_trace()
        check_mean = [
            torch.allclose(p, q,atol=1e-7) for (p, q) in zip(params_with_gs_mean, params_with_g)
        ]
        assert all(check_mean), "not good"

    def test_augmentation_multiplicity3(self):
        images, _ = next(iter(self.dl))
        images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=(28, 28), padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        # images_duplicates = transform(images_duplicates) WOULD DO THE SAME TRANSFORM ON EACH IMAGE
        images_duplicates = transforms.Lambda(
            lambda x: torch.stack([transform(x_) for x_ in x])
        )(images_duplicates)
        images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(
            images
        )

        # self.grad_sample_module_copy.max_batch_len = self.DATA_SIZE DOES NOT WORK
        self.original_model = self.original_model.train()
        self.grad_sample_module_copy = self.grad_sample_module_copy.train()
        gs_out = self.original_model(images_duplicates[:K])
        gs_out_copy = self.grad_sample_module_copy(images_duplicates)
        loss1 = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss2 = self.criterion(gs_out_copy, torch.zeros_like(gs_out_copy))

        loss1.backward()
        loss2.backward()

        params_with_g = [
            p.grad for p in self.original_model.parameters() if p.grad is not None
        ]
        params_with_gs_copy = [
            p.grad_sample
            for p in self.grad_sample_module_copy.parameters()
            if p.grad_sample is not None
        ]
        assert len(params_with_g) == len(
            params_with_gs_copy
        ), f"original:{len(params_with_g)} vs copy:{len(params_with_gs_copy)}"
        params_with_gs_first = [p[0] for p in params_with_gs_copy]

        check_mean = [
            torch.allclose(p, q,atol=1e-7) for (p, q) in zip(params_with_gs_first, params_with_g)
        ]

        assert all(check_mean), "not good"

    def test_augmentation_multiplicity4(self):
        images, _ = next(iter(self.dl))
        images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=(28, 28), padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        # images_duplicates = transform(images_duplicates) WOULD DO THE SAME TRANSFORM ON EACH IMAGE
        images_duplicates = transforms.Lambda(
            lambda x: torch.stack([transform(x_) for x_ in x])
        )(images_duplicates)
        optimizer1 = optim.SGD(self.grad_sample_module_copy.parameters(), lr=1)
        optimizer2 = optim.SGD(self.original_model.parameters(), lr=1)
        privacy_engine = PrivacyEngine()
        self.grad_sample_module_copy, optimizer1, train_loader = privacy_engine.make_private(
            module=self.grad_sample_module_copy,
            optimizer=optimizer1,
            data_loader=self.dl,
            noise_multiplier=0,
            max_grad_norm=100000,
            poisson_sampling=True,
            K=K
        )
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        # self.grad_sample_module_copy.max_batch_len = self.DATA_SIZE DOES NOT WORK
        self.original_model = self.original_model.train()
        self.grad_sample_module_copy = self.grad_sample_module_copy.train()
        gs_out = self.original_model(images_duplicates)
        gs_out_copy = self.grad_sample_module_copy(images_duplicates)
        loss1 = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss2 = self.criterion(gs_out_copy, torch.zeros_like(gs_out_copy))

        loss1.backward()
        loss2.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        images, _ = next(iter(self.dl))
        images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
        # images_duplicates = transform(images_duplicates) WOULD DO THE SAME TRANSFORM ON EACH IMAGE
        images_duplicates = transforms.Lambda(
            lambda x: torch.stack([transform(x_) for x_ in x])
        )(images_duplicates)
        images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(
            images
        )

        # self.grad_sample_module_copy.max_batch_len = self.DATA_SIZE DOES NOT WORK
        self.original_model = self.original_model.train()
        self.grad_sample_module_copy = self.grad_sample_module_copy.train()
        gs_out = self.original_model(images_duplicates[:K])
        gs_out_copy = self.grad_sample_module_copy(images_duplicates)
        loss1 = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss2 = self.criterion(gs_out_copy, torch.zeros_like(gs_out_copy))

        loss1.backward()
        loss2.backward()

        params_with_g = [
            p.grad for p in self.original_model.parameters() if p.grad is not None
        ]
        params_with_gs_copy = [
            p.grad_sample
            for p in self.grad_sample_module_copy.parameters()
            if p.grad_sample is not None
        ]
        assert len(params_with_g) == len(
            params_with_gs_copy
        ), f"original:{len(params_with_g)} vs copy:{len(params_with_gs_copy)}"
        params_with_gs_first = [p[0] for p in params_with_gs_copy]

        check_mean = [
            torch.allclose(p, q,atol=1e-6) for (p, q) in zip(params_with_gs_first, params_with_g)
        ]

        assert all(check_mean), "not good"

class SampleConvNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 8, 2, padding=3, bias=False)
        self.conv2 = nn.Conv2d(16, 32, 4, 2)
        self.fc1 = nn.Linear(32 * 4 * 4, 32)
        self.fc2 = nn.Linear(32, 10)
        self.ln = nn.LayerNorm((13,13))

    def forward(self, x):
        # x of shape [B, 3, 28, 28]
        x = F.relu(self.conv1(x))  # -> [B, 16, 14, 14]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 16, 13, 13]
        x = self.ln(x)
        x = F.relu(self.conv2(x))  # -> [B, 32, 5, 5]
        x = F.max_pool2d(x, 2, 1)  # -> [B, 32, 4, 4]
        x = x.view(-1, 32 * 4 * 4)  # -> [B, 512]
        x = F.relu(self.fc1(x))  # -> [B, 32]
        x = self.fc2(x)  # -> [B, 10]
        return x

    def name(self):
        return "SampleConvNet"

class AugmentationMultiplicityTest_Layer_norm(unittest.TestCase):
    def setUp(self):
        self.original_model = SampleConvNet3()
        self.augmentation = AugmentationMultiplicity(K)
        copy_of_original_model = SampleConvNet3()
        copy_of_original_model.load_state_dict(
            self.original_model.state_dict(), strict=True
        )

        self.grad_sample_module_copy = GradSampleModule(
            copy_of_original_model, batch_first=True,K=K
        )

        self.grad_sample_module_copy.GRAD_SAMPLERS[
            torch.nn.modules.conv.Conv2d
        ] = self.augmentation.augmented_compute_conv_grad_sample
        self.grad_sample_module_copy.GRAD_SAMPLERS[
            torch.nn.modules.linear.Linear
        ] = self.augmentation.augmented_compute_linear_grad_sample
        self.grad_sample_module_copy.GRAD_SAMPLERS[
            nn.LayerNorm
        ] = self.augmentation.augmented_compute_layer_norm_grad_sample
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

    def test_augmentation_multiplicity_level2(self):
        # with randomized transforms
        images, _ = next(iter(self.dl))
        images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=(28, 28), padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        # images_duplicates = transform(images_duplicates) WOULD DO THE SAME TRANSFORM ON EACH IMAGE
        images_duplicates = transforms.Lambda(
            lambda x: torch.stack([transform(x_) for x_ in x])
        )(images_duplicates)

        # self.grad_sample_module_copy.max_batch_len = self.DATA_SIZE
        self.original_model = self.original_model.train()
        self.grad_sample_module_copy = self.grad_sample_module_copy.train()
        gs_out = self.original_model(images_duplicates)
        gs_out_copy = self.grad_sample_module_copy(images_duplicates)
        loss1 = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss2 = self.criterion(gs_out_copy, torch.zeros_like(gs_out_copy))

        loss1.backward()  # classic pytorch
        loss2.backward()  # opacus

        params_with_g = [
            p.grad for p in self.original_model.parameters() if p.grad is not None
        ]
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
            torch.allclose(p, q, atol=1e-07) for (p, q) in zip(params_with_gs_mean, params_with_g)
        ]
        assert all(check_mean), "not good"

    def test_augmentation_multiplicity(self):
        images, _ = next(iter(self.dl))
        images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
        transform = transforms.Compose(
            [transforms.CenterCrop(size=(28, 28)), transforms.RandomHorizontalFlip(p=1)]
        )
        # images_duplicates = transform(images_duplicates) WOULD DO THE SAME TRANSFORM ON EACH IMAGE
        images_duplicates = transforms.Lambda(
            lambda x: torch.stack([transform(x_) for x_ in x])
        )(images_duplicates)
        images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(
            images
        )

        # self.grad_sample_module_copy.max_batch_len = self.DATA_SIZE DOES NOT WORK
        self.original_model = self.original_model.train()
        self.grad_sample_module_copy = self.grad_sample_module_copy.train()
        gs_out = self.original_model(images)
        gs_out_copy = self.grad_sample_module_copy(images_duplicates)
        loss1 = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss2 = self.criterion(gs_out_copy, torch.zeros_like(gs_out_copy))

        loss1.backward()
        loss2.backward()

        params_with_g = [
            p.grad for p in self.original_model.parameters() if p.grad is not None
        ]
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
        # import pdb;pdb.set_trace()
        check_mean = [
            torch.allclose(p, q,atol=1e-6) for (p, q) in zip(params_with_gs_mean, params_with_g)
        ]
        assert all(check_mean), "not good"

    def test_augmentation_multiplicity3(self):
        images, _ = next(iter(self.dl))
        images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=(28, 28), padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        # images_duplicates = transform(images_duplicates) WOULD DO THE SAME TRANSFORM ON EACH IMAGE
        images_duplicates = transforms.Lambda(
            lambda x: torch.stack([transform(x_) for x_ in x])
        )(images_duplicates)
        images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(
            images
        )

        # self.grad_sample_module_copy.max_batch_len = self.DATA_SIZE DOES NOT WORK
        self.original_model = self.original_model.train()
        self.grad_sample_module_copy = self.grad_sample_module_copy.train()
        gs_out = self.original_model(images_duplicates[:K])
        gs_out_copy = self.grad_sample_module_copy(images_duplicates)
        loss1 = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss2 = self.criterion(gs_out_copy, torch.zeros_like(gs_out_copy))

        loss1.backward()
        loss2.backward()

        params_with_g = [
            p.grad for p in self.original_model.parameters() if p.grad is not None
        ]
        params_with_gs_copy = [
            p.grad_sample
            for p in self.grad_sample_module_copy.parameters()
            if p.grad_sample is not None
        ]
        assert len(params_with_g) == len(
            params_with_gs_copy
        ), f"original:{len(params_with_g)} vs copy:{len(params_with_gs_copy)}"
        params_with_gs_first = [p[0] for p in params_with_gs_copy]

        check_mean = [
            torch.allclose(p, q,atol=1e-6) for (p, q) in zip(params_with_gs_first, params_with_g)
        ]

        assert all(check_mean), "not good"

    def test_augmentation_multiplicity4(self):
        images, _ = next(iter(self.dl))
        images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
        transform = transforms.Compose(
            [
                transforms.RandomCrop(size=(28, 28), padding=4, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
        # images_duplicates = transform(images_duplicates) WOULD DO THE SAME TRANSFORM ON EACH IMAGE
        images_duplicates = transforms.Lambda(
            lambda x: torch.stack([transform(x_) for x_ in x])
        )(images_duplicates)
        optimizer1 = optim.SGD(self.grad_sample_module_copy.parameters(), lr=1)
        optimizer2 = optim.SGD(self.original_model.parameters(), lr=1)
        privacy_engine = PrivacyEngine()
        self.grad_sample_module_copy, optimizer1, train_loader = privacy_engine.make_private(
            module=self.grad_sample_module_copy,
            optimizer=optimizer1,
            data_loader=self.dl,
            noise_multiplier=0,
            max_grad_norm=100000,
            poisson_sampling=True,
            K=K
        )
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        # self.grad_sample_module_copy.max_batch_len = self.DATA_SIZE DOES NOT WORK
        self.original_model = self.original_model.train()
        self.grad_sample_module_copy = self.grad_sample_module_copy.train()
        gs_out = self.original_model(images_duplicates)
        gs_out_copy = self.grad_sample_module_copy(images_duplicates)
        loss1 = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss2 = self.criterion(gs_out_copy, torch.zeros_like(gs_out_copy))

        loss1.backward()
        loss2.backward()
        optimizer1.step()
        optimizer2.step()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
        
        images, _ = next(iter(self.dl))
        images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
        # images_duplicates = transform(images_duplicates) WOULD DO THE SAME TRANSFORM ON EACH IMAGE
        images_duplicates = transforms.Lambda(
            lambda x: torch.stack([transform(x_) for x_ in x])
        )(images_duplicates)
        images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(
            images
        )

        # self.grad_sample_module_copy.max_batch_len = self.DATA_SIZE DOES NOT WORK
        self.original_model = self.original_model.train()
        self.grad_sample_module_copy = self.grad_sample_module_copy.train()
        gs_out = self.original_model(images_duplicates[:K])
        gs_out_copy = self.grad_sample_module_copy(images_duplicates)
        loss1 = self.criterion(gs_out, torch.zeros_like(gs_out))
        loss2 = self.criterion(gs_out_copy, torch.zeros_like(gs_out_copy))

        loss1.backward()
        loss2.backward()

        params_with_g = [
            p.grad for p in self.original_model.parameters() if p.grad is not None
        ]
        params_with_gs_copy = [
            p.grad_sample
            for p in self.grad_sample_module_copy.parameters()
            if p.grad_sample is not None
        ]
        assert len(params_with_g) == len(
            params_with_gs_copy
        ), f"original:{len(params_with_g)} vs copy:{len(params_with_gs_copy)}"
        params_with_gs_first = [p[0] for p in params_with_gs_copy]

        check_mean = [
            torch.allclose(p, q,atol=1e-6) for (p, q) in zip(params_with_gs_first, params_with_g)
        ]

        assert all(check_mean), "not good"
if __name__ == "__main__":
    unittest.main()