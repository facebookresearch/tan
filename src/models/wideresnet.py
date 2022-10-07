#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""
Adapted from timm:
https://github.com/xternalz/WideResNet-pytorch/blob/master/wideresnet.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, nb_groups, order):
        super(BasicBlock, self).__init__()
        self.order = order
        self.bn1 = nn.GroupNorm(nb_groups, in_planes) if nb_groups else nn.Identity()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=1
        )
        self.bn2 = nn.GroupNorm(nb_groups, out_planes) if nb_groups else nn.Identity()
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(
            out_planes, out_planes, kernel_size=3, stride=1, padding=1
        )

        self.equalInOut = in_planes == out_planes
        self.bnShortcut = (
            (not self.equalInOut)
            and nb_groups
            and nn.GroupNorm(nb_groups, in_planes)
            or (not self.equalInOut)
            and nn.Identity()
            or None
        )
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, padding=0
            )
        ) or None

    def forward(self, x):
        skip = x
        assert self.order in [0, 1, 2, 3]
        if self.order == 0:  # DM accuracy good
            if not self.equalInOut:
                skip = self.convShortcut(self.bnShortcut(self.relu1(x)))
            out = self.conv1(self.bn1(self.relu1(x)))
            out = self.conv2(self.bn2(self.relu2(out)))
        elif self.order == 1:  # classic accuracy bad
            if not self.equalInOut:
                skip = self.convShortcut(self.relu1(self.bnShortcut(x)))
            out = self.conv1(self.relu1(self.bn1(x)))
            out = self.conv2(self.relu2(self.bn2(out)))
        elif self.order == 2:  # DM IN RESIDUAL, normal other
            if not self.equalInOut:
                skip = self.convShortcut(self.bnShortcut(self.relu1(x)))
            out = self.conv1(self.relu1(self.bn1(x)))
            out = self.conv2(self.relu2(self.bn2(out)))
        elif self.order == 3:  # normal in residualm DM in others
            if not self.equalInOut:
                skip = self.convShortcut(self.relu1(self.bnShortcut(x)))
            out = self.conv1(self.bn1(self.relu1(x)))
            out = self.conv2(self.bn2(self.relu2(out)))
        return torch.add(skip, out)


class NetworkBlock(nn.Module):
    def __init__(
        self, nb_layers, in_planes, out_planes, block, stride, nb_groups, order
    ):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, nb_groups, order
        )

    def _make_layer(
        self, block, in_planes, out_planes, nb_layers, stride, nb_groups, order
    ):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(
                block(
                    i == 0 and in_planes or out_planes,
                    out_planes,
                    i == 0 and stride or 1,
                    nb_groups,
                    order,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        depth,
        num_classes,
        widen_factor=1,
        nb_groups=16,
        init=0,
        order1=0,
        order2=0,
    ):
        if order1 == 0:
            print("order1=0: In the blocks: like in DM, BN on top of relu")
        if order1 == 1:
            print("order1=1: In the blocks: not like in DM, relu on top of BN")
        if order1 == 2:
            print(
                "order1=2: In the blocks: BN on top of relu in residual (DM), relu on top of BN ortherplace (clqssique)"
            )
        if order1 == 3:
            print(
                "order1=3: In the blocks:  relu on top of BN  in residual (classic), BN on top of relu otherplace (DM)"
            )
        if order2 == 0:
            print("order2=0: outside the blocks:  like in DM, BN on top of relu")
        if order2 == 1:
            print("order2=1: outside the blocks:  not like in DM, relu on top of BN")
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1)
        # 1st block
        self.block1 = NetworkBlock(
            n, nChannels[0], nChannels[1], block, 1, nb_groups, order1
        )
        # 2nd block
        self.block2 = NetworkBlock(
            n, nChannels[1], nChannels[2], block, 2, nb_groups, order1
        )
        # 3rd block
        self.block3 = NetworkBlock(
            n, nChannels[2], nChannels[3], block, 2, nb_groups, order1
        )
        # global average pooling and classifier
        self.bn1 = nn.GroupNorm(nb_groups, nChannels[3]) if nb_groups else nn.Identity()
        self.relu = nn.ReLU()
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        if init == 0:  # as in Deep Mind's paper
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    s = 1 / (max(fan_in, 1)) ** 0.5
                    nn.init.trunc_normal_(m.weight, std=s)
                    m.bias.data.zero_()
                elif isinstance(m, nn.GroupNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    s = 1 / (max(fan_in, 1)) ** 0.5
                    nn.init.trunc_normal_(m.weight, std=s)
                    m.bias.data.zero_()
        if init == 1:  # old version
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif isinstance(m, nn.GroupNorm):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.bias.data.zero_()
        self.order2 = order2

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(self.relu(out)) if self.order2 == 0 else self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
