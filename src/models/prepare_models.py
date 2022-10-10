#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import timm 
import src.models.NFnet as NFnet
from src.models.NFnet import MyScaledStdConv2d, MyScaledStdConv2dSame,Expand
import torch.nn as nn
from opacus.validators import ModuleValidator
import torchvision
from src.data.dataset import get_data_loader,get_data_loader_augmented, populate_dataset, getImagenetTransform, build_transform
import torch.optim as optim
from src.models.augmented_grad_samplers import AugmentationMultiplicity
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import numpy as np
import torch
from torch.utils.data import Subset

def prepare_model(architecture):
    if "nf" in architecture:
        print("using NFnets")
        model = timm.create_model(architecture,False,{"skipinit":True})
        conv = model.stem.conv
        replace_conv_start = MyScaledStdConv2d(conv.in_channels, conv.out_channels, kernel_size= conv.kernel_size, stride=conv.stride,padding=conv.padding)
        nn.init.kaiming_normal_(replace_conv_start.weight, mode='fan_in', nonlinearity='linear')
        if replace_conv_start.bias is not None: nn.init.zeros_(replace_conv_start.bias)
        model.stem.conv = replace_conv_start
    elif architecture == "resnet50":
        print("using resnet50")
        model = torchvision.models.resnet50(pretrained=False)
        if not ModuleValidator.is_valid(model):
            model = ModuleValidator.fix(model)
    else:
        print("only NFresnet and Resnets are implemented")
        raise NotImplementedError
    return model

def prepare_dataloaders(args):
    if args.transform:
        transforms = []
        if "deit" in args.type_of_augmentation:
            for _ in range(args.transform):
                transforms.append(build_transform(True,args))
        else:
            for _ in range(args.transform):
                transforms.append(getImagenetTransform(args.type_of_augmentation, img_size=256, crop_size=224, normalization=True, as_list=False, differentiable=False, params=None))
        train_loader = get_data_loader_augmented(
            args,
            split='train',
            transforms= transforms,
            shuffle=True,
        )
    else:
        train_loader = get_data_loader(
            args,
            split='train',
            transform=args.train_transform,
            shuffle=True
        )

    test_loader = get_data_loader(
        args,
        split='valid',
        transform='center',
        shuffle=False
    )
    return train_loader,test_loader

def prepare_optimizer(weights, args):
    if args.optimizer == "AdamW":
        optimizer = optim.AdamW(weights, args.AdamW_lr,(args.AdamW_beta1,args.AdamW_beta1), args.AdamW_eps)
    elif args.optimizer =="Adam":
        optimizer = optim.Adam(weights, args.AdamW_lr,(args.AdamW_beta1,args.AdamW_beta1), args.AdamW_eps)
    else:
        optimizer = optim.SGD(weights, lr=args.lr, momentum=args.momentum, dampening=args.dampening)
    return optimizer


def prepare_augmult(model, transform):
    if transform:  
        augmentation = AugmentationMultiplicity(transform)
        model.GRAD_SAMPLERS[nn.modules.linear.Linear] = augmentation.augmented_compute_linear_grad_sample
        model.GRAD_SAMPLERS[MyScaledStdConv2d] = augmentation.augmented_compute_wsconv_grad_sample
        model.GRAD_SAMPLERS[Expand] = augmentation.augmented_compute_expand_grad_sample

def prepare_data_cifar(data_root,batch_size,proportion):
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),])
    train_dataset = CIFAR10(root=data_root, train=True, download=True, transform=transform)
    l = len(train_dataset)
    if proportion <1:
        indices = np.random.choice(l,int(proportion*l),replace=False)
        train_dataset = Subset(train_dataset,indices)
    test_dataset = CIFAR10(root=data_root, train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return(train_dataset,train_loader,test_loader)

def prepare_augmult_cifar(model, transform):
    if (transform):  
        augmentation = AugmentationMultiplicity(transform)
        model.GRAD_SAMPLERS[torch.nn.modules.conv.Conv2d] = augmentation.augmented_compute_conv_grad_sample
        model.GRAD_SAMPLERS[torch.nn.modules.linear.Linear] = augmentation.augmented_compute_linear_grad_sample
        model.GRAD_SAMPLERS[nn.GroupNorm] = augmentation.augmented_compute_group_norm_grad_sample
