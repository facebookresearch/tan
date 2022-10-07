#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

'''
Used for ImageNet
'''

from logging import getLogger
import torch
from torchvision import transforms
from torch.utils.data import Subset
import numpy as np
from timm.data import create_transform
from torch.utils.data import Subset
from .folder import DatasetFolder, MultiViewDatasetFolder

logger = getLogger()

DATASETS = {
    'imagenet': {
        'train': '',
        'valid': '',
        'num_classes': 1000,
        'img_size': 256,
        'crop_size': 224,
        'max_batch_size': 32768,
    }
}


def populate_dataset(params):
    assert params.dataset in DATASETS
    DATASETS[params.dataset]['train']=params.train_path
    DATASETS[params.dataset]['valid']=params.val_path
    if params.num_classes == -1:
        params.num_classes = DATASETS[params.dataset]['num_classes']

    if params.img_size is None:
        params.img_size = DATASETS[params.dataset]['img_size']
    if params.crop_size is None:
        params.crop_size = DATASETS[params.dataset]['crop_size']



NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
NORMALIZE_CIFAR = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])


def getImagenetTransform(name, img_size=256, crop_size=224, normalization=True, as_list=False, differentiable=False, params=None):
    transform = []
    if differentiable:
        assert False
    else:
        if name == "random":
            transform = [
                transforms.RandomResizedCrop(crop_size),
                transforms.RandomHorizontalFlip(),
            ]
        elif name == "tencrop":
            transform = [
                transforms.Resize(img_size),
                transforms.TenCrop(crop_size),
            ]
        elif name == "center":
            transform = [
                transforms.Resize(img_size),
                transforms.CenterCrop(crop_size),
            ]
        elif name == 'simclr':
            transform = [
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),], p=0.3),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2),
                transforms.RandomResizedCrop((crop_size, crop_size))
            ]
        elif name == 'Ours':
            transform = [
                transforms.Resize(img_size),
                transforms.CenterCrop(crop_size),
                transforms.RandomCrop(size=crop_size, padding=20, padding_mode="reflect"),
            ]
        elif name == 'Ours_flip':
            transform = [
                transforms.Resize(img_size),
                transforms.CenterCrop(crop_size),
                transforms.RandomCrop(size=crop_size, padding=20, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
            ]
        elif name == 'Ours2':
            transform = [
                transforms.Resize(img_size),
                transforms.CenterCrop(crop_size),
                transforms.RandomCrop(size=crop_size, padding=20, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2),
            ]
        elif name == 'Ours3':
            transform = [
                transforms.Resize(img_size),
                transforms.CenterCrop(crop_size),
                transforms.RandomCrop(size=crop_size, padding=20, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),], p=0.3),
            ]
        elif name == 'Ours3_padding10':
            transform = [
                transforms.Resize(img_size),
                transforms.CenterCrop(crop_size),
                transforms.RandomCrop(size=crop_size, padding=10, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),], p=0.3),
            ]
        elif name == 'Ours3_padding40':
            transform = [
                transforms.Resize(img_size),
                transforms.CenterCrop(crop_size),
                transforms.RandomCrop(size=crop_size, padding=10, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),], p=0.3),
            ]
        elif name == 'OursBest':
            transform = [
                transforms.Resize(img_size),
                transforms.CenterCrop(crop_size),
                transforms.RandomCrop(size=crop_size, padding=20, padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),], p=0.3),
            ]
        elif name == 'Ours4':
            transform = [
                transforms.Resize(img_size),
                transforms.CenterCrop(crop_size),
                transforms.RandomCrop(size=crop_size, padding=20, padding_mode="reflect"),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),], p=0.3),
            ]
        elif name == 'Ours_rotation':
            transform = [
                transforms.Resize(img_size),
                transforms.RandomRotation(180),
                transforms.CenterCrop(crop_size),
                transforms.RandomCrop(size=crop_size, padding=20, padding_mode="reflect"),
                transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),], p=0.3),
            ]
        elif name == 'beit':
            return create_transform(
                input_size=crop_size,
                is_training=True,
                color_jitter=params.color_jitter,
                auto_augment=params.aa,
                interpolation=params.train_interpolation,
                re_prob=params.reprob,
                re_mode=params.remode,
                re_count=params.recount,
                mean=NORMALIZE_IMAGENET.mean,
                std=NORMALIZE_IMAGENET.std,
            )
        else:
            assert name == "none"

    if name == "tencrop":
        postprocess = [
            transforms.Lambda(lambda crops: [transforms.ToTensor()(crop) for crop in crops])
        ]
    else:
        postprocess = [
            transforms.ToTensor()
        ]

    if normalization:
        if name == "tencrop":
            postprocess.append(transforms.Lambda(lambda crops: torch.stack([NORMALIZE_IMAGENET(crop) for crop in crops])))
        else:
            postprocess.append(NORMALIZE_IMAGENET)

    if as_list:
        return transform + postprocess
    else:
        if differentiable:
            return transform, transforms.Compose(postprocess)
        else:
            return transforms.Compose(transform + postprocess)

def build_transform(is_train, args):
    if is_train:
        transform = create_transform(
            input_size=args.crop_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        return transform

    t = []
    size = int((256 / 224) * args.crop_size)
    t.append(
        transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.crop_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(NORMALIZE_IMAGENET.mean, NORMALIZE_IMAGENET.std))
    return transforms.Compose(t)



def get_data_loader(params, split, transform, shuffle):
    """
    Get data loader over imagenet dataset.
    """
    assert params.dataset in DATASETS
    assert shuffle == (split == "train")
    
    if params.num_classes == -1:
        params.num_classes = DATASETS[params.dataset]['num_classes']

    # Transform
    if transform=="deit":
        transform = build_transform(split=="train", params)
    else:
        transform = getImagenetTransform(transform, img_size=params.img_size, crop_size=params.crop_size, normalization=True, params=params)

    # Data
    data = DatasetFolder(root=DATASETS[params.dataset][split], transform=transform)
    l = len(data)
    if params.proportion <1:
        indices = np.random.choice(l,int(params.proportion*l),replace=False)
        data = Subset(data,indices)

    if split == 'train':
        # data loader
        if data is not None:
            data_loader = torch.utils.data.DataLoader(
                data,
                batch_size=params.batch_size,
                shuffle=shuffle,
                num_workers=30,
                pin_memory=True,
            )
        else:
            data_loader = []
    else:
        data_loader = torch.utils.data.DataLoader(
            data,
            batch_size=params.test_batch_size,
            shuffle=False,
            num_workers=10,
            pin_memory=True
        )


    return data_loader

def get_data_loader_augmented(params, split, transforms, shuffle):
    """
    Get data loader over imagenet dataset.
    """
    assert params.dataset in DATASETS
    
    if params.num_classes == -1:
        params.num_classes = DATASETS[params.dataset]['num_classes']

    # Transform

    # Data
    data = MultiViewDatasetFolder(root=DATASETS[params.dataset][split], transform=transforms)
    l =len(data)
    if params.proportion<1:
        indices = np.random.choice(l,int(params.proportion*l), replace=False)
        data = Subset(data,indices)
    # data loader
    if data is not None:
        data_loader = torch.utils.data.DataLoader(
            data,
            batch_size=params.batch_size,
            shuffle=shuffle,
            num_workers=30,
            pin_memory=True,
        )
    else:
        data_loader = []

    return data_loader