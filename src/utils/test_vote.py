#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from src.models.wideresnet import WideResNet
import torch.nn as nn
import torch.optim as optim
from opacus import PrivacyEngine
from tqdm import tqdm
import numpy as np
from src.data.dataset import get_data_loader,get_data_loader_augmented, populate_dataset, getImagenetTransform, build_transform
import src.models.NFnet as NFnet
import timm
from src.models.NFnet import MyScaledStdConv2d, MyScaledStdConv2dSame,Expand

import time


# from EMA import EMA
from src.models.augmented_grad_samplers import AugmentationMultiplicity
from math import ceil
from src.utils.utils import (
    bool_flag,
    accuracy,
    accuracy_with_vote,
    print_params,
    reload_checkpoint,
    save_checkpoint,
    state_dict,
)
import copy




def test_vote(model,args):
    """
    Test the model on the testing set and the training set
    """
    device = args.local_rank
    K = args.transform
    args.batch_size = args.batch_size if K == 0 else args.batch_size//K
    print(f"Batch size used:{args.batch_size}")
    model.eval()
    test_top1_acc = []
    transforms = []
    for _ in range(K):
        transforms.append(getImagenetTransform(args.type_of_augmentation, img_size=256, crop_size=224, normalization=True, as_list=False, differentiable=False, params=None))
    test_loader = get_data_loader_augmented(
            args,
            split='valid',
            transforms= transforms,
            shuffle=True
    )
    with torch.no_grad():
        for images, target in tqdm(test_loader):
            if K:
                images = images.view([-1]+list(images.shape[2:]))
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy_with_vote(preds, labels,K)

            test_top1_acc.append(acc)

    test_top1_avg = np.mean(test_top1_acc)
    # print(f"\tTest set:"f"Loss: {np.mean(losses):.6f} "f"Acc: {top1_avg * 100:.6f} ")
    return (test_top1_avg)

def main():  ## for non poisson, divide bs by world size
    args = parse_args()
    data = torch.load(args.checkpoints)
    model = timm.create_model(args.architecture) ## why se?
    conv = model.stem.conv
    replace_conv_start = MyScaledStdConv2d(conv.in_channels, conv.out_channels, kernel_size= conv.kernel_size, stride=conv.stride,padding=conv.padding)
    model.stem.conv = replace_conv_start
    model.cuda()
    ema = copy.deepcopy(model)
    ema.cuda()
    model.load_state_dict(data['model'])
    ema.load_state_dict(data['ema'])
    print(f"voting accuracy of the ema model:{test_vote(ema,args)}")
    print(f"voting accuracy of the non ema model:{test_vote(model,args)}")


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch imagenet DP testing")

    parser.add_argument(
        "--batch_size",
        default=512,
        type=int,
        metavar="B",
        help="batch size",
    )

    parser.add_argument(
        "--proportion",
        default=1,
        type=float,
        help="proportion of the training set to use for training",
    )

    parser.add_argument(
        "--transform",
        default=8,
        type=int,
        metavar="K",
        help="number of transform",
    )

    parser.add_argument(
        "--num_classes",
        default=1000,
        type=int,
        help="number of classes",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        help="which dataset.only imagenet is supported.",
    )

    parser.add_argument(
        "--architecture",
        type=str,
        choices=['nf_resnet50'],
        default='nf_resnet50',
        help="type of nfresetn",
    )

    parser.add_argument(
        "--type_of_augmentation",
        type=str,
        default="pierre3",
        help="type of augmentation",
    )

    parser.add_argument(
        "--checkpoints",
        type=str,
        default="/checkpoints/tomsander/dp_gigantic_scale/iimagenet_20_08_256_Pierre2/expe/checkpoint.pth",
        help="checkpoint of the model to test",
    )

    parser.add_argument("--local_rank", type=int, default=0)

    return parser.parse_args()
if __name__ == "__main__":
    main()
