#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

'''
Training a NF-ResNet or a ResNet on ImageNet with DP-SGD
'''


import argparse
from math import ceil
from selectors import EpollSelector
import torch
import torchvision
import torchvision.transforms as transforms
from src.models.wideresnet import WideResNet
import torch.nn as nn
from torchvision.datasets import CIFAR10
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from src.data.dataset import get_data_loader,get_data_loader_augmented, populate_dataset, getImagenetTransform, build_transform
from src.models.prepare_models import  prepare_model, prepare_dataloaders, prepare_optimizer, prepare_augmult
import src.models.NFnet as NFnet
import time


# from EMA import EMA
from src.models.EMA_without_class import create_ema, update
from src.models.augmented_grad_samplers import AugmentationMultiplicity
from math import ceil
from src.utils.utils import (
    init_distributed_mode,
    initialize_exp,
    bool_flag,
    accuracy,
    get_noise_from_bs,
    get_epochs_from_bs,
    print_params,
    reload_checkpoint,
    save_checkpoint,
)

from src.utils.test_vote import test_vote

import json

from torch.nn.parallel import DistributedDataParallel as DDP
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from src.opacus_augmented.privacy_engine_augmented import PrivacyEngineAugmented
import warnings
from opacus import GradSampleModule
warnings.simplefilter("ignore")

def train(
    model,
    ema,
    train_loader,
    optimizer,
    epoch,
    max_nb_steps,
    device,
    K,
    logger,
    losses,
    train_acc,
    test_loader,
    is_main_worker,
    args,
    nb_steps,
    per_param_norms_epoch
):
    """
    Trains the model for one epoch. If it is the last epoch, it will stop at max_nb_steps iterations.
    If the model is being shadowed for EMA, we update the model at every step.
    """

    model.train()
    criterion = nn.CrossEntropyLoss()

    ######### VARIABLES FOR LOGGER ###########
    steps_per_epoch = len(train_loader)
    if is_main_worker:
        print(f"steps_per_epoch:{steps_per_epoch}")
    losses_epoch, train_acc_epoch, grad_sample_norms = [], [], []
    max_physical_batch_size_with_augnentation = args.max_physical_batch_size if K == 0 else args.max_physical_batch_size // K
    nb_examples_epoch = 0
    data_loading_times, forward_times, backward_times, optimizer_times = [], [], [], []
    optimizer_time = time.time()
    ##########################################

    #Physicial batch size for gradient accumulation to simulate mega batch size training
    with BatchMemoryManager(data_loader=train_loader,max_physical_batch_size=max_physical_batch_size_with_augnentation,optimizer=optimizer,) as memory_safe_data_loader: 
        for (images, target) in memory_safe_data_loader:

            #### VARIABLES FOR LOGGER ###
            start = time.time()
            data_loading_times.append(start-optimizer_time)
            nb_examples_epoch+=len(images)
            #############################
            optimizer.zero_grad(set_to_none=True) 
            # Duplicating labels for Augmentation Multiplicity. Inputs are reshaped from (N,K,*) to (N*K,*) s.t. augmentations of each image are neighbors
            if K:
                images = images.view([-1]+list(images.shape[2:]))
                target = torch.repeat_interleave(target, repeats=K, dim=0)
            ###  Forward ###
            images, target = images.to(device), target.to(device)
            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)
            losses_epoch.append(loss.item())
            train_acc_epoch.append(acc)
            forward_time = time.time()
            forward_times.append(forward_time -start)
            #############
            ### Backward ###
            loss.backward()
            backward_time = time.time()
            backward_times.append(backward_time -forward_time)
            is_updated = not (optimizer._check_skip_next_step(pop_next=False))  # check if we are at the end of a true batch without incrementing the count.
            ############

            #### LOGGING STATISTICS ABOUT THE GRADIENTS ###
            if is_main_worker:
                if nb_steps==0 and is_updated:logger.info("__log:"+ json.dumps({"nb_params": [torch.numel(g) for g in model.parameters() if g.grad_sample is not None]}))
                per_param_norms = torch.stack([g.grad_sample.view(len(g.grad_sample), -1).norm(2, dim=-1).detach().cpu() for g in model.parameters() if g.grad_sample is not None])
                if per_param_norms_epoch is None:
                    per_param_norms_epoch = per_param_norms
                else:
                    per_param_norms_epoch = torch.hstack([per_param_norms_epoch,per_param_norms])
                per_sample_norms = per_param_norms.norm(2, dim=0).tolist()
                l = len(images) // K if K else len(images)
                grad_sample_norms += per_sample_norms[:l]  # in case of poisson sampling we dont want the 0s
            ################################################
            ### GRADIENT STEP ###
            optimizer.step() #True step is performed if the whole batch was explored, gradients are accumulted otherwise
            optimizer_time = time.time()
            optimizer_times.append(optimizer_time-backward_time)
            #####################
            ### UPDATES  ANF LOGS ONLY IF THE WHOLE BATCH WAS EXPLORED ###
            if is_updated:
                nb_steps += 1  
                if nb_steps==1:
                    print(f"nb_examples_before_first_step:{nb_examples_epoch}")
                update(model, ema, nb_steps, decay=args.model_ema_decay) ## Updating the EMA nodel
                if is_main_worker:
                    losses.append(np.mean(losses_epoch))
                    train_acc.append(np.mean(train_acc_epoch))
                    losses_epoch, train_acc_epoch= [],[]
                    if nb_steps % args.freq_log == 0:
                        print(f"epoch:{epoch},step:{nb_steps}")
                        logger.info(
                            "__log:"
                            + json.dumps(
                                {   "step":nb_steps,
                                    "train_acc": np.mean(train_acc[-args.freq_log :]),
                                    "loss": np.mean(losses[-args.freq_log :]),
                                    "current_noise":optimizer.noise_multiplier,
                                    "current_lr":optimizer.param_groups[0]['lr'],
                                    "current_momentum":optimizer.param_groups[0]['momentum'],
                                    "nb_examples_since_last_update_main_worker":len(grad_sample_norms),
                                    "grad_sample_gradients_norms_means": np.mean(grad_sample_norms),
                                    "grad_sample_gradients_norms_lowerC": np.mean(np.array(grad_sample_norms)<args.max_per_sample_grad_norm),
                                    "grad_sample_gradients_norms_medians": np.median(grad_sample_norms),
                                    "grad_sample_gradients_norms_max": max(grad_sample_norms),
                                    "grad_sample_gradients_norms_last_percentile": np.quantile(grad_sample_norms,0.99),
                                    #"grad_sample_gradients_norms_hist":list(np.histogram(grad_sample_norms,bins=np.arange(100), density=True)[0]),
                                    "per_param_norms":per_param_norms_epoch.mean(1).tolist(),
                                    "optimizer_time":np.mean(optimizer_times),
                                    "forward_time":np.mean(forward_times),
                                    "backward_time":np.mean(backward_times),
                                    "data_loading_time":np.mean(data_loading_times),
                                }
                            )
                        )
                        optimizer_times,forward_times, backward_times, data_loading_times = [],[],[],[]
                        grad_sample_norms = []
                        per_param_norms = None
                    if nb_steps % args.freq_log_val == 0: #Performance Evaluation on the test set
                        test_acc_ema = test(ema, test_loader, device)
                        test_acc_non_ema = test(model, test_loader, device)
                        model.train()
                        print(f"epoch:{epoch},step:{nb_steps}")
                        logger.info(
                            "__log:"
                            + json.dumps(
                                {   "step":nb_steps,
                                    "test_acc_ema": test_acc_ema,
                                    "test_acc_non_ema": test_acc_non_ema,
                                }
                            )
                        )
                if nb_steps >= max_nb_steps:
                    break
        if is_main_worker:
            print(f"nb_examples_epoch_per_worker:{nb_examples_epoch}")
        return nb_steps, per_param_norms_epoch


def test(model, test_loader,device):
    """
    Test the model on the testing set and the training set
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    test_top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            test_top1_acc.append(acc)

    test_top1_avg = np.mean(test_top1_acc)
    # print(f"\tTest set:"f"Loss: {np.mean(losses):.6f} "f"Acc: {top1_avg * 100:.6f} ")
    return (test_top1_avg)


def main():  ## for non poisson, divide bs by world size
    args = parse_args()
    init_distributed_mode(args)
    logger = initialize_exp(args)

    ## Loading the model
    model = prepare_model(args.architecture)
    model.cuda()
    print_params(model)
    if args.multi_gpu: model = DPDDP(model)
        
    rank = args.local_rank
    is_main_worker = args.global_rank == 0
    weights = model.module.parameters() if args.multi_gpu else model.parameters()
    if is_main_worker: print(model)
    populate_dataset(args)

    ## Creating ImageNet Dataloaders with or without AugmentationMultiplicity depending on args.transform
    train_loader,test_loader = prepare_dataloaders(args)

    ## Preparing the optimizer (SGD, Adam, AdamW or LAMB)
    optimizer = prepare_optimizer(weights,args)

    ## IMPORTANT: We choose sigma (noise of dp-sgd) using the TAN scaling strategy
    sigma = get_noise_from_bs(args.batch_size, args.ref_noise, args.ref_B)

    ##We use our PrivacyEngine Augmented to take into accoung the eventual augmentation multiplicity
    privacy_engine = PrivacyEngineAugmented(GradSampleModule.GRAD_SAMPLERS)
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=args.max_per_sample_grad_norm,
        poisson_sampling=args.poisson_sampling,
        K=args.transform
    )

    ## Changes the grad samplers to work with augmentation multiplicity
    prepare_augmult(model, args.transform)
    
    ## We create a shadow model called ema, with parameters equal to an exponential moving average of our model's weights
    ema = None
    ema = create_ema(model)
    ## if args.checkpoint_path does not contain a checkpoint, nothing is loaded
    reload_checkpoint(args,model,ema,optimizer,logger, checkpoint_path=args.checkpoint_path)


    (train_acc,test_acc,losses,top1_accs,) = (0, 0, [], [])
    per_param_norms_epoch = None
    E = get_epochs_from_bs(args.batch_size, args.ref_nb_steps, len(train_loader.dataset))
    if is_main_worker: print(f"E:{E},sigma:{sigma}, BATCH_SIZE:{args.batch_size}, noise_multiplier:{sigma}, EPOCHS:{E}")
        
    nb_steps = args.starting_step
    K = args.transform
    for epoch in range(args.starting_epoch,E+1):
        if nb_steps >= args.ref_nb_steps:
            break
        nb_steps, per_param_norms_epoch = train(
            model,
            ema,
            train_loader,
            optimizer,
            epoch,
            args.ref_nb_steps,
            rank,
            K,
            logger,
            losses,
            top1_accs,
            test_loader,
            is_main_worker,
            args,
            nb_steps,
            per_param_norms_epoch
        )
        if is_main_worker:
            print(f"epoch:{epoch}, Current loss:{losses[-1]:.2f},nb_steps:{nb_steps}, top1_acc of model (not ema){top1_accs[-1]:.2f}")
            save_checkpoint(args,model,ema,epoch+1,nb_steps,logger)
    if is_main_worker:
        ## evaluating and logging
        args.batch_size = args.test_batch_size
        test_acc= test(ema, test_loader, rank)
        test_acc_vote = test_vote(ema,args)  if args.transform!=0 else 0
        logger.info("__log:"+ json.dumps({"final_test_acc_ema": test_acc,"final_test_acc_ema_voted":test_acc_vote}))
        test_acc= test(model, test_loader, rank)
        test_acc_voted= test_vote(model,args) if args.transform!=0 else 0
        print(f"final test acc of non ema model:{test_acc:.2f}, final train acc of non ema model:{train_acc:.2f}")
        logger.info("__log:"+ json.dumps({"final_test_acc_non_ema": test_acc,"final_test_acc_non_ema_voted":test_acc_voted}))


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch imagenet DP Training")


    parser.add_argument("--batch_size", default=256,type=int,metavar="B",help=r"Batch size for simulated training. It will automatically set the noise $\sigma$ s.t. $B/\sigma = B_{ref}/sigma_{ref}$",)

#. If B is grater than this max physical batch size, we will perform gradient accumulation, using aggragated gradients untill the full batch is explored. Usually set to the maximum number of examples that one GPU can handle."
    parser.add_argument("--max_physical_batch_size",default=150,type=int,help="max_physical_batch_size for BatchMemoryManager",)
    parser.add_argument("--num_classes",default=-1,type=int,help="Number of classes in the data set.",)
    parser.add_argument("--proportion",default=1,type=float,help="Training only on a subset of the training set. It will randomly select proportion training data",)
    parser.add_argument("--lr","--learning_rate",default=32,type=float,help="Learning rate during training",)
    parser.add_argument("--test_batch_size",default=512,type=int,help="What batch size to use when evaluating the model the test set",)
    parser.add_argument("--momentum",default=0,type=float,metavar="M",help="SGD momentum term",)
    parser.add_argument("--dampening",default=0,type=float,help="SGD dampening term")
    parser.add_argument("--experiment",default=0,type=int,help="seed for initializing training. Usefull when doing multiple experimemts in parrallel to get stds on the results",)

##Privacy related
    parser.add_argument("-c","--max_per_sample_grad_norm",type=float,default=1,metavar="C",help="Clip per-sample gradients to this norm (default 1.0).")
    parser.add_argument("--delta",type=float,default=8e-7,metavar="D",help="Delta for the DP accountant")
    parser.add_argument("--ref_B",type=int,default=32768,help="reference batch size used with reference noise and number of steps to create for the TAN sclaling stratedy",)
    parser.add_argument("--ref_nb_steps",default=18000,type=int,help="reference number of steps used with reference noise and batch size to create our physical constant",)
    parser.add_argument("--ref_noise",type=float,default=2.5,help="reference noise used with reference batch size and number of steps",)
    parser.add_argument("--poisson_sampling",type=bool_flag,default=True,help="using Poisson sampling - we only performed experiments when set to true because it is the only way to have valid DP guarantees")

## Data loading related
    parser.add_argument("--dataset",type=str,default="imagenet",help="Path to train data",)
    parser.add_argument( "--train_path",type=str,default="",help="name of training set")
    parser.add_argument("--val_path",type=str,default="",help="path to validation data",)

## logging/checkpointing related
    parser.add_argument("--dump_path",type=str,default="",help="Where results will be stored")
    parser.add_argument("--checkpoint_path",type=str,default="",help="where to force reloading from",)
    parser.add_argument("--freq_log",type=int,default=50,help="every each freq_log steps, we log.")
    parser.add_argument("--freq_log_val",type=int,default=1000,help="every each freq_log steps, we evaluate the model on the validation set.")
    parser.add_argument("--starting_epoch",type=int,default=0,help="At what epoch of training to start (usefull when starting from a checkpoint). If set up correctly, this value will be accurately updated when loading the checkpoint.",)
    parser.add_argument("--starting_step",type=int,default=0,help="At what step of training to start (usefull when starting from a checkpoint). If set up correctly, this value will be accurately updated when loading the checkpoint.")

##Data Augentation
    parser.add_argument("--transform",type=int,default=0,help="Order of the Augmentation multplicity (AugMult). If non 0, each image in each batch will be duplicated 'transform' times, and randomly applied 'type_of_augmentation'.",)
    parser.add_argument("--type_of_augmentation", default="OursBest", help="type of AugMult that will be used if 'args.transform' is non 0.")
    parser.add_argument("--train_transform", choices=["random", "flip", "center", "simclr", "beit","resize"], default="center", help="equivalent to AugMult of order 1, i.e classic data augmentation.")
    

    


    parser.add_argument("--architecture",type=str,default='nf_resnet50',help="Type of architecture of the network.",)
    parser.add_argument('--model_ema_decay', type=float, default=0.99999, help='')

#optimizer
    parser.add_argument("--optimizer",type=str,choices=["SGD", "Adam", "AdamW","lamb"],default='SGD',help="Type of optimizer",)
    parser.add_argument('--AdamW_lr', type=float, default=0.001, help='')
    parser.add_argument('--AdamW_beta1', type=float, default=0.9, help='')
    parser.add_argument('--AdamW_beta2', type=float, default=0.999, help='')
    parser.add_argument('--AdamW_eps', type=float, default=1e-08, help='')


    parser.add_argument("--exp_name",type=str, default="bypass")
    parser.add_argument("--init", type=int, default=0)

    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--master_port", type=int, default=-1)
    parser.add_argument("--debug_slurm", type=bool_flag, default=False)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--crop_size", type=int, default=None)

    
#DEIT
####FOR TRANSFORMERS
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--drop_block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    return parser.parse_args()


if __name__ == "__main__":
    main()