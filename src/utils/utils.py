# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import re
import sys
import pickle
import random
import socket
import inspect
import getpass
import argparse
import itertools
import subprocess
from logging import getLogger
import torch
from torch import optim
import time
import numpy as np
from .logger import create_logger
from math import ceil


FALSY_STRINGS = {'off', 'false', '0'}
TRUTHY_STRINGS = {'on', 'true', '1'}

DUMP_PATH = '/checkpoint/%s/dumped' if 'private' in __file__ else '/mnt/vol/gfsai-east/ai-group/users/%s/dumped'
DUMP_PATH = DUMP_PATH % getpass.getuser()


logger = getLogger()

def print_params(net):
    n_params = sum([p.numel() for p in net.parameters() if p.requires_grad])
    if n_params > 1e9:
        print(f"{n_params // 1e9}B parameters")
    elif n_params > 1e6:
        print(f"{n_params // 1e6}M parameters")
    elif n_params > 1e3:
        print(f"{n_params // 1e3}K parameters")
    else:
        print(f"{n_params} parameters")

def bool_flag(s):
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")

def accuracy(preds, labels):
    return (preds == labels).mean()

def accuracy_with_vote(preds, labels,K):
    preds = preds.reshape((-1,K))
    preds = np.apply_along_axis(lambda x: np.bincount(x, minlength=1000), axis=1, arr=preds).argmax(1)
    return (preds == labels).mean()

def get_noise_from_bs(B, ref_noise, ref_B,beta=1):
    """
    output the noise necessary to keep our "physical constant" eta constant.
    """
    return(ref_noise/((ref_B/B)*beta))

# def get_noise_from_bs_tom(B, ref_noise, ref_B):
#     """
#     output the noise necessary to keep our "physical constant" eta constant.
#     """
#     return(ref_noise/np.sqrt((ref_B/B)))

def get_epochs_from_bs(B, ref_nb_steps, size_dataset):
    """
    output the approximate number of epochs necessary to keep our "physical constant" eta constant.
    We use a ceil, but please not that the last epoch will stop when we reach 'ref_nb_steps' steps.
    """
    return(ceil(ref_nb_steps*B/size_dataset))


def repeat(l, r):
    """
    Repeat r times each value of list l.
    """
    return list(itertools.chain.from_iterable(itertools.repeat(x, r) for x in l))


def repeat_to(l, r):
    """
    Repeat values in list l so that it has r values
    """
    assert r % len(l) == 0

    return repeat(l, r // len(l))

def round_to_multiple(l, r):
    return ceil(l / r) * r


def init_distributed_mode(params):
    """
    Handle single and multi-GPU / multi-node / SLURM jobs.
    Initialize the following variables:
        - n_nodes
        - node_id
        - local_rank
        - global_rank
        - world_size
    """
    params.is_slurm_job = 'SLURM_JOB_ID' in os.environ and not params.debug_slurm
    logger.info("SLURM job: %s" % str(params.is_slurm_job))

    # SLURM job
    if params.is_slurm_job:

        assert params.local_rank == -1   # on the cluster, this is handled by SLURM

        SLURM_VARIABLES = [
            'SLURM_JOB_ID',
            'SLURM_JOB_NODELIST', 'SLURM_JOB_NUM_NODES', 'SLURM_NTASKS', 'SLURM_TASKS_PER_NODE',
            'SLURM_MEM_PER_NODE', 'SLURM_MEM_PER_CPU',
            'SLURM_NODEID', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_TASK_PID'
        ]

        PREFIX = "%i - " % int(os.environ['SLURM_PROCID'])
        for name in SLURM_VARIABLES:
            value = os.environ.get(name, None)
            logger.info(PREFIX + "%s: %s" % (name, str(value)))

        # # job ID
        params.job_id = os.environ['SLURM_JOB_ID']

        # number of nodes / node ID
        params.n_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
        params.node_id = int(os.environ['SLURM_NODEID'])

        # local rank on the current node / global rank
        params.local_rank = int(os.environ['SLURM_LOCALID'])
        params.global_rank = int(os.environ['SLURM_PROCID'])

        # number of processes / GPUs per node
        params.world_size = int(os.environ['SLURM_NTASKS'])
        params.n_gpu_per_node = params.world_size // params.n_nodes

        # define master address and master port
        hostnames = subprocess.check_output(['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']])
        params.master_addr = hostnames.split()[0].decode('utf-8')
        assert 10001 <= params.master_port <= 20000 or params.world_size == 1
        logger.info(PREFIX + "Master address: %s" % params.master_addr)
        logger.info(PREFIX + "Master port   : %i" % params.master_port)

        # set environment variables for 'env://'
        os.environ['MASTER_ADDR'] = params.master_addr
        os.environ['MASTER_PORT'] = str(params.master_port)
        os.environ['WORLD_SIZE'] = str(params.world_size)
        os.environ['RANK'] = str(params.global_rank)

    # multi-GPU job (local or multi-node) - jobs started with torch.distributed.launch
    elif params.local_rank != -1:

        assert params.master_port == -1

        # read environment variables
        params.global_rank = int(os.environ['RANK'])
        params.world_size = int(os.environ['WORLD_SIZE'])
        params.n_gpu_per_node = int(os.environ['NGPU'])

        # number of nodes / node ID
        params.n_nodes = params.world_size // params.n_gpu_per_node
        params.node_id = params.global_rank // params.n_gpu_per_node

    # local job (single GPU)
    else:
        assert params.local_rank == -1
        assert params.master_port == -1
        params.n_nodes = 1
        params.node_id = 0
        params.local_rank = 0
        params.global_rank = 0
        params.world_size = 1
        params.n_gpu_per_node = 1

    # sanity checks
    assert params.n_nodes >= 1
    assert 0 <= params.node_id < params.n_nodes
    assert 0 <= params.local_rank <= params.global_rank < params.world_size
    assert params.world_size == params.n_nodes * params.n_gpu_per_node

    # define whether this is the master process / if we are in distributed mode
    params.is_master = params.node_id == 0 and params.local_rank == 0
    params.multi_node = params.n_nodes > 1
    params.multi_gpu = params.world_size > 1

    # summary
    PREFIX = "%i - " % params.global_rank
    logger.info(PREFIX + "Number of nodes: %i" % params.n_nodes)
    logger.info(PREFIX + "Node ID        : %i" % params.node_id)
    logger.info(PREFIX + "Local rank     : %i" % params.local_rank)
    logger.info(PREFIX + "Global rank    : %i" % params.global_rank)
    logger.info(PREFIX + "World size     : %i" % params.world_size)
    logger.info(PREFIX + "GPUs per node  : %i" % params.n_gpu_per_node)
    logger.info(PREFIX + "Master         : %s" % str(params.is_master))
    logger.info(PREFIX + "Multi-node     : %s" % str(params.multi_node))
    logger.info(PREFIX + "Multi-GPU      : %s" % str(params.multi_gpu))
    logger.info(PREFIX + "Hostname       : %s" % socket.gethostname())

    # set GPU device
    torch.cuda.set_device(params.local_rank)

    # initialize multi-GPU
    if params.multi_gpu:

        # http://pytorch.apachecn.org/en/0.3.0/distributed.html#environment-variable-initialization
        # 'env://' will read these environment variables:
        # MASTER_PORT - required; has to be a free port on machine with rank 0
        # MASTER_ADDR - required (except for rank 0); address of rank 0 node
        # WORLD_SIZE - required; can be set either here, or in a call to init function
        # RANK - required; can be set either here, or in a call to init function

        logger.info("Initializing PyTorch distributed ...")
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
        )


def initialize_exp(params):
    """
    Initialize the experience:
    - dump parameters
    - create a logger
    """
    # dump parameters
    get_dump_path(params)
    pickle.dump(params, open(os.path.join(params.dump_path, 'params.pkl'), 'wb'))

    # get running command
    command = ["python", sys.argv[0]]
    for x in sys.argv[1:]:
        if x.startswith('--'):
            assert '"' not in x and "'" not in x
            command.append(x)
        else:
            assert "'" not in x
            if re.match('^[a-zA-Z0-9_]+$', x):
                command.append("%s" % x)
            else:
                command.append("'%s'" % x)
    command = ' '.join(command)
    params.command = command + ' --exp_id "%s"' % params.exp_id

    # check experiment name
    assert len(params.exp_name.strip()) > 0

    # create a logger
    logger = create_logger(os.path.join(params.dump_path, 'train.log'), rank=getattr(params, 'global_rank', 0))
    logger.info("============ Initialized logger ============")
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(params)).items())))
    logger.info("The experiment will be stored in %s\n" % params.dump_path)
    logger.info("Running command: %s" % command)
    logger.info("")
    return logger


def get_dump_path(params):
    """
    Create a directory to store the experiment.
    """
    if params.exp_name == "bypass":
        dump_path = params.dump_path.rstrip("/")
        params.exp_id = os.path.basename(dump_path)
        sweep_path = os.path.dirname(dump_path)
        if sweep_path == '':
            sweep_path = "/tmp"
    else:
        dump_path = DUMP_PATH if params.dump_path == '' else params.dump_path
        assert len(params.exp_name) > 0
        assert os.path.isdir(dump_path)
        # create the sweep path if it does not exist
        sweep_path = os.path.join(dump_path, params.exp_name)
        if not os.path.exists(sweep_path):
            subprocess.Popen("mkdir -p %s" % sweep_path, shell=True).wait()

    # create an ID for the job if it is not given in the parameters.
    # if we run on the cluster, the job ID is the one of Chronos.
    # otherwise, it is randomly generated
    if params.exp_id == '':
        chronos_job_id = os.environ.get('CHRONOS_JOB_ID')
        slurm_job_id = os.environ.get('SLURM_JOB_ID')
        assert chronos_job_id is None or slurm_job_id is None
        exp_id = chronos_job_id if chronos_job_id is not None else slurm_job_id
        if exp_id is None:
            chars = 'abcdefghijklmnopqrstuvwxyz0123456789'
            while True:
                exp_id = ''.join(random.choice(chars) for _ in range(10))
                if not os.path.isdir(os.path.join(sweep_path, exp_id)):
                    break
        else:
            assert exp_id.isdigit()
        params.exp_id = exp_id

    # create the dump folder / update parameters
    params.dump_path = os.path.join(sweep_path, params.exp_id)
    if not os.path.isdir(params.dump_path):
        subprocess.Popen("mkdir -p %s" % params.dump_path, shell=True).wait()


class AdamInverseSqrtWithWarmup(optim.Adam):
    """
    Decay the LR based on the inverse square root of the update number.
    We also support a warmup phase where we linearly increase the learning rate
    from some initial learning rate (`warmup-init-lr`) until the configured
    learning rate (`lr`). Thereafter we decay proportional to the number of
    updates, with a decay factor set to align with the configured learning rate.
    During warmup:
        lrs = torch.linspace(warmup_init_lr, lr, warmup_updates)
        lr = lrs[update_num]
    After warmup:
        lr = decay_factor / sqrt(update_num)
    where
        decay_factor = lr * sqrt(warmup_updates)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_updates=4000, warmup_init_lr=1e-7):
        super().__init__(
            params,
            lr=warmup_init_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        self.warmup_updates = warmup_updates
        self.warmup_init_lr = warmup_init_lr
        # linearly warmup for the first warmup_updates
        warmup_end_lr = lr
        self.lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates
        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * warmup_updates ** 0.5
        self._num_updates = 0

    def get_lr_for_step(self, num_updates):
        if num_updates < self.warmup_updates:
            return self.warmup_init_lr + num_updates * self.lr_step
        else:
            return self.decay_factor * (num_updates ** -0.5)

    def step(self, closure=None):
        super().step(closure)
        self._num_updates += 1
        # update learning rate
        new_lr = self.get_lr_for_step(self._num_updates)
        for param_group in self.param_groups:
            param_group['lr'] = new_lr


def get_schedule(s, steps):
    lr_schedule = None
    if s.startswith("cos:"):
        lr_init = float(s[4:])
        lr_schedule = lr_init * (1 + np.cos(np.pi * np.arange(steps) / steps)) / 2
    else:
        lr_schedule = [float(lr) for lr in s.split("-")]

    if len(lr_schedule) == 1:
        # There is no "schedule" if the learning rate stays constant
        lr_schedule = None
    else:
        lr_schedule = repeat_to(lr_schedule, round_to_multiple(steps, len(lr_schedule)))

    skip = max(len(lr_schedule) // 100, 1)
    logger.info(f"Schedule: {lr_schedule[::skip]}")
    
    return lr_schedule


def get_optimizer(params, parameters, lr_init):
    params.optimizer = params.optimizer
    optim_params = {
        "lr": lr_init
    }

    if params.optimizer == 'adadelta':
        optim_fn = optim.Adadelta
    elif params.optimizer == 'adagrad':
        optim_fn = optim.Adagrad
    elif params.optimizer == 'adam':
        optim_fn = optim.Adam
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif params.optimizer == 'adamw':
        optim_fn = optim.AdamW
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif params.optimizer == 'adam_inverse_sqrt':
        optim_fn = AdamInverseSqrtWithWarmup
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif params.optimizer == 'sparseadam':
        optim_fn = optim.SparseAdam
        optim_params['betas'] = (optim_params.get('beta1', 0.9), optim_params.get('beta2', 0.999))
        optim_params.pop('beta1', None)
        optim_params.pop('beta2', None)
    elif params.optimizer == 'adamax':
        optim_fn = optim.Adamax
    elif params.optimizer == 'asgd':
        optim_fn = optim.ASGD
    elif params.optimizer == 'rmsprop':
        optim_fn = optim.RMSprop
    elif params.optimizer == 'rprop':
        optim_fn = optim.Rprop
    elif params.optimizer == 'sgd':
        optim_fn = optim.SGD
        assert 'lr' in optim_params
    else:
        raise Exception('Unknown optimization params.optimizer: "%s"' % params.optimizer)

    # check that we give good parameters to the optimizer
    expected_args = inspect.getargspec(optim_fn.__init__)[0]
    assert expected_args[:2] == ['self', 'params']
    if not all(k in expected_args[2:] for k in optim_params.keys()):
        raise Exception('Unexpected parameters: expected "%s", got "%s"' % (
            str(expected_args[2:]), str(optim_params.keys())))

    return optim_fn(parameters, **optim_params)


def clip_parameters(model, clip):
    """
    Clip model weights.
    """
    if clip > 0:
        for x in model.parameters():
            x.data.clamp_(-clip, clip)


def to_cuda(*args):
    """
    Move tensors to CUDA.
    """
    return [None if x is None else x.cuda() for x in args]

def pad_tensor(tensor, n, pad_value=-1):
    sz = list(tensor.size())
    sz[0] = n

    padded_tensor = pad_value * torch.ones(sz, dtype=tensor.dtype)
    padded_tensor[:tensor.size(0)] = tensor

    return padded_tensor

def state_dict(params,model,ema,epoch,step):
    r"""
    
    """
    model = model._module.module if params.multi_gpu else model._module
    ema = ema._module.module if params.multi_gpu else ema._module


    data = {
        'model': model.state_dict(),
        'ema':ema.state_dict(),
        'epoch': epoch,
        'step':step,
        'params': vars(params)
    }
    #data['optimizer'] = optimizer.state_dict()
    # TODO: put a state dict
    # if params.private:
    #     data['privacy_engine'] = self.privacy_engine.state_dict()

    return data

def save_checkpoint(params,model,ema,epoch,step,logger):
    """
    Checkpoint the experiment.
    """
    data = state_dict(params,model,ema,epoch,step)
    checkpoint_path = os.path.join(params.dump_path, 'checkpoint.pth')
    logger.info("Saving checkpoint to %s ..." % checkpoint_path)
    torch.save(data, checkpoint_path)

def reload_checkpoint(params,model,ema,optimizer,logger,checkpoint_path=""):
    """
    Reload a checkpoint if we find one.
    """
    if checkpoint_path =="": checkpoint_path = params.dump_path
    checkpoint_path = os.path.join(checkpoint_path, 'checkpoint.pth')
    if not os.path.isfile(checkpoint_path):
        return
    logger.warning('Reloading checkpoint from %s ...' % checkpoint_path)
    data = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))

    # reload model parameters
    if params.multi_gpu:
        model._module.module.load_state_dict(data['model'])
        ema._module.module.load_state_dict(data['ema'])
    else:
        model._module.load_state_dict(data['model'])
        ema._module.load_state_dict(data['ema'])

    # TODO: put a state dict
    # if params.private:
    #     self.privacy_engine.load_state_dict(data['privacy_engine'])

    #optimizer.load_state_dict(data['optimizer'])

    # reload main metrics
    params.starting_epoch = data['epoch']
    if 'step' in data:
        params.starting_step = data['step']
    else:
        print("did not fimd the starting step")
    #self.data_loader.batch_sampler.set_step(self.step)

    logger.warning(f'Checkpoint reloaded. Resuming at epoch {params.starting_epoch}')


def trainable_parameters(module):
    """
    Recursively iterates over all parameters, returning those that
    are trainable (ie they want a grad).
    """
    yield from (
        (p_name, p) for (p_name, p) in module.named_parameters() if p.requires_grad
    )

