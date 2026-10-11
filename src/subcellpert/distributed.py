"""One-process-per-device training with global cell-batch semantics."""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.nn.functional import all_reduce
from torch.nn.parallel import DistributedDataParallel


def world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def primary():
    return rank() == 0


def barrier():
    if world_size() > 1:
        dist.barrier()


def broadcast(value):
    if world_size() == 1:
        return value
    values = [value if primary() else None]
    dist.broadcast_object_list(values, src=0)
    return values[0]


def gather(value):
    if world_size() == 1:
        return [value]
    values = [None] * world_size()
    dist.all_gather_object(values, value)
    return values


def local_batch_size(global_batch, size=None):
    size = world_size() if size is None else size
    if type(global_batch) is not int or global_batch < 1 or size < 1 or global_batch % size:
        raise ValueError("Global training.batch_size must be a positive integer divisible by WORLD_SIZE")
    return global_batch // size


def profile_encoder_batch(profile):
    local = local_batch_size(profile["batch_size"], profile.get("world_size", 1))
    encoder = profile.get("encoder_batch_size")
    if type(encoder) is not int or not 1 <= encoder <= local:
        raise ValueError("Encoder batch must be a positive integer no larger than the per-rank training batch")
    return encoder


@contextmanager
def launch(device_name):
    """Use torchrun's rendezvous; no process group or CUDA access for one rank."""
    size = int(os.environ.get("WORLD_SIZE", "1"))
    if size == 1:
        yield str(device_name)
        return
    device = torch.device(device_name)
    if device.type == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank >= torch.cuda.device_count():
            raise ValueError("Each torchrun process needs a distinct visible CUDA device")
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("Distributed training supports CUDA/NCCL or CPU/Gloo")
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo", timeout=timedelta(minutes=60))
    try:
        yield str(device)
    finally:
        dist.destroy_process_group()


def synchronized_batch_norm(module, values):
    """Differentiable global moments, including global unbiased running variance.

    Complete-cell shards are equal-sized. GEARS normalizes either cells or
    cell×gene rows, so every rank contributes the same number of observations.
    Works on Gloo for equivalence tests and NCCL for GPU training.
    """
    count = len(values) * world_size()
    work = values.float() if values.dtype in (torch.float16, torch.bfloat16) else values
    mean = all_reduce(work.mean(0)) / world_size()
    variance = all_reduce((work - mean).square().mean(0)) / world_size()
    if module.track_running_stats:
        with torch.no_grad():
            module.num_batches_tracked.add_(1)
            momentum = module.momentum if module.momentum is not None else 1 / float(module.num_batches_tracked)
            module.running_mean.lerp_(mean.detach().to(module.running_mean.dtype), momentum)
            module.running_var.lerp_((variance.detach() * count / (count - 1)).to(module.running_var.dtype), momentum)
    result = (work - mean) * torch.rsqrt(variance + module.eps)
    if module.affine:
        result = result * module.weight + module.bias
    return result.to(values.dtype)


class _ObjectiveModule(nn.Module):
    def __init__(self, model, objective):
        super().__init__()
        self.model, self.objective = model, objective

    def forward(self, batch, **kwargs):
        return self.objective(self.model, batch, **kwargs)


class DistributedObjective:
    """DDP wraps the actual objective forward, preserving plain checkpoint keys."""

    def __init__(self, model, objective, device):
        device = torch.device(device)
        objective.distributed = True
        for module in model.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.synchronize_distributed = True
        self.ddp = DistributedDataParallel(
            _ObjectiveModule(model, objective),
            device_ids=[device.index] if device.type == "cuda" else None,
            # Static gene graphs are large; BatchNorm buffers are synchronized
            # by global moments and do not need a per-step graph broadcast.
            broadcast_buffers=False,
        )

    def __call__(self, model, batch, **kwargs):
        return self.ddp(batch, **kwargs)
