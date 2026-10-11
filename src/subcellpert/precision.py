"""Explicit neural compute precision; observations and optimizer state stay FP32."""

from contextlib import nullcontext

import torch


def validate_precision(precision):
    if precision not in ("fp32", "bf16"):
        raise ValueError("training.precision must be fp32 or bf16")
    return precision


def neural_autocast(precision, device):
    validate_precision(precision)
    if precision == "fp32":
        return nullcontext()
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("BF16 mixed precision requires a CPU or CUDA device")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def check_precision_device(precision, device):
    validate_precision(precision)
    device = torch.device(device)
    if precision == "bf16" and device.type == "cuda":
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                raise ValueError("Selected CUDA device does not support BF16")
