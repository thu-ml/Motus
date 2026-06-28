import logging
import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def add_lora_to_linear(
    module: nn.Linear,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> None:
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")
    if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
        module.lora_A.requires_grad = True
        module.lora_B.requires_grad = True
        return

    dtype = module.weight.dtype
    device = module.weight.device
    a = torch.empty((rank, module.in_features), device=device, dtype=torch.float32)
    nn.init.kaiming_uniform_(a, a=math.sqrt(5))
    b = torch.zeros((module.out_features, rank), device=device, dtype=torch.float32)

    module.register_parameter("lora_A", nn.Parameter(a.to(dtype=dtype)))
    module.register_parameter("lora_B", nn.Parameter(b.to(dtype=dtype)))
    module.lora_scaling = float(alpha) / float(rank)
    module.lora_dropout = float(dropout)
    module.weight.requires_grad = False
    if module.bias is not None:
        module.bias.requires_grad = False

    def _lora_forward_hook(linear: nn.Linear, inputs, output):
        if getattr(linear, "lora_disabled", False):
            return output
        x = inputs[0]
        if linear.lora_dropout > 0:
            x = F.dropout(x, p=linear.lora_dropout, training=linear.training)
        delta = F.linear(F.linear(x, linear.lora_A), linear.lora_B) * linear.lora_scaling
        return output + delta

    module._lora_hook_handle = module.register_forward_hook(_lora_forward_hook)


def add_lora_to_linear_modules(
    root: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> Dict[str, int]:
    count = 0
    params = 0
    for module in root.modules():
        if isinstance(module, nn.Linear):
            add_lora_to_linear(module, rank=rank, alpha=alpha, dropout=dropout)
            count += 1
            params += module.lora_A.numel() + module.lora_B.numel()
    return {"linear_modules": count, "linear_lora_params": params}


def mark_only_lora_as_trainable(model: nn.Module) -> Dict[str, int]:
    total = 0
    trainable = 0
    for name, param in model.named_parameters():
        is_lora = "lora_A" in name or "lora_B" in name
        param.requires_grad = is_lora
        total += param.numel()
        if is_lora:
            trainable += param.numel()
    logger.info("LoRA trainable parameters: %.2fM / %.2fB", trainable / 1e6, total / 1e9)
    return {"total_params": total, "trainable_lora_params": trainable}
