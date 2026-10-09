"""Device helpers for running Motus on CUDA, Intel XPU, or CPU.

The upstream code targets CUDA and hardcodes ``torch.amp.autocast('cuda', ...)``
along with CUDA-only device APIs. These helpers keep the original CUDA behaviour
byte-for-byte while routing device-sensitive calls to the backend that actually
owns the tensors, so the same checkpoints run unchanged on Intel GPUs.
"""

from __future__ import annotations

import torch

_SUPPORTED_TYPES = ("cuda", "xpu", "cpu")


def xpu_is_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def device_type(device) -> str:
    """Normalize a torch.device / str / tensor to its type string."""
    if device is None:
        return "cpu"
    if isinstance(device, str):
        return device.split(":", 1)[0]
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, torch.Tensor):
        return device.device.type
    return getattr(device, "type", "cpu")


def resolve_device(preference: str = "auto") -> torch.device:
    """Pick a device, preferring CUDA so existing CUDA setups are unaffected."""
    if preference in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if xpu_is_available():
            return torch.device("xpu")
        return torch.device("cpu")

    requested = device_type(preference)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but this PyTorch build cannot access a CUDA device")
    if requested == "xpu" and not xpu_is_available():
        raise RuntimeError("XPU requested, but this PyTorch build cannot access an Intel XPU")
    return torch.device(preference)


def amp_autocast(device, dtype=None, enabled: bool = True):
    """``torch.amp.autocast`` bound to the device type that owns the tensors."""
    resolved = device_type(device)
    if resolved not in _SUPPORTED_TYPES:
        resolved = "cpu"
    # CPU autocast only implements reduced precision, so an fp32 request is a no-op there.
    if resolved == "cpu" and dtype in (None, torch.float32):
        return torch.amp.autocast("cpu", enabled=False)
    if dtype is None:
        return torch.amp.autocast(resolved, enabled=enabled)
    return torch.amp.autocast(resolved, dtype=dtype, enabled=enabled)


def autocast_dtype(device) -> torch.dtype:
    """BF16 when the device reports support for it, FP16 on accelerators, FP32 otherwise."""
    resolved = device_type(device)
    if resolved == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if resolved == "xpu":
        return torch.bfloat16 if torch.xpu.is_bf16_supported() else torch.float16
    return torch.float32


def empty_device_cache(device=None) -> None:
    resolved = device_type(device) if device is not None else None
    if resolved in (None, "cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if resolved in (None, "xpu") and xpu_is_available():
        torch.xpu.empty_cache()


def set_device(local_rank: int, device=None) -> None:
    resolved = device_type(device) if device is not None else None
    if resolved in (None, "cuda") and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    elif resolved in (None, "xpu") and xpu_is_available():
        torch.xpu.set_device(local_rank)


def current_device_index(device=None) -> int:
    resolved = device_type(device) if device is not None else None
    if resolved in (None, "cuda") and torch.cuda.is_available():
        return torch.cuda.current_device()
    if resolved in (None, "xpu") and xpu_is_available():
        return torch.xpu.current_device()
    return 0


def manual_seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if xpu_is_available():
        torch.xpu.manual_seed_all(seed)


def distributed_backend(device=None) -> str:
    """NCCL on CUDA, XCCL on Intel XPU, Gloo elsewhere."""
    resolved = device_type(device) if device is not None else None
    if resolved in (None, "cuda") and torch.cuda.is_available():
        return "nccl"
    if resolved in (None, "xpu") and xpu_is_available():
        return "xccl"
    return "gloo"
