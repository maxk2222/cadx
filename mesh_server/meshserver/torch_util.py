"""Torch backend classification shared by adapters and the registry.

ROCm PyTorch builds reuse the CUDA device namespace: `torch.cuda.is_available()`
is True, tensors live on device string "cuda", and `torch.version.hip` is set.
Adapters therefore keep using `.to("cuda")` internally — "rocm" is only the
*tier label* for reporting and for the few places where genuinely CUDA-only
code paths exist (diso flash decoder, TRELLIS.2 extensions).
"""

from __future__ import annotations

from typing import Any


def _torch() -> Any | None:
    try:
        import torch

        return torch
    except Exception:
        return None


def is_rocm() -> bool:
    """True when the installed torch is a ROCm/HIP build."""
    torch = _torch()
    if torch is None:
        return False
    return getattr(getattr(torch, "version", None), "hip", None) is not None


def cuda_ok() -> bool:
    """A working torch "cuda" device exists — NVIDIA or ROCm alike."""
    torch = _torch()
    try:
        return bool(torch and torch.cuda.is_available())
    except Exception:
        return False


def mps_ok() -> bool:
    torch = _torch()
    try:
        return bool(torch and torch.backends.mps.is_available())
    except Exception:
        return False


def torch_backend() -> str:
    """"cuda" (NVIDIA) | "rocm" (AMD via HIP) | "mps" | "cpu" | "" (no torch)."""
    if cuda_ok():
        return "rocm" if is_rocm() else "cuda"
    if mps_ok():
        return "mps"
    return "cpu" if _torch() is not None else ""


class TorchDevice:
    """Class-level `device` descriptor: reports "rocm" instead of "cuda" on
    HIP builds so /v1/models stays honest, "mps"/"cpu" where applicable.

    Usage: `device = TorchDevice("cuda")` — the argument is the preferred
    backend when several could match."""

    def __init__(self, prefer: str = "cuda") -> None:
        self.prefer = prefer

    def __get__(self, obj: Any, owner: type | None = None) -> str:
        b = torch_backend()
        return b or self.prefer
