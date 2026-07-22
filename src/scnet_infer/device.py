"""Explicit compute-device validation.

Auto-detection chooses CUDA when available and CPU otherwise; explicit requests
are honored or rejected without silent fallback. MPS remains disabled until a
real SCNet parity run verifies it.
Reads: torch backend availability.
"""

from __future__ import annotations

import re

import torch


def resolve_device(requested: str | None) -> torch.device:
    """Resolve one concrete supported device."""

    value = "auto" if requested is None else requested.lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "mps":
        raise ValueError("MPS is not supported: SCNet checkpoint parity has not been verified on MPS")
    if value == "cuda":
        value = "cuda:0"
    match = re.fullmatch(r"cuda:(\d+)", value)
    if match:
        index = int(match.group(1))
        if not torch.cuda.is_available():
            raise RuntimeError(f"explicit device {value!r} requested but CUDA is unavailable")
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"explicit device {value!r} requested but only {torch.cuda.device_count()} CUDA device(s) exist")
        return torch.device(value)
    raise ValueError("device must be one of: auto, cpu, cuda, cuda:N")

