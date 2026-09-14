"""Explicit compute-device validation.

Auto-detection chooses CUDA when available and CPU otherwise; explicit requests
are honored or rejected without silent fallback. `auto` deliberately never
resolves to MPS. Vocabulary: `"auto"`, `"cpu"`, `"cuda"`, `"cuda:N"` -- `"mps"`
always raises; see `resolve_device()`'s docstring.
Reads: torch backend availability.
"""

from __future__ import annotations

import re

import torch


def resolve_device(requested: str | None) -> torch.device:
    """Resolve one concrete supported device.

    `mps` is unsupported by org decision (2026-09-14) and raises
    unconditionally, regardless of whether this machine has a working MPS
    build -- this is not an availability check, it is a scope decision:
    torch's MPS backend was measured to intermittently mis-compute this
    model's chunked forward pass outright (a large, repeating wrong answer,
    roughly 1-2 times in 3 calls). `device="cpu"` has no such issue.

    `auto` keeps its legacy meaning -- CUDA if available, else CPU -- and
    never promotes a Mac caller onto MPS on its own regardless.
    """

    value = "auto" if requested is None else requested.lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if value == "cpu":
        return torch.device("cpu")
    if value == "mps":
        raise ValueError(
            "device='mps' is unsupported: torch's MPS backend was measured to "
            "intermittently mis-compute this model's chunked forward pass (a "
            "large, repeating wrong answer, roughly 1-2 times in 3 calls -- see "
            "CHANGELOG.md). Use device='cpu' or a 'cuda'/'cuda:N' device instead."
        )
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

