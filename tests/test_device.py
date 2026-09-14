"""Explicit device-selection contracts.

Unavailable explicit choices must fail instead of falling back silently.
Reads: scnet_infer.device and monkeypatched torch availability.
"""

from __future__ import annotations

import pytest

from scnet_infer import device


def test_cpu_and_invalid_device() -> None:
    assert str(device.resolve_device("cpu")) == "cpu"
    with pytest.raises(ValueError, match="device must"):
        device.resolve_device("gpu")


def test_auto_and_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: False)
    assert str(device.resolve_device("auto")) == "cpu"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        device.resolve_device("cuda")


def test_explicit_mps_always_raises() -> None:
    """`device="mps"` is unsupported by org decision and always raises --
    this is a scope decision, not an availability check."""
    with pytest.raises(ValueError, match="intermittently mis-compute"):
        device.resolve_device("mps")


def test_auto_never_promotes_to_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy auto-selection is preserved: MPS is never selected, so existing
    Mac callers keep the exact compute path -- and the exact outputs -- they
    had before."""
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: False)
    assert str(device.resolve_device(None)) == "cpu"
    assert str(device.resolve_device("auto")) == "cpu"


def test_explicit_cuda_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(device.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(device.torch.cuda, "device_count", lambda: 1)
    assert str(device.resolve_device("cuda")) == "cuda:0"
    with pytest.raises(RuntimeError, match="only 1 CUDA"):
        device.resolve_device("cuda:1")

