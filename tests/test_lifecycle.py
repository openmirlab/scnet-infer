"""Session lifecycle, release, and load-once regression tests.

A counting model loader proves two sequential inference calls do not reconstruct
the resident model—the behavior the session exists to provide.
Reads: scnet_infer.api with offline test doubles.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from scnet_infer import SCNetSession, api


class FakeModel(torch.nn.Module):
    pass


def install_doubles(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, int]:
    calls = {"loads": 0, "infers": 0}
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"test")
    monkeypatch.setattr(api, "resolve_checkpoint", lambda *args, **kwargs: checkpoint)
    monkeypatch.setattr(api, "resolve_device", lambda value: torch.device("cpu"))

    def load_model(*args: object, **kwargs: object) -> FakeModel:
        calls["loads"] += 1
        return FakeModel()

    def fake_demix(spec: object, model: object, mixture: np.ndarray, device: object) -> dict[str, np.ndarray]:
        calls["infers"] += 1
        return {name: mixture.copy() for name in ("drums", "bass", "other", "vocals")}

    monkeypatch.setattr(api, "load_runtime_model", load_model)
    monkeypatch.setattr(api, "demix", fake_demix)
    return calls


def test_ready_only_and_no_reload_across_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = install_doubles(monkeypatch, tmp_path)
    session = SCNetSession()
    with pytest.raises(RuntimeError, match="requires a ready"):
        session.infer(np.zeros((2, 16), dtype=np.float32), sample_rate=44100)
    assert session.load().load() is session
    session.infer(np.zeros((2, 16), dtype=np.float32), sample_rate=44100)
    session.infer(np.zeros((2, 16), dtype=np.float32), sample_rate=44100)
    assert calls == {"loads": 1, "infers": 2}


def test_release_reload_close_and_context(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = install_doubles(monkeypatch, tmp_path)
    session = SCNetSession().load()
    session.release()
    assert session.status == "released"
    session.load()
    assert calls["loads"] == 2
    session.close()
    session.close()
    assert session.status == "closed"
    with pytest.raises(RuntimeError, match="closed"):
        session.load()


def test_failed_load_is_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(api, "resolve_device", lambda value: (_ for _ in ()).throw(RuntimeError("no device")))
    session = SCNetSession()
    with pytest.raises(RuntimeError, match="no device"):
        session.load()
    assert session.status == "failed"

