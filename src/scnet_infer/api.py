"""One-shot facade and independent reusable SCNet lifecycle.

Sessions own checkpoint resolution, resident model memory, and terminal state;
one-shot calls intentionally build and dispose their own session.
Reads: audio boundary, checkpoint resolver, device validator, runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .audio import load_audio
from .checkpoints import CheckpointSpec, cache_info as inspect_cache
from .checkpoints import get_spec, resolve_checkpoint
from .device import resolve_device
from .runtime import demix, load_runtime_model


@dataclass(frozen=True)
class SeparationResult:
    """Named channel-first float32 source waveforms sharing one sample rate."""

    stems: Mapping[str, np.ndarray]
    sample_rate: int


class SCNetSession:
    """Load-once SCNet model with explicit, observable lifecycle state."""

    def __init__(
        self,
        model_id: str | None = None,
        *,
        device: str | None = "auto",
        checkpoint_path: str | Path | None = None,
        checkpoint_url: str | None = None,
        checkpoint_sha256: str | None = None,
        cache_dir: str | Path | None = None,
        manifest_path: str | Path | None = None,
    ) -> None:
        self._spec: CheckpointSpec = get_spec(model_id, manifest_path)
        self._device_request = device
        self._checkpoint_path = checkpoint_path
        self._checkpoint_url = checkpoint_url
        self._checkpoint_sha256 = checkpoint_sha256
        self._cache_dir = cache_dir
        self._model: torch.nn.Module | None = None
        self._device: torch.device | None = None
        self._status = "new"

    @property
    def status(self) -> str:
        """Return new, ready, released, failed, or closed."""

        return self._status

    @property
    def model_id(self) -> str:
        """Return the stable checkpoint identifier."""

        return self._spec.model_id

    @property
    def device(self) -> str | None:
        """Return the concrete loaded device, if ready."""

        return str(self._device) if self._device is not None else None

    def cache_info(self) -> dict[str, object]:
        """Inspect the same checkpoint target used by load, without downloading."""

        return inspect_cache(
            self._spec,
            cache_dir=self._cache_dir,
            checkpoint_path=self._checkpoint_path,
            checkpoint_url=self._checkpoint_url,
        )

    def load(self) -> SCNetSession:
        """Resolve and construct the model once; repeated ready loads are no-ops."""

        if self._status == "closed":
            raise RuntimeError("cannot load a closed SCNetSession")
        if self._status == "ready":
            return self
        try:
            device = resolve_device(self._device_request)
            checkpoint = resolve_checkpoint(
                self._spec,
                cache_dir=self._cache_dir,
                checkpoint_path=self._checkpoint_path,
                checkpoint_url=self._checkpoint_url,
                checkpoint_sha256=self._checkpoint_sha256,
            )
            model = load_runtime_model(self._spec, checkpoint, device)
        except Exception:
            self._model = None
            self._device = None
            self._status = "failed"
            raise
        self._model = model
        self._device = device
        self._status = "ready"
        return self

    def infer(self, audio: str | Path | np.ndarray, *, sample_rate: int | None = None) -> SeparationResult:
        """Separate audio using the already-loaded resident model."""

        if self._status != "ready" or self._model is None or self._device is None:
            raise RuntimeError("infer() requires a ready SCNetSession; call load() first")
        mixture = load_audio(audio, sample_rate, self._spec.sample_rate)
        stems = demix(self._spec, self._model, mixture, self._device)
        return SeparationResult(stems=stems, sample_rate=self._spec.sample_rate)

    def release(self) -> None:
        """Release resident model/device memory while retaining disk cache."""

        if self._status == "closed":
            return
        self._model = None
        self._device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._status = "released"

    def close(self) -> None:
        """Terminally close this session; repeated calls are safe."""

        if self._status == "closed":
            return
        self.release()
        self._status = "closed"

    def __enter__(self) -> SCNetSession:
        return self.load()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def separate(
    audio: str | Path | np.ndarray,
    *,
    sample_rate: int | None = None,
    model_id: str | None = None,
    device: str | None = "auto",
    checkpoint_path: str | Path | None = None,
    checkpoint_url: str | None = None,
    checkpoint_sha256: str | None = None,
    cache_dir: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> SeparationResult:
    """Separate one input with a disposable, non-global session."""

    with SCNetSession(
        model_id,
        device=device,
        checkpoint_path=checkpoint_path,
        checkpoint_url=checkpoint_url,
        checkpoint_sha256=checkpoint_sha256,
        cache_dir=cache_dir,
        manifest_path=manifest_path,
    ) as session:
        return session.infer(audio, sample_rate=sample_rate)

