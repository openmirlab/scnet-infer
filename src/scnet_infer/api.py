"""Task-level source-separation facade and lifecycle contract.

This scaffold fixes the public names before runtime implementation while
failing loudly for every unsupported operation.
Reads: no runtime resources in the scaffold stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class SeparationResult:
    """Named source waveforms sharing one sample rate."""

    stems: Mapping[str, np.ndarray]
    sample_rate: int


class SCNetSession:
    """Reusable SCNet inference session (runtime implementation pending)."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError("SCNet runtime is implemented on the feature branch")


def separate(*args: object, **kwargs: object) -> SeparationResult:
    """Separate one input with a disposable session."""

    raise NotImplementedError("SCNet runtime is implemented on the feature branch")

