"""Input normalization contracts shared by all public calls.

Arrays and paths converge on float32 channel-first stereo without resampling.
Reads: scnet_infer.audio.
"""

from __future__ import annotations

import numpy as np
import pytest

from scnet_infer.audio import load_audio


def test_array_shapes_normalize_to_stereo() -> None:
    mono = np.zeros(32, dtype=np.float64)
    assert load_audio(mono, 44100, 44100).shape == (2, 32)
    channels_last = np.zeros((32, 2), dtype=np.float32)
    assert load_audio(channels_last, 44100, 44100).shape == (2, 32)


def test_array_requires_rate_and_exact_checkpoint_rate() -> None:
    with pytest.raises(ValueError, match="sample_rate is required"):
        load_audio(np.zeros(8), None, 44100)
    with pytest.raises(ValueError, match="requires 44100"):
        load_audio(np.zeros(8), 48000, 44100)

