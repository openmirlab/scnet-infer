"""Offline MLX model tests: weight-conversion audit, wiring, and numeric
parity against the real Torch model classes on tiny synthetic configs.

Skipped whole-file when the ``[mlx]`` extra is not installed. These do not
need a real checkpoint -- they build a tiny random-weight model (small
enough to construct and forward-pass in well under a second per family) and
compare directly against the real Torch `SCNet`/`SCNet` (masked)/
`SCNet_Tran` classes loaded with the *same* converted weights, which is the
strongest offline check obtainable without downloading anything. Real-
checkpoint Torch-vs-MLX numeric parity through the public API, including the
silence fixtures, lives in ``test_mlx_parity.py`` (``realweights``,
deselected by default).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_spectro")

TINY_KWARGS = {
    "sources": ("drums", "bass", "other", "vocals"),
    "audio_channels": 2,
    "dims": (4, 6, 8),
    "nfft": 64,
    "hop_size": 16,
    "win_size": 64,
    "normalized": True,
    "band_SR": (0.175, 0.392, 0.433),
    "band_stride": (1, 4, 16),
    "band_kernel": (3, 4, 16),
    "conv_depths": (1, 1, 1),
    "compress": 2,
    "conv_kernel": 3,
    "num_dplayer": 2,
    "expand": 1,
}

TINY_TRAN_KWARGS = dict(
    TINY_KWARGS,
    tran_rotary_embedding_dim=8,
    tran_depth=1,
    tran_heads=2,
    tran_dim_head=8,
    tran_attn_dropout=0.0,
    tran_ff_dropout=0.0,
    tran_flash_attn=False,
)

# Well within one tiny model's noise floor (measured ~1e-7 to 1e-8 across all
# three families); headroom kept for run-to-run float32 accumulation noise.
MAX_ABS_TOLERANCE = 1e-4

CHUNK_SAMPLES = 16 * 40  # a few hop windows


def _audio(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((1, 2, CHUNK_SAMPLES)) * 0.1).astype(np.float32)


def _compare(torch_model, mlx_model, weights) -> float:
    from scnet_infer.mlx.convert import load_converted_weights

    load_converted_weights(mlx_model, weights)  # must not raise
    torch_model.eval()
    mlx_model.eval()

    audio = _audio()
    with torch.no_grad():
        torch_out = torch_model(torch.from_numpy(audio)).numpy()
    mlx_out = mlx_model(mx.array(audio))
    mx.eval(mlx_out)
    mlx_out = np.array(mlx_out)

    assert torch_out.shape == mlx_out.shape
    return float(np.abs(torch_out - mlx_out).max())


def test_scnet_family_converts_loads_and_matches_torch():
    from scnet_infer._model.scnet import SCNet
    from scnet_infer.mlx.convert import convert_torch_to_mlx_weights
    from scnet_infer.mlx.model import SCNetMLX

    torch.manual_seed(0)
    torch_model = SCNet(**TINY_KWARGS)
    mlx_model = SCNetMLX(**TINY_KWARGS)
    weights = convert_torch_to_mlx_weights(torch_model.state_dict(), "scnet")

    worst = _compare(torch_model, mlx_model, weights)
    assert worst < MAX_ABS_TOLERANCE, f"scnet: Torch-vs-MLX max abs {worst:.3e}"


def test_scnet_masked_family_converts_loads_and_matches_torch():
    from scnet_infer._model.scnet_masked import SCNet as SCNetMasked
    from scnet_infer.mlx.convert import convert_torch_to_mlx_weights
    from scnet_infer.mlx.model import SCNetMaskedMLX

    torch.manual_seed(0)
    torch_model = SCNetMasked(**TINY_KWARGS)
    mlx_model = SCNetMaskedMLX(**TINY_KWARGS)
    weights = convert_torch_to_mlx_weights(torch_model.state_dict(), "scnet_masked")

    worst = _compare(torch_model, mlx_model, weights)
    assert worst < MAX_ABS_TOLERANCE, f"scnet_masked: Torch-vs-MLX max abs {worst:.3e}"


def test_scnet_tran_family_converts_loads_and_matches_torch():
    from scnet_infer._model.scnet_tran import SCNet_Tran
    from scnet_infer.mlx.convert import convert_torch_to_mlx_weights
    from scnet_infer.mlx.model import SCNetTranMLX

    torch.manual_seed(0)
    torch_model = SCNet_Tran(**TINY_TRAN_KWARGS)
    mlx_model = SCNetTranMLX(**TINY_TRAN_KWARGS)
    weights = convert_torch_to_mlx_weights(torch_model.state_dict(), "scnet_tran")

    worst = _compare(torch_model, mlx_model, weights)
    assert worst < MAX_ABS_TOLERANCE, f"scnet_tran: Torch-vs-MLX max abs {worst:.3e}"


@pytest.mark.parametrize(
    "family,build_torch,build_mlx,kwargs",
    [
        ("scnet", "scnet_infer._model.scnet:SCNet", "scnet_infer.mlx.model:SCNetMLX", TINY_KWARGS),
        (
            "scnet_masked", "scnet_infer._model.scnet_masked:SCNet",
            "scnet_infer.mlx.model:SCNetMaskedMLX", TINY_KWARGS,
        ),
        (
            "scnet_tran", "scnet_infer._model.scnet_tran:SCNet_Tran",
            "scnet_infer.mlx.model:SCNetTranMLX", TINY_TRAN_KWARGS,
        ),
    ],
)
def test_load_converted_weights_raises_on_a_dropped_model_parameter(family, build_torch, build_mlx, kwargs):
    """Deleting a converted tensor must surface as a raised error naming the
    shortfall, not a silent partial load (the bug `strict=False` alone would
    hide)."""
    import importlib

    from scnet_infer.mlx.convert import (
        convert_torch_to_mlx_weights,
        load_converted_weights,
    )

    torch_module, torch_name = build_torch.split(":")
    mlx_module, mlx_name = build_mlx.split(":")
    torch_cls = getattr(importlib.import_module(torch_module), torch_name)
    mlx_cls = getattr(importlib.import_module(mlx_module), mlx_name)

    torch.manual_seed(0)
    torch_model = torch_cls(**kwargs)
    mlx_model = mlx_cls(**kwargs)
    weights = convert_torch_to_mlx_weights(torch_model.state_dict(), family)

    victim_key = next(iter(weights))
    del weights[victim_key]

    with pytest.raises(ValueError, match="unmatched"):
        load_converted_weights(mlx_model, weights)


def test_load_converted_weights_raises_on_an_unconsumed_converted_tensor():
    from scnet_infer._model.scnet import SCNet
    from scnet_infer.mlx.convert import (
        convert_torch_to_mlx_weights,
        load_converted_weights,
    )
    from scnet_infer.mlx.model import SCNetMLX

    torch.manual_seed(0)
    torch_model = SCNet(**TINY_KWARGS)
    mlx_model = SCNetMLX(**TINY_KWARGS)
    weights = convert_torch_to_mlx_weights(torch_model.state_dict(), "scnet")

    weights["bogus.unmapped.key"] = mx.array([1.0])

    with pytest.raises(ValueError, match="dropped"):
        load_converted_weights(mlx_model, weights)


def test_exact_zero_safe_rfft_swaps_and_restores_mx_fft_rfft():
    """The guard must actually change which implementation runs while active,
    and put the original back afterwards -- a no-op wrapper protects nothing,
    and a wrapper that fails to restore would corrupt every later call."""
    from scnet_infer.mlx.model import exact_zero_safe_rfft

    original = mx.fft.rfft
    with exact_zero_safe_rfft():
        assert mx.fft.rfft is not original
        frame = mx.zeros((4, 8))
        result = mx.fft.rfft(frame, axis=-1)
        mx.eval(result)
    assert mx.fft.rfft is original
