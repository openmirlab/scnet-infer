"""Torch(CPU)-vs-MLX output parity on the real default checkpoint, including
silence, exercised through the public `separate()`/`SCNetSession` API on a
real WAV file on disk (not a bare module forward pass).

Marked ``realweights`` and deselected by default: needs the ``[mlx]``
extra, an Apple Silicon Mac, and the default checkpoint already on disk
(this test never downloads it itself if missing -- it skips).

Run explicitly:  pytest -m realweights tests/test_mlx_parity.py -v

The silence cases are the point of this file, not signal alone: every
chunk's tail can legitimately be silent (music has rests, and any track
shorter than one chunk is zero-padded up to `chunk_size` by
`runtime.demix()`'s own chunking), so a fixture without a genuinely silent
region would not exercise that path. See
`scnet_infer/mlx/model.py::exact_zero_safe_rfft` for the guard this
package's MLX port carries for the general shape of this hazard, and that
module's docstring for why it was measured (not assumed) inert for this
specific architecture.

**The Torch reference runs on CPU, not MPS -- deliberately, and this
replaced an earlier MPS-based version of this file.** `device="mps"` is now
refused on the Torch backend (see `scnet_infer.device.resolve_device` and
CHANGELOG.md) because it was found, during this very investigation, to
intermittently mis-compute this model's chunked forward pass outright: a
Torch CPU reference held fixed while looping Torch-MPS calls three times in
one process (same model, same input) caught MPS disagreeing with CPU by a
large, exactly-repeating wrong value in two of the three trials, while MLX
agreed with that same CPU reference exactly all three times:

    trial 0: cpu-vs-mps=5.1443e-01 (WRONG)   cpu-vs-mlx=6.9261e-05 (clean)
    trial 1: cpu-vs-mps=1.2979e-05 (clean)   cpu-vs-mlx=6.9261e-05 (clean)
    trial 2: cpu-vs-mps=5.1443e-01 (WRONG)   cpu-vs-mlx=6.9261e-05 (clean)

That evidence is what justified refusing `device="mps"` on the Torch
backend rather than keeping it with a caveat -- a silently wrong stem one to
two times in three is not a caveat, it is a broken path. It also settles
which side of *this* file's own comparison to trust: CPU, not MPS, and MLX
was never the unreliable side to begin with. See
`tests/test_device_parity.py`'s module docstring for the full CPU-vs-MPS
bisection this refusal rests on, and CHANGELOG.md for the complete record.

SCNet-XL is an expensive model: its default chunk_size is 485100 samples
(~11s) and `num_overlap=4` with `batch_size=4` means one region typically
means a single batched (4-chunk) forward pass, not four separate ones. The
fixture below is sized to exactly one chunk so `runtime.demix()`'s own
reflect-padding path (`length > 2*border`) never triggers -- the smallest
input that still exercises the real chunked overlap-add end to end. A CPU
forward pass at this size is not fast either (comparable order of magnitude
to what MPS took), so this file's runtime did not improve by switching the
reference -- what improved is that the reference is now trustworthy.

Recorded measurement (2026-07-31, Apple M-series, torch 2.13.0, mlx 0.31.2),
worst-case max-abs Torch(CPU)-vs-MLX divergence per tail, through the public
`.infer()` API on a real WAV file:

    signal        1.691e-05
    zeros         6.764e-05
    near_silent   6.926e-05 (matches the triangulation above exactly, as
                             expected: same model, same input, same code path)

The Metal rfft guard's contribution was isolated directly against the
(then-current) MPS reference: remove-the-fix, zeros tail, with guard
7.4215e-05, without guard 7.7039e-05, restored 7.4215e-05 (bit-identical to
the original with-guard run, confirming the swap/restore is exact).
Removing the guard moved parity by ~4%, not an order of magnitude --
**measured inert** for this architecture, matching the reasoning in
`scnet_infer/mlx/model.py::exact_zero_safe_rfft`'s docstring (GroupNorm's
eps=1e-5 is three orders above the ~4.5e-07 artifact and no operation here
discards a token's own magnitude and renormalizes it the way the sibling
Roformer packages' `L2Norm` does). Kept anyway: cheap, harmless, and
consistent with the org's pattern. That ablation was run against MPS before
the refusal above was put in place; the guard's own mechanics do not depend
on which Torch device is being compared against, so the finding stands.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

pytestmark = pytest.mark.realweights

SEED = 1
# The gate is set from what was actually measured, with headroom for
# run-to-run noise, not widened to make an unrelated future regression pass
# quietly. CPU has no known reliability issue on this path (see module
# docstring), so unlike the earlier MPS-referenced version of this file,
# there is no separate "suspect" threshold or tiebreaker here -- a failure
# here means investigate the MLX port.
MAX_ABS_TOLERANCE = 5e-4


def _mlx_available():
    try:
        import mlx.core  # noqa: F401
        import mlx_spectro  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.fixture(scope="module")
def spec_and_checkpoint():
    if not _mlx_available():
        pytest.skip("MLX extra not installed: pip install 'scnet-infer[mlx]'")

    from scnet_infer.checkpoints import ChecksumError, get_spec, resolve_checkpoint

    spec = get_spec()
    try:
        checkpoint = resolve_checkpoint(spec, checkpoint_path=None)
    except (OSError, ChecksumError) as exc:  # pragma: no cover - network/env dependent
        pytest.skip(f"{spec.model_id} checkpoint unavailable: {exc}")
    if not checkpoint.is_file():
        pytest.skip(f"{spec.model_id} checkpoint unavailable")
    return spec, checkpoint


def _fixture_audio(spec, tail: str) -> np.ndarray:
    """One chunk's worth of stereo audio, (samples, channels): real signal up
    front, `tail` behaviour after it -- matching what `soundfile` hands
    `load_audio()`."""
    rng = np.random.default_rng(SEED)
    chunk_size = spec.chunk_size
    signal_samples = chunk_size // 2
    chunk = (rng.standard_normal((chunk_size, 2)) * 0.1).astype(np.float32)
    if tail == "zeros":
        chunk[signal_samples:, :] = 0.0
    elif tail == "near_silent":
        chunk[signal_samples:, :] *= 1e-6
    return chunk


@pytest.mark.parametrize("tail", ["signal", "zeros", "near_silent"])
def test_mlx_matches_torch_including_silence(tmp_path, spec_and_checkpoint, tail):
    from scnet_infer.api import SCNetSession

    spec, checkpoint = spec_and_checkpoint
    audio = _fixture_audio(spec, tail)

    wav_path = tmp_path / f"{tail}.wav"
    sf.write(str(wav_path), audio, spec.sample_rate, subtype="FLOAT")

    with SCNetSession(
        spec.model_id, backend="torch", device="cpu", checkpoint_path=checkpoint
    ) as torch_session:
        torch_result = torch_session.infer(wav_path)

    with SCNetSession(
        spec.model_id, backend="mlx", device=None, checkpoint_path=checkpoint
    ) as mlx_session:
        mlx_result = mlx_session.infer(wav_path)

    assert set(torch_result.stems) == set(mlx_result.stems)
    worst = 0.0
    for stem, reference in torch_result.stems.items():
        candidate = mlx_result.stems[stem]
        assert reference.shape == candidate.shape, f"{stem}: shape moved between backends"
        diff = float(np.abs(reference - candidate).max())
        worst = max(worst, diff)
    print(f"\n[mlx parity] tail={tail} worst_max_abs={worst:.3e}")
    assert worst < MAX_ABS_TOLERANCE, (
        f"tail={tail}: Torch(CPU)-vs-MLX max abs {worst:.3e} exceeds "
        f"{MAX_ABS_TOLERANCE:.0e}. If this fires only for a silent tail, "
        f"suspect exact_zero_safe_rfft in scnet_infer/mlx/model.py -- "
        f"investigate rather than widen the tolerance"
    )
