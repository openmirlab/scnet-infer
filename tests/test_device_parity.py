"""Real-checkpoint CPU-vs-MPS output parity -- the evidence record behind
`device="mps"` being *refused* on the Torch backend (see CHANGELOG.md and
`scnet_infer.device.resolve_device`). This file does not gate a supported
path; `device="mps"` raises unconditionally on the Torch backend now. It
stays because the measurements here, plus the reliability finding at the
bottom of this docstring, are exactly what justifies that refusal being
evidence-based rather than merely inherited from the package's original
"unverified" wording -- deleting this file would throw away the record a
future maintainer needs before ever reconsidering the refusal.

Marked ``realweights`` and deselected by default: it needs the real default
checkpoint already present on disk and an Apple Silicon Mac. It never
downloads -- a test that silently fetches ~200MB is not an offline test.

Run explicitly:  pytest -m realweights tests/test_device_parity.py -v

Exercises a single full-chunk forward pass (not the 4x-overlap chunked
`demix()` path) directly through the model, because SCNet-XL's chunk_size
(485100 samples, ~11s) makes even one CPU forward pass take tens of seconds
on this architecture -- num_overlap=4 chunked demix() would multiply that
fourfold per device per tail. A single forward pass at chunk_size still
exercises the complete model math (STFT -> encoder -> separation_net ->
decoder -> ISTFT) end to end; `demix()`'s overlap-add windowing is a linear
combination of device-identical weights applied to per-device outputs, so it
does not need its own separate device-parity check.

The silence cases are the point, not signal alone: this package's MLX port
(added alongside this MPS work) carries a Metal rfft workaround for a
related but distinct numerical hazard, and a fixture without a genuinely
silent/near-silent region would not build confidence that *this* path -- the
Torch MPS backend -- behaves the same on silence as it does on signal.

Recorded measurement (2026-07-31, Apple M-series, torch 2.13.0), worst-case
max-abs CPU-vs-MPS divergence per tail, single full-chunk forward, and its
size **relative to each stem's own peak** (absolute numbers alone are not
enough to judge a divergence -- 1.5e-05 against a peak of 1.0 is inaudible;
against a peak of 0.01 it would not be):

    tail          max_abs     worst stem  peak(stem)  dBFS-of-diff
    signal        4.608e-06   other       0.1382      -89.5
    zeros         1.488e-05   other       0.4080      -88.8
    near_silent   1.498e-05   other       0.4080      -88.7

Per-stem worst case across all three tails is the "bass" stem at -80.7 dBFS
(max_abs 6.66e-06 against its own peak of 0.072) -- about 15 dB *above* a
16-bit dithered noise floor (~-93 dBFS), so this is not universally "below a
bit of significance," but it is also nowhere near audible or measurable by
ear in practice; every stem sits between -80.7 and -95.9 dBFS across all
three tails. This is roughly two orders of magnitude looser than the sibling
`bs-roformer-infer` package's CPU-vs-MPS measurement (1.136e-07 absolute),
which was worth explaining rather than waving away as "noise":

Bisected by hooking every major submodule (encoder stages, `separation_net`,
each decoder stage's `FusionLayer`/`SUlayer`) and comparing CPU vs MPS
activations at each stage on the same input (zeros tail). Divergence starts
small and stays small through the encoder and the 8-layer LSTM
`separation_net` (about 1e-6 to 5e-6 relative to each stage's own peak
activation). The largest single jump in the whole network -- 5.1e-06 to
9.7e-05 relative, about 19x -- lands at `decoder[0]`'s `FusionLayer`, right
after `separation_net`'s output, where activation magnitude peaks at its
highest point in the network (~250). Re-isolating that one layer (feeding
CPU-computed, therefore device-identical, inputs to both a CPU and an MPS
copy of just that layer) shows the jump is fresh, not carried forward: the
elementwise skip-add and channel-repeat are bit-identical (0.0 diff, as
elementwise ops on identical inputs must be), the dense (non-grouped, full
channel-mixing) `Conv2d` introduces 4.5e-06 relative divergence on its own,
and the following `GLU` roughly quadruples it to 1.8e-05. This pattern
repeats at each of the three decoder stages' `FusionLayer`s, compounding to
~1e-4 relative by mid-decode before the final ISTFT/overlap-add settles it
back down. The architectural difference from the Roformer siblings: SCNet's
decoder does real 2D convolution across a joint frequency x time grid at
channel widths up to several hundred, where the Roformer family's
mask-estimator path has no equivalent dense spatial Conv2d at that width --
this looks like MPS's Conv2d kernel accumulating its channel-reduction sum
in a different order than CPU specifically at that width, at the point in
the network where activations are largest. This is treated as a documented
property of this architecture's use of MPS conv kernels, not investigated
further as a bug: it is measured, explained, and stable (see below), not
merely characterized as "noise."

The gate below is set from what was actually measured (worst single-tail
max_abs 1.498e-05), with headroom for run-to-run noise, not widened to make
an unrelated future regression pass quietly. On its own, this ~1e-5-scale,
bisected-and-explained divergence would not necessarily justify refusing
`device="mps"` -- it is quiet, and the mechanism is understood. It is the
finding below that actually disqualifies the path.

**This is the finding that matters most in this file.** This package's own
MLX parity work (`tests/test_mlx_parity.py`) independently discovered that
**Torch's own MPS backend intermittently mis-computes this specific large
model's chunked forward pass outright**, returning a wildly wrong,
exactly-repeating value (5.14e-01 max abs, vs. an expected ~1e-5) in roughly
one to two of every three repeated calls -- proven by looping `demix()` on
CPU (held fixed as ground truth) against `demix()` on MPS three times in the
same process, same model, same input: two of three MPS calls disagreed with
CPU by 5.1443e-01 (the identical wrong value both times), one agreed to
1.2979e-05. See that file's module docstring for the full triangulation.
This is why `device="mps"` now raises unconditionally on the Torch
backend -- not the quiet, explained ~1e-5 divergence measured above, but a
silently wrong stem roughly a third to two-thirds of the time. Reopening
that refusal requires new evidence that this specific failure mode no
longer occurs (a newer torch release, primarily), not a re-reading of the
~1e-5 numbers above, which were never the disqualifying evidence.

A caution for whoever edits this file next: because `test_cpu_and_mps_agree_
including_silence` below computes exactly the CPU-vs-MPS comparison that
caught the flake above, it can itself hit the same Torch-MPS misbehavior and
report a spurious failure. If this test ever fails with a divergence far
above the ~1e-5 range recorded here (not a modest 2x-5x overshoot, but
multiple orders of magnitude), that is the known flake recurring, not a new
regression -- re-run it in isolation to confirm, and do not treat a clean
re-run as grounds to lift the Torch-backend MPS refusal.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from scnet_infer.checkpoints import ChecksumError, get_spec, resolve_checkpoint
from scnet_infer.device import mps_available
from scnet_infer.runtime import build_model

pytestmark = pytest.mark.realweights

SEED = 1
MAX_ABS_TOLERANCE = 1e-4


def _cached_checkpoint(spec):
    try:
        return resolve_checkpoint(spec, checkpoint_path=None)
    except (OSError, ChecksumError):  # pragma: no cover - network/env dependent
        return None


@pytest.fixture(scope="module")
def loaded_model():
    if not mps_available():
        pytest.skip("MPS unavailable: needs an Apple Silicon Mac and an arm64 torch build")
    spec = get_spec()
    checkpoint = _cached_checkpoint(spec)
    if checkpoint is None or not checkpoint.is_file():
        pytest.skip(f"{spec.model_id} checkpoint unavailable; run resolve_checkpoint() first")

    model = build_model(spec)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return spec, model


def _fixture(spec, tail: str) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    chunk_size = spec.chunk_size
    signal_samples = chunk_size // 2
    chunk = (rng.standard_normal((2, chunk_size)) * 0.1).astype(np.float32)
    if tail == "zeros":
        chunk[:, signal_samples:] = 0.0
    elif tail == "near_silent":
        chunk[:, signal_samples:] *= 1e-6
    return chunk


@pytest.mark.parametrize("tail", ["signal", "zeros", "near_silent"])
def test_cpu_and_mps_agree_including_silence(loaded_model, tail):
    import gc

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    spec, model_cpu = loaded_model
    mix = _fixture(spec, tail)

    outputs = {}
    with torch.inference_mode():
        for device_name in ("cpu", "mps"):
            device = torch.device(device_name)
            model = model_cpu.to(device).eval()
            x = torch.from_numpy(mix).unsqueeze(0).to(device)
            outputs[device_name] = model(x)[0].cpu().numpy()
    model_cpu.to(torch.device("cpu"))

    worst = 0.0
    # dBFS is negative; a LARGER (less negative) value means a bigger
    # divergence relative to that stem's own peak, so the worst case is the
    # maximum, not the minimum -- got this backwards on a first pass, caught
    # by cross-checking against the per-stem table computed independently in
    # the investigation this test's docstring records.
    worst_dbfs = float("-inf")
    for i, stem in enumerate(spec.sources):
        reference, candidate = outputs["cpu"][i], outputs["mps"][i]
        diff = np.abs(reference - candidate)
        peak = float(np.abs(reference).max())
        worst = max(worst, float(diff.max()))
        if peak > 0 and diff.max() > 0:
            worst_dbfs = max(worst_dbfs, 20 * np.log10(float(diff.max()) / peak))
    print(
        f"\n[device parity] tail={tail} worst_max_abs={worst:.3e} "
        f"worst_dBFS_relative_to_stem_peak={worst_dbfs:.1f}"
    )
    assert worst < MAX_ABS_TOLERANCE, (
        f"tail={tail}: CPU-vs-MPS max abs {worst:.3e} exceeds {MAX_ABS_TOLERANCE:.0e} "
        f"(torch {torch.__version__}); investigate rather than widen the tolerance"
    )
