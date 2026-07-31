# Changelog

## 0.1.0 - Unreleased

- Add the constitution-conformant package scaffold and grounded runtime plan.
- Add faithful SCNet, masked SCNet, and SCNet-Tran inference families.
- Add verified package-owned checkpoint metadata and configurable cache/download paths.
- Add explicit device validation, one-shot separation, reusable lifecycle, and CLI.
- Record a deterministic untouched-upstream golden fixture and bit-identical parity tooling.
- **`device="mps"` on the Torch backend is refused -- re-grounded on
  stronger evidence, not the original unverified refusal reinstated
  unchanged.** `device.py` originally refused MPS unconditionally
  ("SCNet checkpoint parity has not been verified on MPS"). During this
  release's MLX work, that refusal was briefly lifted after measuring
  CPU-vs-MPS divergence on the real `scnet-xl-ihf-v1.0.15` checkpoint and
  finding it small in absolute terms. That measurement is kept below because
  it is real and it is what led to the finding that mattered -- but the
  conclusion drawn from it was wrong, and the refusal is back, verified
  rather than merely unverified this time:

  CPU-vs-MPS single full-chunk forward, absolute and relative to each
  stem's own peak:

      tail          max_abs     worst stem   dBFS relative to that stem's own peak
      signal        4.608e-06   other        -89.5
      zeros         1.488e-05   bass         -80.8
      near_silent   1.498e-05   bass         -80.7

  This alone is not disqualifying -- quiet, and bisection traced the ~100x
  gap from the sibling `bs-roformer-infer` package's own CPU-vs-MPS number
  (1.136e-07) to a real, named mechanism: hooking every major submodule and
  diffing CPU vs MPS activations at each stage shows the divergence starts
  small (~1e-6 relative) through the encoder and the 8-layer LSTM
  `separation_net`, then jumps ~19x (5.1e-06 to 9.7e-05 relative)
  specifically at the first decoder stage's `FusionLayer`, right after
  `separation_net`'s output where activation magnitude peaks highest in the
  network. Re-isolating that one layer with device-identical inputs on both
  sides shows the jump is fresh, not carried forward: the elementwise
  skip-add and channel-repeat are bit-identical, but the dense (non-grouped,
  full channel-mixing) `Conv2d` introduces divergence on its own and the
  following `GLU` roughly quadruples it. SCNet's decoder does real 2D
  convolution across a joint frequency x time grid at channel widths up to
  several hundred; the Roformer sibling family's mask-estimator path has no
  equivalent dense spatial `Conv2d` at that width -- this looks like MPS's
  `Conv2d` kernel accumulating its channel-reduction sum in a different
  order than CPU at that width. Full record in
  `tests/test_device_parity.py`'s module docstring.

  **What actually disqualifies the path**: investigating an anomalous
  Torch(MPS)-vs-MLX result during the MLX parity work below found that
  `torch==2.13.0`'s MPS backend *intermittently mis-computes this model's
  chunked (`batch_size=4`) forward pass outright* -- not a small numeric
  divergence, a large, exactly-repeating wrong answer. Proven directly: a
  Torch CPU reference held fixed while looping `demix()` on MPS three times
  in the same process, same model, same input --

      trial 0: cpu-vs-mps=5.1443e-01 (WRONG)   cpu-vs-mlx=6.9261e-05 (clean)
      trial 1: cpu-vs-mps=1.2979e-05 (clean)   cpu-vs-mlx=6.9261e-05 (clean)
      trial 2: cpu-vs-mps=5.1443e-01 (WRONG)   cpu-vs-mlx=6.9261e-05 (clean)

  -- MPS disagreed with CPU by the identical large wrong value two times out
  of three; MLX agreed with CPU exactly every time. A path that silently
  returns a wrong stem one to two times in three is not a caveat to document
  and ship, it is a broken path, and "usually correct" does not survive
  contact with the article-2 standard the rest of this campaign was held to
  (a fixture that measured nothing, a tolerance widened without a cause, a
  device silently ignored were all rejected on the same principle). The
  original refusal's premise -- "parity has not been verified on MPS" -- was
  wrong to begin with in one direction (it undersold how close the
  *reliable* MPS runs are) and right in the direction that matters (MPS
  should not be trusted here): it is now verified, and verified as
  intermittently wrong, which is stronger grounds for the refusal than the
  original "unverified" wording ever was.

  `device="mps"` now raises `ValueError` on the Torch backend unconditionally
  -- naming the real reason and pointing at `backend="mlx"` -- regardless of
  whether MPS is actually available on the machine, because this is a
  reliability refusal, not an availability check. `device="auto"`/`None`
  keep their legacy CUDA-else-CPU meaning, unchanged throughout this whole
  investigation. Restoring the refusal costs Mac users nothing: unlike when
  the original refusal was written, Apple Silicon acceleration is now
  available through the MLX backend below, which is both faster and, on
  this evidence, the reliable choice Torch-MPS is not.
- Add an optional MLX backend (Apple Silicon native execution) behind a new
  additive `backend="torch"` (default, unchanged) / `"mlx"` / `"auto"` axis on
  `SCNetSession`, `separate()`, and the CLI's `--backend`; `device` keeps its
  existing meaning and MLX owns its own execution target (`None`/`"auto"`/
  `"mps"` -- MLX's own device sentinel, independent of the Torch backend's
  `device="mps"` refusal above). Ships MLX ports of all three registry
  architecture families (`scnet`, `scnet_masked`, `scnet_tran`, the last
  including a from-checkpoint rotary-embedding load rather than an assumed
  formula, since a sibling package's checkpoint proved those can genuinely
  drift from the theta=10000 default during training -- this package's own
  `scnet-tran-v1.0.14` checkpoint turned out to match that default
  bit-for-bit, but the loader does not assume that in advance). Measured
  Torch(CPU)-vs-MLX parity on the real default checkpoint through the public
  `.infer()` API: signal 1.691e-05, zeros 6.764e-05, near_silent 6.926e-05
  (CPU, not MPS, is the reference here and throughout this package's tests
  now, for the reliability reason above). The Metal rfft workaround
  (`exact_zero_safe_rfft`) was measured, not assumed, inert for this
  architecture: an ablation against the (then-current) MPS reference moved
  the zeros-tail figure from 7.4215e-05 to 7.7039e-05 with the guard
  removed, a ~4% shift, not the ~36,000x jump the sibling
  `bs-roformer-infer` package measured for its own (attention-based)
  architecture; kept anyway as cheap, harmless insurance. See README.md and
  `CLAUDE.md` for the module layout and the `[mlx]` optional extra.
- Correct `ChunkingPlan`'s docstring: it claimed to be the single
  chunk-arithmetic owner for both backends, but only `MLXBackend` consumes
  it -- `runtime.demix()` still computes the identical formula inline
  (`runtime.py:60-63`). The docstring now states that split and records
  migrating `demix()` onto `ChunkingPlan` as a post-merge follow-up.
  Added `tests/test_chunking_plan.py` (cross-checks both formulas against
  the registry's real chunk configs, offline -- the prior single-spec
  version of this check lived behind `importorskip("mlx.core")` and never
  actually ran in CI) and `tests/test_trunk_identity.py` (pins the
  deliberately-triplicated SD/SU trunk and `FeatureConversion` copy across
  the three `_model` variants so a one-sided edit fails loudly instead of
  drifting silently). No inference code changed.
