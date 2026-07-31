# scnet-infer maintainer guide

## Scope and status

This standalone package extracts inference-only SCNet architecture code from
starrytong/SCNet and current integration behavior from ZFTurbo/MSST. Runtime is
implemented on `feat/scnet-infer-runtime` for `scnet`, `scnet_masked`, and
`scnet_tran`, with the v1.0.15 10.0891-SDR SCNet checkpoint as default. Training,
evaluation, datasets, experiments, GUI code, and weights are permanently outside
the shipped boundary.

The public facade is `separate(...)`, `SeparationResult`, and `SCNetSession`.
Sessions implement idempotent load, ready-only infer, reloadable release,
terminal close, status, cache inspection, and context management. Explicit CPU
and CUDA choices never fall back silently -- an unavailable explicit choice
raises rather than downgrading. `device="mps"` on the **Torch** backend
raises unconditionally, regardless of availability: it is a reliability
refusal, not an availability check -- `torch==2.13.0`'s MPS backend was
measured to intermittently mis-compute this model's chunked forward pass
outright (see `device.py` and CHANGELOG.md's `[0.1.0]` entry for the full
trial-by-trial record). `device="auto"`/`None` keep their legacy
CUDA-else-CPU meaning and never promote a Mac caller onto MPS on their own.
Apple Silicon acceleration is available and reliable through
`backend="mlx"` instead. Model construction, weight loading, and separation
are delegated to a `backends.SeparationBackend` (`backend="torch"` default,
`"mlx"` on Apple Silicon, `"auto"`) -- see "Backends: Torch / MLX" below.

## Backends: Torch / MLX

`backend` (`"torch"` default, `"mlx"`, `"auto"`) and `device` are separate
axes -- `backend` picks the compute framework, `device` keeps its existing
Torch meaning and is not overloaded to mean "Apple Silicon." See README's
"Backends and devices" for the public contract.

- `src/scnet_infer/backends/` -- the compute seam. `base.py` holds the
  `SeparationBackend` protocol (one `(channels, samples)` mixture in, named
  stems out) and `ChunkingPlan` (mirrors `runtime.demix()`'s own inline
  `step`/`fade`/`border`/`batch_size` arithmetic so the two cannot drift
  apart silently). `torch_backend.py` wraps `runtime.build_model`/
  `load_runtime_model`/`demix` without forking them. `mlx_backend.py` is the
  MLX implementation; `separate()` reproduces `runtime.demix()`'s chunked
  overlap-add bit-for-bit, including a batch-level edge-window quirk in the
  original Torch loop (see that module's docstring) -- reproduced
  deliberately, not "fixed," because parity is measured against what Torch
  actually computes. `__init__.py` resolves a backend by name; `auto` tries
  MLX first (only when the checkpoint's `family` has an MLX model class) and
  falls back to Torch. Backend modules import lazily, so `import scnet_infer`
  never pulls in `mlx` -- `tests/test_backends.py` asserts this via a
  subprocess (an *earlier* test in the same process legitimately importing
  real `mlx` as a side effect of a real `is_available()` call is not the same
  thing as `import scnet_infer` doing it).
- `src/scnet_infer/mlx/` -- **a from-scratch MLX port, not a vendored
  upstream** (unlike the sibling `bs-roformer-infer`/`mdxnet-infer`
  packages: no `mlx-audio-separator`-style project ships an MLX SCNet
  anywhere). `model.py` is derived directly from `_model/scnet.py`,
  `_model/scnet_masked.py`, and `_model/scnet_tran.py`, module tree shape
  preserved so `convert.py`'s key mapping stays a rename rather than a
  restructure. Every `Conv1d`/`Conv2d`/`ConvTranspose2d`/`GroupNorm` call
  goes through a channels-first wrapper (`Conv1dCF`/`Conv2dCF`/
  `ConvTranspose2dCF`/`GroupNormCF`) so the rest of the model can be ported
  nearly line-for-line from Torch's NCHW/NCL code without re-deriving every
  transpose (MLX's native conv/norm layers are channels-last). `model.py`
  also carries `exact_zero_safe_rfft()`, the same Metal rfft workaround as
  the sibling packages (see its docstring for why it was *measured*, not
  assumed, to matter for this architecture's `FeatureConversion` module).
  Rotary embeddings (`scnet_tran` only) are always loaded from the
  checkpoint's own buffer and inverted for `mx.fast.rope`, never derived
  from a theta formula -- see `model.py`'s file-top docstring for the
  measurement this follows from a sibling package's checkpoint that had
  genuinely drifted (this package's own `scnet-tran-v1.0.14` checkpoint
  turned out to match the theta=10000 default bit-for-bit, but the loader
  does not special-case that). `convert.py` (LSTM families) and
  `convert_tran.py` (Transformer family) hold `load_converted_weights()`,
  which raises rather than silently loading a partial checkpoint -- see
  their module docstrings.

**Measured parity** (real `scnet-xl-ihf-v1.0.15` checkpoint, public
`.infer()` API, 2026-07-31, Apple M-series, torch 2.13.0, mlx 0.31.2):
Torch(CPU)-vs-MLX worst max-abs is 1.691e-05 (signal) / 6.764e-05
(zero-padded tail) / 6.926e-05 (near-silent tail) -- CPU, not MPS, is the
reference throughout this package's tests, for the reliability reason
below. CPU-vs-MPS (Torch only, kept as the evidence record that justifies
refusing `device="mps"` on the Torch backend, not as a supported path) is
4.608e-06 / 1.488e-05 / 1.498e-05 absolute, roughly 100x looser than the
sibling `bs-roformer-infer` package's own CPU-vs-MPS number and bisected to
SCNet's decoder-stage dense `Conv2d`+`GLU` (absent from the Roformer
family's mask-estimator path) rather than dismissed as noise. **The reason
`device="mps"` is refused is stronger than that gap alone**: investigating
an anomalous result during this work proved, via a fixed CPU reference held
against three looped Torch-MPS calls in one process, that `torch==2.13.0`'s
MPS backend intermittently mis-computes this model's chunked forward pass
outright -- a large, exactly-repeating wrong answer, two of three trials,
while MLX agreed with the same CPU reference exactly every time. Full
numbers, the bisection, and the trial-by-trial record live in
`tests/test_device_parity.py`'s and `tests/test_mlx_parity.py`'s module
docstrings and in CHANGELOG.md. **Read those docstrings before touching
either test or before ever reconsidering the MPS refusal**: reopening it
requires evidence *for* MPS reliability on this model, not merely time
having passed.

## Attribution and license

The official SCNet authors and Roman Solovyev remain credited in README and
NOTICE. Both source repositories were verified as MIT. Upstream checkpoints are
not covered by a verified license grant and remain `NOASSERTION`.

## File convention

Every load-bearing Python file starts with a module docstring describing its
purpose, rationale, and `Reads:` dependencies. Keep algorithms faithful and put
each config value behind one owner.

## Testing philosophy

Golden evidence is recorded from untouched MSST revision
`83d495dfc81b2ede9bc62f4209619f8bdfd14995` before adapted runtime commits. Unit
tests stay offline; `tools/generate_upstream_fixture.py` and
`tools/verify_upstream_parity.py` form the separate real-checkpoint gate. Float
digests guard on torch/CUDA/device environment.

## Required verification commands

```bash
uv sync --extra dev
uv run pytest -q
uv run python -m build
uv run python tools/verify_inference_only.py
uv run python tools/verify_wheel.py
uv run python tools/generate_upstream_fixture.py --fixture tests/fixtures/music_delta_disco_10s
uv run python tools/verify_upstream_parity.py --fixture tests/fixtures/music_delta_disco_10s
uv run python -c "import scnet_infer; print(scnet_infer.__version__)"
git status --short
git log -1 --oneline

# MLX backend, on an Apple Silicon Mac (the [mlx] extra is Apple-Silicon-only;
# CI never installs it -- see .github/workflows/ci.yml):
uv sync --extra dev --extra mlx
uv run pytest -m realweights tests/test_mlx_parity.py -v
uv run pytest -m realweights tests/test_device_parity.py -v
```
