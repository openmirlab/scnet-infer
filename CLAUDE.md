# scnet-infer maintainer guide

**`docs/`** is local-only by policy (2026-09-14): kept on disk, gitignored,
never pushed to GitHub.

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
raises rather than downgrading. `device="mps"` raises unconditionally,
regardless of availability: `torch==2.13.0`'s MPS backend was measured to
intermittently mis-compute this model's chunked forward pass outright (see
`device.py` and CHANGELOG.md's `[0.1.0]` entry for the full trial-by-trial
record). `device="auto"`/`None` keep their legacy CUDA-else-CPU meaning and
never promote a Mac caller onto MPS on their own. Apple Silicon (MPS/MLX)
is out of scope for this package by org decision (2026-09-14) -- there is
no `backend=` axis and no accelerated Apple Silicon path; `runtime.py`'s
`build_model`/`load_runtime_model`/`demix` are the only compute path.

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

Python 3.10 reads the package checkpoint manifest using the conditional
`tomli` dependency; Python 3.11+ uses `tomllib`. CI selects the requested
matrix interpreter explicitly rather than relying on local Python discovery.

## Required verification commands

```bash
uv sync --extra dev
uv run pytest -q
# Verify the lowest supported interpreter explicitly
uv sync --python 3.10 --extra dev
uv run --python 3.10 pytest -q
uv run python -m build
uv run python tools/verify_inference_only.py
uv run python tools/verify_wheel.py
uv run python tools/generate_upstream_fixture.py --fixture tests/fixtures/music_delta_disco_10s
uv run python tools/verify_upstream_parity.py --fixture tests/fixtures/music_delta_disco_10s
uv run python -c "import scnet_infer; print(scnet_infer.__version__)"
git status --short
git log -1 --oneline
```
