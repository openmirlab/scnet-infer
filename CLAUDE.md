# scnet-infer maintainer guide

## Scope and status

This standalone package extracts inference-only SCNet architecture code from
starrytong/SCNet and current integration behavior from ZFTurbo/MSST. The initial
main-branch scaffold exposes stable names but intentionally raises until the
feature branch implements runtime. Training, evaluation, datasets, experiments,
GUI code, and weights are permanently outside the shipped boundary.

## Attribution and license

The official SCNet authors and Roman Solovyev remain credited in README and
NOTICE. Both source repositories were verified as MIT. Upstream checkpoints are
not covered by a verified license grant and remain `NOASSERTION`.

## File convention

Every load-bearing Python file starts with a module docstring describing its
purpose, rationale, and `Reads:` dependencies. Keep algorithms faithful and put
each config value behind one owner.

## Testing philosophy

Golden evidence must be recorded from untouched upstream before adapting numeric
code. Unit tests stay offline; real checkpoint parity is a separate explicit gate.

## Required verification commands

```bash
uv sync --all-extras --dev
uv run pytest -q
uv run python -m build
uv run python tools/verify_inference_only.py
uv run python tools/verify_wheel.py
uv run python tools/generate_upstream_fixture.py --fixture tests/fixtures/music_delta_disco_10s
uv run python tools/verify_upstream_parity.py --fixture tests/fixtures/music_delta_disco_10s
uv run python -c "import scnet_infer; print(scnet_infer.__version__)"
git status --short
git log -1 --oneline
```

