# scnet-infer

Inference-only SCNet music source separation with verified upstream checkpoints.

## Why this exists

[SCNet](https://github.com/starrytong/SCNet) is the official research
implementation of Sparse Compression Network source separation, but its
repository is organized around training. `scnet-infer` extracts the faithful
model and overlap-add inference path into a standalone installable library with
explicit devices, verified downloads, and a reusable model lifecycle.

## Acknowledgments

- Weinan Tong, Jiaxu Zhu, Jun Chen, Shiyin Kang, Tao Jiang, Yang Li, Zhiyong Wu,
  and Helen Meng authored SCNet and its official implementation.
- [starrytong/SCNet](https://github.com/starrytong/SCNet) is the architecture source.
- Roman Solovyev maintains
  [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training),
  the source of current SCNet integration code and GitHub Release checkpoints.

## Citation

```bibtex
@misc{tong2024scnet,
  title={SCNet: Sparse Compression Network for Music Source Separation},
  author={Weinan Tong and Jiaxu Zhu and Jun Chen and Shiyin Kang and Tao Jiang and Yang Li and Zhiyong Wu and Helen Meng},
  year={2024},
  eprint={2401.13276},
  archivePrefix={arXiv},
  primaryClass={eess.AS}
}
```

## Features

- One-shot `separate(...)` facade and load-once `SCNetSession`.
- Four named stereo stems: drums, bass, other, and vocals.
- Config-driven `scnet`, `scnet_masked`, and `scnet_tran` families.
- Explicit `auto`, `cpu`, `cuda`, and `cuda:N` device selection on the Torch
  backend.
- Native Apple Silicon execution via `backend="mlx"` (see "Backends and
  devices" below) -- `device="mps"` on the Torch backend is refused; see
  that section for why.
- Atomic auto-download, SHA-256 verification, manual checkpoints, direct URL
  overrides, and configurable cache location.
- CLI output as float WAV stems.

## Scope

The package ships model construction, checkpoint resolution, audio validation,
overlap-add inference, and output writing. Training loops, datasets, evaluation
metrics, experiment orchestration, GUI code, and checkpoints are out of scope
forever.

Inputs are mono or stereo paths/NumPy arrays at 44.1 kHz. Array input requires
`sample_rate=44100`; path input reads its rate from the file. Resampling is not
silently performed.

## Install

```bash
pip install scnet-infer
```

Python 3.10 through 3.14 are claimed and exercised in CI.

## Quick Start

One-shot calls load and release a model for each call:

```python
from scnet_infer import separate

result = separate("song.wav", device="cuda:0")
vocals = result.stems["vocals"]  # shape: (2, samples)
```

Use a session to avoid reloading across songs:

```python
from scnet_infer import SCNetSession

with SCNetSession(device="cuda:0") as session:
    first = session.infer("first.wav")
    second = session.infer("second.wav")  # same resident model
```

`infer()` is ready-only. `release()` frees the model but permits a later
`load()`; `close()` is terminal and idempotent. `status` reports `new`, `ready`,
`released`, `failed`, or `closed`.

CLI usage:

```bash
scnet-infer song.wav separated/ --device cuda:0
```

## Backends and devices

`backend` and `device` are separate axes. `backend` picks the compute
framework; `device` keeps its existing Torch meaning and is not overloaded to
mean "Apple Silicon."

| `backend` | Resolves to | `device` accepted |
|---|---|---|
| `"torch"` (default) | Torch | `None`, `"auto"`, `"cpu"`, `"cuda"`, `"cuda:N"` -- `"mps"` raises, see below |
| `"mlx"` | MLX, or raise if unavailable | `None`, `"auto"`, `"mps"` (MLX's own device sentinel) -- anything else raises |
| `"auto"` | MLX on Apple Silicon when the `[mlx]` extra is importable and the checkpoint's family has an MLX port, else Torch | per resolved backend |

`backend` defaults to `"torch"`, so a caller who never mentions it keeps
today's exact behaviour. `device="auto"`/`None` keep their legacy
CUDA-else-CPU meaning and never promote a Mac caller onto MPS on their own.

**`device="mps"` on the Torch backend raises, on purpose.** It is not simply
unimplemented: it was measured, and `torch==2.13.0`'s MPS backend was found
to intermittently mis-compute this model's chunked forward pass outright --
a large, exactly-repeating wrong answer, roughly one to two times in every
three calls (see CHANGELOG.md for the full trial-by-trial record). A
one-in-three chance of a silently wrong separated stem is not a caveat worth
shipping with a warning label. Apple Silicon acceleration is available
through `backend="mlx"` instead, which matched a Torch CPU reference exactly
on every trial in that same investigation.

```python
from scnet_infer import SCNetSession

with SCNetSession(backend="mlx") as session:       # Apple Silicon, native MLX
    result = session.infer("song.wav")

with SCNetSession(device="cpu") as session:         # Torch, portable everywhere
    result = session.infer("song.wav")
```

```bash
scnet-infer song.wav separated/ --backend mlx
scnet-infer song.wav separated/ --device cpu
```

### The MLX backend

Install it with the extra, which is never part of the core install:

```bash
pip install 'scnet-infer[mlx]'
```

All three registry architecture families (`scnet`, `scnet_masked`,
`scnet_tran`) have an MLX port; an explicit `backend="mlx"` request for a
family this port does not support raises rather than silently falling back to
Torch -- `"auto"` is the only backend value that ever substitutes.

Measured Torch(CPU)-vs-MLX parity on the real default `scnet-xl-ihf-v1.0.15`
checkpoint, through the public `.infer()` API, worst-case max-abs divergence
(2026-07-31, Apple M-series, torch 2.13.0, mlx 0.31.2):

| tail | worst max-abs |
|---|---:|
| clean signal | 1.691e-05 |
| zero-padded tail | 6.764e-05 |
| near-silent tail | 6.926e-05 |

MLX is dramatically faster than the Torch CPU fallback on this checkpoint,
and was independently measured faster than Torch's MPS backend too, before
that path was found unreliable for this model and refused. See `CLAUDE.md`
and `tests/test_mlx_parity.py` (`realweights`, deselected by default) for
the full record.

MPS and MLX both need an **arm64 Python interpreter**. Under Rosetta/x86_64
they report as unavailable rather than failing loudly -- an x86_64
interpreter makes `torch.backends.mps.is_available()` return `False`, and
MLX fails to run correctly, so an accelerated path just looks absent rather
than misconfigured. This is easy to hit without noticing: an x86_64 `uv`
resolves x86_64 interpreters, so `uv sync` can silently produce an
environment where the accelerated paths structurally cannot exist. Check
with `python -c "import platform; print(platform.machine())"` -- it must
print `arm64`.

## Models and checkpoints

| Stable ID | Family | Release-reported SDR | Role |
|---|---|---:|---|
| `scnet-xl-ihf-v1.0.15` | `scnet` | 10.0891 | default |
| `scnet-masked-xl-v1.0.17` | `scnet_masked` | 9.8286 | alternative |
| `scnet-tran-v1.0.14` | `scnet_tran` | 8.9272 | alternative |

The default cache is `~/.cache/scnet-infer/`. Override it with `cache_dir=` or
`SCNET_INFER_CACHE`. `cache_info()` uses the same resolver as `load()` and never
downloads merely to inspect status.

For offline installation, download the exact asset named in
`src/scnet_infer/config/checkpoints.toml`, verify its recorded SHA-256, then pass:

```python
session = SCNetSession(checkpoint_path="/offline/model.ckpt", device="cpu")
```

The default upstream URL is:

```text
https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.15/model_scnet_ep_36_sdr_10.0891.ckpt
```

Generic direct URL overrides require an accompanying `checkpoint_sha256=` so a
custom host cannot disable integrity checking.

## What this project will NEVER bundle

Model weights are downloaded directly from their upstream GitHub Releases and
verified before use. No primary source grants a weights license, so every asset
is truthfully recorded as `NOASSERTION`. Source code is MIT; checkpoint use is a
separate downstream decision, and public release remains blocked pending review.

## Development

```bash
uv sync --extra dev
uv run pytest -q
uv run python -m build
uv run python tools/verify_inference_only.py
uv run python tools/verify_wheel.py

# MLX backend, on an Apple Silicon Mac (the [mlx] extra has no Linux wheels,
# so it is never part of --all-extras / CI -- install it explicitly):
uv sync --extra dev --extra mlx
uv run pytest -m realweights tests/test_mlx_parity.py -v
uv run pytest -m realweights tests/test_device_parity.py -v
```

The real-checkpoint accuracy gates and exact commands are recorded in `CLAUDE.md`.

## License

MIT for source code. See `NOTICE` for exact revisions and the separate weights layer.

## Support

Use the project issue tracker after repository creation is approved.

