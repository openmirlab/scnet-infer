# Changelog

## 0.1.0 - Unreleased

- Fix Python 3.10 checkpoint imports with a conditional `tomli` backport;
  select CI matrix interpreters explicitly for dependency sync and tests.

- Add the constitution-conformant package scaffold and grounded runtime plan.
- Add faithful SCNet, masked SCNet, and SCNet-Tran inference families.
- Add verified package-owned checkpoint metadata and configurable cache/download paths.
- Add explicit device validation, one-shot separation, reusable lifecycle, and CLI.
- Record a deterministic untouched-upstream golden fixture and bit-identical parity tooling.
- `device="mps"` on the Torch backend is refused unconditionally, regardless
  of availability: `torch==2.13.0`'s MPS backend was measured to
  intermittently mis-compute this model's chunked forward pass outright (a
  large, exactly-repeating wrong answer, roughly one to two times in every
  three calls) -- see `device.py`. `device="auto"`/`None` keep their legacy
  CUDA-else-CPU meaning and never promote a Mac caller onto MPS on their own.
- Added `tests/test_trunk_identity.py`, which pins the deliberately-
  triplicated SD/SU trunk (`Swish`/`ConvolutionModule`/`FusionLayer`/
  `SDlayer`/`SUlayer`/`SDblock`) and `FeatureConversion`'s copy across the
  three `_model` variants, so a one-sided edit fails loudly instead of
  drifting silently. No inference code changed.
- **Removed** the MLX backend and all Torch-MPS integration added on
  2026-07-31 (org decision, 2026-09-14: MLX/MPS is out of scope for this
  package). `backend=` is gone entirely from `SCNetSession`, `separate()`,
  and the CLI -- Torch is the only compute path again, with no dispatch
  seam. `device="mps"` still raises a plain `ValueError`; the vocabulary is
  `"auto"` | `"cpu"` | `"cuda"` | `"cuda:N"`.
