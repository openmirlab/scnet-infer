# SCNet Infer Implementation Plan

## Goal and acceptance

Create `scnet-infer`, an inference-only, standalone Python package that exposes
one-shot source separation, the reusable `SCNetSession` lifecycle, and the
`scnet-infer` CLI. The accuracy gate is an end-to-end comparison against the
untouched MSST inference path on a deterministic 10-second stereo mixture made
from the four read-only Music Delta - Disco MUSDB stems.

## Verified sources of truth

- Official architecture: `starrytong/SCNet` revision
  `e0e3f4037dad3fc9437499051e73aed466bd2766` (2024-12-18), MIT license in the
  repository `LICENSE`. This source provides the paper implementation and
  authorship/citation text.
- Integration and current model variants: `ZFTurbo/Music-Source-Separation-Training`
  revision `83d495dfc81b2ede9bc62f4209619f8bdfd14995` (2026-07-12), MIT license in
  the repository `LICENSE`. Extract only its SCNet model and generic overlap-add
  inference behavior; training, datasets, metrics, optimization, augmentation,
  GUI, and experiment orchestration are permanently excluded.
- Primary citation: arXiv:2401.13276, "SCNet: Sparse Compression Network for
  Music Source Separation," with authors copied from the official repository's
  citation block rather than inferred.
- Checkpoint host: ZFTurbo's GitHub Releases. Checkpoint licensing is not stated
  by a primary license grant and is therefore `NOASSERTION`.

## Checkpoint decision

Use `scnet-xl-ihf-v1.0.15` as the default. Its release-reported SDR is 10.0891,
the strongest verified practical SCNet checkpoint in the surveyed official MSST
releases. Keep `scnet-masked-xl-v1.0.17` (9.8286) and
`scnet-tran-v1.0.14` (8.9272) as explicit alternatives. The runtime manifest
will pin original asset URLs, sizes, SHA-256 values where GitHub supplies them
or local verification is completed, source revision, provenance, checked date,
family, and license. No weights are bundled or re-hosted.

## Architecture

1. **Faithful model core** — extract only `scnet`, `scnet_masked`, `scnet_tran`,
   shared separation layers, and the minimal attention primitive required by
   `scnet_tran`. Preserve upstream tensor operations and state-dict names.
2. **Configuration and checkpoints** — package-owned `checkpoints.toml` is the
   only checkpoint registry. A single resolver handles manual path, direct URL,
   configurable cache directory, environment cache override, verified download,
   and read-only `cache_info()` inspection.
3. **Runtime** — validate `auto`, `cpu`, `cuda`, `cuda:N`, and available `mps`;
   load model/config once; reproduce MSST normalization and overlap-add inference;
   return named stereo stems at 44.1 kHz.
4. **Facade** — `separate(...)` owns a disposable session. `SCNetSession`
   implements idempotent `load()`, ready-only `infer()`, reloadable `release()`,
   terminal/idempotent `close()`, status, cache info, and context management.
5. **I/O and CLI** — accept paths and NumPy arrays, normalize through one input
   boundary, write deterministic WAV stems, and keep audio codec scope limited to
   formats supported by the declared pure runtime dependencies.

## Stages and evidence

### Stage 1 — constitutional scaffold

Create src layout, hatchling packaging, single-sourced version, Python 3.10-3.14
CI matrix, README/CLAUDE paired documentation, MIT LICENSE with verified upstream
copyrights, NOTICE, CHANGELOG, nav headers, and an import/version smoke test.
Commit this on `main`, confirm log/status, then create `feat/scnet-infer-runtime`.

### Stage 2 — golden baseline and inference-only core

Record the deterministic input mixture and untouched-upstream output metadata
before adapting runtime behavior. Extract only the minimum model files and prove
the shipped package scan contains no training/evaluation/dataset/experiment
surface. Commit the fixture evidence separately from runtime API work.

### Stage 3 — checkpoint resolver, runtime, and lifecycle

Implement model/config parsing, download integrity, explicit devices, overlap-add
inference, facade, lifecycle, and CLI. Tests cover malformed manifest/config,
package data, checksum failures, URL/path overrides, device errors, lifecycle
transitions, release/cache semantics, and a call-counting no-reload assertion
across sequential `infer()` calls.

### Stage 4 — release gate

Run all required commands exactly as recorded in `CLAUDE.md`: dependency sync,
full tests, wheel-from-sdist build, inference-only scan, clean-wheel install/import,
upstream fixture generation, upstream parity, import/version smoke, git status,
and git log. Real-checkpoint scripts must report a precise skip reason only when
network, CUDA, or checkpoint acquisition genuinely prevents execution.

## Known release constraint

MIT covers the extracted source code from both referenced repositories. The
published checkpoint assets have no verified primary license grant; local use
and direct upstream download can function, but public release remains blocked
pending human review of the `NOASSERTION` weights layer.
