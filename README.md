# scnet-infer

Inference-only SCNet music source separation with verified upstream checkpoints.

## Why this exists

[SCNet](https://github.com/starrytong/SCNet) is the official research
implementation of Sparse Compression Network source separation. This package
extracts its inference surface into a standalone, installable library without
the upstream training stack.

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

Runtime implementation is in progress on `feat/scnet-infer-runtime`. The stable
surface is `separate(...)`, `SCNetSession`, and the `scnet-infer` CLI.

## Scope

This package will include model construction, checkpoint resolution, audio
normalization, overlap-add inference, and output writing. It will never include
training loops, datasets, evaluation metrics, experiment orchestration, GUI code,
or bundled checkpoints.

## Install

```bash
pip install scnet-infer
```

## Quick Start

The runtime is not implemented on `main` yet. Every inference entry point fails
loudly instead of returning placeholder output.

## What this project will NEVER bundle

Model weights are downloaded directly from their upstream GitHub Releases and
verified before use. Their license is currently `NOASSERTION`; source code is MIT.

## Development

See `CLAUDE.md` for exact verification commands and
`docs/implementation-plan.md` for the grounded implementation plan.

## License

MIT for source code. See `NOTICE` for provenance and the separate weights layer.

## Support

Use the eventual project issue tracker after repository creation is approved.

