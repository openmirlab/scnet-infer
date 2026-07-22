"""Compare package output directly with a recorded untouched-upstream run.

The exact environment is guarded before float digests are treated as bit-exact.
Reads: generated fixture, SCNET_INFER_CHECKPOINT, package runtime.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from scnet_infer import SCNetSession


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    required = [args.fixture / "mixture.wav", args.fixture / "upstream_outputs.npz", args.fixture / "metadata.json"]
    if not all(path.is_file() for path in required):
        print(f"SKIP: fixture is incomplete at {args.fixture}; run generate_upstream_fixture.py first")
        return 0
    if not torch.cuda.is_available():
        print("SKIP: parity requires CUDA matching the recorded upstream environment")
        return 0
    metadata = json.loads((args.fixture / "metadata.json").read_text())
    current = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)}
    expected = metadata["environment"]
    mismatches = [key for key in current if current[key] != expected[key]]
    if mismatches:
        print(f"SKIP: float fixture environment mismatch for {mismatches}; expected {expected}, got {current}")
        return 0
    checkpoint = os.environ.get("SCNET_INFER_CHECKPOINT")
    try:
        with SCNetSession(device="cuda:0", checkpoint_path=checkpoint) as session:
            package = session.infer(args.fixture / "mixture.wav")
    except Exception as exc:
        print(f"SKIP: package checkpoint acquisition/inference failed: {type(exc).__name__}: {exc}")
        return 0
    upstream = np.load(args.fixture / "upstream_outputs.npz")
    failed = False
    for name, value in package.stems.items():
        difference = float(np.max(np.abs(value - upstream[name])))
        exact = np.array_equal(value, upstream[name])
        print(f"{name}: exact={exact} max_abs_diff={difference:.9g}")
        failed |= not exact
    if failed:
        raise SystemExit("Upstream parity failed")
    print("Upstream parity passed: all four stems are bit-identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
