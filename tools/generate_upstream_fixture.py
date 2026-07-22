"""Record a deterministic golden output through untouched MSST code.

The tool imports model and demix functions directly from the pinned read-only
reference and writes environment-bound float evidence.
Reads: MSST_REF, MUSDB_STEMS, upstream checkpoint cache, checkpoint manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from ml_collections import ConfigDict

from scnet_infer.checkpoints import get_spec, resolve_checkpoint


DEFAULT_MSST = Path("/home/worzpro/Desktop/dev/openmirlab/.dev-cache/scnet-probe/msst")
DEFAULT_STEMS = Path("/home/worzpro/Desktop/dev/rytho-ai/archive/agent-daw/refs/DawDreamer/tests/assets/Music Delta - Disco")
REVISION = "83d495dfc81b2ede9bc62f4209619f8bdfd14995"


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    args = parser.parse_args()
    msst = Path(os.environ.get("MSST_REF", DEFAULT_MSST))
    stems = Path(os.environ.get("MUSDB_STEMS", DEFAULT_STEMS))
    if not torch.cuda.is_available():
        print("SKIP: upstream golden generation requires CUDA for the recorded float environment")
        return 0
    if not (msst / ".git").exists() or not all((stems / f"{name}.wav").is_file() for name in ("drums", "bass", "other", "vocals")):
        print(f"SKIP: missing read-only MSST reference ({msst}) or Music Delta - Disco stems ({stems})")
        return 0
    import subprocess
    revision = subprocess.check_output(["git", "-C", str(msst), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        print(f"SKIP: MSST reference revision is {revision}, expected {REVISION}")
        return 0
    spec = get_spec()
    checkpoint_override = os.environ.get("SCNET_INFER_CHECKPOINT")
    try:
        checkpoint = resolve_checkpoint(spec, checkpoint_path=checkpoint_override)
    except Exception as exc:
        print(f"SKIP: checkpoint acquisition failed: {type(exc).__name__}: {exc}")
        return 0
    sys.path.insert(0, str(msst))
    from models.scnet.scnet import SCNet
    from utils.model_utils import demix

    start, frames = 30 * spec.sample_rate, 10 * spec.sample_rate
    parts = []
    for name in spec.sources:
        audio, rate = sf.read(stems / f"{name}.wav", start=start, frames=frames, dtype="float32", always_2d=True)
        if rate != spec.sample_rate or audio.shape != (frames, 2):
            raise RuntimeError(f"unexpected fixture input shape/rate for {name}: {audio.shape}, {rate}")
        parts.append(audio.T)
    mixture = np.sum(parts, axis=0, dtype=np.float32)
    model = SCNet(**dict(spec.model), sources=list(spec.sources))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    model.eval().to("cuda:0")
    config = ConfigDict({
        "audio": {"chunk_size": spec.chunk_size},
        "inference": {"batch_size": spec.batch_size, "num_overlap": spec.num_overlap},
        "training": {"instruments": list(spec.sources), "target_instrument": None, "use_amp": spec.use_amp},
    })
    outputs = demix(config, model, mixture, torch.device("cuda:0"), "scnet", pbar=False)
    args.fixture.mkdir(parents=True, exist_ok=True)
    mixture_path = args.fixture / "mixture.wav"
    if mixture_path.is_file():
        existing, rate = sf.read(mixture_path, dtype="float32", always_2d=True)
        if rate != spec.sample_rate or not np.array_equal(existing.T, mixture):
            sf.write(mixture_path, mixture.T, spec.sample_rate, subtype="FLOAT")
    else:
        sf.write(mixture_path, mixture.T, spec.sample_rate, subtype="FLOAT")
    np.savez_compressed(args.fixture / "upstream_outputs.npz", **outputs)
    metadata = {
        "source_revision": REVISION,
        "checkpoint_sha256": spec.sha256,
        "fixture": "Music Delta - Disco stems, seconds 30-40",
        "sample_rate": spec.sample_rate,
        "shape": list(mixture.shape),
        "mixture_sha256": digest(mixture),
        "outputs": {name: {"shape": list(value.shape), "sha256": digest(value)} for name, value in outputs.items()},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
    }
    (args.fixture / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
