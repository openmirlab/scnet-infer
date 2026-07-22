"""Committed upstream golden-fixture integrity checks.

Float output digests are environment-bound and real inference comparison remains
in tools/verify_upstream_parity.py.
Reads: tests/fixtures/music_delta_disco_10s.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


FIXTURE = Path(__file__).parent / "fixtures" / "music_delta_disco_10s"


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def test_committed_upstream_fixture_matches_metadata() -> None:
    metadata = json.loads((FIXTURE / "metadata.json").read_text())
    outputs = np.load(FIXTURE / "upstream_outputs.npz")
    assert metadata["source_revision"] == "83d495dfc81b2ede9bc62f4209619f8bdfd14995"
    assert set(outputs.files) == {"drums", "bass", "other", "vocals"}
    for name in outputs.files:
        assert list(outputs[name].shape) == metadata["outputs"][name]["shape"]
        assert digest(outputs[name]) == metadata["outputs"][name]["sha256"]

