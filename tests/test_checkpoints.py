"""Offline checkpoint manifest and resolver contracts.

Tests cover package data, malformed overrides, cache inspection, and integrity
failure without making network requests.
Reads: scnet_infer.checkpoints and temporary files.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scnet_infer.checkpoints import (
    CheckpointConfigError,
    ChecksumError,
    cache_info,
    get_spec,
    load_manifest,
    resolve_checkpoint,
)


def test_packaged_manifest_inventory() -> None:
    default, specs = load_manifest()
    assert default == "scnet-xl-ihf-v1.0.15"
    assert {spec.family for spec in specs.values()} == {"scnet", "scnet_masked", "scnet_tran"}
    assert all(spec.license == "NOASSERTION" for spec in specs.values())
    assert all(len(spec.sha256) == 64 for spec in specs.values())
    assert all(spec.url.startswith("https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/") for spec in specs.values())


def test_malformed_manifest_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text("schema_version = 2\n")
    with pytest.raises(CheckpointConfigError, match="schema_version"):
        load_manifest(path)


def test_cache_info_is_read_only(tmp_path: Path) -> None:
    spec = get_spec()
    cache = tmp_path / "cache"
    info = cache_info(spec, cache_dir=cache)
    assert info["exists"] is False
    assert not cache.exists()


def test_manual_path_checksum_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "wrong.ckpt"
    path.write_bytes(b"wrong")
    with pytest.raises(ChecksumError, match="checksum mismatch"):
        resolve_checkpoint(get_spec(), checkpoint_path=path)


def test_direct_url_requires_integrity_override() -> None:
    with pytest.raises(CheckpointConfigError, match="sha256"):
        resolve_checkpoint(get_spec(), checkpoint_url="https://example.invalid/model.ckpt")


def test_manual_path_accepts_matching_generic_hash(tmp_path: Path) -> None:
    path = tmp_path / "custom.ckpt"
    path.write_bytes(b"custom")
    digest = hashlib.sha256(b"custom").hexdigest()
    resolved = resolve_checkpoint(get_spec(), checkpoint_path=path, checkpoint_sha256=digest)
    assert resolved == path


def test_direct_url_download_is_atomic_and_verified(tmp_path: Path) -> None:
    source = tmp_path / "source.ckpt"
    source.write_bytes(b"download")
    digest = hashlib.sha256(b"download").hexdigest()
    cache = tmp_path / "cache"
    resolved = resolve_checkpoint(
        get_spec(), cache_dir=cache, checkpoint_url=source.as_uri(), checkpoint_sha256=digest
    )
    assert resolved.read_bytes() == b"download"
    assert not list(resolved.parent.glob("*.part"))
