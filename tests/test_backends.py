"""Backend seam contract: resolution, import purity, and honest failure.

These are offline and hardware-independent. They guard the two properties the
seam exists to protect -- that a requested backend is honoured or refused,
never silently swapped, and that the default import path stays free of
optional frameworks.
"""

from __future__ import annotations

import pytest

from scnet_infer.backends import (
    BACKEND_NAMES,
    DEFAULT_BACKEND,
    BackendUnavailable,
    ChunkingPlan,
    get_backend,
    resolve_backend_name,
)
from scnet_infer.checkpoints import get_spec


def test_default_and_none_resolve_to_torch():
    assert resolve_backend_name(None) == DEFAULT_BACKEND == "torch"
    assert resolve_backend_name("torch") == "torch"


def test_auto_resolves_to_a_registered_backend():
    """`auto` is the one place a fallback is what the caller asked for."""
    assert resolve_backend_name("auto") in BACKEND_NAMES


def test_unknown_backend_name_raises_value_error():
    with pytest.raises(ValueError):
        resolve_backend_name("onnx")


def test_unavailable_backend_raises_rather_than_substituting(monkeypatch):
    """An explicit request is honoured or fails loudly -- never downgraded.

    A silent substitution is only ever discovered by noticing the wrong
    hardware was busy.
    """
    from scnet_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: False))
    with pytest.raises(BackendUnavailable):
        resolve_backend_name("mlx")


def test_auto_falls_back_to_torch_when_mlx_is_unavailable(monkeypatch):
    from scnet_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: False))
    assert resolve_backend_name("auto") == "torch"


def test_auto_prefers_mlx_when_available_and_family_supported(monkeypatch):
    from scnet_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: True))
    assert resolve_backend_name("auto", family="scnet") == "mlx"


def test_explicit_mlx_raises_for_unsupported_family(monkeypatch):
    """A checkpoint family with no MLX head is refused for an *explicit*
    request, never silently run through the wrong construction path."""
    from scnet_infer.backends import mlx_backend

    monkeypatch.setattr(mlx_backend.MLXBackend, "is_available", classmethod(lambda cls: True))
    with pytest.raises(BackendUnavailable):
        resolve_backend_name("mlx", family="unknown_future_family")


def test_torch_backend_satisfies_the_protocol_surface():
    backend = get_backend("torch")
    assert backend.name == "torch"
    assert backend.is_available() is True
    for method in ("separate", "release"):
        assert hasattr(backend, method), f"TorchBackend is missing {method}"


def test_mlx_backend_declares_the_protocol_surface():
    backend = get_backend("mlx")
    assert backend.name == "mlx"
    for method in ("separate", "release", "from_checkpoint", "is_available", "supports_family"):
        assert hasattr(backend, method), f"MLXBackend is missing {method}"


def test_mlx_backend_supports_every_registry_family():
    """Every family the package's own checkpoint registry ships must have an
    MLX head -- support is measured from the model class registry, so this
    also catches a family that was added to the registry but never wired
    into `mlx_backend._FAMILY_MODELS`."""
    from scnet_infer.backends.mlx_backend import supported_families
    from scnet_infer.checkpoints import load_manifest

    _, specs = load_manifest()
    registry_families = {spec.family for spec in specs.values()}
    assert registry_families <= supported_families()


def test_importing_the_package_does_not_pull_in_an_optional_framework():
    """`pip install scnet-infer` must stay MLX-free and import-clean.

    Run in a fresh subprocess rather than this test process: on a machine
    that actually has the ``[mlx]`` extra installed, an *earlier* test in
    this same file legitimately imports real ``mlx`` as a side effect of
    calling the real (unmonkeypatched) ``MLXBackend.is_available()`` -- that
    is correct behaviour for `resolve_backend_name("auto")`, not a leak, and
    asserting against this process's already-polluted `sys.modules` would
    make the test's pass/fail depend on execution order instead of on what
    `import scnet_infer` itself does.
    """
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "import scnet_infer\n"
        "optional = {'mlx', 'mlx_spectro', 'mlx_audio_io'}\n"
        "leaked = sorted({m.split('.')[0] for m in sys.modules} & optional)\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"import scnet_infer pulled in optional frameworks: {leaked}"


def test_importing_backends_package_does_not_pull_in_mlx():
    """Even importing the seam itself must not import mlx -- only requesting
    the mlx backend by name should. Subprocess-isolated for the same reason
    as the test above."""
    import subprocess
    import sys as _sys

    code = (
        "import sys\n"
        "import scnet_infer.backends\n"
        "optional = {'mlx', 'mlx_spectro'}\n"
        "leaked = sorted({m.split('.')[0] for m in sys.modules} & optional)\n"
        "print(','.join(leaked))\n"
    )
    result = subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    leaked = [name for name in result.stdout.strip().split(",") if name]
    assert not leaked, f"import scnet_infer.backends pulled in: {leaked}"


def test_chunking_plan_matches_runtime_demix_arithmetic():
    """`ChunkingPlan` must not silently drift from `runtime.demix()`'s own
    inline `step`/`fade`/`border` formula -- this test ties the two together
    so a future edit to one is caught by the other."""
    spec = get_spec()
    plan = ChunkingPlan.from_spec(spec)

    expected_step = spec.chunk_size // spec.num_overlap
    assert plan.chunk_size == spec.chunk_size
    assert plan.num_overlap == spec.num_overlap
    assert plan.batch_size == spec.batch_size
    assert plan.step == expected_step
    assert plan.fade_size == spec.chunk_size // 10
    assert plan.border == spec.chunk_size - expected_step
