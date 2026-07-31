"""Cross-check `ChunkingPlan` against `runtime.demix()`'s inline arithmetic.

`ChunkingPlan` (backends/base.py) is the chunk/step/fade/border owner for the
*MLX* backend only -- `runtime.demix()` (the Torch path) still computes the
same `step`/`fade`/`border` formula inline at `runtime.py:60-63`, and nothing
ties the two together at runtime. That is one design decision encoded in two
places: a real audit finding (fable-reviewed 2026-07-30), deliberately not
fixed here because touching the shipped Torch path is out of scope of this
change -- see `backends/base.py`'s `ChunkingPlan` docstring for the full
rationale and the recorded post-merge follow-up to migrate `demix()` onto
this class.

Until that migration lands, a hand edit to either formula (e.g. changing
`fade = chunk_size // 10` in one place but not the other) would silently
desync the two backends' overlap-add windows. This test recomputes both
formulas independently -- once via `ChunkingPlan.from_spec`, once via the
exact arithmetic copied from `runtime.demix()` -- for every checkpoint the
package's own registry ships, and fails loudly if they ever disagree. It is
entirely offline: no checkpoint weights are loaded, only `CheckpointSpec`
metadata from the packaged manifest, and it does not import `mlx` (unlike
`tests/test_mlx_model.py`'s narrower single-spec version of this same check,
which is skipped whenever the `[mlx]` extra is not installed and therefore
never runs in CI -- see `.github/workflows/ci.yml`). Placing the check here,
in a plain `backends`-only test module, is what makes it actually run
everywhere.
"""

from __future__ import annotations

import pytest

from scnet_infer.backends.base import ChunkingPlan
from scnet_infer.checkpoints import load_manifest


def _registry_specs():
    _, specs = load_manifest()
    return sorted(specs.values(), key=lambda spec: spec.model_id)


@pytest.mark.parametrize("spec", _registry_specs(), ids=lambda spec: spec.model_id)
def test_chunking_plan_matches_demix_inline_arithmetic(spec):
    """`ChunkingPlan.from_spec` must equal `runtime.demix()`'s inline formula.

    The right-hand side of every assertion below is copied verbatim from
    `runtime.demix()` (`runtime.py:60-63`), not re-derived, so this test
    fails the moment either copy of the formula changes without the other.
    """

    plan = ChunkingPlan.from_spec(spec)

    # runtime.py:60-63, verbatim:
    #   step = chunk_size // spec.num_overlap
    #   border = chunk_size - step
    #   fade = chunk_size // 10
    expected_step = spec.chunk_size // spec.num_overlap
    expected_border = spec.chunk_size - expected_step
    expected_fade = spec.chunk_size // 10

    assert plan.chunk_size == spec.chunk_size
    assert plan.num_overlap == spec.num_overlap
    assert plan.batch_size == spec.batch_size
    assert plan.step == expected_step
    assert plan.fade_size == expected_fade
    assert plan.border == expected_border
