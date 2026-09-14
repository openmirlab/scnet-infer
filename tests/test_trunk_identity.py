"""Byte-identity pin for the triplicated SCNet trunk classes.

`Swish`, `ConvolutionModule`, `FusionLayer`, `SDlayer`, `SUlayer`, and
`SDblock` are defined three times over, verbatim, in `_model/scnet.py`
(lines 16-238), `_model/scnet_masked.py` (lines 15-237), and
`_model/scnet_tran.py` (lines 285-507) -- roughly 222 lines triplicated
rather than shared. `FeatureConversion` is duplicated a second way:
`_model/scnet_tran.py` (lines 167-197) carries its own copy of the class
also defined in `_model/separation.py` (lines 12-42). A real audit
(fable-reviewed 2026-07-30) confirmed every one of these copies is
byte-identical today (`inspect.getsource` diffed empty, before this test
existed).

This triplication is deliberate and *not* fixed by this test: extracting
the trunk into one shared module is a live-code restructuring left for a
post-merge verbatim extraction (dedup deferred, drift locked). Left
unchecked, a one-sided edit to any copy -- a bug
fix applied to `scnet.py`'s `SDlayer` but forgotten in `scnet_tran.py`'s --
would silently desync the three families' shared trunk behaviour with no
signal until numeric parity broke somewhere downstream. This test makes
that drift fail loudly and immediately instead: it reads each class's
source via `inspect.getsource` (cleaner than hand-slicing line ranges,
which would rot the moment any file gains or loses a line above the
target class), normalizes incidental whitespace, and asserts every copy is
still identical to the reference (`scnet.py`, and `separation.py` for
`FeatureConversion`).
"""

from __future__ import annotations

import inspect

import pytest

from scnet_infer._model import scnet, scnet_masked, scnet_tran, separation

TRUNK_CLASS_NAMES = (
    "Swish",
    "ConvolutionModule",
    "FusionLayer",
    "SDlayer",
    "SUlayer",
    "SDblock",
)

_MODULES = {
    "scnet": scnet,
    "scnet_masked": scnet_masked,
    "scnet_tran": scnet_tran,
}


def _normalized_source(cls) -> str:
    """Source text with incidental trailing whitespace stripped per line.

    Normalizing (rather than comparing raw text) means this test pins the
    code the copies actually run, not accidental formatting noise -- while
    still failing on any real line, ordering, or logic difference.
    """

    return "\n".join(line.rstrip() for line in inspect.getsource(cls).strip().splitlines())


@pytest.mark.parametrize("class_name", TRUNK_CLASS_NAMES)
@pytest.mark.parametrize("other_module_name", ["scnet_masked", "scnet_tran"])
def test_trunk_class_is_byte_identical_to_scnet(class_name, other_module_name):
    reference = _normalized_source(getattr(scnet, class_name))
    other = _normalized_source(getattr(_MODULES[other_module_name], class_name))
    assert other == reference, (
        f"{class_name} has drifted between scnet.py and {other_module_name}.py -- "
        "the trunk is deliberately triplicated pending a post-merge verbatim "
        "extraction; a one-sided edit must be reconciled across all three copies "
        "(or promoted into that extraction), not left to drift silently."
    )


def test_feature_conversion_is_byte_identical_between_scnet_tran_and_separation():
    tran_copy = _normalized_source(scnet_tran.FeatureConversion)
    separation_copy = _normalized_source(separation.FeatureConversion)
    assert tran_copy == separation_copy, (
        "FeatureConversion has drifted between scnet_tran.py and separation.py -- "
        "these are two copies of the same class, deliberately left duplicated "
        "pending a post-merge verbatim extraction; a one-sided edit must be "
        "reconciled across both copies, not left to drift silently."
    )
