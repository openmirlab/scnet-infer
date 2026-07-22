"""Scaffold smoke tests.

These checks prove package metadata and public imports agree from commit one.
Reads: scnet_infer public facade.
"""

import scnet_infer


def test_version_and_public_symbols() -> None:
    assert scnet_infer.__version__ == "0.1.0"
    assert scnet_infer.SCNetSession.__name__ == "SCNetSession"
    assert callable(scnet_infer.separate)

