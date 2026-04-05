"""Smoke tests: verify package imports and basic structure."""

import tqbridge
from tqbridge import wire, compression, dma, bridge, thermal, metrics
from tqbridge.kernels import metal, cuda


def test_version():
    assert tqbridge.__version__ == "0.1.0"


def test_modules_importable():
    """All tqbridge submodules should be importable."""
    assert wire is not None
    assert compression is not None
    assert dma is not None
    assert bridge is not None
    assert thermal is not None
    assert metrics is not None
    assert metal is not None
    assert cuda is not None
