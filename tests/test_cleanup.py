"""Tests for safe cleanup of KVBridge and NativeBridge.

These tests verify that:
  1. __del__ is safe to call on partially-constructed objects (no AttributeError
     when __init__ raises before attributes are assigned).
  2. close() cleans up ALL resources even if one cleanup step raises (no leaks
     if thermal.stop() or native.close() throws).
  3. close() re-raises errors to explicit callers but __del__ swallows them.

These tests do not require any hardware (NV or Metal).
"""

import sys
import gc
import warnings
import pytest

from tqbridge.bridge import KVBridge
from tqbridge.wire import Format


# ---------------------------------------------------------------------------
# Helpers — fake resources that can be injected / made to fail
# ---------------------------------------------------------------------------

class _FakeResource:
    """Pretends to be a native/cuda compressor or thermal monitor."""

    def __init__(self, raise_on_close: bool = False, name: str = "fake"):
        self.closed = False
        self.stopped = False
        self.raise_on_close = raise_on_close
        self.name = name

    def close(self):
        if self.raise_on_close:
            raise RuntimeError(f"{self.name} close() failed")
        self.closed = True

    def stop(self):
        if self.raise_on_close:
            raise RuntimeError(f"{self.name} stop() failed")
        self.stopped = True


class _MinimalBridge:
    """A KVBridge-like object with only the close() method under test.

    Constructing a real KVBridge requires either NV or Metal devices.
    This shim bypasses __init__ and exercises close() directly against the
    exact code path from KVBridge.close().
    """

    def __init__(self, thermal=None, native=None, cuda=None):
        self.thermal_monitor = thermal
        self._native_compressor = native
        self._cuda_compressor = cuda

    # Import the real close() method from KVBridge so we test what ships.
    close = KVBridge.close
    __del__ = KVBridge.__del__


# ---------------------------------------------------------------------------
# AttributeError on partial __init__ — the original bug
# ---------------------------------------------------------------------------

def test_del_on_partial_init_does_not_raise():
    """__del__ on an object missing expected attrs should not AttributeError.

    This reproduces the original bug from the pytest warning:
      'NativeBridge' object has no attribute '_ptr'
      'KVBridge' object has no attribute 'thermal_monitor'
    """
    # Create a bare instance without calling __init__
    bridge = _MinimalBridge.__new__(_MinimalBridge)
    # None of the attrs exist at all
    assert not hasattr(bridge, "thermal_monitor")
    assert not hasattr(bridge, "_native_compressor")
    assert not hasattr(bridge, "_cuda_compressor")

    # close() must not raise AttributeError
    bridge.close()

    # __del__ must also not raise (would normally warn via
    # PytestUnraisableExceptionWarning)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        del bridge
        gc.collect()


def test_del_on_none_attributes_is_safe():
    """All attrs present but None — close() must not fail."""
    bridge = _MinimalBridge(thermal=None, native=None, cuda=None)
    bridge.close()  # no-op, no errors


def test_native_bridge_del_without_init():
    """NativeBridge.__del__ on a fresh __new__() instance is safe.

    Directly tests NativeBridge (no hardware required — we never actually
    call the real __init__ that loads the C library).
    """
    from tqbridge.native import NativeBridge
    bridge = NativeBridge.__new__(NativeBridge)
    # The bug: neither _lib nor _ptr are set
    assert not hasattr(bridge, "_ptr")
    assert not hasattr(bridge, "_lib")

    # Must not raise
    bridge.close()

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        del bridge
        gc.collect()


def test_native_bridge_close_happy_path_with_mock_lib():
    """Test the close() happy path: lib.tq_bridge_free is actually called.

    Uses a mock lib to avoid requiring the real C library.
    """
    import ctypes
    from tqbridge.native import NativeBridge

    class _MockLib:
        def __init__(self):
            self.free_called_with = None

        def tq_bridge_free(self, ptr):
            self.free_called_with = ptr

    bridge = NativeBridge.__new__(NativeBridge)
    bridge._lib = _MockLib()
    bridge._ptr = ctypes.c_void_p(0xDEADBEEF)

    bridge.close()

    # tq_bridge_free was called with the pointer
    assert bridge._lib.free_called_with is not None
    # _ptr was reset
    assert not bridge._ptr  # c_void_p() is falsy


def test_native_bridge_del_swallows_close_errors():
    """NativeBridge.__del__ must swallow exceptions from close()."""
    import ctypes
    from tqbridge.native import NativeBridge

    class _FailingLib:
        def tq_bridge_free(self, ptr):
            raise RuntimeError("simulated free failure")

    bridge = NativeBridge.__new__(NativeBridge)
    bridge._lib = _FailingLib()
    bridge._ptr = ctypes.c_void_p(0xDEADBEEF)

    # Explicit close() raises
    with pytest.raises(RuntimeError, match="simulated free failure"):
        bridge.close()

    # But __del__ must not propagate
    bridge._lib = _FailingLib()
    bridge._ptr = ctypes.c_void_p(0xDEADBEEF)
    del bridge
    gc.collect()
    # Test passes if no exception escapes gc


# ---------------------------------------------------------------------------
# Resource leak prevention — all resources cleaned up even on errors
# ---------------------------------------------------------------------------

def test_close_cleans_all_resources_on_success():
    thermal = _FakeResource(name="thermal")
    native = _FakeResource(name="native")
    cuda = _FakeResource(name="cuda")

    bridge = _MinimalBridge(thermal=thermal, native=native, cuda=cuda)
    bridge.close()

    assert thermal.stopped
    assert native.closed
    assert cuda.closed

    # Attributes should be cleared to None after close
    assert bridge.thermal_monitor is None
    assert bridge._native_compressor is None
    assert bridge._cuda_compressor is None


def test_close_continues_when_thermal_stop_raises():
    """If thermal.stop() raises, native/cuda must still be cleaned up.

    The failing resource's reference is preserved so callers can inspect
    or retry. Successfully-closed resources are cleared to None.
    """
    thermal = _FakeResource(raise_on_close=True, name="thermal")
    native = _FakeResource(name="native")
    cuda = _FakeResource(name="cuda")

    bridge = _MinimalBridge(thermal=thermal, native=native, cuda=cuda)

    with pytest.raises(RuntimeError, match="thermal stop"):
        bridge.close()

    # Critically: native and cuda were still cleaned up
    assert native.closed, "native was leaked when thermal.stop() raised"
    assert cuda.closed, "cuda was leaked when thermal.stop() raised"

    # Successful resources cleared; failing resource preserved for diagnosis
    assert bridge.thermal_monitor is thermal, "failed resource should be preserved"
    assert bridge._native_compressor is None
    assert bridge._cuda_compressor is None


def test_close_continues_when_native_close_raises():
    """If native.close() raises, cuda must still be cleaned up."""
    thermal = _FakeResource(name="thermal")
    native = _FakeResource(raise_on_close=True, name="native")
    cuda = _FakeResource(name="cuda")

    bridge = _MinimalBridge(thermal=thermal, native=native, cuda=cuda)

    with pytest.raises(RuntimeError, match="native close"):
        bridge.close()

    assert thermal.stopped
    assert cuda.closed, "cuda was leaked when native.close() raised"

    # Successful resources cleared; failing resource preserved
    assert bridge.thermal_monitor is None
    assert bridge._native_compressor is native, "failed resource should be preserved"
    assert bridge._cuda_compressor is None


def test_close_continues_when_cuda_close_raises():
    """If cuda.close() raises, thermal and native already ran."""
    thermal = _FakeResource(name="thermal")
    native = _FakeResource(name="native")
    cuda = _FakeResource(raise_on_close=True, name="cuda")

    bridge = _MinimalBridge(thermal=thermal, native=native, cuda=cuda)

    with pytest.raises(RuntimeError, match="cuda close"):
        bridge.close()

    assert thermal.stopped
    assert native.closed


def test_close_all_three_raising_reports_first_error():
    """If every resource raises, the first error propagates and all
    failing references are preserved for diagnosis."""
    thermal = _FakeResource(raise_on_close=True, name="thermal")
    native = _FakeResource(raise_on_close=True, name="native")
    cuda = _FakeResource(raise_on_close=True, name="cuda")

    bridge = _MinimalBridge(thermal=thermal, native=native, cuda=cuda)

    with pytest.raises(RuntimeError, match="thermal stop"):
        bridge.close()

    # All references preserved — all three failed, none should be cleared
    assert bridge.thermal_monitor is thermal
    assert bridge._native_compressor is native
    assert bridge._cuda_compressor is cuda


def test_retry_close_after_transient_failure():
    """Callers should be able to retry close() after a transient failure."""
    thermal = _FakeResource(raise_on_close=True, name="thermal")
    native = _FakeResource(name="native")

    bridge = _MinimalBridge(thermal=thermal, native=native, cuda=None)

    # First close: thermal fails, native succeeds
    with pytest.raises(RuntimeError, match="thermal stop"):
        bridge.close()
    assert native.closed
    assert bridge._native_compressor is None
    assert bridge.thermal_monitor is thermal  # preserved

    # Fix the transient issue and retry
    thermal.raise_on_close = False
    bridge.close()  # should succeed now
    assert thermal.stopped
    assert bridge.thermal_monitor is None


def test_double_close_is_safe():
    """Calling close() twice must not raise."""
    thermal = _FakeResource(name="thermal")
    native = _FakeResource(name="native")

    bridge = _MinimalBridge(thermal=thermal, native=native, cuda=None)
    bridge.close()
    # Second call must be a no-op — attributes are now None
    bridge.close()


# ---------------------------------------------------------------------------
# __del__ swallows exceptions — it must never raise into GC
# ---------------------------------------------------------------------------

def test_del_swallows_cleanup_errors():
    """__del__ must not propagate exceptions from close().

    Python's garbage collector logs but does not re-raise exceptions from
    __del__. We verify this by forcing close() to raise and confirming
    no exception escapes del/gc.
    """
    thermal = _FakeResource(raise_on_close=True, name="thermal")
    native = _FakeResource(raise_on_close=True, name="native")

    bridge = _MinimalBridge(thermal=thermal, native=native, cuda=None)

    # Trigger __del__ by removing our last reference
    # Python logs errors from __del__ but does not re-raise
    del bridge
    gc.collect()
    # If we got here, __del__ didn't crash the interpreter
