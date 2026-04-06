"""Tests for the KV bridge pipeline (bridge.py + dma.py).

Tests marked @pytest.mark.hardware require both Metal and NV devices live.
Run with: pytest tests/test_bridge.py -m hardware
"""

import pytest
import numpy as np

tinygrad = pytest.importorskip("tinygrad")
from tinygrad import Tensor, Device

from tqbridge.bridge import KVBridge
from tqbridge.dma import DMAManager
from tqbridge.metrics import TransferMetrics, PipelineMetrics
from tqbridge.wire import Format


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_nv():
    try:
        Device["NV"]
        return True
    except Exception:
        return False


def _has_metal():
    try:
        Device["METAL"]
        return True
    except Exception:
        return False


hardware = pytest.mark.skipif(
    not (_has_nv() and _has_metal()),
    reason="Requires both Metal and NV devices",
)


# ---------------------------------------------------------------------------
# DMA tests
# ---------------------------------------------------------------------------

@hardware
class TestDMA:
    def test_transfer_metal_to_nv(self):
        """Transfer a tensor from Metal to NV and verify values."""
        dma = DMAManager(src_device="METAL", dst_device="NV")
        x = Tensor.rand(1024, device="METAL").realize()
        x_np = x.numpy()

        y, ms = dma.transfer(x)
        assert y.device == "NV"
        np.testing.assert_allclose(x_np, y.numpy(), atol=1e-6)
        assert ms > 0

    def test_transfer_nv_to_metal(self):
        """Transfer a tensor from NV to Metal and verify values."""
        dma = DMAManager(src_device="NV", dst_device="METAL")
        x = Tensor.rand(1024, device="NV").realize()
        x_np = x.numpy()

        y, ms = dma.transfer(x)
        assert y.device == "METAL"
        np.testing.assert_allclose(x_np, y.numpy(), atol=1e-6)

    def test_transfer_dict(self):
        """Transfer a dict of tensors, preserving non-tensor values."""
        dma = DMAManager(src_device="METAL", dst_device="NV")
        d = {
            "tensor_a": Tensor.rand(64, device="METAL").realize(),
            "tensor_b": Tensor.rand(32, device="METAL").realize(),
            "scalar": 42,
            "string": "hello",
        }

        result, ms = dma.transfer_dict(d)
        assert result["tensor_a"].device == "NV"
        assert result["tensor_b"].device == "NV"
        assert result["scalar"] == 42
        assert result["string"] == "hello"
        assert ms > 0

    def test_transfer_round_trip(self):
        """Metal -> NV -> Metal should preserve values."""
        x = Tensor.rand(4096, device="METAL").realize()
        x_np = x.numpy()

        dma_to_nv = DMAManager("METAL", "NV")
        dma_to_metal = DMAManager("NV", "METAL")

        y, _ = dma_to_nv.transfer(x)
        z, _ = dma_to_metal.transfer(y)

        assert z.device == "METAL"
        np.testing.assert_allclose(x_np, z.numpy(), atol=1e-6)

    def test_transfer_large(self):
        """Transfer 16 MB tensor and measure bandwidth."""
        n = 16 * 1024 * 1024 // 4  # 16 MB float32
        dma = DMAManager("METAL", "NV")
        x = Tensor.rand(n, device="METAL").realize()

        y, ms = dma.transfer(x)
        bw_gbps = (n * 4 / 1e9) / (ms / 1e3)
        print(f"  16 MB transfer: {ms:.1f} ms, {bw_gbps:.2f} GB/s")
        assert ms > 0
        assert y.shape == x.shape


# ---------------------------------------------------------------------------
# Bridge single-layer tests
# ---------------------------------------------------------------------------

@hardware
class TestBridgeSingleLayer:
    def test_transfer_layer_turbo3(self):
        """Single layer transfer with turbo3 compression."""
        bridge = KVBridge(head_dim=128, fmt_k=Format.Q8_0, fmt_v=Format.TURBO3)

        k = Tensor.rand(8, 128, device="METAL").realize()
        v = Tensor.rand(8, 128, device="METAL").realize()
        k_np, v_np = k.numpy(), v.numpy()

        k_out, v_out, metrics = bridge.transfer_layer(k, v)

        # On NV device
        assert k_out.device == "NV"
        assert v_out.device == "NV"

        # Shape preserved
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape

        # Quality: K (q8_0) should be very close, V (turbo3) has more error
        k_mse = np.mean((k_np - k_out.numpy()) ** 2)
        v_mse = np.mean((v_np - v_out.numpy()) ** 2)
        assert k_mse < 1e-4, f"K MSE too high: {k_mse}"
        assert v_mse < 0.05, f"V MSE too high: {v_mse}"

    def test_transfer_layer_turbo4(self):
        """Single layer transfer with turbo4 compression."""
        bridge = KVBridge(head_dim=128, fmt_k=Format.TURBO4, fmt_v=Format.TURBO4)

        k = Tensor.rand(16, 128, device="METAL").realize()
        v = Tensor.rand(16, 128, device="METAL").realize()

        k_out, v_out, metrics = bridge.transfer_layer(k, v)

        assert k_out.device == "NV"
        assert v_out.device == "NV"
        assert metrics.compression_ratio > 2.5

    def test_metrics_populated(self):
        """Verify all metrics fields are populated after transfer."""
        bridge = KVBridge(head_dim=128, fmt_k=Format.Q8_0, fmt_v=Format.TURBO3)

        k = Tensor.rand(8, 128, device="METAL").realize()
        v = Tensor.rand(8, 128, device="METAL").realize()

        _, _, metrics = bridge.transfer_layer(k, v)

        assert metrics.compress_time_ms > 0
        assert metrics.transfer_time_ms > 0
        assert metrics.decompress_time_ms > 0
        assert metrics.original_bytes > 0
        assert metrics.compressed_bytes > 0
        assert metrics.compression_ratio > 1.0
        assert metrics.total_time_ms > 0

    def test_asymmetric_kv(self):
        """Asymmetric: q8_0 K + turbo3 V should have different error profiles."""
        bridge = KVBridge(head_dim=128, fmt_k=Format.Q8_0, fmt_v=Format.TURBO3)

        data = Tensor.rand(16, 128, device="METAL").realize()
        data_np = data.numpy()

        k_out, v_out, _ = bridge.transfer_layer(data, data)

        k_mse = np.mean((data_np - k_out.numpy()) ** 2)
        v_mse = np.mean((data_np - v_out.numpy()) ** 2)

        # K (q8_0) should be much more precise than V (turbo3)
        assert k_mse < v_mse / 10, f"K MSE {k_mse} should be << V MSE {v_mse}"


# ---------------------------------------------------------------------------
# Bridge multi-layer tests
# ---------------------------------------------------------------------------

@hardware
class TestBridgeMultiLayer:
    def test_multi_layer_transfer(self):
        """Transfer 4 KV cache layers and verify all arrive on NV."""
        n_layers, n_heads, seq_len, head_dim = 4, 4, 32, 128
        bridge = KVBridge(head_dim=head_dim, fmt_k=Format.Q8_0, fmt_v=Format.TURBO3)

        k_cache = Tensor.rand(n_layers, n_heads, seq_len, head_dim, device="METAL").realize()
        v_cache = Tensor.rand(n_layers, n_heads, seq_len, head_dim, device="METAL").realize()

        k_layers, v_layers, pipeline = bridge.transfer_kv_cache(k_cache, v_cache)

        assert len(k_layers) == n_layers
        assert len(v_layers) == n_layers
        assert len(pipeline.layers) == n_layers

        for i in range(n_layers):
            assert k_layers[i].device == "NV"
            assert v_layers[i].device == "NV"
            assert k_layers[i].shape == (n_heads, seq_len, head_dim)

    def test_layer_range(self):
        """Transfer only a subset of layers."""
        n_layers, n_heads, seq_len, head_dim = 8, 4, 16, 128
        bridge = KVBridge(head_dim=head_dim, fmt_k=Format.Q8_0, fmt_v=Format.TURBO3)

        k_cache = Tensor.rand(n_layers, n_heads, seq_len, head_dim, device="METAL").realize()
        v_cache = Tensor.rand(n_layers, n_heads, seq_len, head_dim, device="METAL").realize()

        # Only transfer layers 2-5
        k_layers, v_layers, pipeline = bridge.transfer_kv_cache(
            k_cache, v_cache, layer_range=range(2, 6)
        )

        assert len(k_layers) == 4
        assert len(pipeline.layers) == 4

    def test_pipeline_metrics_summary(self):
        """Pipeline summary should report aggregate stats."""
        n_layers, n_heads, seq_len, head_dim = 2, 2, 16, 128
        bridge = KVBridge(head_dim=head_dim, fmt_k=Format.Q8_0, fmt_v=Format.TURBO3)

        k_cache = Tensor.rand(n_layers, n_heads, seq_len, head_dim, device="METAL").realize()
        v_cache = Tensor.rand(n_layers, n_heads, seq_len, head_dim, device="METAL").realize()

        _, _, pipeline = bridge.transfer_kv_cache(k_cache, v_cache)

        summary = pipeline.summary()
        assert "2 layers" in summary
        assert "compression" in summary
        assert pipeline.avg_compression_ratio > 1.0
        assert pipeline.total_time_ms > 0


# ---------------------------------------------------------------------------
# Metrics unit tests (no hardware required)
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_transfer_metrics_ratio(self):
        m = TransferMetrics(
            layer_idx=0,
            compress_time_ms=1.0,
            transfer_time_ms=2.0,
            decompress_time_ms=0.5,
            original_bytes=4096,
            compressed_bytes=1024,
        )
        assert m.compression_ratio == 4.0
        assert m.total_time_ms == 3.5

    def test_transfer_metrics_bandwidth(self):
        m = TransferMetrics(
            layer_idx=0,
            transfer_time_ms=10.0,  # 10ms
            original_bytes=60_000_000,  # 60 MB
        )
        # 60 MB / 10ms = 6 GB/s
        assert abs(m.effective_bandwidth_gbps - 6.0) < 0.1

    def test_pipeline_metrics(self):
        pipeline = PipelineMetrics()
        pipeline.add(TransferMetrics(0, 1, 2, 1, 4096, 1024))
        pipeline.add(TransferMetrics(1, 1, 2, 1, 4096, 1024))

        assert pipeline.total_time_ms == 8.0
        assert pipeline.avg_compression_ratio == 4.0
        assert "2 layers" in pipeline.summary()

    def test_empty_pipeline(self):
        pipeline = PipelineMetrics()
        assert pipeline.total_time_ms == 0.0
        assert "No layers" in pipeline.summary()
