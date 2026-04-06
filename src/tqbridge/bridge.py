"""Bridge controller: compress -> DMA -> decompress pipeline for KV cache transfer.

Orchestrates the end-to-end pipeline:
1. Compress KV tensors on source device (Metal) using tinygrad ops
2. Transfer compressed payload over TB5 via Tensor.to()
3. Decompress on destination device (NV)

Supports per-layer streaming with metrics collection.
"""

from __future__ import annotations

from tinygrad import Tensor, Device

from tqbridge.compression_tg import TinygradCompressor
from tqbridge.dma import DMAManager
from tqbridge.metrics import Timer, TransferMetrics, PipelineMetrics
from tqbridge.wire import Format


class KVBridge:
    """Cross-device KV cache bridge with TurboQuant compression."""

    def __init__(
        self,
        head_dim: int = 128,
        fmt_k: Format = Format.Q8_0,
        fmt_v: Format = Format.TURBO3,
        seed: int = 42,
        src_device: str = "METAL",
        dst_device: str = "NV",
    ):
        self.head_dim = head_dim
        self.fmt_k = fmt_k
        self.fmt_v = fmt_v
        self.compressor = TinygradCompressor(head_dim=head_dim, seed=seed)
        self.dma = DMAManager(src_device=src_device, dst_device=dst_device)
        self.src_device = src_device
        self.dst_device = dst_device

    def transfer_layer(
        self,
        k_cache: Tensor,
        v_cache: Tensor,
        layer_idx: int = 0,
    ) -> tuple[Tensor, Tensor, TransferMetrics]:
        """Compress, transfer, and decompress one KV layer.

        Args:
            k_cache: Key tensor shape (..., head_dim) on src_device
            v_cache: Value tensor shape (..., head_dim) on src_device
            layer_idx: layer index for metrics

        Returns:
            (k_out, v_out, metrics) where k_out/v_out are on dst_device
        """
        original_bytes = k_cache.numpy().nbytes + v_cache.numpy().nbytes

        # Step 1: Compress on source device
        with Timer() as t_compress:
            k_compressed = self.compressor.compress(k_cache, self.fmt_k)
            v_compressed = self.compressor.compress(v_cache, self.fmt_v)

        compressed_bytes = (
            self.compressor.compressed_size_bytes(k_compressed)
            + self.compressor.compressed_size_bytes(v_compressed)
        )

        # Step 2: Transfer compressed tensors to destination
        k_transferred, k_ms = self.dma.transfer_dict(k_compressed)
        v_transferred, v_ms = self.dma.transfer_dict(v_compressed)
        transfer_ms = k_ms + v_ms

        # Step 3: Decompress on destination device
        with Timer() as t_decompress:
            k_out = self.compressor.decompress(k_transferred)
            v_out = self.compressor.decompress(v_transferred)

        metrics = TransferMetrics(
            layer_idx=layer_idx,
            compress_time_ms=t_compress.ms,
            transfer_time_ms=transfer_ms,
            decompress_time_ms=t_decompress.ms,
            original_bytes=original_bytes,
            compressed_bytes=compressed_bytes,
        )

        return k_out, v_out, metrics

    def transfer_kv_cache(
        self,
        k_cache: Tensor,
        v_cache: Tensor,
        layer_range: range | None = None,
    ) -> tuple[list[Tensor], list[Tensor], PipelineMetrics]:
        """Transfer multiple KV cache layers sequentially.

        Args:
            k_cache: shape (n_layers, n_heads, seq_len, head_dim) on src
            v_cache: shape (n_layers, n_heads, seq_len, head_dim) on src
            layer_range: which layers to transfer (default: all)

        Returns:
            (k_layers_out, v_layers_out, pipeline_metrics)
        """
        n_layers = k_cache.shape[0]
        if layer_range is None:
            layer_range = range(n_layers)

        pipeline = PipelineMetrics()
        k_out_layers = []
        v_out_layers = []

        for layer_idx in layer_range:
            # Extract single layer: (n_heads, seq_len, head_dim)
            k_layer = k_cache[layer_idx].realize()
            v_layer = v_cache[layer_idx].realize()

            # Reshape to (n_vectors, head_dim) for compression
            orig_shape = k_layer.shape
            k_flat = k_layer.reshape(-1, self.head_dim)
            v_flat = v_layer.reshape(-1, self.head_dim)

            k_out, v_out, metrics = self.transfer_layer(k_flat, v_flat, layer_idx)

            # Reshape back
            k_out_layers.append(k_out.reshape(*orig_shape))
            v_out_layers.append(v_out.reshape(*orig_shape))
            pipeline.add(metrics)

        return k_out_layers, v_out_layers, pipeline
