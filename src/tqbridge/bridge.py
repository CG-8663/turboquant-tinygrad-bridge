"""Bridge controller: compress -> DMA -> decompress pipeline for KV cache transfer.

Orchestrates the end-to-end pipeline:
1. Compress KV tensors on source device (Metal) using tinygrad ops
2. Transfer compressed payload over TB5 via Tensor.to()
3. Decompress on destination device (NV)

Supports both sequential and pipelined (double-buffer) modes.
Thermal monitoring with throttle gating and optional CLI display.
"""

from __future__ import annotations

import os
import threading

from tinygrad import Tensor, Device

from tqbridge.compression_tg import TinygradCompressor
from tqbridge.dma import DMAManager, RingBuffer
from tqbridge.metrics import Timer, TransferMetrics, PipelineMetrics
from tqbridge.thermal import ThermalMonitor, print_thermal_header, print_thermal_row, print_thermal_footer
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
        beam: int | None = None,
        thermal: bool = False,
        thermal_interval_s: float = 2.0,
        temp_limit_c: float = 85.0,
        show_thermal: bool = False,
    ):
        self.head_dim = head_dim
        self.fmt_k = fmt_k
        self.fmt_v = fmt_v
        self.beam = beam
        self.show_thermal = show_thermal
        if beam is not None:
            os.environ["JITBEAM"] = str(beam)
        self.compressor = TinygradCompressor(head_dim=head_dim, seed=seed)
        self.dma = DMAManager(src_device=src_device, dst_device=dst_device)
        self.src_device = src_device
        self.dst_device = dst_device

        # Thermal monitor
        self.thermal_monitor: ThermalMonitor | None = None
        if thermal or show_thermal:
            self.thermal_monitor = ThermalMonitor(
                interval_s=thermal_interval_s,
                temp_limit_c=temp_limit_c,
            )
            self.thermal_monitor.start()

    def close(self) -> None:
        """Stop thermal monitor."""
        if self.thermal_monitor is not None:
            self.thermal_monitor.stop()

    def __del__(self):
        self.close()

    def _thermal_gate(self) -> None:
        """Block if thermal throttle is active."""
        if self.thermal_monitor is not None:
            self.thermal_monitor.wait_if_throttled()

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
        self._thermal_gate()

        n_elements = 1
        for s in k_cache.shape:
            n_elements *= s
        original_bytes = n_elements * 4 * 2  # K + V, float32

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

        if self.show_thermal:
            print_thermal_header()

        with Timer() as t_wall:
            for layer_idx in layer_range:
                self._thermal_gate()

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

                if self.show_thermal:
                    snap = self.thermal_monitor.latest if self.thermal_monitor else None
                    print_thermal_row(
                        layer_idx, metrics.compress_time_ms, metrics.transfer_time_ms,
                        metrics.decompress_time_ms, metrics.compression_ratio, snap,
                    )

        pipeline.wall_time_ms = t_wall.ms

        if self.show_thermal:
            snap = self.thermal_monitor.latest if self.thermal_monitor else None
            print_thermal_footer(pipeline.summary(), snap)

        return k_out_layers, v_out_layers, pipeline

    def transfer_kv_cache_pipelined(
        self,
        k_cache: Tensor,
        v_cache: Tensor,
        layer_range: range | None = None,
        ring_slots: int = 4,
    ) -> tuple[list[Tensor], list[Tensor], PipelineMetrics]:
        """Transfer KV cache with double-buffered pipeline.

        Producer thread: compress + transfer each layer
        Consumer thread: decompress each layer on destination
        Overlap: compress+transfer of layer N+1 runs while layer N decompresses.

        Args:
            k_cache: shape (n_layers, n_heads, seq_len, head_dim) on src
            v_cache: shape (n_layers, n_heads, seq_len, head_dim) on src
            layer_range: which layers to transfer (default: all)
            ring_slots: ring buffer depth (default 4)

        Returns:
            (k_layers_out, v_layers_out, pipeline_metrics)
        """
        n_layers = k_cache.shape[0]
        if layer_range is None:
            layer_range = range(n_layers)
        layers = list(layer_range)

        ring = RingBuffer(slots=ring_slots)
        pipeline = PipelineMetrics()
        results: dict[int, tuple[Tensor, Tensor]] = {}
        producer_error: list[Exception] = []

        if self.show_thermal:
            print_thermal_header()

        def _producer():
            try:
                for layer_idx in layers:
                    self._thermal_gate()

                    k_layer = k_cache[layer_idx].realize()
                    v_layer = v_cache[layer_idx].realize()
                    orig_shape = k_layer.shape
                    k_flat = k_layer.reshape(-1, self.head_dim)
                    v_flat = v_layer.reshape(-1, self.head_dim)

                    # Estimate original size without .numpy() (avoid device sync in producer)
                    n_elements = 1
                    for s in k_flat.shape:
                        n_elements *= s
                    original_bytes = n_elements * 4 * 2  # K + V, float32

                    # Compress on source device
                    with Timer() as t_compress:
                        k_comp = self.compressor.compress(k_flat, self.fmt_k)
                        v_comp = self.compressor.compress(v_flat, self.fmt_v)

                    compressed_bytes = (
                        self.compressor.compressed_size_bytes(k_comp)
                        + self.compressor.compressed_size_bytes(v_comp)
                    )

                    # Transfer to destination
                    k_xfer, k_ms = self.dma.transfer_dict(k_comp)
                    v_xfer, v_ms = self.dma.transfer_dict(v_comp)

                    ring.put({
                        "layer_idx": layer_idx,
                        "k_xfer": k_xfer,
                        "v_xfer": v_xfer,
                        "orig_shape": orig_shape,
                        "compress_ms": t_compress.ms,
                        "transfer_ms": k_ms + v_ms,
                        "original_bytes": original_bytes,
                        "compressed_bytes": compressed_bytes,
                    })
            except Exception as e:
                producer_error.append(e)
            finally:
                ring.done()

        # Start producer thread
        producer = threading.Thread(target=_producer, daemon=True)

        with Timer() as t_wall:
            producer.start()

            # Consumer: decompress on destination device (main thread)
            while True:
                item = ring.get()
                if RingBuffer.is_sentinel(item):
                    break

                with Timer() as t_decompress:
                    k_out = self.compressor.decompress(item["k_xfer"])
                    v_out = self.compressor.decompress(item["v_xfer"])

                layer_idx = item["layer_idx"]
                results[layer_idx] = (
                    k_out.reshape(*item["orig_shape"]),
                    v_out.reshape(*item["orig_shape"]),
                )

                metrics = TransferMetrics(
                    layer_idx=layer_idx,
                    compress_time_ms=item["compress_ms"],
                    transfer_time_ms=item["transfer_ms"],
                    decompress_time_ms=t_decompress.ms,
                    original_bytes=item["original_bytes"],
                    compressed_bytes=item["compressed_bytes"],
                )
                pipeline.add(metrics)

                if self.show_thermal:
                    snap = self.thermal_monitor.latest if self.thermal_monitor else None
                    print_thermal_row(
                        layer_idx, metrics.compress_time_ms, metrics.transfer_time_ms,
                        metrics.decompress_time_ms, metrics.compression_ratio, snap,
                    )

            producer.join()
            if producer_error:
                raise producer_error[0]

        pipeline.wall_time_ms = t_wall.ms

        if self.show_thermal:
            snap = self.thermal_monitor.latest if self.thermal_monitor else None
            print_thermal_footer(pipeline.summary(), snap)

        # Return in layer order
        k_out_layers = [results[i][0] for i in layers]
        v_out_layers = [results[i][1] for i in layers]
        return k_out_layers, v_out_layers, pipeline
