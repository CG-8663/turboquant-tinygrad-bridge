"""Bridge controller: compress -> DMA -> decompress pipeline for KV cache transfer.

Orchestrates the end-to-end pipeline:
1. Compress KV tensors on source device using tinygrad ops or native C
2. Transfer compressed payload over TB5 via Tensor.to()
3. Decompress on destination device

Supports both sequential and pipelined (double-buffer) modes.
Backend selection: "tinygrad" (on-device tensor ops) or "native" (C via ctypes).
Thermal monitoring with throttle gating and optional CLI display.
"""

from __future__ import annotations

import os
import threading
from typing import Literal

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
        backend: Literal["tinygrad", "native"] = "tinygrad",
    ):
        self.head_dim = head_dim
        self.fmt_k = fmt_k
        self.fmt_v = fmt_v
        self.beam = beam
        self.show_thermal = show_thermal
        self.backend = backend
        if beam is not None:
            os.environ["JITBEAM"] = str(beam)
        self.compressor = TinygradCompressor(head_dim=head_dim, seed=seed)
        self.dma = DMAManager(src_device=src_device, dst_device=dst_device)
        self.src_device = src_device
        self.dst_device = dst_device

        # Native C compressor (lazy init — only loaded when backend="native")
        self._native_compressor = None
        self._seed = seed
        if backend == "native":
            from tqbridge.native import NativeCompressor
            self._native_compressor = NativeCompressor(head_dim=head_dim, seed=seed)

        # Thermal monitor
        self.thermal_monitor: ThermalMonitor | None = None
        if thermal or show_thermal:
            self.thermal_monitor = ThermalMonitor(
                interval_s=thermal_interval_s,
                temp_limit_c=temp_limit_c,
            )
            self.thermal_monitor.start()

    def close(self) -> None:
        """Stop thermal monitor and release native resources."""
        if self.thermal_monitor is not None:
            self.thermal_monitor.stop()
        if self._native_compressor is not None:
            self._native_compressor.close()
            self._native_compressor = None

    def __del__(self):
        self.close()

    def _thermal_gate(self) -> None:
        """Block if thermal throttle is active."""
        if self.thermal_monitor is not None:
            self.thermal_monitor.wait_if_throttled()

    def warmup(self, n_heads: int = 4, seq_len: int = 8) -> float:
        """Pre-compile all kernel shapes by running a dummy transfer cycle.

        Forces tinygrad to compile and cache kernels for compress, transfer,
        and decompress at the expected tensor shapes. Subsequent calls reuse
        cached kernels, eliminating first-run compilation latency.

        Args:
            n_heads: number of attention heads for warmup shape
            seq_len: sequence length for warmup shape

        Returns:
            Warmup time in milliseconds.
        """
        with Timer() as t:
            shape = (n_heads, seq_len, self.head_dim)
            k_dummy = Tensor.rand(*shape, device=self.src_device).realize()
            v_dummy = Tensor.rand(*shape, device=self.src_device).realize()
            k_flat = k_dummy.reshape(-1, self.head_dim)
            v_flat = v_dummy.reshape(-1, self.head_dim)
            self.transfer_layer(k_flat, v_flat, layer_idx=0)
        return t.ms

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

        if self.backend == "native":
            return self._transfer_layer_native(k_cache, v_cache, layer_idx)
        return self._transfer_layer_tinygrad(k_cache, v_cache, layer_idx)

    def _transfer_layer_tinygrad(
        self,
        k_cache: Tensor,
        v_cache: Tensor,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor, TransferMetrics]:
        """Compress/decompress via tinygrad tensor ops (on-device)."""
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

    def _transfer_layer_native(
        self,
        k_cache: Tensor,
        v_cache: Tensor,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor, TransferMetrics]:
        """Compress/decompress via native C library (CPU path).

        Flow: GPU→CPU (.numpy) → C compress → transfer bytes → C decompress → CPU→GPU
        The C compress/decompress is much faster than tinygrad tensor ops,
        offsetting the CPU round-trip cost for large tensors.
        """
        import numpy as np

        orig_shape = k_cache.shape
        n_elements = 1
        for s in orig_shape:
            n_elements *= s
        original_bytes = n_elements * 4 * 2

        nc = self._native_compressor

        # Step 1: Pull to CPU and compress via C
        with Timer() as t_compress:
            k_np = k_cache.reshape(-1, self.head_dim).numpy()
            v_np = v_cache.reshape(-1, self.head_dim).numpy()
            k_comp = nc.compress(k_np, self.fmt_k)
            v_comp = nc.compress(v_np, self.fmt_v)

        compressed_bytes = (
            nc.compressed_size_bytes(k_comp) + nc.compressed_size_bytes(v_comp)
        )

        # Step 2: Transfer compressed bytes as a tensor to destination device
        # For native path, the compressed data is already CPU bytes.
        # We transfer the raw bytes as a 1D tensor to the destination,
        # then pull back to CPU for C decompression.
        with Timer() as t_transfer:
            # Minimal transfer: just move the compressed bytes over the bus
            k_bytes_t = Tensor(k_comp["compressed_bytes"].astype(np.uint8)).to(self.dst_device).realize()
            v_bytes_t = Tensor(v_comp["compressed_bytes"].astype(np.uint8)).to(self.dst_device).realize()
            Device[self.dst_device].synchronize()
        transfer_ms = t_transfer.ms

        # Step 3: Decompress via C on destination side
        with Timer() as t_decompress:
            # Pull compressed bytes back to CPU for C decompression
            k_comp["compressed_bytes"] = k_bytes_t.numpy().astype(np.uint8)
            v_comp["compressed_bytes"] = v_bytes_t.numpy().astype(np.uint8)
            k_result = nc.decompress(k_comp)
            v_result = nc.decompress(v_comp)
            # Push decompressed float32 to destination device
            k_out = Tensor(k_result, device=self.dst_device).reshape(*orig_shape).realize()
            v_out = Tensor(v_result, device=self.dst_device).reshape(*orig_shape).realize()

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

        def _producer_tinygrad():
            try:
                for layer_idx in layers:
                    self._thermal_gate()

                    k_layer = k_cache[layer_idx].realize()
                    v_layer = v_cache[layer_idx].realize()
                    orig_shape = k_layer.shape
                    k_flat = k_layer.reshape(-1, self.head_dim)
                    v_flat = v_layer.reshape(-1, self.head_dim)

                    n_elements = 1
                    for s in k_flat.shape:
                        n_elements *= s
                    original_bytes = n_elements * 4 * 2

                    with Timer() as t_compress:
                        k_comp = self.compressor.compress(k_flat, self.fmt_k)
                        v_comp = self.compressor.compress(v_flat, self.fmt_v)

                    compressed_bytes = (
                        self.compressor.compressed_size_bytes(k_comp)
                        + self.compressor.compressed_size_bytes(v_comp)
                    )

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
                        "backend": "tinygrad",
                    })
            except Exception as e:
                producer_error.append(e)
            finally:
                ring.done()

        def _producer_native():
            import numpy as np
            try:
                nc = self._native_compressor
                for layer_idx in layers:
                    self._thermal_gate()

                    k_layer = k_cache[layer_idx].realize()
                    v_layer = v_cache[layer_idx].realize()
                    orig_shape = k_layer.shape

                    n_elements = 1
                    for s in orig_shape:
                        n_elements *= s
                    original_bytes = n_elements * 4 * 2

                    # Pull to CPU and compress via C
                    with Timer() as t_compress:
                        k_np = k_layer.reshape(-1, self.head_dim).numpy()
                        v_np = v_layer.reshape(-1, self.head_dim).numpy()
                        k_comp = nc.compress(k_np, self.fmt_k)
                        v_comp = nc.compress(v_np, self.fmt_v)

                    compressed_bytes = (
                        nc.compressed_size_bytes(k_comp)
                        + nc.compressed_size_bytes(v_comp)
                    )

                    # Transfer compressed bytes to destination
                    with Timer() as t_transfer:
                        k_bytes_t = Tensor(k_comp["compressed_bytes"].astype(np.uint8)).to(self.dst_device).realize()
                        v_bytes_t = Tensor(v_comp["compressed_bytes"].astype(np.uint8)).to(self.dst_device).realize()
                        Device[self.dst_device].synchronize()

                    ring.put({
                        "layer_idx": layer_idx,
                        "k_comp": k_comp,
                        "v_comp": v_comp,
                        "k_bytes_t": k_bytes_t,
                        "v_bytes_t": v_bytes_t,
                        "orig_shape": orig_shape,
                        "compress_ms": t_compress.ms,
                        "transfer_ms": t_transfer.ms,
                        "original_bytes": original_bytes,
                        "compressed_bytes": compressed_bytes,
                        "backend": "native",
                    })
            except Exception as e:
                producer_error.append(e)
            finally:
                ring.done()

        _producer = _producer_native if self.backend == "native" else _producer_tinygrad

        # Start producer thread
        producer = threading.Thread(target=_producer, daemon=True)

        with Timer() as t_wall:
            producer.start()

            # Consumer: decompress on destination device (main thread)
            while True:
                item = ring.get()
                if RingBuffer.is_sentinel(item):
                    break

                if item["backend"] == "native":
                    import numpy as np
                    nc = self._native_compressor
                    with Timer() as t_decompress:
                        item["k_comp"]["compressed_bytes"] = item["k_bytes_t"].numpy().astype(np.uint8)
                        item["v_comp"]["compressed_bytes"] = item["v_bytes_t"].numpy().astype(np.uint8)
                        k_result = nc.decompress(item["k_comp"])
                        v_result = nc.decompress(item["v_comp"])
                        k_out = Tensor(k_result, device=self.dst_device).reshape(*item["orig_shape"]).realize()
                        v_out = Tensor(v_result, device=self.dst_device).reshape(*item["orig_shape"]).realize()
                else:
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
