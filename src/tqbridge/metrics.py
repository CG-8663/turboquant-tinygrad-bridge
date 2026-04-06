"""Bridge metrics: per-layer timing, compression ratio, pipeline utilisation."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class TransferMetrics:
    """Metrics for a single layer KV transfer."""

    layer_idx: int
    compress_time_ms: float = 0.0
    transfer_time_ms: float = 0.0
    decompress_time_ms: float = 0.0
    original_bytes: int = 0
    compressed_bytes: int = 0

    @property
    def compression_ratio(self) -> float:
        if self.compressed_bytes == 0:
            return 0.0
        return self.original_bytes / self.compressed_bytes

    @property
    def total_time_ms(self) -> float:
        return self.compress_time_ms + self.transfer_time_ms + self.decompress_time_ms

    @property
    def effective_bandwidth_gbps(self) -> float:
        """Effective bandwidth in GB/s based on original (uncompressed) size."""
        if self.transfer_time_ms == 0:
            return 0.0
        return (self.original_bytes / 1e9) / (self.transfer_time_ms / 1e3)


@dataclass
class PipelineMetrics:
    """Aggregate metrics across a multi-layer KV transfer."""

    layers: list[TransferMetrics] = field(default_factory=list)

    def add(self, m: TransferMetrics) -> None:
        self.layers.append(m)

    @property
    def total_time_ms(self) -> float:
        return sum(m.total_time_ms for m in self.layers)

    @property
    def total_original_bytes(self) -> int:
        return sum(m.original_bytes for m in self.layers)

    @property
    def total_compressed_bytes(self) -> int:
        return sum(m.compressed_bytes for m in self.layers)

    @property
    def avg_compression_ratio(self) -> float:
        if self.total_compressed_bytes == 0:
            return 0.0
        return self.total_original_bytes / self.total_compressed_bytes

    @property
    def avg_effective_bandwidth_gbps(self) -> float:
        total_transfer_ms = sum(m.transfer_time_ms for m in self.layers)
        if total_transfer_ms == 0:
            return 0.0
        return (self.total_original_bytes / 1e9) / (total_transfer_ms / 1e3)

    def summary(self) -> str:
        n = len(self.layers)
        if n == 0:
            return "No layers transferred"
        return (
            f"{n} layers | {self.total_time_ms:.1f} ms total | "
            f"{self.avg_compression_ratio:.1f}x compression | "
            f"{self.avg_effective_bandwidth_gbps:.1f} GB/s effective"
        )


class Timer:
    """Context manager for timing a block in milliseconds."""

    def __init__(self) -> None:
        self.ms: float = 0.0

    def __enter__(self) -> Timer:
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.ms = (time.perf_counter() - self._start) * 1000
