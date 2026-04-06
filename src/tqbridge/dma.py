"""DMA manager: cross-device tensor transfer via tinygrad Tensor.to().

Wraps tinygrad's built-in cross-device transfer with timing instrumentation.
The actual data path is CPU-mediated (copyin/copyout) through TB5 PCIe 4.0 x4.
TurboQuant compression reduces the transfer payload by 4.6x, making the
effective bandwidth ~28 GB/s vs ~6 GB/s raw.
"""

from __future__ import annotations

from tinygrad import Tensor, Device

from tqbridge.metrics import Timer


class DMAManager:
    """Manages cross-device tensor transfers between Metal and NV backends."""

    def __init__(self, src_device: str = "METAL", dst_device: str = "NV"):
        self.src_device = src_device
        self.dst_device = dst_device

    def transfer(self, tensor: Tensor) -> tuple[Tensor, float]:
        """Transfer a tensor from src to dst device. Blocking.

        Args:
            tensor: tinygrad Tensor on src_device

        Returns:
            (transferred_tensor, transfer_time_ms)
        """
        with Timer() as t:
            result = tensor.to(self.dst_device).realize()
            Device[self.dst_device].synchronize()
        return result, t.ms

    def transfer_dict(self, d: dict) -> tuple[dict, float]:
        """Transfer all Tensor values in a dict to dst_device.

        Used for compressed KV payloads (norms, indices, scales, quants).
        Non-Tensor values are passed through unchanged.

        Returns:
            (transferred_dict, total_transfer_time_ms)
        """
        result = {}
        total_ms = 0.0
        for k, v in d.items():
            if isinstance(v, Tensor):
                transferred, ms = self.transfer(v)
                result[k] = transferred
                total_ms += ms
            else:
                result[k] = v
        return result, total_ms
