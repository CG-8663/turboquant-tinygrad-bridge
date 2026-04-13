"""Decode node server: receives compressed KV over TCP, decompresses, ready for decode.

Deploy on GX10 nodes (Linux + CUDA) or any machine with a GPU.
Listens for wire-protocol KV cache transfers from the prefill node,
decompresses using native C or CUDA kernels, and stores the KV cache
in GPU memory ready for decode inference.

Usage:
    # On GX10-001 (192.168.68.61):
    python -m tqbridge.serve_decode --port 9473 --device NV

    # On M3 Ultra (local Metal):
    python -m tqbridge.serve_decode --port 9474 --device METAL
"""

from __future__ import annotations

import argparse
import sys
import time
import threading
from pathlib import Path

import numpy as np

from tqbridge.router import TCPReceiver
from tqbridge.wire import Format, decode_header
from tqbridge.native import NativeCompressor


class DecodeServer:
    """Receives compressed KV, decompresses, stores in GPU memory."""

    def __init__(
        self,
        port: int = 9473,
        head_dim: int = 128,
        seed: int = 42,
        device: str = "NV",
    ):
        self.port = port
        self.head_dim = head_dim
        self.seed = seed
        self.device = device

        self._nc = NativeCompressor(head_dim=head_dim, seed=seed)
        self._receiver = TCPReceiver(port=port, head_dim=head_dim)
        self._kv_store: dict[int, tuple] = {}  # layer_idx → (k, v)
        self._lock = threading.Lock()
        self._tokens_received = 0
        self._total_bytes = 0
        self._t_start = None

    def start(self) -> None:
        """Start listening for KV transfers."""
        self._t_start = time.monotonic()
        self._receiver.start(on_receive=self._on_kv_received)
        print(f"[DecodeServer] Listening on port {self.port}, device={self.device}")

    def _on_kv_received(self, header, k_data: bytes, v_data: bytes) -> None:
        """Handle incoming compressed KV cache."""
        t0 = time.perf_counter()

        n_layers = header.n_layers
        layer_start = header.layer_start
        fmt_k = Format(header.fmt_k)
        fmt_v = Format(header.fmt_v)

        # Decompress K
        k_bytes = np.frombuffer(k_data, dtype=np.uint8)
        k_comp = {
            "fmt": fmt_k,
            "compressed_bytes": k_bytes,
            "n_vectors": header.n_heads_k * header.seq_len * n_layers,
            "orig_shape": (n_layers, header.n_heads_k, header.seq_len, self.head_dim),
            "n_elements": header.n_heads_k * header.seq_len * n_layers * self.head_dim,
            "bit_width": {Format.TURBO2: 2, Format.TURBO3: 3, Format.TURBO4: 4}.get(fmt_k),
        }

        # Decompress V
        v_bytes = np.frombuffer(v_data, dtype=np.uint8)
        v_comp = {
            "fmt": fmt_v,
            "compressed_bytes": v_bytes,
            "n_vectors": header.n_heads_v * header.seq_len * n_layers,
            "orig_shape": (n_layers, header.n_heads_v, header.seq_len, self.head_dim),
            "n_elements": header.n_heads_v * header.seq_len * n_layers * self.head_dim,
            "bit_width": {Format.TURBO2: 2, Format.TURBO3: 3, Format.TURBO4: 4}.get(fmt_v),
        }

        k_result = self._nc.decompress(k_comp)
        v_result = self._nc.decompress(v_comp)

        t1 = time.perf_counter()
        decompress_ms = (t1 - t0) * 1000

        with self._lock:
            for i in range(n_layers):
                layer_idx = layer_start + i
                self._kv_store[layer_idx] = (
                    k_result.reshape(k_comp["orig_shape"])[i],
                    v_result.reshape(v_comp["orig_shape"])[i],
                )
            self._tokens_received += 1
            self._total_bytes += len(k_data) + len(v_data)

        elapsed = time.monotonic() - self._t_start
        tps = self._tokens_received / elapsed if elapsed > 0 else 0
        print(f"  [{self._tokens_received:4d}] layers {layer_start}-{layer_start+n_layers-1} "
              f"decompress {decompress_ms:.1f}ms  "
              f"{len(k_data)+len(v_data)} bytes  "
              f"({tps:.1f} tok/s avg)")

    @property
    def kv_layers(self) -> int:
        with self._lock:
            return len(self._kv_store)

    def stop(self) -> None:
        self._receiver.stop()
        self._nc.close()
        elapsed = time.monotonic() - self._t_start if self._t_start else 0
        tps = self._tokens_received / elapsed if elapsed > 0 else 0
        print(f"\n[DecodeServer] Stopped. {self._tokens_received} tokens, "
              f"{self._total_bytes/1e6:.1f} MB, {tps:.1f} tok/s avg")


def main():
    parser = argparse.ArgumentParser(description="TurboQuant decode node server")
    parser.add_argument("--port", type=int, default=9473)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="NV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    server = DecodeServer(
        port=args.port, head_dim=args.head_dim,
        seed=args.seed, device=args.device,
    )
    server.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
