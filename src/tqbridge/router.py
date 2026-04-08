"""Multi-node KV cache router for distributed split inference.

Routes compressed KV cache layers from a source (prefill) node to
multiple destination (decode) nodes. Each node owns a contiguous range
of layers and receives only the KV for its assigned layers.

Supports two transport modes:
  - "local": tinygrad Tensor.to() for same-machine devices (Metal↔NV via TB5)
  - "tcp": async TCP for network-connected nodes (GX10 via 200GbE)

Usage:
    router = KVRouter(head_dim=128, seed=42)
    router.add_node("m3", device="METAL", layers=range(0, 16), transport="local")
    router.add_node("gx10", host="192.168.68.60", layers=range(16, 32), transport="tcp")
    router.warmup()

    # Fan out KV from prefill
    results = router.distribute(k_cache, v_cache)
"""

from __future__ import annotations

import socket
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from tqbridge.compression import _get_rotation, _get_codebook
from tqbridge.metrics import Timer, TransferMetrics
from tqbridge.wire import Format, encode_header, WireHeader


# ---------------------------------------------------------------------------
# Node configuration
# ---------------------------------------------------------------------------

@dataclass
class NodeConfig:
    """Configuration for a single decode node."""
    name: str
    layers: range
    transport: Literal["local", "tcp"]
    fmt_k: Format = Format.TURBO3
    fmt_v: Format = Format.TURBO3

    # Local transport (tinygrad device)
    device: str | None = None

    # TCP transport
    host: str | None = None
    port: int = 9473  # "TQKV" → 9473


@dataclass
class DistributeResult:
    """Result of distributing KV to a single node."""
    node: str
    layers: range
    transfer_ms: float
    compressed_bytes: int
    success: bool
    error: str | None = None


# ---------------------------------------------------------------------------
# TCP transport — lightweight async sender
# ---------------------------------------------------------------------------

class TCPSender:
    """Send compressed KV cache over TCP with wire protocol header."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        self._sock.connect((self.host, self.port))

    def send_kv(
        self,
        k_data: bytes,
        v_data: bytes,
        header: WireHeader,
    ) -> float:
        """Send K+V compressed data with wire header. Returns send time in ms."""
        if self._sock is None:
            self.connect()

        hdr_bytes = encode_header(header)
        payload = hdr_bytes + k_data + v_data

        t0 = time.perf_counter()
        self._sock.sendall(payload)
        t1 = time.perf_counter()
        return (t1 - t0) * 1000

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None


class TCPReceiver:
    """Receive compressed KV cache over TCP. Runs in a background thread."""

    def __init__(self, port: int = 9473, head_dim: int = 128):
        self.port = port
        self.head_dim = head_dim
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._on_receive = None

    def start(self, on_receive=None) -> None:
        """Start listening. on_receive(header, k_data, v_data) called per message."""
        self._on_receive = on_receive
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        self._sock.settimeout(1.0)
        self._sock.bind(("0.0.0.0", self.port))
        self._sock.listen(4)
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def _listen_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
                conn.settimeout(5.0)
                self._handle_conn(conn)
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_conn(self, conn: socket.socket) -> None:
        try:
            while not self._stop.is_set():
                # Read 40-byte header
                hdr_data = self._recv_exact(conn, 40)
                if hdr_data is None:
                    break

                from tqbridge.wire import decode_header
                header = decode_header(hdr_data)

                # Read K+V payload
                payload = self._recv_exact(conn, header.payload_bytes)
                if payload is None:
                    break

                # Split K and V (each is half the payload)
                mid = len(payload) // 2
                k_data = payload[:mid]
                v_data = payload[mid:]

                if self._on_receive:
                    self._on_receive(header, k_data, v_data)
        finally:
            conn.close()

    def _recv_exact(self, conn: socket.socket, n: int) -> bytes | None:
        buf = bytearray()
        while len(buf) < n:
            try:
                chunk = conn.recv(n - len(buf))
                if not chunk:
                    return None
                buf.extend(chunk)
            except socket.timeout:
                if self._stop.is_set():
                    return None
        return bytes(buf)

    def stop(self) -> None:
        self._stop.set()
        if self._sock:
            self._sock.close()
        if self._thread:
            self._thread.join(timeout=3.0)


# ---------------------------------------------------------------------------
# KV Router
# ---------------------------------------------------------------------------

class KVRouter:
    """Routes compressed KV cache to multiple decode nodes.

    Splits layers across nodes and fans out compressed KV after prefill.
    Supports both local (tinygrad) and network (TCP) transport.
    """

    def __init__(
        self,
        head_dim: int = 128,
        n_kv_heads: int = 8,
        seed: int = 42,
        src_device: str = "NV",
    ):
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.seed = seed
        self.src_device = src_device
        self.nodes: dict[str, NodeConfig] = {}
        self._tcp_senders: dict[str, TCPSender] = {}
        self._local_bridges: dict[str, object] = {}

    def add_node(
        self,
        name: str,
        layers: range,
        transport: Literal["local", "tcp"] = "local",
        device: str | None = None,
        host: str | None = None,
        port: int = 9473,
        fmt_k: Format = Format.TURBO3,
        fmt_v: Format = Format.TURBO3,
    ) -> None:
        """Register a decode node."""
        node = NodeConfig(
            name=name, layers=layers, transport=transport,
            device=device, host=host, port=port,
            fmt_k=fmt_k, fmt_v=fmt_v,
        )
        self.nodes[name] = node

        if transport == "tcp":
            self._tcp_senders[name] = TCPSender(host, port)
        elif transport == "local":
            from tqbridge.bridge import KVBridge
            bridge = KVBridge(
                head_dim=self.head_dim,
                fmt_k=fmt_k, fmt_v=fmt_v,
                src_device=self.src_device,
                dst_device=device,
                backend="cuda",
                seed=self.seed,
            )
            self._local_bridges[name] = bridge

    def warmup(self) -> float:
        """Pre-compile kernels and pre-allocate buffers for all nodes."""
        with Timer() as t:
            for name, node in self.nodes.items():
                n_layers = len(node.layers)
                if node.transport == "local":
                    bridge = self._local_bridges[name]
                    bridge.warmup(
                        n_heads=self.n_kv_heads, seq_len=1, n_layers=n_layers,
                    )
                elif node.transport == "tcp":
                    self._tcp_senders[name].connect()
        return t.ms

    def distribute(
        self,
        k_cache,
        v_cache,
        parallel: bool = True,
    ) -> list[DistributeResult]:
        """Fan out KV cache layers to all registered nodes.

        Args:
            k_cache: (n_layers, n_kv_heads, seq_len, head_dim) on src_device
            v_cache: same shape as k_cache
            parallel: if True, send to all nodes concurrently

        Returns:
            List of DistributeResult, one per node.
        """
        # tinygrad is not thread-safe — only use parallel for TCP-only routes
        has_local = any(n.transport == "local" for n in self.nodes.values())
        if parallel and len(self.nodes) > 1 and not has_local:
            return self._distribute_parallel(k_cache, v_cache)
        return self._distribute_sequential(k_cache, v_cache)

    def _distribute_sequential(self, k_cache, v_cache) -> list[DistributeResult]:
        results = []
        for name, node in self.nodes.items():
            r = self._send_to_node(name, node, k_cache, v_cache)
            results.append(r)
        return results

    def _distribute_parallel(self, k_cache, v_cache) -> list[DistributeResult]:
        results = [None] * len(self.nodes)
        threads = []

        for idx, (name, node) in enumerate(self.nodes.items()):
            def _send(i=idx, n=name, nd=node):
                results[i] = self._send_to_node(n, nd, k_cache, v_cache)
            t = threading.Thread(target=_send)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        return results

    def _send_to_node(
        self, name: str, node: NodeConfig, k_cache, v_cache
    ) -> DistributeResult:
        """Send assigned layers to a single node."""
        try:
            layers = node.layers
            # Extract this node's layers
            k_slice = k_cache[layers.start:layers.stop]
            v_slice = v_cache[layers.start:layers.stop]

            if node.transport == "local":
                return self._send_local(name, node, k_slice, v_slice)
            else:
                return self._send_tcp(name, node, k_slice, v_slice)
        except Exception as e:
            return DistributeResult(
                node=name, layers=node.layers,
                transfer_ms=0, compressed_bytes=0,
                success=False, error=str(e),
            )

    def _send_local(self, name, node, k_slice, v_slice) -> DistributeResult:
        """Transfer via tinygrad local device (TB5)."""
        bridge = self._local_bridges[name]

        with Timer() as t:
            k_out, v_out, metrics = bridge.transfer_kv_bulk(k_slice, v_slice)

        return DistributeResult(
            node=name, layers=node.layers,
            transfer_ms=t.ms, compressed_bytes=metrics.compressed_bytes,
            success=True,
        )

    def _send_tcp(self, name, node, k_slice, v_slice) -> DistributeResult:
        """Transfer via TCP (network nodes)."""
        from tqbridge.native import NativeCompressor

        nc = NativeCompressor(head_dim=self.head_dim, seed=self.seed)
        sender = self._tcp_senders[name]

        n_layers = k_slice.shape[0]
        seq_len = k_slice.shape[2] if len(k_slice.shape) > 2 else 1

        # Compress on CPU via native C (fast, no GPU dependency)
        k_np = k_slice.reshape(-1, self.head_dim).numpy()
        v_np = v_slice.reshape(-1, self.head_dim).numpy()

        k_comp = nc.compress(k_np, node.fmt_k)
        v_comp = nc.compress(v_np, node.fmt_v)

        k_bytes = bytes(k_comp["compressed_bytes"])
        v_bytes = bytes(v_comp["compressed_bytes"])

        # Build wire header
        header = WireHeader(
            fmt_k=node.fmt_k, fmt_v=node.fmt_v,
            n_layers=n_layers, layer_start=node.layers.start,
            seq_len=seq_len, n_heads_k=self.n_kv_heads,
            n_heads_v=self.n_kv_heads, head_dim=self.head_dim,
            flags=0, payload_bytes=len(k_bytes) + len(v_bytes),
        )

        with Timer() as t:
            sender.send_kv(k_bytes, v_bytes, header)

        nc.close()

        return DistributeResult(
            node=name, layers=node.layers,
            transfer_ms=t.ms,
            compressed_bytes=len(k_bytes) + len(v_bytes),
            success=True,
        )

    @property
    def total_layers(self) -> int:
        """Total layers covered by all nodes."""
        return sum(len(n.layers) for n in self.nodes.values())

    @property
    def layer_map(self) -> dict[str, range]:
        """Map of node name → layer range."""
        return {n: cfg.layers for n, cfg in self.nodes.items()}

    def close(self) -> None:
        for sender in self._tcp_senders.values():
            sender.close()
        for bridge in self._local_bridges.values():
            bridge.close()
        self._tcp_senders.clear()
        self._local_bridges.clear()

    def __del__(self):
        self.close()
