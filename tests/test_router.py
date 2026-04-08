"""Tests for multi-node KV cache router."""

from __future__ import annotations

import numpy as np
import pytest

tinygrad = pytest.importorskip("tinygrad")
from tinygrad import Tensor, Device

from tqbridge.router import KVRouter, NodeConfig, TCPReceiver
from tqbridge.wire import Format


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


class TestNodeConfig:
    def test_create_local_node(self):
        node = NodeConfig(
            name="m3", layers=range(0, 16), transport="local",
            device="METAL", fmt_k=Format.TURBO3, fmt_v=Format.TURBO3,
        )
        assert node.name == "m3"
        assert len(node.layers) == 16
        assert node.transport == "local"

    def test_create_tcp_node(self):
        node = NodeConfig(
            name="gx10", layers=range(16, 32), transport="tcp",
            host="192.168.68.60", port=9473,
        )
        assert node.host == "192.168.68.60"
        assert node.port == 9473


class TestRouterSetup:
    def test_add_nodes(self):
        router = KVRouter(head_dim=128, n_kv_heads=8)
        router.add_node("m3", layers=range(0, 16), transport="local", device="METAL")
        router.add_node("gx10", layers=range(16, 32), transport="tcp", host="192.168.68.60")

        assert router.total_layers == 32
        assert router.layer_map["m3"] == range(0, 16)
        assert router.layer_map["gx10"] == range(16, 32)
        router.close()

    def test_layer_map(self):
        router = KVRouter(head_dim=128)
        router.add_node("a", layers=range(0, 8), transport="local", device="METAL")
        router.add_node("b", layers=range(8, 24), transport="local", device="METAL")
        router.add_node("c", layers=range(24, 32), transport="local", device="METAL")

        assert router.total_layers == 32
        assert len(router.layer_map) == 3
        router.close()


@hardware
class TestRouterLocal:
    def test_single_node_distribute(self):
        """Route all layers to one local node."""
        router = KVRouter(head_dim=128, n_kv_heads=4, src_device="METAL")
        router.add_node("nv", layers=range(0, 4), transport="local",
                         device="NV", fmt_k=Format.TURBO3, fmt_v=Format.TURBO3)
        router.warmup()

        k = Tensor.rand(4, 4, 2, 128, device="METAL").realize()
        v = Tensor.rand(4, 4, 2, 128, device="METAL").realize()

        results = router.distribute(k, v)
        assert len(results) == 1
        assert results[0].success
        assert results[0].transfer_ms > 0
        assert results[0].compressed_bytes > 0
        router.close()

    def test_two_node_split(self):
        """Split layers between Metal and NV."""
        router = KVRouter(head_dim=128, n_kv_heads=4, src_device="METAL")
        router.add_node("local_metal", layers=range(0, 2), transport="local",
                         device="METAL", fmt_k=Format.TURBO3, fmt_v=Format.TURBO3)
        router.add_node("nv", layers=range(2, 4), transport="local",
                         device="NV", fmt_k=Format.TURBO3, fmt_v=Format.TURBO3)
        router.warmup()

        k = Tensor.rand(4, 4, 1, 128, device="METAL").realize()
        v = Tensor.rand(4, 4, 1, 128, device="METAL").realize()

        results = router.distribute(k, v)
        assert len(results) == 2
        assert all(r.success for r in results)
        router.close()


class TestTCPTransport:
    def test_receiver_start_stop(self):
        """TCP receiver starts and stops cleanly."""
        receiver = TCPReceiver(port=19473)
        receiver.start()
        receiver.stop()

    def test_loopback_send_receive(self):
        """Send compressed KV over TCP loopback."""
        received = []

        def on_receive(header, k_data, v_data):
            received.append((header, k_data, v_data))

        receiver = TCPReceiver(port=19474, head_dim=128)
        receiver.start(on_receive=on_receive)

        from tqbridge.router import TCPSender
        from tqbridge.wire import WireHeader

        sender = TCPSender("127.0.0.1", 19474)
        sender.connect()

        header = WireHeader(
            fmt_k=Format.TURBO3, fmt_v=Format.TURBO3,
            n_layers=4, layer_start=0, seq_len=1,
            n_heads_k=8, n_heads_v=8, head_dim=128,
            flags=0, payload_bytes=200,
        )

        k_data = b'\x01' * 100
        v_data = b'\x02' * 100
        ms = sender.send_kv(k_data, v_data, header)
        assert ms > 0

        import time
        time.sleep(0.5)

        assert len(received) == 1
        hdr, k, v = received[0]
        assert hdr.n_layers == 4
        assert k == k_data
        assert v == v_data

        sender.close()
        receiver.stop()
