"""Cluster tok/s benchmark: prefill on RTX → fan out to decode fleet.

Drives the KVRouter to distribute compressed KV from the prefill node
(RTX PRO 6000 or Metal) to all decode nodes (local + remote).

Architecture:
  RTX (prefill) → CUDA compress → fan out:
    ├── TB5 → M3 Ultra (local, layers 0-7)
    ├── TCP → GX10-001 at .60 (layers 8-15)
    ├── TCP → GX10-002 at .61 (layers 16-23)
    └── TB5 → M1 Max (local, layers 24-31) [optional]

Remote nodes must be running: python -m tqbridge.serve_decode --port 9473

Usage:
    # Local-only (M3 + RTX on same machine):
    python benchmarks/cluster_toks.py --local-only

    # Full cluster:
    python benchmarks/cluster_toks.py

    # Custom config:
    python benchmarks/cluster_toks.py --nodes m3:METAL:0-16 --nodes gx10:192.168.68.60:16-32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tinygrad"))

import numpy as np
from tinygrad import Tensor, Device

from tqbridge.router import KVRouter
from tqbridge.wire import Format

# Default cluster config
CLUSTER_CONFIGS = {
    "local-only": [
        # Only M3 + RTX on the same machine
        ("m3", "local", "METAL", range(0, 16)),
        ("rtx", "local", "NV", range(16, 32)),
    ],
    "local-m3": [
        # All layers to M3 via local
        ("m3", "local", "METAL", range(0, 32)),
    ],
    "full-cluster": [
        ("m3", "local", "METAL", range(0, 8)),
        ("gx10-001", "tcp", "192.168.68.60:9473", range(8, 16)),
        ("gx10-002", "tcp", "192.168.68.61:9473", range(16, 24)),
        ("rtx", "local", "NV", range(24, 32)),
    ],
    "gx10-pair": [
        ("gx10-001", "tcp", "192.168.68.60:9473", range(0, 16)),
        ("gx10-002", "tcp", "192.168.68.61:9473", range(16, 32)),
    ],
}


def parse_node_spec(spec: str):
    """Parse node spec like 'm3:METAL:0-16' or 'gx10:192.168.68.60:9473:0-16'."""
    parts = spec.split(":")
    name = parts[0]
    layer_parts = parts[-1].split("-")
    layers = range(int(layer_parts[0]), int(layer_parts[1]))

    if len(parts) == 3:
        # name:device_or_host:layers
        target = parts[1]
        if "." in target:
            return (name, "tcp", f"{target}:9473", layers)
        return (name, "local", target, layers)
    elif len(parts) == 4:
        # name:host:port:layers
        return (name, "tcp", f"{parts[1]}:{parts[2]}", layers)
    raise ValueError(f"Invalid node spec: {spec}")


def check_remote(host: str, port: int) -> bool:
    """Quick TCP connect check."""
    import socket
    try:
        s = socket.create_connection((host, port), timeout=2)
        s.close()
        return True
    except (ConnectionRefusedError, TimeoutError, OSError):
        return False


def main():
    parser = argparse.ArgumentParser(description="Cluster tok/s benchmark")
    parser.add_argument("--config", choices=list(CLUSTER_CONFIGS.keys()), default=None)
    parser.add_argument("--nodes", action="append", help="Node spec: name:device:start-end")
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--model", default="qwen3-8b")
    parser.add_argument("--tokens", type=int, default=100)
    parser.add_argument("--src-device", default="METAL")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Model configs
    MODELS = {
        "qwen3-8b": (32, 8, 128),
        "qwen3-14b": (40, 8, 128),
        "llama3-8b": (32, 8, 128),
    }
    n_layers, n_kv_heads, head_dim = MODELS[args.model]

    # Resolve cluster config
    if args.nodes:
        node_specs = [parse_node_spec(s) for s in args.nodes]
    elif args.local_only or args.config == "local-only":
        node_specs = CLUSTER_CONFIGS["local-only"]
    elif args.config:
        node_specs = CLUSTER_CONFIGS[args.config]
    else:
        node_specs = CLUSTER_CONFIGS["local-only"]

    # Check devices
    try:
        Device[args.src_device]
    except Exception:
        print(f"ERROR: Source device {args.src_device} not available")
        sys.exit(1)

    print(f"\n{'=' * 80}")
    print(f"  Cluster tok/s — {args.model} ({n_layers} layers)")
    print(f"  Source: {args.src_device}")
    print(f"  Nodes:")
    for name, transport, target, layers in node_specs:
        print(f"    {name:12s} {transport:5s} → {target:20s} layers {layers.start}-{layers.stop-1}")
    print(f"  Tokens: {args.tokens}")
    print(f"{'=' * 80}\n")

    # Check remote nodes
    for name, transport, target, layers in node_specs:
        if transport == "tcp":
            host, port = target.rsplit(":", 1)
            port = int(port)
            ok = check_remote(host, port)
            status = "OK" if ok else "UNREACHABLE"
            print(f"  {name}: {host}:{port} → {status}")
            if not ok:
                print(f"    Run on {name}: python -m tqbridge.serve_decode --port {port}")

    # Build router
    router = KVRouter(
        head_dim=head_dim, n_kv_heads=n_kv_heads,
        seed=42, src_device=args.src_device,
    )

    for name, transport, target, layers in node_specs:
        if transport == "local":
            router.add_node(name, layers=layers, transport="local", device=target,
                            fmt_k=Format.TURBO3, fmt_v=Format.TURBO3)
        else:
            host, port = target.rsplit(":", 1)
            router.add_node(name, layers=layers, transport="tcp",
                            host=host, port=int(port),
                            fmt_k=Format.TURBO3, fmt_v=Format.TURBO3)

    print(f"\n  Warming up...")
    warmup_ms = router.warmup()
    print(f"  Warmup: {warmup_ms:.0f} ms\n")

    # Generate KV cache on source device
    shape = (n_layers, n_kv_heads, 1, head_dim)
    k = Tensor.rand(*shape, device=args.src_device).realize()
    v = Tensor.rand(*shape, device=args.src_device).realize()

    # Warmup transfers
    for _ in range(5):
        router.distribute(k, v)

    # Benchmark
    token_times = []
    node_times: dict[str, list] = {n: [] for n, _, _, _ in node_specs}

    t_start = time.perf_counter()
    for tok in range(args.tokens):
        t0 = time.perf_counter()
        results = router.distribute(k, v)
        t1 = time.perf_counter()

        token_times.append((t1 - t0) * 1000)
        for r in results:
            node_times[r.node].append(r.transfer_ms)

    t_end = time.perf_counter()
    wall_s = t_end - t_start
    toks = args.tokens / wall_s

    print(f"  {'Node':12s} {'Median ms':>10s} {'Best ms':>10s} {'Layers':>8s}")
    print(f"  {'─' * 44}")
    for name, transport, target, layers in node_specs:
        times = node_times[name]
        med = sorted(times)[len(times) // 2]
        best = min(times)
        print(f"  {name:12s} {med:10.2f} {best:10.2f} {len(layers):>8d}")

    med_total = sorted(token_times)[len(token_times) // 2]
    best_total = min(token_times)
    print(f"\n  Total: {toks:.1f} tok/s  (median {med_total:.2f} ms/tok, best {best_total:.2f} ms/tok)")

    kv_bytes = n_layers * n_kv_heads * head_dim * 4 * 2
    raw_bw = kv_bytes * toks / 1e9
    print(f"  Raw KV bandwidth: {raw_bw:.3f} GB/s ({kv_bytes/1024:.0f} KB/tok)")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps({
            "model": args.model,
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "tokens": args.tokens,
            "toks_per_s": toks,
            "median_ms": med_total,
            "best_ms": best_total,
            "node_results": {
                name: {"median_ms": sorted(node_times[name])[len(node_times[name])//2],
                       "best_ms": min(node_times[name]),
                       "layers": len(layers)}
                for name, _, _, layers in node_specs
            },
        }, indent=2, default=float))
        print(f"  Saved to {out}")

    router.close()
    print()


if __name__ == "__main__":
    main()
