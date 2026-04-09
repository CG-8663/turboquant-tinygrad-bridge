<p align="center">
  <img src="https://raw.githubusercontent.com/CG-8663/turboquant-tinygrad-bridge/main/docs/logo.png" alt="Chronara" width="180">
</p>

# TQBridge — Decode Node Server

Compressed KV cache bridge for distributed LLM inference across heterogeneous GPUs.

## Quick Start

```bash
docker run -p 9473:9473 chronaragroup/chronara-bridge
```

That's it. The server listens for compressed KV cache transfers on port 9473, decompresses them, and is ready for decode inference.

## What It Does

TQBridge compresses the KV cache (the model's working memory during inference) by 4-5x using TurboQuant, then transfers it between GPUs over the network. This lets you:

- **Split models** across a Mac, NVIDIA GPUs, and server nodes
- **Run longer contexts** — 131K context on 16GB machines
- **Decode faster** — turbo4 is 71% faster than f16 at 32K context

## Usage

### Single Node
```bash
docker run -p 9473:9473 chronaragroup/chronara-bridge
```

### Custom Configuration
```bash
docker run -p 9473:9473 chronaragroup/chronara-bridge --head-dim 128 --seed 42
```

### Multiple Nodes (Docker Compose)
```yaml
# On GX10-001 (192.168.68.60):
docker run -d -p 9473:9473 --name tqbridge --restart unless-stopped chronaragroup/chronara-bridge

# On GX10-002 (192.168.68.61):
docker run -d -p 9473:9473 --name tqbridge --restart unless-stopped chronaragroup/chronara-bridge
```

### Without Docker
Build from source (requires only a C compiler):
```bash
cc -O2 -o tqbridge-server tqbridge-server.c tqbridge.c tqbridge_net.c -lm
./tqbridge-server --port 9473
```

## Architecture

```
Prefill Node (NVIDIA GPU)
  ├── Compress KV cache (0.2ms)
  ├── Send via TCP to decode nodes
  └── 26KB per token (compressed from 256KB)

Decode Node (this container)
  ├── Receive compressed KV
  ├── Decompress (0.3ms)
  └── Ready for decode inference
```

## Best Practices & Recommendations

### Which TurboQuant Config Should I Use?

| Scenario | K cache | V cache | Compression | Speed impact |
|----------|---------|---------|-------------|--------------|
| **Safe default (recommended)** | q8_0 | turbo3 | 4.6x | Within 4% of f16 |
| Speed priority at long context | q8_0 | turbo4 | 3.8x | +71% faster than f16 at 32K |
| Maximum compression | q8_0 | turbo2 | 6.4x | +6.5% PPL, for memory pressure |
| **Never do this** | turbo3 | turbo3 | — | Symmetric kills quality on 27B+ |

**Rule: Always keep K at q8_0.** All quality loss comes from K compression. V is "free" to compress — even turbo2 V has near-zero quality impact.

### Which Models Work Best?

Tested on M3 Ultra with llama.cpp TurboQuant branch:

| Model | Turbo tolerance | Notes |
|-------|----------------|-------|
| **Qwen3.5-9B** | Excellent | Flat decode curve, turbo3 symmetric beats f16 at 32K |
| **Qwen3.5-27B** | Excellent (asymmetric) | Must use asymmetric — symmetric collapses |
| **Qwen3-8B** | Good | turbo4 +71% faster at 32K |
| **Llama 3 8B** | Good | Similar to Qwen3-8B |
| **MoE models (35B+)** | Best | MoE models benefit the most from KV compression |

### Cluster Setup Examples

**Example 1: Mac Studio + 2 decode nodes (home lab)**
```bash
# On decode node 1 (any Linux machine):
docker run -d -p 9473:9473 --restart unless-stopped chronaragroup/chronara-bridge

# On decode node 2:
docker run -d -p 9473:9473 --restart unless-stopped chronaragroup/chronara-bridge

# On Mac (runs the model + prefill):
./llama-server -m model.gguf -ctk q8_0 -ctv turbo3 -ngl 99
```

**Example 2: Cloud cluster (3 × GPU nodes)**
```bash
# On each GPU node:
docker run -d -p 9473:9473 --gpus all --restart unless-stopped chronaragroup/chronara-bridge

# Bridge distributes KV: node1 gets layers 0-10, node2 gets 11-21, node3 gets 22-31
```

**Example 3: Mixed hardware (Mac + NVIDIA eGPU + remote server)**
```
Mac M3 Ultra (128GB)     — prefill + 16 layers decode
RTX PRO 6000 eGPU (96GB) — 8 layers decode via TB5
Remote server (GPU)      — 8 layers decode via Ethernet
```

### Performance by Context Depth

The key insight: **TurboQuant gets faster relative to baseline as context grows.**

```
Qwen3-8B Q8_0 — M3 Ultra decode tok/s:

Context    f16    turbo4(asym)    Advantage
  0       67.8      54.2          -20% (turbo slower)
  4K      49.6      49.9          CROSSOVER
  8K      39.5      45.7          +15.6%
  16K     28.0      38.3          +36.8%
  32K     17.1      29.2          +70.8% ← turbo wins big
```

At short context, f16 is faster. But as context grows, KV cache bandwidth becomes the bottleneck and turbo's 4x compression advantage kicks in.

### When NOT to Use TQBridge

- **Short conversations (<2K context)** — overhead isn't worth it, f16 is faster
- **Symmetric turbo on large models** — always use asymmetric (q8_0 K + turbo V)
- **Quality-critical applications** — run needle-in-haystack validation first
- **Single GPU with plenty of memory** — no need for the bridge if everything fits

## Supported Platforms

| Platform | Architecture | Status |
|----------|-------------|--------|
| Linux x86_64 | amd64 | ✅ |
| Linux aarch64 | arm64 (GX10, Graviton) | ✅ |
| macOS (Docker Desktop) | arm64 (Apple Silicon) | ✅ |

## Image Details

- **Size**: ~1 MB (static binary on `FROM scratch`)
- **Dependencies**: None
- **Port**: 9473 (TQKV protocol)
- **Wire format**: 40-byte header with CRC32, TurboQuant compression

## Links

- [Benchmark Results](https://github.com/CG-8663/turboquant-tinygrad-bridge/blob/main/docs/results-summary.md)
- [TurboQuant Paper (ICLR 2026)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [Source Code](https://github.com/CG-8663/turboquant-tinygrad-bridge)
