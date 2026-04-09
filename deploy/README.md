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
