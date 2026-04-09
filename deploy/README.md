<img src="https://raw.githubusercontent.com/CG-8663/turboquant-tinygrad-bridge/main/docs/logo.png" alt="Chronara" width="150">

# TQBridge — Decode Node Server

Compressed KV cache bridge for distributed LLM inference across heterogeneous GPUs.

## Quick Start

```bash
docker run -p 9473:9473 chronaragroup/chronara-bridge
```

That's it. The server listens for compressed KV cache transfers on port 9473, decompresses them, and is ready for decode inference.

## Dependencies

**The Docker image has zero dependencies.** It's a statically-linked C binary on an empty base image (`FROM scratch`). No Python, no tinygrad, no CUDA toolkit, no GPU drivers needed on the decode node.

| Component | Required? | Notes |
|-----------|-----------|-------|
| Docker | Yes (or build from C source) | Any platform with Docker |
| Python | No | Not needed on decode nodes |
| tinygrad | No | Only needed on the prefill/orchestration node |
| CUDA / Metal | No | Decode runs on CPU — GPU optional for acceleration |
| TurboQuant (llama.cpp fork) | Only on prefill node | The model runs there, not on decode nodes |
| NumPy / SciPy | No | Codebooks are precomputed into the binary |

**Prefill node requirements** (the machine running the model):
- [llama.cpp TurboQuant fork](https://github.com/TheTom/llama-cpp-turboquant) on the `feature/turboquant-kv-cache` branch
- Or the TQBridge Python library with tinygrad (for custom kernel path)

## What It Does

TQBridge compresses the KV cache (the model's working memory during inference) by 4-5x using TurboQuant, then transfers it between machines over the network. This lets you:

- **Split models** across a Mac, NVIDIA GPUs, and server nodes
- **Run longer contexts** — 131K context on 16GB machines
- **Decode faster** — turbo4 is 71% faster than f16 at 32K context

## Use Case Scenarios

### Scenario 1: "My model doesn't fit on one machine"

**Problem:** You have a 27B model (27GB Q8_0) but only 16GB machines.

**Solution:** Run the model on a machine with enough RAM for the weights, use TQBridge to distribute the KV cache to decode nodes. Each decode node handles a subset of layers.

```
Machine A (32GB, runs the model):
  → Prefill with llama.cpp + TurboQuant
  → Compresses KV cache 4.6x (27GB model, KV goes from 256KB to 56KB per token)

Machine B (16GB, Docker decode node):
  docker run -p 9473:9473 chronaragroup/chronara-bridge
  → Receives layers 0-15, decompresses, ready for decode

Machine C (16GB, Docker decode node):
  docker run -p 9473:9473 chronaragroup/chronara-bridge
  → Receives layers 16-31, decompresses, ready for decode
```

### Scenario 2: "Long context is too slow"

**Problem:** At 32K context, your decode speed drops from 67 tok/s to 17 tok/s because the KV cache doesn't fit in fast memory.

**Solution:** Use asymmetric TurboQuant (q8_0 K + turbo4 V). The compressed KV cache stays in fast memory.

```bash
# On llama.cpp:
./llama-server -m model.gguf -ctk q8_0 -ctv turbo4 -c 32768 -ngl 99

# Result: 29.2 tok/s at 32K (vs 17.1 with f16 = 71% faster)
```

No Docker needed for this scenario — it's purely a llama.cpp config change.

### Scenario 3: "I want to use my Mac and a cloud GPU together"

**Problem:** You have an M3 Mac (128GB unified memory) and a cloud instance with an A100. You want them to split the work.

**Solution:** Mac runs prefill + some layers. Cloud instance runs a TQBridge decode node for the rest.

```
Mac M3 Ultra (local):
  → Runs model, prefills, handles 16 layers
  → Compresses remaining 16 layers via TQBridge

Cloud A100 (remote):
  docker run -p 9473:9473 chronaragroup/chronara-bridge
  → Receives 16 layers over internet, decompresses
  → At 4.6x compression: 56KB per token over the wire
```

### Scenario 4: "I have multiple GPUs across different machines"

**Problem:** You have 2 NVIDIA boxes, a Mac, and an old workstation. Different architectures, different drivers.

**Solution:** Docker normalises everything. Each machine runs the same image.

```bash
# On every decode node (Linux x86, Linux arm64, or Mac Docker):
docker run -d -p 9473:9473 --restart unless-stopped chronaragroup/chronara-bridge

# The bridge doesn't care about GPU type — it decompresses on CPU
# GPU acceleration is optional and automatic when available
```

### Scenario 5: "I want to serve multiple users from one model"

**Problem:** You have one powerful machine running a model, but multiple users need inference.

**Solution:** Prefill once, distribute KV cache to per-user decode nodes. Each user gets their own context without re-prefilling.

```
Prefill Server (RTX 6000, 96GB):
  → Loads model once
  → Prefills for each user request
  → Distributes compressed KV to user's decode node

User A decode node: docker run -p 9473:9473 chronaragroup/chronara-bridge
User B decode node: docker run -p 9474:9473 chronaragroup/chronara-bridge
User C decode node: docker run -p 9475:9473 chronaragroup/chronara-bridge
```

## Usage

### Single Node
```bash
docker run -p 9473:9473 chronaragroup/chronara-bridge
```

### Custom Configuration
```bash
docker run -p 9473:9473 chronaragroup/chronara-bridge --head-dim 128 --seed 42
```

### Daemon Mode (restart on reboot)
```bash
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
Prefill Node (runs the model)
  ├── Compress KV cache (0.2ms per token)
  ├── Send via TCP to decode nodes
  └── 26KB per token (compressed from 256KB)

Decode Node (this container)
  ├── Receive compressed KV over TCP
  ├── Decompress via C library (0.3ms)
  └── KV cache ready for decode inference
```

## Best Practices

### Which TurboQuant Config Should I Use?

| Scenario | K cache | V cache | Compression | Speed impact |
|----------|---------|---------|-------------|--------------|
| **Safe default (recommended)** | q8_0 | turbo3 | 4.6x | Within 4% of f16 |
| Speed priority at long context | q8_0 | turbo4 | 3.8x | +71% faster than f16 at 32K |
| Maximum compression | q8_0 | turbo2 | 6.4x | +6.5% PPL, for memory pressure |
| **Never do this** | turbo3 | turbo3 | — | Symmetric kills quality on 27B+ |

**Rule: Always keep K at q8_0.** All quality loss comes from K compression. V is "free" to compress.

### Which Models Work Best?

| Model | Turbo tolerance | Notes |
|-------|----------------|-------|
| **Qwen3.5-9B** | Excellent | Flat decode curve, beats f16 at 32K |
| **Qwen3.5-27B** | Excellent (asymmetric only) | Must use asymmetric |
| **Qwen3-8B** | Good | turbo4 +71% faster at 32K |
| **MoE models (35B+)** | Best | MoE models benefit the most |

### When NOT to Use TQBridge

- **Short conversations (<2K context)** — f16 is faster at short context
- **Symmetric turbo on large models** — always use asymmetric (q8_0 K + turbo V)
- **Single GPU with plenty of memory** — no need for the bridge if everything fits

## Supported Platforms

| Platform | Architecture | Status |
|----------|-------------|--------|
| Linux x86_64 | amd64 | ✅ |
| Linux aarch64 | arm64 (GX10, Graviton, Jetson) | ✅ |
| macOS (Docker Desktop) | arm64 (Apple Silicon) | ✅ |

## Image Details

- **Size**: ~1 MB (static binary on `FROM scratch`)
- **Dependencies**: None
- **Port**: 9473 (TQKV protocol)
- **Wire format**: 40-byte header with CRC32, TurboQuant compression

## Built With

TQBridge builds on the work of several excellent projects:

- **[TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)** (Google Research, ICLR 2026) — the KV cache compression algorithm at the heart of this bridge
- **[TurboQuant+](https://github.com/TheTom/llama-cpp-turboquant)** by Tom Turney — the llama.cpp implementation with asymmetric K/V, sparse V dequant, and Metal kernels that made TurboQuant practical for local inference
- **[tinygrad](https://github.com/tinygrad/tinygrad)** by George Hotz — the GPU runtime we use for Metal and CUDA kernel dispatch on the orchestration node

We're not replacing any of these — TQBridge is the transport layer that connects them across machines. Run TurboQuant+ on your prefill node, distribute the compressed KV cache through TQBridge to your decode fleet.

## Links

- [Benchmark Results](https://github.com/CG-8663/turboquant-tinygrad-bridge/blob/main/docs/results-summary.md)
- [TurboQuant Paper (ICLR 2026)](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [TurboQuant+ (llama.cpp fork)](https://github.com/TheTom/llama-cpp-turboquant)
- [tinygrad](https://github.com/tinygrad/tinygrad)
- [Source Code](https://github.com/CG-8663/turboquant-tinygrad-bridge)
- [Docker Hub](https://hub.docker.com/r/chronaragroup/chronara-bridge)
