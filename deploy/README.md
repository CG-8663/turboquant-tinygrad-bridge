<img src="https://raw.githubusercontent.com/CG-8663/turboquant-tinygrad-bridge/main/docs/logo.png" alt="Chronara" width="150">

# TQBridge — Decode Node Server

Compressed KV cache bridge for distributed LLM inference across heterogeneous GPUs. Originally built to solve the eGPU serving problem — getting maximum throughput from external GPUs (Thunderbolt 5, USB4) where per-transfer latency is high and every byte over the bus matters.

> **Work in progress.** TQBridge is under active development. The per-token bridge throughput (531 tok/s) is near the Thunderbolt 5 latency ceiling — the GPU kernels finish in 0.5ms but each USB4 round-trip costs 1.5ms. That's physics, not software.
>
> Where TQBridge makes the biggest impact today is **prefill distribution and hardware utilisation** — compressing the KV cache lets you spread a 27B model across machines that individually can't hold it, serve longer contexts without running out of memory, and keep every GPU in your cluster productive instead of idle. The bridge turns a collection of mismatched hardware into a single inference system.

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

## What We Solve (Beyond tinygrad Alone)

tinygrad is an excellent GPU runtime — it compiles and dispatches tensor operations across Metal, CUDA, and other backends. TQBridge solves the problems that emerge when you try to use tinygrad for **cross-device KV cache transfer** in production:

| Problem | tinygrad alone | With TQBridge |
|---------|---------------|---------------|
| **Cross-device transfer** | `Tensor.to()` — raw uncompressed, 256KB/token | Compressed to 26KB/token (9.8x), wire protocol with CRC32 |
| **Compression speed** | tinygrad tensor ops: 4.3ms per token | Custom Metal/CUDA kernels: 0.5ms per token (8.6x faster) |
| **Thread safety** | SQLite disk cache + device access fail in threads | Sequential fallback for tinygrad, true threading for native C |
| **eGPU latency** | Every `.realize()` syncs over USB4 (~1ms each) | Pre-allocated buffers, single sync point (531 tok/s vs 75) |
| **Multi-node** | No network transport — local devices only | TCP transport with retries, timeouts, auto-reconnect |
| **Deployment** | Python + pip + tinygrad + numpy + scipy | 1MB static C binary, zero dependencies |
| **Kernel dispatch** | ~1ms overhead per kernel launch over eGPU | Fused compress/decompress kernels, amortized launch cost |

**TQBridge doesn't replace tinygrad** — it uses tinygrad for GPU kernel compilation on the orchestration node, and adds the transport, compression, and deployment layers that tinygrad wasn't designed to provide.

## Why eGPU? The Origin Story

TQBridge was built to solve a specific problem: **getting usable throughput from an RTX PRO 6000 Blackwell connected to a Mac via Thunderbolt 5.**

The TB5 eGPU link has ~1.5ms latency per round-trip — regardless of payload size. Sending uncompressed KV cache (256KB per token) wastes this expensive link. By compressing to 26KB with TurboQuant, we send 10x less data per round-trip. Combined with custom Metal/CUDA kernels that eliminate tinygrad's per-dispatch overhead, we went from **12 tok/s to 531 tok/s** on the same hardware.

The same architecture works for any high-latency link: cloud GPUs, remote servers, or multi-node clusters over Ethernet.

## Kernel & Backend Support

TQBridge includes custom GPU kernels that bypass tinygrad's tensor dispatch overhead. Here's what's supported:

| Backend | Compress | Decompress | Kernel type | Speed |
|---------|----------|------------|-------------|-------|
| **CUDA** (NVIDIA) | ✅ 0.12ms | ✅ 0.12ms | Custom `.cu` kernels | 4,117 tok/s |
| **Metal** (Apple Silicon) | ✅ 0.15ms | ✅ 0.15ms | Custom `.metal` shaders | 3,384 tok/s |
| **Native C** (any CPU) | ✅ 1.7ms | ✅ 2.7ms | Pure C, zero deps | 295 tok/s |
| **tinygrad** (fallback) | ✅ 5.0ms | ✅ 5.0ms | tinygrad tensor ops | 75 tok/s |

### Supported NVIDIA Architectures

| Architecture | GPUs | Status | Notes |
|-------------|------|--------|-------|
| **Pascal** (sm_61) | GTX 1060, 1070, 1080 | ✅ | Via [pascal-egpu](https://github.com/TheTom/pascal-egpu) on macOS eGPU |
| **Turing** (sm_75) | RTX 2060-2080 | ✅ | Standard CUDA |
| **Ampere** (sm_86) | RTX 3060-3090, A100 | ✅ | Standard CUDA |
| **Ada Lovelace** (sm_89) | RTX 4060-4090 | ✅ | Standard CUDA |
| **Blackwell** (sm_120) | RTX 5090, PRO 6000 | ✅ Tested | Primary development hardware |

Kernels use only basic CUDA features (`__shared__`, `__syncthreads__`, `sqrtf`) — no warp shuffles, no tensor cores. Compiles cleanly from sm_61 through sm_120.

The Docker decode node uses the **Native C** path — fast enough for network-connected nodes where the bottleneck is transfer, not compute. The CUDA/Metal kernels are used on the orchestration node where sub-millisecond compress matters.

**tinygrad limitations the kernels solve:**
- tinygrad's SQLite disk cache is not thread-safe — custom kernels don't use it
- tinygrad's `ALLOW_DEVICE_USAGE` blocks GPU access from non-main threads — C kernels have no such restriction
- tinygrad dispatches each tensor op as a separate kernel launch (~1ms each over eGPU) — custom kernels fuse the entire compress/decompress into a single launch
- tinygrad's `Device.synchronize()` round-trips over USB4 — pre-allocated buffers avoid redundant syncs

## Use Case Scenarios

### Scenario 1: "My model doesn't fit on one machine"

**Problem:** You have a 27B model (27GB Q8_0) but only 16GB machines.

**Step-by-step setup:**

```bash
# Step 1: On the prefill machine (32GB+), install llama.cpp with TurboQuant
git clone https://github.com/TheTom/llama-cpp-turboquant
cd llama-cpp-turboquant
git checkout feature/turboquant-kv-cache
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release  # or -DGGML_CUDA=ON
cmake --build build -j

# Step 2: On decode machine B (16GB), start the bridge
docker run -d -p 9473:9473 --name tqbridge --restart unless-stopped chronaragroup/chronara-bridge

# Step 3: On decode machine C (16GB), start the bridge
docker run -d -p 9473:9473 --name tqbridge --restart unless-stopped chronaragroup/chronara-bridge

# Step 4: Run inference on the prefill machine with TurboQuant KV
./build/bin/llama-server -m Qwen3.5-27B-Q8_0.gguf \
    -ctk q8_0 -ctv turbo3 -ngl 99 -c 8192

# The bridge distributes compressed KV to machines B and C
# Each handles a subset of layers for decode
```

### Scenario 2: "Long context is too slow"

**Problem:** At 32K context, decode drops from 67 tok/s to 17 tok/s.

**Step-by-step setup:**

```bash
# Step 1: Install llama.cpp TurboQuant fork (one time)
git clone https://github.com/TheTom/llama-cpp-turboquant
cd llama-cpp-turboquant && git checkout feature/turboquant-kv-cache
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Step 2: Run with asymmetric TurboQuant
./build/bin/llama-server -m model.gguf -ctk q8_0 -ctv turbo4 -c 32768 -ngl 99

# Result: 29.2 tok/s at 32K (vs 17.1 with f16 = 71% faster)
# No Docker needed — this is purely a llama.cpp config change
```

### Scenario 3: "Mac + cloud GPU together"

**Problem:** M3 Mac (128GB) + cloud A100. You want them to split the work.

**Step-by-step setup:**

```bash
# Step 1: On the cloud A100, start decode node
docker run -d -p 9473:9473 --restart unless-stopped chronaragroup/chronara-bridge

# Step 2: On your Mac, install TQBridge Python library
cd turboquant-tinygrad-bridge
pip install -e .

# Step 3: Configure the bridge to route layers
python -c "
from tqbridge.router import KVRouter
from tqbridge.wire import Format

router = KVRouter(head_dim=128, n_kv_heads=8, src_device='METAL')
router.add_node('local', layers=range(0, 16), transport='local', device='METAL')
router.add_node('cloud', layers=range(16, 32), transport='tcp', host='<cloud-ip>')
router.warmup()

# Now distribute KV during inference
# Mac handles layers 0-15 locally, cloud handles 16-31
# At 4.6x compression: 56KB per token over the wire
"
```

### Scenario 4: "Multiple GPUs, different machines"

**Problem:** 2 NVIDIA boxes, a Mac, and an old workstation. Different architectures.

**Step-by-step setup:**

```bash
# Step 1: On EVERY decode machine (any arch), one command:
docker run -d -p 9473:9473 --name tqbridge --restart unless-stopped chronaragroup/chronara-bridge

# Step 2: Verify they're all running
for host in 192.168.1.10 192.168.1.11 192.168.1.12; do
    curl -s --connect-timeout 2 $host:9473 && echo "$host: OK" || echo "$host: not ready"
done

# Step 3: Configure the router on your prefill machine
python -c "
from tqbridge.router import KVRouter
router = KVRouter(head_dim=128, n_kv_heads=8)
router.add_node('nvidia1', layers=range(0, 8), transport='tcp', host='192.168.1.10')
router.add_node('nvidia2', layers=range(8, 16), transport='tcp', host='192.168.1.11')
router.add_node('workstation', layers=range(16, 24), transport='tcp', host='192.168.1.12')
router.add_node('local', layers=range(24, 32), transport='local', device='METAL')
router.warmup()
"

# Docker normalises everything — same 1MB image on x86, arm64, Mac
```

### Scenario 5: "eGPU serving (Thunderbolt / USB4)"

**Problem:** RTX GPU in an eGPU enclosure connected via Thunderbolt 5. Raw transfers are slow (1.5ms latency per round-trip).

**Step-by-step setup:**

```bash
# Step 1: Install TQBridge with tinygrad (for Metal/CUDA kernels)
cd turboquant-tinygrad-bridge
pip install -e .

# Step 2: Use the CUDA backend with pre-allocated buffers
python -c "
from tqbridge.bridge import KVBridge
from tqbridge.wire import Format

bridge = KVBridge(
    head_dim=128,
    fmt_k=Format.TURBO3, fmt_v=Format.TURBO3,
    backend='cuda',              # Custom GPU kernels on both ends
    src_device='METAL',          # Mac compresses via Metal shader
    dst_device='NV',             # RTX decompresses via CUDA kernel
)
bridge.warmup(n_heads=8, seq_len=1, n_layers=32)

# Result: 531 tok/s (vs 12 tok/s without TQBridge)
# Compress: 0.2ms (Metal) → Transfer: 1.5ms (TB5) → Decompress: 0.3ms (CUDA)
"
```

This is the scenario TQBridge was built for. The 1.5ms TB5 latency is physics — TQBridge minimises everything else to make that latency the only cost.

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
