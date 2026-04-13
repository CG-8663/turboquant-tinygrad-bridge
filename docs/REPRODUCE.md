# Reproducing TQBridge Cluster Results

All numbers in results-summary.md were measured on real hardware on 2026-04-13. Here's how to reproduce them.

## Hardware Required

| Node | Hardware | VRAM | Link |
|------|----------|------|------|
| Mac with Apple Silicon | M3 Ultra / M1 Max / M2 etc | 32-96 GB unified | Local |
| NVIDIA GPU (Linux) | Any CUDA GPU (GB10, RTX, etc) | 8+ GB | 10GbE or faster |
| Optional: NVIDIA eGPU | RTX via Thunderbolt 3/4/5 | 8+ GB | TB5 via tinygrad |

Minimum: one Mac + one Linux box with NVIDIA GPU.

## Software Setup

### Mac (MLX inference + orchestrator)

```bash
# Python 3.12+ required (tinygrad needs match statements)
/opt/homebrew/bin/python3.12 -m pip install mlx mlx-lm scipy numpy --break-system-packages

# Clone TQBridge
git clone https://github.com/CG-8663/turboquant-tinygrad-bridge
cd turboquant-tinygrad-bridge
git submodule update --init

# Build C monitor and menu
cc -O2 -o deploy/bin/tqbridge-monitor deploy/src/tqbridge-monitor.c -lpthread
cc -O2 -o deploy/bin/tqbridge-menu deploy/src/tqbridge-menu.c -lpthread

# For eGPU (RTX via TB5): install TinyGPU.app
# tinygrad auto-downloads it on first Device["NV"] access
```

### Linux NVIDIA node (CUDA inference + decode server)

```bash
# Build tqbridge-server (C, no dependencies)
cc -O2 -o tqbridge-server tqbridge-server.c tqbridge.c tqbridge_net.c -lm -lpthread

# Self-test
./tqbridge-server --self-test

# Start decode server
./tqbridge-server --port 9473 --head-dim 128 &

# Build llama.cpp with CUDA
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc)
```

### Download a model

```bash
# Any GGUF model works. We tested with Qwen3-8B Q8_0:
# Download from HuggingFace or use llmfit
llmfit download qwen3:8b
```

## Running the Tests

### 1. Monitor (Terminal 1)

```bash
deploy/bin/tqbridge-monitor
```

Shows cluster status, GPU detection, node health.

### 2. Round-Robin GPU Test

```bash
/opt/homebrew/bin/python3.12 benchmarks/cluster_roundrobin.py --tokens 50
```

Tests each GPU sequentially: M3 MLX → GX10 CUDA → RTX tinygrad → M1 MLX.

### 3. GX10 CUDA Benchmark (with GPU proof)

```bash
# On the NVIDIA Linux node:
nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw \
  --format=csv,noheader -l 1 > /tmp/gpu_log.csv &

llama-bench -m /path/to/model.gguf -ngl 99 -p 512 -n 128

pkill -f nvidia-smi
cat /tmp/gpu_log.csv  # Should show 96% GPU utilization
```

### 4. RTX eGPU with BEAM Optimization

```bash
# BEAM=2 searches for optimal kernel implementations (one-time, cached)
cd tinygrad
DEV=NV BEAM=2 python3.12 -m tinygrad.apps.llm \
  --model /path/to/model.gguf --benchmark 20

# After BEAM: 53 tok/s decode (up from 6.8 without BEAM)
```

### 5. TQBridge Pipeline (sustained)

```bash
# Start decode servers on all nodes first:
# On each NVIDIA node: ./tqbridge-server --port 9473 --head-dim 128 &

# Run sustained pipeline
/opt/homebrew/bin/python3.12 benchmarks/sustained_bridge_test.py --scenario chat --duration 30

# Check server logs for proof of decompression:
ssh user@nvidia-node "tail /tmp/tqbridge.log"
```

### 6. Generative Cluster (real text generation)

```bash
# Start llama-server on NVIDIA node:
llama-server -m /path/to/model.gguf -ngl 99 --host 0.0.0.0 --port 8080

# Run generative test
/opt/homebrew/bin/python3.12 benchmarks/generative_cluster.py
```

### 7. Full Demo Menu

```bash
deploy/bin/tqbridge-menu
```

Interactive menu with all tests, llmfit model fitting, and visual demos.

## Key Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `DEV=NV` | Force tinygrad to use NVIDIA GPU | `DEV=NV python3 script.py` |
| `BEAM=2` | Search for fastest kernel implementations | `BEAM=2 DEV=NV python3 ...` |
| `LLMFIT_MODELS_DIR` | Override llmfit model download directory | `/Volumes/18TB-Mirror/models/gguf` |
| `LLAMA_CPP_PATH` | Point llmfit to llama.cpp binaries | `~/llama.cpp/build/bin` |

## Expected Results

These are our measured numbers. Your results will vary with hardware.

| Node | Model | Prefill | Decode | How to verify |
|------|-------|---------|--------|---------------|
| NVIDIA GB10 (CUDA) | Qwen3-8B Q8_0 | 2,030 tok/s | 28.5 tok/s | nvidia-smi shows 96% |
| NVIDIA GB10 (CUDA) | Qwen3.5-35B MoE | 1,857 tok/s | 55.6 tok/s | nvidia-smi shows 96% |
| M3 Ultra (MLX) | Qwen2.5-7B 4bit | — | 94.3 tok/s | macmon shows Metal active |
| M3 Ultra (MLX) | Qwen2.5-32B 4bit | 28 tok/s | 34.7 tok/s | macmon shows Metal active |
| M1 Max (MLX) | Qwen2.5-7B 4bit | 14 tok/s | 120.0 tok/s | macmon shows Metal active |
| RTX Blackwell (tinygrad TB5) | Qwen3-8B Q8_0 | 155.7 (pp512) | 53.0 (BEAM=2) | tqbridge-monitor shows temp rise |
| TQBridge compression | — | — | 9.8x (turbo3) | C driver self-test passes |
| TriAttention eviction | — | — | 1.1x (90% retention) | Tom Turney V3 validated |

## What's NOT Claimed

- No "50x" or "24,000x" compression — corrected to ~11x combined (9.8x TQ × 1.1x TriAtt)
- No "lossless" — TurboQuant is within 1.5% of baseline quality at 32K context
- TriAttention 10x is paper-only on reasoning workloads, not reproduced on general text
- RTX 6.8 tok/s (without BEAM) is a tinygrad + TB5 limitation, not the GPU
- Pipeline tok/s numbers are compression throughput, not model inference tok/s

## Credits

- **TurboQuant**: Google Research (PolarQuant), Tom Turney (TurboQuant+)
- **TriAttention**: Weian Mao et al.
- **tinygrad**: George Hotz (GPU runtime, TinyGPU eGPU driver)
- **llmfit**: Alex Jones (model fitting tool)
- **TQBridge**: James Tervit, Chronara Group
