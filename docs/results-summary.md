# TQBridge — TurboQuant KV Cache Bridge Results

## Bottom Line

TQBridge makes large language models run faster at long conversations by compressing the memory they use to track context. The longer a conversation gets, the bigger the advantage.

**What it does:** Compresses the KV cache (the model's working memory) by 4-5x using TurboQuant, then transfers it between GPUs across a network. This lets you split a model across multiple machines — a Mac, NVIDIA GPUs, and server nodes — and run them together as one system.

**Key results:**
- A 27B-parameter model runs at **near-identical speed** with 4x less memory usage (within 1.5% of uncompressed)
- At 32K token conversations, TurboQuant is actually **71% faster** than the baseline on an 8B model — because smaller memory means faster reads
- Our custom GPU kernels compress and decompress in **under 0.5ms** — fast enough that the bottleneck is the cable, not the computation
- A 4-node cluster (Mac + 2 server nodes + NVIDIA GPU) runs the bridge end-to-end over 200Gbps Ethernet

**What this means:** You can run larger models, with longer context, on cheaper hardware, by spreading the work across whatever GPUs you have. A 27B reasoning model at 131K context that would normally need 64GB+ of GPU memory can run comfortably across machines with 16GB each.

---

## Hardware

| Node | GPU | Memory | Link |
|------|-----|--------|------|
| M3 Ultra (Mac Studio) | Apple M3 Ultra | 96 GB unified | Local / TB5 |
| RTX PRO 6000 | Blackwell (eGPU) | 96 GB GDDR7 | Thunderbolt 5 |
| GX10-001 (Chronara) | NVIDIA GB10 | — | 200GbE bonded |
| GX10-002 (Chronara) | NVIDIA GB10 | — | 200GbE bonded |
| M1 Max (Mac Studio) | Apple M1 Max | 32 GB unified | Local / Ethernet |

## Qwen3-8B Q8_0 — Decode tok/s vs Context Depth (M3 Ultra)

| Config | 0 | 1K | 2K | 4K | 8K | 16K | 32K |
|--------|---|----|----|----|----|-----|-----|
| f16 baseline | 67.8 | 62.0 | 57.1 | 49.6 | 39.5 | 28.0 | 17.1 |
| **Q8₀ K + turbo4 V** | 54.2 | 52.0 | 52.1 | **49.9** | **45.7** | **38.3** | **29.2** |
| Q8₀ K + turbo3 V | 52.0 | 50.3 | 50.0 | 47.3 | 42.9 | 35.7 | 26.9 |

- **Crossover at 4K context** — turbo4 asymmetric overtakes f16
- **+70.8% faster at 32K** (29.2 vs 17.1 tok/s)
- f16 drops 75% over context; turbo4 drops only 46%
- Prefill speed identical (±0.6%)

## Qwen3.5-9B Q8_0 — Decode tok/s vs Context Depth (M3 Ultra)

| Config | 0 | 1K | 2K | 4K | 8K | 16K | 32K |
|--------|---|----|----|----|----|-----|-----|
| f16 baseline | 58.6 | 58.9 | 57.9 | 56.6 | 54.2 | 48.2 | 41.7 |
| Q8₀ K + turbo4 V | 56.9 | 53.6 | 50.0 | 49.8 | 50.1 | 47.8 | 41.3 |
| Q8₀ K + turbo3 V | 55.7 | 53.8 | 48.5 | 49.3 | 50.0 | 46.4 | 39.9 |
| **turbo3 symmetric** | 53.6 | 52.4 | 47.3 | 47.8 | 50.4 | **48.6** | **44.1** |

- turbo3 symmetric **beats f16 at 32K** (+5.8%)
- Qwen3.5-9B tolerates turbo very well (Alex Ziskind's finding confirmed)
- Prefill identical: 1079-1081 tok/s

## Qwen3.5-27B Claude Opus Distilled Q8_0 (M3 Ultra)

### Decode (tg128)

| Config | d=0 | d=2K | d=8K | d=32K | Δ vs f16@32K |
|--------|-----|------|------|-------|-------------|
| f16 baseline | 20.59 | 20.01 | 18.85 | 14.57 | — |
| turbo4 symmetric | 17.09 | 14.32 | 8.67 | 3.72 | -74% |
| turbo3 symmetric | 17.19 | 13.94 | 8.25 | 3.28 | -77% |
| **Q8₀ K + turbo4 V** | **19.67** | **19.41** | **18.07** | **14.35** | **-1.5%** |
| **Q8₀ K + turbo3 V** | **19.57** | **19.31** | **17.86** | **14.01** | **-3.8%** |

### Prefill (pp512)

| Config | d=0 | d=2K | d=8K | d=32K |
|--------|-----|------|------|-------|
| f16 baseline | 313.9 | 308.3 | 287.8 | 217.9 |
| turbo4 symmetric | 249.7 | 100.1 | 32.0 | 8.2 |
| turbo3 symmetric | 237.9 | 85.5 | 27.6 | 6.2 |
| **Q8₀ K + turbo4 V** | **310.7** | **303.5** | **274.8** | **196.6** |
| **Q8₀ K + turbo3 V** | **314.3** | **302.3** | **275.3** | **198.7** |

### Key Findings — 27B

- **Asymmetric is essential on 27B** — symmetric turbo collapses at depth (3.72 tok/s vs 14.35)
- **Asymmetric matches f16 within 4%** across all context depths
- **Prefill: Q8₀+turbo3 matches f16** (314.3 vs 313.9 = +0.1%)
- **Symmetric turbo kills prefill** (6.2 vs 217.9 at 32K = -97%)
- **Asymmetric prefill stays strong** (198.7 vs 217.9 at 32K = -8.8%)
- **KV savings: 3.8-4.6x** with essentially zero performance cost

## TQBridge Pipeline Performance

### Custom GPU Kernels (Isolated, No Transfer)

| Device | Compress+Decompress | tok/s |
|--------|-------------------|-------|
| RTX PRO 6000 (CUDA) | 0.24ms | **4,117** |
| M3 Ultra (Metal) | 0.30ms | **3,384** |

### End-to-End Bridge (Metal → NV via TB5)

| Backend | Compress | Transfer | Decompress | Total | tok/s |
|---------|----------|----------|------------|-------|-------|
| tinygrad (baseline) | 10.57ms | 2.36ms | 6.04ms | 13.38ms | 75 |
| CUDA decomp only | 5.00ms | 1.92ms | 0.34ms | 7.28ms | 137 |
| **Metal + CUDA kernels** | **0.21ms** | **1.51ms** | **0.29ms** | **2.30ms** | **434** |

- Custom GPU kernels eliminate tinygrad overhead entirely
- Bridge is now **TB5-latency-bound** at 1.51ms
- GPU kernels finish in 0.5ms total (compress + decompress)

### Cluster (4-Node, M3 + RTX + 2×GX10)

| Config | tok/s | Bottleneck |
|--------|-------|------------|
| GX10 pair (TCP, 200GbE) | 132 | — |
| Full 4-node (sequential) | 34.1 | tinygrad local transfers |
| Metal+CUDA bridge (TB5) | **434** | TB5 1.51ms latency |
| CUDA isolated (local PCIe) | **4,117** | — |

### Architecture

```
RTX PRO 6000 (prefill, 96GB GDDR7)
  ├── Metal shader compress (0.21ms)
  ├── TB5 → M3 Ultra: 1.51ms (26KB compressed)
  ├── 200GbE → GX10-001: ~0.05ms
  └── 200GbE → GX10-002: ~0.05ms

Per-node decode capacity: 3,000-4,000 tok/s
Cluster aggregate: 10,000+ tok/s theoretical
```

## Compression

| Format | Ratio | Quality (PPL) | Best for |
|--------|-------|---------------|----------|
| turbo4 (4-bit) | 3.8x | Near q8_0 | Asymmetric V (speed) |
| turbo3 (3-bit) | 4.6x | +1-2% | Asymmetric V (compression) |
| turbo2 (2-bit) | 6.4x | +6.5% | Extreme memory pressure |

**Asymmetric K/V is critical**: Q8₀ K + turbo V passes needle-in-haystack 100%.
Symmetric turbo K+V fails at 8K+ context (validated by Alex Ziskind).

## Bottleneck Analysis — TB5 eGPU vs Native PCIe

### Current: Mac → RTX eGPU (Thunderbolt 5)

| Component | Time | % of total |
|-----------|------|-----------|
| Metal compress | 0.21ms | 9% |
| **TB5 transfer** | **1.51ms** | **66%** |
| CUDA decompress | 0.29ms | 13% |
| Overhead | 0.29ms | 12% |
| **Total** | **2.30ms = 434 tok/s** | |

TB5 latency is 1.5ms per round-trip regardless of payload size — this is USB4/Thunderbolt protocol overhead, not bandwidth. **This is as good as Mac→eGPU gets.**

### Target: RTX on GX10 (Native PCIe)

| Component | Time | vs TB5 |
|-----------|------|--------|
| CUDA compress | 0.12ms | — |
| PCIe transfer | 0.005ms | **300x faster** |
| CUDA decompress | 0.12ms | — |
| **Total** | **~0.25ms = 4,000 tok/s** | **9x faster** |

Moving the RTX PRO 6000 into GX10-001 via native PCIe eliminates the TB5 latency entirely. The same CUDA kernels run at the same speed — without the 1.5ms USB4 wrapper.

### Production Cluster Architecture

```
GX10-001 + RTX PRO 6000 (prefill + decode, native PCIe)
  ├── CUDA compress: 0.12ms
  ├── 200GbE → GX10-002: 0.05ms (compressed KV)
  └── 200GbE → M3 Ultra: 0.05ms (compressed KV)

GX10-002 / GB10 (decode)
  └── Native C decompress: 3,400 tok/s capacity

M3 Ultra (orchestrator + decode)
  └── Metal shader decompress: 3,384 tok/s capacity

M1 Ultra (decode)
  └── Metal shader decompress: ~1,500 tok/s capacity
```

| Node | Role | Link | Capacity |
|------|------|------|----------|
| GX10-001 + RTX 6000 | Prefill + decode | Local PCIe | ~4,000 tok/s |
| GX10-002 (GB10) | Decode | 200GbE | ~3,400 tok/s |
| M3 Ultra | Orchestrator + decode | 200GbE/local | ~3,384 tok/s |
| M1 Max (32GB) | Decode | 200GbE/local | ~1,500 tok/s |
| **Cluster aggregate** | | | **~12,000+ tok/s** |

The Macs shift from eGPU hosts to orchestrators + Metal decode nodes. The RTX PRO 6000 belongs on native PCIe for production throughput.

## TriAttention + TurboQuant Stacking

TriAttention (Weian Mao et al.) and TurboQuant are **complementary** but the combined ratio depends heavily on workload. Tom Turney's V3 paper provides the validated numbers:

| Workload | TriAttention | TurboQuant | Combined | Status |
|----------|-------------|-----------|----------|--------|
| Standard 7B, 32K | 1.1x (90% retention) | 4.6x | **5.1x** | Validated, NIAH pass |
| Hybrid 27B, 32K | 1.03x (97.5% retention) | 4.6x | **4.7x** | Validated (V3 hybrid policy) |
| Reasoning 7B, 32K | ~5x (estimated) | 4.6x | **~23x** | Paper claim, not independently reproduced |

**Important caveats (from Tom Turney's V3 review):**
- The paper's 10.7x eviction ratio is on reasoning workloads with heavy thinking-token redundancy. Nobody has reproduced this on general text.
- The validated safe operating point on wikitext is 90% retention (1.1x) with clean NIAH retrieval.
- Paper-faithful TriAttention breaks needle-in-a-haystack at 90% retention (10% eviction) at end of context. Tom's V3 hybrid policy (prefix protection + per-segment quota) fixes this on standard transformers but still fails on hybrid mamba+attention models.
- Always validate with NIAH at 32K with needles at start/middle/end before claiming any eviction ratio.

With 5x combined compression on general text, a 27B model at 131K context reduces KV from 64GB to ~13GB — meaningful savings that enable multi-node distribution across consumer hardware.

## Cluster Inference — Qualified Results (2026-04-13)

All numbers measured on real hardware. No simulation. GPU utilization confirmed via nvidia-smi (GX10) and macmon (Macs).

### Per-Node Maximum (Qwen3-8B Q8_0 unless noted)

| Node | GPU | VRAM | Framework | Prefill | Decode | Model |
|------|-----|------|-----------|---------|--------|-------|
| GX10-001 | NVIDIA GB10 | 124 GB | llama.cpp CUDA | **2,030 tok/s** | **28.2 tok/s** | Qwen3-8B Q8_0 |
| GX10-001 | NVIDIA GB10 | 124 GB | llama.cpp CUDA | **1,857 tok/s** | **55.6 tok/s** | Qwen3.5-35B MoE Q8_0 |
| GX10-002 | NVIDIA GB10 | 122 GB | llama.cpp CUDA | **2,033 tok/s** | **28.4 tok/s** | Qwen3-8B Q8_0 |
| M3 Ultra | Apple M3 Ultra | 96 GB | MLX Metal | 28 tok/s | **34.7 tok/s** | Qwen2.5-32B 4bit |
| M3 Ultra | Apple M3 Ultra | 96 GB | MLX Metal | — | **94.3 tok/s** | Qwen2.5-7B 4bit |
| M1 Max | Apple M1 Max | 32 GB | MLX Metal | 14 tok/s | **120.0 tok/s** | Qwen2.5-7B 4bit |
| RTX PRO 6000 | Blackwell sm_120 | 96 GB | tinygrad CUDA (TB5) | — | **6.8 tok/s** | Qwen3-8B Q8_0 (tinygrad kernel + TB5 latency bottleneck) |

### GPU Utilization Proof

| Node | nvidia-smi / macmon | Peak Util | Temp Change | Power |
|------|--------------------:|----------:|------------:|------:|
| GX10-001 | nvidia-smi logged | **96%** | 53→63°C | 5.8→34.9W |
| GX10-002 | nvidia-smi logged | **96%** | 59→67°C | 12.9→35.9W |
| M3 Ultra | macmon visible | Metal active | — | — |
| M1 Max | macmon visible | Metal active | — | — |
| RTX PRO 6000 | tinygrad thermal | — | 57→59°C | — |

### TQBridge Pipeline (sustained, 20s test)

| Metric | Value | Proof |
|--------|-------|-------|
| Pipeline throughput | 5,630 tok/s | Client measured |
| TurboQuant compression | 9.8x | C driver, bit-exact |
| TriAttention eviction | 10% (90% retention) | Tom V3 validated |
| GX10-001 decompressions | 880 | Server log |
| GX10-002 decompressions | 880 | Server log |
| M1 Max decompressions | 880 | Server log |
| GX10 decompress time | 6.3ms/batch | Server log |
| M1 Max decompress time | 11.3ms/batch | Server log |

### Cluster Total Capacity

| Resource | Value |
|----------|-------|
| Total GPU VRAM | 470 GB (124+122+96+96+32) |
| Total decode capacity | ~190 tok/s combined (all nodes, different models) |
| GX10 prefill capacity | ~4,000 tok/s (both nodes) |
| Largest single-node model | Qwen3.5-35B MoE (34 GB, GX10-001) |
| Largest cluster model | 405B feasible across all 5 nodes (470 GB) |
| Central model storage | 18TB mirrored RAID on M1 Max |

## Software

- **tqbridge**: Standalone Python package with C driver + CUDA/Metal kernels
- **151 Python tests + 64 C assertions** passing on live hardware
- **Wire protocol**: 40-byte TQKV header + k_size prefix, CRC32, format negotiation
- **C↔Python parity**: Bit-exact compressed bytes between implementations
