# TurboQuant-Tinygrad Bridge

**Compressed KV cache as a cross-backend wire format for Metal + CUDA split inference**

---

## What This Is

This project is building the first cross-backend KV cache bridge that enables split LLM inference between Apple Metal (M3 Ultra) and NVIDIA CUDA (RTX PRO 6000 Blackwell via tinygrad direct) over Thunderbolt 5.

The core insight: TurboQuant's compressed KV format (`turbo3`/`turbo4`) is byte-identical on both Metal and CUDA backends -- same block structs from `ggml-common.h`, same WHT rotation seed, same Lloyd-Max codebook. Instead of building a custom serialisation layer, we use the **compressed KV cache itself as the wire format** for cross-device streaming. Compression isn't just saving bandwidth -- it's the bridge.

## Why This Was Started

### The Interconnect Wall

eGPU setups for LLM inference hit a hard bandwidth ceiling. Thunderbolt 5 PCIe 4.0 x4 tops out at ~6 GB/s -- two orders of magnitude below GPU memory bandwidth. For long-context inference with KV cache migration, this makes cross-device transfer the bottleneck.

Every existing heterogeneous inference system (PARALLAX, PetS, FlexGen, tinygrad multi-device) treats the interconnect as a fixed cost and tries to minimise crossings. Nobody has used **compression as the bandwidth strategy**.

### The X Posts That Started It All

This project started from a conversation on X (Twitter) between [@__tinygrad__](https://x.com/__tinygrad__) and the TurboQuant community. TheTom posted about the Metal + CUDA kernel work and tagged tinygrad:

> *"that setup is perfect for turboquant. we have metal kernels and cuda kernels now so both the M3 and the RTX Pro 6000 can run compressed KV cache. turbo3 at 4.6x compression means way less data moving over that TB5 link during inference"*
>
> *"if you're splitting layers between Mac and eGPU, smaller KV cache = less cross-device transfer per token. that's where the compression wins even if compute isn't the bottleneck"*

And then the key observation that defined this project:

> *"heads up though: right now metal and cuda work independently. there's no cross-backend bridge yet for split inference across both devices. that would need some work to coordinate the KV cache across the TB5 link. would love to experiment with that if i ever get my hands on one of these setups"*
>
> *`-ctk turbo3 -ctv turbo3` and go*

The links shared:
- Fork with Metal + CUDA support: https://github.com/TheTom/llama-cpp-turboquant
- Benchmark scripts + papers + results: https://github.com/TheTom/turboquant_plus

That gap -- "no cross-backend bridge yet" -- is exactly what this project fills.

### The Opportunity

TheTom's [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) fork added Metal and CUDA kernels for TurboQuant compression (`kernel_set_rows_turbo3/4` on Metal, `set-rows.cu` on CUDA). Both backends can independently compress and decompress KV cache at up to 4.6x compression versus FP16.

The turbo KV format is identical bytes on both backends -- same struct in `ggml-common.h`, same WHT rotation seed. Cross-device deserialisation is architecturally clean. And the compression works in your favour twice: less data to serialise AND less data moving over TB5 during inference. Up to 4.6x less versus FP16, about 2.4x versus Q8_0.

Using compressed KV as the wire format for cross-backend streaming is a new angle. tinygrad and others have done heterogeneous multi-device inference but not with compression as the bandwidth strategy. This could make eGPU setups actually practical for long-context workloads where the interconnect is normally the wall.

### The Hardware

| Node | Memory | Backend | Link |
|------|--------|---------|------|
| Mac M3 Ultra | 96 GB unified | Metal / MLX | -- (host) |
| RTX PRO 6000 Blackwell (300W) | 96 GB GDDR7 | tinygrad direct | TB5 PCIe 4.0 x4 (~6 GB/s) |
| **Combined** | **192 GB** | **Heterogeneous** | |

The RTX PRO 6000 uses the GB202 die (same as RTX 5090), which tinygrad already supports. The GPU is addressed via tinygrad's `PCIIface` backend using IOKit on macOS -- no Linux host, no NVIDIA drivers, no CUDA runtime required.

This setup puts 192 GB of compute memory on a single desk, spanning two fundamentally different memory architectures. The bridge makes them work as one.

### Why Not Just Use One Device?

A single M3 Ultra (96 GB) can run 70B models comfortably. But 405B models (224 GB at Q4_K_M) require more memory than any single consumer device offers. By bridging Metal and CUDA:

- **192 GB combined pool** -- enough for 405B Q4_K_M with headroom
- **Extends M3 Ultra into M5 territory** -- matching projected M5 Ultra memory capacity (~200 GB) today, on current hardware
- **No cloud, no data egress** -- sovereign inference for sensitive workloads

## Architecture: Double-Buffer Compressed KV Streaming

```
                    TB5 PCIe 4.0 x4 (~6 GB/s)
                    =============================

  Metal (M3 Ultra)                              CUDA (RTX PRO 6000)
  +--------------+                              +--------------+
  | Compress      |    +---------+              | Decompress    |
  | KV layer N    |--->| Buf A   |---- DMA ---->| KV layer N    |--> Compute
  |               |    +---------+              |               |
  | Compress      |    +---------+              | Decompress    |
  | KV layer N+1  |--->| Buf B   |---- DMA ---->| KV layer N+1  |--> (queued)
  +--------------+    (ping-pong)               +--------------+
```

The double-buffer pipeline overlaps DMA with compute:
- **Producer (Metal):** compresses KV layer via `kernel_set_rows_turbo3`, strips block padding, writes to ring buffer
- **DMA:** stripped turbo3 blocks traverse TB5 as raw bytes -- no interpretation needed
- **Consumer (CUDA):** restores padding, decompresses, loads into tinygrad KV cache

### Why Compression Changes Everything

| Format | KV per layer (70B, 32K ctx) | 80-layer transfer @ 6 GB/s |
|--------|---------------------------|--------------------------|
| FP16 | ~32 MB | ~427 ms |
| Q8_0 | ~16 MB | ~213 ms |
| **turbo3 (stripped)** | **~0.86 MB** | **~11 ms** |

With turbo3 stripped wire format + double buffering, the pipeline is **compute-bound, not DMA-bound**. The interconnect wall becomes a speed bump.

### Quality Impact

From TheTom/turboquant_plus benchmarks (M5 Max 128GB, Metal):

| Cache | Compression | PPL (wikitext-2) | vs q8_0 |
|-------|-------------|-----------------|---------|
| q8_0 | 1.9x | 6.111 | baseline |
| turbo4 | 3.8x | 6.125 | +0.23% |
| turbo3 | 4.6x | 6.176 | +1.06% |

Needle-In-A-Haystack retrieval: 100% accuracy through 32K context for both turbo3 and turbo4.

## Two Approaches: Single Bridge and Multi-Cluster

This project defines two deployment architectures that share the same compressed KV wire format but operate at different scales.

### Approach 1: Single Bridge (M3 Ultra + eGPU)

The simplest and most immediate configuration. One Mac, one eGPU, one TB5 cable.

```
  Mac M3 Ultra (96 GB)              Razer Core X V2
  +------------------+              +------------------+
  | Metal / MLX      |  TB5 PCIe   | RTX PRO 6000     |
  | Layers 0 -> M    |<----------->| Layers M -> N    |
  | KV: turbo3/4     |  ~6 GB/s    | KV: turbo3/4     |
  | (host + bridge)  |             | tinygrad direct   |
  +------------------+              +------------------+
        192 GB combined · 2 backends · 1 link
```

**How it works:**
- Layers are split proportionally by memory: M3 Ultra takes ~50% (layers 0-39 for 80-layer 70B), RTX PRO 6000 takes ~50% (layers 40-79)
- During inference, only hidden state activations (16 KB/token) cross the TB5 link per decode step -- negligible latency
- For KV cache migration (rebalancing, prefill handoff), the double-buffer pipeline streams compressed KV layer-by-layer over TB5
- The M3 Ultra runs the bridge coordinator: compress on Metal, DMA stripped blocks, decompress on CUDA
- No network, no cluster orchestration, no drivers -- just two devices on one desk

**What it gets you:**
- 192 GB pool -- run 405B Q4_K_M locally
- M5 Ultra-class memory capacity on current M3 hardware
- Sovereign inference: no cloud, no data egress
- Single point of failure (TB5 link), simple recovery (fall back to Metal-only)

**Best for:** Individual researchers, small teams, privacy-sensitive workloads, model experimentation at 405B scale.

### Approach 2: Multi-Cluster (exo + RDMA + eGPU)

The same compressed KV bridge scales to a multi-node cluster by adding RDMA-connected Mac nodes via [exo](https://github.com/exo-explore/exo). Each node can optionally attach its own eGPU.

```
  +------------------+   RDMA / MLX Jaccl   +------------------+
  | Mac M3 Ultra     |   (~10 GB/s, TB5)    | Mac M1 Max       |
  | 96 GB · Metal    |<------------------->| 32 GB · Metal    |
  | Layers 0 -> A    |                      | Layers A -> B    |
  | + eGPU bridge    |                      | (or + own eGPU)  |
  +--------+---------+                      +------------------+
           |
           | TB5 PCIe (~6 GB/s)
           |
  +--------+---------+
  | RTX PRO 6000     |
  | 96 GB · CUDA     |
  | Layers B -> N    |
  | tinygrad direct  |
  +------------------+

  Total: 224+ GB · 3+ nodes · 2 link types · 2+ backends
```

**How it works:**
- exo manages the cluster topology: node discovery, layer assignment, inference orchestration
- RDMA links (MLX Jaccl over TB5) handle Mac-to-Mac hidden state transfer at ~10 GB/s with ~5 us RTT
- The TurboQuant bridge handles Metal-to-CUDA KV streaming within each host that has an eGPU
- Both link types benefit from compression: RDMA KV migration uses turbo3 (12 ms vs 64 ms for 20 layers at 32K context), PCIe eGPU uses the double-buffer pipeline
- Layer assignment is proportional to memory across all nodes: a 3-node setup with 96+32+96=224 GB can run 405B Q4_K_M

**The key insight for multi-cluster:** the compressed wire format is the same regardless of transport. Whether KV travels over RDMA between two Macs or over PCIe to an eGPU, it's the same stripped turbo3/turbo4 blocks. The bridge code doesn't care about the link -- it just produces and consumes compressed bytes.

**What it gets you:**
- 224+ GB pool across heterogeneous nodes
- Linear memory scaling: add more Macs or eGPUs to grow the pool
- Each node contributes proportional compute -- no idle hardware
- Mix of link types (RDMA + PCIe) handled transparently by the same wire format

**Scaling further:**
- Add more Mac nodes via RDMA for memory + Metal compute
- Add more eGPUs (one per TB5 port) for CUDA compute density
- Each eGPU bridge is independent -- no cross-eGPU coordination needed
- exo's Spark integration (in progress by Alex Cheema) will handle cluster-level orchestration

**Best for:** Research labs, teams running multiple large models, production-adjacent inference, scaling beyond 400 GB.

### How the Approaches Connect

Approach 1 is a subset of Approach 2. The single bridge is the building block:

1. **Build and validate the single bridge first** (this project's current scope)
2. **Plug it into exo** when Spark integration lands -- the bridge becomes one node in a larger topology
3. **Scale by adding nodes** -- each with its own optional eGPU bridge

The compressed wire format, ring buffer, and stream coordinator are identical in both approaches. The only difference is who calls `stream_kv_migration`: in Approach 1 it's the local host; in Approach 2 it's exo's cluster orchestrator.

---

## Project Status

**Phase: Pre-hardware validation**

The RFC is drafted, the architecture is designed, the test strategy is defined. Implementation is blocked on building the physical rig (M3 Ultra + RTX PRO 6000 in Razer Core X V2).

### Prerequisites
- [ ] tinygrad micro-PR: Add RTX PRO 6000 Blackwell Max-Q PID to GB202 device table (`tinygrad/runtime/ops_nv.py:544`)
- [ ] Build eGPU rig: RTX PRO 6000 + Razer Core X V2 + 650W+ ATX PSU

### Sprint 0 -- Hardware Validation (Go/No-Go Gates)
- [ ] tinygrad enumerates RTX PRO 6000 via PCIIface/IOKit
- [ ] TB5 sustained throughput >= 5 GB/s
- [ ] turbo3/turbo4 compress/decompress on Metal (M3 Ultra)
- [ ] turbo3/turbo4 compress/decompress on CUDA (RTX PRO 6000)
- [ ] Lloyd-Max codebook consistency between Metal and CUDA implementations

### Implementation
- [ ] Ring buffer for double-buffered KV streaming
- [ ] KV stream coordinator using `KVCacheCompressor` API
- [ ] Wire format: stripped turbo3/turbo4 blocks + layer header
- [ ] Cross-backend round-trip integration tests
- [ ] End-to-end benchmarks

## Repository Structure

```
docs/
  RFC-metal-cuda-kv-bridge.md       # Parent RFC: hardware topology, layer assignment
  RFC-double-buffer-kv-bridge.md    # This RFC: compressed KV streaming bridge
  RFC-double-buffer-kv-bridge.pdf   # PDF version with cover page and TOC
```

## Key References

| Resource | Link |
|----------|------|
| TurboQuant paper (ICLR 2026) | [arXiv:2504.19874](https://arxiv.org/abs/2504.19874) |
| Metal + CUDA fork (TheTom) | [llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) |
| Benchmarks + papers (TheTom) | [turboquant_plus](https://github.com/TheTom/turboquant_plus) |
| tinygrad | [tinygrad/tinygrad](https://github.com/tinygrad/tinygrad) |
| exo (future integration) | [exo-explore/exo](https://github.com/exo-explore/exo) |
| ggml block formats | [ggml-quants.h](https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.h) |

## Acknowledgements

- **TheTom** -- Metal and CUDA TurboQuant kernels, turbo4 resurrection, benchmark infrastructure
- **George Hotz / tinygrad** -- Direct GPU backend enabling driverless NVIDIA access on macOS
- **Alex Cheema / exo** -- Distributed inference framework (future integration target)
- **Google Research** -- TurboQuant paper (arXiv:2504.19874)

## License

MIT

---

*Chronara Group Limited | AS200840*
