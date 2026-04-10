# Generative Inference Mesh — Architecture

## Vision

A Docker-based distributed inference system where any machine — Mac, NVIDIA GPU, cloud instance, edge device — joins the mesh with a single command and contributes to generative AI inference.

```
docker run -p 9473:9473 chronaragroup/chronara-bridge
```

The mesh handles model splitting, KV cache compression, token eviction, and secure transport automatically.

## Stack

```
┌─────────────────────────────────────────────────────┐
│                  User Request                        │
├─────────────────────────────────────────────────────┤
│            Orchestrator (M3 Ultra)                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ TriAttention │  │  TurboQuant  │  │  KV Router │ │
│  │ (evict)      │→│  (compress)  │→│  (distribute)│ │
│  └─────────────┘  └──────────────┘  └────────────┘ │
├──────────┬──────────┬──────────┬────────────────────┤
│          │          │          │                      │
│  RTX 6000│  GX10-001│  GX10-002│    Docker nodes     │
│  (prefill│  (decode)│  (decode)│    (decode)          │
│   + CUDA)│  (200GbE)│  (200GbE)│    (anywhere)        │
└──────────┴──────────┴──────────┴────────────────────┘
```

## Compression Pipeline

Each token's KV cache goes through two stages before transfer:

### Stage 1: TriAttention (Token Eviction)
- Scores every KV token using trigonometric frequency-domain analysis
- Evicts lowest-scoring tokens (redundant thinking traces, repeated patterns)
- 90% retention on standard models, 97.5% on hybrid (Mamba+Attention)
- Result: fewer tokens in the cache

### Stage 2: TurboQuant (Bit Compression)
- Compresses remaining tokens via PolarQuant + asymmetric K/V
- Q8_0 for keys (precision), turbo3 for values (compression)
- 4.6x compression ratio
- Result: fewer bits per token

### Combined Effect

| Workload | TriAttention | TurboQuant | Combined | Per-token over wire |
|----------|-------------|-----------|----------|---------------------|
| Short context (<2K) | 1x (skip) | 4.6x | 4.6x | ~11 KB |
| Standard 7B, 32K | 1.11x (90%) | 4.6x | 5.1x | ~10 KB |
| Reasoning 7B, 32K | ~5x (est.) | 4.6x | ~23x | ~2.2 KB |
| Hybrid 27B, 32K | 1.03x (97.5%) | 4.6x | 4.7x | ~11 KB |

At 2.2 KB/token on reasoning workloads, the mesh works over WiFi.

## Transport Security

The TQKV wire protocol supports pluggable transport:

| Layer | Current | Future |
|-------|---------|--------|
| Transport | Raw TCP | TLS 1.3 + post-quantum (ML-KEM/Kyber) |
| Addressing | IPv4 | IPv6 (globally routable mesh nodes) |
| Authentication | None | mTLS with node certificates |
| Integrity | CRC32 (header) | AEAD (full payload) |

IPv6 eliminates NAT traversal — every Docker node has a routable address. Post-quantum key exchange future-proofs the mesh against quantum computing attacks on the model's inference traffic.

## Node Types

| Type | Role | Requirements | Deploy |
|------|------|-------------|--------|
| **Prefill** | Run model, prefill context, compress KV | GPU (NVIDIA/Metal), model weights | Manual setup |
| **Decode** | Receive compressed KV, decode tokens | Docker only | `docker run chronaragroup/chronara-bridge` |
| **Orchestrator** | Route requests, manage mesh topology | Python + TQBridge library | `pip install tqbridge` |
| **Edge** | Low-latency local decode | Any CPU | Docker or native binary |

## GPU Support

| Backend | Compress | Decompress | Docker decode |
|---------|----------|------------|---------------|
| NVIDIA CUDA (sm_61-sm_120) | ✅ 4,117 tok/s | ✅ 4,117 tok/s | ✅ CPU fallback |
| Apple Metal (M1-M5) | ✅ 3,384 tok/s | ✅ 3,384 tok/s | ✅ CPU fallback |
| AMD HIP (RDNA/CDNA) | ✅ Ready | ✅ Ready | ✅ CPU fallback |
| Intel OpenCL (Arc/UHD) | ✅ Ready | ✅ Ready | ✅ CPU fallback |
| CPU (any) | ✅ 295 tok/s | ✅ 295 tok/s | ✅ Native |

## Current Status

| Component | Status | Performance |
|-----------|--------|-------------|
| TQBridge (compression + transfer) | ✅ Production | 531 tok/s (TB5), 4117 tok/s (PCIe) |
| Docker decode node | ✅ Production | 1 MB image, zero deps |
| Multi-node router (TCP) | ✅ Production | Error handling + retries |
| CUDA/Metal/C kernels | ✅ Production | 159 tests passing |
| TriAttention integration | 🔄 In progress | MLX calibration running |
| IPv6 + post-quantum transport | 📋 Planned | Wire protocol ready |
| AMD/Intel GPU kernels | 📋 Ready (untested) | HIP + OpenCL written |
| vGPU support | 📋 Ready | Via underlying backend |
