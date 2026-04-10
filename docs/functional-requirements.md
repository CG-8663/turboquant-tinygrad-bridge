# TQBridge — Functional Requirements for Team Review

## Current Status (Validated)

| Feature | Status | Evidence |
|---------|--------|----------|
| KV compression (TurboQuant) | ✅ Production | 9.8x ratio, bit-exact C↔Python |
| Custom GPU kernels (CUDA/Metal) | ✅ Production | 4117/3384 tok/s |
| Bridge Metal→NV (TB5) | ✅ Production | 484-531 tok/s |
| Multi-node router (TCP) | ✅ Production | Error handling + retries |
| Docker decode node | ✅ Production | 1MB image, multi-arch |
| TriAttention integration | ✅ Validated | MLX, norm-only scoring |
| Generative pipeline | ✅ Validated | 7B + 27B models |
| AMD HIP kernels | ✅ Written | Untested on hardware |
| Intel OpenCL kernels | ✅ Written | Untested on hardware |
| C decode server | ✅ Production | 73KB binary, zero deps |
| 159 Python + 64 C tests | ✅ Passing | Live hardware validated |

## FR-1: Decentralised Mesh (Priority: HIGH)

Docker-based peer mesh where nodes discover each other and self-organise.

| Requirement | Detail |
|-------------|--------|
| FR-1.1 | Peer discovery via mDNS or DHT on local network |
| FR-1.2 | Secure node join via mTLS (mesh CA issues certs) |
| FR-1.3 | IPv6 native addressing (no NAT traversal) |
| FR-1.4 | Automatic layer assignment based on node capabilities |
| FR-1.5 | Node health monitoring and failover |
| FR-1.6 | Graceful node departure (redistribute layers) |

## FR-2: Secure Transport (Priority: HIGH)

All KV cache traffic encrypted and authenticated.

| Requirement | Detail |
|-------------|--------|
| FR-2.1 | TLS 1.3 minimum for all mesh traffic |
| FR-2.2 | Mutual TLS (mTLS) — both nodes must present valid certs |
| FR-2.3 | Post-quantum key exchange (ML-KEM/Kyber) ready |
| FR-2.4 | Per-node certificate management (issue, revoke, rotate) |
| FR-2.5 | IPv6 dual-stack support |

## FR-3: Generative Inference Pipeline (Priority: HIGH)

End-to-end generative inference across the mesh.

| Requirement | Detail |
|-------------|--------|
| FR-3.1 | TriAttention token eviction before compression |
| FR-3.2 | TurboQuant KV compression (asymmetric K/V) |
| FR-3.3 | Support reasoning models (`<think>` traces, long generation) |
| FR-3.4 | Model splitting across nodes (layer assignment) |
| FR-3.5 | Prefill on high-compute node, decode on fleet |
| FR-3.6 | KV cache streaming (not batch — per-token transfer) |

## FR-4: Multi-GPU Backend (Priority: MEDIUM)

Support all major GPU vendors.

| Requirement | Detail |
|-------------|--------|
| FR-4.1 | NVIDIA CUDA (sm_61 through sm_120) — validated |
| FR-4.2 | Apple Metal (M1-M5) — validated |
| FR-4.3 | AMD HIP (RDNA/CDNA) — kernels written, needs hardware test |
| FR-4.4 | Intel OpenCL (Arc/UHD/Iris) — kernels written, needs test |
| FR-4.5 | vGPU support (NVIDIA GRID, AMD MxGPU, Intel GVT-g) |
| FR-4.6 | iGPU/unified memory (80% limit for OS stability) |
| FR-4.7 | GPU auto-detection and backend selection |

## FR-5: Docker Deployment (Priority: HIGH)

One-command deployment for decode nodes.

| Requirement | Detail |
|-------------|--------|
| FR-5.1 | Multi-arch image (amd64 + arm64) |
| FR-5.2 | FROM scratch (minimal attack surface, <2MB) |
| FR-5.3 | Pre-compiled for all supported architectures |
| FR-5.4 | GPU passthrough for accelerated decompress |
| FR-5.5 | Docker Compose for multi-node clusters |
| FR-5.6 | Health check endpoint for orchestration |

## FR-6: Older GPU Support (Priority: MEDIUM)

Enable participation of legacy hardware in the mesh.

| Requirement | Detail |
|-------------|--------|
| FR-6.1 | Pascal eGPU (GTX 10-series) via CPU decode path |
| FR-6.2 | Ampere+ eGPU via TinyGPU/GSP (full GPU compute) |
| FR-6.3 | CPU-only decode on any hardware (native C, 295 tok/s) |
| FR-6.4 | Automatic fallback: GPU kernels → CPU if unavailable |

## FR-7: exo Integration (Priority: MEDIUM)

Integration with exo-explore/exo for heterogeneous cluster inference.

| Requirement | Detail |
|-------------|--------|
| FR-7.1 | TQBridge as transport layer for exo engine shards |
| FR-7.2 | Compressed KV cache topic for pub/sub routing |
| FR-7.3 | Process isolation for BEAM kernel tuning |
| FR-7.4 | Kernel warmup at startup (eliminates 26s→745ms TTFT) |

## FR-8: Observability (Priority: LOW)

Monitoring and debugging for production deployments.

| Requirement | Detail |
|-------------|--------|
| FR-8.1 | Per-node throughput metrics (tok/s, compression ratio) |
| FR-8.2 | Thermal monitoring with throttle gating |
| FR-8.3 | Transfer latency histogram |
| FR-8.4 | KV cache memory usage per node |
| FR-8.5 | Mesh topology visualisation |
