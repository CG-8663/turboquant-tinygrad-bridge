# RFC: Heterogeneous Split Inference Bridge
## TurboQuant + exo-explorer | Mac RDMA Cluster + tinygrad eGPU Direct Backend

**Status:** Draft  
**Author:** James Tervit, Chronara Group Limited (AS200840)  
**Date:** 2026-03-30  
**Repository target:** exo-explore/exo  
**Prerequisite micro-PR:** tinygrad — add RTX PRO 6000 Blackwell Max-Q PID to GB202 device table

---

## Hardware Topology

### Phase 1 (Current — in scope for this RFC)

```
┌──────────────────────────────────┐    RDMA over Thunderbolt 5     ┌──────────────────────────────────┐
│          M3 ULTRA NODE           │   (MLX Jaccl, macOS 26.2+)    │          M1 MAX NODE             │
│  Apple M3 Ultra                  │ ◄────────────────────────────► │  Apple M1 Max                    │
│  96 GB unified memory            │    ~10 GB/s · 5–9 µs RTT       │  32 GB unified memory            │
│  819 GB/s memory bandwidth       │                                │  400 GB/s memory bandwidth       │
│  Metal / MLX backend             │                                │  Metal / MLX backend             │
│  Layers: 0 → floor(N × 0.75)     │                                │  Layers: floor(N × 0.75) → N    │
│  KV cache: local · Q5_K_M        │                                │  KV cache: local · Q5_K_M       │
└──────────────────────────────────┘                                └──────────────────────────────────┘
  Primary node / exo master · 96 GB                                   Worker node · 32 GB
  Combined Phase 1 pool: 128 GB unified
```

### Phase 2 (eGPU — in scope for this RFC, pending PID micro-PR)

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              M3 ULTRA NODE                                           │
│                                                                                      │
│  ┌─────────────────────────┐          TB5 PCIe 4.0 x4 (~6 GB/s)                    │
│  │  MLX backend            │    ┌──────────────────────────────────────────────┐    │
│  │  96 GB unified          │    │  Razer Core X V2 eGPU enclosure              │    │
│  │  Layers 0 → M           │◄──►│  └── RTX PRO 6000 Blackwell Max-Q            │    │
│  │                         │    │       300W TDP · 96 GB GDDR7                 │    │
│  │  TurboQuant bridge      │    │       1,792 GB/s bandwidth                   │    │
│  │  (tensor handoff)       │    │       tinygrad direct backend                │    │
│  └─────────────────────────┘    │       GB202 die (same as RTX 5090)           │    │
│                                 │       PID: one-line addition to device table │    │
│                                 │       Layers M → N                           │    │
│                                 └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────────────┘
     RDMA ↕ MLX Jaccl
┌──────────────────────────────────┐
│          M1 MAX NODE             │
│  32 GB unified · Metal / MLX    │
│  Layers floor(N × 0.75) → N     │
│  (Phase 1 only, see §7.2)        │
└──────────────────────────────────┘
```

**Phase 2 combined pool:**

| Node | Memory | Backend | Transport to M3 Ultra |
|------|--------|---------|----------------------|
| M3 Ultra | 96 GB unified | Metal / MLX | — (master) |
| M1 Max | 32 GB unified | Metal / MLX | RDMA · MLX Jaccl · ~10 GB/s |
| RTX PRO 6000 Max-Q | 96 GB GDDR7 | tinygrad direct | TB5 PCIe 4.0 x4 · ~6 GB/s |
| **Total** | **224 GB** | Heterogeneous | |

---

## 1. Summary

This RFC proposes a split inference bridge enabling pipeline-parallel LLM inference across
heterogeneous nodes and backends using the `exo-explore/exo` framework.

**Phase 1** is a two-node Apple Silicon RDMA cluster: M3 Ultra (96 GB) + M1 Max (32 GB) connected
via MLX Jaccl RDMA over Thunderbolt 5 (macOS 26.2). This operates entirely within exo's existing
MLX Jaccl topology. The contribution is the asymmetric memory placement policy, Q5_K_M as the
recommended default, and a TurboQuant KV migration module for context handoff.

**Phase 2** adds the RTX PRO 6000 Blackwell Max-Q in a Razer Core X V2 eGPU enclosure, connected
to the M3 Ultra via TB5 (PCIe 4.0 x4). The GPU is addressed via **tinygrad's direct backend** —
no Linux host, no CUDA runtime, no NVIDIA macOS drivers required. The RTX PRO 6000 shares the
GB202 die with the RTX 5090, which tinygrad already supports; adding the RTX PRO 6000's PID to
tinygrad's device table is a one-line micro-PR prerequisite. `TinygradDynamicShardInferenceEngine`
already exists in exo — this is the natural CUDA-class backend for the eGPU node.

**TurboQuant** (Google Research, ICLR 2026) is the tensor bridge between the MLX backend (BF16,
unified memory) and the tinygrad backend (GDDR7 across PCIe 4.0 x4). It compresses KV cache
vectors to 2–4 bits for cross-backend handoff, and handles the format translation at the
memory domain boundary. This is TurboQuant's primary role in this architecture — not just
compression for bandwidth saving, but as the bridge enabling coherent KV state across two
fundamentally different memory models.

---

## 2. Motivation

### 2.1 Why tinygrad direct, not CUDA + Linux

tinygrad's direct backend talks to GPU hardware via libusb without requiring an OS-level compute
stack. For NVIDIA GB202 (RTX 5090, RTX PRO 6000 Blackwell Max-Q), tinygrad already has working
support. The RTX PRO 6000 Max-Q uses the same GB202 die, same architecture, same ISA — the only
difference is the USB/PCIe product ID in tinygrad's device enumeration table.

This eliminates the Linux host requirement entirely:

| Approach | Linux host | NVIDIA drivers | New exo code |
|----------|-----------|----------------|-------------|
| Linux + CUDA + vLLM | ✓ required | ✓ required | Heavy (new runner, transport) |
| Linux + tinygrad CUDA | ✓ required | ✓ required | Medium |
| **tinygrad direct (this RFC)** | **✗ not needed** | **✗ not needed** | **Minimal — PID + existing engine** |

`TinygradDynamicShardInferenceEngine` is already implemented in exo
(`exo/inference/tinygrad/inference.py`). The eGPU node plugs into this existing engine.

### 2.2 The memory domain problem — why TurboQuant is the bridge

When the M3 Ultra (MLX backend, BF16, unified memory) hands off a KV cache slice to the RTX PRO
6000 (tinygrad backend, GDDR7, across PCIe 4.0 x4), there is no shared address space. The tensor
must be serialised, transmitted across the PCIe 4.0 x4 link (~6 GB/s ceiling), and deserialised
into a completely different memory model.

For hidden state activations (16 KB per token for 70B), this is negligible. For KV cache
migration (hundreds of MB at long context), raw FP16 transfer is slow and wastes the limited
PCIe bandwidth. TurboQuant's 3-bit compression reduces a 70B/32K-context cross-backend KV
migration from ~640 MB to ~120 MB — a 5.3× reduction — making context handoff practical at
interactive latencies.

TurboQuant's role in this architecture is therefore dual:

1. **Bandwidth optimisation** — reduce transfer size for context migration
2. **Memory domain bridge** — serialize MLX tensors into a backend-neutral compressed wire
   format, deserialize into tinygrad tensor format on the other side

### 2.3 The gap in exo today

Current exo (January 2026 rewrite) supports homogeneous MLX topologies only. Phase 1 of this
RFC operates entirely within that constraint. Phase 2 activates `TinygradDynamicShardInferenceEngine`
— which exists in exo but is currently unused in the new architecture — and wires it to the eGPU
node via a new PCIe transport layer.

---

## 3. Architecture: Phase 1 (M3 Ultra + M1 Max, RDMA)

### 3.1 Layer assignment (asymmetric 3:1 split)

M3 Ultra holds 96 GB, M1 Max holds 32 GB → 3:1 memory ratio → proportional layer assignment:

```
Layers 0 → floor(N × 0.75)    →  M3 Ultra
Layers floor(N × 0.75) → N    →  M1 Max
```

For Llama 3 70B (80 layers): M3 Ultra layers 0–59 (~36 GB Q5_K_M), M1 Max layers 60–79
(~12 GB Q5_K_M).

KV cache at 32K context (GQA 8 heads): M3 Ultra ~864 MB, M1 Max ~288 MB — both trivially
within their respective pools.

### 3.2 Inference data flow (steady state)

```
Token → M3 Ultra (exo master)
    │
    ▼ [MLX backend]
    Embedding + layers 0–59
    KV update: local Q5_K_M
    Output: hidden state h, [batch, seq, 8192] FP16 = 16 KB/token
    │
    │  MLX Jaccl RDMA · ~1.6 µs at 10 GB/s
    ▼ [MLX backend]
    M1 Max: layers 60–79
    KV update: local Q5_K_M
    Argmax → token ID (4 bytes) returned via RDMA
    │
    ▼
Next token → loop
```

### 3.3 Context migration (TurboQuant)

Not on the hot path. Triggered only for failover or deliberate rebalancing:

```python
async def migrate_kv_cache(
    source: MLXShardedEngine,
    dest: MLXShardedEngine,
    request_id: str,
    layer_range: Tuple[int, int],
    compression: Literal["none", "turboquant_3bit"] = "turboquant_3bit"
) -> None:
    kv = await source.export_kv_cache(request_id, layer_range)
    kv_wire = turboquant_compress(kv, bits=3) if compression == "turboquant_3bit" else kv
    await dest.import_kv_cache(request_id, layer_range, kv_wire)
```

Migration cost at 32K context, M1 Max layer range (20 layers):

| Format | Size | RDMA time @ 10 GB/s |
|--------|------|---------------------|
| FP16 | ~640 MB | ~64 ms |
| Q5_K_M | ~220 MB | ~22 ms |
| TurboQuant 3-bit | ~120 MB | **~12 ms** |

---

## 4. Architecture: Phase 2 (+ RTX PRO 6000 Max-Q via tinygrad direct)

### 4.1 Prerequisite micro-PR: tinygrad PID addition

**Repository:** tinygrad  
**File:** `tinygrad/runtime/ops_nv.py` (or equivalent GB202 device table)  
**Change:** Add RTX PRO 6000 Blackwell Max-Q PID alongside the RTX 5090 entry

The RTX PRO 6000 Max-Q is a GB202 die at 300W TDP with 24,064 CUDA cores, identical
architecture to the RTX 5090. tinygrad's existing GB202 codepaths — shader compilation, memory
management, command ring submission — apply without modification. Only the USB/PCIe product
ID differs.

This micro-PR is a standalone contribution to tinygrad that benefits the broader community and
is a natural prerequisite to land before the exo Phase 2 PR.

### 4.2 Three-node layer assignment

With M3 Ultra (96 GB), M1 Max (32 GB), and RTX PRO 6000 Max-Q (96 GB), total pool is 224 GB.
Proportional split across all three nodes:

```
M3 Ultra (96/224 = 42.9%):    layers 0 → floor(N × 0.429)
M1 Max   (32/224 = 14.3%):    layers floor(N × 0.429) → floor(N × 0.572)
RTX PRO  (96/224 = 42.9%):    layers floor(N × 0.572) → N
```

For Llama 3 405B Q4_K_M (224 GB total, 126 layers):

| Node | Layer range | Layers | Weight size |
|------|-------------|--------|-------------|
| M3 Ultra | 0–53 | 54 | ~96 GB |
| M1 Max | 54–71 | 18 | ~32 GB |
| RTX PRO 6000 Max-Q | 72–125 | 54 | ~96 GB |

### 4.3 TurboQuant as the Metal↔tinygrad bridge

The critical data flow for Phase 2 is at the M3 Ultra → RTX PRO 6000 handoff boundary.

**Steady-state (hidden state only):**
```
M3 Ultra [MLX · BF16 unified memory]
    └── hidden state: [batch, seq, hidden_dim] BF16
         → cast to FP16
         → PCIe 4.0 x4 · ~6 GB/s
         → cast to tinygrad tensor dtype
    RTX PRO 6000 [tinygrad · GDDR7]
```

At 16 KB/token for 70B (hidden_dim=8192), PCIe transfer is ~2.7 µs per token — negligible.

**Context migration (KV cache):**
```
M3 Ultra [MLX KV cache · BF16 or Q5_K_M]
    └── turboquant_compress(kv, bits=3)      ← PolarQuant rotation + Lloyd-Max quantise
         → compressed wire format             ← backend-neutral flatbuffer slice
         → PCIe 4.0 x4 transfer
         → turboquant_decompress()
         → tinygrad.Tensor(dtype=dtypes.half)  ← tinygrad memory space
    RTX PRO 6000 [tinygrad KV cache]
```

The compressed wire format is **backend-neutral** — it is not an MLX tensor or a tinygrad
tensor, but a raw quantised byte buffer + FP32 scale factors. This is TurboQuant's function
as a bridge: both backends can produce and consume it without knowledge of each other's memory
model.

Context migration cost at 32K context, layers 72–125 (RTX PRO range, 54 layers):

| Format | Size | PCIe 4.0 x4 time @ 6 GB/s |
|--------|------|---------------------------|
| BF16 / FP16 | ~1.73 GB | ~289 ms |
| Q5_K_M | ~596 MB | ~99 ms |
| TurboQuant 3-bit | ~324 MB | **~54 ms** |

### 4.4 Inference data flow (three-node steady state)

```
Token → M3 Ultra (exo master)
    │
    ▼ [MLX backend · BF16 unified memory]
    Embedding + layers 0–53
    KV update: local (BF16 or Q5_K_M)
    Output: hidden state h₅₃, 16 KB (FP16)
    │
    │  MLX Jaccl RDMA · ~1.6 µs
    ▼ [MLX backend · BF16 unified memory]
    M1 Max: layers 54–71
    KV update: local Q5_K_M
    Output: hidden state h₇₁, 16 KB (FP16)
    │
    │  RDMA back to M3 Ultra · ~1.6 µs
    │  then PCIe 4.0 x4 to eGPU · ~2.7 µs
    ▼ [tinygrad direct · GDDR7]
    RTX PRO 6000 Max-Q: layers 72–125
    KV update: local (tinygrad tensor)
    Argmax → token ID → return to M3 Ultra
    │
    ▼
Next token → loop
```

Total inter-node transfer per token: ~7 µs. Well below any compute latency.

---

## 5. Quantisation

### 5.1 Q4_K_M (`block_q4_K` · 144 bytes / 256 elements · 4.5 bpw)

```c
typedef struct {
    ggml_half  d;           // FP16 super-block scale for scales
    ggml_half  dmin;        // FP16 super-block scale for mins
    uint8_t    scales[12];  // 8×6-bit scales + 8×6-bit mins, packed
    uint8_t    qs[128];     // 256 × 4-bit values, 2 per byte
} block_q4_K;
```

### 5.2 Q5_K_M (`block_q5_K` · 176 bytes / 256 elements · 5.5 bpw)

```c
typedef struct {
    ggml_half  d;           // FP16 super-block scale
    ggml_half  dmin;        // FP16 super-block scale for mins
    uint8_t    scales[12];  // 8×6-bit scales + 8×6-bit mins, packed
    uint8_t    qh[32];      // High bits: bit 4 of each 5-bit value
    uint8_t    qs[128];     // Lower 4 bits per element
} block_q5_K;
```

Both block layouts are **binary-identical across all ggml backends** — Metal, CUDA, tinygrad all
read the same struct from the same mmap'd GGUF file without conversion.

### 5.3 Format selection

| Model | Q4_K_M | Q5_K_M | Phase 1 (128 GB) | Phase 2 (224 GB) | Recommendation |
|-------|--------|--------|-------------------|-------------------|----------------|
| 8B | 4.7 GB | 5.7 GB | ✓ | ✓ | Q5_K_M |
| 70B | 39 GB | 48 GB | ✓ | ✓ | **Q5_K_M** (default) |
| 405B | 224 GB | 274 GB | ✗ | ✓ Q4 only | Q4_K_M |
| 671B MoE | ~190 GB | ~232 GB | ✗ | ✓ Q4 only | Q4_K_M |

Quality (LLaMA 2 7B, WikiText-2 perplexity, lower is better):

| Format | Perplexity | Δ vs FP16 |
|--------|-----------|-----------|
| FP16 | 5.91 | baseline |
| Q5_K_M | 5.93 | +0.023 |
| Q4_K_M | 5.97 | +0.054 |

---

## 6. exo Integration Points

### 6.1 Existing engine — `TinygradDynamicShardInferenceEngine`

**File:** `exo/inference/tinygrad/inference.py` (already exists)

No new runner required. Phase 2 wires the eGPU to this existing engine. The `ShardMetadata`
for the eGPU node specifies `backend="tinygrad"` and the assigned layer range.

### 6.2 ShardMetadata extension

**File:** `src/exo/shared/types/worker/shards.py`

```python
@dataclass
class ShardMetadata:
    start_layer: int
    end_layer: int
    node_id: str
    quant_format: str = "q5_k_m"    # "q4_k_m" | "q5_k_m"
    backend: str = "mlx"            # "mlx" | "tinygrad"
```

### 6.3 Asymmetric placement guard

**File:** `src/exo/master/placement_utils.py` → `allocate_layers_proportionally()`

Add `min_layers_per_node` guard. Below ~10 layers assigned, pipeline overhead exceeds the
node's compute contribution:

```python
def allocate_layers_proportionally(
    nodes: List[NodeInfo],
    total_layers: int,
    min_layers_per_node: int = 10
) -> Dict[str, Tuple[int, int]]:
    ...
    # Exclude nodes receiving < min_layers; fall back to next largest node
```

### 6.4 PCIe transport layer (Phase 2 new file)

**File:** `src/exo/networking/pcie_tb5.py`

Thin asyncio TCP wrapper over the TB5 PCIe 4.0 x4 link between M3 Ultra and eGPU:

```python
class PCIeTB5Transport:
    """
    TCP transport for M3 Ultra ↔ RTX PRO 6000 Max-Q hidden state and
    TurboQuant KV migration over TB5 PCIe 4.0 x4 (~6 GB/s ceiling).
    """
    async def send_activation(self, tensor: np.ndarray) -> None:
        # TCP_NODELAY for low-latency hidden state transfer (16 KB/token)
        ...

    async def send_kv_migration(self, compressed: bytes) -> None:
        # Large send buffers for TurboQuant bulk transfer
        ...
```

### 6.5 TurboQuant bridge module (Phase 2 new file)

**File:** `src/exo/inference/turboquant_bridge.py`

```python
from turboquant import compress_kv, decompress_kv

class TurboQuantBridge:
    """
    Backend-neutral KV cache bridge between MLX and tinygrad memory domains.
    Produces a raw compressed byte buffer + FP32 scale factors that neither
    backend needs to understand — only compress/decompress.
    """
    def compress(
        self,
        kv: Union[mlx.core.array, np.ndarray],
        bits: int = 3
    ) -> bytes:
        # PolarQuant rotation + Lloyd-Max scalar quantisation
        return compress_kv(np.array(kv), bits=bits)

    def decompress(
        self,
        wire: bytes,
        target_backend: Literal["mlx", "tinygrad"]
    ) -> Union[mlx.core.array, tinygrad.Tensor]:
        arr = decompress_kv(wire)    # → np.ndarray FP16
        if target_backend == "mlx":
            return mlx.core.array(arr)
        return tinygrad.Tensor(arr)
```

### 6.6 KV migration orchestration

**File:** `src/exo/master/kv_migration.py` (new, used by both Phase 1 and Phase 2)

```python
async def migrate_kv_cache(
    source_engine: InferenceEngine,
    dest_engine: InferenceEngine,
    request_id: str,
    layer_range: Tuple[int, int],
    transport: Union[MLXJacclTransport, PCIeTB5Transport],
    compression: Literal["none", "turboquant_3bit"] = "turboquant_3bit"
) -> None:
    bridge = TurboQuantBridge()
    kv = await source_engine.export_kv_cache(request_id, layer_range)
    wire = bridge.compress(kv, bits=3) if compression == "turboquant_3bit" else bytes(kv)
    await transport.send_kv_migration(wire)
    await dest_engine.import_kv_cache(request_id, layer_range, wire)
```

---

## 7. Performance Projections

### 7.1 Phase 1: two-node RDMA (70B Q5_K_M, batch 1)

| Stage | Node | Est. time |
|-------|------|-----------|
| Layers 0–59 | M3 Ultra | ~25 ms |
| RDMA hidden state | Link | ~0.002 ms |
| Layers 60–79 | M1 Max | ~10 ms |
| RDMA token return | Link | ~0.001 ms |
| **Total** | | **~35 ms → ~28 tok/s** |

M3 Ultra solo on 70B Q5_K_M: approximately 20–22 tok/s. Net gain from M1 Max: ~25–35%.

### 7.2 Phase 2: three-node (405B Q4_K_M, batch 1)

405B becomes locally runnable for the first time on a desk-deployable cluster.
Benchmark target rather than firm projection — depends on tinygrad direct backend
throughput on the RTX PRO 6000 Max-Q at 300W TDP.

| Stage | Node | Est. time |
|-------|------|-----------|
| Layers 0–53 | M3 Ultra | ~65 ms |
| RDMA to M1 Max | Link | ~0.002 ms |
| Layers 54–71 | M1 Max | ~22 ms |
| RDMA to M3 Ultra + PCIe to eGPU | Links | ~0.005 ms |
| Layers 72–125 | RTX PRO 6000 | ~45 ms |
| PCIe token return | Link | ~0.003 ms |
| **Total** | | **~132 ms → ~7.5 tok/s** |

Not fast — but 405B locally, no cloud, no data egress.

### 7.3 Pipeline bubble at 75/25 Phase 1 split

M1 Max idles ~60% of each decode step at batch size 1. Micro-batching 2–4 concurrent requests
raises both-node utilisation to >80%. RDMA context-switch overhead (5–9 µs) is negligible
versus compute time.

---

## 8. PR Structure

### Micro-PR 0 — tinygrad (prerequisite, standalone)
- **Repo:** tinygrad
- **File:** `tinygrad/runtime/ops_nv.py`
- **Change:** Add RTX PRO 6000 Blackwell Max-Q PID to GB202 device table
- **Scope:** One line. No logic changes.

### PR 1 — exo Phase 1 (this RFC, main deliverable)
- `placement_utils.py` — `min_layers_per_node` guard
- `shards.py` — `backend` and `quant_format` fields on `ShardMetadata`
- `kv_migration.py` — TurboQuant orchestration module (new)
- `turboquant_bridge.py` — backend-neutral compress/decompress (new)
- Benchmark: Llama 3 70B Q5_K_M, M3 Ultra + M1 Max via MLX Jaccl

### PR 2 — exo Phase 2 (follow-on, after Micro-PR 0 lands in tinygrad)
- `pcie_tb5.py` — PCIe TCP transport for eGPU link (new)
- `placement.py` — backend-aware cycle filtering for heterogeneous nodes
- Wire `TinygradDynamicShardInferenceEngine` to eGPU node
- Benchmark: Llama 3 405B Q4_K_M across all three nodes

---

## 9. Open Questions

1. **MLX Jaccl on M1 Max as client** — macOS 26.2 required. M3 Ultra as host is confirmed.
   Does MLX Jaccl function with M1 Max as the RDMA client? Generation gap (M1→M3) may affect
   Jaccl compatibility — needs lab validation before PR 1 benchmarks.

2. **tinygrad direct on eGPU PCIe** — The Razer Core X V2 presents as PCIe 4.0 x4 to the
   host, not USB. Does tinygrad's direct backend enumerate PCIe-attached GPUs, or is device
   discovery USB-specific? May require a small enumeration shim for PCIe presentation.

3. **GGUF layer-range loading** — Does the llama.cpp/MLX GGUF loader support loading only an
   assigned layer range, or does each node load the full model and discard unused layers?
   For 405B Q4_K_M (224 GB) the latter would require each node to read the entire file.

4. **min_layers threshold** — At what layer count does M1 Max contribution become net negative
   due to pipeline overhead? Estimate ~10 layers; needs benchmarking on the actual hardware.

5. **TurboQuant wire format versioning** — The compressed byte buffer format between backends
   should be versioned from day one. PolarQuant rotation matrix must be deterministic given the
   same seed so both ends can compress/decompress consistently without transmitting the matrix.

6. **RTX PRO 6000 Max-Q PSU** — The Razer Core X V2 requires a user-supplied ATX PSU.
   300W GPU TDP + 230W enclosure overhead = 530W minimum draw. A 650W+ ATX unit is required.
   Confirm the Core X V2 PSU bay accepts standard ATX form factor at this wattage.

---

## 10. References

- exo-explore/exo: https://github.com/exo-explore/exo
- tinygrad: https://github.com/tinygrad/tinygrad
- Apple TN3205 (RDMA over Thunderbolt): https://developer.apple.com/documentation/technotes/tn3205
- TurboQuant arXiv:2504.19874: https://arxiv.org/abs/2504.19874
- turboquant pip (HuggingFace drop-in): https://github.com/back2matching/turboquant
- turboquant Triton + vLLM: https://github.com/0xSero/turboquant
- turboquant_plus (llama.cpp Metal+CUDA): https://github.com/TheTom/turboquant_plus
- PARALLAX heterogeneous inference: https://gradient.network/parallax.pdf
- Razer Core X V2: https://www.razer.com/gaming-egpus/razer-core-x-v2
- RTX PRO 6000 Blackwell Max-Q: https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000-family/
- GGUF block format: https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.h

---

*Chronara Group Limited | AS200840 | Draft RFC — exo-explore/exo*
