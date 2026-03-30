# RFC: Compressed KV Streaming as Cross-Backend Wire Format
## Double-Buffer Deserialisation Bridge for Metal ↔ CUDA Split Inference

**Status:** Draft
**Author:** James Tervit, Chronara Group Limited (AS200840)
**Date:** 2026-03-30
**Target repos:** TheTom/llama-cpp-turboquant, TheTom/turboquant_plus
**Prerequisite:** tinygrad — RTX PRO 6000 Blackwell Max-Q PID addition (one-line micro-PR)
**Future integration:** exo-explore/exo (pending Spark integration by Alex Cheema)

---

## Hardware Topology

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                              M3 ULTRA NODE                                           │
│                                                                                      │
│  ┌─────────────────────────┐          TB5 PCIe 4.0 x4 (~6 GB/s)                    │
│  │  Metal backend (MLX)    │    ┌──────────────────────────────────────────────┐    │
│  │  96 GB unified memory   │    │  Razer Core X V2 eGPU enclosure              │    │
│  │  819 GB/s bandwidth     │◄──►│  └── RTX PRO 6000 Blackwell (300W thermal)   │    │
│  │                         │    │       96 GB GDDR7                             │    │
│  │  TurboQuant bridge      │    │       1,792 GB/s bandwidth                   │    │
│  │  (this RFC)             │    │       tinygrad direct backend (PCIIface)     │    │
│  └─────────────────────────┘    │       GB202 die · PCIe enumeration via IOKit │    │
│                                 └──────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────────────┘
     Combined pool: 192 GB · Two backends · One TB5 link
```

---

## 1. The Problem: Interconnect Is the Wall

eGPU setups for LLM inference hit a hard bandwidth ceiling. Thunderbolt 5 PCIe 4.0 x4 tops
out at ~6 GB/s — two orders of magnitude below GPU memory bandwidth (819 GB/s for M3 Ultra,
1,792 GB/s for RTX PRO 6000). For short-context, small-activation workloads this is
acceptable. For long-context inference with KV cache migration, it's the wall.

**The conventional approach:** minimise cross-device transfers. Keep KV cache local, only send
hidden state activations (16 KB/token for 70B) across the link.

**This RFC's approach:** make cross-device transfers cheap enough to stream continuously.
Use TurboQuant compressed KV cache as the wire format between Metal and CUDA backends,
achieving up to 4.6× compression versus FP16. Pair with a double-buffer pipeline to overlap
DMA with compute, hiding interconnect latency entirely for sequential decode.

### 1.1 Why This Is Novel

Heterogeneous multi-device inference exists (PARALLAX, PetS, FlexGen, tinygrad). None use
**compression as the bandwidth strategy**. They treat the interconnect as a fixed cost and
minimise crossings. This RFC treats the interconnect as a continuous stream and minimises
the **per-crossing cost**.

The TurboQuant compressed format is **byte-identical on both backends**. The WHT rotation
seed is deterministic (seeded via `np.random.default_rng(seed)` + QR decomposition), the
quantisation block layout is shared, the Lloyd-Max codebook is derived deterministically
from the rotation. Cross-device deserialisation is architecturally clean — no format
conversion, no negotiation, no endianness concerns.

### 1.2 Format Selection: turbo3 vs turbo4

The bridge architecture is format-agnostic. Both turbo3 and turbo4 use the same streaming
mechanism. Selection depends on the quality/bandwidth tradeoff:

| Format | Bits/val | Compression | PPL (wiki-2) | vs q8_0 | Decode speed |
|--------|----------|-------------|-------------|---------|-------------|
| q8_0 | 8.5 | 1.9× | 6.111 | baseline | baseline |
| **turbo4** | **4.25** | **3.8×** | **6.125** | **+0.23%** | **0.93×** |
| q4_0 | 4.5 | 3.6× | 6.142 | +0.52% | — |
| **turbo3** | **3.5** | **4.6×** | **6.176** | **+1.06%** | **0.90×** |
| turbo2 | 2.5 | 6.4× | 6.507 | +6.48% | — |

*Benchmarks: M5 Max 128GB, Metal, wikitext-2 512 context. Source: TheTom/turboquant_plus*

**Recommendation:** turbo4 as default (best quality/compression ratio), turbo3 for maximum
bandwidth savings when interconnect is the bottleneck. turbo4 also avoids the L2 cache
pressure issue seen with turbo3 on pre-M5 Apple Silicon.

### 1.3 Compression Advantage Quantified

KV cache per layer for 70B (GQA 8 heads, hidden_dim 8192, 32K context):

| Format | Size/layer | 80 layers | Compression vs FP16 |
|--------|-----------|-----------|-------------------|
| FP16 | ~32 MB | ~2.56 GB | 1.0× (baseline) |
| Q8_0 | ~16 MB | ~1.28 GB | 2.0× |
| **turbo4** | **~8.4 MB** | **~674 MB** | **3.8×** |
| **turbo3** | **~6.9 MB** | **~555 MB** | **4.6×** |

Transfer time over TB5 PCIe 4.0 x4 (~6 GB/s) for full 80-layer KV migration at 32K context:

| Format | Size | Transfer time |
|--------|------|--------------|
| FP16 | 2.56 GB | ~427 ms |
| Q8_0 | 1.28 GB | ~213 ms |
| turbo4 | 674 MB | ~112 ms |
| **turbo3** | **555 MB** | **~93 ms** |

With double-buffered layer-granular streaming, the effective latency is hidden behind compute.

---

## 2. Architecture: Double-Buffer KV Streaming

### 2.1 Core Insight

Both the Metal backend (via `kernel_set_rows_turbo3/4` in ggml-metal.metal) and the CUDA
backend (via quantize.cu / set-rows.cu) can independently compress and decompress KV cache.
The compressed block format is binary-identical across both backends. The WHT rotation
matrix is deterministic given the model-level seed — no per-session negotiation required.

This means:
1. Metal compresses a KV layer → raw bytes
2. Raw bytes traverse TB5 via DMA → no interpretation needed in transit
3. CUDA decompresses raw bytes → native KV tensor

The bridge is the **orchestration**, not the format.

### 2.2 Double-Buffer Pipeline

```
                    TB5 PCIe 4.0 x4 (~6 GB/s)
                    ════════════════════════════

  Metal (M3 Ultra)                              CUDA (RTX PRO 6000)
  ┌──────────────┐                              ┌──────────────┐
  │ Compress      │    ┌─────────┐              │ Decompress    │
  │ KV layer N    │───►│ Buf A   │──── DMA ────►│ KV layer N    │──► Compute
  │               │    └─────────┘              │               │
  │ Compress      │    ┌─────────┐              │ Decompress    │
  │ KV layer N+1  │───►│ Buf B   │──── DMA ────►│ KV layer N+1  │──► (queued)
  │               │    └─────────┘              │               │
  │ Compress      │    ┌─────────┐              │ Decompress    │
  │ KV layer N+2  │───►│ Buf A   │──── DMA ────►│ KV layer N+2  │──► (queued)
  └──────────────┘    (ping-pong)               └──────────────┘
```

**Pipeline stages per layer:**

| Stage | Location | Time (70B, 32K ctx, turbo3) |
|-------|----------|-------------------|
| 1. Compress KV layer | Metal GPU (kernel_set_rows) | ~1.2 ms |
| 2. DMA compressed buffer | TB5 link | ~1.15 ms |
| 3. Decompress + load to VRAM | CUDA GPU | ~0.8 ms |
| 4. Compute (layer forward pass) | CUDA GPU | ~0.4 ms |

With double-buffering, stages 1–2 for layer N+1 overlap with stages 3–4 for layer N.
**Effective per-layer latency: max(compress+DMA, decompress+compute) ≈ 2.35 ms**.

Without compression: DMA alone is ~5.3 ms per layer (32 MB FP16 @ 6 GB/s).
**Compression makes the pipeline compute-bound instead of DMA-bound.**

### 2.3 Prefetch Depth

To fully hide DMA latency, the producer (Metal) must stay ahead of the consumer (CUDA) by
enough layers that a buffer is always ready when compute completes.

```
Prefetch depth = ceil(DMA_time / compute_time)
               = ceil(1.15 ms / 0.4 ms)
               = 3 layers
```

A ring buffer of 3–4 slots (not just 2) provides margin for jitter. Each slot holds one
compressed KV layer (~6.9 MB for 70B turbo3). Total ring buffer memory: ~28 MB — trivial.

### 2.4 Steady-State Decode vs KV Migration

**Steady-state decode (per token):**

Only hidden state activations cross the link. At 16 KB/token (70B, hidden_dim=8192), TB5
transfer is ~2.7 µs. No compression needed — activation transfer is negligible.

**KV migration (context handoff / layer rebalancing):**

This is where compressed streaming matters. Triggered when:
- Layer assignment changes (rebalancing between Metal and CUDA)
- Prefill on Metal, decode on CUDA (prompt processing split)
- Context handoff between backends

The double-buffer pipeline streams the KV cache layer-by-layer at the compressed rate,
overlapping DMA with decompression. Total migration time for 40 layers (half of 80-layer
70B model, representative CUDA assignment) at 32K context:

| Approach | Total time | Bottleneck |
|----------|-----------|-----------|
| FP16, blocking | ~213 ms | DMA |
| turbo3, blocking | ~46 ms | DMA |
| **turbo3, double-buffered** | **~40 ms** | **Compute (decompress)** |

---

## 3. Wire Format Specification

### 3.1 Compressed KV Block Layout (turbo3)

The on-GPU storage format uses 128-byte blocks (aligned to GPU cache lines). For the wire
format, we **strip padding** to avoid wasting 87.5% of DMA bandwidth on zeros:

```
On-GPU block (128 bytes per 32 elements):
  qs[8]      — 2-bit low indices (2 bits × 32 = 8 bytes)
  signs[4]   — 1-bit high bits (1 bit × 32 = 4 bytes)
  norm[4]    — float32 L2 norm
  pad[112]   — padding to 128-byte cache line alignment

Wire block (16 bytes per 32 elements — padding stripped):
  qs[8]      — 2-bit low indices
  signs[4]   — 1-bit high bits
  norm[4]    — float32 L2 norm
```

Stripping padding reduces wire transfer by 7×:
- On-GPU: 128 bytes / 32 elements → 4 bytes/element
- Wire: 16 bytes / 32 elements → 0.5 bytes/element

For 70B at 32K context, wire format for 80 layers:
- With padding: ~555 MB → ~93 ms @ 6 GB/s
- **Stripped: ~69 MB → ~12 ms @ 6 GB/s**

### 3.2 Compressed KV Block Layout (turbo4)

```
On-GPU/Wire block (68 bytes per 128 elements — no padding needed):
  qs[64]     — 4-bit nibble packed (128 × 4 bits = 64 bytes)
  norm[4]    — float32 L2 norm

Wire: 68 bytes / 128 elements → 0.53 bytes/element
```

turbo4 blocks are already compact — no stripping needed.

### 3.3 Per-Layer Wire Packet

```
┌──────────────────────────────────────────────┐
│ Layer Header (16 bytes)                       │
│   layer_idx:    uint32                        │
│   seq_len:      uint32                        │
│   n_heads:      uint32                        │
│   flags:        uint32  (K=0x1, V=0x2, KV=0x3)│
├──────────────────────────────────────────────┤
│ Compressed K cache (stripped turbo blocks)     │
│   n_blocks = n_heads × seq_len × head_dim / 32│
│   Each block: 16 bytes (turbo3) or            │
│               68 bytes per 128 elem (turbo4)  │
├──────────────────────────────────────────────┤
│ Compressed V cache (same layout as K)         │
│   V uses TurboQuantMSE (no QJL stage)         │
│   Block format identical, different codebook  │
└──────────────────────────────────────────────┘
```

Both `-ctk turbo3` and `-ctv turbo3` (or turbo4) are applied — K and V caches use matching
compression. Note: K cache uses full TurboQuant (PolarQuant + QJL for inner product
preservation), V cache uses TurboQuantMSE (PolarQuant only, MSE-optimal).

### 3.4 Endianness and Alignment

All block structs are little-endian, naturally aligned. Both Apple Silicon (ARM64) and
NVIDIA (sm_100, Blackwell) are little-endian. No byte-swapping required. The stripped
buffer can be DMA'd as raw bytes without interpretation.

### 3.5 Codebook Consistency

Both backends must use the same Lloyd-Max codebook for the post-WHT distribution.

**Known consideration:** Metal uses theoretical Lloyd-Max centroids for post-FWHT
distribution (d=128, σ=0.0884, outer centroid=0.1739). CUDA community implementations
have used empirical centroids from real KV data (outer centroid=0.2416). The difference
is ~0.2-0.3% quality impact. For the bridge to work correctly, both sides must use the
**same codebook**. This is guaranteed when both use the centroids from
`turboquant_plus/turboquant/codebook.py` (Lloyd's algorithm with seeded init).

**Validation:** Sprint 0 test S0.6 verifies C codebook matches Python reference at d=128.

---

## 4. Implementation

### 4.1 Real API Surface (from turboquant_plus)

The Python reference implementation provides:

```python
# turboquant_plus/turboquant/kv_cache.py
class KVCacheCompressor:
    def __init__(self, head_dim: int, k_bits: int = 3, v_bits: int = 3, seed: int = 42)

    def compress(self, k_cache: np.ndarray, v_cache: np.ndarray) -> CompressedKVCache:
        """Input shapes: (num_layers, num_heads, seq_len, head_dim)"""

    def decompress(self, compressed: CompressedKVCache) -> tuple[np.ndarray, np.ndarray]:
        """Returns: (k_cache, v_cache) same shapes as input"""

# turboquant_plus/turboquant/turboquant.py
@dataclass
class CompressedVector:
    mse_indices: np.ndarray      # PolarQuant indices, (b-1)-bit integers
    vector_norms: np.ndarray     # original ||x||_2 for rescaling
    qjl_signs: np.ndarray        # QJL sign bits, int8 {+1, -1}
    residual_norms: np.ndarray   # ||residual||_2
    bit_width: int
```

The llama.cpp integration uses Metal/CUDA kernels directly:
- **Metal:** `kernel_set_rows_turbo3` / `kernel_set_rows_turbo4` in `ggml-metal.metal`
- **CUDA:** `set-rows.cu`, `quantize.cu` in `ggml/src/ggml-cuda/`
- **Flags:** `-ctk turbo3 -ctv turbo3` parsed in `common/arg.cpp` → `params.cache_type_k/v`

### 4.2 Ring Buffer

**File:** `src/turboquant_bridge/kv_ring_buffer.py`

```python
import asyncio
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class KVSlot:
    """One slot in the ring buffer, holds one compressed KV layer."""
    layer_idx: int = -1
    data: Optional[bytes] = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)

class KVRingBuffer:
    """
    Ring buffer for streaming compressed KV layers from producer
    (Metal compress) to consumer (CUDA decompress).

    Slot count >= prefetch_depth + 1 to avoid stalls.
    """
    def __init__(self, n_slots: int = 4):
        self.slots = [KVSlot() for _ in range(n_slots)]
        self.n_slots = n_slots
        self.write_idx = 0
        self.read_idx = 0
        self._write_sem = asyncio.Semaphore(n_slots)
        self._read_sem = asyncio.Semaphore(0)

    async def acquire_write(self) -> KVSlot:
        await self._write_sem.acquire()
        slot = self.slots[self.write_idx % self.n_slots]
        slot.ready.clear()
        return slot

    def release_write(self, slot: KVSlot) -> None:
        slot.ready.set()
        self.write_idx += 1
        self._read_sem.release()

    async def acquire_read(self) -> KVSlot:
        await self._read_sem.acquire()
        slot = self.slots[self.read_idx % self.n_slots]
        await slot.ready.wait()
        return slot

    def release_read(self) -> None:
        self.read_idx += 1
        self._write_sem.release()
```

### 4.3 KV Stream Coordinator

**File:** `src/turboquant_bridge/kv_stream.py`

```python
import asyncio
import struct
from typing import Tuple
from turboquant.kv_cache import KVCacheCompressor
import numpy as np

HEADER_FMT = "<IIII"  # layer_idx, seq_len, n_heads, flags
HEADER_SIZE = struct.calcsize(HEADER_FMT)
FLAG_KV = 0x3

class KVStreamCoordinator:
    """
    Coordinates layer-granular KV cache streaming between Metal and CUDA
    backends using TurboQuant compressed wire format over a double-buffered
    ring.

    Producer side (Metal / M3 Ultra):
        - Exports KV for one layer from Metal KV cache
        - Compresses via KVCacheCompressor (turbo3/turbo4)
        - Strips block padding for wire efficiency
        - Writes stripped bytes + header into ring buffer slot

    Consumer side (CUDA / RTX PRO 6000 via tinygrad):
        - Reads stripped bytes from ring buffer slot
        - Restores block padding
        - Decompresses via KVCacheCompressor
        - Loads into tinygrad.Tensor KV cache
    """

    def __init__(self, ring: 'KVRingBuffer', head_dim: int = 128,
                 k_bits: int = 3, v_bits: int = 3, seed: int = 42):
        self.ring = ring
        self.compressor = KVCacheCompressor(
            head_dim=head_dim, k_bits=k_bits, v_bits=v_bits, seed=seed
        )

    async def produce(
        self,
        kv_cache: np.ndarray,  # [2, num_layers, num_heads, seq_len, head_dim]
        layer_range: Tuple[int, int],
    ) -> None:
        """Producer coroutine — runs on Metal side."""
        for layer_idx in range(layer_range[0], layer_range[1]):
            slot = await self.ring.acquire_write()

            # Extract single layer: [num_heads, seq_len, head_dim]
            k_layer = kv_cache[0, layer_idx:layer_idx+1]  # keep dim for API
            v_layer = kv_cache[1, layer_idx:layer_idx+1]

            # Compress via KVCacheCompressor
            compressed = self.compressor.compress(k_layer, v_layer)

            # Serialize to wire format (stripped padding)
            wire = self._to_wire(compressed, layer_idx)

            slot.layer_idx = layer_idx
            slot.data = wire
            self.ring.release_write(slot)

        # Send sentinel
        slot = await self.ring.acquire_write()
        slot.data = None
        self.ring.release_write(slot)

    async def consume(
        self,
        dest_kv_cache: np.ndarray,  # target KV cache to populate
    ) -> None:
        """Consumer coroutine — runs on CUDA side."""
        while True:
            slot = await self.ring.acquire_read()
            if slot.data is None:
                self.ring.release_read()
                break

            # Deserialize from wire format
            layer_idx, k_data, v_data = self._from_wire(slot.data)

            # Decompress and write into dest KV cache
            compressed = self._reconstruct_compressed(k_data, v_data)
            k_layer, v_layer = self.compressor.decompress(compressed)
            dest_kv_cache[0, layer_idx] = k_layer[0]
            dest_kv_cache[1, layer_idx] = v_layer[0]

            self.ring.release_read()

    def _to_wire(self, compressed, layer_idx: int) -> bytes:
        """Serialize CompressedKVCache to stripped wire format."""
        header = struct.pack(HEADER_FMT,
                             layer_idx, compressed.seq_len,
                             compressed.num_heads, FLAG_KV)
        # Serialize compressed arrays — strip GPU padding
        k_bytes = self._serialize_compressed_vectors(compressed.k_compressed[0])
        v_bytes = self._serialize_v_data(compressed.v_indices[0], compressed.v_norms[0])
        return header + struct.pack("<II", len(k_bytes), len(v_bytes)) + k_bytes + v_bytes

    def _from_wire(self, data: bytes):
        """Deserialize stripped wire format."""
        layer_idx, seq_len, n_heads, flags = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
        k_len, v_len = struct.unpack("<II", data[HEADER_SIZE:HEADER_SIZE+8])
        offset = HEADER_SIZE + 8
        k_data = data[offset:offset+k_len]
        v_data = data[offset+k_len:offset+k_len+v_len]
        return layer_idx, k_data, v_data

    def _serialize_compressed_vectors(self, vectors) -> bytes:
        """Serialize list of CompressedVector to compact bytes."""
        parts = []
        for cv in vectors:
            parts.append(cv.mse_indices.tobytes())
            parts.append(cv.vector_norms.tobytes())
            parts.append(cv.qjl_signs.tobytes())
            parts.append(cv.residual_norms.tobytes())
        return b''.join(parts)

    def _serialize_v_data(self, indices_list, norms_list) -> bytes:
        """Serialize V cache data (TurboQuantMSE — no QJL)."""
        parts = []
        for indices, norms in zip(indices_list, norms_list):
            parts.append(indices.tobytes())
            parts.append(norms.tobytes())
        return b''.join(parts)

    def _reconstruct_compressed(self, k_data, v_data):
        """Reconstruct CompressedKVCache from wire bytes. Implementation
        depends on head configuration — deserializes into the dataclass
        structure expected by KVCacheCompressor.decompress()."""
        # Implementation deferred to integration phase — requires
        # knowledge of n_heads, seq_len, head_dim to reshape arrays
        raise NotImplementedError("Wire deserialization — see §9.1")


async def stream_kv_migration(
    source_kv: np.ndarray,
    dest_kv: np.ndarray,
    layer_range: Tuple[int, int],
    head_dim: int = 128,
    k_bits: int = 3,
    v_bits: int = 3,
    seed: int = 42,
    n_ring_slots: int = 4,
) -> None:
    """
    Top-level entry point: stream KV cache from Metal to CUDA using
    double-buffered TurboQuant compressed wire format.

    Producer and consumer run concurrently — DMA overlaps with compute.
    """
    from .kv_ring_buffer import KVRingBuffer
    ring = KVRingBuffer(n_slots=n_ring_slots)
    coordinator = KVStreamCoordinator(
        ring, head_dim=head_dim, k_bits=k_bits, v_bits=v_bits, seed=seed
    )

    await asyncio.gather(
        coordinator.produce(source_kv, layer_range),
        coordinator.consume(dest_kv),
    )
```

### 4.4 tinygrad Integration Point

KV cache in tinygrad is stored as a Tensor in `TransformerBlock`
(`tinygrad/apps/llm.py:145-177`):

```python
# Existing tinygrad KV cache structure:
self.cache_kv = Tensor.empty(
    2, x.shape[0], self.config.n_kv_heads,
    self.config.max_context, self.config.head_dim,
    device=x.device
)
```

**No `export_kv_cache` / `import_kv_cache` methods exist.** These must be added.
The bridge interfaces with tinygrad by:

1. Reading `self.cache_kv` as numpy: `cache_kv.numpy()` → source for producer
2. Writing decompressed data back: `Tensor(decompressed_array)` → consumer import

For Metal (MLX), the equivalent extraction from `mlx.core.array` is similarly direct.

### 4.5 tinygrad PID Prerequisite

**File:** `tinygrad/runtime/ops_nv.py:544`

tinygrad's `PCIIface` class uses IOKit on macOS to scan the PCI bus. The device table:

```python
devices=((0xff00, (0x2200,0x2400,0x2500,0x2600,0x2700,0x2800,0x2b00,0x2c00,0x2d00,0x2f00)),)
```

Adding the RTX PRO 6000 Blackwell Max-Q PID is a one-line addition to this tuple.
The chip name mapping at `tinygrad/runtime/support/nv/nvdev.py:114` maps `"GB2"` →
`"GB202"`, which already covers the GB202 die shared with RTX 5090.

**PCIe enumeration confirmed:** `PCIIface` uses IOKit's `IORegistryEntryCreateCFProperty`
for vendor-id/device-id matching (`tinygrad/runtime/support/system.py:57-74`). A GPU in
a Razer Core X V2 eGPU enclosure presents as a standard PCIe device visible to IOKit.
This is **not** USB-dependent.

---

## 5. Performance Model

### 5.1 Pipeline Throughput (70B, 32K context)

| Metric | turbo3 | turbo4 |
|--------|--------|--------|
| Compressed KV per layer | ~6.9 MB | ~8.4 MB |
| Wire bytes per layer (stripped) | ~0.86 MB | ~8.4 MB (no strip) |
| DMA time per layer (6 GB/s) | ~0.14 ms (stripped) | ~1.4 ms |
| Compress time per layer (Metal) | ~1.2 ms | ~0.9 ms |
| Decompress time per layer (CUDA) | ~0.8 ms | ~0.6 ms |
| Compute time per layer (RTX PRO 6000) | ~0.4 ms | ~0.4 ms |
| **Double-buffered pipeline rate** | **~1.6 ms/layer** | **~1.4 ms/layer** |
| **Blocking FP16 transfer rate** | **~5.3 ms/layer** | **~5.3 ms/layer** |
| **Speedup vs FP16 blocking** | **3.3×** | **3.8×** |

### 5.2 Full Migration Latency (40 layers, 32K context)

| Approach | turbo3 | turbo4 |
|----------|--------|--------|
| FP16, blocking | ~213 ms | ~213 ms |
| Compressed, blocking | ~46 ms | ~56 ms |
| **Compressed, double-buffered** | **~40 ms** | **~36 ms** |

### 5.3 Steady-State Decode Impact

During normal autoregressive decode, only the **current token's KV update** crosses the link
if the layer is assigned to the remote device. Per token:

| Data | Size | Transfer time |
|------|------|--------------|
| Hidden state activation | 16 KB | ~2.7 µs |
| KV update (1 token, compressed) | ~192 bytes | negligible |

The double-buffer is not needed for steady-state decode. It activates for bulk KV migration.

### 5.4 Quality Impact (from TheTom/turboquant_plus benchmarks)

**Long-context stability (32K, 50-chunk wikitext-103):**
- q8_0: 7.0638 PPL (baseline)
- turbo3: 7.1796 PPL (+1.64%) — no degradation trend with context length

**Needle-In-A-Haystack retrieval (multi-key, RULER MK-NIAH):**
- q8_0: 1/1 at all depths (4K/8K/16K/32K)
- turbo3: 1/1 at all depths (4K/8K/16K/32K) — 100% retrieval accuracy

**Real-world PDF benchmark (24K context, Qwen3.5-35B):**
- q8_0: 68.2 tok/s decode
- turbo4: 63.7 tok/s (0.93×)
- turbo3: 53.3 tok/s (0.78×)

---

## 6. Failure Modes and Recovery

### 6.1 TB5 Hot-Unplug During Stream

If the eGPU disconnects mid-stream:
- Producer detects DMA failure → sets error flag on ring buffer
- Consumer reads error sentinel → aborts import
- Host falls back to Metal-only inference on M3 Ultra
- KV cache on Metal side is still intact (producer exports are non-destructive)

### 6.2 Backpressure (Consumer Slower Than Producer)

Ring buffer semaphores handle this naturally:
- If all write slots are full, producer blocks on `acquire_write`
- No data loss, no corruption — just a slower pipeline
- Degraded case converges to blocking transfer (still correct)

### 6.3 Compression/Decompression Mismatch

Cannot happen if both backends use the same TurboQuant implementation:
- Same block struct layout
- Same WHT rotation seed (per-model, from GGUF metadata or fixed seed=42)
- Same Lloyd-Max codebook (deterministic from rotation + Lloyd's algorithm)

If implementations diverge (e.g., different codebook versions), add a version byte to the
wire header and reject mismatches at connection time.

---

## 7. Test Strategy

### 7.1 Sprint 0 — Hardware Validation (Go/No-Go Gates)

| # | Test | Pass criteria | Blocks |
|---|------|--------------|--------|
| S0.1 | tinygrad enumerates RTX PRO 6000 via Razer Core X V2 | `tinygrad.Device["NV"]` returns device info via PCIIface/IOKit | All CUDA work |
| S0.2 | TB5 sustained throughput benchmark | ≥5 GB/s sustained for 60s (bulk transfer) | Performance model |
| S0.3 | turbo3/turbo4 compress/decompress on Metal (M3 Ultra) | Correct round-trip, measure latency per layer | Compress timing |
| S0.4 | turbo3/turbo4 compress/decompress on CUDA (RTX PRO 6000) | Correct round-trip, measure latency per layer | Decompress timing |
| S0.5 | Codebook consistency | Lloyd-Max centroids from Python `codebook.py` at d=128 match C implementation centroids in Metal and CUDA kernels | Cross-backend correctness |

### 7.2 Layer 1 — Double-Buffer Correctness

| # | Test | Method |
|---|------|--------|
| L1.1 | Cross-backend round-trip fidelity | Compress on Metal → stripped wire → decompress on CUDA → compare vs FP16 reference. Max absolute error < threshold per layer |
| L1.2 | Ring buffer ordering guarantee | Synthetic producer/consumer with random delays (0–5ms). Verify consumer reads layers in strict order |
| L1.3 | WHT rotation determinism | Same model seed → same rotation matrix on Metal and CUDA. `memcmp` compressed output of identical input on both backends |
| L1.4 | Backpressure correctness | Producer 3× faster than consumer. Verify no data loss, no corruption, producer blocks gracefully |
| L1.5 | Error propagation on DMA failure | Simulate TB5 disconnect (close FD). Verify clean error in consumer, no partial layer import |
| L1.6 | Padding strip/restore round-trip | Strip padding from on-GPU turbo3 blocks, transmit, restore padding, verify decompression matches original |

### 7.3 Layer 2 — Streaming Integration

| # | Test | Method |
|---|------|--------|
| L2.1 | Full KV migration, 70B/32K, 40 layers | Stream Metal→CUDA, run inference on CUDA, compare output vs single-device baseline (perplexity delta < 0.05) |
| L2.2 | Migration under concurrent inference | One request actively decoding while another's KV migrates. Verify no interference |
| L2.3 | Throttled link (2 GB/s artificial cap) | Verify graceful degradation: slower but correct |
| L2.4 | TB5 hot-unplug during migration | Disconnect eGPU mid-stream. Verify: clean error, Metal KV intact, fallback to Metal-only |

### 7.4 Layer 3 — End-to-End Benchmarks

| # | Test | Target |
|---|------|--------|
| L3.1 | 70B Q5_K_M, M3 Ultra + RTX PRO 6000, turbo3 | Measure tok/s decode, compare vs M3 Ultra solo |
| L3.2 | KV migration latency, 32K context, 40 layers | ≤45 ms (turbo3 double-buffered, stripped wire) |
| L3.3 | KV migration latency, 128K context, 40 layers | Measure and report |
| L3.4 | Perplexity at 128K context with turbo3 KV | Measure degradation vs FP16 baseline |
| L3.5 | turbo3 vs turbo4 comparison | Compare quality, latency, and throughput on same workload |

---

## 8. PR Structure

### PR 0 — tinygrad micro-PR (prerequisite, standalone)
- **Repo:** tinygrad
- **File:** `tinygrad/runtime/ops_nv.py`
- **Change:** Add RTX PRO 6000 Blackwell Max-Q PID to GB202 device tuple
- One line, no logic changes

### PR 1 — TurboQuant Cross-Backend Bridge (this RFC)
- **Repo:** turboquant-tinygrad-bridge (new)
- `kv_ring_buffer.py` — Double-buffer ring for compressed KV streaming
- `kv_stream.py` — Layer-granular stream coordinator using KVCacheCompressor API
- Wire format spec: stripped turbo3/turbo4 blocks + layer header
- Tests: ring buffer unit tests, cross-backend round-trip integration tests
- Benchmarks: migration latency, throughput, quality metrics

### PR 2 — exo Integration (future, post-Spark)
- Wire `stream_kv_migration` into exo's topology manager
- Backend-aware layer placement (Metal vs CUDA)
- End-to-end benchmarks with exo cluster

---

## 9. Open Questions

1. **Wire deserialization implementation** — The `_reconstruct_compressed` method in §4.3
   needs to reshape raw bytes back into `CompressedKVCache` dataclass structure. This
   requires knowing the exact byte layout per head, which varies with seq_len and head_dim.
   Needs a serialization spec that encodes array shapes alongside data.

2. **turbo3 compress latency on Metal** — The ~1.2 ms/layer estimate is extrapolated from
   CPU benchmarks. Metal compute shader performance (via `kernel_set_rows_turbo3`) may
   differ significantly. Needs Sprint 0 measurement on actual M3 Ultra hardware.

3. **Ring buffer memory pinning** — For optimal DMA throughput, ring buffer slots should
   use pinned (page-locked) memory. MLX handles this automatically for unified memory;
   tinygrad's allocator may need explicit pinning for GDDR7 staging buffers.

4. **Prefill split** — Can prompt processing (prefill) be split across backends using the
   same streaming mechanism? Prefill is compute-bound with different DMA/compute overlap
   ratios. turbo4 prefill matches q8_0 speed (1.01×); turbo3 is comparable.

5. **turbo3 quality at 128K+ context** — Benchmarks show stability through 32K. At 128K,
   cumulative quantisation error across 80 layers needs measurement. turbo4 may be safer
   for very long context.

6. **Sparse V dequant integration** — turboquant_plus implements sparse V dequant (skip
   positions where softmax < 1e-6) for +22.8% decode speedup. Can this be applied at the
   bridge level, or only within a single backend?

---

## 10. References

- TheTom/llama-cpp-turboquant: https://github.com/TheTom/llama-cpp-turboquant (Metal + CUDA fork)
- TheTom/turboquant_plus: https://github.com/TheTom/turboquant_plus (benchmarks, papers, Python reference)
- TurboQuant arXiv:2504.19874: https://arxiv.org/abs/2504.19874 (ICLR 2026)
- tinygrad: https://github.com/tinygrad/tinygrad (PCIIface NV backend)
- ggml-common.h block formats: https://github.com/ggml-org/ggml/blob/master/src/ggml-quants.h
- turbo4-resurrection.md: turboquant_plus/docs/papers/turbo4-resurrection.md (bug analysis, final format)
- turboquant-recommendations.md: turboquant_plus/docs/turboquant-recommendations.md (model-specific guidance)
- PARALLAX heterogeneous inference: https://gradient.network/parallax.pdf
- exo-explore/exo: https://github.com/exo-explore/exo (future integration, pending Spark)

---

*Chronara Group Limited | AS200840 | Draft RFC — Compressed KV Streaming Bridge*
