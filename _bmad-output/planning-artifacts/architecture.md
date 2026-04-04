---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
inputDocuments:
  - docs/RFC-double-buffer-kv-bridge.md
  - docs/RFC-metal-cuda-kv-bridge.md
  - turboquant_plus/PLAN.md
  - README.md
workflowType: 'architecture'
lastStep: 8
status: 'complete'
completedAt: '2026-04-02'
project_name: 'turboquant-tinygrad-bridge'
user_name: 'Jamesm3'
date: '2026-04-01'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Context Analysis

### Requirements Overview

**Functional Requirements:**
- FR1: Double-buffered ring buffer for streaming compressed KV layers Metal → CUDA
- FR2: TurboQuant compress on Metal (kernel_set_rows_turbo3/4) → stripped wire format
- FR3: DMA stripped blocks over TB5 PCIe 4.0 x4 as raw bytes
- FR4: Decompress on CUDA/NV backend → tinygrad.Tensor KV cache
- FR5: Wire format: 28-byte layer header (with separate k_format/v_format fields for asymmetric compression) + stripped turbo3 blocks (16 bytes/32 elements) or turbo4 blocks (68 bytes/128 elements) or q8_0 blocks. Header must include head_dim and block_count for consumer-side reshaping. Head_dim validation guard must reject unsupported values (>128) until corresponding FA kernel instantiations are confirmed.
- FR6: Codebook consistency validation — Lloyd-Max centroids bit-exact across Metal and CUDA
- FR7: Graceful degradation on TB5 disconnect — error sentinel, fallback to Metal-only
- FR8: Backpressure handling — semaphore-based, no data loss under slow consumer
- FR9: tinygrad PID micro-PR — RTX PRO 6000 Blackwell Max-Q added to GB202 device table

**Non-Functional Requirements:**
- NFR1: TB5 sustained throughput ≥5 GB/s (validates performance model)
- NFR2: KV migration ≤45 ms for 40 layers @ 32K context (turbo3 double-buffered)
- NFR3: PPL degradation <1.1% vs q8_0 baseline through 32K context. **Validated beyond 32K (Tom review 2026-04-04):** KL divergence does NOT drift with context length — mean KLD actually decreases slightly. Sjoerd Maessen running turbo3 in production at 2×104K context on 122B model across dual L40S with no quality degradation. turbo3 is validated well beyond 32K; turbo4 remains slightly safer but turbo3 is production-proven at 128K+.
- NFR4: Thermal stability at 300W sustained streaming load in eGPU enclosure. Real-world validation from RTX 6000 Pro Ada testing (themarkymark/Waivio) confirms 300W as optimal operating point: 95.4% throughput, +44% efficiency vs 600W, stable power draw. Native 300W Blackwell part (GB202, TSMC N4P, GDDR7) expected to match or exceed. Risk: eGPU enclosure airflow — requires Sprint 0 thermal soak test (S0.6). Do NOT under-power below 300W — empirically shown to reduce efficiency, not improve it (250W draws higher peak spikes and delivers only 84% throughput).
- NFR5: Zero-copy where hardware allows — Metal _as_buffer() for reads, pinned staging for CUDA DMA
- NFR6: Wire format byte-identical across backends — no endianness conversion, no format negotiation

**Scale & Complexity:**
- Primary domain: Systems infrastructure / cross-backend GPU data plane
- Complexity level: Medium-high
- Estimated architectural components: 6 (ring buffer, stream coordinator, wire serializer, compress/decompress adapters, tinygrad device bridge, thermal monitor)

### Technical Constraints & Dependencies

| Constraint | Detail |
|-----------|--------|
| TB5 bandwidth ceiling | ~6 GB/s PCIe 4.0 x4 — hard physical limit |
| No Metal↔CUDA P2P in tinygrad | Cross-backend goes through CPU-mediated path. Decision (AD1): manage DMA directly with cuMemHostAlloc pinned staging. tinygrad's BufferCopy not designed for sustained streaming with pinned ring buffer slots. |
| tinygrad PID prerequisite | RTX PRO 6000 PID must land in tinygrad before any NV backend work |
| 300W thermal envelope | Razer Core X V2 + 650W+ ATX PSU. 300W is the empirically validated sweet spot — do not throttle below. Single GPU system draw estimated ~400-450W. Junction temp must stay <83°C sustained. |
| Metal SharedMode buffers | CPU-accessible via _as_buffer() — zero-copy reads on producer side |
| CUDA host staging | cuMemHostAlloc for pinned memory; async HtoD/DtoH via cuMemcpyAsync |
| NV HCQ DMA | COPY_CLASS subchannel with LAUNCH_DMA — more efficient for bulk transfers, but _transfer() only works within same device family |
| Codebook determinism | Seeded np.random.default_rng(seed) + QR decomposition → identical WHT rotation on both backends |
| Wire format versioning | Must be versioned from day one per RFC §9 |
| Wire header gap | Current header (layer_idx, seq_len, n_heads, flags) missing head_dim and block_count — required for consumer-side array reshaping |

### Cross-Cutting Concerns

1. **Transfer path architecture (Party Mode finding + Tom review 2026-04-04):** The NV vs CUDA backend question is a red herring. Neither backend's `_transfer()` works cross-family. **Decision: Direct DMA management (option b).** Tom (TurboQuant author) explicitly recommends bypassing tinygrad's BufferCopy: "the ring buffer memory layout is performance-critical and tinygrad's BufferCopy abstraction was not designed for sustained streaming with pinned staging buffers." Direct management gives control over pinning, alignment, and buffer lifetime at the cost of a maintenance surface outside tinygrad's abstractions.

2. **Memory pinning:** Ring buffer slots must use page-locked memory for optimal DMA. Metal handles this via unified memory. CUDA/NV side needs explicit cuMemHostAlloc or BufferSpec(host=True) in tinygrad.

3. **Codebook determinism:** The WHT rotation seed and Lloyd-Max codebook must produce bit-identical results on both backends. Sprint 0 gate S0.5 validates this. Divergence = silent quality degradation.

4. **Thermal management (Party Mode finding):** 300W is the validated operating point — do not implement power throttling as thermal strategy. If eGPU enclosure overheats, the answer is better cooling, not lower power limits. Sprint 0 gate S0.6: 5-minute sustained DMA + compute at full pipeline rate, monitoring junction temp, verifying no clock reduction. Pass: junction <83°C. Fail: requires enclosure cooling modifications.

5. **Error propagation:** TB5 hot-unplug, DMA failure, backpressure — all must propagate cleanly through the async ring buffer without data corruption or hangs.

6. **Wire deserialization gap:** RFC's _reconstruct_compressed is NotImplementedError. Wire header needs head_dim and block_count fields added so consumer can reshape flat byte buffer into per-head compressed vectors without out-of-band metadata.

### Empirical References

- **RTX 6000 Pro 300W power efficiency:** themarkymark (Waivio, 2026) — dual RTX 6000 Pro Ada 600W cards at 300W cap: 95.4% throughput retained, +44% tokens/watt efficiency, 816W system draw. Validates 300W as optimal operating point. Native Blackwell 300W part expected to match or exceed due to N4P process and GDDR7.

## Technology Stack & Project Foundation

### Primary Technology Domain

Systems infrastructure — cross-backend GPU data plane for compressed KV cache streaming between Apple Metal and NVIDIA Blackwell via tinygrad.

### Technology Stack (Domain-Determined)

No starter template evaluation applicable — technology choices are dictated by the problem domain and existing ecosystem:

| Layer | Technology | Rationale |
|-------|-----------|-----------|
| Language | Python 3.11+ | tinygrad, turboquant_plus, and the bridge coordination layer are all Python |
| GPU (Metal) | tinygrad `ops_metal.py` | Zero-copy buffer access via `_as_buffer()` on MTLResourceStorageModeShared |
| GPU (CUDA/NV) | tinygrad `ops_nv.py` | HCQ-based DMA, COPY_CLASS subchannel, timeline signal sync |
| Compression | TurboQuant (turboquant_plus) | `KVCacheCompressor` API — turbo3 (4.6x) / turbo4 (3.8x) |
| Wire format | Custom binary (stripped turbo blocks) | 16-byte headers + packed turbo3/turbo4 blocks, little-endian, no padding |
| Async runtime | Python asyncio | Producer/consumer pipeline, ring buffer semaphores |
| Testing | pytest | Matches tinygrad and turboquant_plus conventions |
| Benchmarking | Custom (latency, throughput, PPL) | Sprint 0 gates + Layer 1-3 test strategy from RFC §7 |

### Project Structure

```
src/turboquant_bridge/
├── __init__.py
├── kv_ring_buffer.py      # Double-buffer ring (RFC §4.2)
├── kv_stream.py           # Stream coordinator (RFC §4.3)
├── wire_format.py         # Stripped block serialize/deserialize
├── device_bridge.py       # tinygrad Metal↔NV buffer management
└── thermal_monitor.py     # GPU junction temp monitoring (S0.6)
tests/
├── test_ring_buffer.py    # L1.2, L1.4 — ordering, backpressure
├── test_wire_format.py    # L1.6 — strip/restore round-trip
├── test_cross_backend.py  # L1.1, L1.3 — fidelity, WHT determinism
└── test_thermal.py        # S0.6 — sustained load junction temp
pyproject.toml
```

### Dependencies

- `tinygrad` (local, from `./tinygrad/`) — GPU device management, buffer allocation, Metal + NV backends
- `turboquant_plus` (local, from `./turboquant_plus/`) — `KVCacheCompressor`, codebook, rotation matrix
- `numpy` — array manipulation, wire format packing
- `pytest` — test framework

## Core Architectural Decisions

### Decision Priority Analysis

**Critical Decisions (Block Implementation):**
1. Transfer path architecture — direct DMA management (cuMemHostAlloc + cuMemcpyAsync)
2. Wire format versioning — version byte in packet header with separate k_format/v_format
3. Compression kernel source — turboquant_plus Python for Sprint 0
6. Asymmetric K/V compression — support q8_0 K + turbo V configs

**Important Decisions (Shape Architecture):**
4. Ring buffer memory model — Python bytes, upgrade to pinned arrays if jitter measured
5. Error propagation — sentinel through ring buffer channel

**Deferred Decisions (Post-Sprint 0):**
- tinygrad custom ops for on-GPU compress/decompress (evaluate after Sprint 0 validation)
- Fallback to tinygrad BufferCopy (only if direct DMA management proves too fragile across tinygrad updates)
- Pre-allocated pinned ring buffer slots (only if GC jitter measured in L3 benchmarks)

### Decision 1: Transfer Path — Direct DMA Management

- **Choice:** Manage Metal→host→CUDA DMA directly using `cuMemHostAlloc` for pinned staging buffers, bypassing tinygrad's `BufferCopy` abstraction
- **Rationale:** Tom (TurboQuant author) recommends this approach: "the ring buffer memory layout is performance-critical and tinygrad's BufferCopy abstraction was not designed for sustained streaming with pinned staging buffers." The ring buffer requires persistent pinned allocations with controlled lifetime and alignment — `BufferCopy`'s per-transfer allocation model adds overhead and prevents the memory layout optimisations needed for sustained streaming. Neither backend's `_transfer()` supports cross-family anyway — both paths go through CPU regardless.
- **Tradeoff:** Creates a maintenance surface outside tinygrad's abstractions. Ring buffer memory management, pinning, and DMA scheduling are owned by `device_bridge.py`. Changes to tinygrad's NV backend internals may require updates.
- **Metal side:** `_as_buffer()` → memoryview → pinned host buffer (zero-copy read into DMA-ready memory)
- **CUDA side:** `cuMemHostAlloc` pinned staging → `cuMemcpyAsync` HtoD into VRAM
- **Affects:** device_bridge.py, all integration tests, ring buffer slot allocation

### Decision 2: Wire Format Versioning — Per-Packet Header

- **Choice:** Extend wire header to 28 bytes: add `version: uint8`, `k_format: uint8` (turbo3=3, turbo4=4, q8_0=8), `v_format: uint8`, `reserved: uint8`
- **Rationale:** Self-describing packets with per-tensor format fields. Supports asymmetric K/V compression (Decision 6). No connection handshake needed — producer and consumer are coroutines in the same process. Local PCIe bridge, not a network protocol.
- **Affects:** wire_format.py, kv_stream.py serialization

### Decision 3: Compression Source — turboquant_plus Python

- **Choice:** Use `KVCacheCompressor` from turboquant_plus for Sprint 0 and initial implementation
- **Rationale:** Already exists, tested, reference implementation. CPU-bound but sufficient for architecture validation. On-GPU tinygrad custom ops are the production path — evaluate after Sprint 0 proves the architecture.
- **Affects:** kv_stream.py, dependencies on turboquant_plus

### Decision 4: Ring Buffer Memory — Python bytes

- **Choice:** Ring buffer slots hold `bytes` objects. GC manages lifetime.
- **Rationale:** Matches RFC §4.2 design. Simple, no sizing assumptions. Turbo3 block sizes are deterministic given model config, so upgrade to pre-allocated pinned arrays is clean if needed.
- **Fallback:** Pre-allocated pinned `np.ndarray` per slot if GC jitter appears in L3 benchmarks
- **Affects:** kv_ring_buffer.py

### Decision 5: Error Propagation — Ring Buffer Sentinel

- **Choice:** Errors flow as sentinel values through the ring buffer (slot.data = None + error flag). Consumer detects and raises.
- **Rationale:** Explicit, inspectable, ordered. The ring buffer is the communication channel — errors should flow through the same channel as data. Avoids non-deterministic task cancellation during decompress with shared mutable KV cache state.
- **Affects:** kv_ring_buffer.py, kv_stream.py

### Decision 6: Asymmetric K/V Compression Support

- **Choice:** Support asymmetric compression configs where K and V use different formats (e.g., `q8_0 K + turbo3 V`), not just symmetric `turbo3 K + turbo3 V`
- **Rationale:** Tom's review confirms: "V errors scale linearly while K errors amplify through softmax." Asymmetric configs (q8_0 K + turbo3 V) give better quality on sensitive models while retaining most bandwidth savings — V is the bulk of the transfer. This is a quality-for-free optimisation on the wire format.
- **Wire format impact:** The per-packet header must carry separate `k_format` and `v_format` fields (replacing the single `format` byte from Decision 2). Strip/restore logic in `wire_format.py` must dispatch per-tensor based on format.
- **Head dimension caveat (Tom, 2026-04-04):** Asymmetric configs had a rotation matrix initialisation bug on `head_dim=256` that produced corrupt output. Fixed in Tom's latest branches. The bridge must pin to fixed branches and add a head_dim validation guard in `wire_format.py` that rejects unsupported head_dim values until the corresponding FA kernel instantiations are confirmed.
- **Affects:** wire_format.py (header, strip/restore dispatch), kv_stream.py (separate k_bits/v_bits already supported in KVCacheCompressor API), test_wire_format.py (asymmetric round-trip tests)

### Decision Impact Analysis

**Implementation Sequence:**
1. Wire format (Decision 2) — defines the packet structure everything else builds on
2. Ring buffer (Decisions 4, 5) — transport layer with error handling
3. Stream coordinator with turboquant_plus compression (Decision 3) — producer/consumer pipeline
4. Device bridge via tinygrad BufferCopy (Decision 1) — connects to real GPU buffers

**Cross-Component Dependencies:**
- Wire format header size (28 bytes) flows into stream coordinator serialization and test fixtures
- Transfer path choice (BufferCopy) determines how device_bridge.py allocates and moves buffers
- Compression source (Python) means compress/decompress latency is CPU-bound — pipeline timing in tests must account for this
- Error sentinel design must be consistent between ring buffer and stream coordinator

## Implementation Patterns & Consistency Rules

### Naming Conventions

**Module naming:** snake_case, descriptive nouns
- `kv_ring_buffer.py`, `wire_format.py`, `device_bridge.py`

**Class naming:** PascalCase
- `KVRingBuffer`, `KVStreamCoordinator`, `WirePacket`

**Function naming:** snake_case, verb_noun
- `compress_layer()`, `strip_padding()`, `acquire_write()`

**Constants:** UPPER_SNAKE_CASE
- `HEADER_FMT`, `HEADER_SIZE`, `FLAG_KV`, `TURBO3_BLOCK_SIZE`

**Domain vocabulary — use these terms consistently:**

| Concept | Canonical term | NOT |
|---------|---------------|-----|
| One KV cache layer's compressed data | `layer packet` | chunk, slab, frame |
| Wire header + compressed blocks | `wire packet` | message, payload, segment |
| Ring buffer position | `slot` | cell, entry, position |
| Metal → CUDA data movement | `transfer` | copy, move, send |
| TurboQuant compress | `compress` | encode, quantize, pack |
| TurboQuant decompress | `decompress` | decode, dequantize, unpack |
| Remove block padding for wire | `strip` | trim, compact, shrink |
| Restore block padding from wire | `restore` | pad, expand, unstrip |

### Wire Format Patterns

**All integer fields:** Little-endian (`<` prefix in struct format strings). No exceptions.

**Header struct format string:** Always reference `HEADER_FMT` and `HEADER_SIZE` constants from `wire_format.py`. Never hardcode `"<IIIIII"` or `28` in other modules.

**Block size calculations:** Always derive from constants, never hardcode:
```python
# CORRECT
TURBO3_WIRE_BLOCK = 16   # bytes per 32 elements (qs[8] + signs[4] + norm[4])
TURBO4_WIRE_BLOCK = 68   # bytes per 128 elements (qs[64] + norm[4])
n_blocks = (n_heads * seq_len * head_dim) // 32  # turbo3
wire_size = n_blocks * TURBO3_WIRE_BLOCK

# WRONG — hardcoded magic numbers
wire_size = n_heads * seq_len * head_dim // 32 * 16
```

### Async Patterns

**Semaphore discipline:** Every `acquire()` must have a matching `release()` in the same method or a clearly documented handoff. Use try/finally for release:
```python
# CORRECT
slot = await self.ring.acquire_write()
try:
    slot.data = self._serialize(compressed)
    slot.layer_idx = layer_idx
finally:
    self.ring.release_write(slot)
```

**Coroutine naming:** Async functions that are pipeline stages use `produce_*` / `consume_*` prefix. Helper coroutines use verb_noun.

**No bare asyncio.sleep()** in pipeline code. If waiting is needed, it flows through ring buffer semaphores.

### Error Handling Patterns

**Pipeline errors:** Always sentinel through the ring buffer (Decision 5). Never raise from producer directly.

**Validation errors** (bad header, wrong version, size mismatch): Raise `WireFormatError(msg)` — a custom exception defined in `wire_format.py`.

**Device errors** (buffer alloc failure, DMA timeout): Raise `DeviceBridgeError(msg)` — defined in `device_bridge.py`.

**Test errors:** Use pytest.raises with specific exception types. Never catch bare `Exception`.

### Test Patterns

**Test location:** `tests/` directory, mirroring source structure:
- `tests/test_ring_buffer.py` tests `src/turboquant_bridge/kv_ring_buffer.py`

**Fixture strategy:**
- Synthetic compressed data fixtures for unit tests (no GPU required)
- `@pytest.fixture` for ring buffer instances, wire packets, mock device buffers
- Mark hardware-dependent tests with `@pytest.mark.hardware` — skipped when GPU unavailable

**Test naming:** `test_<what>_<condition>_<expected>`:
- `test_ring_buffer_full_producer_blocks()`
- `test_wire_header_missing_fields_raises_format_error()`
- `test_strip_restore_roundtrip_matches_original()`

**Assertion style:** One logical assertion per test. Use `np.testing.assert_array_equal` for array comparisons, `pytest.approx` for floats.

### Buffer Management Patterns

**tinygrad buffer access:** Always go through `device_bridge.py`. No direct `Buffer.copyin()`/`copyout()` calls in stream coordinator or ring buffer code.

**Metal reads:** Use `_as_buffer()` → memoryview → pinned host buffer. Document that this is zero-copy into DMA-ready memory.

**CUDA writes:** Use `cuMemHostAlloc` for pinned staging → `cuMemcpyAsync` HtoD into VRAM. Ring buffer slots own their pinned allocations — no per-transfer allocation. Document the copy chain.

**Head dimension validation:** Before compress/decompress, validate `head_dim` against supported values. Models with `head_dim=256` or `head_dim=512` (Gemma 4, Qwen3.5-122B) require specific FA kernel instantiations. Reject unsupported values with `WireFormatError` until kernels are confirmed available. See Decision 6 for asymmetric config caveats.

### Enforcement Guidelines

**All AI agents MUST:**
- Use domain vocabulary table above — grep for banned synonyms before submitting
- Reference wire format constants from `wire_format.py`, never hardcode
- Pair every semaphore acquire with a finally-guarded release
- Route all device buffer operations through `device_bridge.py`
- Mark GPU-dependent tests with `@pytest.mark.hardware`

**Anti-Patterns (reject on review):**
- Magic numbers for block sizes or header offsets
- Bare `except Exception` in pipeline code
- `asyncio.sleep()` in producer/consumer coroutines
- Direct tinygrad `Buffer` operations outside `device_bridge.py`
- Tests that require GPU hardware without `@pytest.mark.hardware`

## Project Structure & Boundaries

### Complete Project Directory Structure

```
turboquant-tinygrad-bridge/
├── README.md
├── LICENSE
├── pyproject.toml
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml                    # pytest (unit only, no GPU)
│
├── src/
│   └── turboquant_bridge/
│       ├── __init__.py               # Package exports: KVRingBuffer, KVStreamCoordinator
│       ├── wire_format.py            # FR5: Header constants, serialize/deserialize, WireFormatError
│       ├── kv_ring_buffer.py         # FR1, FR8: Double-buffer ring with sentinel error propagation
│       ├── kv_stream.py              # FR2-FR4: Producer/consumer coordinator using KVCacheCompressor
│       ├── device_bridge.py          # FR3: Direct DMA — Metal _as_buffer() + cuMemHostAlloc pinned staging
│       └── thermal_monitor.py        # NFR4: GPU junction temp polling, throttle detection
│
├── tests/
│   ├── conftest.py                   # Shared fixtures: synthetic compressed data, ring buffers
│   ├── test_wire_format.py           # Header pack/unpack, version validation, strip/restore roundtrip (L1.6)
│   ├── test_ring_buffer.py           # Ordering (L1.2), backpressure (L1.4), sentinel propagation
│   ├── test_kv_stream.py             # Producer/consumer pipeline with synthetic data
│   ├── test_device_bridge.py         # BufferCopy integration (requires @pytest.mark.hardware)
│   ├── test_cross_backend.py         # Metal→NV roundtrip fidelity (L1.1), WHT determinism (L1.3)
│   ├── test_thermal.py              # Sustained load junction temp (S0.6, @pytest.mark.hardware)
│   └── benchmarks/
│       ├── bench_migration.py        # L3.2, L3.3: KV migration latency at 32K/128K
│       ├── bench_throughput.py        # S0.2: TB5 sustained throughput
│       └── bench_quality.py           # L3.4: PPL measurement at various contexts
│
├── docs/
│   ├── RFC-double-buffer-kv-bridge.md
│   └── RFC-metal-cuda-kv-bridge.md
│
├── tinygrad/                          # Git submodule or local clone — GPU runtime
├── turboquant_plus/                   # Git submodule or local clone — KVCacheCompressor
└── llama-cpp-turboquant/             # Reference — Metal/CUDA kernel source
```

### Architectural Boundaries

**Module Boundary Map:**

```
┌─────────────────────────────────────────────────────┐
│                   kv_stream.py                       │
│   Producer/Consumer pipeline orchestration           │
│   IMPORTS: wire_format, kv_ring_buffer,              │
│            device_bridge, turboquant_plus             │
├──────────┬──────────────┬───────────────────────────┤
│          │              │                           │
│  wire_format.py   kv_ring_buffer.py   device_bridge.py
│  Header pack/     Slot management,    tinygrad Buffer
│  unpack, block    semaphores,         ops, Metal
│  strip/restore    sentinel errors     _as_buffer(),
│  IMPORTS: struct  IMPORTS: asyncio    NV BufferCopy
│                                       IMPORTS: tinygrad
├──────────┴──────────────┴───────────────────────────┤
│                thermal_monitor.py                     │
│   Independent — polled by kv_stream or called         │
│   externally. No pipeline dependencies.               │
└─────────────────────────────────────────────────────┘
```

**Import rules (strict):**
- `wire_format.py` — no project imports. Only `struct`, `numpy`.
- `kv_ring_buffer.py` — no project imports. Only `asyncio`, `dataclasses`.
- `device_bridge.py` — imports `tinygrad` only. No other bridge modules.
- `thermal_monitor.py` — imports `tinygrad` only. Independent module.
- `kv_stream.py` — imports all other bridge modules + `turboquant_plus`. This is the integration point.

**No circular imports.** Dependency flows one direction: `kv_stream` → everything else.

### Requirements to Structure Mapping

| Requirement | File | Test |
|------------|------|------|
| FR1: Double-buffer ring | `kv_ring_buffer.py` | `test_ring_buffer.py` |
| FR2: TurboQuant compress | `kv_stream.py` (calls turboquant_plus) | `test_kv_stream.py` |
| FR3: DMA over TB5 | `device_bridge.py` | `test_device_bridge.py` |
| FR4: Decompress to tinygrad.Tensor | `kv_stream.py` + `device_bridge.py` | `test_cross_backend.py` |
| FR5: Wire format | `wire_format.py` | `test_wire_format.py` |
| FR6: Codebook consistency | (validated at turboquant_plus level) | `test_cross_backend.py` |
| FR7: TB5 disconnect recovery | `kv_ring_buffer.py` sentinel | `test_ring_buffer.py` |
| FR8: Backpressure | `kv_ring_buffer.py` semaphores | `test_ring_buffer.py` |
| FR9: tinygrad PID | (separate micro-PR to tinygrad) | manual verification |
| NFR1: ≥5 GB/s throughput | `device_bridge.py` | `benchmarks/bench_throughput.py` |
| NFR2: ≤45 ms migration | `kv_stream.py` pipeline | `benchmarks/bench_migration.py` |
| NFR3: PPL <1.1% delta | turboquant_plus quality | `benchmarks/bench_quality.py` |
| NFR4: Thermal stability | `thermal_monitor.py` | `test_thermal.py` |

### Data Flow

```
Metal GPU (M3 Ultra)                           CUDA/NV GPU (RTX PRO 6000)
┌──────────────┐                               ┌──────────────┐
│ KV cache      │                               │ KV cache      │
│ (tinygrad     │                               │ (tinygrad     │
│  Tensor)      │                               │  Tensor)      │
└──────┬───────┘                               └──────▲───────┘
       │ _as_buffer() → memoryview (zero-copy)        │ cuMemcpyAsync HtoD from pinned host
       ▼                                               │
┌──────────────┐                               ┌──────────────┐
│ KVStream      │                               │ KVStream      │
│ .produce()    │                               │ .consume()    │
│               │                               │               │
│ compress via  │    ┌─────────────────┐        │ decompress    │
│ KVCache       │───►│ KVRingBuffer    │───────►│ via KVCache   │
│ Compressor    │    │ (Python bytes   │        │ Compressor    │
│               │    │  in slots)      │        │               │
│ strip padding │    └─────────────────┘        │ restore pad   │
│ pack header   │     wire_format.py            │ unpack header │
└──────────────┘     serialize/deserialize       └──────────────┘

    device_bridge.py manages both sides:
    Metal: _as_buffer() → memoryview (zero-copy read)
    NV: BufferSpec(host=True) → pinned staging → copyin()
```

## Architecture Validation Results

### Coherence Validation ✅

**Decision Compatibility:** All six core decisions (direct DMA transfer, per-packet versioning with split K/V format, turboquant_plus Python compression, Python bytes ring buffer, sentinel error propagation, asymmetric K/V compression) form a coherent stack. Deferred decisions each have explicit trigger conditions (S0.2 failure, GC jitter in L3, post-Sprint 0 evaluation).

**Pattern Consistency:** Domain vocabulary, wire format constants, async semaphore discipline, and import rules are internally consistent. No contradictions between patterns and decisions.

**Structure Alignment:** Module boundary map enforces one-directional dependency flow. Every FR/NFR maps to a specific source file and test file.

### Requirements Coverage ✅

| Category | Count | Status |
|----------|-------|--------|
| Functional Requirements (FR1-FR9) | 9 | All mapped to source files and tests |
| Non-Functional Requirements (NFR1-NFR6) | 6 | All mapped to tests or benchmarks |
| Sprint 0 Gates (S0.1-S0.6) | 6 | S0.6 added from Party Mode thermal analysis |
| Layer 1-3 Test Cases | 16 | All mapped to test files or benchmark scripts |

**No coverage gaps.** FR9 (tinygrad PID) is tracked as a separate prerequisite micro-PR.

### Implementation Readiness ✅

**Decision Completeness:** All critical decisions documented with rationale, fallback conditions, and affected files. Technology versions specified (Python 3.11+, tinygrad local, turboquant_plus local).

**Structure Completeness:** Full directory tree with every file mapped to requirements. Import rules prevent architectural drift.

**Pattern Completeness:** Domain vocabulary (8 canonical terms), wire format constants, async discipline, error handling hierarchy, test naming/fixture/assertion patterns all specified. Anti-pattern list provides clear rejection criteria.

### Gap Analysis

**Critical Gaps:** None

**Important Gaps (Post-Sprint 0):**
1. Unsupported wire version handling → `WireFormatError`, propagate via sentinel
2. `_reconstruct_compressed` reshape logic → implementation story, not architecture gap
3. CI without GPU → unit tests only in CI, hardware tests manual or self-hosted

**Deferred (Post-Sprint 0):**
- Profiling/tracing instrumentation
- exo integration API versioning

### Architecture Completeness Checklist

**✅ Requirements Analysis**
- [x] Project context analyzed with Party Mode multi-perspective review
- [x] Scale and complexity assessed (medium-high)
- [x] Technical constraints identified (10 constraints documented)
- [x] Cross-cutting concerns mapped (6 concerns including Party Mode findings)
- [x] Empirical thermal data integrated (themarkymark RTX 6000 Pro benchmarks)

**✅ Architectural Decisions**
- [x] 6 critical/important decisions documented with rationale (updated 2026-04-04 per Tom review)
- [x] 3 deferred decisions with explicit trigger conditions
- [x] Technology stack specified
- [x] Implementation sequence defined
- [x] Cross-component dependencies mapped

**✅ Implementation Patterns**
- [x] Domain vocabulary table (8 terms)
- [x] Wire format constant rules
- [x] Async semaphore discipline
- [x] Error handling hierarchy (WireFormatError, DeviceBridgeError, sentinel)
- [x] Test patterns (naming, fixtures, hardware marks)
- [x] Buffer management patterns
- [x] Anti-pattern rejection list

**✅ Project Structure**
- [x] Complete directory tree
- [x] Module boundary map with import rules
- [x] FR/NFR to file mapping table
- [x] Data flow diagram
- [x] No circular dependencies

### Architecture Readiness Assessment

**Overall Status:** READY FOR IMPLEMENTATION

**Confidence Level:** High — requirements fully covered, decisions coherent, patterns comprehensive, empirical thermal data integrated.

**Key Strengths:**
- "Simple first, optimise with data" philosophy gives clear Sprint 0 → production upgrade path
- Strict module boundaries and import rules prevent architectural drift across AI agents
- Domain vocabulary eliminates naming ambiguity
- Party Mode surfaced two critical findings (transfer path reframing, thermal validation gate) that would have been missed in solo analysis

**Areas for Future Enhancement:**
- On-GPU tinygrad custom ops for compress/decompress (post-Sprint 0)
- Fallback to tinygrad BufferCopy if direct DMA management proves too fragile
- Pre-allocated pinned ring buffer slots if GC jitter observed
- exo integration layer (PR 2, post-Spark)

### Implementation Handoff

**AI Agent Guidelines:**
- Follow all architectural decisions exactly as documented
- Use domain vocabulary table — reject banned synonyms
- Respect module import rules — no circular dependencies
- Reference wire format constants from wire_format.py, never hardcode
- Mark all GPU-dependent tests with @pytest.mark.hardware
- Route all tinygrad Buffer operations through device_bridge.py

**First Implementation Priority:**
1. `wire_format.py` — header constants, pack/unpack, strip/restore, WireFormatError
2. `kv_ring_buffer.py` — slots, semaphores, sentinel error propagation
3. `kv_stream.py` — producer/consumer with turboquant_plus KVCacheCompressor
4. `device_bridge.py` — Metal _as_buffer() reads, NV BufferCopy writes
5. Tests for each module (unit tests first, hardware tests when rig is built)

