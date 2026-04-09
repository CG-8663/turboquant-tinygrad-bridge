# X Thread Draft — TQBridge Launch

---

**Tweet 1 (hook):**

We built TQBridge — a compressed KV cache bridge for distributed LLM inference.

531 tok/s over Thunderbolt 5 eGPU. 71% faster than f16 at 32K context. 1MB Docker image, zero dependencies.

Split a 27B model across a Mac, NVIDIA GPUs, and server nodes.

docker run -p 9473:9473 chronaragroup/chronara-bridge

🧵👇

---

**Tweet 2 (the problem):**

The problem: eGPUs have 1.5ms latency per USB4 round-trip. Sending uncompressed KV cache (256KB/token) wastes this expensive link.

TurboQuant compresses KV 4-5x. Our custom Metal + CUDA kernels compress in 0.2ms. The bus is the only bottleneck left.

We went from 12 tok/s → 531 tok/s on the same hardware.

---

**Tweet 3 (the numbers):**

Qwen3-8B on M3 Ultra — decode tok/s vs context depth:

f16:     67 → 40 → 28 → 17 (drops 75%)
turbo4:  54 → 50 → 38 → 29 (drops 46%)

Turbo4 beats f16 at 4K+ context.
+71% faster at 32K.

Prefill speed identical (±0.6%).

Built on @TheTom's TurboQuant+ fork of llama.cpp.

---

**Tweet 4 (how it works):**

Architecture:

Prefill node (NVIDIA/Mac) compresses KV → 26KB/token
  ↓ TCP or Thunderbolt 5
Decode nodes (Docker, any platform) decompress → ready for inference

Custom kernels:
• CUDA: 4,117 tok/s (RTX PRO 6000)
• Metal: 3,384 tok/s (M3 Ultra)
• C fallback: 295 tok/s (any CPU)

---

**Tweet 5 (cluster):**

4-node cluster operational:
• M3 Ultra (Metal) — 3,384 tok/s
• RTX PRO 6000 (CUDA) — 4,117 tok/s
• 2× ASUS GX10 GB10 (200GbE) — decode nodes

All connected. KV compressed and routed via TQBridge.

Target: 12,000+ tok/s aggregate with RTX on native PCIe.

---

**Tweet 6 (for devs):**

For developers:

• 159 Python tests + 64 C assertions
• Bit-exact C↔Python compression parity
• Wire protocol: 40-byte header, CRC32, format negotiation
• TCP transport with retries + auto-reconnect
• Multi-arch Docker (amd64 + arm64)

github.com/CG-8663/turboquant-tinygrad-bridge

---

**Tweet 7 (acknowledgments):**

Built on the shoulders of:

@TheTom — TurboQuant+ llama.cpp fork. Asymmetric K/V, sparse V dequant, Metal kernels. The compression that makes this possible.

@__tinygrad__ — GPU runtime for Metal + CUDA kernel compilation.

@Prince_Canuma — inspiration from MLX ecosystem (mlx-audio, mlx-vlm). Showing what's possible when you build native for Apple Silicon.

Google Research — TurboQuant paper (ICLR 2026).

@AlexZiskind — community validation that asymmetric turbo works on real hardware.

---

**Tweet 8 (call to action):**

Try it:

docker run -p 9473:9473 chronaragroup/chronara-bridge

Full results & benchmarks:
github.com/CG-8663/turboquant-tinygrad-bridge/blob/main/docs/results-summary.md

Docker Hub:
hub.docker.com/r/chronaragroup/chronara-bridge

What models are you running? Let us know what you see.

@chronaboratory

---

## Tag suggestions:
- @TheTom (Tom Turney, TurboQuant+)
- @__tinygrad__ (tinygrad)
- @AlexZiskind (Alex Ziskind, community testing)
- @Prince_Canuma (Prince Canuma, MLX ecosystem inspiration)
- @chronaboratory (Chronara)
- @exaboratory (exo, if pitching integration)

## Hashtags:
#TurboQuant #LLM #LocalAI #eGPU #ThunderboltAI #DistributedInference
