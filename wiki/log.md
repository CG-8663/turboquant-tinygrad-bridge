# TurboQuant Bridge Wiki — Log

## [2026-04-04] ingest | Initial wiki seeding — TurboQuant ecosystem research

Seeded wiki with public research on key contributors and ecosystem state. Sources: Twitter/X profiles, GitHub repos, llama.cpp discussions, benchmark data.

**Pages created:** 3 entities, 5 concepts, 9 sources
**Pages updated:** None (initial seed)
**Key findings:**
- Tom Turney's turbo3/4 blocks are byte-identical on Metal and CUDA — validates our wire format assumption
- TurboQuant+ now extends to weight compression (Q8_0 GGUFs), not just KV cache
- Independent researchers confirming findings on RTX 4090 (sztlink)
- Prince Canuma's MLX already supports CUDA backend (`pip install mlx[cuda]`)
- 104B model at 128K context running on MacBook with turbo3 (74GB peak, PPL 4.024)
- No one has attempted cross-device KV streaming using compressed format — our bridge is novel
