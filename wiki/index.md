# TurboQuant Bridge Wiki — Index

## Entities

- [Tom Turney (@no_stp_on_snek)](entities/tom-turney.md) — TurboQuant llama.cpp implementer, Metal kernels, turbo2/3/4
- [Prince Canuma (@Prince_Canuma)](entities/prince-canuma.md) — MLX ecosystem lead, mlx-vlm, mlx-audio, cross-backend inference
- [Andrej Karpathy (@karpathy)](entities/andrej-karpathy.md) — LLM Wiki pattern author, knowledge compounding via persistent wiki

## Concepts

- [TurboQuant Compression](concepts/turboquant-compression.md) — KV cache compression: turbo2/3/4, WHT rotation, Lloyd-Max codebooks
- [TurboQuant+ Weight Compression](concepts/turboquant-plus-weights.md) — Post-training weight compression using same WHT+Lloyd-Max principles
- [Cross-Backend Wire Format](concepts/cross-backend-wire-format.md) — Using compressed KV blocks as DMA wire protocol between Metal and CUDA
- [LLM Wiki Pattern](concepts/llm-wiki-pattern.md) — Persistent compiled knowledge base maintained by LLMs
- [MLX Cross-Backend](concepts/mlx-cross-backend.md) — Apple MLX framework running on both Metal and CUDA

## Sources

- [Tom Turney — TurboQuant llama.cpp Implementation](sources/turney-turboquant-llama-cpp.md)
- [Tom Turney — TurboQuant+ Weight Compression](sources/turney-turboquant-plus-weights.md)
- [Tom Turney — Independent Validation](sources/turney-independent-validation.md)
- [Prince Canuma — Voxtral on MLX + CUDA](sources/canuma-voxtral-mlx.md)
- [Prince Canuma — MLX-VLM](sources/canuma-mlx-vlm.md)
- [Prince Canuma — MLX-Audio](sources/canuma-mlx-audio.md)
- [TurboQuant llama.cpp Discussion #20969](sources/llamacpp-discussion-20969.md)
- [TurboQuant Benchmarks — Apple Silicon](sources/turboquant-benchmarks-apple-silicon.md)
- [Karpathy — LLM Wiki Pattern](sources/karpathy-llm-wiki.md)
