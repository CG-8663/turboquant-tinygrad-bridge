---
tags: [person, llm-wiki, knowledge-management, training]
sources: 1
last_updated: 2026-04-04
---

# Andrej Karpathy (@karpathy)

## Role

AI researcher, former Tesla AI lead, OpenAI founding member. Author of the [LLM Wiki pattern](../concepts/llm-wiki-pattern.md) — a framework for building persistent knowledge bases maintained by LLMs.

## LLM Wiki Pattern

Published April 2026. Three-layer architecture:
1. **Raw sources** — immutable source documents
2. **Wiki** — LLM-maintained markdown pages with cross-references
3. **Schema** — conventions and workflows (e.g. CLAUDE.md)

Key operations: ingest, query, lint.

Core insight: RAG retrieves and re-derives knowledge every query. A wiki compiles once and compounds over time. The LLM handles the bookkeeping humans abandon.

## Relevance to Bridge

Two integration points planned:
1. **Flywheel training** (chronaracli project) — wiki layer between raw Elasticsearch interaction logs and QLoRA fine-tuning. Structured training signal instead of raw logs.
2. **Bridge quality validation** — same sources ingested through Metal-only, CUDA-only, and split inference paths. Identical wiki output = compression doesn't degrade downstream quality. Stronger signal than perplexity scores.

## Sources

- [LLM Wiki pattern document](../sources/karpathy-llm-wiki.md)
