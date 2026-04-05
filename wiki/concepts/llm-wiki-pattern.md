---
tags: [knowledge-management, training, flywheel, quality-gate]
sources: 1
last_updated: 2026-04-04
---

# LLM Wiki Pattern

Framework by [Andrej Karpathy](../entities/andrej-karpathy.md) (April 2026) for building persistent knowledge bases maintained by LLMs.

## Architecture

Three layers:
1. **Raw sources** — immutable source documents (articles, papers, data). LLM reads but never modifies.
2. **Wiki** — LLM-generated markdown files. Summaries, entity pages, concept pages, cross-references. LLM owns entirely.
3. **Schema** — conventions and workflows (CLAUDE.md, AGENTS.md). How the wiki is structured, what to do on ingest/query/lint.

## Operations

- **Ingest** — new source added, LLM reads it, writes summary, updates index, updates relevant entity/concept pages. Single source may touch 10-15 wiki pages.
- **Query** — questions answered against the wiki with citations. Good answers filed back as new wiki pages.
- **Lint** — health check: contradictions, stale claims, orphan pages, missing cross-references, data gaps.

## Core Insight

RAG re-derives knowledge from scratch every query. A wiki compiles once and compounds. Cross-references already exist. Contradictions already flagged. Synthesis already reflects everything ingested.

## Integration with This Project

### 1. Flywheel Training (chronaracli — direct dependency)

The [chronaracli flywheel](/Volumes/Chronara-Storage-1/Projects/chronaracli/) uses the wiki pattern as a layer between raw Elasticsearch interaction logs and QLoRA fine-tuning on the GX10 cluster. The wiki produces structured training signal instead of raw logs.

**This project is a direct dependency of the flywheel.** The bridge enables cross-device inference that the flywheel needs for training at scale.

### 2. Bridge Quality Validation

Same sources ingested through three inference paths:
- Metal-only (M3 Ultra)
- CUDA-only (RTX PRO 6000)
- Split inference with compressed KV over TB5

If wiki output is identical across all three paths, compression didn't degrade downstream quality. This is a stronger end-to-end signal than perplexity scores, which only measure token-level prediction.

## Source

- [Karpathy — LLM Wiki pattern](../sources/karpathy-llm-wiki.md)
