# KnowledgeBase

This document collects **durable, cross-cutting insights** about AbstractCore’s design and operational behavior.
It is intentionally brief and additive: we append new learnings, and deprecate by moving to a dedicated section.

## Media processing: no silent truncation policy

- **Principle**: Processors (TextProcessor, PDFProcessor, OfficeProcessor) output **full content without truncation**. Silent data loss is considered worse than context overflow errors.
- **Token estimation**: All text-based processors add `estimated_tokens` and `content_length` to `MediaContent.metadata` using `TokenUtils.estimate_tokens()`.
- **Handler consolidation**: `BaseProviderMediaHandler.estimate_tokens_for_media()` uses metadata's `estimated_tokens` when available; subclasses override `_estimate_image_tokens()` for provider-specific image calculations.
- **Caller responsibility**: The caller (or higher layers) decides how to handle large content: error, summarize via `BasicSummarizer`, chunk, or proceed.
- **Rationale**: Different models have different context limits (8K to 200K+). Truncating at the processor level would be wrong for at least one use case. The SOTA approach is to surface the problem and let the user decide.

## Capabilities database: embedding models

- **Canonical signal**: Embedding-only models should be marked with `model_type: "embedding"` in `abstractcore/assets/model_capabilities.json`.
- **Neutral generation fields**: For embedding models, set `max_output_tokens: 0`, `tool_support: "none"`, and `structured_output: "none"` (they do not produce chat text and do not call tools).
- **Alias hygiene**: Provider-specific identifiers (e.g. LMStudio quantization suffixes like `@q6_k`) should be added as aliases of the canonical model entry so architecture detection does not fall back to generative defaults.

## DEPRECATED

(none yet)

