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

## Portkey gateway: explicit-parameter forwarding

- **Pass-through reality**: Portkey routes requests to many backends (OpenAI, Anthropic, Gemini, Grok, etc.) and forwards payloads verbatim. Defaults injected by AbstractCore can break strict models.
- **Rule**: Only forward optional generation parameters (`temperature`, `top_p`, token caps) when the user explicitly sets them.
- **Reasoning models**: OpenAI reasoning families (gpt-5/o1) reject `temperature`/`top_p` and require `max_completion_tokens`; drop unsupported params with a warning.
- **Routing modes**: Config ID, virtual key, and provider-direct headers are mutually exclusive — mixing them causes gateway errors.

## Configuration UX: provider-agnostic prompts

- **Principle**: CLI prompts and docs must not hard-code provider lists when the runtime accepts any provider/model pair.
- **Why**: Restrictive prompts mislead users and silently constrain valid setups; examples should be illustrative, not exhaustive.
- **Practice**: Collect provider/model as free-form input and include optional examples without validation gates.

## DEPRECATED

(none yet)

