# Proposed: Remote prompt-cache session parity and observability

## Metadata
- Created: 2026-05-06
- Status: Proposed
- Completed: N/A

## Context

`CachedSession` already gives the strongest behavior with local providers that can expose
in-process prefix/KV caches: MLX, HuggingFace transformers, and supported HuggingFace GGUF
paths. Remote providers are more limited today: AbstractCore can pass a stable
`prompt_cache_key` and provider-specific retention hints, but the backend may treat that key as a
best-effort cache-bucketing hint rather than a controllable cache object.

Relevant current docs/code:

- `docs/prompt-caching.md`
- `examples/prompt_caching/cached_session_quickstart.py`
- `examples/prompt_caching/prompt_cache_repl_demo.py`
- `abstractcore/core/cached_session.py`
- `abstractcore/providers/base.py`
- remote provider implementations for OpenAI, Anthropic, OpenAI-compatible, OpenRouter, and
  Portkey.

## Problem

Users can see strong local cache wins, but remote cache behavior is harder to reason about:

- support differs by provider and model;
- some providers expose cache usage only in response usage metadata;
- some OpenAI-compatible servers may accept cache fields but ignore them;
- `CachedSession(prompt_cache_strategy="auto")` cannot always tell whether remote cache reuse is
  actually happening;
- there is no consistent user-facing observability story for remote prompt-cache hits/misses.

This creates a documentation and ergonomics gap, not a correctness bug.

## Proposal

Improve remote prompt-cache parity in three phases:

1. Add provider capability probes that distinguish:
   - no remote cache support;
   - accepts cache key/hint only;
   - reports cache usage in response metadata;
   - exposes a real control plane.
2. Normalize cache usage telemetry into `GenerateResponse.metadata`, for example:
   - `prompt_cache.mode`
   - `prompt_cache.key`
   - `prompt_cache.requested`
   - `prompt_cache.cached_tokens`
   - `prompt_cache.provider_raw`
3. Teach `CachedSession` and examples to display remote cache observability when available, while
   still making clear that remote caching is best-effort and provider-owned.

## Why

This would make `CachedSession` less local-provider-specific without pretending remote caches are
equivalent to in-process KV caches. Users could make informed decisions about whether a remote
workflow is benefiting from cache reuse.

## Evidence needed before promotion

Promote this to `planned/` only if at least two of these are true:

- OpenAI or Anthropic exposes stable enough response metadata to normalize cache usage.
- A target OpenAI-compatible backend used by AbstractCore users supports cache keys or cache
  metrics.
- Users report confusion about whether `CachedSession` helps with remote providers.
- Server or Docker examples need a first-class "large stable context + remote provider" story.

## Suggested implementation

- Extend provider capability metadata rather than adding special cases directly in `CachedSession`.
- Keep cache telemetry optional and JSON-safe.
- Prefer a small normalized metadata block over a new public class unless the shape becomes large.
- Update `docs/prompt-caching.md` and `examples/prompt_caching/README.md`.
- Add unit tests with fake provider responses for:
  - OpenAI cached token metadata;
  - Anthropic cache-control enabled/disabled behavior;
  - OpenAI-compatible backend that accepts the key but reports no metrics.

## Non-goals

- Do not promise deterministic remote cache hits.
- Do not emulate remote caches inside AbstractCore Server.
- Do not make remote cache control-plane APIs look equivalent to local KV cache save/load.
- Do not require remote providers for `CachedSession` to remain useful.

## Validation ideas

- Unit tests for metadata normalization.
- Live smoke tests gated behind provider keys and explicit env flags.
- Documentation examples that show both "cache requested" and "cache observed" states.

## Guidance for future agents

Inspect current provider usage payloads and response metadata before implementing. Provider cache
semantics change over time, so treat current provider docs and live responses as more authoritative
than this proposed item.
