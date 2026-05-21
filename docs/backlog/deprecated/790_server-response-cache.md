# 790 — Response caching for repeated requests

## Metadata
- Created: 2026-02-21
- Status: Deprecated
- Deprecated: 2026-05-21

## Summary
Add an optional response cache at the server/gateway layer to reuse identical responses and reduce latency/cost for repeated prompts.

## Why
Many production workloads repeat identical or near-identical prompts (health checks, FAQ, evaluation harnesses). A cache reduces spend, improves latency, and protects upstream providers. The server docs already call out a caching layer for repeated requests as a future enhancement.

## Scope
### In scope
- Request hashing that includes model, messages, tool schema, and generation parameters.
- Safe cache eligibility rules (no tools execution, no streaming, deterministic configs like `temperature=0`).
- TTL-based invalidation with size limits.
- Opt-in config (`server.cache.enabled`) + per-request override.
- Admin endpoints to inspect/clear cache with clear warnings when disabled (#FALLBACK).
- Docs updates and examples.

### Out of scope
- Semantic cache (embedding similarity).
- Cross-user personalization-aware caching.
- Caching for tool outputs (handled by runtime/tool layer).

## Dependencies
- Optional Redis for shared cache across workers.
- Cache key hashing library (standard `hashlib` is sufficient).

## Expected outcomes
- Lower latency for repeated requests.
- Reduced upstream token spend.
- Predictable behavior with explicit opt-in controls.

## Testing
- Unit tests for cache key stability and TTL expiry.
- Integration tests for cache hit/miss behavior.
- Regression tests to ensure streaming/tool-call requests bypass cache.

## ADR note
If approved, add an ADR describing cache eligibility rules and risk boundaries.

## Deprecation report (2026-05-21)

This item is superseded by `planned/2026-05-07_multimodal-generation-and-deterministic-inference-cache.md`,
which already scopes the same server-side caching work (and more importantly, includes the security,
tenant/auth namespace, and binary-output considerations that make caching safe).

Keep the work tracked in the broader planned item so there is one source of truth for cache keying,
eligibility rules, and endpoint wiring.
