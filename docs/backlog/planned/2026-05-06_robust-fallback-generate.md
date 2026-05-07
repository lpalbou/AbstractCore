# Planned: Robust Fallback Generate

## Metadata
- Created: 2026-05-06
- Status: Planned
- Completed: N/A

## Context
AbstractCore has provider-level retry logic for transient failures and several domain-specific
fallback systems, especially media/vision fallback. It does not yet have a reusable generation API
that tries a caller-provided chain of provider/model candidates until one succeeds.

A common application need is:

1. Try a preferred remote model that may require network access.
2. If the remote provider is unavailable, times out, or cannot be reached, try another provider.
3. If all remote options fail, fall back to a locally served model such as Ollama, LM Studio,
   HuggingFace GGUF, MLX, vLLM, or an OpenAI-compatible local endpoint.

This is different from retrying the same provider/model. It is controlled cross-provider
degradation.

## Current Code Reality
- `create_llm(provider, model=...)` in `abstractcore/core/factory.py` creates one provider instance.
- Provider discovery and defaults live in `abstractcore/providers/registry.py`.
- `BaseProvider` owns timeout normalization, retry, telemetry, and circuit breaker behavior for one
  provider/model in `abstractcore/providers/base.py`.
- `abstractcore/core/retry.py` retries the same operation; it does not select another provider/model.
- `GenerateResponse.metadata` can carry fallback trace details without changing provider-specific
  response bodies.
- `abstractcore/config/manager.py` has media-oriented fallback chains such as vision fallback, but no
  generic LLM generation fallback chain.
- `abstractcore/media/vision_fallback.py` is a useful traceable fallback pattern, but it is scoped to
  image captioning for text-only models.
- `README.md` and `docs/troubleshooting.md` show manual provider fallback snippets using
  `try`/`except`, but there is no packaged `robust_generate` helper or wrapper provider.
- `docs/fallbacks.md` documents provider/runtime control fallbacks and reinforces that fallback
  behavior must be explicit, not silent.

## Problem
Applications need graceful degradation across provider/model choices, but every app currently has to
write its own loop. These local loops can be fragile:
- they may catch the wrong exceptions;
- they may silently switch models without telling the user;
- they may retry invalid requests that should fail fast;
- they may lose the error trail needed for debugging;
- they may not preserve normal `generate(...)` kwargs consistently;
- they may be hard to use with `BasicSession` or higher-level processing helpers.

## What We Want To Do
Create a robust fallback generation API that accepts an ordered provider/model chain, tries each
candidate until generation succeeds, and returns the successful `GenerateResponse` with explicit
fallback trace metadata.

The first provider/model is the preferred path. Later providers/models are fallback paths. The final
fallback can be a local model for offline or degraded operation.

## Why
- Applications can keep working when a remote provider, API key, route, or network connection is
  unavailable.
- Local fallback models make offline-first and resilient deployment stories more practical.
- A shared implementation can enforce the "no silent fallback" rule consistently.
- The same API can support scripts, apps, sessions, and server-side routing later.

## Requirements
- Accept an ordered fallback chain of provider/model specs, for example:
  - `("openai", "gpt-4o-mini")`
  - `("anthropic", "claude-haiku-4-5")`
  - `("ollama", "qwen3:4b")`
  - dicts or provider/model strings split on the first slash.
- Preserve nested model ids such as `openrouter/anthropic/claude-...`.
- Try candidates in order and stop at the first successful non-streaming generation.
- Support async and sync usage:
  - `arobust_generate(...)`
  - `robust_generate(...)`
  - possibly a wrapper class that implements `generate(...)` and `agenerate(...)`.
- Reuse normal `generate(...)` kwargs such as `messages`, `system_prompt`, `tools`, `media`,
  `temperature`, `seed`, `thinking`, `timeout`, and `response_model` where supported.
- Make fallback visible:
  - log when moving from one candidate to the next;
  - attach `metadata["fallback"]` or equivalent to the successful response;
  - include attempted provider/model list, selected index, failure types/messages, and latencies.
- Provide an option to surface a user-facing warning or event when fallback changes provider/model
  behavior in application contexts.
- Use a clear default error policy:
  - retry/fallback on provider API errors, timeouts, connectivity errors, rate limits, model loading
    failures, and provider initialization failures;
  - fail fast by default on invalid request shape, unsupported feature, and authentication/config
    errors unless the caller opts into broader fallback.
- Allow callers to override fallback policy with an exception predicate or named policy.
- Do not swallow all failures. If every candidate fails, raise a structured exception that includes
  the full bounded attempt trace.
- Avoid repeated provider initialization when a wrapper object is reused.
- Ensure fallback is not confused with provider-side retry. Retries for one candidate should happen
  inside that provider according to its normal `RetryManager`; fallback starts after that candidate
  fails.

## Suggested Implementation
1. Create a small shared provider/model spec parser, ideally shared with consensus generation.
2. Add dataclasses in a core module, for example `abstractcore/core/fallback_generate.py`:
   - `FallbackAttempt`
   - `FallbackChain`
   - `FallbackPolicy`
   - `FallbackGenerateError`
3. Implement a lightweight `FallbackGenerator` or `RobustGenerator` wrapper that can cache provider
   instances and implement the `generate(...)` / `agenerate(...)` surface.
4. Add convenience functions:
   - `robust_generate(prompt, chain=[...], **kwargs)`
   - `arobust_generate(prompt, chain=[...], **kwargs)`
5. Normalize failure classification near the wrapper rather than modifying every provider.
6. Attach fallback trace metadata to the returned `GenerateResponse`:
   - `used_fallback: bool`
   - `selected_provider`
   - `selected_model`
   - `selected_index`
   - `attempts`
   - `policy`
7. Add optional integration with centralized config after the base API is stable, for example an
   app-level default fallback chain.
8. Document the remote-to-local use case in `docs/api.md`, `docs/getting-started.md`, or
   `docs/troubleshooting.md`.

## Scope
- Text generation fallback for non-streaming calls.
- Provider/model spec normalization shared with consensus generation.
- Explicit attempt tracing and final error reporting.
- Unit tests with fake providers and fake initialization failures.
- Documentation examples showing cloud-first and local-final fallback chains.

## Non-Goals
- Do not silently alter model choice without metadata/logs.
- Do not replace provider-level retry or circuit breakers.
- Do not guarantee semantic equivalence between primary and fallback models.
- Do not implement load balancing or cost-based routing in this item.
- Do not add a global fallback chain that affects all calls before the explicit API is proven.
- Do not support streaming fallback in v1 unless there is a clear design for partial stream failure.

## Dependencies And Related Tasks
- `abstractcore/core/factory.py`
- `abstractcore/providers/registry.py`
- `abstractcore/core/interface.py`
- `abstractcore/core/types.py`
- `abstractcore/providers/base.py`
- `abstractcore/core/retry.py`
- `abstractcore/config/manager.py`
- `abstractcore/media/vision_fallback.py`
- `docs/fallbacks.md`
- Related future item: consensus generation should share provider/model spec parsing.

## Expected Outcomes
- Users can define a robust fallback chain such as remote-first, local-final generation.
- The first successful provider/model response is returned through the normal `GenerateResponse`
  shape.
- The response metadata clearly shows whether fallback happened and why.
- All-failed chains produce actionable errors with enough detail to debug provider availability.
- Higher-level apps can reuse one implementation instead of hand-written fallback loops.

## Validation
- Unit tests for provider/model spec normalization, including nested model ids.
- Unit tests proving first-candidate success does not call later candidates.
- Unit tests proving transient failure falls through to the next candidate and records metadata.
- Unit tests proving invalid request/auth errors fail fast under the default policy.
- Unit tests proving policy override can broaden or narrow fallback conditions.
- Unit tests proving all-failed chains raise `FallbackGenerateError` with bounded attempt traces.
- Async tests proving `arobust_generate(...)` follows the same policy.
- Documentation examples checked for import paths and metadata fields.

## Progress Checklist
- [ ] Define public API shape and fallback policy defaults.
- [ ] Implement shared provider/model spec parsing.
- [ ] Implement fallback attempt tracing.
- [ ] Implement sync robust generation.
- [ ] Implement async robust generation.
- [ ] Add wrapper-provider support if needed for `BasicSession`.
- [ ] Add failure classification and all-failed error type.
- [ ] Add tests.
- [ ] Update docs and examples.

## Guidance For The Implementing Agent
Reassess current retry, provider, config, and media fallback code before implementation. Treat
fallback visibility as part of correctness. Keep the first version explicit and local to the helper
or wrapper API before considering global config-driven fallback behavior.
