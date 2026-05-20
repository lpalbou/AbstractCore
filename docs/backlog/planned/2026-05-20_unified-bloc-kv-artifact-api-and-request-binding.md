# Planned: Unified bloc KV artifact API and request-time binding

## Metadata
- Created: 2026-05-20
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: Needs a new ADR before closure if this becomes the durable public contract for
  exact local cache binding across providers

## Context

Prompt-cache and bloc-KV support are release surface for AbstractCore, not private experiment
helpers. Callers should be able to use durable bloc cache artifacts from both Python and the server
without knowing provider internals.

The current MLX path already proves the shape:

1. persist extracted text as one bloc
2. compile or validate one exact provider/model artifact for that bloc
3. load or fork that artifact into one runtime cache key
4. use the returned key on a later generation request

The next step is to make that shape explicit as an AbstractCore abstraction that MLX, HuggingFace
transformers, and supported HuggingFace GGUF can implement behind one Python/server surface.

## Current code reality

- `abstractcore/core/bloc_kv.py` implements durable exact bloc artifacts only for MLX today.
- `ensure_bloc_kv_artifact(...)` and `load_bloc_kv_artifact(...)` are importable from
  `abstractcore.core.bloc_kv`, but they are not exported from `abstractcore` or
  `abstractcore.core`.
- `AbstractEndpoint` and the gateway expose `/acore/blocs/kv/ensure`,
  `/acore/blocs/kv/load`, and `/acore/blocs/kv/manifest`.
- `load_bloc_kv_artifact(...)` validates artifact identity at load time and reloads a stable key
  when the key is missing or stale.
- Chat requests still accept only generic `prompt_cache_key`; they do not validate that the key is
  still bound to the artifact the caller loaded.
- `MLXProvider.generate(...)` still creates an empty cache when a supplied `prompt_cache_key` is
  missing, so `prompt_cache_key` remains best-effort by itself.
- HuggingFace transformers and supported HuggingFace GGUF already expose local prompt-cache
  control-plane operations, including save/load, but do not yet participate in the durable bloc
  artifact manifest/route/helper flow.
- `../ai-space` uses `superbloc` for grouped bloc membership. AbstractCore does not currently
  expose a superbloc artifact compiler.

## Problem

The cache system has the right primitives, but the public abstraction is incomplete:

- Python callers need a stable public import path for bloc artifact helpers.
- Server callers need the same behavior through gateway and endpoint routes.
- Providers should not leak filesystem paths, provider-private model paths, or cache-store internals
  as the public binding contract.
- `prompt_cache_key` should stay a best-effort cache handle for generic providers, but local exact
  bloc artifacts need an optional stricter use-time check.
- Backend-specific artifact formats should not fork the public API into three separate concepts.

## What we want to do

Define one small provider-neutral bloc artifact API with backend-specific implementations:

- MLX: existing `.safetensors` bloc artifact path remains the baseline.
- HuggingFace transformers: add a durable exact bloc artifact path using transformers cache
  save/load support.
- HuggingFace GGUF: add a durable exact bloc artifact path only for chat formats with exact cached
  prompt renderers.

Add opt-in request-time binding:

- `/acore/blocs/kv/load` returns `artifact.key` plus an opaque `binding_id`.
- Python `load_bloc_kv_artifact(...)` returns the same binding field.
- Generation may pass `prompt_cache_key` plus `expected_prompt_cache_binding` or equivalent.
- If the expected binding is supplied and the key is missing or rebound, generation fails with a
  structured cache-binding error.
- If the expected binding is omitted, existing best-effort `prompt_cache_key` behavior is preserved.

## Requirements

- Export the public bloc artifact helpers from the package API intentionally.
- Keep the API available from both Python and server routes.
- Keep one exact bloc recipe boundary for v1: one rendered attached-file-box prompt compiled into
  one provider/model artifact.
- Keep provider-specific serialization details behind the abstraction.
- Add a small provider hook for key metadata/binding validation instead of reading private stores
  from unrelated layers.
- Make `binding_id` an AbstractCore-owned opaque digest over stable manifest identity fields.
- Do not require callers to send artifact paths, provider-private model paths, tokenizer internals,
  or rendered prompt hashes back to chat.
- Preserve loaded-runtime thread affinity for local providers.
- Return structured errors for unsupported providers, unsupported model classes, unsupported GGUF
  chat formats, stale bindings, and missing keys.
- Document that remote providers keep best-effort `prompt_cache_key` semantics and do not support
  exact local binding.

## Suggested implementation

Use the existing MLX implementation as the behavioral baseline, but refactor toward a small
backend adapter boundary inside `abstractcore.core.bloc_kv`:

- resolve bloc record and canonical rendered recipe once
- select a backend adapter by provider capability and provider family
- compile/load/fork through provider prompt-cache control-plane operations
- persist a common manifest envelope with backend-specific metadata
- compute an opaque binding id from the manifest envelope
- store binding metadata on the loaded prompt-cache key through a provider-level public hook

Expose the same contract through:

- Python helpers from `abstractcore` and `abstractcore.core`
- `AbstractEndpoint` routes
- gateway loaded-runtime routes
- chat request/response models for optional strict binding

## Scope

- Public Python/server API for exact bloc artifact ensure/load/manifest.
- Optional strict binding for local exact bloc artifacts.
- Common manifest envelope shared by MLX, transformers, and supported GGUF.
- Backend-specific planned items for the two HuggingFace implementations.

## Non-goals

- Do not make generic `prompt_cache_key` strict by default.
- Do not promise exact binding for remote providers.
- Do not implement superbloc artifacts here.
- Do not allow arbitrary KV-cache composition or merging.
- Do not make cache artifacts the durable memory source of truth; bloc storage remains primary.
- Do not require GGUF `CachedSession` KV source-of-truth parity.

## Dependencies and related tasks

- `docs/backlog/completed/2026-05-18_memory-bloc-mlx-kv-compiler-loader.md`
- `docs/backlog/planned/2026-05-20_hf-transformers-bloc-kv-artifact-compiler-loader.md`
- `docs/backlog/planned/2026-05-20_hf-gguf-bloc-kv-artifact-compiler-loader.md`
- `docs/backlog/proposed/2026-05-20_composable-prompt-cache-recipes-for-immutable-memory-clusters.md`
- `docs/backlog/deprecated/2026-05-20_transformers-and-gguf-prompt-cache-parity-for-exact-blocs-and-superblocs.md`
- `docs/prompt-caching.md`
- `docs/memory-blocs.md`

## Expected outcomes

- External clients can use exact bloc artifacts through documented Python and server APIs.
- MLX, HuggingFace transformers, and supported HuggingFace GGUF share one public shape.
- Strict request-time binding is available when callers need correctness over best-effort fallback.
- Existing best-effort prompt-cache behavior remains available for general generation.

## Validation

- Unit tests for binding id generation and provider key metadata validation.
- Python tests proving `from abstractcore import ...` exposes the intended bloc artifact helpers.
- MLX regression tests for manifest validation, reload-on-miss, replacement, fork, cleanup, and
  strict stale-binding failure.
- Server tests for `/acore/blocs/kv/load` returning `binding_id` and chat failing when
  `expected_prompt_cache_binding` no longer matches.
- Gateway loaded-runtime tests proving strict binding checks run on the provider worker thread.
- Negative tests proving remote providers and unsupported local providers return structured
  unsupported errors instead of silently pretending to be exact.

## Progress checklist

- [ ] Define the public Python/server API names and request fields.
- [ ] Add provider-level metadata/binding validation hooks.
- [ ] Extend MLX bloc-KV results/routes with opaque binding ids.
- [ ] Export the public helpers intentionally.
- [ ] Add optional strict binding to generation paths.
- [ ] Implement the HuggingFace transformers backend item.
- [ ] Implement the supported HuggingFace GGUF backend item.
- [ ] Update docs and examples.

## Guidance for the implementing agent

Keep this small. The important distinction is:

- `prompt_cache_key` is a volatile runtime cache handle.
- `binding_id` is an optional exact-artifact proof for local bloc-derived caches.
- the bloc record remains the durable memory source of truth.

Do not design a general cache proof system, superbloc compiler, or remote cache contract here.
