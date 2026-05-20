# Planned: HuggingFace transformers bloc KV artifact compiler and loader

## Metadata
- Created: 2026-05-20
- Status: Completed
- Completed: 2026-05-20

## ADR status
- Governing ADRs: `docs/adr/0007-durable-memory-bloc-cache-binding.md`
- ADR impact: Covered by ADR 0007.

## Context

HuggingFace transformers already has the prompt-cache primitives needed for durable exact bloc
artifacts:

- local prompt-cache control-plane support
- `prompt_cache_update(...)`
- `prompt_cache_fork(...)`
- `prompt_cache_prepare_modules(...)`
- `prompt_cache_save(...)`
- `prompt_cache_load(...)`
- KV source-of-truth support for `CachedSession`

The missing work is to wire those primitives into the same durable bloc artifact compiler/loader
contract currently implemented for MLX.

## Current code reality

- `HuggingFaceProvider` with `model_type="transformers"` reports
  `PromptCacheCapabilities(mode="local_control_plane")`.
- Transformers cache persistence already writes/reads `.safetensors` through provider
  `prompt_cache_save(...)` and `prompt_cache_load(...)`.
- `tests/huggingface/test_transformers_prompt_cache_control_plane_unit.py` covers capabilities,
  module preparation, fork, update, and save/load round-trip.
- `abstractcore/core/bloc_kv.py` currently rejects non-MLX providers before it can compile or load
  a bloc artifact.
- The existing MLX manifest and reload/fork semantics are the behavioral baseline.
- The unified API and request-time binding work is tracked in
  `docs/backlog/completed/2026-05-20_unified-bloc-kv-artifact-api-and-request-binding.md`.

## Problem

Users running local transformers models can reuse in-process prompt caches, but they cannot yet
compile one persisted bloc into one durable, versioned, exact provider/model artifact through the
same Python/server API as MLX.

## What we want to do

Add a HuggingFace transformers backend for the unified bloc KV artifact helper:

- compile one bloc's canonical attached-file-box recipe into a transformers KV cache
- persist that cache as a durable artifact
- write a manifest with common AbstractCore fields plus transformers-specific metadata
- load or fork the artifact into a runtime prompt-cache key
- expose the behavior through Python helpers, `AbstractEndpoint`, and gateway loaded runtimes
- support optional request-time binding through the unified binding API

## Requirements

- Support only transformers model classes where the provider already supports prompt-cache
  save/load safely.
- Use the same public helper and server route shape as MLX.
- Preserve the same load semantics as MLX:
  - stable key reload-on-miss
  - stale metadata detection
  - optional fork into a working key
  - default-key preservation unless explicitly requested
  - cleanup on failure
- Include enough manifest metadata to reject stale or incompatible artifacts:
  - provider family
  - model id and resolved model identity when available
  - tokenizer or cache serializer version data needed for safe reload
  - recipe id/version
  - rendered recipe hash
  - bloc content hash
  - artifact hash
  - backend artifact format version
- Use provider-level public hooks for prompt-cache metadata/binding checks; do not reach into
  private provider stores from higher layers.
- Return structured unsupported errors for transformers models without compatible cache support.

## Suggested implementation

Extend `abstractcore.core.bloc_kv` with a backend adapter for HuggingFace transformers:

- reuse the existing bloc resolution and rendered recipe flow
- create a temporary cache key
- call `provider.prompt_cache_update(...)` with the canonical bloc message
- call `provider.prompt_cache_save(...)` to persist the artifact
- validate and commit the manifest with the same manifest-last pattern as MLX
- load through `provider.prompt_cache_load(...)`
- fork through `provider.prompt_cache_fork(...)` when a separate working key is requested

## Scope

- Durable exact bloc artifacts for HuggingFace transformers local text-generation models.
- Python and server access through the unified bloc artifact API.
- Binding metadata integration for request-time exactness.

## Non-goals

- Do not support vision/custom transformer backends unless the provider advertises safe prompt-cache
  save/load for that path.
- Do not implement superbloc artifacts.
- Do not add arbitrary cache composition.
- Do not duplicate `CachedSession`; this is a persisted bloc artifact flow, not a session
  transcript bootstrap.
- Do not change remote HuggingFace-compatible server behavior.

## Dependencies and related tasks

- `docs/backlog/completed/2026-05-20_unified-bloc-kv-artifact-api-and-request-binding.md`
- `docs/backlog/completed/2026-05-18_memory-bloc-mlx-kv-compiler-loader.md`
- `docs/prompt-caching.md`
- `docs/memory-blocs.md`
- `tests/huggingface/test_transformers_prompt_cache_control_plane_unit.py`

## Expected outcomes

- A local transformers provider can ensure/load an exact bloc artifact through the same API as MLX.
- Existing transformers prompt-cache control-plane support becomes durable across process runs.
- The server and Python APIs expose the same behavior.
- Unsupported transformer models fail explicitly.

## Validation

- Unit tests with a lightweight/fake transformers provider proving compile, manifest validation,
  load, reload-on-miss, fork, replacement, cleanup, and default-key preservation.
- Integration-style tests extending existing HuggingFace prompt-cache control-plane coverage.
- Server tests for `/acore/blocs/kv/ensure`, `/acore/blocs/kv/load`, and strict binding on a loaded
  transformers runtime.
- Negative tests for unsupported transformers model classes.
- Documentation examples for Python and server usage.

## Progress checklist

- [x] Add transformers backend selection to the unified bloc artifact helper.
- [x] Define transformers-specific manifest metadata.
- [x] Implement compile/save/load/fork using provider control-plane operations.
- [x] Add binding metadata propagation.
- [x] Add Python and server tests.
- [x] Update docs and examples.

## Guidance for the implementing agent

Do not special-case a second public API for transformers. If the implementation cannot use the
same helper/route names as MLX, fix the shared abstraction first.

## Completion report

Completed: 2026-05-20.

Summary:
- Implemented HuggingFace transformers durable bloc artifacts through the unified
  `ensure_bloc_kv_artifact(...)` / `load_bloc_kv_artifact(...)` flow.
- Added provider render metadata for `hf-transformers`, provider artifact format
  `abstractcore-transformers-prompt-cache/v1`, and `.safetensors` artifact selection.
- Preserved provider-owned transformers cache save/load behavior and carried bloc metadata through
  safetensors metadata without exposing private provider internals.
- Added shared contract tests that cover transformers backend manifests, debug payloads, binding
  validation, load/reload/fork behavior, and request-time binding.

Validation:
- `pytest -q` -> `1409 passed, 243 skipped`.
- Focused: `pytest -q tests/test_bloc_kv.py tests/huggingface/test_transformers_prompt_cache_control_plane_unit.py`.
- Real-provider smoke proof with `sshleifer/tiny-gpt2` on CPU: local-control-plane capability,
  `.safetensors` artifact, backend `hf-transformers`, 4,668-byte artifact, binding validation, and
  generation with `prompt_cache_binding` all succeeded.

Residual risks:
- Only standard text-generation transformer models with provider-advertised prompt-cache
  save/load are supported. Vision/custom transformer paths remain out of scope.
- Real model proof depends on a locally cached model and is intentionally not default CI.
