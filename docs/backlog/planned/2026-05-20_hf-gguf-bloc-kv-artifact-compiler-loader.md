# Planned: HuggingFace GGUF bloc KV artifact compiler and loader

## Metadata
- Created: 2026-05-20
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: Covered by the unified bloc artifact API ADR if that ADR is created

## Context

HuggingFace GGUF already has useful prompt-cache support, but it is narrower than transformers:

- keyed in-process cache support is broadly available
- local prompt-cache control-plane support exists only when AbstractCore can render the active
  llama.cpp chat format exactly
- current exact renderers are `chatml-function-calling` and `llama-3`
- unsupported chat formats must remain keyed-only

That is still enough to add durable exact bloc artifacts for supported GGUF paths, as long as the
implementation is explicit about renderer limits.

## Current code reality

- `HuggingFaceProvider` with `model_type="gguf"` always supports keyed prompt-cache selection.
- GGUF reports `mode="local_control_plane"` only when the exact cached prompt renderer is
  available for the active chat format.
- Existing GGUF provider code can save/load llama.cpp state snapshots for supported prompt-cache
  control-plane paths.
- `tests/huggingface/test_gguf_prompt_cache_control_plane.py` covers supported exact renderers,
  keyed-only downgrade for unsupported formats, prefix reuse, and save/load behavior.
- `abstractcore/core/bloc_kv.py` currently rejects non-MLX providers before it can compile or load
  a bloc artifact.
- GGUF does not currently support `CachedSession` KV source-of-truth mode and this item should not
  imply that it does.

## Problem

GGUF users with exact-renderer chat formats can reuse local prompt caches, but they cannot yet
compile one persisted bloc into one durable, versioned exact artifact through the same Python/server
API as MLX.

## What we want to do

Add a HuggingFace GGUF backend for the unified bloc KV artifact helper:

- compile one bloc's canonical attached-file-box recipe into a llama.cpp prompt-cache state
- persist that state as a durable artifact
- write a manifest with common AbstractCore fields plus GGUF-specific renderer/cache metadata
- load or fork the artifact into a runtime prompt-cache key
- expose the behavior through Python helpers, `AbstractEndpoint`, and gateway loaded runtimes
- support optional request-time binding through the unified binding API

## Requirements

- Support only GGUF chat formats with exact cached prompt renderers.
- Keep unsupported GGUF formats keyed-only and return structured unsupported errors for
  compile/load attempts.
- Use the same public helper and server route shape as MLX and transformers.
- Preserve the same load semantics as MLX:
  - stable key reload-on-miss
  - stale metadata detection
  - optional fork into a working key
  - default-key preservation unless explicitly requested
  - cleanup on failure
- Include enough manifest metadata to reject stale or incompatible artifacts:
  - provider family
  - model id and resolved model identity when available
  - chat format
  - exact renderer id/version
  - recipe id/version
  - rendered recipe hash
  - bloc content hash
  - artifact hash
  - llama.cpp/cache artifact format version
- Use provider-level public hooks for prompt-cache metadata/binding checks; do not reach into
  private provider stores from higher layers.
- Preserve loaded-runtime thread affinity for gateway-local GGUF runtimes.

## Suggested implementation

Extend `abstractcore.core.bloc_kv` with a backend adapter for supported GGUF paths:

- reuse the existing bloc resolution and rendered recipe flow
- require `provider.prompt_cache_supports_operation("save")` and `"load"`
- require the exact-renderer capability before compile/load
- create a temporary cache key
- call `provider.prompt_cache_update(...)` with the canonical bloc message
- call `provider.prompt_cache_save(...)` to persist the llama.cpp state snapshot
- validate and commit the manifest with the same manifest-last pattern as MLX
- load through `provider.prompt_cache_load(...)`
- fork through `provider.prompt_cache_fork(...)` when a separate working key is requested

## Scope

- Durable exact bloc artifacts for supported HuggingFace GGUF local runtimes.
- Python and server access through the unified bloc artifact API.
- Binding metadata integration for request-time exactness.

## Non-goals

- Do not promise support for all GGUF chat formats.
- Do not claim GGUF supports `CachedSession` KV source-of-truth mode.
- Do not implement superbloc artifacts.
- Do not add arbitrary cache composition.
- Do not change remote OpenAI-compatible llama.cpp server behavior.

## Dependencies and related tasks

- `docs/backlog/planned/2026-05-20_unified-bloc-kv-artifact-api-and-request-binding.md`
- `docs/backlog/completed/2026-05-18_memory-bloc-mlx-kv-compiler-loader.md`
- `docs/prompt-caching.md`
- `docs/memory-blocs.md`
- `tests/huggingface/test_gguf_prompt_cache_control_plane.py`

## Expected outcomes

- Supported GGUF chat formats can ensure/load exact bloc artifacts through the same API as MLX.
- Unsupported GGUF formats fail explicitly instead of pretending durable artifacts exist.
- Existing GGUF save/load capabilities become usable by external bloc-memory clients.
- Server and Python APIs expose the same behavior.

## Validation

- Unit tests with a lightweight/fake GGUF provider proving compile, manifest validation, load,
  reload-on-miss, fork, replacement, cleanup, and default-key preservation.
- Tests for supported exact renderers: `chatml-function-calling` and `llama-3`.
- Negative tests for unsupported GGUF chat formats remaining keyed-only.
- Server tests for `/acore/blocs/kv/ensure`, `/acore/blocs/kv/load`, and strict binding on a loaded
  GGUF runtime.
- Documentation examples for Python and server usage.

## Progress checklist

- [ ] Add GGUF backend selection to the unified bloc artifact helper.
- [ ] Define GGUF-specific manifest metadata.
- [ ] Gate compile/load on exact cached prompt renderer support.
- [ ] Implement compile/save/load/fork using provider control-plane operations.
- [ ] Add binding metadata propagation.
- [ ] Add Python and server tests.
- [ ] Update docs and examples.

## Guidance for the implementing agent

Treat GGUF as capability-gated, not generally equivalent to MLX or transformers. The correct user
experience for unsupported chat formats is a clear unsupported response, not a quiet best-effort
artifact.
