# Proposed: Clarify transformers and GGUF parity boundaries for exact blocs and shelves

## Metadata
- Created: 2026-05-20
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: None until this is promoted into a committed cross-provider exact-artifact contract

## Context

AbstractCore now has three distinct prompt-cache layers that are easy to conflate if a backlog item
just asks for "parity":

1. provider-level local prompt-cache control plane
2. `CachedSession` KV source-of-truth behavior
3. durable exact bloc artifacts today, and any future shelf-derived exact artifacts later

The current exact bloc/shelf workflow was productized first through MLX because MLX already had the
right persistence and loader hooks. Since then, HuggingFace transformers and parts of HuggingFace
GGUF have gained much stronger local prompt-cache capabilities than this proposal originally
assumed.

The backlog item should preserve one useful open question:

> should AbstractCore extend exact bloc/shelf artifact workflows beyond MLX?

It should not imply that transformers and GGUF are still missing the same baseline control-plane
features. They are not.

## Current code reality

Files and behavior inspected before rewriting this proposal:

- `abstractcore/providers/base.py`
  - defines the generic prompt-cache contract:
    - capability modes `none`, `keyed`, `local_control_plane`
    - control-plane operations `set`, `clear`, `update`, `fork`, `prepare_modules`, `save`,
      `load`, `stats`
  - `prompt_cache_prepare_modules(...)` is already provider-agnostic at the API level.
- `abstractcore/providers/huggingface_provider.py`
  - transformers path reports `mode="local_control_plane"` and supports:
    - `update`
    - `fork`
    - `prepare_modules`
    - `save`
    - `load`
  - transformers path also reports `prompt_cache_supports_kv_source_of_truth() is True`, so
    `CachedSession` can treat the cache as the active context source-of-truth there.
  - GGUF path always supports keyed cache selection, but `mode="local_control_plane"` only when
    AbstractCore has an exact cached prompt renderer for the active chat format.
  - current exact GGUF renderers:
    - `chatml-function-calling`
    - `llama-3`
  - other GGUF chat formats downgrade to `mode="keyed"` and do not support
    `prepare_modules` / `fork` / `update`.
  - GGUF does not currently report `prompt_cache_supports_kv_source_of_truth()`, so it does not
    have the same `CachedSession` KV semantics as MLX or transformers.
- `tests/huggingface/test_transformers_prompt_cache_control_plane_unit.py`
  - proves transformers control-plane parity for:
    - capabilities
    - module preparation
    - fork
    - update
    - save/load round-trip
- `tests/huggingface/test_gguf_prompt_cache_control_plane.py`
  - proves GGUF local control-plane support for supported exact renderers
  - proves keyed-only downgrade for unsupported chat formats
  - proves prefix reuse and save/load behavior for the supported path
- `docs/prompt-caching.md`
  - already documents that:
    - transformers have local control-plane support
    - GGUF has local control-plane support only for exact renderers
    - `CachedSession` KV mode is for MLX and transformers, not GGUF
- `abstractcore/core/bloc_kv.py`
  - explicitly calls `_require_mlx_provider(...)`
  - durable bloc-artifact compile/load/fork flows are currently MLX-only
  - artifact manifests, reload-on-miss handling, and exact rendered-recipe validation are all
    written around the MLX bloc compiler/loader path
- `tests/test_bloc_kv.py`
  - covers the durable exact bloc artifact workflow only for MLX-style providers
- `docs/backlog/planned/2026-05-18_memory-bloc-mlx-kv-compiler-loader.md`
  - the MLX-first durable artifact layer is already implemented and should be treated as the
    baseline, not reopened here
- `abstractcore/server/app.py`
  - current worktree routes loaded-runtime cache control, bloc KV ensure/load, and generation
    through a single provider worker thread (`_run_loaded_gateway_runtime(...)` /
    `_stream_loaded_gateway_runtime(...)`)
  - this loaded-runtime thread-affinity path is a prerequisite for trustworthy external
    "load then next use" semantics, especially for MLX and other mutable in-process caches

## Problem or opportunity

The old wording treats "parity" as if there is one missing feature across all local backends. That
is no longer accurate.

Today the real boundaries are:

1. local prompt-cache control-plane parity already exists for transformers
2. local prompt-cache control-plane parity exists only partially for GGUF, gated by exact
   chat-format renderers
3. durable exact bloc/shelf artifact parity does not exist outside MLX
4. external "load into key K, then trust the next request using K" binding semantics are being
   hardened at the gateway/runtime layer and should not be treated as already settled everywhere

If this proposal keeps using one undifferentiated parity label, future agents are likely to:

- duplicate already-landed transformers/GGUF control-plane work
- overclaim GGUF session parity that does not exist
- assume `bloc_kv.py` is cross-provider when it is explicitly MLX-only
- mix remote-provider concerns into a strictly local-provider artifact question

## Proposed direction

Rescope this item into a decision gate, not an implementation promise.

Use three explicit parity levels:

1. Control-plane parity
   - "Can the provider support `set` / `update` / `fork` / `prepare_modules` / `save` / `load`
     for local prompt-cache state?"
   - Current answer:
     - MLX: yes
     - transformers: yes
     - GGUF: only for supported exact renderers; otherwise keyed-only
2. Session parity
   - "Can `CachedSession` treat the prompt cache as the context source-of-truth?"
   - Current answer:
     - MLX: yes
     - transformers: yes
     - GGUF: no
3. Durable exact bloc/shelf artifact parity
   - "Can AbstractCore compile a versioned exact bloc/shelf artifact, persist it, reload or fork
     it later, and validate its rendered recipe externally?"
   - Current answer:
     - MLX: yes
     - transformers: not implemented
     - GGUF: not implemented

If work beyond MLX is ever promoted, split it into a backend-specific planned item rather than
keeping one vague "parity" task. Likely candidates would be:

- a transformers exact-bloc artifact compiler/loader item
- a GGUF exact-renderer artifact compiler/loader item
- a docs-only item if the only missing work is clearer capability language

## Why it might matter

This keeps AbstractCore honest about what already works versus what is still MLX-first:

- users get accurate expectations
- backlog work stays aligned with actual code reality
- future backend expansion can be justified by evidence instead of terminology drift
- exact memory shelf experiments do not accidentally depend on unsupported provider semantics

## Promotion criteria

Promote this proposal to `planned/` only when the scope is narrowed to one backend family and one
specific parity level, and the following evidence exists:

- the loaded-runtime thread-affinity patch in `abstractcore/server/app.py` has landed with tests
  for local cache/bloc flows, so external runtime semantics are stable enough to rely on
- a real workload needs exact bloc/shelf reuse on a non-MLX backend, or benchmark results show a
  non-MLX backend is materially competitive for the exact warm-prefix workflow
- the backend boundaries are explicit:
  - transformers model class limitations, or
  - GGUF chat-format limitations and exact renderer scope
- the follow-up can state one concrete artifact story:
  - docs-only clarification, or
  - provider-specific durable artifact compiler/loader work

Do not promote this item just because "parity sounds nice." Promotion should require evidence that
non-MLX exact artifacts are worth the extra serializer, manifest, and correctness surface area.

## Validation ideas

Before promotion, keep validation narrow and decision-oriented.

For transformers:

- verify exact prefix reuse with `prepare_modules(...)`, `fork(...)`, and `update(...)`
- verify save/load round-trips for the exact prefix under consideration
- benchmark cold prefill versus warm reuse for one realistic exact shelf workload
- explicitly record which model classes are excluded from support

For GGUF:

- run the same checks on supported exact renderers:
  - `chatml-function-calling`
  - `llama-3`
- verify unsupported chat formats remain keyed-only and are documented as such
- measure warm-request behavior both with and without the control-plane generation fast path

For any durable non-MLX artifact proposal:

- prove the provider can serialize one exact rendered file-box recipe and reload it later without
  ambiguous prompt reconstruction
- define provider-specific manifest fields and versioning before implementation work is promoted
- test stale-artifact detection, reload-on-miss behavior, and key-binding safety

For gateway/external usage:

- verify "load artifact into key K, then next request using K" on a loaded runtime
- verify no silent stale-key success when replacing one exact shelf with another

## Non-goals

- Do not reinterpret the implemented MLX bloc-KV compiler/loader as provider-agnostic.
- Do not promise GGUF `CachedSession` KV source-of-truth parity.
- Do not promise support for all HuggingFace models; transformers prompt-cache support is narrower
  than "any model loadable by HuggingFaceProvider".
- Do not mix this item with remote prompt-cache parity or observability work.
- Do not broaden this item into general cache composability or memory-cluster caching.

## Related

- `docs/backlog/planned/2026-05-18_memory-bloc-mlx-kv-compiler-loader.md`
- `docs/backlog/proposed/2026-05-06_remote-prompt-cache-session-parity.md`
- `docs/backlog/proposed/2026-05-20_exact-bloc-and-shelf-cache-binding-for-external-clients.md`
- `docs/backlog/proposed/2026-05-20_composable-prompt-cache-recipes-for-immutable-memory-clusters.md`
- `docs/prompt-caching.md`
- `docs/memory-blocs.md`

## Recent validation

Validated against current tests on 2026-05-20:

- `pytest -q tests/huggingface/test_transformers_prompt_cache_control_plane_unit.py`
- `pytest -q tests/huggingface/test_gguf_prompt_cache_control_plane.py`
- `pytest -q tests/test_bloc_kv.py`

## Guidance for future agents

Re-read the inspected code before promoting or implementing anything. If the only gap is wording,
update `docs/prompt-caching.md` or `docs/memory-blocs.md` instead of opening new backend work.

Treat "exact shelf" as an exact rendered prefix question, not a generic memory abstraction. If you
do promote beyond MLX, create one backend-specific planned item per artifact strategy so the
correctness and serializer boundaries stay explicit.
