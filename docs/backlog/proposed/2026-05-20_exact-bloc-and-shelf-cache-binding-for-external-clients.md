# Proposed: Exact bloc cache binding contract for external clients

## Metadata
- Created: 2026-05-20
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: Needs a new ADR if this is promoted into a committed public request-time
  cache-binding contract

## Context

AbstractCore now has a real MLX bloc-KV artifact flow:

1. persist extracted text as a bloc
2. compile or validate one durable MLX artifact for that bloc
3. load or fork that artifact into a cache key
4. reuse the returned key on the next chat call

This is already exposed in the gateway and endpoint control plane, and `docs/memory-blocs.md`
explicitly tells callers to use the returned `artifact.key` as `prompt_cache_key` on the next chat
request.

At the same time, `docs/prompt-caching.md` still describes prompt caching as best-effort in
general, and treats local KV state as a source of truth only when the caller keeps cache state and
conversation state aligned deliberately.

That leaves one narrow design question worth preserving:

**for external clients using MLX bloc-derived cache keys, is load-time exactness enough, or do we
eventually need an opt-in request-time binding contract too?**

## Current code reality

- `ensure_bloc_kv_artifact(...)` compiles and validates exact MLX bloc artifacts using the rendered
  prompt recipe, provider/model identity, `path_in_prompt`, content hash, and artifact hash.
- `load_bloc_kv_artifact(...)` reloads a `stable_cache_key` when the key is missing or its stored
  metadata no longer matches the expected manifest, then optionally forks into a working key.
- `load_bloc_kv_artifact(...)` preserves the caller's default cache key and cleans the working key
  on failure.
- `MLXProvider.prompt_cache_load(...)` replaces the in-process cache entry for the target key; it
  is not an append operation.
- `MLXProvider.generate(...)` still treats `prompt_cache_key` as best-effort: if the key is
  missing, it can create an empty cache and continue instead of failing.
- Pending gateway changes in `abstractcore/server/app.py` serialize bloc-KV operations,
  prompt-cache control-plane calls, and chat generation onto one stable provider worker thread for a
  loaded runtime. That reduces MLX thread-affinity hazards but does not add request-time cache
  identity checks.
- `tests/test_bloc_kv.py` already covers manifest rebuilds, reload-on-miss, reload-on-artifact
  drift, default-key preservation, and cleanup on failure.
- `tests/server/test_server_loaded_runtime_control_plane.py` already covers gateway-local
  `/acore/blocs/kv/*` reuse and the loaded-runtime threading model.
- There is no first-class shelf KV compiler/loader in the inspected code or docs. Current landed
  durable artifact support is bloc-only.

## Problem or opportunity

External clients can already ensure and load an exact bloc-derived artifact, but the public chat
path only accepts a generic `prompt_cache_key`.

That means the current contract is stronger at load time than at use time:

- load-time artifact identity is validated
- stable-key reload-on-miss is implemented
- working-key cleanup on load/fork failure is implemented
- request-time key identity is not re-validated
- a missing key can still degrade to a new empty cache instead of failing explicitly

The current proposal text also mixes in shelf guarantees that are not yet grounded in the landed
code.

## Proposed direction

Keep this item narrow and MLX-first:

- document the honest current contract for bloc-derived cache keys
- if stronger safety becomes necessary, add an opt-in request-time binding check for bloc-derived
  keys on local runtimes or gateway-loaded runtimes
- extend the same idea to shelves only after a first-class shelf artifact recipe exists

One possible later shape:

- `/acore/blocs/kv/load` returns the working key plus a stable binding identifier
- a later chat call can supply that expected binding alongside `prompt_cache_key`
- if the key is missing or no longer matches the expected loaded artifact, the request fails with a
  structured cache-binding error instead of silently continuing

This keeps the request external and black-box testable without pretending general cache
composability is solved.

## Why it might matter

- External clients could trust "load then ask" flows without reading provider-private state.
- The contract would stay aligned with current AbstractCore philosophy: exact prefix artifacts are
  derived, model-locked, and provider-local, while stricter use-time guarantees remain explicit and
  opt-in.
- It would avoid overloading generic `prompt_cache_key` semantics for every provider and every
  cache mode.

## Promotion criteria

Promote this to `planned/` only when all of the following are true:

- an external client workflow depends on `/acore/blocs/kv/load` plus later `prompt_cache_key`
  reuse as a committed public contract, not just an internal experiment
- the current load-time validation in `abstractcore/core/bloc_kv.py` is insufficient because
  request-time false positives or silent empty-cache fallback create real risk
- the first implementation scope is explicitly limited to MLX bloc-derived artifacts on local
  runtimes or gateway-loaded runtimes
- if shelf language remains in scope, a first-class shelf artifact recipe or shelf KV helper exists
  in code first

## Validation ideas

- Add a black-box test that loads two different bloc artifacts into the same destination key and
  proves the second load replaces the first.
- Add an end-to-end failure test for a strict binding mode: clear or rebind the key after
  `/acore/blocs/kv/load`, then verify the next generation request fails with a structured
  cache-binding error instead of creating a blank cache.
- Keep current validation for manifest drift, reload-on-miss, default-key preservation, and cleanup
  on failure.
- Keep gateway-local threading validation so cache control and generation stay serialized on the
  same loaded-runtime provider worker.

## Non-goals

- Do not treat this as a request for general `prompt_cache_key` correctness across all providers or
  remote backends.
- Do not broaden this into prompt-cache composability, transcript bootstrap, or grouped-memory
  research.
- Do not claim shelf KV artifacts are already supported today.
- Do not make cache artifacts the durable memory source of truth; bloc storage remains primary.

## Related backlog items

- `docs/backlog/planned/2026-05-18_memory-bloc-mlx-kv-compiler-loader.md`
- `docs/backlog/proposed/2026-05-20_composable-prompt-cache-recipes-for-immutable-memory-clusters.md`
- `docs/backlog/proposed/2026-05-20_transformers-and-gguf-prompt-cache-parity-for-exact-blocs-and-shelves.md`
- `docs/backlog/proposed/2026-05-06_remote-prompt-cache-session-parity.md`

## Guidance for future agents

Re-check `MLXProvider.generate(...)` before promoting this item. If a missing `prompt_cache_key`
still auto-creates an empty cache, this remains a proposal for a stricter optional contract, not a
description of behavior that already exists.

If shelf KV work lands later, decide whether it should inherit this item's binding rules or move to
its own proposal rather than silently broadening the scope again.
