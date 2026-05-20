# Proposed: Public prompt-cache persistence control plane or explicit de-scope

## Metadata
- Created: 2026-05-20
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: `docs/adr/0007-durable-memory-bloc-cache-binding.md`
- ADR impact: Needs a new ADR or an explicit ADR 0007 follow-up if this is promoted into a public server/runtime contract

## Context

AbstractCore now has a strong public prompt-cache story in two places:

- provider-level prompt-cache APIs for local runtimes;
- durable bloc KV artifacts for exact-prefix persistence and binding.

That is enough for Python callers and for exact durable memory workflows. It is not enough for a
clean gateway/runtime/server control-plane contract when a higher layer wants to persist or reload
one provider-owned in-process cache snapshot without bypassing Runtime or reaching into provider
internals.

This matters because Gateway cleanup now removed most direct Core bypasses. The one remaining
architectural exception is provider-private prompt-cache `save/load`.

## Current code reality

Inspected on 2026-05-20:

- `abstractcore/providers/base.py`
  - public provider methods exist:
    - `prompt_cache_save(...)`
    - `prompt_cache_load(...)`
- `abstractcore/providers/mlx_provider.py`
  - MLX overrides `prompt_cache_save(...)` / `prompt_cache_load(...)`
- `abstractcore/providers/huggingface_provider.py`
  - transformers and GGUF override `prompt_cache_save(...)` / `prompt_cache_load(...)`
- `abstractcore/core/bloc_kv.py`
  - durable bloc helpers already use the public provider methods internally
- `abstractcore/server/app.py`
  - exposes `/acore/prompt_cache/stats|capabilities|set|update|fork|clear|prepare_modules`
  - exposes `/acore/blocs/kv/manifest|ensure|load`
  - does **not** expose public `/acore/prompt_cache/save` or `/acore/prompt_cache/load`
- `abstractcore/endpoint/app.py`
  - same prompt-cache control-plane gap: no public save/load routes
- `docs/prompt-caching.md`
  - documents provider-level persistence methods
  - still says `abstractgateway` can save/load local caches
- `abstractvoice/abstractvoice/omnivoice/prompt_cache.py`
  - package-owned OmniVoice prompt-token cache only
- `abstractvoice/docs/backlog/proposed/042_capability_residency_hooks.md`
  - clone/TTS residency work only

Important distinction:

- `abstractcore` already solved **exact durable persistence** for supported local providers through
  bloc KV artifacts and `prompt_cache_binding`.
- `abstractcore` has **not** solved a generic public server/runtime contract for ad hoc
  provider-owned prompt-cache snapshots.
- `abstractvoice` does not change that conclusion. Its caches are package-owned voice-conditioning
  caches, not a cross-framework prompt-cache persistence surface for Gateway.

## Problem or opportunity

Today there is no clean public Core contract for generic prompt-cache persistence across the
runtime/server boundary.

That leaves higher layers with three bad options:

1. reach into provider-private runtime state;
2. keep a Gateway-local special case forever;
3. overuse bloc KV artifacts for workflows that are not really durable exact-memory artifacts.

The first option is the current architectural exception and should end. The second is a boundary
smell. The third conflates two different semantics:

- exact, durable, binding-aware bloc artifacts;
- ad hoc, model-locked, best-effort local runtime cache snapshots.

There is also a security/design concern: a naive HTTP `save/load` route that accepts arbitrary
filesystem paths would be the wrong abstraction.

## Concrete things higher layers should be able to do, but cannot do cleanly today

### Example 1: Save a warmed local runtime cache after a long prefill

Suppose a loaded MLX or HuggingFace runtime has already paid the cost to build a large local cache
key:

- provider: `mlx`
- model: `mlx-community/Qwen3.6-27B-4bit`
- key: `work:orbit`

At the framework level, a clean operator flow should be possible:

1. Gateway asks Runtime to save that live cache key.
2. Runtime asks Core to persist it as a Core-owned artifact.
3. Core returns an opaque artifact reference plus metadata.
4. Gateway returns that result without ever touching provider-private cache state.

What cannot be done cleanly today:

- Gateway cannot forward a public `save_prompt_cache(key="work:orbit")` call through Runtime to a
  public Core route.
- The only way to do this generically is to hold a provider instance and call
  `provider.prompt_cache_save(...)` directly, which is exactly the boundary violation we are trying
  to eliminate.

### Example 2: Reload a saved local cache into a fresh runtime after restart or eviction

Suppose a runtime was unloaded, restarted, or replaced, but an operator wants to restore a
previously saved local cache snapshot into a new runtime key:

- artifact: `artifact_abc123`
- target provider/model: same local provider/model pair
- target key: `work:orbit-restored`

The clean flow should be:

1. Gateway asks Runtime to load `artifact_abc123` into the selected loaded runtime.
2. Runtime forwards that to a public Core load route.
3. Core validates provider/model compatibility and loads the cache on the correct provider thread.
4. Core returns the new active key and artifact metadata.

What cannot be done cleanly today:

- there is no public Core HTTP/control-plane route to accept an artifact reference and perform that
  load;
- higher layers therefore cannot forward this capability without reintroducing provider-instance
  reach-through.

### Example 3: Runtime host facade cannot expose a fully clean prompt-cache persistence API

Runtime can expose host/admin facades only for surfaces that Core actually owns publicly.

Today Runtime can cleanly forward:

- `/acore/prompt_cache/stats`
- `/acore/prompt_cache/capabilities`
- `/acore/prompt_cache/set`
- `/acore/prompt_cache/update`
- `/acore/prompt_cache/fork`
- `/acore/prompt_cache/clear`
- `/acore/prompt_cache/prepare_modules`
- `/acore/blocs/kv/*`

But Runtime cannot expose a clean generic persistence contract like:

- `save_prompt_cache(runtime_id=..., key=...)`
- `load_prompt_cache(runtime_id=..., artifact=..., key=...)`

unless it either:

- invents a Runtime-owned persistence implementation that bypasses Core, or
- forces Gateway to keep the current special case.

Both are the wrong abstraction.

### Example 4: Bloc KV is not a substitute for ad hoc live cache persistence

Bloc KV already solves a different problem well:

- compile one immutable text/file memory snapshot into one exact provider/model artifact;
- load it with binding-aware correctness;
- use `prompt_cache_binding` to prove exactness at generation time.

What bloc KV does **not** cleanly model:

- “save the current live session cache after system/tools/discussion have evolved”
- “reload that exact live runtime snapshot later as operator/admin state”
- “preserve one working keyed cache as a reusable runtime warm state without claiming it is a
  durable memory source of truth”

Trying to force all generic save/load workflows through bloc KV would blur an important semantic
difference:

- exact durable memory artifact
- local admin snapshot of a live cache

That distinction should remain explicit.

## Proposed direction

AbstractCore should make an explicit product decision and own it publicly:

### Preferred direction

Add a **public artifact-backed local-admin persistence contract** for prompt-cache save/load.

That means:

- no raw caller-controlled filesystem paths in the public HTTP contract;
- Core server owns where cache artifacts are written or retrieved;
- save returns a Core-owned artifact descriptor or durable store reference;
- load accepts an artifact id/reference plus a target key/runtime selection;
- responses stay capability-gated and structured, like the existing prompt-cache control plane.

This would let Runtime and Gateway forward the feature cleanly without reaching through provider
internals.

### Acceptable alternative

If maintainers do not want to support generic cache snapshot persistence as a public server
feature, then de-scope it explicitly:

- keep `prompt_cache_save(...)` / `prompt_cache_load(...)` as in-process Python/CLI APIs;
- keep bloc KV artifacts as the only public persistent HTTP cache artifact surface;
- remove or narrow docs that imply Gateway/server-level generic cache save/load is part of the
  supported public contract.

## Why this is the cleanest framework shape

The cleanest design for the framework is:

`Flow / thin clients -> Gateway -> Runtime -> Core public control plane -> provider hooks`

not:

`Gateway -> provider instance internals`

and not:

`Runtime reimplements provider-owned persistence outside Core`

The reasons are concrete:

1. **Provider serialization already belongs to Core**
   - MLX, transformers, and GGUF each serialize different native cache formats.
   - That logic is already encapsulated behind Core provider hooks.
   - Re-implementing or bypassing that elsewhere is duplication and drift.

2. **Thread-affine local runtimes already belong to Core/Runtime control**
   - loaded local runtimes keep prompt-cache operations on the provider executor thread;
   - save/load should follow the same rule as stats/update/fork/clear;
   - a public Core control plane keeps that execution rule in the right layer.

3. **Gateway should forward opaque artifacts, not own provider cache formats**
   - Gateway should not know whether the underlying cache artifact is `.safetensors`, `.npz`, or
     some future provider-native format;
   - Gateway should pass opaque artifact references and structured results only.

4. **Thin clients should never see filesystem paths**
   - Flow, Assistant, or future operator tools should work with runtime ids, keys, and artifact
     refs;
   - they should not construct provider-specific filenames or server-local paths.

5. **The same surface can serve multiple higher layers**
   - Runtime host/admin facade
   - Gateway operator endpoints
   - CLI/admin tools
   - possible future run-authored admin steps, if explicitly desired

This is why the proposal is artifact-backed rather than path-backed.

## Candidate public contract shape

This proposal does **not** require locking the exact JSON today, but the intended shape should be
clear enough for future agents.

### Save

Example request shape:

```json
{
  "runtime_id": "abc123",
  "key": "work:orbit",
  "format_hint": null,
  "meta": {
    "label": "orbit-prefill",
    "source": "operator"
  }
}
```

Example response shape:

```json
{
  "supported": true,
  "operation": "save",
  "runtime_id": "abc123",
  "provider": "mlx",
  "model": "mlx-community/Qwen3.6-27B-4bit",
  "key": "work:orbit",
  "artifact": {
    "$artifact": "artifact_abc123"
  },
  "artifact_meta": {
    "kind": "prompt_cache_snapshot",
    "format": "abstractcore-prompt-cache/v1"
  }
}
```

### Load

Example request shape:

```json
{
  "runtime_id": "abc123",
  "artifact": {
    "$artifact": "artifact_abc123"
  },
  "key": "work:orbit-restored",
  "make_default": true
}
```

Example response shape:

```json
{
  "supported": true,
  "operation": "load",
  "runtime_id": "abc123",
  "provider": "mlx",
  "model": "mlx-community/Qwen3.6-27B-4bit",
  "key": "work:orbit-restored",
  "artifact": {
    "$artifact": "artifact_abc123"
  },
  "loaded": true
}
```

Important properties of that shape:

- artifact reference is opaque;
- provider/model validation stays in Core;
- higher layers do not pass server-local file paths;
- higher layers do not parse provider-native cache internals;
- unsupported providers still return structured `supported=false` responses.

## Clean leverage path for Runtime, Gateway, and others

If Core provides this contract, the framework can align cleanly:

### Runtime

- host facade adds `save_prompt_cache(...)` / `load_prompt_cache(...)`
- facade forwards directly to Core public control-plane routes
- no provider-instance peeking in Runtime

### Gateway

- Gateway forwards operator/admin requests to Runtime only
- Gateway no longer needs the remaining prompt-cache special case
- Gateway remains agnostic to cache serialization details

### Flow and other thin clients

- Flow should only surface this if/when the operator UX is truly needed
- if surfaced, it should operate on runtime ids, keys, and returned artifact refs
- it should never construct filenames or talk to Core directly

### Assistant / scripts / future tools

- same public surface can be consumed from CLI/admin tooling without extra one-off logic
- ledger-aware higher layers can record the returned artifact refs or load requests if they choose
  to expose this in a workflow

In other words, this proposal is not only about filling a missing route. It is about preserving a
single framework boundary where all cache persistence flows remain:

- Core-owned in semantics
- Runtime-owned in forwarding/orchestration
- Gateway-owned in HTTP exposure only
- thin-client-safe at the edges

## Why it might matter

- It removes the last known Gateway prompt-cache boundary violation without inventing a new bypass.
- It keeps Runtime as the sole execution/control-plane owner for forwarded work.
- It preserves the distinction between:
  - exact durable memory artifacts;
  - local admin cache snapshots.
- It prevents insecure public contracts based on arbitrary filenames or provider-private paths.

## Promotion criteria

Promote only when one of these becomes true:

1. Gateway must keep generic prompt-cache save/load as a supported operator feature.
2. Runtime needs a public facade for cache snapshot persistence, not only bloc artifacts.
3. Core maintainers decide that the current docs promise too much and want an explicit public
   contract or de-scope.

Before promotion, confirm which semantic family this belongs to:

- local-admin snapshotting, or
- durable exact artifacts.

Do not promote until that distinction is explicit.

## Validation ideas

- Negative design review:
  - reject any proposal that exposes raw filesystem paths in public HTTP routes.
- If promoted into a public contract:
  - route tests for save/load capability gating and structured unsupported/error payloads;
  - artifact-store round-trip tests through server/runtime, not just provider unit tests;
  - model-mismatch and stale-artifact rejection tests;
  - docs updates that separate generic snapshot persistence from bloc KV exact binding.
- If de-scoped instead:
  - remove or revise docs that imply Gateway/server generic save/load support;
  - verify higher layers no longer advertise or depend on that feature.

## Non-goals

- Do not weaken ADR 0007 exact-binding semantics for bloc KV artifacts.
- Do not expose provider-private cache-store metadata as the public contract.
- Do not treat voice-conditioning caches in `abstractvoice` as a substitute for Core prompt-cache
  persistence.
- Do not make remote-provider best-effort cache hints look like local exact artifacts.

## Guidance for future agents

Start from the boundary question, not from the existing Gateway workaround.

Re-check:

- whether Gateway still carries any prompt-cache save/load special case;
- whether Runtime already exposes a public facade that needs a Core counterpart;
- whether Core docs still claim Gateway/server save/load support.

If the answer is “generic save/load should remain public”, design it as an artifact-backed local
admin surface. If the answer is “no”, de-scope it explicitly and keep bloc KV as the only public
persistent cache contract.
