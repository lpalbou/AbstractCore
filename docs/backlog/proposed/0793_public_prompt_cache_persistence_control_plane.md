# Proposed: De-scope or opt-in local-admin prompt-cache snapshots

## Metadata
- Created: 2026-05-20
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs:
  - `docs/adr/0004-operator-control-and-server-trust-boundary.md`
  - `docs/adr/0007-durable-memory-bloc-cache-binding.md`
- ADR impact: Branch B needs a new ADR, or an explicit ADR 0007 follow-up, before any
  HTTP/server/runtime snapshot contract is accepted. Live prompt-cache snapshots must not be folded
  into ADR 0007 exact bloc binding.

## Context

AbstractCore now has two separate local cache surfaces:

- exact durable bloc artifacts for immutable text/file memory snapshots;
- in-process prompt-cache keys with provider-level `prompt_cache_save(...)` and
  `prompt_cache_load(...)` hooks.

That is enough for Python callers and exact durable memory workflows. The unresolved product
question is whether Core should expose any supported server/runtime admin surface for saving and
restoring opaque live local prompt-cache snapshots.

Important framework conclusion from the 2026-05-20 cross-repo review:

- This item is **not** the primary path for app-facing durable prompt caching through Gateway.
- The clean public durable path for apps is:
  1. persist reusable text/file memory as blocs;
  2. compile/load exact provider/model bloc artifacts through `/acore/blocs/*`;
  3. carry `prompt_cache_binding` on generation when exactness matters;
  4. let Runtime/Gateway orchestrate stable session keys and cache preparation on top.
- Live prompt-cache snapshots remain an optional operator optimization for hot local runtime state.

These snapshots are provider-native runtime state. They are not memory blocs, not exact durable
memory artifacts, and must not return `prompt_cache_binding`, `binding_id`, or any other exact
prompt proof.

## Recommended framework direction

For the framework-wide public contract, the preferred durable path should be:

- immutable/extracted content becomes a bloc;
- Core compiles or loads one provider/model cache artifact for that bloc;
- higher layers reuse that artifact through a stable cache key and optional
  `prompt_cache_binding`;
- session/workflow/app cache reuse is expressed through deterministic keys,
  module preparation, or explicit bloc loads, not opaque live snapshot files.

In other words:

- exact durable app-visible persistence should center on `/acore/blocs/kv/*`,
  `stable_cache_key`, and `prompt_cache_binding`;
- live prompt-cache snapshots, if they ever exist beyond Python/CLI, should be
  treated only as an operator/admin optimization for preserving hot local state
  across restart or eviction.

This item is therefore not the blocker for durable Gateway/app prompt caching.
The main public-contract gaps are in higher layers:

- Runtime does not yet expose a public bloc-KV facade analogous to its prompt-cache
  host facade.
- Runtime/Gateway do not yet propagate `prompt_cache_binding` through their normal
  generation surfaces, so higher layers cannot currently consume Core exact
  binding semantics cleanly.
- Gateway still carries provider-private prompt-cache save/load hacks that should
  not become the app-facing durability model.

## Current code reality

Inspected on 2026-05-20:

- `abstractcore/providers/base.py` exposes public provider methods:
  `prompt_cache_save(...)` and `prompt_cache_load(...)`.
- `abstractcore/providers/mlx_provider.py` overrides save/load for MLX native cache artifacts.
- `abstractcore/providers/huggingface_provider.py` overrides save/load for HuggingFace
  transformers and GGUF native cache artifacts.
- `abstractcore/core/bloc_kv.py` uses those provider hooks internally to build exact durable bloc
  artifacts.
- `abstractcore/server/app.py` exposes
  `/acore/prompt_cache/stats|capabilities|set|update|fork|clear|prepare_modules`.
- `abstractcore/server/app.py` exposes `/acore/blocs/kv/manifest|ensure|load`.
- `abstractcore/server/app.py` does not expose public
  `/acore/prompt_cache/snapshots/save` or `/acore/prompt_cache/snapshots/load`.
- `abstractcore/endpoint/app.py` has the same gap: prompt-cache controls exist, but no generic
  snapshot save/load routes.
- `docs/prompt-caching.md` documents provider-level persistence methods and now distinguishes them
  from public persistent HTTP bloc artifacts.
- `ArtifactStoreLike` already exists for generated artifacts, but it is not currently wired to
  prompt-cache snapshot persistence.

Important distinction:

- Core already solved exact durable persistence for supported local providers through bloc KV
  artifacts and `prompt_cache_binding`.
- Core has not solved a generic server/runtime admin contract for ad hoc provider-owned live
  prompt-cache snapshots.
- Runtime and Gateway still need their own public facades if the framework wants apps to consume
  Core's existing bloc-KV durability through Gateway without bypassing Runtime.
- Non-Core caches, including voice-conditioning caches, are out of scope for this item.

## Problem or opportunity

Today there is no clean Core contract for generic live prompt-cache snapshot persistence
across the runtime/server boundary.

That leaves higher layers with three bad options:

1. reach into provider runtime state;
2. keep a Gateway-local special case forever;
3. overuse bloc KV artifacts for workflows that are not really durable exact-memory artifacts.

The first option is a boundary violation. The second is a maintenance smell. The third conflates two
different semantics:

- exact, durable, binding-aware bloc artifacts;
- ad hoc, model-locked, best-effort local runtime snapshots.

There is also a security/design concern: a naive HTTP save/load route that accepts arbitrary
filesystem paths would be the wrong abstraction.

## What apps actually need for durable caching

When an app quits and relaunches, the clean framework-level durable behavior is:

1. the app (or its server-side host) still knows its provider/model/session/template identity;
2. durable memory text/file content still exists as blocs in Core storage;
3. provider/model-specific bloc artifacts still exist on disk and can be revalidated or rebuilt;
4. Runtime/Gateway can re-load those artifacts into stable cache keys and optionally fork them for
   session-specific live work;
5. generation calls can carry `prompt_cache_binding` when exact bloc identity must be enforced.

That workflow already matches Core's public durable model. What is still missing is mostly above
Core:

- Runtime does not yet expose a public bloc-KV facade comparable to its existing prompt-cache and
  discovery facades.
- Runtime/Gateway do not yet flow `prompt_cache_binding` through their durable LLM execution
  contract.
- Gateway session prompt-cache lifecycle currently orients around stable keys and prepared modules,
  but not yet around durable bloc selectors/artifacts as a first-class public app contract.

So the framework should not treat this proposal as a prerequisite for durable app-facing caching.
The real durable path is blocs + bindings. This proposal only matters if maintainers also want
opaque live-cache checkpoint/restore across process restart or eviction.

## Recommendation after 2026-05-20 code audit

For the concrete framework goal of **app quit/relaunch reuse through Gateway for local MLX,
HuggingFace transformers, and HuggingFace GGUF**, this item is **not** the primary missing piece.

Current code already solves most of the durable reuse problem in a cleaner way:

- immutable text/file memory can be persisted and looked up through
  `/acore/blocs/upsert_text` and `/acore/blocs/record`;
- supported local providers can compile/load durable provider-native KV artifacts through
  `/acore/blocs/kv/ensure` and `/acore/blocs/kv/load`;
- callers can require exact request-time reuse with `prompt_cache_binding`.

What is still missing for that user goal is mostly **upper-layer exposure**, not a new Core cache
snapshot primitive:

- Runtime does not yet expose a public bloc facade analogous to the current host/discovery/run
  facades;
- Gateway does not yet expose a client-facing durable bloc surface backed by Runtime;
- apps therefore cannot yet use the already-clean durable Core mechanism through Gateway alone.

That means:

- if the product goal is **durable reusable memory/prefixes for apps**, prefer bloc KV and binding;
- if the product goal is **hot runtime state restore without rebuilding the prefix**, that is where
  live snapshots may still be useful.

This distinction is critical. Bloc KV is the public durable application contract. Snapshot save/load
would be an optional operator optimization.

## Concrete use cases if Branch B is chosen

### Save a warmed local runtime cache after a long prefill

A loaded MLX, HuggingFace transformers, or HuggingFace GGUF runtime has already paid the cost to
build a large local cache key.

A clean operator flow would be:

1. Gateway or Runtime asks Core to save that live cache key for the selected runtime.
2. Core persists provider-native state through provider hooks on the correct provider thread.
3. Core returns an opaque snapshot id plus compatibility metadata.
4. Higher layers forward the opaque result without touching provider internals.

This is intentionally **not** the recommended first answer for ordinary app relaunch reuse. If the
same reusable prefix can be represented as immutable text/file memory, the cleaner path is:

1. persist/update the bloc source of truth with `/acore/blocs/upsert_text`;
2. read or resolve the durable selector with `/acore/blocs/record` when needed;
3. compile or validate the provider-native bloc artifact;
4. load it into a runtime key when needed;
5. use `prompt_cache_binding` on generation when exactness matters.

### Reload a saved local cache after restart or eviction

A local runtime was restarted, unloaded, or replaced, and an operator wants to restore a saved live
snapshot into a compatible loaded runtime under an explicit target key.

A clean operator flow would be:

1. Gateway or Runtime asks Core to load a snapshot id into a selected loaded runtime.
2. Core validates snapshot compatibility before native provider load.
3. Core performs load on the provider thread and under key-level serialization.
4. Core returns the active target key and snapshot metadata.

### Keep bloc KV exactness separate

Bloc KV already solves a different problem:

- compile one immutable text/file memory snapshot into one exact provider/model artifact;
- load it with binding-aware correctness;
- use `prompt_cache_binding` to prove exactness at generation time.

Live snapshots do not cleanly model that. They can represent evolved runtime state, but they do not
prove a durable text/file source of truth. They also do not restore transcript, tools, run ledger, or
memory state.

## Concrete examples of what snapshots would solve that blocs do not

Examples that **are already solved more cleanly by blocs**:

- "Keep a durable file/text memory chunk across app restarts and reload it later for the same
  provider/model."
- "Reuse the same immutable memory bloc for many sessions."
- "Prove that a request still references the exact durable memory artifact that was prepared
  earlier."

Examples that blocs do **not** directly solve, and where Branch B could still be useful:

- "Save the exact hot local prompt-cache state of an already-running session after a long chat
  history prefill, then restore that opaque state after the worker restarts."
- "Checkpoint a provider-native cache that reflects transient module ordering or evolved history
  not represented as durable blocs."
- "Warm-restore a long local context without reconstructing it from transcript/bloc inputs."

Those are legitimate operator/performance use cases, but they are not the same as public durable
memory artifacts.

For app relaunch through Gateway, the intended layering should therefore be:

- app persists its own session/thread identity plus any durable bloc ids or sha256s;
- Gateway exposes durable bloc routes to apps;
- Runtime forwards those requests to Core through a public bloc facade;
- Core compiles/loads the provider-native artifact and returns `prompt_cache_binding`;
- generation reuses the loaded artifact through normal request parameters.

Snapshot save/load only enters the design if rebuild latency after runtime restart is still too
expensive even when the app already has a durable bloc source of truth.

## Decision branches

### Branch A: Explicit de-scope

Keep `prompt_cache_save(...)` and `prompt_cache_load(...)` as in-process Python, CLI, or
provider-admin hooks only.

Keep `/acore/blocs/kv/*` as the only public persistent HTTP cache artifact surface.

Update docs that imply Gateway/server generic prompt-cache save/load is supported as a Core public
contract.

If Branch A is chosen, the framework can still deliver durable app-facing caching by:

- adding Runtime facades for `/acore/blocs/*`;
- forwarding `prompt_cache_binding` through Runtime/Gateway execution;
- teaching Gateway session cache workflows to reference durable bloc artifacts instead of provider
  snapshot files.

This is currently the recommended branch for the framework's app-facing contract unless and until a
separate operator need for hot snapshot restore is proven.

### Branch B: Authenticated opt-in local-admin snapshots

Add an opt-in trusted admin surface for opaque live local prompt-cache snapshots.

This branch should be evaluated only after Runtime/Gateway expose bloc KV cleanly and real operator
measurements still show unacceptable restart/rebuild cost for local MLX/transformers/GGUF runtimes.

Promotion into this branch requires:

- route names that say `snapshot`, for example `/acore/prompt_cache/snapshots/save` and
  `/acore/prompt_cache/snapshots/load`;
- authenticated local-admin exposure only;
- opt-in endpoint exposure, or no endpoint exposure unless routed through an authenticated gateway;
- no raw caller-controlled filesystem paths;
- no caller `format_hint`;
- provider-native serialization chosen behind Core/provider hooks;
- Core-owned snapshot ids and manifests;
- provider-thread execution and key-level serialization;
- explicit compatibility validation before load;
- structured unsupported/error responses;
- quota, retention, deletion, and size-limit policy;
- no `prompt_cache_binding`, `binding_id`, or exact durable memory proof.

## Candidate constraints if Branch B is promoted

This section is not implementation approval. It records constraints that a future planned item must
satisfy if maintainers choose Branch B.

### Route shape

Use snapshot-named routes, not generic save/load routes:

- `POST /acore/prompt_cache/snapshots/save`
- `POST /acore/prompt_cache/snapshots/load`

The admin payload should work with runtime selectors, keys, and opaque snapshot ids only.

It must not accept or return:

- raw server-local filesystem paths;
- `artifact_path` or `manifest_path`;
- provider-native cache filenames;
- caller-selected native formats;
- `prompt_cache_binding` or `binding_id`.

Default load behavior should avoid accidental key poisoning:

- require an explicit target key;
- default `make_default=false`;
- validate runtime/provider/model ownership before touching live state.

### Snapshot manifest

A snapshot manifest should record enough metadata to reject unsafe loads without parsing
provider-private internals:

- `snapshot_manifest_format`;
- snapshot id and server/scope id;
- provider family and resolved model identity;
- model file identity or revision when known;
- cache backend and cache implementation id;
- native artifact format and native format/schema version;
- provider package/library versions when available;
- tokenizer/chat-template/rendering identifiers when known;
- adapter, quantization, context-length, device/runtime, and GGUF runtime parameters when relevant;
- prompt-cache options that affect serialized state, including thinking/template controls and lossy
  quantization options;
- artifact hash, size, created time, and token count when available;
- source key, target key, and whether load replaces or preserves existing key state.

The manifest is compatibility metadata only. It is not an exact prompt proof and must never be
accepted as `prompt_cache_binding`.

### Storage and security

Snapshot state can encode system prompts, tools, private transcript state, file contents, and other
sensitive context. Any such admin surface must therefore define:

- authenticated admin access;
- deny-by-default route exposure;
- safe server-owned storage root or configured artifact store;
- atomic write plus manifest-last commit;
- redacted logs;
- retention and deletion behavior;
- quotas and artifact-size limits;
- corrupt-artifact and hash-mismatch rejection;
- server-scope rules for proxied Gateway-to-endpoint deployments.

### Execution model

Save/load must follow the same provider-thread rule as loaded-runtime prompt-cache operations:

- use the selected loaded runtime;
- serialize against concurrent update, generation, fork, clear, and snapshot work for the same key;
- reject unsupported providers with structured `supported=false`;
- never silently fall back to an empty cache when a snapshot load fails.

## Why this might matter

- It removes a potential Gateway prompt-cache boundary violation without inventing a new provider
  bypass.
- It keeps Runtime as the forwarding/orchestration owner for admin work.
- It preserves the distinction between exact durable memory artifacts and local admin snapshots.
- It prevents unsafe contracts based on arbitrary filenames or provider-private paths.
- It avoids using this item to solve the wrong problem: durable app-facing memory reuse already has
  a cleaner Core primitive in bloc KV artifacts.

## Promotion criteria

Keep this item in `proposed/` until maintainers choose Branch A or Branch B.

If Branch A is chosen, the next work is docs cleanup and any higher-layer de-scope required to stop
advertising generic Gateway/server prompt-cache save/load as a Core public contract.

If Branch B is chosen, promotion requires:

- one named consumer and operator workflow;
- an explicit authenticated-admin trust model;
- artifact-store or Core-owned storage ownership and retention policy;
- route names and payloads that are snapshot-specific and pathless;
- clear non-exact semantics versus bloc KV;
- manifest compatibility requirements;
- provider/model/backend/template/version mismatch behavior;
- corrupt, missing, oversized, and stale snapshot failure behavior;
- provider-thread and key-serialization validation plan;
- docs plan that separates live snapshots from exact durable bloc artifacts.

## Validation ideas

If Branch B is promoted into implementation, tests should cover:

- auth and opt-in exposure;
- no-path payload validation;
- unsupported provider response shape;
- missing source key on save;
- explicit target key on load;
- provider-thread execution;
- key-level serialization with concurrent update/generation/fork/clear;
- provider/backend/model mismatch;
- model revision or model-file mismatch where available;
- tokenizer/chat-template/rendering mismatch where available;
- native format or serializer-version mismatch;
- artifact hash mismatch;
- corrupt native artifact rejection;
- oversized artifact rejection;
- proxy/server-scope mismatch;
- snapshot load usability, not just deserialization success;
- no `prompt_cache_binding` or `binding_id` in snapshot responses.

If Branch A is chosen instead, validation should cover:

- docs no longer imply public Gateway/server generic prompt-cache save/load support;
- higher layers do not advertise or depend on that feature;
- bloc KV remains documented as the only public persistent HTTP cache artifact surface.

## Non-goals

- Do not weaken ADR 0007 exact-binding semantics for bloc KV artifacts.
- Do not return or accept `prompt_cache_binding`, `binding_id`, or exact durable memory proof for
  live snapshots.
- Do not treat restored snapshots as durable memory source of truth.
- Do not restore transcript, tools, system prompt, run ledger, or memory state from a snapshot.
- Do not expose this to thin clients or run-authored workflows by default.
- Do not expose provider-private cache metadata as the public contract.
- Do not expose raw filesystem paths or caller-selected provider artifact formats.
- Do not make snapshots portable across provider/model/backend versions unless proven explicitly.
- Do not let load silently fall back to an empty cache.
- Do not fold this into superbloc or exact-prefix recipe research.
- Do not make remote-provider best-effort cache hints look like local exact artifacts.

## Related

- `docs/adr/0004-operator-control-and-server-trust-boundary.md`
- `docs/adr/0007-durable-memory-bloc-cache-binding.md`
- `docs/prompt-caching.md`
- `docs/memory-blocs.md`
- `docs/backlog/completed/2026-05-20_unified-bloc-kv-artifact-api-and-request-binding.md`
- `docs/backlog/completed/2026-05-20_hf-transformers-bloc-kv-artifact-compiler-loader.md`
- `docs/backlog/completed/2026-05-20_hf-gguf-bloc-kv-artifact-compiler-loader.md`

## Guidance for future agents

Start from the boundary decision, not from any existing workaround.

Re-check:

- whether Gateway still carries any prompt-cache save/load special case;
- whether Runtime exposes or needs a public facade that requires a Core counterpart;
- whether Core docs still claim Gateway/server save/load support.

If generic save/load should be supported beyond Python/CLI, design it as authenticated opt-in
local-admin snapshots.
If not, de-scope it explicitly and keep bloc KV as the only public persistent HTTP cache artifact
contract.
