# Proposed: Exact-prefix cache recipes for immutable memory clusters

## Metadata
- Created: 2026-05-20
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: None
- ADR impact: Needs a new ADR if this is promoted beyond research into a public compiled-cluster
  cache contract

## Context

AbstractCore now has a credible exact-prefix prompt-cache story for narrow local artifacts:

- one exact bloc text snapshot -> one per-model MLX KV artifact
- if a higher layer materializes one immutable shelf text snapshot, it becomes one per-model exact
  prefix candidate
- one ordered `system -> tools -> discussion` prefix -> one provider-managed module chain

That makes a later cluster-level question worth preserving:

> if future memory systems repeatedly activate the same immutable groups of memories, should
> AbstractCore compile those groups as exact prefixes rather than recomputing them every time?

This is not required for current shelf-based memory work. It is a research question about whether a
larger exact-prefix unit ever becomes worth compiling.

## Current code reality

- [`abstractcore/core/bloc_kv.py`](/Users/albou/tmp/abstractframework/abstractcore/abstractcore/core/bloc_kv.py:80) currently enforces an MLX-only durable artifact path, validates one exact rendered recipe, and binds artifacts to provider/model/rendered hash metadata.
- [`abstractcore/core/bloc_kv.py`](/Users/albou/tmp/abstractframework/abstractcore/abstractcore/core/bloc_kv.py:237) compiles only one canonical bloc recipe today: one rendered attached-file-box prompt into one artifact.
- [`abstractcore/core/bloc_kv.py`](/Users/albou/tmp/abstractframework/abstractcore/abstractcore/core/bloc_kv.py:681) proves load/reload/fork semantics for exact artifacts, but only as load-one-artifact then use-one-key behavior.
- [`abstractcore/core/cached_session.py`](/Users/albou/tmp/abstractframework/abstractcore/abstractcore/core/cached_session.py:138) prepares ordered prefix caches for `system` and `tools`, then forks them into a session key.
- [`abstractcore/core/cached_session.py`](/Users/albou/tmp/abstractframework/abstractcore/abstractcore/core/cached_session.py:751) treats KV mode as context source-of-truth and sends only delta prompts after prefix prefill.
- [`abstractcore/providers/base.py`](/Users/albou/tmp/abstractframework/abstractcore/abstractcore/providers/base.py:4073) already exposes hierarchical prefix reuse, but it does so by deriving deterministic keys for an ordered module list, cloning the longest known prefix, and appending missing modules. It does not merge arbitrary persisted cache artifacts.
- There is no first-class shelf KV compiler/loader in the inspected Core code. Any future shelf
  work would first need to materialize one immutable shelf text boundary above this layer.
- [`abstractcore/providers/huggingface_provider.py`](/Users/albou/tmp/abstractframework/abstractcore/abstractcore/providers/huggingface_provider.py:340) shows provider capability asymmetry:
  - transformers: local control plane with save/load
  - GGUF: local control plane only when an exact cached prompt renderer exists
  - other GGUF chat formats: keyed only
- [`docs/prompt-caching.md`](/Users/albou/tmp/abstractframework/abstractcore/docs/prompt-caching.md:192) documents module caches and file boxes as stable prefix reuse, not arbitrary cache composition.
- [`docs/memory-blocs.md`](/Users/albou/tmp/abstractframework/abstractcore/docs/memory-blocs.md:41) documents bloc KV artifacts as optional per-model derived artifacts and explicitly points to the MLX bloc compiler/loader.
- [`docs/backlog/planned/2026-05-18_memory-bloc-mlx-kv-compiler-loader.md`](/Users/albou/tmp/abstractframework/abstractcore/docs/backlog/planned/2026-05-18_memory-bloc-mlx-kv-compiler-loader.md:52) already records the key constraint: KV caches are exact prefix artifacts, not reusable blocks that can be merged in arbitrary order.
- [`docs/backlog/planned/2026-05-18_mlx-provider-continuous-batching.md`](/Users/albou/tmp/abstractframework/abstractcore/docs/backlog/planned/2026-05-18_mlx-provider-continuous-batching.md:303) intentionally limits batched MLX v1 to keyed reuse and explicitly says not to attempt arbitrary cache composition there.

## Problem or opportunity

If future memory retrieval repeatedly selects the same immutable evidence groups, shelf-only and
single-bloc caching may stop being the right prefix size.

The danger is scope drift:

- the current codebase proves exact-prefix reuse, not general cache composability
- provider capabilities differ materially across MLX, transformers, and GGUF
- `CachedSession` ordering and transcript semantics are stricter than “load some caches and combine
  them”
- continuous batching work is intentionally avoiding broader control-plane composition in v1

So the opportunity is real, but the wrong framing would over-promise behavior the runtime does not
currently support.

## Proposed direction

Keep this item as research into **compiled exact-prefix recipes for immutable memory clusters**.

The unit of study should remain:

- one explicit cluster definition
- one deterministic rendered prompt recipe
- one backend/model-specific artifact or derived prefix key

Recommended framing order:

1. Evaluate whether a cluster-sized exact prefix is ever better than shelf-sized exact prefixes for
   real local-memory workloads.
2. If yes, define one narrow recipe at a time, for example:
   - cluster-only full-memory prefix
   - later, possibly, an explicitly compiled `system + tools + cluster` prefix
3. Treat any “recipe graph” only as recipe lineage metadata between exact compiled prefixes, not as
   permission to splice or merge independent KV artifacts.

If future work needs broader cache assembly, that should become a separate design item with a much
higher proof burden than this proposal.

## Why it might matter

- Larger immutable evidence groups may become the dominant repeated prefix in retrieval-heavy local
  memory systems.
- A cluster-level exact prefix could offer a middle ground between tiny per-memory caches and one
  giant cache for everything.
- Narrow research memory is useful here because the risk is not only performance; it is also
  accidentally promising a composability model that breaks correctness.

## Promotion criteria

Promote this item only when all of the following are true:

- real workloads or benchmarks show repeated activation of stable memory groups that shelves do not
  already cover well enough
- the exact bloc/shelf path is already stable enough to treat as baseline, not moving target
- cluster materialization rules live above Core and already define one immutable cluster boundary
- the work is narrowed to one deterministic recipe boundary, not “general composability”
- at least one target backend has an exact renderer/control-plane path capable of proving the
  recipe end to end
- load/use semantics are explicit for that backend: exact artifact or derived prefix key, exact
  next-use behavior, explicit stale detection
- interactions with ongoing batching/runtime work are defined rather than assumed

Promotion should stay MLX-first unless the related backend-parity research item proves another local
backend is strong enough to deserve equal priority.

## Validation ideas

- Benchmark the same workload with:
  - uncached construction
  - single-memory or single-bloc reuse
  - shelf-level reuse
  - one candidate cluster-level exact prefix
- Verify correctness against uncached prompt construction, not only latency.
- Require explicit recipe metadata for any candidate cluster artifact:
  - recipe id/version
  - rendered hash
  - model/provider binding
  - stable cluster membership definition or membership hash
- Reuse the same working-key safety expectation as exact bloc/shelf work:
  - loading a new cluster into the key must replace the old one or fail explicitly
- If multi-backend ambitions remain, compare MLX, transformers, and GGUF separately instead of
  assuming one answer applies to all three.

## Non-goals

- Do not read this item as approval for arbitrary KV-cache composition or merging.
- Do not claim `CachedSession` can hydrate transcript state from independent memory artifacts.
- Do not broaden this into remote-provider cache parity.
- Do not make this a prerequisite for current shelf-prefilter or bloc-only acceleration work.
- Do not assume continuous batching v1 will support advanced prompt-cache control-plane behavior.

## Related

- `docs/backlog/planned/2026-05-18_memory-bloc-mlx-kv-compiler-loader.md`
- `docs/backlog/planned/2026-05-18_mlx-provider-continuous-batching.md`
- `docs/backlog/proposed/2026-05-20_exact-bloc-and-shelf-cache-binding-for-external-clients.md`
- `docs/backlog/proposed/2026-05-20_transformers-and-gguf-prompt-cache-parity-for-exact-blocs-and-shelves.md`
- `docs/prompt-caching.md`
- `docs/memory-blocs.md`

## Guidance for future agents

Do not start from the filename word “composable.” Start from current code reality:

- exact prefixes
- ordered recipes
- explicit hashes
- explicit backend constraints

If this ever promotes, rewrite it again around one exact recipe and one target workload before
turning it into planned work.
