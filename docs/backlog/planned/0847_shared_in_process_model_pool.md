# 0847 — Shared in-process model pool: extend beyond MLX, and decide the tenancy of a shared cache store

**Status**: planned · **Priority**: P1 · **Created**: 2026-09-18
**Package**: abstractcore · **Related**: abstractgateway 0846, abstractruntime 0845

## Current code update — 2026-09-20

The historical gaps below predate the two-model native MLX scheduler/session work.
Native 27B/Flash now share loaded sessions with owner leases and isolated request
facades; consult `providers/mlx_native_session.py`, `providers/mlx_runtime.py` and
`docs/native-mlx-runtime.md` before treating item 5 below as wholly unimplemented.
Runtime's instance-aware admission repair is tracked by abstractruntime-0846.

One measured remaining boundary matters for application concurrency:
`NativeRuntime._group_key` groups by draft depth AND APC manager identity.
Different session/cache managers therefore cannot currently share a batch, even
though requests reach the scheduler concurrently. A 2026-09-20 Runtime boundary
probe saw separate keyed local requests at peak batch 1 and unkeyed requests at
peak batch 4. This is not proof of an outer Runtime serialization defect.
Any cross-key batching optimization must preserve tenant/cache isolation, ownership,
prefix correctness and eviction honesty; benchmark separate-session application
traffic, not only unkeyed calls, before claiming application-wide throughput gains.
Do not merge keys or bypass isolation simply to improve the reported batch size.
Related pressure eviction is separately scoped by planned native-inference 0852.

## Why

A provider INSTANCE is a cheap facade (generation defaults, scoped config, a tracer). The
expensive parts are the weights and the KV/prompt caches computed on them — and both were
per instance. A host that builds one provider per service, and rebuilds them whenever a
workflow is published, therefore paid for both repeatedly.

Measured on the operator's gateway, 2026-09-17: one 15 GB model resident **four** times
(physical footprint 113 GB, peak 129 GB on a 128 GB machine, ~85 GB compressed or swapped,
system swap 18.3 of 19.5 GB used), a model load per service on every publish, and every
session's prompt cache orphaned by each rebuild. Decode speed under that pressure was
14 tok/s against a 26–28 tok/s baseline.

Operator ruling, 2026-09-17: *"'One loaded model should be shared across services': yes at
the moment"*.

## What already landed (2026-09-17, uncommitted at time of writing)

`abstractcore/providers/mlx_provider.py`:

- `_SharedMLXModel` + a process-wide `weakref.WeakValueDictionary` keyed by the resolved
  load target. The first provider loads; later providers for the same model adopt the
  `llm`, the `tokenizer`, **and** the `prompt_cache_store`, `_hybrid_snapshots`,
  `_hybrid_snapshot_lock` and `_delta_feed_warned_keys` — a KV cache is only reusable with
  the weights it was computed on, so the two travel together.
- Weak values, strong references held by the providers: when the last provider for a model
  goes away, the weights are freed exactly as before.
- `unload_model()` releases this provider's hold; if other providers still use the model it
  detaches (private, empty cache state) and leaves their weights and caches alone.
- Opt-out: `create_llm(..., share_loaded_model=False)` or `ABSTRACTCORE_MLX_SHARE_MODELS=0`.

Measured on `mlx-community/Qwen3.5-4B-MLX-4bit`:

| | before | after |
|---|---|---|
| second provider, same model | 2.37 → 4.80 GB, `a.llm is b.llm` False | 2.37 → 2.50 GB, same object |
| second provider sees session cache | no | yes (`hit_restore cached_tokens=468`) |
| per-service scoped config | separate | still separate |
| all providers dropped + gc | 0.00 GB | 0.00 GB |

## The gap

1. **MLX only.** The HuggingFace lanes (transformers and GGUF) construct their own model
   per provider instance and have the same duplication and the same cache orphaning. The
   gateway reaches them through the same `create_llm` seam.
2. **Tenancy of the shared store is now implicit.** One `PromptCacheStore` per model is
   shared by every provider in the process, including providers built for different
   principals in a multi-user gateway. Keys embed the session id (and provider/model/node),
   so cross-talk needs a key collision rather than a mistake — but "principals share a cache
   store" is a decision that must be written down, and probably gated
   (per-principal namespace, or an explicit opt-in for multi-tenant processes).
3. **The bound is now shared.** `PromptCacheStore` defaults to 32 entries. Shared, those 32
   are shared by every session and node of that model: concurrent sessions can evict each
   other silently. Owned jointly with abstractruntime 0845 (fairness + eviction honesty).
4. **No visibility.** Nothing reports which providers hold a model, how many caches the
   shared store holds, or how much memory that is. `get_prompt_cache_stats()` describes one
   store; there is no process view.
5. **Sharing is untested for the awkward paths**: two providers with *different* llm_kwargs
   for the same model (who wins?), vision add-ons and MTP/speculation lanes (`_enter_mtp_lane`
   returns before the shared path), `unload_model` while another provider is mid-generation
   (the process-wide generate lock is per (provider, model) in abstractruntime, not here).

## Scope

### In scope

- Extend the pool to the HuggingFace transformers and GGUF lanes, with the same
  weights-plus-caches-travel-together rule and the same opt-out.
- Include the MTP/speculation and vision-add-on load paths, or declare them explicitly
  unshared with a reason.
- Decide and document the tenancy rule for the shared store; implement whatever the decision
  requires (namespacing, opt-in, or a refusal to share across principals).
- A process-level view: models resident, holders per model, cache entries and approximate
  bytes per model.
- Tests for: adoption, per-instance config isolation, `unload_model` with and without other
  holders, last-holder release frees memory, differing `llm_kwargs`, and a concurrency test
  that two providers generating on one model do not corrupt each other's caches.

### Out of scope

- Cross-process sharing.
- Changing the eviction policy itself (abstractruntime 0845 owns fairness/honesty).
- Any change to prefix identity or the delta-feed lanes.

## Acceptance criteria

- [ ] A second provider for the same HF/GGUF model adopts the resident one; memory does not
      double; the first provider's session cache is visible to the second.
- [ ] The tenancy decision is documented in the provider docs and enforced in code.
- [ ] `unload_model` on one holder never disturbs another holder's weights or caches, and
      the last holder's unload frees the memory (measured, not asserted).
- [ ] A process-level residency/cache view exists and is exercised by a test.
- [ ] Differing `llm_kwargs` for one model has a defined, tested behaviour.
