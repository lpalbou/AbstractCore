# Hybrid-model KV rewind and fork — checkpoint ring so /rewind and forking survive recurrent state

## Metadata
- Created: 2026-08-27
- Status: PLANNED (operator mandate, 2026-08-27: "we would like to create a /rewind at
  some point or forking. so it would be nice to find a way to make it work" — raised
  while validating Qwen3.8-Flash-Next on the GGUF lane)
- Area: providers/huggingface_provider.py (GGUF prompt-cache state ring),
  providers/mlx_provider.py (same class of hybrid caches), core/bloc_kv.py
  (durable checkpoint artifacts), providers/base.py (prompt-cache control plane)

## ADR status
- Governing ADRs: `docs/adr/0001-engineering-guardrails-and-no-silent-degradation.md`
  (a rewind that silently reuses state PAST the edit point is a silently-wrong cache;
  full re-prefill is the honest fallback and must remain the default when no
  checkpoint covers the edit).
- ADR impact: none expected while the work stays inside the existing prompt-cache
  control plane; a new ADR only if checkpoint artifacts become durable cross-process
  contracts (then it folds into the 0817 validity-key family).

## Context
Hybrid linear-attention models cannot rewind their recurrent state. Verified
2026-08-27 on Qwen3.8-Flash-Next (GGUF arch `qwen4exp`, Gated DeltaNet + sparse
attention, llama.cpp PR #27742 build):

- `llama_memory_hybrid_idx::seq_rm` refuses partial ranges (the recurrent cache is
  asked first and can veto; `src/llama-memory-hybrid-idx.cpp:142` in the PR tree), so
  llama-cpp-python's built-in longest-prefix reuse silently degrades to FULL
  re-prefill: an identical 4,078-token prompt re-evaluated at cold speed (4.27s vs
  4.17s cold). Correctness is preserved; the speedup is simply gone.
- abstractcore's keyed snapshot cache (`generate(..., prompt_cache_key=...)`) works
  because it restores a full state snapshot and appends: turn-2 of the same
  transcript prefilled in 0.60s (7.4x). But it keeps only the LATEST state per key,
  so it is append-only: any edit before the tip — exactly what a `/rewind` or a fork
  from an earlier turn produces — falls back to re-prefill from zero.
- Measured state size for qwen4exp: ~256 MB per snapshot (fixed-size recurrent state
  dominates; the 12 full-attention layers' KV is small at 2 KV heads).

The same limitation class applies to every hybrid/SSM family already in the registry
(qwen35, qwen35moe, qwen3next, qwen3_5/3_6/3_8 GGUFs, MLX-side Qwen3.5+ hybrids).

## Goal
`/rewind` to an earlier message and forking a conversation from a mid-point should
cost "re-prefill from the nearest checkpoint", not "re-prefill from zero" — on hybrid
models, on the local GGUF and MLX lanes.

## Mechanisms to evaluate (in order)
1. **Checkpoint ring per cache key (provider-level, no upstream dependency).** The
   GGUF lane already snapshots full llama.cpp states at message boundaries (the
   boundary-snapshot machinery pinned by
   `tests/providers/test_gguf_boundary_snapshot_unit.py`) and the control plane
   already declares `supports_fork`. Keep the last N boundary snapshots per key
   (ring, budgeted in bytes — at ~256 MB/snapshot a 4-deep ring is ~1 GB) instead of
   only the tip. Rewind/fork = restore the newest checkpoint at-or-before the edit
   point, re-eval only the delta. Eviction policy and a byte budget knob are part of
   the design, not an afterthought.
2. **Native rollback: `llama_context_params.n_rs_seq`.** The PR-era llama.cpp adds an
   experimental per-sequence recurrent-state snapshot count for rollback
   ("`n_rs_seq` — number of recurrent-state snapshots per seq for rollback, 0 = no
   rollback"). Exposing it through llama-cpp-python (one ctypes field + one kwarg)
   could make partial `seq_rm` succeed natively, restoring the implicit prefix-reuse
   path too. Verify semantics + memory cost before preferring it over (1); note the
   llama-server analog is context checkpoints (`--ctx-checkpoints`, upstream PR
   #15293) — same idea, server-side.
3. **Fork semantics audit.** `supports_fork=True` is already declared on the GGUF
   lane; verify fork of a hybrid state is a full deep copy (llama state bytes), pin
   it with a divergence test (two forks continue differently and both stay correct).
4. **Durable checkpoints (bloc_kv).** Only after (1): persisting ring members as
   KV artifacts inherits the full 0817 validity-key requirements (engine fingerprint
   matters MORE here — recurrent state layout is engine-version-sensitive).

## Acceptance
- On a hybrid model: edit/rewind to turn k of an n-turn transcript re-prefills only
  from the nearest checkpoint ≤ k (measured, not inferred: prefill token count in
  usage), never from zero when a covering checkpoint exists.
- Fork from turn k yields two independently-continuing sessions; both answer
  correctly; neither mutates the other's state.
- Honesty guard: when no checkpoint covers the edit point, the lane re-prefills from
  zero and says so (stats/telemetry), never reuses state past the edit (ADR-0001).
- Ring memory stays within its declared byte budget under a many-turn soak.

## Non-goals
- Token-level rewind granularity (checkpoints are message-boundary granularity by
  design; recurrent state cannot interpolate between checkpoints).
- Server-lane (`llama-server`) checkpoint tuning — that is upstream's
  `--ctx-checkpoints`; this item is the in-process lanes.
