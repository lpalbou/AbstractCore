# 0858 — Native MLX lane: the per-call prefix snapshot lands before trailing system riders

**Status**: completed on branch `fix/prefix-cache-reuse` (2026-10-27), UNRELEASED: ships in the next AbstractCore
release · **Priority**: P1 · **Created**: 2026-10-27
**Cross-package**: AbstractRuntime 0855 (prefix-stability contract of consecutive LLM calls); root
[1060](../../../../docs/backlog/recurrent/1060_gateway_security_resilience_and_latency_standing_duty.md) (latency
budget) and [1056](../../../../docs/backlog/proposed/1056_programmable_ai_systems_workflows_as_the_central_primitive.md)
(workflows as the central primitive: every agent step is an LLM call whose prefill is the latency).

## Why
Operator report 2026-10-27: a gateway agent run on `mlx · Jundot/Qwen3.8-27B-oQ4e-mtp` showed "prefill 16,384 /
24,552 tokens" at step 10. Ledger `runtime/ledger_ce182561-….jsonl`: 10 of 10 tool-loop calls `outcome: cold`,
`apc: {exact_stores: 1, exact_hits: 0}`; the run just before it (same gateway process, model and lane, no
attachments) restored all but the new turns (`hit_restore`, cached = previous prompt − 8).

Root cause. Qwen3.8-27B is hybrid (`qwen3_5`: 3 linear-attention layers per full-attention layer), so mlx-vlm
restores only an EXACT stored prefix (`apc.py` `_checkpoint_match_len`), and the native lane stores one snapshot per
call at `len(prompt) − exact_cache_guard_tokens` (8; `apc_coordinator.py` `checkpoint_len`). The runtime re-injects
the session attachment index as a TRAILING `system` message on every call (abstractruntime `effect_handlers.py`, by
design since its backlog 0212). Rendered, that rider is the last 170 tokens of the prompt; the snapshot sat inside
it, the next call renders the rider after its new turns, and no snapshot was ever a prefix of a later prompt. The
provider payloads of the run are otherwise append-only (system prompt and tools byte-identical across 11 calls).

## What shipped
- `MLXProvider._native_rider_tail_tokens` (`providers/mlx_provider.py`): when the request's messages end in
  `system`/`developer` messages and no `prompt` follows them, the rider tail is MEASURED as `len(full) − LCP(full,
  head)` over token ids, where `head` is the same request rendered without the riders and without a generation
  prompt (same renderer, thinking and effort settings). Computed on the direct native lane only.
- `CheckpointGuardView` (`providers/mlx_native_session.py`): the call's view of the session's APC manager with
  `exact_cache_guard_tokens = manager guard + rider tail`; every other attribute and method resolves on the real
  manager (one lock, budget, snapshot store). The shared manager is never mutated.
- Telemetry: `prompt_cache.snapshot_excludes_tail_tokens`.
- Tests: `tests/providers/test_mlx_native_rider_snapshot_unit.py` (golden Qwen3.8 tokenizer + hermetic template;
  tool-loop and cross-turn restore; no-rider cases keep the default; the guard reaches a fake `mlx_vlm.generate` on a
  per-call view and does not leak into the next call; the installed mlx-vlm's `checkpoint_len` honours the view).
  Red on removal: helper returning None → 4 failed; wrap removed → 1 failed.

## Evidence (real weights: the operator ledger's first 4 provider payloads replayed through `create_llm("mlx", …,
speculation={"mode": "native_mtp"})` with one `prompt_cache_key`, main vs this branch, same machine)
| step | prompt | main: prefilled | branch: restored / prefilled | main → branch |
|---|---|---|---|---|
| 1 | 8,714 | 8,714 (cold) | 0 / 8,714 (cold) | 30 s → 27 s |
| 2 | 11,602 | 11,602 (cold) | 8,536 / 3,066 | 49 s → 11 s |
| 3 | 14,942 | 14,942 (cold) | 11,424 / 3,518 | 50 s → 13 s |
| 4 | 17,636 | 17,636 (cold) | 14,764 / 2,872 | 54 s → 11 s |
Token-level replay of all 10 steps (tokenizer only): step 10 prefills 673 of 24,564 instead of 24,564.
Cross-turn (real weights): turn 2's first call (history + answer + new user turn + rider) after turn 1's first call:
main 8,747 prefilled, 26.2 s; branch 8,536 restored / 211 prefilled, 1.3 s.

## Follow-ups (not done)
- Scheduled lane (`mlx_batching=True`, `mlx_runtime.py` worker + `NativeCacheStore`): one `BatchGenerator` and one
  manager serve every request, so a per-request guard needs a field on `NativeRequest` and a per-row checkpoint
  length in mlx-vlm's batch path (`generate/ar.py` `_apc_exact_checkpoint_len`). Until then that lane keeps the
  default placement.
- Any other end-of-prompt-snapshot backend (llama.cpp / LM Studio hybrid models) has the same exposure to a trailing
  rider; the rule to carry over is "snapshot before trailing system riders".
