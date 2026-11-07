# 0233 — Real run cancellation: abort in-flight LLM calls (abstractcore slice)

**Status**: completed · **Completed**: 2026-11-07 · **Priority**: P1 · **Created**: 2026-08-02
**Master item**: `docs/backlog/planned/0233_real_run_cancellation_abort_in_flight_llm_calls.md`
(framework root) — read it first; it carries the full evidence and the cross-package plan.

## Why this is here
`/cancel` today only writes `status=CANCELLED`; the in-flight generation runs to completion
and keeps consuming GPU/paid tokens. CONFIRMED 2026-08-02 by tracing
abstractcode-tui → gateway `_apply_run_control` → `abstractruntime/core/runtime.py:1302`,
plus LM Studio slot-span analysis (5 overlapping generation pairs, largest 211s).

## This package's slice
- Accept a cancellation token on `generate()`/`stream()` and honour it at the HTTP
  boundary (close the response/stream so the server stops generating), not only in backoff waits.
- Extend the existing `core/retry.py:319` `cancel_event` rather than adding a second mechanism.
- This is the load-bearing piece: sequence it first.

## Validation (shared)
Cancel a long local generation mid-flight; the LM Studio log must show it stopping within seconds,
no slot left busy, ledger shows `cancelled_in_flight`, tokens stop accruing, and cancelling run A
must not disturb run B.

## Completion report (2026-11-07)

- Summary: shipped. `abstractcore/providers/generation_cancel.py` holds the host-cancellable in-flight
  registry (`InflightGenerations`); providers declare cancel support (openai-compatible, ollama, HuggingFace,
  MLX) and `unload_model` cancels this instance's in-flight requests first (`huggingface_provider.py` ~980,
  ~1028). User docs: `docs/generation-cancel.md`.
- Tests: `tests/providers/test_generation_cancel_unit.py`, `test_generation_cancel_http_unit.py`,
  `test_generation_cancel_inprocess_unit.py`.
- Behavior change: a cancelled run stops consuming GPU or paid tokens instead of running to completion.
- Residual / follow-ups: non-streaming calls to the hosted OpenAI and Anthropic providers cannot be interrupted
  mid-request (the HTTP call has no cancel point until it returns); streaming calls can. Left as a note for the
  framework-root master item, which owns the cross-package plan.
- Closed by the 2026-11-07 hygiene pass after a code audit; the framework-root master item is already completed.
