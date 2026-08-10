# 0233 — Real run cancellation: abort in-flight LLM calls (abstractcore slice)

**Status**: planned · **Priority**: P1 · **Created**: 2026-08-02
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
