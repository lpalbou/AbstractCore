# 0879 — ADR 0011 enforcement: the missing test files, the ADR status and the release numbers

> Type: tests + process
> Created: 2026-11-07
> Priority: normal
> Labels: residency, concurrency, llama.cpp, adr
> Links: [ADR 0011](../../adr/0011-shared-weights-and-concurrent-requests.md), [0870](../completed/0870_per_model_slot_scheduler_and_shared_gguf_weights.md) (completed), [0871](../planned/0871_continuous_batching_llamacpp_and_mlx_lm.md)

## Summary

`0870` shipped shared GGUF weights and the per-model slot scheduler in 2.32.0 (`providers/gguf_shared.py`,
`providers/gguf_slots.py`, `tests/providers/test_gguf_shared_weights_unit.py`,
`tests/providers/test_model_slot_scheduler_unit.py`). Three pieces ADR 0011 asks for are still missing, so the
ADR cannot honestly move from Proposed to Accepted.

## Current code reality

- Present: the two test files above; the slot pool with queued events and per-request cancel.
- Missing: `tests/providers/test_shared_weights_every_path_unit.py` (one weights object across admin pool, bob
  pool, gateway-side client and `/v1`), the two-streams-interleave / cancel-one-of-four stream test, and
  `test_slot_fit_check_unit.py` (a slot is fit-checked under ADR 0010 and `n_ctx` is never lowered to fit).
- ADR 0011 status line still reads Proposed.
- No release report records the N = 4 wall-clock figure (< 0.7 x serial) per engine.

## Scope

- Write the three tests against the shipped code; fix what they find.
- Record the per-engine numbers once (a real-weights run on the Linux card and on Apple silicon).
- Move ADR 0011 to Accepted with the backlog trace, or state in the ADR what still blocks it.

## Promotion criteria

Promote to `planned/` with the next residency or concurrency change, or before the next release wave that
touches `gguf_slots.py`.
