# 0870 — Per-model slot scheduler and shared GGUF weights (ADR 0011, this wave)

> Type: feature + tests
> Created: 2026-10-29
> Priority: P0
> Labels: residency, concurrency, llama.cpp, mlx, scheduler, memory
> Links: [ADR 0011](../../adr/0011-shared-weights-and-concurrent-requests.md), [ADR 0010](../../adr/0010-model-residency-fit-and-eviction.md), [0847](0847_shared_in_process_model_pool.md) (the MLX sharing this generalises), [0871](0871_continuous_batching_llamacpp_and_mlx_lm.md) (the later half), gateway ADR 0002, root backlog 1064

## Summary

Implement ADR 0011's decision in Core: one weights object per (provider, artifact, load parameters)
per process for every request path, and a per-model slot scheduler so that concurrent requests on
the shared weights run in parallel up to a bound, queue visibly past it, and cancel one at a time.
This item is the tracking record for the wave's three fixes (R1: shared llama.cpp weights with a slot
pool; R2 in the runtime/gateway: gateway-side generations borrow the principal's pool; R3: one
`EmbeddingManager` per (provider, model), PR #28 for `/v1/embeddings`) and the test files the ADR's
Enforcement section names.

## Why

The 2026-10-29 audit (gateway `untracked/residency-probe/`): with GGUF every holder (admin pool, bob
pool, gateway-side client) loads its own copy (11.8 → 23.3 → 34.9 GB RSS for a 2.7 GB file; the model
host child 12 → 24 GB after one sandbox call; two builds again after a restart), while MLX shares one
weights object. Sharing the instance as it is would serialise every request on that model on the
instance's one worker (`_InstanceGenerationWorker`). The operator requires both: one pool of resources
AND concurrency.

## Scope

### In scope (this wave)

1. `abstractcore/providers/gguf_shared.py` (R1, branch `feat/gguf-shared-weights`): a process-wide
   registry of `LlamaModel`s keyed by resolved path + `n_gpu_layers` + `use_mmap` + `use_mlock`; weak
   holders; freed with the last holder; borrowed contexts never close the model.
2. `abstractcore/providers/model_scheduler.py`: the slot pool per (weights key, `n_ctx`): the
   run/stream interface of `_InstanceGenerationWorker`, one worker thread per slot, lazy slot allocation
   through the ADR 0010 fit arithmetic (one slot's KV + compute; weights resident; no eviction; the
   refusal becomes the queued state's `slot_note`), `max_concurrent` per engine default and per model
   setting (`config.model_concurrency`, exact-or-refused), bounded FIFO queue (`max_queue` 32,
   `ModelQueueFullError`), `queue_timeout_s` 120 (`ModelQueueTimeoutError`), per-request cancel from the
   queue and from a slot, `slot_idle_ttl_s` 300, `stats()`.
3. The `queued` phase in `providers/generation_progress.py` (`queue_position`, `queue_ahead`, `slots`,
   `slots_max`, `slot_note`), response metadata `execution`, and the `/v1` header + SSE comment lines in
   `server/app.py`; the pool fields on residency rows and claims (`process_residency`).
4. The mlx-lm lane and the transformers lane attach to the same scheduler with one slot (their two
   requests never overlap on the weights). The native lane exposes its `NativeRuntime.stats()` and
   errors through the same surface.
5. Tests (ADR 0011 Enforcement; red without the fix):
   - `tests/providers/test_shared_weights_every_path_unit.py` (a)
   - `tests/providers/test_gguf_shared_weights_unit.py` (R1: one `LlamaModel` across N providers, same
     pointer, freed with the last holder, a borrowed context's `close()` does not free the model)
   - `tests/providers/test_model_slot_scheduler_unit.py` (b: wall-clock with N = 4 and N = bound + 2,
     queue bound, timeout, admission order; the contextvar that binds `self.llm` to the borrowed slot
     resolves to THAT slot inside the worker for two concurrent calls)
   - `tests/providers/test_model_slot_scheduler_stream_unit.py` (c: interleaving; d: cancel one of N,
     cancel a queued request)
   - `tests/providers/test_slot_fit_check_unit.py` (e: one slot's bytes, no eviction, refusal in words,
     a person's concurrency refused never lowered)
   - real weights, opt-in `ABSTRACT_REAL_WEIGHTS=1`: `tests/providers/test_gguf_shared_weights_real.py`,
     `tests/providers/test_gguf_concurrency_real.py` (Qwen3.5-4B-GGUF Q4_K_M, Metal + CUDA: N = 4,
     one pointer, wall-clock < 0.7 × serial, RSS/GPU sampled every 0.5 s flat within the slots' measured
     bytes + 256 MB), `tests/providers/test_mlx_lm_concurrency_real.py` (Qwen1.5-0.5B-4bit: N = 4
     serial, no crash, `queued` events seen, wall-clock ≈ serial).

### Out of scope (0871)

Continuous batching inside one context (llama.cpp `n_seq_max` / `kv_unified` through the low-level
batch API; the mlx-lm lane on `BatchGenerator`); transformers' continuous batching.

## Acceptance criteria

- [ ] One weights copy per model per process across admin pool, bob pool, gateway-side client, `/v1`,
      embeddings, before and after a model-host restart (test a green; the audit probe re-run shows
      `builds_of_model_total == 1` per child generation).
- [ ] N = 4 concurrent GGUF requests on one shared model: one pointer, wall-clock < 0.7 × serial on
      Metal and CUDA, RSS/GPU flat within the fit check (tests b, e).
- [ ] Two concurrent streams interleave; cancelling one of four leaves three complete (tests c, d).
- [ ] Every wait is visible: `queued` events, `execution` metadata, `/v1` header, residency rows; the
      queue bound and timeout are typed errors.
- [ ] No `n_ctx` is ever lowered to fit a slot; a person's `model_concurrency` is used exactly or
      refused with the sentence.
- [ ] Release report: the numbers of ADR 0011 Validation per engine.
