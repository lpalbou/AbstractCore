# 0871 — Continuous batching on one context: llama.cpp `n_seq_max` and the mlx-lm `BatchGenerator` lane (ADR 0011, later)

> Type: feature
> Created: 2026-10-29
> Priority: high (after 0870 ships)
> Labels: concurrency, llama.cpp, mlx, batching, throughput
> Links: [ADR 0011](../../adr/0011-shared-weights-and-concurrent-requests.md) §3, [0870](0870_per_model_slot_scheduler_and_shared_gguf_weights.md), `docs/native-mlx-runtime.md` (the native lane, the reference shape), root backlog 1064

## Summary

0870 gives every engine shared weights and a bounded slot scheduler; on llama.cpp a slot is a whole
context (its own KV at the reserved `n_ctx`), on the mlx-lm lane the bound is one. This item replaces
"N contexts" with "N sequences in one execution loop" where the engine supports it, so that
concurrency stops costing a full KV reservation per slot and tokens are decoded for every active
request in one step:

1. **llama.cpp**: one context built with `n_seq_max = max_concurrent` and `kv_unified`, requests
   admitted as sequences through the low-level batch API (`llama_batch` with per-token `seq_id`,
   `llama_kv_cache_seq_rm` / `seq_cp` for cancel and prefix sharing), llama-server's slot model inside
   the provider. The KV budget is ONE `n_ctx` shared by the sequences (the ADR 0010 reservation does not
   multiply); the fit check accounts the per-sequence compute buffers instead of a KV per slot.
   llama-cpp-python 0.3.35's high-level `Llama` does this only in embedding mode; the generation path
   needs the internals (`LlamaContext`, `LlamaBatch`, the sampler per sequence) and a per-sequence
   prompt-cache mapping (`LlamaState` is per context; a sequence's prefix lives in the shared cache).
2. **mlx-lm lane**: generation through `mlx_lm.generate.BatchGenerator` (continuous batching: `insert`,
   `remove`, `next`, per-uid prompt caches, `stats`), `max_concurrent` default 4; the prompt-cache store,
   hybrid snapshots and delta feed mapped onto `BatchGenerator`'s per-uid caches; the native lane's
   `NativeRuntime` admission and cancel semantics reused.
3. **transformers**: evaluate the library's continuous-batching manager once it is stable; not a
   release gate.

User-visible: the same API; more requests in parallel on the same memory; the queued state appears
later (at the sequence bound instead of the slot bound).

## Acceptance criteria

- [ ] 0870's tests (b)–(e) pass unchanged against the batched implementation (same contract, more
      throughput): N = 4 on one GGUF model within one `n_ctx` of KV, wall-clock < 0.5 × serial on Metal.
- [ ] mlx-lm lane: N = 4 concurrent requests batch (one `BatchGenerator`, `stats()["active"] == 4`),
      prompt-cache reuse per uid proven with the existing prefix-reuse tests.
- [ ] The 1064 benchmark's 4×4 matrix shows the throughput gain over 0870's slot pool; numbers in the
      release report.
