### Fixed

- **One llama.cpp weights instance per GGUF model in a process** (`providers/gguf_shared.py`, ADR 0011). Before, every provider instance of a resident GGUF loaded the weights again: a second account's runtime pool, a gateway-side client (run summary, Ask, sandbox, entity chat) or a server request each held a full copy. The 2026-10-29 audit measured 11.8, then 23.3, then 34.9 GB RSS for three holders of one 4B model, and the residency failsafe then refused the extra copies. Now every llama.cpp context of a model borrows the one resident `LlamaModel`. The weights are freed with the last context.

### Added

- **Concurrent generations on one weights instance: a per-model slot pool** (`providers/gguf_slots.py`, ADR 0011).
  - A slot is one llama.cpp context with its own KV, at the model's reserved context. Every slot of a pool has the same n_ctx.
  - Provider instances own no context. A generation borrows a slot for its duration, and the provider's `llm` is that slot while the call runs.
  - Bounds: 4 slots on a GPU or Metal and 1 on CPU-only by default. `model_concurrency` (`config/model_contexts.py`) sets the bound per model: the slots are built at load, or the load is refused with the numbers.
  - Extra slots are built lazily, only when their KV fits free memory now. A slot never evicts another model and never gets a smaller context; when it does not fit, the request waits and the reason is kept.
  - The queue is FIFO and bounded. `ModelQueueFullError` (503) and `ModelQueueTimeoutError` (504) are request-local and carry a Retry-After. A cancel while queued is a `GenerationCancelledError`.
  - Idle extra slots are freed after 300 s.
  - The queued state is visible in a new progress phase `queued` (with `queue_position`) and on the residency rows: `slots`, `slots_max`, `active`, `queued`, `slot_refusal_reason`, `slot_bytes`. A pool with work running or waiting counts as in use (never evicted).
  - Measured on Metal with Qwen3.5-4B Q4_K_M: RSS was 3.14 GiB for one holder, 3.14 GiB for two and 3.86 GiB with 4 slots. A short request completed in 0.10 s during a 3.95 s decode. Four equal decodes took 0.94 x the serial wall clock (separate contexts do not batch weight reads; batched decode is backlog 0871).
