# ADR 0010: Model residency failsafe — fit before load, evict unlocked LRU, refuse with a typed error

Status: Accepted (2026-10-29).

## Context

Loading a model must never crash the gateway. On the Linux test server (one 16 GB GPU, 26 GB RAM, two
accounts' runtimes in one process) a load used whatever memory it could get: the llama.cpp context
ladder retried smaller contexts, then fell back to the CPU, and a model that did not fit next to the
resident ones either squeezed them, filled RAM, or failed deep inside llama.cpp with no sentence a
person can act on. Memory was never *managed*: nothing chose what to free.

Core already owns the three facts this needs (ADR 0008): which instances hold which weights in this
process (`hf_residency`, `mlx_residency`), who claims them (`process_residency` claimants: runtime
pools, locks, builds), and how big a load is (`utils.kv_geometry`: weights + KV at a context).

## Decision

Core owns the failsafe, in `process_residency`, because the decision needs every holder in the
process, every claim, and the estimate; Runtime and Gateway only declare locks and surface the outcome.

1. **One fit check, one place.** Before a provider allocates (`HuggingFaceProvider._load_gguf_model`
   first; other in-process lanes adopt the same call), Core computes the need = file weights + KV at
   the smallest context it would accept (the device rule of `docs/memory-management.md` then sizes the
   context in what is left) + the reserve (max(1 GiB, 10% of the device)), and compares it with the
   memory free *now* on the device the load targets (CUDA: `torch.cuda.mem_get_info`; CPU lane:
   available RAM). One function: `process_residency.fit_check(...)`.
2. **What locks a model (nothing else does):** an admin's "Keep loaded" pin (the runtime's
   `lock_model_residency`, the Models page toggle), and in-flight use (a generation running on an
   instance of it). Pools, configured defaults, sessions and prompt caches do not lock.
3. **Last used, per model:** the newest `last_used_at` over the model's instances (stamped when a
   generation begins and ends; the load time until the first call).
4. **Eviction order:** when the need does not fit, evict resident in-process models that are not
   locked and not the one being loaded, least recently used first, one at a time (process-wide
   eject: every holder of that model), re-measuring free memory after each, until it fits or none is
   evictable. Locked models are never evicted by this process. An evicted model reloads on its next
   call (the eject contract), so a runtime pool keeps working.
5. **The typed refusal:** if it still does not fit, the load fails with
   `ModelDoesNotFitError(ProviderError)` carrying `model`, `needed_bytes`, `free_bytes`,
   `device`, `locked` (the models kept, in words), `evicted`, and `suggestions` (the catalog's
   smaller artifacts of the same model: smaller quant, smaller size, that fit `free + evictable`),
   and the sentence: "MODEL needs 13.7 GB but only 9.2 GB is free on the GPU (kept: X, locked).
   Choose a smaller model, for example: A (5.6 GB), B (4.1 GB), or unlock X." Nothing is allocated.
6. **Broadcast:** Runtime records the error type and its fields on the failed LLM_CALL / load in the
   ledger (`error.code = "model_does_not_fit"`); the Gateway shows the sentence as a console toast and
   on the Models page, and thin clients' chat renders the same sentence (the kit's error sentence).
7. **Multi-account runtimes:** a model held by several accounts' runtimes is one resident model (the
   claims of all owners count once); it is locked if ANY owner locks it or any instance of it is in
   use; evicting it ejects every holder (AbstractRuntime 0.15.x: dropped copies are released by their
   owner, unbuilt defaults claim nothing). The refusal names locks by owner label ("the runtime of
   account default/bob"), never by id (ADR 0043 of the framework).

8. **A load reserves the MAXIMUM context, explicitly** (operator ruling 2026-10-29). When nothing is set, the context
   is the smaller of the model's advertised context and the largest whose weights + KV + compute fit the device's free
   memory minus a safety buffer (5 % by default, `model_memory_buffer` in the AbstractCore config), computed by
   inverting the KV-size formula (`_gguf_max_context_that_fits`); if the allocation still fails the context halves,
   down to 4,096. No fixed rungs, no "half of the free memory". The choice is logged ("reserving 131,072 of 262,144
   tokens (limited by memory; ... 5 % buffer)") and carried on the residency claim (`context_length`,
   `advertised_context_length`, `context_source`, `memory_buffer`, `context_reservation`), so the Models row shows it
   ("131,072 of 262,144; limited by memory, 5 % buffer"). A person can set a model's context
   (`config.model_contexts.set_model_context`): used exactly or refused with the readable sentence, never shrunk. A
   request longer than the loaded context NEVER gets cut: it fails with `ContextWindowExceededError`. No per-route
   caps (framework ADR 0026). The `model_loaded` ledger event with these numbers is AbstractRuntime backlog 0865.

## Consequences

- A load either fits (possibly after evicting idle unlocked models) or fails before allocating with a
  sentence and alternatives; the CPU fallback is subject to the same check against RAM.
- Eviction is visible: each evicted model is logged and reported in `evicted` of the load result.
- Estimates are estimates (labelled `kv_cache_estimated`): the context ladder and its probe still
  guard the real allocation; the fit check only prevents the loads that cannot fit.

## Enforcement

Tests in `tests/providers/test_residency_fit_and_evict_unit.py` go red when a locked or in-use model
is evicted, when the order is not LRU, when the refusal lacks its numbers or suggestions, or when a
load allocates without the check. A real-weights proof on the Linux test server (lock A, refuse B;
unlock A, B evicts A) is recorded in the PR.
