# ADR 0012: Idle unload — a model no call has used for 60 minutes is unloaded; a kept-loaded model never is

Status: Accepted (2026-11-04). Amends [ADR 0010](0010-model-residency-fit-and-eviction.md). Operator, 2026-11-03,
verbatim: "a model loaded in the framework ... must have a default TTL of 60mn. if the model is never used for 60mn,
it must automatically unload from memory. this must be customizable and in particular, in gateway, when a model is
'locked', it means TTL is infinite and it can't unload as long as the lock is not removed." Root backlog 1115;
design: option C of the architect's note (Core mechanism, host policy).

## Context

Before this ADR nothing unloaded a model that sat unused. The only automatic eviction was ADR 0010's failsafe,
which frees memory when a new load does not fit. A gateway kept every model it had ever loaded until someone ejected
it or the process restarted. The facts an idle rule needs already live in Core: which instances hold which weights
(`process_residency`), who claims them (runtime pools, locks, builds), and when each model was last used
(`InflightGenerations.last_used_at`).

## Decision

1. **Idle is a second eviction trigger, under ADR 0010's rule.** A resident in-process model (HuggingFace:
   transformers and GGUF; MLX; in-process embeddings) is unloaded when no call has used it for the idle-unload time.
   What keeps it is exactly what 0010 says locks a model: an admin's Keep loaded lock (a claim with `locked: True`)
   and use. Use means a call running on any instance, or a claimant still building the model (a `building` claim).
   Pools, configured defaults, sessions and prompt caches do not keep a model loaded.
2. **The default is 60 minutes; a host sets it.** Core owns the mechanism and the default
   (`process_residency.DEFAULT_IDLE_UNLOAD_S = 3600`). A host passes one number for its process,
   `process_residency.set_idle_unload(seconds)`, where `0` or `None` turns it off. `get_idle_unload()` reads it.
   AbstractRuntime relays the host's setting; the gateway owns the setting, its launch flag and the UI.
3. **Last used.** The time of a model is the newest `last_used_at` over its instances. Every call stamps it when it
   begins and ends, whether or not it carries a cancel event, and every call is counted in flight. A model that no
   call has used yet is timed from when the reaper first saw it resident.
4. **Never cancel, never decide while memory moves.**
   - The reaper skips a tick while a load runs in the process (`load_lock`).
   - Before an eject it closes the in-flight gate of every holder. Calls that begin then wait instead of running.
   - With the gates closed it checks the model again. A call that began since the first check makes the model in
     use, and it is kept.
   - So an idle unload never cancels a call. A waiting call runs once the gates reopen, and the provider reloads the
     model on demand (the eject contract).
   - Embeddings are not gated: `EmbeddingManager.unload` waits for running encodes and never cancels them.
5. **The eject is the process-wide one.** It unloads every holder, then garbage-collects and returns the backend's
   allocator cache (`mx.clear_cache`, torch's caches), with `reason="idle"`. Each unload is logged by
   `abstractcore.residency` at INFO: `unloaded <model> after <N> min idle; freed <X.Y GB>` (seconds under two minutes). When something is left
   resident, the line is a WARNING with the holders that refused.
6. **Where it runs.** One daemon thread per process starts on the first in-process load and ticks every
   `min(60 s, idle / 4)`. `reap_idle(now=None)` runs one pass and returns
   `[{model, backend, idle_s, freed_bytes, ok}]`.
7. **Rows.** `resident_model_states()` reports every lane with `last_used_at`, `in_use`, `locked_by` and
   `idle_unload_at`. `idle_unload_at` is None when the setting is off or the model is locked or in use. ADR 0010's
   failsafe now sees MLX and embedding models too.

## Consequences

- After an hour of no use the next call pays the load, and a full prefill: the in-memory prompt caches go with the
  weights. Keep loaded is the answer for a model that must stay warm. An hourly automation sits on the boundary
  (lock it, or raise the time).
- A per-load `ttl_s` / `keep_alive` stays unsupported for in-process providers; their warning names the
  process-wide idle unload instead. External servers (Ollama, LM Studio) keep their own TTLs.
- Media, voice and music plugin engines do not report a last use yet, and are not covered (root backlog 0991).
- Lock persistence across restarts is the gateway's (root backlog 1115). Core reads the claims it is given.

## Enforcement

`tests/providers/test_idle_unload_unit.py` goes red, rule by rule, when:
- a locked model, a running call or a building model is unloaded;
- a call that starts between the check and the eject is cancelled;
- a call runs during the eject;
- a tick runs during a load;
- MLX or embedding models are missing from the states;
- a call without a cancel event is not counted;
- the reaper does not start with the first load.

ADR 0010's tests (`test_residency_fit_and_evict_unit.py`) still pass on the extended states.
