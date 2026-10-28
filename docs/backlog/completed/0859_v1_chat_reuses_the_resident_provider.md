# 0859 — `/v1/chat/completions` reuses the model already resident in the process

**Status**: completed (2026-10-28, 2.31.1) · **Priority**: P1 · **Created**: 2026-10-28
**Cross-package**: AbstractGateway (serves `/v1` through this app via AbstractRuntime's `server_facade`; its
runtime pool holds the model the gateway loads); root release-closing wave 2026-10-28, fresh-Linux test QA 4
(framework root `untracked/release-closing/ovh-test.log`).

## Why
Fresh Linux install (RTX, 16 GB): the gateway loaded `unsloth/Qwen3-4B-Instruct-2507-GGUF` (2.5 GB) through its
runtime pool; `POST /v1/chat/completions` answered 500 "Failed to load GGUF model ... Failed to create
llama_context" three times; nvidia-smi showed 12.3 GB held by the gateway for one model. `/v1/chat/completions`
(`server/app.py`) took the server registry's runtime (`_get_loaded_gateway_runtime`) or else called `create_llm`
FOR EVERY REQUEST: the runtime pool's instance was invisible to it, and a HuggingFace provider instance never
shares weights (`hf_residency`: each holder is a full copy plus its llama.cpp context and compute buffers).

Reproduced on this Mac (Metal, `unsloth/Qwen3.5-4B-GGUF` Q4_K_M, scratch gateway 0.17.1, `POST
/api/gateway/models/load` then three `/v1/chat/completions`): holders 1 -> 2 -> 3 -> 4, process RSS 11.2 -> 21.9 ->
32.6 -> 43.4 GB.

## What shipped
- `process_residency.resident_provider(provider, model, task=None)`: the first alive registered `huggingface`
  holder whose model name or path matches (`model_matches`, the eject's matcher), else None; an instance the
  server built for one request (`_abstractcore_request_scoped`, set by the chat route: it may be unloaded when
  that request ends) is never lent. Only the huggingface backend lends: MLX instances already share weights,
  embeddings are not chat providers.
- `HuggingFaceProvider._generation_worker` (`_InstanceGenerationWorker`): every operation on the instance's model
  runs on ONE worker thread of that instance, one at a time: generations (a stream is produced there into an
  unbounded queue, so it never waits for its consumer; an early close stops the producer at its next chunk),
  and, through `_on_model_thread`, the prompt-cache backend create/clone/append, `prompt_cache_set/clear/save/load`
  and the free step of `unload_model` (`_free_model_on_unload`, after the in-flight stop). A stream is in flight
  (`_inflight_generations`) from the call until its production ends; the ejected-model reload is decided on the
  worker; the caller's contextvars are copied to the call; a caller interrupted while waiting (Ctrl-C) sets the
  call's cancel event; a call made from the worker runs inline. Before, only the GGUF prompt-cache prefill took an
  instance lock (`_gguf_prompt_cache_lock`); `create_chat_completion` and the streams took none, so a pool serving
  two runs could already run two generations on one context.
- `server/app.py` chat route: without a server-registry runtime, `_resident_provider_for_request` lends the
  resident holder before `create_llm`. Never with `unload_after` (it would unload the holder's model), never with
  request-specific provider settings other than tracing and the timeout (an explicit provider key or a
  `base_url` builds the request's own instance, as before). A lent instance answers off the event loop with the
  disconnect-watched cancel (it queues behind the holder's runs). It keeps the holder's construction defaults
  (seed, default max output tokens, context size), shares its prompt-cache store and, with tracing on, its trace
  buffer.
- `hf_residency`: registry adds and snapshots under one lock (a WeakSet iterated while another thread adds raises).

## Evidence
- Adversarial review round 1 (Opus, read-only): the first version lent an instance with no generation
  serialization (a scratch repro ran 2 generations on 1 instance at once) and lent an `unload_after` request's
  own instance, which was then unloaded under the borrower. Both fixed (a per-instance lock held across a stream's
  yields; request-scoped instances never lent).
- Adversarial review round 2 (Opus, read-only), against the worker design that replaced the lock (the lock deadlocked
  a call waiting on the event loop behind a paused stream, and two streams advanced by one thread). Confirmed and
  fixed: prompt-cache operations ran on the shared context during a generation (now on the worker); a stream
  produced before it was registered in flight and the ejected-reload check ran before queueing (eject could free
  the model under them); a lent `/v1` call waited on the event loop behind the holder's runs; contextvars did not
  reach the worker (an `endpoint:<id>` route in the vision fallback); Ctrl-C did not stop the generation; the
  registry snapshot could raise under a concurrent add; the docs described the lock. Not issues (verified): request
  generation settings stay per call (`gen_kwargs`); no hang on early close, mid-stream errors or re-entry;
  `unload_after`/`base_url`/explicit key never borrow; MLX/embeddings never lent; idle worker threads exit with
  their instance. Recorded, not changed: borrowers inherit the holder's construction defaults, prompt-cache store
  and trace buffer (CHANGELOG).
- `tests/providers/test_hf_instance_generation_gate.py` (11): one worker thread for calls from many threads; a
  paused stream never blocks a waiting call; two streams on one thread never overlap; early close frees the
  instance; start errors raise at the call; re-entry inline; every model operation (8 methods) runs on the worker
  and never overlaps a stream; a stream is in flight from the call to the end of its production; an eject while
  a call waits is seen on the worker (reload once); the caller's contextvars reach the lane; SIGINT while waiting
  cancels the generation. With the worker replaced by a passthrough: 3 failed; with each round-2 fix removed
  (in-flight registration, context copy, cancel on interrupt, reload placement, one decorator): 5 failed.
- `tests/server/test_server_reuses_resident_provider.py`: three chats on a registered holder build no instance
  (`create_llm` never called, the holder serves all 3, never on the event loop); `unload_after`, another model, a
  `base_url`, an explicit key, `mlx`, and a holder whose weights were released all build their own. 2 passed;
  with the previous `server/app.py`, 2 failed; with the off-loop lane removed, 1 failed.
- Metal, the scratch-gateway script (`untracked/release-closing/w5/scripts/v1_holders.sh`) with this branch
  installed: METAL_AFTER
