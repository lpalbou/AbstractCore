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
  holder whose model name or path matches (`model_matches`, the eject's matcher), else None. Only the
  huggingface backend lends: MLX instances already share weights, embeddings are not chat providers. One
  instance is safe across threads (GGUF generation serialized by the instance's `_gguf_prompt_cache_lock`,
  transformers on MPS by `_MPS_GENERATION_LOCK`).
- `server/app.py` chat route: without a server-registry runtime, `_resident_provider_for_request` lends the
  resident holder before `create_llm`. Never with `unload_after` (it would unload the holder's model), never with
  request-specific provider settings other than tracing and the timeout (an explicit provider key or a
  `base_url` builds the request's own instance, as before).

## Evidence
- `tests/server/test_server_reuses_resident_provider.py`: three chats on a registered holder build no instance
  (`create_llm` never called, the holder serves all 3); `unload_after`, another model, a `base_url`, an explicit
  key, `mlx`, and a holder whose weights were released all build their own. 2 passed; with the previous
  `server/app.py`, 2 failed.
- Metal, the same scratch-gateway script with this branch installed: holders stay at 1 and RSS at 11.2 GB over
  three `/v1` calls (all HTTP 200).
