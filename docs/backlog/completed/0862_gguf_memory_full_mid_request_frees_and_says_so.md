# 0862: A GGUF context that fills up mid-request is freed and the request fails in words

**Status**: completed 2026-10-29 (unreleased; stacked on `fix/gguf-kv-context-honesty`, PR #9).
**Found by**: the OVH release-test server (Core 2.31.0, `unsloth/gemma-4-E4B-it-GGUF` on a 16 GB GPU): gateway.log
ends with `decode: failed to find a memory slot for batch of size 512` repeated on every later call.

## Cause

On the delegated llama-cpp-python lanes (`create_chat_completion`, plain and streamed), a decode that cannot place a
batch in the context memory raises `RuntimeError("llama_decode returned 1")`. The GGUF lane turned it into an
"Error: llama_decode returned 1" answer and left the engine's memory as it was, so the next request met the same
full memory. (PR #9 handles the prompt-cache lane and prompts that do not fit before eval.)

## Fix

`HuggingFaceProvider._gguf_raise_if_decode_failed`: on a decode failure, `Llama.reset()` and the context's
`memory_clear(True)` (`kv_cache_clear()` on an older llama-cpp-python), then `ContextWindowExceededError` with the
sentence "<model> ran out of context memory while answering (its N-token context is full; llama.cpp could not
place the next batch, code 1). The request was stopped and that memory freed. Start a new conversation or shorten
this one, or load the model with a larger context if the device has room." Applied in the plain lane's handler and
in `_gguf_context_checked_stream` (a streamed failure).

## Evidence

`tests/providers/test_gguf_memory_full_unit.py` (memory freed + the sentence; a streamed failure; the older API);
red when the freeing is removed. `tests/providers`: 1727 passed.
