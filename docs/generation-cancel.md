# Stopping a generation (host cancel) and ejecting a model

A host stops an in-flight generation by passing a `threading.Event` as
`cancel_event=` to `generate()` and setting it later. AbstractRuntime does this
for every LLM call (a UI **Stop** sets the event of the running effect);
AbstractGateway adds a kill switch on top. The contract lives in
`abstractcore/providers/generation_cancel.py`.

```python
import threading
from abstractcore import create_llm
from abstractcore.exceptions import GenerationCancelledError

llm = create_llm("lmstudio", model="llama-3.2-1b-instruct")
stop = threading.Event()
threading.Timer(3.0, stop.set).start()        # e.g. the user pressed Stop
try:
    llm.generate("Write a very long story.", cancel_event=stop)
except GenerationCancelledError as e:
    print("stopped:", e)                        # never an "Error: ..." answer, never retried
```

## What every provider guarantees

- An event that is already set raises before anything is sent to the model.
- A stop raises `GenerationCancelledError` (`request_local`: never retried, never
  counted against endpoint health). A cancelled answer is never returned as a
  complete one, and never as an `"Error: ..."` string with `finish_reason="error"`.
- Streams are checked between chunks for every provider, and the upstream stream
  is closed.
- A call whose event is set that fails for any reason (for example a severed
  socket) is reported as the typed stop, not as a transport error.

## Per lane

| Lane | Where the stop is observed | Measured (2026-09-23) |
|---|---|---|
| MLX (native runtime, mlx-vlm, mlx-lm) | every sampled token; mlx-lm also between prefill chunks | decode stopped 5 ms after the cancel reached it (4B + MTP, hermetic gateway) |
| HuggingFace transformers | a `StoppingCriteria` that raises, every decode step; chunked prefill checks between chunks | 2-8 ms (SmolLM2-135M, MPS) |
| HuggingFace GGUF, control-plane lane | every sampled token; prefill evaluated in `n_batch` slices with a check between slices | decode 3-5 ms; prefill 34 ms (was 2.4 s before the slicing) |
| HuggingFace GGUF, `create_chat_completion` lane | a logits processor forces end-of-generation at the next token, then the provider raises | 1-5 ms |
| LM Studio (OpenAI-compatible and native `/api/v1/chat`) | the request is **severed** (see below) | client returned in ~1 ms; LM Studio worker CPU 50 % → 0 % |
| Ollama (`/api/chat`, `/api/generate`, `/v1`) | the request is severed | client ~1 ms; runner CPU 36-40 % → 0 % |
| llama.cpp server, vLLM, any OpenAI-compatible server | the request is severed | llama-server logged `stop: cancel task` for streaming, non-streaming and mid-prefill requests |

### HTTP lanes: the request is severed

Checking "between chunks" is not enough over HTTP: a non-streaming request sits
in a blocking socket read for the whole generation, and a streaming request sits
there for the whole prefill. Neither the per-chunk check nor an injected
exception (the gateway kill switch) can reach a thread blocked in `recv`.

So when an event is passed, the request runs on its own connection under an
`HttpCancelGuard`: a watcher thread waits on the event and shuts the socket down
(`shutdown(SHUT_RDWR)`), which wakes the blocked read immediately. The server sees
the client disconnect and stops decoding. Measured: LM Studio 0.4.20 (MLX engine),
Ollama 0.20.2 and llama.cpp's `llama-server` stop on disconnect, streaming or not.
Ollama finishes a prompt evaluation already in progress before it notices; LM
Studio and llama-server stop mid-prefill. Uncancellable calls keep using the
provider's pooled client, exactly as before.

### Not interruptible from Python

- One native call: one transformers prefill forward (chunks of
  `ABSTRACTCORE_TRANSFORMERS_PREFILL_STEP` tokens), one llama.cpp `n_batch`
  evaluation inside `create_chat_completion`, one mlx-vlm prefill op. The stop
  lands when that call returns.
- The Outlines structured-output lane and custom `infer()` models (for example
  DeepSeek-OCR) own their loop: they are checked before they start only.

## Ejecting a model (`unload_model`) while it generates

`unload_model()` first cancels the instance's in-flight generations and waits for
them (`generation_cancel.InflightGenerations`, bounded by
`DEFAULT_EJECT_DRAIN_TIMEOUT_S`, 30 s, logged). In-process providers (MLX,
HuggingFace) refuse the unload with a `ProviderAPIError` if a call is still running
at the deadline, so memory is never freed under a running decode (`Llama.close()`
under a running llama.cpp evaluation crashes the process). HuggingFace gives every
call a private event when the host passed none, so any call can be stopped.

HTTP providers cancel **our** requests before asking the server to unload. LM Studio
answers an unload under a running request with an in-stream "Model unloaded" error,
which the retry layer would resample, reloading the model the operator just ejected.

After an eject:

- `load_model()` reloads an MLX or HuggingFace instance. It was missing, so a
  gateway `POST /models/load` after an unload of the default model could not
  warm it again.
- A generation on an ejected MLX or HuggingFace instance reloads the model on
  demand and logs a WARNING. Before, the gateway recorded a *completed* run whose
  answer was the string `"Error: MLX model not loaded"`.
- The sync httpx client of LM Studio, Ollama and OpenAI-compatible providers is
  replaced, not left closed. A closed client made every later request of a pooled
  instance raise.

Measured through the hermetic gateway routes (`POST /api/gateway/models/unload`),
with the settled process phys_footprint:

| Provider | Loaded | After eject | Eject during a running generation |
|---|---|---|---|
| MLX 4B + MTP | 4,110 MB | 769 MB (settled in 0.8 s) | stopped 6-9 ms after, `cancelled_by: model_eject`; 1,039 MB |
| HuggingFace SmolLM2-135M | 1,985 MB | 702 MB | stopped 10 ms after; 893 MB |
| GGUF Qwen3.5-2B (in-process) | 4,724 MB | 267 MB | stopped 3-10 ms after |
| LM Studio llama-3.2-1b | `loaded_instances: [id]` | `[]` | stopped 1 ms after; not JIT-reloaded |
| Ollama gemma3:1b | `/api/ps`: 1.41 GB | `[]` | stopped 0-1 ms after |

The operating system reclaims freed Metal buffers asynchronously, so a footprint
read right after the unload can still show the weights for up to about a second.
Use the MLX allocator (`mx.get_active_memory()`, which is 0 immediately) or wait
for the reading to settle.

## Remote AbstractCore server

`RemoteAbstractCoreLLMClient` (AbstractRuntime) severs its request to the
AbstractCore server when the effect's event is set. The server treats the client
disconnect as a cancel: HTTP-backed providers run off the event loop with a
disconnect watcher, and their own guard severs the upstream request in turn.
There is no abort endpoint and no request id: the wire form of a Stop is the
closed connection. Measured chain runtime → server → LM Studio: the client
returned in about 0 ms and the LM Studio worker went from 49.5 % CPU to 0 %.
In-process providers on the server (transformers, llama.cpp) stay serialized on
the event loop and are not cancelled by a disconnect.
