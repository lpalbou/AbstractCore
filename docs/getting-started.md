# Getting Started

AbstractCore is a unified Python interface for cloud, gateway, and local LLM providers. The default install runs every remote provider; two install settings add local engines.

## Prerequisites

- Python 3.9+
- `pip`

## Installation

Pick one of the three install settings:

```bash
pip install abstractcore             # light: every remote provider, tools, media, server, plugins
pip install "abstractcore[apple]"    # Apple silicon: light + every local engine a Mac can run
pip install "abstractcore[gpu]"      # NVIDIA / AMD: light + every local engine a GPU machine can run
```

The light install covers OpenAI, Anthropic, OpenRouter, Portkey, and any OpenAI-compatible
server (Ollama, LM Studio, vLLM, llama.cpp, LocalAI, ...): point AbstractCore at the server base
URL. It also carries the built-in tools, media inputs (images, PDFs, Office documents), the
HTTP server, and the capability plugins (`llm.voice` / `llm.audio` via AbstractVoice, `llm.vision`
via AbstractVision, `llm.music` via AbstractMusic) with their remote backends.

`apple` and `gpu` add the local engines: MLX (Apple) or vLLM (GPU), HuggingFace/GGUF, local
embeddings, and local voice, image, video and music engines. See [Installation](installation.md)
for the full contents and platform requirements, and [Prerequisites](prerequisites.md) for
provider setup.

See: [Capabilities](capabilities.md) and [Server](server.md).

For generative vision, AbstractCore does not hardcode a local model default.
Configure an AbstractVision/OpenAI-compatible image default and omit `model`, or
route explicitly with `model="diffusers/default"`,
`model="diffusers/<huggingface-repo>"`, `model="sdcpp/default"`, or
`model="openai-compatible/<model>"` on `/v1/images/generations` /
`/v1/images/edits`. Local Diffusers runs cache-only unless you opt in to
downloads.

## Providers and models

AbstractCore uses a provider ID plus a model name:

```python
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o-mini")
# llm = create_llm("anthropic", model="claude-haiku-4-5")
# llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")
# llm = create_llm("lmstudio", model="qwen/qwen3-4b-2507")
# llm = create_llm("openai-compatible", model="default", base_url="http://localhost:1234/v1")
```

Tip: you can omit `model=...`, but it’s usually better to pass an explicit model to avoid surprises when defaults change.

Open-source-first: start with local providers (Ollama, LMStudio, MLX, HuggingFace), then add cloud or gateway providers as needed.

Gateway providers (OpenRouter, Portkey) examples:

```python
from abstractcore import create_llm

llm_openrouter = create_llm("openrouter", model="openai/gpt-4o-mini")
llm_portkey = create_llm("portkey", model="gpt-5-mini", api_key="PORTKEY_API_KEY", config_id="pcfg_...")
```

Note: gateway providers only forward optional generation params (e.g. `temperature`, `top_p`, `max_output_tokens`) when you explicitly set them.

## Your first call

OpenAI example (works with the light install; set `OPENAI_API_KEY`):

```python
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o-mini")
resp = llm.generate("What is the capital of France?")
print(resp.content)
```

## Request plus output

The simplest calls stay prompt-first, but direct Core multimodal callers can now use the lower-level
keyword form:

```python
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o-mini")

resp = llm.generate(
    request={"text": "A red ceramic mug on a white table."},
    output={"modality": "image", "format": "png"},
)
```

This is equivalent in spirit to the older compatibility forms:

```python
resp = llm.generate("A red ceramic mug on a white table.", output={"modality": "image"})
resp = llm.generate(text="Hello from AbstractCore.", output={"modality": "voice", "voice": "coral"})
```

Manual route pins are still supported when you need them:

```python
resp = llm.generate(
    request={"text": "Slow dolly shot over a misty valley."},
    output={
        "modality": "video",
        "provider": "mlx-gen",
        "model": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    },
)
```

When you do not pin a route, capability defaults can supply provider/model/base URL and, for
reasoning-capable text routes, a default reasoning level. See
[Request and Output](request-output.md) and [Centralized Config](centralized-config.md).

## Sessions (multi-turn)

Use a session to keep conversation state (system prompt + message history) across turns:

```python
from abstractcore import BasicSession, create_llm

llm = create_llm("openai", model="gpt-4o-mini")
session = BasicSession(provider=llm, system_prompt="You are a helpful assistant.")

print(session.generate("Hello!").content)
print(session.generate("Now continue.").content)
```

For prompt-cache-aware long chats (reuse stable prefixes like system/tools/files), use `CachedSession`:
- See [Prompt Caching](prompt-caching.md).

## Thinking / reasoning (best-effort)

Many modern models can optionally emit a reasoning/thinking trace (sometimes in a separate channel, sometimes inline). AbstractCore exposes a single unified control:

```python
from abstractcore import create_llm

llm = create_llm("lmstudio", model="qwen3.5-27b@q4_k_m", base_url="http://localhost:1234/v1")

# Disable thinking (tries to suppress any reasoning trace)
resp = llm.generate("Compute 17*23 - 19*11. Reply with the integer only.", thinking="none")
print(resp.content)

# Enable thinking (levels are best-effort; not all backends support budgets)
resp = llm.generate("Solve a hard logic puzzle.", thinking="high")
print(resp.content)
print(resp.metadata.get("reasoning"))  # when the backend exposes it
```

Notes:
- For **Qwen3 / Qwen3.5 on LM Studio**, AbstractCore uses LM Studio’s model template variables (`enable_thinking` / `enableThinking`) and a Qwen template “hard switch” for `thinking="none"` (empty `<think></think>`), rather than injecting “Reasoning effort …” text into the system prompt.
- For **Qwen3 / Qwen3.5 GGUF via HuggingFaceProvider (llama-cpp-python)**, there is no template-kwargs knob exposed by llama-cpp-python today, so `thinking="none"` also uses the Qwen hard-switch marker. If GGUF loading fails due to huge advertised context windows, AbstractCore will retry with smaller `n_ctx` values (best-effort); you can also pass `max_tokens=...` when constructing `HuggingFaceProvider()` to explicitly control llama.cpp `n_ctx`.
- For **Ollama**, enabling thinking may consume a lot of output tokens in the thinking channel; consider using a larger `max_output_tokens` when `thinking` is enabled.

For server usage (OpenAI-compatible HTTP), see [Server](server.md) and [Generation Parameters](generation-parameters.md).

## Streaming

```python
from abstractcore import create_llm

llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")
for chunk in llm.generate("Write a short poem about distributed systems.", stream=True):
    print(chunk.content or "", end="", flush=True)
```

The last chunk of a stream carries the call's `finish_reason` and, when the provider reports it,
`usage`. On MLX it also carries `metadata["prompt_cache"]` when the call has a `prompt_cache_key`,
the same record a non-streamed call returns (see [Prompt Caching](prompt-caching.md)).
Models that think in `<think>` blocks stream their reasoning as `chunk.metadata["reasoning_delta"]`
while they think, and the answer as text after it. When the chat template itself opens the block
(Qwen3.x with thinking on), this works on the lanes that render the template in-process: MLX,
HuggingFace transformers and GGUF (llama.cpp). OpenAI-compatible servers (LM Studio, vLLM,
llama.cpp server, ...) render the template on the server, so AbstractCore cannot tell that the
block was opened: unless the server returns the reasoning separately (most do), the reasoning of
such a model arrives in one piece when the model closes the block. The same happens on a GGUF
model whose embedded chat template AbstractCore cannot render: a warning is logged and the stream
carries `metadata["thinking_stream"] == "held_until_close"`.
For GPT-OSS models, the Harmony channels are split as the stream arrives: the `final` channel is
the text, `analysis` is reasoning, and a reply cut off before `final` ends with empty text and
`finish_reason: "length"` (see [Tool Calling](tool-calling.md#tool-calls-while-streaming)).

Streamed usage on OpenAI-compatible servers: AbstractCore asks for it with
`stream_options: {"include_usage": true}`. A server that rejects that field is retried without it,
and from then on that provider instance does not send it (`_stream_options_unsupported` is `True`).
Usage still arrives if the server includes it in a chunk on its own, as LM Studio does on the last
chunk; a server that does neither gives streamed calls without usage. AbstractCore does not make
an extra non-streamed call to count tokens.

## Tool calling

AbstractCore supports native tool calling (when the provider supports it) and prompted tool syntax (when it doesn’t).

By default, tool execution is pass-through (`execute_tools=False`): you get tool calls in `resp.tool_calls`, and your host/runtime decides how to execute them.

In the AbstractFramework ecosystem, **AbstractRuntime** is the recommended runtime for executing tool calls durably (policy, retries, persistence). See [Architecture](architecture.md) and [Tool Calling](tool-calling.md).

```python
from abstractcore import create_llm, tool

@tool
def get_weather(city: str) -> str:
    return f"{city}: 22°C and sunny"

llm = create_llm("openai", model="gpt-4o-mini")
resp = llm.generate("What's the weather in Paris? Use the tool.", tools=[get_weather])

print(resp.content)
print(resp.tool_calls)
```

See [Tool Calling](tool-calling.md) and [Tool Syntax Rewriting](tool-syntax-rewriting.md) (`tool_call_tags`, server `agent_format`).

Note:
- If you pass both `tools=[...]` and `response_model=...` to `generate()`, AbstractCore uses a 2-pass hybrid flow (tool-capable call, then structured-output call). Streaming is not supported in this hybrid mode.

### Built-in tools

The light install includes a ready-made toolset for agentic scripts. Import it from
`abstractcore.tools.common_tools`:

- `skim_websearch` vs `web_search`: compact/filtered links vs full results
- `skim_url` vs `fetch_url`: fast URL triage (small output) vs full fetch + parsing for web documents and feeds (HTML/JSON/XML/RSS/Atom/PDF when supported)

See [Tool Calling](tool-calling.md) for a recommended workflow and the full built-in tool list.

## Structured output

Pass a Pydantic model via `response_model=...` to get a typed result back (instead of parsing JSON yourself):

```python
from pydantic import BaseModel
from abstractcore import create_llm

class Answer(BaseModel):
    title: str
    bullets: list[str]

llm = create_llm("openai", model="gpt-4o-mini")
answer = llm.generate("Summarize HTTP/3 in 3 bullets.", response_model=Answer)
print(answer.bullets)
```

See [Structured Output](structured-output.md) for strategy details and limitations.

## Media input (images/audio/video + documents)

Images and document extraction are part of the light install (Pillow, pypdf, Office parsers).

```python
from abstractcore import create_llm

llm = create_llm("anthropic", model="claude-haiku-4-5")
resp = llm.generate("Describe the image.", media=["./image.png"])
print(resp.content)
```

Audio and video attachments are also supported, but they are **policy-driven** (no silent semantic changes):
- audio: `audio_policy` (`native_only|speech_to_text|auto|caption`)
- video: `video_policy` (`native_only|frames_caption|auto`)

Speech-to-text fallback (`audio_policy="speech_to_text"`) typically requires installing `abstractvoice` (capability plugin).

What you need (quick checklist):
- **Images**: either a vision-capable model (VLM/VL) **or** configured vision fallback (`abstractcore --set-vision-provider PROVIDER MODEL`).
- **Video**: `ffmpeg`/`ffprobe` on `PATH` + either a vision-capable model **or** configured vision fallback (for frame sampling). Native video input is model/provider dependent.
- **Audio**: either an audio-capable model **or** speech-to-text fallback via `abstractvoice` + `audio_policy="auto"`/`"speech_to_text"`.

Defaults can be configured via the config CLI (`abstractcore --config`, `abstractcore --status`). See [Centralized Config](centralized-config.md).

If your main model is text-only, you can configure vision fallback (two-stage captioning) so images are automatically described and injected as short observations. See [Media Handling](media-handling-system.md), [Vision Capabilities](vision-capabilities.md), and [Centralized Config](centralized-config.md).

For long documents, AbstractCore can optionally apply Glyph visual-text compression. It is part of the light install; see [Glyph Visual-Text Compression](glyphs.md).

## Async

```python
import asyncio
from abstractcore import create_llm

async def main():
    llm = create_llm("openai", model="gpt-4o-mini")
    resp = await llm.agenerate("Give me 3 bullet points about HTTP caching.")
    print(resp.content)

asyncio.run(main())
```

## CLI (optional)

```bash
# Configure defaults and API keys
abstractcore --config
abstractcore --status

# Interactive chat
abstractcore-chat --provider openai --model gpt-4o-mini
```

## Next steps

- [Native MLX Runtime](native-mlx-runtime.md) — Qwen3.8 on Apple Silicon, vision, MTP and concurrent serving
- [Prerequisites](prerequisites.md) — provider setup (keys, base URLs, hardware notes)
- [FAQ](faq.md) — common questions and setup gotchas
- [Examples](examples.md) — end-to-end patterns and recipes
- [API (Python)](api.md) — public API map and common patterns
- [API Reference](api-reference.md) — complete function/class listing
- [Troubleshooting](troubleshooting.md) — common errors and fixes
- [Server](server.md) — OpenAI-compatible HTTP gateway
- [Endpoint](endpoint.md) — single-model OpenAI-compatible endpoint (one provider/model per worker)
