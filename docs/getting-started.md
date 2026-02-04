# Getting Started

AbstractCore is a unified Python interface for cloud + local LLM providers. The default install is lightweight; add features via extras.

## Prerequisites

- Python 3.9+
- `pip`

## Installation

```bash
# Core (small, lightweight default)
pip install abstractcore

# Providers (install only what you use)
pip install "abstractcore[openai]"       # OpenAI SDK
pip install "abstractcore[anthropic]"    # Anthropic SDK
pip install "abstractcore[huggingface]"  # Transformers / torch (heavy)
pip install "abstractcore[mlx]"          # Apple Silicon local inference (heavy)
pip install "abstractcore[vllm]"         # GPU inference server integrations (heavy)

# Optional features
pip install "abstractcore[tools]"        # built-in tools (web/file/command helpers)
pip install "abstractcore[media]"        # images, PDFs, Office docs
pip install "abstractcore[compression]"  # glyph visual-text compression (Pillow renderer)
pip install "abstractcore[embeddings]"   # EmbeddingManager + local embedding models
pip install "abstractcore[tokens]"       # precise token counting (tiktoken)
pip install "abstractcore[server]"       # OpenAI-compatible HTTP gateway

# Combine extras (zsh: keep quotes)
pip install "abstractcore[openai,media,tools]"
```

Local OpenAI-compatible servers (Ollama, LMStudio, vLLM, llama.cpp, LocalAI, etc.) work with the core install; you just point AbstractCore at the server base URL. See [Prerequisites](prerequisites.md) for provider setup.

Optional capability plugins (deterministic multimodal outputs):

```bash
pip install abstractvoice   # enables llm.voice / llm.audio (TTS/STT)
pip install abstractvision  # enables llm.vision (generative vision; typically via an OpenAI-compatible images endpoint)
```

See: [Capabilities](capabilities.md) and [Server](server.md).

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

## Your first call

OpenAI example (requires `pip install "abstractcore[openai]"`):

```python
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o-mini")
resp = llm.generate("What is the capital of France?")
print(resp.content)
```

## Streaming

```python
from abstractcore import create_llm

llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M")
for chunk in llm.generate("Write a short poem about distributed systems.", stream=True):
    print(chunk.content or "", end="", flush=True)
```

## Tool calling

AbstractCore supports native tool calling (when the provider supports it) and prompted tool syntax (when it doesn’t).

By default, tool execution is pass-through (`execute_tools=False`): you get tool calls in `resp.tool_calls`, and your host/runtime decides how to execute them.

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

Images and document extraction require `pip install "abstractcore[media]"` (Pillow + PDF/Office deps).

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

If your main model is text-only, you can configure vision fallback (two-stage captioning) so images are automatically described and injected as short observations. See [Media Handling](media-handling-system.md), [Vision Capabilities](vision-capabilities.md), and [Centralized Config](centralized-config.md).

For long documents, AbstractCore can optionally apply Glyph visual-text compression. Install `pip install "abstractcore[compression]"` (and `pip install "abstractcore[media]"` for PDFs) and see [Glyph Visual-Text Compression](glyphs.md).

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
abstractcore --configure
abstractcore --status

# Interactive chat
abstractcore-chat --provider openai --model gpt-4o-mini
```

## Next steps

- [Prerequisites](prerequisites.md) — provider setup (keys, base URLs, hardware notes)
- [FAQ](faq.md) — common questions and setup gotchas
- [Examples](examples.md) — end-to-end patterns and recipes
- [API (Python)](api.md) — public API map and common patterns
- [API Reference](api-reference.md) — complete function/class listing
- [Troubleshooting](troubleshooting.md) — common errors and fixes
- [Server](server.md) — OpenAI-compatible HTTP gateway
