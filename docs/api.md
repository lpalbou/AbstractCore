# API (Python)

This page is a user-facing map of the **public Python API** exposed from `abstractcore` (see `abstractcore/__init__.py`). For a complete listing of functions/classes (including events), see **[API Reference](api-reference.md)**.

Implementation pointers (source of truth):
- `create_llm`: `abstractcore/core/factory.py` → `abstractcore/providers/registry.py`
- `BasicSession`: `abstractcore/core/session.py`
- Response/types: `abstractcore/core/types.py`
- Tool decorator: `abstractcore/tools/core.py`

## Core entrypoints

### `create_llm(...)`

Create a provider instance:

```python
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o-mini")  # requires: pip install "abstractcore[openai]"
resp = llm.generate("Hello!")
print(resp.content)
```

Provider IDs (common): `openai`, `anthropic`, `openrouter`, `portkey`, `ollama`, `lmstudio`, `vllm`, `openai-compatible`, `huggingface`, `mlx`.

### Gateway providers (OpenRouter, Portkey)

```python
from abstractcore import create_llm

llm_openrouter = create_llm("openrouter", model="openai/gpt-4o-mini")
llm_portkey = create_llm("portkey", model="gpt-5-mini", api_key="PORTKEY_API_KEY", config_id="pcfg_...")
```

Gateway notes:
- OpenRouter uses `OPENROUTER_API_KEY` (model names like `openai/...`).
- Portkey uses `PORTKEY_API_KEY` plus a config id (`PORTKEY_CONFIG`).
- Optional generation parameters (`temperature`, `top_p`, `max_output_tokens`, etc.) are only forwarded when explicitly set.

### `BasicSession`

Keep conversation state:

```python
from abstractcore import BasicSession, create_llm

session = BasicSession(create_llm("anthropic", model="claude-haiku-4-5"))  # requires: abstractcore[anthropic]
print(session.generate("Give me 3 name ideas.").content)
print(session.generate("Pick the best one.").content)
```

### `tool` (decorator)

Define tools in Python with a decorator, then pass them to `generate()` / `agenerate()`:

```python
from abstractcore import create_llm, tool

@tool
def get_weather(city: str) -> str:
    return f"{city}: 22°C and sunny"

llm = create_llm("openai", model="gpt-4o-mini")
resp = llm.generate("Use the tool.", tools=[get_weather])
print(resp.tool_calls)
```

## Responses (`GenerateResponse`)

Most calls return a `GenerateResponse` object (or an iterator of them for streaming). Common fields:

- `content`: cleaned assistant text
- `tool_calls`: structured tool calls (pass-through by default)
- `usage`: token usage (provider-dependent)
- `metadata`: provider/model specific fields (for example extracted reasoning text when configured)

## Model downloads (`download_model`, optional)

`download_model(...)` is an **async generator** that yields `DownloadProgress` updates while a model is being fetched.

Supported providers:
- `ollama`: pulls via the Ollama HTTP API (`/api/pull`)
- `huggingface` / `mlx`: downloads from HuggingFace Hub (requires `pip install "abstractcore[huggingface]"`; pass `token=` for gated models)

Example:

```python
import asyncio
from abstractcore import download_model

async def main():
    async for p in download_model("ollama", "qwen3:4b-instruct-2507-q4_K_M"):
        print(p.status.value, p.message)

asyncio.run(main())
```

Implementation: `abstractcore/download.py`. For provider setup and base URLs, see [Prerequisites](prerequisites.md).

## Tool calling

Tools are passed explicitly to `generate()` / `agenerate()`:

```python
from abstractcore import create_llm, tool

@tool
def get_weather(city: str) -> str:
    return f"{city}: 22°C and sunny"

llm = create_llm("openai", model="gpt-4o-mini")
resp = llm.generate("Use the tool.", tools=[get_weather])
print(resp.tool_calls)
```

See **[Tool Calling](tool-calling.md)** and **[Tool Syntax Rewriting](tool-syntax-rewriting.md)**.

### Built-in tools (optional)

If you want a ready-made toolset (web + filesystem helpers), install:

```bash
pip install "abstractcore[tools]"
```

Then import from `abstractcore.tools.common_tools` (for example `web_search`, `skim_websearch`, `skim_url`, `fetch_url`). See **[Tool Calling](tool-calling.md)** for usage patterns and when to use `skim_*` vs `fetch_*`.

## Structured output

Pass a Pydantic model via `response_model=...` to receive a typed result:

```python
from pydantic import BaseModel
from abstractcore import create_llm

class Answer(BaseModel):
    title: str
    bullets: list[str]

llm = create_llm("openai", model="gpt-4o-mini")
result = llm.generate("Summarize HTTP/3 in 3 bullets.", response_model=Answer)
print(result.bullets)
```

See **[Structured Output](structured-output.md)**.

## Media input

Media handling is opt-in:

```bash
pip install "abstractcore[media]"
```

Then pass `media=[...]` to `generate()` / `agenerate()` (or use the media pipeline). Media behavior is **policy-driven**:

- Images: use a vision-capable model, or configure vision fallback (caption → inject short observations).
- Video: controlled by `video_policy` (native when supported; otherwise frame sampling via `ffmpeg` + vision handling).
- Audio: controlled by `audio_policy` (native when supported; otherwise optional speech-to-text via `abstractvoice`).

See **[Media Handling](media-handling-system.md)**, **[Vision Capabilities](vision-capabilities.md)**, and **[Centralized Config](centralized-config.md)**.

## HTTP API (optional)

If you want an OpenAI-compatible `/v1` gateway, install and run the server:

```bash
pip install "abstractcore[server]"
python -m abstractcore.server.app
```

See **[Server](server.md)**.
