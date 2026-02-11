# AbstractCore: one Python API for every LLM provider (plus optional OpenAI-compatible servers)

Most teams don’t want “an OpenAI wrapper”. They want:
- one interface for **local + cloud models**,
- **streaming** that behaves consistently,
- **tool calling** that works even when providers differ,
- **structured output** so you stop parsing JSON by hand,
- safe, explicit **media handling** (images/audio/video/docs),
- and (sometimes) an **HTTP server** so other apps can call the same stack.

That is what **AbstractCore** is for.

![AbstractCore request flow](assets/abstractcore-flow.svg)

## What you gain

- **Provider portability**: switch providers/models without rewriting your app.
- **Tool normalization**: tool schemas + tool call parsing are consistent across providers.
- **Safety by design**: tools are explicit, and media behavior is policy-driven (no silent changes).
- **A small default install**: heavy features are opt-in via extras (`abstractcore[media]`, `abstractcore[tools]`, …).
- **Two server modes** (optional): a multi-provider gateway and a single-model endpoint (both OpenAI-compatible `/v1`).

## Quick start (library)

### Install

```bash
pip install abstractcore
```

Install only the providers/features you use:

```bash
pip install "abstractcore[openai]"       # OpenAI SDK
pip install "abstractcore[anthropic]"    # Anthropic SDK
pip install "abstractcore[tools]"        # built-in tool bundle
pip install "abstractcore[media]"        # images, PDFs, Office docs
pip install "abstractcore[server]"       # OpenAI-compatible HTTP gateway + endpoint
```

### Your first call

```python
from abstractcore import create_llm

llm = create_llm("ollama", model="qwen3:4b-instruct")   # local
# llm = create_llm("openai", model="gpt-4o-mini")       # cloud (needs OPENAI_API_KEY)

resp = llm.generate("Explain HTTP caching in 3 bullets.")
print(resp.content)
```

## Tools (portable tool calling)

Tools are explicit Python callables, passed to `generate()`:

```python
from abstractcore import create_llm, tool

@tool
def get_weather(city: str) -> str:
    return f"{city}: 22°C and sunny"

llm = create_llm("openai", model="gpt-4o-mini")
resp = llm.generate("Use the tool to answer.", tools=[get_weather])

print(resp.tool_calls)  # pass-through by default; hosts can execute safely
```

If you want a ready-made toolset for agentic scripts, install `abstractcore[tools]` and import from:
`abstractcore.tools.common_tools` (for example `web_search`, `fetch_url`, `skim_url`).

## Structured output (stop parsing JSON)

If your provider supports it (or via safe prompting fallback), you can request typed outputs:

```python
from pydantic import BaseModel
from abstractcore import create_llm

class Answer(BaseModel):
    title: str
    bullets: list[str]

llm = create_llm("openai", model="gpt-4o-mini")
result = llm.generate("Summarize HTTP/3.", response_model=Answer)
print(result.bullets)
```

## Media handling (images/audio/video + documents)

Media is opt-in (install `abstractcore[media]`) and **policy-driven**:

```python
from abstractcore import create_llm

llm = create_llm("anthropic", model="claude-haiku-4-5")
resp = llm.generate("Describe this image.", media=["./chart.png"])
print(resp.content)
```

If your main model is text-only, AbstractCore can be configured to use a **vision fallback**:
an image-capable model captions images and injects short observations into the prompt.

## Optional: run as an OpenAI-compatible server

AbstractCore supports two OpenAI-compatible HTTP modes:

### 1) Gateway server (multi-provider)

```bash
pip install "abstractcore[server]"
python -m abstractcore.server.app --port 8000
```

Then route by model prefix (example):

```bash
curl -sS http://localhost:8000/v1/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model":"ollama/qwen3:4b-instruct","messages":[{"role":"user","content":"hi"}]}'
```

### 2) Endpoint server (single-model)

This is useful when you want “one provider/model per worker” with an OpenAI-compatible `/v1`.

```bash
pip install "abstractcore[server]"
abstractcore-endpoint --help
```

See the dedicated endpoint docs in AbstractCore for configuration details.

## How it fits in AbstractFramework

In the ecosystem, **AbstractCore is the foundation layer**:
- AbstractRuntime emits effects such as `LLM_CALL` / `TOOL_CALLS`.
- Hosts typically handle those effects using AbstractCore so provider differences don’t leak into workflows.

Next up: durability itself — [AbstractRuntime](03-abstractruntime-durable-workflows.md).

---

### Evidence pointers (source of truth)

AbstractCore repo + docs:
- https://github.com/lpalbou/abstractcore
- https://github.com/lpalbou/abstractcore/blob/main/docs/README.md

AbstractCore’s public API is intentionally small; these files are good anchors:
- `abstractcore/core/factory.py` (create_llm)
- `abstractcore/providers/registry.py` (provider ids + defaults)
- `abstractcore/core/types.py` (response types)
- `abstractcore/tools/core.py` (tool decorator)
- `abstractcore/server/app.py` (gateway server)
- `abstractcore/endpoint/app.py` (single-model endpoint)
