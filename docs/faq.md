# FAQ

## What do I get with `pip install abstractcore`?

The default install is intentionally lightweight. It includes the core API (`create_llm`, `BasicSession`, tool definitions, structured output plumbing) and uses only small dependencies (`pydantic`, `httpx`).

Anything heavy (provider SDKs, torch/transformers, PDF parsing, embeddings models, web scraping deps, the HTTP server) is behind install extras. See [Getting Started](getting-started.md) and [Prerequisites](prerequisites.md).

## Which extra do I need for my provider?

- OpenAI: `pip install "abstractcore[openai]"`
- Anthropic: `pip install "abstractcore[anthropic]"`
- HuggingFace (transformers/torch; heavy): `pip install "abstractcore[huggingface]"`
- MLX (Apple Silicon; heavy): `pip install "abstractcore[mlx]"`
- vLLM integration (GPU; heavy): `pip install "abstractcore[vllm]"`

These providers work with the core install (no provider extra): `ollama`, `lmstudio`, `openrouter`, `openai-compatible`.

## How do I combine extras?

```bash
# zsh: keep quotes
pip install "abstractcore[openai,media,tools]"
```

For “turnkey” installs, see `README.md` (`all-apple`, `all-non-mlx`, `all-gpu`).

## Why did my install pull `torch` / take a long time?

You probably installed a heavy extra (most commonly `abstractcore[huggingface]`, `abstractcore[mlx]`, or `abstractcore[all-*]`). The core install (`pip install abstractcore`) does not include torch/transformers.

## What’s the difference between “provider” and “model”?

- **Provider**: a backend adapter (`openai`, `anthropic`, `ollama`, `lmstudio`, …)
- **Model**: a provider-specific model name (for example `gpt-4o-mini` or `qwen3:4b-instruct-2507-q4_K_M`)

```python
from abstractcore import create_llm
llm = create_llm("openai", model="gpt-4o-mini")
```

## How do I connect to a local server (Ollama / LMStudio / vLLM / llama.cpp / LocalAI)?

Use the matching provider and set `base_url` (or the provider’s base-url env var).

Examples:

```python
from abstractcore import create_llm

llm = create_llm("ollama", model="qwen3:4b-instruct-2507-q4_K_M", base_url="http://localhost:11434")
llm = create_llm("lmstudio", model="qwen/qwen3-4b-2507", base_url="http://localhost:1234/v1")
llm = create_llm("vllm", model="Qwen/Qwen3-Coder-30B-A3B-Instruct", base_url="http://localhost:8000/v1")
```

For a generic OpenAI-compatible endpoint, use `openai-compatible`:

```python
llm = create_llm("openai-compatible", model="my-model", base_url="http://localhost:1234/v1")
```

See [Prerequisites](prerequisites.md) for setup details and env var names.

## How do I set API keys and defaults?

You can use environment variables, or persist settings via the config CLI:

```bash
abstractcore --configure
abstractcore --set-api-key openai sk-...
abstractcore --set-api-key anthropic sk-ant-...
abstractcore --status
```

Config is stored in `~/.abstractcore/config/abstractcore.json`. See [Centralized Config](centralized-config.md).

## Why aren’t tools executed automatically?

By default, AbstractCore runs in **pass-through** mode (`execute_tools=False`): it returns tool calls in `resp.tool_calls`, and your host/runtime decides whether/how to execute them.

Automatic execution (`execute_tools=True`) exists but is deprecated for most use cases. See [Tool Calling](tool-calling.md).

## How do I preserve tool-call markup in `response.content` for agentic CLIs?

Use tool-call syntax rewriting:

- Python: pass `tool_call_tags=...` to `generate()` / `agenerate()`
- Server: set `agent_format` in requests

See [Tool Syntax Rewriting](tool-syntax-rewriting.md).

## How do I get structured output (typed objects) instead of parsing JSON?

Pass a Pydantic model via `response_model=...`:

```python
from pydantic import BaseModel
from abstractcore import create_llm

class Answer(BaseModel):
    title: str
    bullets: list[str]

llm = create_llm("openai", model="gpt-4o-mini")
result = llm.generate("Summarize HTTP/3 in 3 bullets.", response_model=Answer)
```

See [Structured Output](structured-output.md).

## Why does structured output retry or fail validation?

Structured output is validated against your schema. If validation fails, AbstractCore retries with feedback (up to the configured retry limit). Common fixes:

- simplify schemas (fewer nested structures; fewer strict constraints)
- tighten prompts (be explicit about allowed values and ranges)
- increase timeouts for slow backends

See [Structured Output](structured-output.md) and [Troubleshooting](troubleshooting.md).

## Why do PDFs / Office docs / images not work?

Those require the media extra:

```bash
pip install "abstractcore[media]"
```

Then pass `media=[...]` to `generate()` or use the media pipeline. See [Media Handling](media-handling-system.md).

## What are “glyphs” and what do they require?

Glyph visual-text compression is an optional feature for long documents. Install:

- `pip install "abstractcore[compression]"` (renderer)
- plus `pip install "abstractcore[media]"` if you want PDF extraction support

See [Glyph Visual-Text Compression](glyphs.md).

## How do I use embeddings?

Embeddings are opt-in:

```bash
pip install "abstractcore[embeddings]"
```

Then import from the embeddings module:

```python
from abstractcore.embeddings import EmbeddingManager
```

See [Embeddings](embeddings.md).

## Do I need the HTTP server?

No. The server is optional and is mainly for:

- exposing one OpenAI-compatible `/v1` endpoint that can route to multiple providers/models
- integrating with OpenAI-compatible clients and agentic CLIs

Install and run:

```bash
pip install "abstractcore[server]"
python -m abstractcore.server.app
```

See [Server](server.md).

## Where are logs and traces?

- Logging (console/file) is configured via the config CLI and config file. See [Structured Logging](structured-logging.md).
- Interaction tracing is opt-in (`enable_tracing=True`). See [Interaction Tracing](interaction-tracing.md).

## I’m getting HTTP timeouts. What should I change?

- Per-provider: pass `timeout=...` to `create_llm(...)` (`timeout=None` means unlimited).
- Process-wide default: set `abstractcore --set-default-timeout 0` (0 = unlimited), or set a larger value.
- Some CLI apps have their own `--timeout` flags; run `--help` for the exact behavior.

See [Troubleshooting](troubleshooting.md) and [Centralized Config](centralized-config.md).

## HuggingFace won’t download models — why?

The HuggingFace provider respects AbstractCore’s offline-first settings. If you want HuggingFace to fetch from the Hub, update `~/.abstractcore/config/abstractcore.json`:

- set `"offline_first": false`
- set `"force_local_files_only": false`

Restart your Python process after changing this (the provider reads these settings at import time).

## Is AbstractCore a full agent/RAG framework?

AbstractCore focuses on provider abstraction + infrastructure (tools, structured output, media handling, tracing). It does not ship a full RAG pipeline or multi-step agent orchestration. See [Capabilities](capabilities.md).
