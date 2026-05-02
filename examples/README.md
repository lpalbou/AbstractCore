# Examples

This folder is a runnable “tour” of AbstractCore: providers, sessions, tools, prompt controls, caching, and common application patterns (RAG, structured output, tracing).

The goal is not just “here’s how to run it”, but:
- what each example demonstrates,
- how it works (which AbstractCore abstraction is doing what),
- and what you should take away when building your own app.

## How AbstractCore Examples Are Structured

Most examples follow the same shape:

1) **Pick a provider** (`create_llm(...)`), which normalizes APIs across local and hosted backends.  
2) **Use a session** (`BasicSession` or `CachedSession`) to manage system prompt, tools, and conversation state.  
3) **Call `generate(...)` / `agenerate(...)`** (optionally streaming), and observe stats/events.

In practice, switching providers usually means changing only:
- `provider=` (e.g. `ollama`, `lmstudio`, `mlx`, `huggingface`)
- `model=...`

## How To Run

If you cloned the repo, install AbstractCore in editable mode first (so `import abstractcore` works from the examples):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Then run from the repo root:
```bash
python examples/<topic>/<script>.py
```

Typical local backends:
- Ollama: `ollama serve` (default `http://localhost:11434`)
- LM Studio: start the OpenAI-compatible server (default `http://localhost:1234/v1`)

Optional extras:
- Embeddings: `pip install "abstractcore[embeddings]"`
- Media/PDF: `pip install "abstractcore[media]"`
- MLX local models: `pip install "abstractcore[mlx]"`
- HuggingFace transformers/GGUF: `pip install "abstractcore[huggingface]"`

## Downloading Your First Local Model (HF / GGUF / MLX)

AbstractCore is offline-first by default, so local providers typically **do not auto-download** weights.

- For Hugging Face / GGUF / MLX model downloads, see `examples/models/README.md`.
- For Ollama/LM Studio, models are managed by the server/app (pull/load them there first).

## Recommended Learning Path

If you want a guided “read + run” walkthrough, start with `examples/learning_path/` (6 parts, heavily commented).

If you prefer “pick a feature and try it”, start here:
- Prompt caching + attachments: `examples/prompt_caching/prompt_cache_repl_demo.py`
- Tool calling basics: `examples/tools/tool_usage_basic.py`
- Reasoning/thinking knobs (local providers): `examples/reasoning/qwen_thinking_repl.py`

## Index (What Each Folder Teaches)

Each topic folder has its own `README.md` that explains the *mechanics* and *takeaways* behind the scripts.

- `examples/learning_path/` — the guided 6-part walkthrough (start at `01_basic_generation.py`).
- `examples/cli/` — building an async CLI around sessions, streaming, events, and tool progress.
- `examples/prompt_caching/` — prompt-cache “box boundaries” (system/tools/history/files) via `CachedSession`.
- `examples/sessions/` — `BasicSession` multi-turn chat + transcript persistence.
- `examples/reasoning/` — how AbstractCore maps `thinking=` across providers and how to verify what the backend returned.
- `examples/tools/` — tool schema, tool routing, streaming tool calls, and safety patterns.
- `examples/embeddings/` — `EmbeddingManager`, similarity search, clustering, and RAG-friendly retrieval patterns.
- `examples/media/` — media detection + processing, plus “glyph” compression examples.
- `examples/models/` — model discovery/filtering patterns (server `/v1/models`) and practical model selection helpers.
- `examples/performance/` — benchmarking knobs and tradeoffs (streaming vs non-streaming, MLX concurrency).
- `examples/observability/` — tracing/telemetry: what to log and how to export it.
- `examples/rag/` — a compact end-to-end RAG pipeline example.
- `examples/structured_output/` — structured output + validation patterns.

## What To Look For When Reading The Code

- “Where is the system prompt/tools/history stored?” → in the `Session` abstraction.
- “How do tool calls get surfaced?” → as provider messages + events, normalized by AbstractCore.
- “What does caching cache?” → provider-specific KV/prefix reuse, controlled by `CachedSession` and provider capabilities.

## Notes

- Some scripts require a local model backend and will error if the server/model isn’t running.
- Scripts labeled benchmark/dev-oriented are intentionally more “instrumented” than “minimal”.
