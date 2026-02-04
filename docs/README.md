# AbstractCore Documentation Index

This folder contains the **canonical user documentation** for AbstractCore. The codebase is the source of truth; if you spot a mismatch, please open an issue.

## Start here (recommended reading order)

1. **[Prerequisites](prerequisites.md)** — install/configure providers (OpenAI, Anthropic, Ollama, LMStudio, …)
2. **[Getting Started](getting-started.md)** — first call (`create_llm`, `generate`), streaming, tools, structured output
3. **[FAQ](faq.md)** — install extras, local servers, common gotchas
4. **[Troubleshooting](troubleshooting.md)** — actionable fixes for common failures
5. **[API (Python)](api.md)** — user-facing map of the public API
6. **[API Reference](api-reference.md)** — complete function/class reference (including events)

## Core guides

- **[Tool Calling](tool-calling.md)** — native + prompted tools; passthrough vs execution
- **[Tool Syntax Rewriting](tool-syntax-rewriting.md)** — normalize tool-call markup for different runtimes/clients
- **[Structured Output](structured-output.md)** — `response_model=...` strategies and limitations
- **[Session Management](session.md)** — conversation state, persistence, compaction
- **[Generation Parameters](generation-parameters.md)** — unified parameter vocabulary + provider quirks
- **[Centralized Config](centralized-config.md)** — config file + config CLI (`abstractcore --configure`)
- **[Events](events.md)** and **[Structured Logging](structured-logging.md)** — observability hooks
- **[Interaction Tracing](interaction-tracing.md)** — record prompts/responses/usage for debugging
- **[Capabilities](capabilities.md)** — what AbstractCore can and cannot do
- **Capability plugins (voice/audio/vision)** — optional deterministic outputs via `llm.voice/llm.audio/llm.vision` (see `capabilities.md` and `server.md`)

## Media, embeddings, and MCP (optional subsystems)

- **[Media Handling System](media-handling-system.md)** — images/audio/video + documents (policies + fallbacks)
- **[Vision Capabilities](vision-capabilities.md)** — image/video input, vision fallback, and how this differs from generative vision
- **[Glyph Visual-Text Compression](glyphs.md)** — optional vision-based document compression (experimental)
- **[Embeddings](embeddings.md)** — `EmbeddingManager` and local embedding models (opt-in)
- **[MCP (Model Context Protocol)](mcp.md)** — consume MCP tool servers (HTTP/stdio) as tool sources

## Server (optional HTTP API)

- **[Server](server.md)** — OpenAI-compatible `/v1` gateway (install `pip install "abstractcore[server]"`)

## Built-in CLI apps

These are convenience CLIs built on top of the core library:

- **[Summarizer](apps/basic-summarizer.md)**
- **[Extractor](apps/basic-extractor.md)**
- **[Judge](apps/basic-judge.md)**
- **[Intent](apps/basic-intent.md)**
- **[DeepSearch](apps/basic-deepsearch.md)**

## Project docs

- **[Changelog](../CHANGELOG.md)** — release notes and upgrade guidance
- **[Contributing](../CONTRIBUTING.md)** — dev setup and PR guidelines
- **[Security](../SECURITY.md)** — responsible vulnerability reporting
- **[Acknowledgements](../ACKNOWLEDGEMENTS.md)** — upstream projects and communities
- **[License](../LICENSE)** — MIT license text

## Docs layout (what’s where)

`docs/` is mostly a flat set of guides plus a few subfolders:

- `docs/apps/` — CLI app guides
- `docs/known_bugs/` — focused notes on known issues (when present)
- `docs/archive/` — superseded/historical docs (see `docs/archive/README.md`)
- `docs/backlog/` — planning notes (see `docs/backlog/README.md`)
- `docs/reports/` — non-authoritative engineering notes (see `docs/reports/README.md`)
- `docs/research/` — non-authoritative experiments (see `docs/research/README.md`)

**Key distinction:**
- `api.md` = API overview (how to use the public API)
- `api-reference.md` = full Python API reference
- `server.md` = HTTP server endpoints and deployment
