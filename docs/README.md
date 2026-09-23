# AbstractCore Documentation Index

This folder contains the **canonical user documentation** for AbstractCore. The codebase is the source of truth; if you spot a mismatch, please open an issue.

AI assistants and documentation indexers can use the repository-level
`llms.txt` and `llms-full.txt` files. MCP clients such as Context7 can also
query the public documentation directly.

## AbstractFramework ecosystem

AbstractCore is one of the core packages of the **AbstractFramework** ecosystem:

- **AbstractFramework (umbrella)**: https://github.com/lpalbou/AbstractFramework
- **AbstractCore (this package)**: unified LLM interface + cross-provider infrastructure (tools, streaming, structured output, media policies)
- **AbstractRuntime**: durable tool/effect execution, workflows, and state persistence (recommended runtime for executing `response.tool_calls`) — https://github.com/lpalbou/abstractruntime

## Start here (recommended reading order)

1. **[Prerequisites](prerequisites.md)** — install/configure providers (Ollama, LMStudio, vLLM, HuggingFace, MLX, OpenAI, Anthropic, OpenRouter, Portkey, …)
2. **[Getting Started](getting-started.md)** — first call (`create_llm`, `generate`), streaming, tools, structured output
3. **[FAQ](faq.md)** — install extras, local servers, common gotchas
4. **[Troubleshooting](troubleshooting.md)** — actionable fixes for common failures
5. **[API (Python)](api.md)** — user-facing map of the public API
6. **[API Reference](api-reference.md)** — complete function/class reference (including events)
7. **[Architecture](architecture.md)** — component ownership, provider boundaries, request lifecycle and native MLX execution

## Core guides

- **[Tool Calling](tool-calling.md)** — native + prompted tools; passthrough vs execution
- **[Tool Syntax Rewriting](tool-syntax-rewriting.md)** — normalize tool-call markup for different runtimes/clients
- **[Web and Document Tools](web-tools.md)** — `fetch_url`, `skim_url`, `web_search`, `skim_websearch`: extraction, result contract, optional JavaScript rendering, and safety
- **[Structured Output](structured-output.md)** — `response_model=...` strategies and limitations
- **[Request and Output](request-output.md)** — canonical `request=` + `output=` shape, structural task inference, and route-inspection basics
- **[Session Management](session.md)** — conversation state, persistence, compaction
- **[Chat Compaction](chat-compaction.md)** — conversation reduction and continuity controls
- **[Async Guide](async-guide.md)** — async generation, streaming and concurrency patterns
- **[Concurrency and Throughput](concurrency.md)** — async versus tensor batching and the historical direct-backend MLX profile
- **[Prompt Caching](prompt-caching.md)** — `prompt_cache_key`, KV/prefix caches, persistence, durable memory bloc bindings, and the measured cold-vs-warm prefill matrix for the MLX, transformers and GGUF lanes
- **[Generation Parameters](generation-parameters.md)** — unified parameter vocabulary, default hierarchy, caller overrides, and provider quirks
- **[Speculative Decoding](speculative-decoding.md)** — native MTP controls, Qwen3.8-Flash-Next Q4 on the in-process MLX provider, selectable draft depth, prefix caching, and measured tradeoffs
- **[Native MLX Runtime](native-mlx-runtime.md)** — concurrent native requests, target continuous batching versus MTP cohorts, shared ownership, cancellation, bounded RAM/SSD prefix caches, and HTTP serving
- **[Native MLX Benchmarks](native-mlx-benchmarks.md)** — M5 Max 128-GiB measurements for Qwen3.8-27B and Flash-Next versus oMLX: actual MTP, concurrent requests, vision, memory and workload-dependent limits
- **[Reasoning Control](reasoning-control.md)** — the unified `thinking=` parameter, what each provider sends for on/off and effort levels, model requirements, and how to verify a request took effect
- **[HuggingFace Model Compatibility](huggingface-model-compatibility.md)** — Transformers/GGUF loading rules, quantized checkpoint caveats, and trusted proof targets
- **[Memory Blocs](memory-blocs.md)** — persistent extracted text snapshots + per-model KV artifacts
- **[Stopping a Generation and Ejecting a Model](generation-cancel.md)** — host `cancel_event` per lane (MLX, transformers, GGUF, LM Studio, Ollama, llama.cpp server / OpenAI-compatible), severed HTTP requests, what cannot be interrupted, and eject safety (in-flight calls stopped first, `load_model`, reload on demand)
- **[Memory and Model Residency](memory-management.md)** — host memory snapshot (`get_memory_snapshot`), host-wide resident-model sweep (`sweep_loaded_models`), per-provider loaded-model listings, what `unload_model()` frees (weights + session caches) and how to verify it, gateway model locks (`/acore/models/lock`), and context calibration/estimation (`estimate_context_fit`)
- **Model/architecture registries (source of truth)** — `abstractcore/assets/model_capabilities.json`, `abstractcore/assets/model_capabilities.schema.json`, and `abstractcore/assets/architecture_formats.json` (see `abstractcore/assets/README.md`)
- **[Local Models](models.md)** — browse the download catalog with fit verdicts for this machine (`abstractcore models catalog|search`), list installed models with sizes, download and delete weights, follow jobs
- **[Local Engines](engines.md)** — detect Ollama / LM Studio / MLX / llama.cpp / vLLM / transformers, the exact install command per OS, `abstractcore engines install`, and the install safety policy
- **[Centralized Config](centralized-config.md)** — config file, config CLI (`abstractcore --config`), and capability route defaults (`input.*`, `output.*`, `embedding.*`, `rerank.*`)
- **[Data-Home Registry](data-registry.md)** — machine-level registry of framework data directories (model/prompt caches, runs, sessions, logs, entity homes) with owner-declared safe-purge verbs
- **[Events](events.md)** and **[Structured Logging](structured-logging.md)** — observability hooks
- **[Interaction Tracing](interaction-tracing.md)** — record prompts/responses/usage for debugging
- **[Capabilities](capabilities.md)** — what AbstractCore can and cannot do
- **[Fallbacks](fallbacks.md)** — explicit fallback behavior and compatibility boundaries
- **[Model Compatibility and Issue Reporting](known_issues.md)** — provider/artifact checks and useful issue reports
- **[Practical Examples](examples.md)** — end-to-end Python recipes
- **[Framework Comparison](comparison.md)** — AbstractCore and AbstractFramework alongside other LLM libraries
- **Capability plugins (voice/audio/vision/music/scene3d/camera)** — optional deterministic outputs via `llm.voice/llm.audio/llm.vision/llm.music/llm.scene3d/llm.camera`, plus shared provider/model discovery (see `capabilities.md` and `server.md`)

## Media, embeddings, and MCP (optional subsystems)

- **[Media Handling System](media-handling-system.md)** — images/audio/video + documents (policies + fallbacks)
- **[Vision Capabilities](vision-capabilities.md)** — image/video input, native MLX image input, vision fallback, and how this differs from generative vision
- **[Glyph Visual-Text Compression](glyphs.md)** — optional vision-based document compression (experimental)
- **[Vision Compression](vision-compression.md)** — rendering, configuration and quality tradeoffs for text-as-image workflows
- **[Embeddings](embeddings.md)** — `EmbeddingManager` and local embedding models (opt-in)
- **[MCP (Model Context Protocol)](mcp.md)** — consume MCP tool servers (HTTP/stdio) as tool sources

## Server (optional HTTP API)

- **[Server](server.md)** — OpenAI-compatible `/v1` gateway (install `pip install "abstractcore[server]"`; run `abstractcore serve`)
- **[Endpoint](endpoint.md)** — single-model OpenAI-compatible `/v1` endpoint (install `pip install "abstractcore[server]"`; run `abstractcore-endpoint`)

## Built-in CLI apps

These are convenience CLIs built on top of the core library:

- **[Interactive Chat CLI](acore-cli.md)** — `abstractcore-chat` usage and controls
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
