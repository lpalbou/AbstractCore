# Planned: RerankManager (query–document reranking)

## Metadata
- Created: 2026-05-22
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: ADR 0002, ADR 0003
- ADR impact: Needs new ADR

## Context
AbstractCore provides:
- text generation via `create_llm(...)` and providers (`AbstractCoreInterface.generate(...)`);
- embeddings via `abstractcore.embeddings.EmbeddingManager` (local and server-backed);
- capability plugins for voice/audio/vision/music.

In modern RAG / agentic retrieval pipelines, a reranker is a distinct component:
- stage 1: cheap candidate generation (BM25, dense embeddings, or hybrid fusion);
- stage 2: cross-encoder reranking of the top-N candidates (often 30–200) to sharpen ordering and
  improve top-k precision.

As of May 2026, production reranking commonly uses either:
- locally-run cross-encoders / decoder-only rerankers (in-process, or served locally behind an HTTP
  rerank endpoint), or
- hosted rerank APIs (often exposed as a Jina-style `POST /v1/rerank` endpoint, including OpenRouter and
  Portkey gateways that route to Cohere/Voyage/Jina-style rerankers).

AbstractCore currently has no first-class reranking abstraction. Users must assemble their own
reranking layer, which breaks the “one abstraction layer per task” goal.

## Current code reality
- No `rerank(...)` concept exists in Core or providers; repo search shows no rerank implementation
  besides a mention in learning examples.
- Embeddings are implemented as a separate contract via `abstractcore/embeddings/manager.py`.
- Model filtering only supports output types `text` and `embeddings` via
  `abstractcore/providers/model_capabilities.py` and `/v1/models` in `abstractcore/server/app.py`.
- `abstractcore/config/capability_defaults.py` already reserves a `rerank.text` route spec with
  package hint `"future reranker manager"`, so config vocabulary is ahead of execution support.
- Text embedding provider discovery now exposes transport/base-URL metadata instead of static
  `local`/`remote` provider identity. The rerank provider catalog should follow the same rule:
  provider/model/base_url define a configured route, and endpoint locality is derived from
  `base_url`, not from provider id.
- `planned/0803_image_embedding_manager_and_multimodal_embeddings.md` and
  `proposed/0804_huggingface_embedding_provider_boundary.md` are adjacent boundary items. Re-check
  them before implementing rerank provider/catalog metadata.
- The public output-selector contract (`abstractcore/core/output_specs.py`) covers text, image,
  voice, music, and transcription; it does **not** include `embeddings` or `rerank` as selectors.
- `MultimodalGenerateResponse` (`abstractcore/core/multimodal_generation.py`) is oriented around
  returning generated media bytes (`GeneratedItem.data: bytes`).
- Architecture/model typing only includes `ModelType.EMBEDDING` for non-generation models; there is
  no reranker model type (`abstractcore/architectures/enums.py`).
- The only explicit reranking mention today is an educational bullet:
  `examples/learning_path/06_production_patterns.py` (“Cross-Encoder: Re-rank retrieved documents”).

## Problem
AbstractCore’s “provider/model abstraction layer” stops at embeddings + generation. Reranking is a
separate, high-leverage primitive for retrieval quality, but today it is:
- not discoverable (no `/v1/models?output_type=...` support);
- route-configurable only as a generic `rerank.text` capability default; there is no
  `RerankManager` execution path that consumes it yet;
- not testable/benchmarkable through AbstractCore’s existing patterns;
- not available as a single stable API for local + hosted rerankers.

## What we want to do
Add a first-class `RerankManager` that provides a unified reranking API across:
- local rerankers (cross-encoders running in-process), and
- hosted rerank APIs (provider-backed HTTP).

This should mirror `EmbeddingManager`’s ergonomics (defaults + caching + strictness + provider
kwargs), without forcing reranking into `generate(...)` semantics.

## Why
- Reranking is now “standard RAG plumbing” for correctness and top-k precision.
- Without an abstraction, every downstream app re-implements batching, truncation, and provider
  request/response parsing.
- A shared `RerankManager` can standardize: result shapes, truncation behavior, error handling,
  and validation.

## Requirements
### Public API
- Provide a stable Python API (sync first; async optional) similar to:
  - `RerankManager.rerank(query, documents, *, top_n=None, instruction=None, **kwargs) -> RerankResponse`
- Support documents as either:
  - `List[str]`, or
  - `List[{"text": str, ...}]` (metadata preserved and returned).
- Return ordered results with:
  - stable `index` into the input list;
  - `relevance_score` (float; provider-specific scale, treated as non-calibrated by default);
  - optional `document` echo (for hosted APIs that support it).
  - metadata about truncation/chunking decisions (when enabled).

### Contract: normalized rerank response (single source of truth)
AbstractCore should expose **one** normalized response contract regardless of backend:
- **Request inputs (normalized)**:
  - `query: str`
  - `documents: List[str] | List[{"text": str, ...}]`
  - `top_n: Optional[int]`
  - `instruction: Optional[str]` (a.k.a. `instruct` for some backends)
  - `return_documents: bool` (default: True in Python APIs; can be False for bandwidth)
  - optional truncation/chunking controls (see below)
- **Response (normalized)**:
  - `results: List[{index:int, relevance_score:float, document?:{text:str, ...}}]` sorted descending
  - `model: str` (as provided to the backend; include provider prefix when that is the public key)
  - `provider: Optional[str]` when available (e.g., OpenRouter/Portkey include it)
  - `id: Optional[str]` when available
  - `usage: Optional[Dict[str, Any]]` when available
  - `meta: Dict[str, Any]` for:
    - input trimming/chunking decisions;
    - score normalization policy;
    - backend type (in-process vs HTTP vs LLM-based).

`RerankManager` may offer `to_dict()` helpers for compatibility, but the dataclass contract above is
the core abstraction boundary.

### Provider limits and performance guards
- Default guidance should prefer reranking a bounded candidate set (typically tens to low hundreds,
  not thousands).
- Enforce provider hard limits defensively:
  - Many providers hard-cap documents (often **1,000** per request; some allow more). Gateways
    (OpenRouter/Portkey) may route to multiple upstreams, so the true limit can be **model- and
    upstream-dependent**, not just “the gateway limit”.
  - Cohere’s rerank endpoint throws an error when the request would exceed their effective
    document/chunk budget (`num_docs * max_chunks_per_doc > 10,000`). When callers exceed hard
    limits, either:
    - `strict=True`: raise with an actionable error; or
    - `strict=False`: trim with explicit metadata (never silent).
  - When chunking is enabled, ensure `num_documents * max_chunks_per_doc` cannot explode; fail fast
  or trim with an explicit error/metadata record (avoid “surprise 10,000+ chunks” requests).

### Evidence-based pitfalls (May 2026 snapshot)
The goal of this subsection is to encode “don’t accidentally build the wrong thing” guidance that
is easy to forget when integrating rerankers.

**Paper: _Drowning in Documents: Consequences of Scaling Reranker Inference_ (Jacob et al., 2024;
arXiv v2 July 2025)**:
- **Result**: When rerankers are evaluated end-to-end (not only as a top-100 rescorer), they show
  **diminishing returns** and can **degrade retrieval quality** beyond a certain reranking depth.
  They also report that rerankers can assign high scores to documents with little/no lexical or
  semantic overlap with the query.
- **Logic**: Scaling reranking depth increases the volume of hard negatives/noise. Cross-encoders
  appear less robust to this noise than the “rerank more = better” intuition suggests, so recall
  can drop as you feed the reranker progressively more candidates.
- **Best practices for AbstractCore design**:
  - Make reranking depth (`top_n`) an explicit, required tuning knob in docs/examples; do not imply
    that “rerank more” is a safe default.
  - Add guardrails: default caps, explicit trimming behavior, and metadata that records when inputs
    were trimmed or chunked.
  - Encourage evaluation across multiple reranking depths for a given retriever+rereanker pairing.

**Paper: _Language Model Re-rankers are Fooled by Lexical Similarities_ (Hagström et al., 2025;
arXiv v2 June 2025)**:
- **Result**: Across three datasets (NQ, LitQA2, DRUID) and six rerankers, they find cases where LM
  rerankers **struggle to outperform BM25** on DRUID, and they attribute many failures to
  **lexical distractors** and **missing document context**. They introduce a separation metric
  (DS) based on BM25 scores to detect when the most BM25-similar non-gold passage outcompetes the
  most BM25-similar gold passage.
- **Logic**: Rerankers can over-index on lexical similarity (or be misled when the gold evidence is
  lexically dissimilar), and they can fail when the passage alone lacks identifying context (e.g.,
  title/source).
- **Best practices for AbstractCore design**:
  - Keep BM25/hybrid retrieval as a first-class “stage 1” recommendation in docs (reranking does
    not obsolete lexical retrieval).
  - Support attaching and optionally prepending **document context** (title/source/url) to the
    text passed into the reranker, with explicit metadata showing what was prepended.
  - Treat `instruction` as best-effort and metadata-visible; prompt tweaks can help in some tasks
    but do not generalize across domains.

### Backends (v0)
AbstractCore should support **three** concrete reranking backend types that cover the real
deployment surface:

1. **HTTP Rerank (Jina-style)** — `POST /v1/rerank`
   - This is the preferred “remote or locally-served” path because it standardizes the wire shape.
   - Must support these provider IDs (aligned with existing provider registry/config):
     - `openrouter` (OpenRouter implements `POST /api/v1/rerank` and returns `results[]` with
       `{index, relevance_score, document}`).
     - `portkey` (Portkey implements `POST /v1/rerank` and provides a unified gateway to Cohere,
       Voyage, Jina, Pinecone, Bedrock, Azure AI, etc).
     - `openai-compatible` (generic base_url; covers self-hosted vLLM/SGLang/llama.cpp deployments).
   - Must normalize **response variants**:
     - object-wrapped responses (`{"results":[...], "model":..., ...}` like OpenRouter/Portkey);
     - list-only responses (some local servers return `[{score/index/...}, ...]`).
   - Must support `top_n`, `return_documents`, and best-effort `instruction` mapping:
     - `instruction` → `instruct` (if backend supports it) OR documented query-prefixing fallback.

2. **In-process CrossEncoder** — local Python
   - Backed by `sentence-transformers` `CrossEncoder`, because:
     - it supports both classic SequenceClassification cross-encoders (e.g. BGE / mixedbread / GTE),
       and decoder-only rerankers (e.g. Qwen3 reranker) via model-specific templates;
     - it supports ONNX/OpenVINO backends for CPU optimization where supported.
   - Must support batching + device selection and must avoid repeated model loads (single instance).
   - Do not assume this backend is the same thing as
     `abstractcore.providers.huggingface_provider.HuggingFaceProvider`. Current text embeddings
     have the same ambiguity: `EmbeddingManager` runs SentenceTransformers directly while the
     provider registry also advertises `huggingface` embedding support. See
     `proposed/0804_huggingface_embedding_provider_boundary.md`.

3. **LLM-based reranking (fallback)** — any `create_llm(...)` provider
   - Needed to satisfy “OpenAI must have *some* reranking option” even without a native rerank API.
   - Implement as an explicit backend (opt-in, not default) that:
     - uses structured outputs to produce `{index, relevance_score}` for each document;
     - defaults to listwise reranking over a bounded `top_n` and requires explicit caps/truncation;
     - records `meta.backend="llm_rerank"` so users understand it is not a dedicated reranker model.

### Local model defaults (v0)
Provide a curated set of “known-good” local reranker models similar to `abstractcore/embeddings/models.py`.

**Curated v0 local defaults (pick 3; user requirement)**:
- `Qwen/Qwen3-Reranker-0.6B`
- `Qwen/Qwen3-Reranker-4B`
- `Qwen/Qwen3-Reranker-8B`

These models expose:
- instruction-aware reranking and long context (~32k) per their model cards;
- **raw (non-calibrated) scores** by default in `sentence-transformers`, with a documented sigmoid option for 0–1
  scores; do not assume cross-provider score comparability.
- practical “instruction/prompt” control via `sentence-transformers` `CrossEncoder(..., prompts=..., default_prompt_name=...)`.

**Optional additional curated picks** (not required for v0, but useful):
- `BAAI/bge-reranker-v2-m3` (widely deployed lightweight multilingual cross-encoder)
- `mixedbread-ai/mxbai-rerank-large-v1` (strong open reranker baseline)
- `Alibaba-NLP/gte-reranker-modernbert-base` (compact CPU-friendly reranker with ONNX potential)

### Truncation / chunking
- Implement explicit, user-visible truncation and chunking rules:
  - Never silently truncate without recording it in returned metadata.
  - Allow callers to opt into `truncation=True` (provider-style) or provide their own chunked
    documents (recommended when callers need deterministic boundaries).
  - Provide a doc-level aggregation strategy when chunking is enabled (default `max` across
    chunks; configurable). This mirrors common hosted-rerank behavior (score each chunk, then take
    the max per document).
  - Add explicit knobs to prevent compute blow-ups:
    - `max_chunks_per_doc` (hard cap; default small, e.g. 1–4 depending on backend);
    - `max_total_chunks` (cap across the whole request);
    - `max_tokens_per_doc`/`max_chars_per_doc` (best-effort, backend-dependent).

### Instruction-following reranking (May 2026 baseline)
- Expose an optional `instruction` parameter in the public API, but implement it in a provider-safe
  way:
  - some providers may support a native instruction field;
  - others (e.g., Voyage’s instruction-following rerankers) can be used by prepending instructions
    to the query string. This must be explicit in metadata so callers understand what was sent.
  - local Qwen3 rerankers support instruction prompting via `sentence-transformers` CrossEncoder
    prompts; treat this as the canonical local “instruction” mechanism.

### Safety / privacy
- Treat rerank inputs as sensitive user data:
  - caching must be opt-in and clearly documented (memory-only default; no disk persistence by
    default for query–document pairs).
  - ensure logs use preview/truncation helpers (similar to existing `preview_text` patterns).

### Integration
- Add a new model/output capability taxonomy for rerankers:
  - `ModelOutputCapability.RERANK`
  - update `abstractcore/providers/model_capabilities.py` so rerankers can be discovered via
    `/v1/models?output_type=rerank` when the provider supports it.
- Add a Core-facing `supports_rerank(model_name)` helper (similar to `supports_embeddings`) that
  checks JSON capabilities or name patterns.
- Add docs page(s) showing “retrieve then rerank” usage and the “rerank depth is a tuning knob”
  pitfalls from the evidence above.

### API surface: unified `generate(..., output=...)` vs task-specific managers
AbstractCore already has a *unified* `generate(..., output=...)` surface for **multimodal generation**
(text + image/voice/music, plus transcription). This is great UX because it feels like “one call”.

Embeddings + rerank *can* plausibly fit a unified “one call” UX, and the current output-selector
infrastructure is already a good foundation. The main tension is not that it is impossible, but
that it must be done **without making the contract confusing** or weakening type clarity:
- **Stateless vs stateful**: it should remain true that *sessions* are for chat/history, while
  embeddings/rerank are stateless utilities. A unified entrypoint should therefore be **one-shot**
  only (not `session.generate(...)`).
- **Batch-shaped inputs**: rerank/embeddings naturally accept lists. This is solvable by evolving
  the unified API to accept a richer request object (or `prompt` as a union type), but that is a
  deliberate interface change and should not be smuggled into a rerank integration.
- **Return typing**: `MultimodalGenerateResponse` today is optimized around media bytes
  (`GeneratedItem.data: bytes`). We can extend this (e.g., allow `GeneratedItem.data: Any` or add
  a parallel `computations: Dict[str, Any]`) but we should do it intentionally so consumers do not
  need type guesses.

**Recommendation (v0 for this item)**:
- Keep **task-specific managers** as the canonical, typed APIs:
  - `EmbeddingManager.embed(...)` / `embed_batch(...)`
  - `RerankManager.rerank(...)`
- Design `RerankManager`’s request/response types to be JSON-serializable and stable, so a future
  unified entrypoint can wrap/return them without inventing a second contract.

**Follow-up (nice UX, explicitly out-of-scope here unless already planned elsewhere)**:
- Evaluate one unified entrypoint for *stateless compute* tasks:
  - either extend output specs (`abstractcore/core/output_specs.py`) to include `embeddings` and
    `rerank` modalities and route them through managers, **or**
  - introduce `abstractcore.compute(task=..., ...)` as a sibling to `generate(...)`.
  The decision should be user-experience driven and include a typing story (Python + server JSON).

### Model registry & asset placement (answering “where do we reference rerank models?”)
**Recommendation (v0)**:
- Put curated local reranker “favorites” in `abstractcore/rerank/models.py` (parallel to `abstractcore/embeddings/models.py`).
- Treat remote/hosted rerank model names as **opaque strings** (do not hardcode vendor catalogs).
- Add provider/catalog metadata that describes transport and execution ownership, not topology:
  - examples: `cross_encoder_inprocess`, `rerank_http`, `llm_rerank`;
  - include `base_url_configurable`, `base_url_env_vars`, and auth hints for HTTP routes;
  - avoid `local`/`remote` booleans. A self-hosted Jina/vLLM rerank endpoint and a hosted gateway
    may use the same HTTP wire contract, and only the configured `base_url` tells the endpoint
    location.
- For `/v1/models?output_type=rerank` discovery, use **best-effort classification**:
  - extend `ModelOutputCapability` with `RERANK`;
  - classify models as `RERANK` when the model id contains `rerank`/`reranker` (case-insensitive);
  - optionally (later) add explicit `model_type: "rerank"` entries in `abstractcore/assets/model_capabilities.json` for
    a *small* curated set to improve precision.

`abstractcore/assets/architecture_formats.json` is not directly useful for dedicated rerankers; it only matters for the
optional **LLM-based rerank backend** because that backend uses `create_llm(...)`.

## Suggested implementation
1. Create a new package `abstractcore/rerank/`:
   - `types.py`: `RerankDocument`, `RerankResult`, `RerankResponse`.
   - `manager.py`: `RerankManager` with provider routing, batching, and optional caching.
   - `models.py`: curated local reranker registry (Qwen3 rerankers required; others optional).
   - `http_client.py` (or `providers/`): HTTP rerank transport (`POST /v1/rerank`) + response normalization.
2. Extend config-manager defaults (mirror embeddings):
   - consume the existing generic `rerank.text` capability default first;
   - add a dedicated `RerankConfig` only if compatibility or CLI ergonomics require a legacy-style
     mirror of embeddings;
   - add an env override for strict-mode (e.g. `ABSTRACTCORE_RERANK_STRICT=1`);
   - document config keys in `abstractcore/config/README.md`.
3. Keep provider selection aligned with `EmbeddingManager`:
   - `provider` + `model` defaults from config-manager if present.
   - `provider_kwargs` passthrough for keys/base URLs.
   - Treat local CrossEncoder execution as a dedicated rerank backend unless/until
     `HuggingFaceProvider` grows a provider-owned rerank/embed contract. Do not overload the
     `huggingface` provider id without resolving `proposed/0804_huggingface_embedding_provider_boundary.md`.
4. Implement HTTP Rerank transport:
   - Primary: OpenRouter + Portkey (both expose `POST /v1/rerank`-style endpoints).
   - Generic: `openai-compatible` base_url for local serving (vLLM/SGLang/llama.cpp).
   - Implement endpoint probing (`/v1/rerank`, `/rerank`, `/v2/rerank`) with caching; record which
     route was used in response metadata.
5. Implement in-process CrossEncoder backend:
   - Use `sentence-transformers` `CrossEncoder` and expose `backend="torch|onnx|openvino"` as an
     option (when supported by the model).
   - Default local model should be one of the required Qwen3 rerankers, with a clear size-based
     recommendation.
6. Implement LLM-based rerank backend (optional) using `create_llm(...)`:
   - Use structured outputs to return results, with strict truncation/caps and provenance metadata.
7. Add strict-mode semantics consistent with `EmbeddingManager`:
   - `strict=True`: raise on provider/model failures.
   - `strict=False` (default): return empty results plus error metadata.
8. Add tests:
   - unit tests with stub provider responses (no network);
   - local backend tests gated behind optional extras (or mocked model) to avoid heavy CI loads.
9. Update docs and examples:
   - new “Reranking” section mirroring the embeddings docs structure.
   - add a small example that: retrieves via embeddings -> reranks -> builds prompt context.

## Scope
- Implement `RerankManager` (core API + HTTP rerank transport + in-process backend + optional LLM fallback).
- Add result types and unit tests for normalization and error-handling.
- Add docs and one end-to-end example (retrieve → rerank → prompt).
- Add (or plan) a consistent capability taxonomy hook for rerankers.

## Non-goals
- Do not refactor `AbstractCoreInterface.generate(...)` to fully support rerank/embeddings as
  first-class `output=...` selectors in this item (keep it a focused rerank integration).
- Do not implement a full vector DB / BM25 layer inside AbstractCore.
- Do not claim one “best” reranker model without workload-specific evaluation.
- Do not treat relevance scores as calibrated probabilities.

## Dependencies and related tasks
- `abstractcore/embeddings/manager.py` (pattern and config defaults)
- `abstractcore/providers/model_capabilities.py` and `abstractcore/server/app.py` (capability
  filtering and discovery)
- `abstractcore/config/capability_defaults.py` (`rerank.text` already exists as a route vocabulary
  placeholder)
- `planned/0803_image_embedding_manager_and_multimodal_embeddings.md` (retrieval lanes and future
  multimodal reranking interaction)
- `proposed/0804_huggingface_embedding_provider_boundary.md` (avoid repeating provider/backend
  identity ambiguity for local CrossEncoder rerankers)
- ADR 0002: validation and evidence requirements
- ADR 0003: provider/capability/output boundaries
- ADR 0008: provider-owned model residency truth
- Related: `docs/backlog/planned/2026-05-06_consensus-generate.md` (similar “new core primitive”
  design/trace patterns)

## Expected outcomes
- Users can call a single stable API to rerank candidate passages/documents.
- Hosted and local rerankers share the same response shape and truncation metadata.
- AbstractCore’s docs clearly explain “retrieve then rerank” as a supported pattern.
- The server can optionally expose rerank models via discovery filtering, without drift.

## Validation
- Unit tests covering:
  - document normalization (strings vs dicts);
  - provider response normalization (index + score ordering);
  - strict vs non-strict error behavior;
  - truncation/chunking metadata surfaces.
- Documentation example that runs with a mocked provider (no network).
- (Optional, if dependencies are added) local backend smoke: run a tiny rerank on 5 candidates and
  verify stable ordering.

## Progress checklist
- [ ] Decide rerank capability taxonomy + ADR
- [ ] Define public types + API
- [ ] Implement HTTP rerank transport (OpenRouter/Portkey/OpenAI-compatible)
- [ ] Implement local CrossEncoder backend (Qwen3 rerankers + optional picks)
- [ ] Implement opt-in LLM-based rerank fallback
- [ ] Add truncation/chunking + aggregation controls
- [ ] Add tests (normalization, errors, metadata)
- [ ] Add docs + example
- [ ] Update server discovery (or document why not)

## Guidance for the implementing agent
Be critical of vendor performance claims and treat them as input hypotheses, not truth. Prefer a
design that makes evaluation easy: explicit truncation behavior, stable result shapes, and
benchmarks/tests that can run without external services.

### External references (for implementation notes; verify again before shipping)
- Jina-style rerank APIs (wire shape to match/normalize):
  - https://openrouter.ai/docs/api/api-reference/rerank/create-rerank/
  - https://openrouter.ai/docs/changelog/2026/4/4
  - https://portkey.ai/docs/api-reference/inference-api/rerank
- Local OpenAI-style rerank endpoints (useful for local serving / integration testing):
  - https://docs.vllm.ai/en/v0.7.0/getting_started/examples/jinaai_rerank_client.html
  - https://docs.vllm.ai/en/v0.10.1/api/vllm/entrypoints/openai/api_server.html
  - https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
- Cohere Rerank API reference and best practices:
  - https://docs.cohere.com/v2/reference/rerank
  - https://docs.cohere.com/docs/reranking-best-practices
- Voyage reranker docs and API reference:
  - https://docs.voyageai.com/docs/reranker
  - https://docs.voyageai.com/reference/reranker-api
- Jina reranker API schema:
  - https://jina.ai/en-US/reranker/
- Open-weight reranker examples:
  - https://huggingface.co/BAAI/bge-reranker-v2-m3
  - https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/blob/main/README.md
  - https://huggingface.co/Qwen/Qwen3-Reranker-4B/blob/main/README.md
  - https://huggingface.co/Qwen/Qwen3-Reranker-8B
  - https://sbert.net/docs/package_reference/cross_encoder/cross_encoder.html

- Reranker pitfalls worth keeping in mind (do not assume “more candidates” is always better):
  - https://arxiv.org/abs/2411.11767
  - https://arxiv.org/abs/2502.17036
