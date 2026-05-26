# Planned: ImageEmbeddingManager and multimodal embedding endpoint

## Metadata
- Created: 2026-05-24
- Status: Planned
- Completed: N/A

## ADR status
- Governing ADRs: ADR 0002, ADR 0003, ADR 0004
- ADR impact: May revise ADR 0003 or require a short follow-up ADR before implementation if
  `/v1/embeddings` is extended beyond strict text-compatible inputs.

## Context
AbstractCore has a mature text embedding surface but no first-class image embedding surface.
The new framework capability route matrix already contains `embedding.image`, and AbstractFlow /
Gateway can now configure route defaults for embeddings, but execution support is text-only.

Image embeddings are not the same thing as VLM image understanding. A VLM consumes image tokens and
generates text; an image embedding model returns dense vectors optimized for similarity search,
retrieval, clustering, or reranking pipelines. Many modern multimodal embedding models can map text
queries and image/document/video inputs into one comparable vector space.

## Current code reality
- `abstractcore/abstractcore/embeddings/manager.py` implements `EmbeddingManager.embed(text: str)`
  and `embed_batch(texts: List[str])`; the type and cache keys are text-oriented.
- `abstractcore/abstractcore/embeddings/models.py` lists text embedding models only.
- `abstractcore/abstractcore/server/app.py` defines `EmbeddingRequest.input` as `str | List[str]`
  and `POST /v1/embeddings` describes text embeddings.
- `abstractcore/abstractcore/server/app.py` supports provider/model parsing, provider-specific
  kwargs, dimensions, OpenAI-compatible remote providers, and local `EmbeddingManager` execution.
- `abstractcore/abstractcore/config/capability_defaults.py` already defines
  `embedding.text` and `embedding.image` route specs.
- Docs currently say `/v1/embeddings` is text embeddings only in several places, including
  `docs/server.md`, `docs/embeddings.md`, and `docs/reports/server-media-status.md`.
- `abstractcore/abstractcore/media/handlers/local_handler.py` can use text embeddings when there
  are no images, but there is no image-vector retrieval pipeline.

## Problem
Users can currently make images searchable only by captioning/OCRing them and embedding the text.
That is useful but incomplete:
- captions miss visual style, layout, objects that were not mentioned, and spatial relations;
- OCR misses non-text visual meaning;
- VLM-generated captions add latency and can hallucinate;
- there is no stable way to store image vectors, compare text-to-image vectors, or expose image
  embedding defaults through `embedding.image`.

## What we want to do
Add first-class image and multimodal embedding support while preserving the existing text embedding
API for current clients.

The intended public shape:
- text embeddings stay available through the current `EmbeddingManager` behavior;
- image embeddings get an explicit Python manager and route default;
- mixed retrieval can use a wrapper that validates the selected text/image models share a compatible
  embedding space before cross-modal similarity is allowed;
- the server keeps the OpenAI-style embedding response shape but supports AbstractCore multimodal
  input objects as an explicit extension.

## Why
This enables local and server-backed visual memory, screenshot search, product/photo retrieval,
visual document retrieval, and image-aware RAG without forcing every caller to invent a separate
pipeline.

It also keeps the framework route model honest: if `embedding.image` is configurable, the execution
host should eventually be able to execute it or report a clear unsupported capability.

## Requirements
- Keep strict text `/v1/embeddings` clients working unchanged.
- Do not silently compare vectors from unrelated embedding spaces.
- Record an `embedding_space_id` or equivalent metadata for every backend/model combination.
- Support both single-vector CLIP/SigLIP-style embeddings and multi-vector/late-interaction visual
  document embeddings, but do not pretend they are interchangeable.
- Keep `embedding.image` separate from `input.image`: image embeddings retrieve; VLM input image
  understanding generates or conditions text.
- Treat captions/OCR text embeddings as a complementary retrieval lane, not a replacement for
  image vectors.
- Keep server-side image ingestion safe: data URLs, uploaded files/artifacts, and allowed remote
  URLs must reuse existing media trust-boundary validation.

## Suggested implementation
### Python API
Introduce small typed contracts under `abstractcore.embeddings`:
- `EmbeddingInput`: `{modality, content, mime_type?, metadata?}` for text/image now and future
  video/audio later.
- `EmbeddingVector`: `{embedding, index, modality, model, provider, embedding_space_id, metadata}`.
- `EmbeddingResponse`: OpenAI-compatible `object`, `data`, `model`, `usage`, plus AbstractCore
  metadata.

Prefer explicit managers:
- `TextEmbeddingManager`: canonical text manager.
- `ImageEmbeddingManager`: `embed_image(...)`, `embed_image_batch(...)`.
- `MultimodalEmbeddingManager`: accepts `EmbeddingInput` lists, routes to text/image managers, and
  allows cross-modal similarity only when the embedding space is compatible.

`EmbeddingManager` may remain as a text alias during migration, but new code should call the
explicit manager that matches the route.

### Model registry
Split model metadata by modality/capability:
- `TEXT_EMBEDDING_MODELS`
- `IMAGE_EMBEDDING_MODELS`
- `MULTIMODAL_EMBEDDING_MODELS`

Add fields:
- `modalities`
- `dimension`
- `embedding_space_id`
- `supports_cross_modal`
- `vector_shape`: `single` or `multi_vector`
- `requires_trust_remote_code`
- `recommended_backend`: `sentence-transformers`, `transformers`, `open_clip`, `vllm`, etc.
- `license`

### Server API
Keep `POST /v1/embeddings` as the canonical endpoint and extend it carefully:
- Existing request remains valid:
  - `input: "text"`
  - `input: ["text 1", "text 2"]`
- Add AbstractCore extension inputs:
  - `{ "type": "text", "text": "..." }`
  - `{ "type": "image_url", "image_url": { "url": "data:image/png;base64,..." } }`
  - `{ "type": "image_file", "file_id": "artifact-or-upload-id" }`
  - lists may mix text and image objects only when the selected model is multimodal.
- Add optional request fields:
  - `input_modality`: explicit override when inference is ambiguous.
  - `route`: optional `embedding.text` or `embedding.image`; defaults by input type.
  - `return_metadata`: include `modality`, `embedding_space_id`, and ingestion metadata.

Response should remain OpenAI-shaped:
```json
{
  "object": "list",
  "data": [
    {"object": "embedding", "embedding": [0.1, 0.2], "index": 0}
  ],
  "model": "provider/model",
  "usage": {"prompt_tokens": 0, "total_tokens": 0},
  "metadata": {
    "route": "embedding.image",
    "embedding_space_id": "nomic-v1.5",
    "modalities": ["image"]
  }
}
```

For strict OpenAI compatibility, clients sending only strings should see the same shape and no
required new fields. For provider-scoped routes, keep `/{provider}/v1/embeddings` behavior aligned.

### Retrieval guidance
The recommended retrieval architecture should be hybrid:
- image vector lane: CLIP/SigLIP/Nomic/BGE/Qwen-style image vectors for visual semantics;
- text vector lane: captions, OCR, object labels, EXIF/file metadata, and user notes embedded as
  text;
- lexical lane: BM25/OCR exact matching;
- optional rerank lane: future `RerankManager`, preferably multimodal where available.

## Candidate open-source/open-weight models
- `nomic-ai/nomic-embed-vision-v1.5` (Apache-2.0): image embeddings aligned with
  `nomic-embed-text-v1.5`; good first local candidate for a clean image/text shared space.
- `BAAI/BGE-VL-base` and `BAAI/BGE-VL-large` (MIT): CLIP-based multimodal retrieval models with
  sentence-transformers usage; useful for text, image, and composed image-text inputs.
- `Qwen/Qwen3-VL-Embedding-2B` and `Qwen/Qwen3-VL-Embedding-8B` (Apache-2.0): newer, heavier
  multimodal embedding models for text, images, screenshots, video, and mixed-modal inputs.
- `mlfoundations/open_clip`: robust implementation and model zoo for CLIP/OpenCLIP baselines;
  useful as a simple local backend, but model weight licenses vary by checkpoint.
- `google/siglip2-*` (Apache-2.0 model cards): strong image-text embedding/classification family;
  good candidate after verifying local feature extraction path and pooling semantics.
- `jinaai/jina-clip-v2` and `jinaai/jina-embeddings-v5-omni-*`: strong open-weight options, but
  local weights are CC BY-NC; treat commercial use separately.

## Scope
- Define the Python contracts.
- Add local image embedding backend support for at least one Apache/MIT model family.
- Add `embedding.image` config resolution.
- Extend `/v1/embeddings` with explicit multimodal request objects.
- Add model catalog/discovery metadata for text/image/multimodal embedding models.
- Document hybrid retrieval patterns and vector-space compatibility rules.

## Non-goals
- Do not implement full visual memory or vector database storage in this item.
- Do not replace VLM image understanding or captioning.
- Do not force every text embedding client through a multimodal manager.
- Do not claim all image embedding models produce comparable vectors.
- Do not implement video/audio embeddings here, except keeping the input contract extensible.

## Dependencies and related tasks
- `planned/0801_rerank_manager.md`: reranking should consume candidates from text/image retrieval
  lanes later.
- `proposed/0804_huggingface_embedding_provider_boundary.md`: settle whether local
  SentenceTransformers-style embedding execution is manager-owned, provider-owned, or split by a
  clearer backend id before broadening provider/catalog metadata for multimodal embeddings.
- `abstractcore/abstractcore/config/capability_defaults.py`: `embedding.image` route already exists.
- `abstractcore/abstractcore/server/app.py`: existing `/v1/embeddings` endpoint and request schema.
- `abstractcore/abstractcore/embeddings/manager.py`: current text embedding behavior.
- ADR 0003: provider/capability boundaries.
- ADR 0004: server trust boundary for remote URLs, files, and operator-controlled routes.

## Expected outcomes
- Users can configure `embedding.image` with provider/model/base URL.
- Python callers can embed images directly and receive numeric vectors.
- Server callers can use `/v1/embeddings` for text or image inputs with the same response family.
- Cross-modal retrieval rejects incompatible embedding spaces with a clear error.
- Documentation explains when to use image embeddings, captions/OCR text embeddings, and VLMs.

## Validation
- Unit tests for `ImageEmbeddingManager` using a small mocked backend.
- Contract tests that text-only `/v1/embeddings` requests remain unchanged.
- Server tests for image data URL input, file/artifact input, invalid MIME, unsupported provider,
  incompatible mixed modalities, `dimensions`, and metadata.
- Real-model smoke test behind an opt-in env flag for one small Apache/MIT candidate
  (`BAAI/BGE-VL-base` or `nomic-ai/nomic-embed-vision-v1.5`).
- Retrieval sanity test: text query retrieves a semantically matching image above a distractor using
  the same embedding space.

## Progress checklist
- [ ] Finalize whether the server extension belongs only on `/v1/embeddings` or also on a named
      `/v1/embeddings/multimodal` helper.
- [ ] Add typed embedding input/result contracts.
- [ ] Split text/image/multimodal model metadata.
- [ ] Implement one local image embedding backend.
- [ ] Wire `embedding.image` defaults into manager initialization.
- [ ] Extend server request parsing and docs.
- [ ] Add focused unit, server, and opt-in real-model validation.

## Guidance for the implementing agent
Re-check model cards, licenses, and runtime requirements before coding. Start with the smallest
clean path: one single-vector model, explicit vector-space metadata, and strict compatibility
checks. Do not start with multi-vector ColPali/ColNomic semantics unless the single-vector contract
is already stable.
