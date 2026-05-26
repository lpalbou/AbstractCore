# Proposed: HuggingFace embedding provider boundary

## Metadata
- Created: 2026-05-25
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: ADR-0003 provider/capability/output boundaries; ADR-0008 provider-owned model residency truth
- ADR impact: May revise existing ADRs if promoted, because this touches provider identity, embedding execution ownership, and residency truth.

## Context
AbstractCore currently exposes `huggingface` as a text embedding provider in the embeddings catalog, but that embedding path is not the same runtime path as `abstractcore.providers.huggingface_provider.HuggingFaceProvider`.

This matters because `huggingface` can mean at least two different things today:

- the broad AbstractCore provider registered in `ProviderRegistry`, with transformers/GGUF text generation, media-adjacent behavior, model residency hooks, and local cache behavior;
- the text embedding path in `EmbeddingManager`, which loads SentenceTransformers models directly in-process and manages its own embedding cache.

The recent embedding provider catalog cleanup removed static `local`/`remote` claims and replaced them with transport metadata. That made the next ambiguity clearer: `transport="python_inprocess"` is true for the current HuggingFace embedding path, but it does not say whether the embedding provider is intended to be the general AbstractCore HuggingFace provider or a dedicated SentenceTransformers embedding backend.

## Current code reality
- `abstractcore/embeddings/models.py` registers embedding provider `huggingface` with `transport="python_inprocess"` and `requires_local_model_files=True`.
- `abstractcore/embeddings/manager.py` handles `provider == "huggingface"` by loading `sentence_transformers.SentenceTransformer(...)` directly, then calling `self.model.encode(...)` for single and batch embeddings.
- `EmbeddingManager` delegates non-HuggingFace embedding providers to provider objects that expose `embed(...)`, such as Ollama, LM Studio, OpenAI, OpenAI-compatible, OpenRouter, Portkey, and vLLM.
- `abstractcore/providers/registry.py` registers the general `huggingface` provider with `supported_features=["chat", "completion", "embeddings", "prompted_tools", "local_models", "structured_output"]`.
- `abstractcore/providers/huggingface_provider.py` implements text generation, GGUF/transformers loading, unload, and residency truth, but current inspection found no `embed(...)` method on `HuggingFaceProvider`.
- `HuggingFaceProvider.get_model_residency(...)` reports text-generation residency for provider-owned loaded models; the SentenceTransformers embedding model residency and cache are owned separately by `EmbeddingManager`.

## Problem or opportunity
The shared provider id `huggingface` is overloaded. A caller may reasonably infer that `huggingface` embeddings go through `HuggingFaceProvider`, but the actual embedding path is a dedicated SentenceTransformers implementation.

That may be acceptable, but it needs an explicit boundary. Without one, future work could accidentally duplicate model loading, report misleading residency, route embedding requests through the wrong provider contract, or make provider discovery claim support that does not match executable behavior.

## Proposed direction
Decide and document one of these boundaries:

1. Keep `huggingface` embeddings as an `EmbeddingManager`-owned SentenceTransformers backend, and make the public metadata explicit with a transport such as `sentence_transformers_inprocess`.
2. Add a provider-owned `embed(...)` path to `HuggingFaceProvider` and move HuggingFace embedding execution under the general provider contract.
3. Split identity: reserve `huggingface` for the general provider and introduce a clearer embedding backend id such as `sentence-transformers` while preserving compatibility aliases.

The likely safest first step is option 1: keep execution where it works, rename/clarify the transport, and document that embedding backends are not automatically the same thing as text-generation providers.

## Why it might matter
- Discovery consumers should not infer execution topology or residency from provider ids.
- The `huggingface` provider registry claims `embeddings` support, but execution does not currently use the provider class.
- Model residency and unload behavior differ between `HuggingFaceProvider` and `EmbeddingManager`.
- Future image/multimodal embedding work could compound the ambiguity if provider identity is not clarified first.

## Promotion criteria
Promote when one of these becomes true:

- a user-facing bug shows HuggingFace embedding requests are routed, discovered, warmed, or unloaded incorrectly;
- a new embedding residency/control-plane feature needs truthful provider-owned loaded-state reporting;
- image or multimodal embeddings require a unified provider/backend identity model;
- `planned/0801_rerank_manager.md` needs to choose whether local CrossEncoder rerankers are
  SentenceTransformers-manager-owned, HuggingFaceProvider-owned, or exposed under a clearer backend id;
- docs or UI need to explain why `huggingface` embeddings do not behave like `HuggingFaceProvider` text models.

## Validation ideas
- Add unit tests proving `huggingface` text embeddings use the intended execution owner.
- Add provider catalog tests that assert HuggingFace embedding transport and local model-file requirements without claiming provider-level locality.
- Add a regression test proving `HuggingFaceProvider` registry metadata and embedding catalog metadata do not drift silently.
- If a provider-owned implementation is chosen, test `HuggingFaceProvider.embed(...)`, batch embeddings, cache behavior, residency, unload, and server `/v1/embeddings` routing.
- If a split identity is chosen, test backward compatibility for `huggingface/<model>` embedding routes.

## Non-goals
- Do not force HuggingFace embeddings through `HuggingFaceProvider` merely for naming symmetry.
- Do not remove existing `huggingface/<model>` embedding compatibility without a migration path.
- Do not treat local cache presence, provider id, or Python process ownership as proof of loaded embedding residency.
- Do not broaden this item into image/multimodal embedding design; that remains tracked separately by `planned/0803_image_embedding_manager_and_multimodal_embeddings.md`.
- Do not broaden this item into reranker implementation; `planned/0801_rerank_manager.md` owns the
  reranker API, but should consume the provider/backend boundary decision here.

## Guidance for future agents
Re-check the current implementation before changing code. If `HuggingFaceProvider` has gained `embed(...)` by then, compare behavior with the SentenceTransformers path before consolidating. If not, prefer the smallest truthful catalog/docs cleanup first: clarify the transport/backend name and state explicitly that `provider/model/base_url` describes a route, while embedding execution ownership may be manager-owned or provider-owned depending on backend.
