# Embeddings Examples

## What This Folder Teaches

How to use AbstractCore embeddings as a *local* building block for:
- similarity search (“semantic search”),
- clustering / deduplication,
- and RAG-style retrieval (finding context to pass to an LLM).

## Prereqs

- Install embeddings support: `pip install "abstractcore[embeddings]"`

## Key AbstractCore Concepts

- `EmbeddingManager`: the main entry point (model selection, caching, batch embedding).
- `embed(...)` / `embed_batch(...)`: produce vectors for text.
- `compute_similarity(...)`: cosine similarity helper for quick relevance scoring.
- Cache behavior: repeated embeddings are reused (memory + optional persistent cache).

## How The Examples Work

Most scripts follow the same pattern:
1) create an `EmbeddingManager` (choose a model)  
2) embed queries + documents (often batched)  
3) compute similarity scores or matrices  
4) sort/filter results to build a retriever

RAG in AbstractCore is deliberately composable: embeddings retrieval is separate from LLM generation.

## Scripts

- `embeddings_hf_demo.py`
  - Demonstrates: the “cleanest” local-only usage (HuggingFace embeddings end-to-end).
  - Takeaway: you can do useful retrieval without any hosted API.

- `embeddings_demo.py`
  - Demonstrates: basic embedding + similarity + a simple RAG prompt scaffold.
  - Takeaway: retrieval is just “pick the best context”, then call an LLM (not coupled).

- `simple_embeddings_examples.py`
  - Demonstrates: copy/paste recipes (single embedding, similarity, document search, batching, RAG scaffold).
  - Takeaway: a small set of primitives covers most production use cases.

- `embeddings_benchmark.py` (dev/benchmark oriented)
  - Demonstrates: comparing multiple embedding models on clustering-like tasks.
  - Takeaway: model choice matters; benchmark with your own domain sentences.

## Key Takeaways

- Use `embed_batch(...)` for throughput; it’s usually faster than many single calls.
- Treat embeddings as a separate “retrieval layer” you can swap independently from your LLM provider.
