# Concurrency / Performance Examples

## What This Folder Teaches

How to reason about latency/throughput tradeoffs in LLM apps:
- streaming vs non-streaming,
- batching/concurrency (especially for local runtimes like MLX),
- and “embeddings-scale” operations (similarity matrices, clustering).

## Prereqs

- MLX-specific benchmarks: `pip install "abstractcore[mlx]"`
- Some scripts also use embeddings: `pip install "abstractcore[embeddings]"`

## Key AbstractCore Concepts

- Provider capabilities differ: some can batch/concurrently decode efficiently; others can’t.
- Streaming is a UX feature first; it can change observed latency even when total work is similar.
- Benchmarks should be reproducible: pin model, temperature, max tokens, and prompt set.

## Scripts

- `streaming_vs_non_streaming.py`
  - Demonstrates: how response delivery changes when you stream tokens.
  - Takeaway: “fast first token” and “fast total” are different metrics.

- `mlx_concurrency_benchmark.py`
  - Demonstrates: how MLX throughput changes with concurrent jobs / pooling.
  - Takeaway: concurrency can help or hurt depending on decode length and memory pressure.

- `matrix_operations_demo.py` (embeddings-scale, dev-oriented)
  - Demonstrates: similarity matrices, query-doc matching, clustering, and performance tricks (normalization/chunking).
  - Takeaway: once you cross into N×N territory, batching + chunking become essential.

## Key Takeaways

- Benchmark on your own prompts and expected output lengths; “throughput” is workload dependent.
- Prefer measuring: TTFT (time-to-first-token), total latency, and tokens/sec separately.
