# Benchmark Prompt Sets (MLX concurrency)

This folder contains fixed prompt sets used by `examples/performance/mlx_concurrency_benchmark.py`.

Why this exists:
- A fixed prompt set lets you compare throughput across runs/models without your workload changing.
- The benchmark script also supports generating synthetic prompts on the fly, but that is less comparable.

Files:
- `mlx_benchmark_prompts_50.json` — 50 short/medium prompts
- `mlx_benchmark_prompts_128.json` — 128 short/medium prompts (useful for wider concurrency sweeps)

Usage:

```bash
python examples/performance/mlx_concurrency_benchmark.py \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --prompts-file examples/performance/assets/mlx_benchmark_prompts_128.json \
  --queries 128 \
  --concurrency-levels 1 2 4 8 16 32 64 128 \
  --max-output-tokens 512
```
