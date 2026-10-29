### Fixed

- **A GGUF prompt-cache snapshot no longer carries half a gigabyte it never uses, and the snapshots stay under a byte
  budget** (8080 pass four, 2026-10-29: the model host child grew 1.65 GB per basic-agent run, 4.60 to 9.57 GB in
  three runs). `Llama.save_state` copies a `scores` array of `n_batch x n_vocab` float32 into every snapshot, 508 MB
  for a Qwen3.5 vocabulary. With `logits_all=False` llama.cpp never writes or reads that array, and nothing counted
  it, so the stats showed 0.42 GB of caches per run while 1.5 GB more stayed resident. A stored snapshot now keeps
  one zero row, which `load_state` broadcasts; with `logits_all=True` the array is kept whole. Per-key `bytes` and
  `cache_state_total_bytes` count the arrays too. The snapshots of every key together are held to
  `ABSTRACTCORE_GGUF_PROMPT_CACHE_MAX_GB` (default: min(8 GiB, a quarter of physical RAM), the MLX prefix cache's
  rule). The least recently used keys are evicted on every insert path through a new `PromptCacheStore` `on_set`
  hook and `evict()`, and `get_prompt_cache_stats()["gguf"]` reports `prompt_cache_budget_bytes` and
  `prompt_cache_total_bytes`. Measured on a Mac with `unsloth/Qwen3.5-4B-GGUF` through a real gateway
  (`--model-host process`), ten basic-agent runs:
  - before: child RSS 14.77 to 29.95 GB, +1.69 GB a run;
  - after: 13.28 to 15.69 GB, +0.27 GB a run while keys accumulate;
  - with a 1.5 GiB budget: the caches hold at 1.61 GB from run 5 to run 15, and RSS stays within 11.2 to 13.5 GB.

  A restore from a compacted snapshot answers token for token like a cold prefill.
