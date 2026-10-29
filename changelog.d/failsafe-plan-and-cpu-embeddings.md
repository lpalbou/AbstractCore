### Fixed

- **The residency failsafe never evicts a model it cannot replace** (ADR 0010). On the OVH 8080 gateway the unlocked
  text model was evicted for a 14B load that was then refused anyway (the free memory after the eviction was still
  short), leaving neither model. `make_room` now plans first: it evicts only when the free memory plus what the
  evictable models hold covers the need; otherwise it refuses with everything kept.
- **The GGUF embedding lane runs entirely on the CPU.** With `n_gpu_layers=0` llama.cpp's CUDA build still placed the
  KV cache and the `n_ubatch`-sized compute buffers on the GPU (`offload_kqv` defaults on): 4.9 GiB of device memory
  for `Qwen3-Embedding-0.6B-GGUF`, shown on no Models row (gateway backlog 0966). Now `offload_kqv=False` (and
  `op_offload=False` where supported) and the context is bounded at 4,096 tokens; measured on the same server, the
  embedder process holds 118 MiB of GPU (the CUDA context) instead of failing to allocate next to the text model.
