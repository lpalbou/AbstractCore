### Fixed

- **A GGUF embedding repository id is served by llama.cpp** (`embeddings/manager.py`, OVH 8080 pass four). `/v1/embeddings` with `huggingface/Qwen/Qwen3-Embedding-0.6B-GGUF` used to reach sentence-transformers, which cannot read a GGUF, and answered 500 "Unrecognized model". A `-GGUF` id now goes straight to the llama.cpp lane.
- **The GGUF embedder reserves the largest input that fits, and says so.** With n_batch = n_ubatch = n_ctx, llama.cpp's compute buffers grow with the input length squared. At 8,192 tokens the 0.6B Qwen3 embedder held 6.6 GB of GPU memory on a 16 GB card, leaving no room for a second LLM slot, and its row read `device_bytes: None`.
  - On a CUDA GPU the encoder now reserves the largest input (a multiple of 256, at most the model's maximum) whose weights, KV and compute buffers fit the free memory minus the safety buffer (`model_memory_buffer`, 5 %). It logs the choice.
  - The row carries `context_length`, `advertised_context_length`, `context_source` (`memory` / `configured`), `memory_buffer` and `context_reservation`, the same fields as an LLM context row.
  - A maximum input an admin set (`model_contexts`, the same per-model setting as an LLM context) is used exactly. On a GPU-only device that cannot hold it, the load is refused in words.
  - A longer text is refused with "split it before embedding it", never cut.
  - `device_bytes` is measured through NVML when torch has no CUDA context, as in the gateway's model host.
