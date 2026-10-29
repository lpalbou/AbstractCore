### Fixed

- **A GGUF embedding repository id is served by llama.cpp** (`embeddings/manager.py`, OVH 8080 pass four). `/v1/embeddings` with `huggingface/Qwen/Qwen3-Embedding-0.6B-GGUF` used to reach sentence-transformers, which cannot read a GGUF, and answered 500 "Unrecognized model". A `-GGUF` id now goes straight to the llama.cpp lane.
- **The GGUF embedder reserves the largest input that fits, and says so.** With n_batch = n_ubatch = n_ctx, llama.cpp's compute buffers grow with the input length squared. At 8,192 tokens the 0.6B Qwen3 embedder held 6.6 GB of GPU memory on a 16 GB card, leaving no room for a second LLM slot, and its row read `device_bytes: None`.
  - On a CUDA GPU the encoder now reserves the largest input (a multiple of 256, at most the model's maximum) whose weights, KV and compute buffers fit the free memory minus the safety buffer (`model_memory_buffer`, 5 %). It logs the choice.
  - The row carries `context_length`, `advertised_context_length`, `context_source` (`memory` / `configured`), `memory_buffer` and `context_reservation`, the same fields as an LLM context row.
  - A maximum input an admin set (`model_contexts`, the same per-model setting as an LLM context) is used exactly. On a GPU-only device that cannot hold it, the load is refused in words.
  - A longer text is refused with "split it before embedding it", never cut.
  - `device_bytes` is measured through NVML when torch has no CUDA context, as in the gateway's model host.
- **The cause of the 6.6 GB: the GGUF embedder's scratch, not its model.** With n_ubatch = 8,192, llama.cpp reserved a 5.1 GB compute buffer: the output projection for every token of the micro-batch, 8,192 × 151,669 vocabulary × 4 bytes, plus the attention scratch. A causal embedder (`<arch>.attention.causal`, or a decoder architecture: Qwen3-Embedding) now reads its full input (n_ctx = n_batch = the model's maximum, 8,192) in 512-token micro-batches with flash attention. A bidirectional encoder keeps the whole input in one micro-batch.
  - Measured with Qwen3-Embedding-0.6B Q8_0 at 8,192 tokens, the same buffer sizes on OVH 8080 (CUDA, llama-cpp-python 0.3.36) and on an Apple Silicon Mac (Metal, 0.3.35). Compute buffer 5,092 → 306 MiB; model 604 MiB and f16 KV 896 MiB unchanged. Total GPU memory on 8080 went from 6,710 to 1,808 MB.
  - Vectors are equivalent: minimum cosine against the previous path over 20 texts (up to 7,581 tokens) is 0.99958 on CUDA and 0.999999 on Metal.
  - With flash attention the default reservation is the model's full maximum. The fit, set-value and refusal logic above remains the safety net.
  - The row now also says which build serves it (`quant`: `Q8_0`, `model_file`) and carries `flash_attn` and `micro_batch`.
