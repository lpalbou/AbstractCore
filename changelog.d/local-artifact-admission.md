### Fixed

- Managed local loaders reuse compatible resident artifacts or refuse conflicting second allocations across MLX, HuggingFace and embedding paths. Explicit incompatible placement, native target/head or Qwen4 PLE settings require unloading first; Transformers callers reuse the resident provider.
- Pinned GGUF concurrency can construct its context slots within a model load while preserving serialization between concurrent loaders.
