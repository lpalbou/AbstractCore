# Model compatibility and issue reporting

Model-family capabilities do not guarantee that every artifact and backend
implements the same inputs, quantization or generation features. Check the
exact provider, checkpoint and installed runtime versions.

- [Vision capabilities](vision-capabilities.md) explains native image delivery,
  delivery metadata, model-specific limits and caption fallback.
- [Native MLX runtime](native-mlx-runtime.md) documents supported Qwen3.8 MTP,
  concurrency, caching and current-turn image inputs.
- [Native MLX measurements](native-mlx-benchmarks.md) records tested artifacts
  and measured behavior; it does not certify other models or runtime versions.
- [HuggingFace compatibility](huggingface-model-compatibility.md) covers
  Transformers/GGUF loading and quantized checkpoint requirements.
- [Troubleshooting](troubleshooting.md) provides symptom-oriented checks and fixes.

When reporting an issue, include the exact model/artifact, provider and runtime
versions, operating system, generation controls, expected/actual behavior and a
minimal example. For vision, include a shareable image and image-delivery
metadata. Remove credentials and private prompt/media content before posting.

Use the [issue tracker](https://github.com/lpalbou/AbstractCore/issues).
