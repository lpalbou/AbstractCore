# HuggingFace Model Compatibility

This guide explains which HuggingFace local model formats AbstractCore can load directly and how to
diagnose model-load failures. These checks are general provider compatibility checks; they are not
specific to prompt caching, memory blocs, tools, or structured output.

## Baseline Install

`pip install "abstractcore[huggingface]"` installs the stable HuggingFace provider stack:

- `transformers`
- `torch`
- `torchvision`
- `torchaudio`
- `sentencepiece`
- `llama-cpp-python`
- `outlines`

This baseline supports standard Transformers checkpoints and GGUF files. It intentionally does not
install every optional Transformers quantization runtime, because those runtimes are
platform-specific and can carry dependency pins that conflict with the rest of the local stack.
Fresh installs resolve the newest compatible Transformers release allowed by AbstractCore's
dependency range. Very new architectures such as Gemma4 require a recent Transformers build.
Audio/voice extras use `abstractvoice>=0.10.14` and `omnivoice>=0.1.5`.

## Quantized Transformers Checkpoints

Some HuggingFace Transformers repositories include a `quantization_config` in `config.json`. Common
families include AWQ, GPTQ, bitsandbytes, and compressed-tensors. Those formats are not just smaller
`.safetensors` files; they require the matching quantization runtime to map packed or compressed
weights back into usable model modules.

If the runtime is missing or incompatible, a model may appear to download but still fail as a valid
runtime target. Treat these as model-load failures:

- missing base weights
- unexpected packed/compressed weight tensors
- mismatched weight shapes
- incorrect output for a trivial prompt such as `Reply with exactly: ready`

AbstractCore preflights known quantization configs and reports actionable errors for common cases
instead of letting a broken quantized checkpoint initialize silently.

## Trusted Model Selection

Prefer official or widely maintained model/runtime pairs for release validation.

For Qwen3.5 4B, the official Qwen namespace currently provides:

- `Qwen/Qwen3.5-4B`
- `Qwen/Qwen3.5-4B-Base`

The official Qwen namespace does not currently provide a `Qwen/Qwen3.5-4B-GPTQ-Int4` or
`Qwen/Qwen3.5-4B-AWQ` repository. Official Qwen3.5 GPTQ-Int4 repositories exist for larger model
families such as 27B and 397B-A17B. Q4 4B artifacts are available through other ecosystems, for
example MLX or GGUF conversions, but those are different provider/runtime paths.

For Qwen3.6, the official Qwen namespace currently provides:

- `Qwen/Qwen3.6-27B`
- `Qwen/Qwen3.6-27B-FP8`
- `Qwen/Qwen3.6-35B-A3B`
- `Qwen/Qwen3.6-35B-A3B-FP8`

The official Qwen namespace does not currently provide GPTQ-Int4 or AWQ repositories for Qwen3.6
27B or 35B-A3B. Qwen3.6 Q4/AWQ/GPTQ artifacts exist through community or runtime-specific
ecosystems, including GGUF, MLX, compressed-tensors, and AWQ conversions. Treat those as
provider/runtime-specific targets rather than official Qwen Transformers proof targets.

FP8 Transformers checkpoints are also runtime-specific. Hugging Face's fine-grained FP8
documentation targets CUDA-class FP8 hardware, and the installed Transformers quantizer requires a
CUDA or XPU runtime for native FP8 execution. Apple Silicon should use Apple-native quantization
paths instead: Qwen's own local-use guidance points Apple users to MLX models, while Transformers
documents Metal quantization separately for MPS with 2/4/8-bit affine kernels.

In local Apple/MPS validation, `Qwen/Qwen3.6-27B-FP8` downloaded and initialized through the
Transformers fallback path, but produced incorrect answers for trivial prompts. AbstractCore
therefore rejects that pairing before cache, tool, or structured-output validation. If upstream
adds a real MPS FP8 path later, enable it only after model-load correctness and semantic smoke tests
pass in CI or an explicit validation report.

Use provider-native paths for those artifacts:

- MLX quantized models: `create_llm("mlx", model="mlx-community/...")`
- GGUF quantized models: `create_llm("huggingface", model="/path/to/model.gguf")`
- Transformers-native quantized models: only when the required Transformers quantization runtime is
  installed and a clean load plus semantic smoke test passes

Gemma4 official HF transformers targets such as `google/gemma-4-E4B-it` are valid
HuggingFace-provider targets when the local Transformers build recognizes `model_type=gemma4`.
Dense Gemma4 E4B-it was validated on Apple/MPS in an isolated Transformers 5.9.0 environment.
Durable bloc cache support depends on preserving dynamic sliding-window cache sequence lengths
across save/load; AbstractCore records that in the transformers cache artifact metadata.

## AbstractCore Policy

AbstractCore should improve compatibility without making the default install fragile:

- Keep optional quantization runtimes out of the base `huggingface` extra unless they are stable
  across supported platforms.
- Add lightweight preflight checks from model config before loading large weights.
- Fail with explicit model/runtime compatibility errors for known missing quantization runtimes.
- Reject quantized loads that report missing base weights or unexpected packed weights.
- Consider narrow optional extras later, such as `huggingface-awq`, `huggingface-gptq`, or
  `huggingface-compressed-tensors`, only after dependency compatibility is verified.
- Use official or trusted model owners for release proofs; treat unknown third-party quantized
  checkpoints as user-supplied compatibility targets, not AbstractCore proof targets.

References:

- HuggingFace Transformers quantization overview:
  <https://huggingface.co/docs/transformers/quantization/overview>
- HuggingFace compressed-tensors integration:
  <https://huggingface.co/docs/transformers/quantization/compressed_tensors>
- Qwen3.5 collection:
  <https://huggingface.co/collections/Qwen/qwen35>
- Qwen3.6 27B:
  <https://huggingface.co/Qwen/Qwen3.6-27B>
- Qwen3.6 27B FP8:
  <https://huggingface.co/Qwen/Qwen3.6-27B-FP8>
- Qwen3.6 35B-A3B:
  <https://huggingface.co/Qwen/Qwen3.6-35B-A3B>
- Qwen3.6 35B-A3B FP8:
  <https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8>
