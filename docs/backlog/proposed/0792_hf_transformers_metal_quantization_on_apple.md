# Proposed: HF Transformers Metal Quantization On Apple

## Metadata
- Created: 2026-05-20
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: ADR 0007
- ADR impact: None unless this becomes a public provider option with cross-provider naming rules

## Context
AbstractCore now has provider-wide durable memory bloc artifacts for MLX, HuggingFace transformers,
and HuggingFace GGUF. The current Apple-local HF transformers proof uses official dense
BF16-style weights such as `Qwen/Qwen3.5-4B`, while the comparable Apple-native MLX and GGUF proofs
use 4-bit artifacts. That makes HF transformers slower and more memory-heavy in Apple-local tests,
but FP8 and GPTQ checkpoints are not native MPS proof targets.

Hugging Face documents a Metal quantization path for Apple Silicon via `MetalConfig(bits=4)`.
This is different from loading a prequantized FP8/GPTQ checkpoint: it quantizes official base
weights into Metal-backed MPS kernels at load time.

## Current code reality
- [`abstractcore/providers/huggingface_provider.py`](/Users/albou/tmp/abstractframework/abstractcore/abstractcore/providers/huggingface_provider.py:2770) loads transformers models through `AutoConfig`, `AutoTokenizer`, and `AutoModelForCausalLM.from_pretrained(...)`, then uses the same prompt-cache control-plane methods for durable memory blocs.
- [`pyproject.toml`](/Users/albou/tmp/abstractframework/abstractcore/pyproject.toml:127) includes the normal `huggingface` extra but does not include the optional `kernels` package required by the Metal quantization path.
- The local environment exposes `transformers.MetalConfig`, but `kernels` is not installed.
- `Qwen/Qwen3.5-4B` works today as the official HF transformers Apple proof target, but it is a dense BF16 model with an 8.68 GiB safetensors payload.
- `google/gemma-4-E4B-it` is an official Gemma4 HF transformers model, but it requires a newer transformers build than the current repo environment. In an isolated transformers 5.9.0 venv, text generation on MPS works for trivial prompts. Durable bloc-cache reuse requires preserving Gemma4 sliding-window cache sequence lengths across save/load; AbstractCore tracks this through the architecture cache-position strategy and dynamic-cache layer metadata.

## Problem or opportunity
Apple users who want the HF transformers provider, not MLX or GGUF, currently have no clean
4-bit local proof target in AbstractCore. Official Qwen and Gemma repositories publish dense
transformers weights, while official quantized variants are generally CUDA/XPU/GPTQ/FP8-oriented
or runtime-specific. Metal quantization may provide an Apple-native HF transformers 4-bit path
without making AbstractCore depend on fragile third-party quantization stacks by default.

## Proposed direction
Add an explicit, optional HF transformers Metal quantization path for Apple Silicon:

- expose one small provider option, for example `transformers_quantization="metal-4bit"` or
  `quantization={"backend": "metal", "bits": 4}`;
- when the provider is `huggingface`, backend is transformers, device is MPS, and the option is
  explicitly set, construct `transformers.MetalConfig(bits=4)` and pass it as
  `quantization_config` to `from_pretrained(...)`;
- add an optional install extra such as `abstractcore[huggingface-metal]` or
  `abstractcore[apple-hf-metal]` that includes `kernels` without adding it to the base
  `huggingface` extra;
- fail early with a clear error when `kernels`, MPS, or compatible Transformers support is absent;
- keep the durable memory bloc API unchanged: `ensure_bloc_kv_artifact(...)`,
  `load_bloc_kv_artifact(...)`, server `/acore/blocs/kv/*`, and `prompt_cache_binding` should not
  care whether the model was dense or Metal-quantized.

## Why it might matter
This could give Apple users a fairer HF transformers comparison against MLX and GGUF:

- same public AbstractCore provider family: `create_llm("huggingface", ...)`;
- official base model weights instead of untrusted third-party Q4 conversions;
- smaller local memory footprint than dense BF16;
- one unified durable memory bloc contract independent of quantization backend.

## Promotion criteria
Promote this only after a local spike proves:

- `Qwen/Qwen3.5-4B` loads on MPS with `MetalConfig(bits=4)` using only optional dependencies;
- generation passes simple semantic smoke tests;
- durable memory bloc compile/load/request binding still works through the existing
  `PromptCacheCapabilities` interface;
- prompt-processing speed and memory footprint are measured against the dense HF transformers
  baseline;
- failure modes are clean when `kernels` is missing or the model/runtime does not support Metal
  quantization.

Gemma4 should remain a secondary validation target because its architecture and dependency
requirements are newer than Qwen3.5. Do not use Gemma4 Metal quantization as the first promotion
gate.

## Validation ideas
- Run the durable bloc benchmark for dense `Qwen/Qwen3.5-4B` and Metal-quantized
  `Qwen/Qwen3.5-4B` in separate processes.
- Record model load time, memory pressure if available, full prompt processing, cached suffix
  processing, artifact load, cached generation, artifact size, token count, and answer
  correctness.
- Verify Python and server paths both accept the same `prompt_cache_binding` after Metal
  quantization.
- Add unit tests that monkeypatch `transformers.MetalConfig` and missing `kernels` so CI can cover
  selection and error messages without requiring Apple hardware.

## Non-goals
- Do not add `kernels` to the base `huggingface` extra.
- Do not auto-enable Metal quantization just because the device is MPS.
- Do not treat Metal quantization as a portable artifact format; durable memory blocs remain
  provider/model/runtime-native.
- Do not claim Gemma4 Metal quantization support until dense Gemma4 HF transformers remains correct
  under the same durable-cache test shape.

## Guidance for future agents
Keep this simple. The abstraction should be one explicit provider loading option plus one optional
dependency profile. If implementation requires model-family-specific cache semantics, split that
into a separate planned item rather than hiding it inside the quantization option.
