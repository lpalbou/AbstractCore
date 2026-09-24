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
Audio/voice capability extras use `abstractvoice>=0.11.2` without installing
OmniVoice, torch, or torchaudio. Local OmniVoice engines are part of the
explicit local aggregate profiles such as `abstractcore[all-apple]` and
`abstractcore[all-gpu]`.

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

`Qwen/Qwen3.6-27B-FP8` is not supported through Transformers on Apple/MPS;
AbstractCore rejects that pairing. A successful weight load alone does not
establish correct execution for an unsupported quantization/runtime pair.
Use an Apple-native MLX artifact or a compatible GGUF artifact instead.

Use provider-native paths for those artifacts:

- MLX quantized models: `create_llm("mlx", model="mlx-community/...")`
- GGUF quantized models: `create_llm("huggingface", model="/path/to/model.gguf")`
- Transformers-native quantized models: require a compatible installed
  quantization runtime and correct generation, not only successful weight loading

Gemma4 official HF transformers targets such as `google/gemma-4-E4B-it` are valid
HuggingFace-provider targets when the local Transformers build recognizes `model_type=gemma4`.
Dense Gemma4 E4B-it was validated on Apple/MPS in an isolated Transformers 5.9.0 environment.
Durable bloc cache support depends on preserving dynamic sliding-window cache sequence lengths
across save/load; AbstractCore records that in the transformers cache artifact metadata.

## Which provider lists a local repository

`mlx` and `huggingface` scan the same directories — the HuggingFace hub cache and the
LM Studio store — so every local repo has to be assigned to exactly one of them. One
rule decides, `abstractcore/providers/mlx_model_rules.py`, and both listings call it,
so the two lists are complements: nothing appears in both pickers, nothing disappears
from both. The same rule drives the `create_llm("huggingface", ...)` → `mlx` re-route
and the MLX-artifact refusal in the Transformers loader.

In order, first match wins:

1. `ABSTRACTCORE_NON_MLX_MODEL_PATTERNS` — operator override, never MLX.
2. `ABSTRACTCORE_MLX_MODEL_PATTERNS` — operator override, always MLX.
3. A GGUF marker in the name — llama.cpp weights, which MLX cannot load.
4. `mlx` anywhere in the name (`mlx-community/*`, `*-MLX-4bit`, …).
5. A publisher that ships MLX by construction: `mlx-community`, `Jundot` (oMLX/oQ).
6. An MLX-only quantizer tag: oMLX's `oQ<n>`, as in `Jundot/Qwen3.8-27B-oQ4e-mtp`.
   The leading `o` is what keeps this off llama.cpp's `Q4_K_M` / `q8_0` GGUF tags.
7. The on-disk signature: `library_name: mlx` in the model card, or an MLX
   `quantization` block in `config.json` (`bits` + `group_size`, no `quant_method`).
   A card naming a generation library (`mlx-gen`, `mflux`) is MLX format but belongs
   to the media backends, so it is not offered as an LLM.

Rules 4–6 read the name alone, which is what makes them work for a repo whose weights
are not downloaded yet: a hub entry that was only resolved has `refs/` and no
`config.json` to inspect. Rule 7 is authoritative whenever the weights are present.

Both override variables take patterns separated by commas or `os.pathsep`. A pattern
containing `*`, `?` or `[` is matched against the whole lowercased `org/model` handle
with `fnmatch`; anything else is a plain substring test:

```bash
export ABSTRACTCORE_MLX_MODEL_PATTERNS="acme/*-mlxq,someorg/one-model"
```

Use these when a publisher ships MLX under a name no rule recognizes, rather than
waiting for a new release of AbstractCore.

## Loading from the local cache (offline-first)

With `offline_first` on (the default), a transformers load never downloads. The provider
finds the model's cached snapshot directory itself (`refs/main`, else the newest snapshot
that has a `config.json` or an `adapter_config.json`). It passes that directory, with
`local_files_only=True`, to `AutoConfig`, `AutoTokenizer` / `AutoProcessor` and the model
class. It never sets `HF_HUB_OFFLINE` or related variables, so child processes are not
affected.

Why a directory rather than the repo id: transformers finds a repo id offline through
`refs/main`. AbstractCore's own downloads pin the commit they listed, and for a pinned
commit huggingface_hub writes no `refs/main`. A load by id from such a snapshot failed
with "couldn't connect to huggingface.co ... couldn't find them in the cached files",
even though every file was present. Transformers also checks the Hub for
`adapter_config.json` even with `local_files_only=True`. Neither happens with a directory.

What a snapshot can be loaded as depends on its files:

| Cached snapshot holds | Result |
|---|---|
| `config.json` + weights | loads, no network |
| `adapter_config.json` (PEFT / LoRA) with a cached base model | the base loads from its snapshot and the adapter is attached from its own directory (needs `peft`) |
| `adapter_config.json` whose base is not cached | `ModelNotFoundError` naming the base and its download command |
| `config.json` without weights | `ModelNotFoundError`: the download did not finish |
| no config at all (for example a README-only diffusion LoRA) | `ModelNotFoundError`: not a transformers model |
| nothing | `ModelNotFoundError`: `download it first: abstractcore models download huggingface <repo>` |

**Adapters need a `peft` that matches your transformers.** AbstractCore does not pin `peft` in
any extra. The minimum is set by transformers itself (`transformers.integrations.peft.MIN_PEFT_VERSION`):
`peft>=0.18.2` for transformers 5.8, and `peft>=0.19.1` for transformers 5.17. With transformers
5.17, peft 0.18.x fails inside `load_adapter` with
`cannot import name '_maybe_shard_state_dict_for_tp'`. The provider checks the installed pair
before it loads the base model. If `peft` is missing, too old, or cannot be imported, the load
raises `ProviderError`, and no raw `ImportError` gets through. For example:
`adapter support needs peft >= 0.19.1 compatible with transformers 5.17.0; installed: peft 0.18.1, transformers 5.17.0. Fix: pip install -U "peft>=0.19.1".`

With `offline_first` and `force_local_files_only` both off, a model that is not fully
cached is passed to transformers by repo id, which may download it. The provider logs
this.

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
