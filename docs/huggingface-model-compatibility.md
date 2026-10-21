# HuggingFace Model Compatibility

This guide explains which HuggingFace local model formats AbstractCore can load directly and how to
diagnose model-load failures. These checks are general provider compatibility checks; they are not
specific to prompt caching, memory blocs, tools, or structured output.

## Baseline Install

`pip install "abstractcore[apple]"` (Apple silicon) and `pip install "abstractcore[gpu]"` (NVIDIA /
AMD) install the stable HuggingFace provider stack:

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
The light install carries `abstractvoice>=0.13.1` without OmniVoice, torch, or torchaudio. Local
OmniVoice engines come with `abstractcore[apple]` and `abstractcore[gpu]`.

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

### Tencent Hy-Memory-30B-A3B (`hy_v3`)

`tencent/Hy-Memory-30B-A3B` is a 30B-total / 3B-active MoE whose specialty seems to be a
long-term memory writer (the write path of Mem0, MemOS and Hy-Memory). AbstractCore runs it on
all three local stacks:

| Stack | Target | Notes |
|---|---|---|
| `huggingface` (transformers) | `tencent/Hy-Memory-30B-A3B` (BF16, 60 GB) | needs a Transformers build with `model_type=hy_v3` (5.8.0 measured) |
| `huggingface` (llama.cpp) | `tencent/Hy-Memory-30B-A3B-GGUF:Q4_K_M` (also `Q6_K`, `Q8_0`) | llama.cpp arch `hy_v3`; llama-cpp-python 0.3.35 measured |
| `mlx` | `mlx-community/Hy-Memory-30B-A3B-4bit` (16.9 GB) or `-8bit` (32.0 GB) | mlx-lm ships no `hy_v3` module; AbstractCore provides one, and the repos bundle it |

`tencent/Hy-Memory-30B-A3B-FP8` (F8_E4M3) is a vLLM/SGLang artifact and has no Apple path.

Things AbstractCore handles for this model:

- **Expert width.** `config.json` says `moe_intermediate_size: 1536`, but every routed and shared
  expert tensor is 768 wide (`expert_hidden_dim: 768`). Transformers 5.8 sizes experts from
  `moe_intermediate_size`, so a plain `from_pretrained` fails on a shape mismatch. The
  transformers loader sizes them from `expert_hidden_dim` and logs the change.
- **MoE kernels.** Transformers 5 runs MoE experts with `grouped_mm` whenever torch exposes the
  op; torch 2.8 has no CPU or MPS kernel for it. The loader probes the device once and uses
  Transformers' `eager` experts loop where the kernel is missing.
- **Thinking.** The chat template has one variable, `reasoning_effort` (`high`, `low` or
  `no_think`), and no `enable_thinking` switch. Unset, it renders `no_think`: thinking is off by
  default. `thinking=False` sends `no_think`, and `thinking="low"` / `"high"` send that level
  (registry `thinking_control.effort_disable_value`). A bare `thinking=True` picks the first of
  medium, high, low that the model declares, which is `high` here: the most expensive level.
  For a memory writer, ask for `thinking="low"` when you want reasoning at a lower cost.
  vLLM receives the value as `chat_template_kwargs.reasoning_effort`; OpenAI-compatible servers
  receive `low`/`high` as the top-level `reasoning_effort`, and `no_think` only inside
  `chat_template_kwargs` (a validating server rejects it top-level).
- **GGUF lanes.** The embedded template renders text requests, so all three values reach it.
  Requests that leave that lane (tool-result history in `messages`, structured output, media)
  go through llama-cpp-python's chat completion, which has no transport for the variable: there,
  `thinking="off"` still holds because the template's default is `no_think`, but
  `thinking="low"` or `"high"` cannot be applied, and a warning says so.
- **Tool calls.** `<tool_calls><tool_call>name<tool_sep><arg_key>k</arg_key><arg_value>v</arg_value></tool_call></tool_calls>`
  is parsed on the plain, streaming and passthrough lanes.

For memory extraction the model card uses `temperature=0.0`, `max_tokens=512` and no thinking:

```python
from abstractcore import create_llm

llm = create_llm("huggingface", model="tencent/Hy-Memory-30B-A3B-GGUF:Q4_K_M")
resp = llm.generate(
    "My name is Alex. I live in Seattle and I'm vegetarian.",
    system_prompt='Extract explicit user facts. Return only a JSON object with a "memories" list of strings.',
    thinking=False,
    temperature=0.0,
    max_tokens=512,
)
```

On Apple/MPS with torch 2.8, `scaled_dot_product_attention(is_causal=True)` in bfloat16 or
float16 returns wrong values once a prompt is longer than about 20 tokens (measured against a
float64 reference: maximum error 3.5 on unit-scale inputs; float32 and an explicit mask are
exact). Every half-precision transformers model on that stack is affected; for
Hy-Memory-30B-A3B the BF16 checkpoint produced unusable text. AbstractCore warns at load on
torch 2.8.x; run with `ABSTRACTCORE_TRANSFORMERS_ATTN_IMPL=eager`, or upgrade torch.

MLX conversions are made with mlx-lm's own `convert`, with AbstractCore's class registered as
`mlx_lm.models.hy_v3`:

```python
import abstractcore.providers.mlx_hy_v3 as hy

hy.register_with_mlx_lm()  # only when the installed mlx-lm has no hy_v3 module
from mlx_lm.convert import convert

convert(hf_path="tencent/Hy-Memory-30B-A3B", mlx_path="Hy-Memory-30B-A3B-4bit",
        quantize=True, q_group_size=64, q_bits=4)
```

The class keeps the MoE router and its correction bias in full precision when quantizing
(`Model.quant_predicate`). Copy `abstractcore/providers/mlx_hy_v3.py` into the output as
`hy_v3.py` and add `"model_file": "hy_v3.py"` to its `config.json` so plain mlx-lm can load it.
For checkpoints whose `config.json` says `model_type: hy_v3`, `create_llm("mlx", ...)` does not
run a bundled `model_file`: it uses AbstractCore's class, or mlx-lm's own module once mlx-lm
ships one. (A checkpoint that declares another `model_type` plus a `model_file` still runs that
file through mlx-lm, as before.) Weights use the mlx-lm MoE names (`mlp.switch_mlp.*`,
`mlp.router.gate`, `mlp.router.expert_bias`), as in `mlx-community/Hy3-preview-*`; the tencent,
transformers and earlier community layouts are mapped onto them at load. A layout the selected
class cannot load fails with an error that names the layout found.

The MLX class follows transformers' `HYV3ForCausalLM` except in two places, both more precise:
it sums the routed experts in float32 (transformers accumulates them in the model dtype), and it
computes an unquantized `lm_head` in float32 when `enable_lm_head_fp32` is set (transformers
ignores the flag). Small bf16 differences against transformers are therefore expected. Measured
on the BF16 weights and a 100-token prompt: the same top-5 next tokens in the same order, logit
correlation 0.9965 at the last position, the same most likely token at 96 of 100 positions.

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

Loading from the snapshot directory rather than the repo id means a cached model loads even
when its repo has no `refs/main` (transformers needs that ref to resolve a repo id offline),
and transformers makes no Hub request for `adapter_config.json`. Other tools that load by repo
id still need `refs/main`; `abstractcore models repair-refs` writes it (see
[Repairing `refs/main`](models.md#repairing-refsmain)).

What a snapshot can be loaded as depends on its files:

| Cached snapshot holds | Result |
|---|---|
| `config.json` + weights | loads, no network |
| `adapter_config.json` (PEFT / LoRA) with a cached base model | the base loads from its snapshot and the adapter is attached from its own directory (needs `peft`) |
| `adapter_config.json` whose base is not cached | `ModelNotFoundError` naming the base and its download command |
| `config.json` without weights | `ModelNotFoundError`: the download did not finish |
| no config at all (for example a README-only diffusion LoRA) | `ModelNotFoundError`: not a transformers model |
| nothing | `ModelNotFoundError`: `download it first: abstractcore models download huggingface <repo>` |

**Adapters need a `peft` that matches your transformers.** AbstractCore does not pin `peft` itself;
it arrives with `abstractcore[apple]` and `abstractcore[gpu]` (through AbstractVision). The minimum is set by transformers itself (`transformers.integrations.peft.MIN_PEFT_VERSION`):
`peft>=0.18.2` for transformers 5.8, and `peft>=0.19.1` for transformers 5.17. With transformers
5.17, peft 0.18.x fails inside `load_adapter` with
`cannot import name '_maybe_shard_state_dict_for_tp'`. The provider checks the installed pair
before it loads the base model. If `peft` is missing, too old, or cannot be imported, the load
raises `ProviderError`, and no raw `ImportError` gets through. For example:
`adapter support needs peft >= 0.19.1 compatible with transformers 5.17.0; installed: peft 0.18.1, transformers 5.17.0. Upgrade with: pip install -U "abstractcore[apple]".` (on Linux the message names
`abstractcore[gpu]`, and so does Windows x86_64; on an Intel Mac or Windows on ARM, where no setting provides the Transformers engine,
it says so).

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
- Consider adding verified quantization runtimes (AWQ, GPTQ, compressed-tensors) to the `apple`
  and `gpu` settings, only after dependency compatibility is verified.
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
