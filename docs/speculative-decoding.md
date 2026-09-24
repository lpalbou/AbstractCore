# Speculative decoding (native MTP)

Multi-token prediction (MTP) makes a model draft several of its own next tokens
and verify them with the target model in one pass. Speed depends on draft
acceptance and verification cost. Compare outputs using the same loaded runtime:
finite-precision kernels and changes of runtime can affect exact token choices.

The native MLX provider exposes this through an optional `speculation` block.
It reports actual execution in `response.metadata["speculation"]`, with a named
reason when acceleration is unavailable. The shared contract does not implement
tensor execution or guarantee MTP support in other provider backends.

## Quick start

```python
from abstractcore import create_llm

llm = create_llm(
    "mlx",
    model="mlx-community/Qwen3.8-27B-4bit",
    speculation={"mode": "native_mtp"},
)

r = llm.generate("Write an LRU cache in Python.", max_output_tokens=400, thinking=False)
print(r.metadata["speculation"])
# {'requested': True, 'mode': 'native_mtp', 'used': True,
#  'drafter': 'mlx-community/Qwen3.8-27B-MTP-4bit', 'num_draft_tokens': 3,
#  'runtime': 'mlx_vlm', 'draft_kind': 'mtp'}
```

## Qwen3.8-Flash-Next on native MLX

Install `abstractcore[mlx]` with MLX ≥0.32.2 and mlx-vlm ≥0.7.1. The `mlx`
provider executes locally in your Python process; no HTTP endpoint or oMLX
installation is used.

```python
from abstractcore import create_llm

llm = create_llm(
    "mlx",
    model="Jundot/Qwen3.8-Flash-Next-oQ4e-mtp",
    speculation={"mode": "native_mtp", "num_draft_tokens": 3,
                 "require_acceleration": True},
)
result = llm.generate("Write a small LRU cache.", thinking=False,
                      temperature=0, max_output_tokens=256)
print(result.content)
print(result.metadata["speculation"])

# Change depth for one request; the following request returns to depth 3.
result = llm.generate("Explain cache expiry.", thinking=False,
                      speculation={"num_draft_tokens": 5})
baseline = llm.generate("Explain cache expiry.", thinking=False, speculation=False)
llm.unload_model(llm.model)
```

`num_draft_tokens` counts proposed tokens, excluding the target seed token.
Depth is fixed per request, not an adaptive ceiling. This is distinct from the
checkpoint's **one trained MTP layer**, which can be reused recurrently.
Direct-mode measurements cover depths 1, 3 and 5; the scheduled two-model
comparison covers depths 1, 2 and 3. Larger depths are accepted but can cost more
memory and run slower. Invalid depths raise immediately. Response metadata
reports actual drafted/accepted counts, rounds and acceptance rate.

The [Jundot Q4 MTP artifact](https://huggingface.co/Jundot/Qwen3.8-Flash-Next-oQ4e-mtp)
occupies approximately 106.32 GB on disk. It uses mixed MLX affine quantization,
not GGUF `IQ4_XS`. Its MTP tensors are embedded and checked during loading.
The [mlx-community 4-bit artifact](https://huggingface.co/mlx-community/Qwen3.8-Flash-Next-4bit)
does not include those tensors. An MTP-related config field alone does not
establish that a quantization includes the trained head.

In the direct-mode M5 Max/128-GiB profile, a 45-token coding prompt generating
256 tokens at temperature 0 measured **59–61 decode tokens/s at depth 3** versus
**37–38 tokens/s with MTP off** (two runs each, same loaded model). Depth 1
measured 54 tokens/s and depth 5 measured 41–43 tokens/s. Outputs matched exactly
across these runs. These are workload-specific measurements, not a guarantee
that every prompt or depth is faster. Peak MLX allocation was approximately
76 GB, excluding OS memory and filesystem cache. These are not measurements of
scheduled-runtime or HTTP throughput. For those metrics and the oMLX comparison,
use the [native MLX benchmark profile](native-mlx-benchmarks.md).

`mlx_ple_offload=True` is the default: n-gram embedding rows are read from the
existing safetensors files with a bounded row cache. No second checkpoint is
written. Full residency is available with `mlx_ple_offload=False`, but needs
approximately 30 GiB more weight memory and is not recommended on a busy
128 GiB machine. Native model weights are shared between provider instances.
Direct mode refuses overlapping calls. Set `mlx_batching=True` for the
[native concurrent runtime](native-mlx-runtime.md): target-only continuous
batching, MTP cohorts, cancellation and optional bounded SSD prefix caching.
Finish or close active streams before unloading their owning provider.

### Native prefix caching and sampling

Pass complete conversation history with each cached request. Use `messages=[]`
for a standalone prompt; an omitted history is rejected to avoid confusing
prefix reuse with hidden-context append semantics.

```python
result = llm.generate(long_prompt, messages=history, prompt_cache_key="session-1",
                      thinking=False, temperature=0)
print(result.usage["cached_input_tokens"])
llm.prompt_cache_clear()
```

Direct mode uses a memory-only cache bounded to 512 MiB with two recurrent-state
checkpoints. Scheduled mode (`mlx_batching=True`) adds a configurable RAM budget
and opt-in SSD persistence; see [native cache controls](native-mlx-runtime.md#bounded-ram-and-optional-ssd-prefix-reuse).
Manual prefill, append, fork, save and load are unsupported. Clearing a specific
key clears the shared native cache and emits an explicit warning.

Temperature, top-p, top-k, min-p and seed are supported with MTP. Non-neutral
presence/frequency/repetition penalties and logit bias require target-only
decoding: use `speculation=False`. With optional acceleration, these controls
produce an explicit unaccelerated outcome; with `require_acceleration=True`,
they raise before generation. They are never silently ignored. Prompted Pydantic
structured output is supported; the Outlines mlx-lm adapter is not used for
Qwen4. Image input is handled by the native vision graph; audio/video containers
must be converted to supported inputs (for video, image frames).

## From the CLI

```bash
# start a chat with the drafter loaded (drafter auto-resolves from the registry)
abstractcore-chat --provider mlx --model mlx-community/Qwen3.8-27B-4bit \
                  --speculation native_mtp

# a drafter that isn't registered, or a hand-tuned draft width
abstractcore-chat --provider mlx --model <target> \
                  --drafter mlx-community/Qwen3.8-27B-MTP-4bit --num-draft-tokens 3

# refuse to start rather than run unaccelerated
abstractcore-chat --provider mlx --model <target> \
                  --speculation native_mtp --require-acceleration
```

Inside the REPL, `/speculation` reports what is actually loaded and lets you
toggle the drafter per turn:

```
> /speculation
⚡ speculation: native MTP lane LOADED
   drafter        : mlx-community/Qwen3.5-4B-MTP-4bit
   draft tokens   : 2
   per-call state : auto (on)

> /speculation off      # lane stays loaded; the drafter is skipped
```

`--speculation` is a **startup** flag, not a runtime one: choosing it decides
which runtime loads the weights. `/speculation off` works mid-session because
skipping a drafter needs no reload; `/speculation on` cannot conjure a lane that
was never loaded, and says so instead of pretending.

## Comparing fairly

This trips people up, so it is worth being explicit. There are two different
comparisons and they answer different questions:

```python
# ❌ measures TWO changes at once
slow = create_llm("mlx", model=M)                                  # runs on mlx-lm
fast = create_llm("mlx", model=M, speculation={"mode": "native_mtp"})  # runs on mlx-vlm
#   -> difference = the drafter AND the library switch

# ✅ measures ONE change
llm = create_llm("mlx", model=M, speculation={"mode": "native_mtp"})
with_mtp    = llm.generate(PROMPT, thinking=False)
without_mtp = llm.generate(PROMPT, thinking=False, speculation={"mode": "off"})
#   -> same object, same weights, same library; only the drafter differs
```

The second snippet reuses **the same `llm`** — that is why no model name appears
on the last line. Nothing is reloaded; `speculation={"mode": "off"}` just tells
that one call to skip the drafter.

Use the first comparison to answer *"is my session faster than before?"* and the
second to answer *"what is MTP itself worth?"*. They can disagree, because
enabling speculation on MLX also swaps mlx-lm for mlx-vlm, and mlx-vlm's own
unaccelerated decoding is not always the same speed — or even the same text —
as mlx-lm's.

## The request block

| field | default | meaning |
|---|---|---|
| `mode` | `"native_mtp"` | `"native_mtp"` or `"off"` |
| `drafter` | from the registry | repo/path of the MTP head, when the runtime needs a separate one |
| `num_draft_tokens` | from the drafter config | how many tokens to draft per step |
| `require_acceleration` | `False` | `True` raises `SpeculationUnavailableError` instead of running unaccelerated |

`speculation=True` is shorthand for `{"mode": "native_mtp"}`; `speculation=False`
pins it off. A malformed block (unknown key, unknown mode, `num_draft_tokens=0`)
raises immediately rather than becoming a silent no-op.

## Which providers can actually do it

Whether the weights contain an MTP head and whether your runtime can *execute*
it are two different questions, and they have different answers per provider.

| provider | native MTP | how |
|---|---|---|
| **mlx** | **yes, in-process** | `speculation={"mode": "native_mtp"}` — this is the only lane AbstractCore accelerates itself |
| **lmstudio** | yes, at load time | `lms load <model> --speculative-draft-mtp`, then use the provider normally |
| openai-compatible → `llama-server` | yes, at launch time | `llama-server --spec-type draft-mtp --spec-draft-n-max 3` |
| **huggingface** (GGUF, in-process) | **no** | see below |
| huggingface (transformers) | no | no MTP execution path |
| vllm | server-side | set `speculative_config` on the vLLM server |
| ollama | no | not exposed |

### Why in-process GGUF cannot do it

llama.cpp itself implements MTP self-speculation, and the graphs are compiled
into the `libllama` that ships with `llama-cpp-python`. What is missing is the
*driver* (`common_speculative_impl_draft_mtp`), which lives in
`libllama-common` — not part of the Python wheel — and `Llama.__init__` exposes
neither `load_mtp` nor `ctx_type`, so there is no way to switch the graphs on.

The provider detects this honestly. It reads the GGUF header for a
`<arch>.nextn_predict_layers` key (evidence, not the filename) and, when a head
is present, warns once with the exact escape hatches. Serve the same file from
`llama-server` or LM Studio and point AbstractCore at it over HTTP.

## Which models have an MTP head

The head is a property of the **artifact**, not the model family, and the two
distributions of the same model differ:

- **GGUF**: the head is embedded. `lmstudio-community/Qwen3.8-27B-GGUF`
  (Q4_K_M) carries `qwen35.nextn_predict_layers=1` and the `blk.64.nextn.*`
  tensors. No sidecar download.
- **MLX**: the head ships **separately**. `mlx-community/Qwen3.8-27B-4bit`
  contains no MTP tensors at all; the head is the 256 MB
  `mlx-community/Qwen3.8-27B-MTP-4bit`.

> **Do not trust `mtp_num_hidden_layers`.** That config key is present in *both*
> MLX repos, including the one whose weights have no MTP tensors. A capability
> probe based on it returns a false positive. AbstractCore's registry entries
> cite artifact-level evidence instead.

Models with a `speculation` block in `model_capabilities.json` today, with the
MLX drafter that gets auto-resolved:

| model | MLX drafter (auto) | status |
|---|---|---|
| `qwen3.8-27b` | `mlx-community/Qwen3.8-27B-MTP-4bit` | tested end-to-end, **1.2–1.5x** |
| `qwen3.6-27b` | `mlx-community/Qwen3.6-27B-MTP-4bit` | drafter config verified, not benchmarked |
| `qwen3.6-35b-a3b` | `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` | drafter config verified, not benchmarked |
| `qwen3.5-9b` | `mlx-community/Qwen3.5-9B-MTP-4bit` | drafter config verified, not benchmarked |
| `qwen3.5-4b` | `mlx-community/Qwen3.5-4B-MTP-4bit` | tested; **too small to benefit** |
| `gemma-4-26b-a4b-it` | `mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit` | tested, **1.12–1.48x**; not output-preserving |
| `qwen3.6-27b-mtp-gguf` | — (GGUF only) | llama.cpp / LM Studio |
| `qwen3.6-35b-a3b-mtp-gguf` | — (GGUF only) | llama.cpp / LM Studio |

"drafter config verified" means the repo's `config.json` was fetched and reports
`model_type: qwen3_5_mtp` with a `text_config.hidden_size` matching its target —
artifact evidence, not a timing measurement.

Any drafter can also be pointed at explicitly, whether or not it is registered:

```python
create_llm("mlx", model="<target>",
           speculation={"mode": "native_mtp", "drafter": "<drafter repo>"})
```

The MLX drafter is a **companion** of the model: `abstractcore models download mlx <model>`
(and the Gateway's model downloads) fetch it in the same job, and the model counts as
installed only when both are there (see [MTP companions](models.md#mtp-companions)). When
the companion is missing, the model still loads and answers, without MTP. The response's
`speculation` block then says so in words, and the Gateway consoles show that text:

```text
{'used': False, 'reason': 'mtp_head_not_cached',
 'message': 'MTP acceleration off: companion mlx-community/Qwen3.5-9B-MTP-4bit (the MTP head
             for mlx-works/Qwen3.5-9B-oQ4e-mtp) is not downloaded; download it with
             `abstractcore models download mlx mlx-community/Qwen3.5-9B-MTP-4bit` ...'}
```

### MTP-preserving checkpoints never load through mlx-lm

Some MLX checkpoints keep the model's `mtp.*` tensors in their own weights, for example
`mlx-works/Qwen3.5-9B-oQ4e-mtp`, `Jundot/Qwen3.8-27B-oQ4e-mtp` and Flash-Next. The MLX
provider loads these checkpoints through **mlx-vlm in every lane**: with or without a drafter,
`speculation` on, off or inherited, batching or not. It detects them from the checkpoint's
`model.safetensors.index.json` `weight_map` (or, for a single file, its safetensors header), a
local read of a few KB. If mlx-vlm cannot load such a checkpoint, the provider raises a
`ProviderAPIError` that names the model and the reason; it never falls back to mlx-lm. The
native lane is also the one that runs the MTP drafter and prefix caching.

If you load these checkpoints outside AbstractCore, use mlx-vlm as well. The `qwen3_5` loader
in released mlx-lm versions (0.31.3 and earlier) treats any `mtp.` tensor as a sign of an
unconverted checkpoint and shifts the RMSNorm weights a second time, which produces meaningless
output without an error (the loader on mlx-lm `main` is corrected by ml-explore/mlx-lm#1623).
`abstractcore models verify <repo>` checks that an installed model answers sensibly (see
[Checking a fresh install](models.md#checking-a-fresh-install-models-verify)).

mlx-community also publishes drafters for `Qwen3.5-122B-A10B`,
`DeepSeek-V4-Flash` (bf16 only) and `Hy3-preview`. Of these, mlx-vlm has a
drafter implementation for DeepSeek-V4 (`deepseek_v4_mtp`) but **not** for Hy3,
so the Hy3 drafter has no in-process consumer here.

### Other families — surveyed from GGUF headers, not model names

Each row below was checked by range-fetching the first 3 MB of a real Q4 GGUF
and reading its `<arch>.nextn_predict_layers` key, so "MTP" means the head is in
the file, not that the name says so.

| family | example artifact | arch | MTP head |
|---|---|---|---|
| DeepSeek-V4 | `antirez/deepseek-v4-gguf` | `deepseek4` | **yes (1)** |
| GLM-5.2 | `unsloth/GLM-5.2-GGUF` | `glm-dsa` | **yes (1)** |
| GLM-5.3-Flash | `unsloth/GLM-5.3-Flash-GGUF` | `glm5next` | **yes (1)** |
| Hunyuan Hy3 | `AngelSlim/Hy3-GGUF` | `hy_v3` | **yes (1)** |
| MiMo | `AesSedai/MiMo-V2.5-GGUF` | `mimo2` | **yes (3)** |
| Ling 3.0 flash | `bloomer010/Ling-3.0-flash-REAP288-73B-A5B-GGUF` | `bailingmoe3` | **yes (1)** |
| Nemotron Labs 3 | `RemySkye/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-GGUF` | `nemotron_h_moe` | **yes (2)** |
| Qwen3-Next 80B | `Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF` | `qwen3next` | no |
| Step 3.5 Flash | `ggml-org/Step-3.5-Flash-GGUF` | `step35` | no |
| Cohere command-a | `bartowski/command-a-plus-05-2026-GGUF` | `cohere2moe` | no |

The last three are cases where llama.cpp implements the MTP graph for the
architecture but the published quants carry no head — the runtime is ready and
the weights are not.

Caveat on the two GLM rows: `glm5next` is **not** among the `graph_mtp` symbols
in the llama.cpp build used here, so that one needs a newer runtime than
`glm-dsa` does.

llama.cpp additionally implements MTP graphs for `deepseek2/32/4`, `glm4_moe`,
`glm_dsa`, `qwen35`, `qwen35moe`, `qwen3next`, `bailingmoe3`, `cohere2moe`,
`hy_v3`, `mimo2`, `nemotron_h_moe` and `step35` — so a GGUF of any of those that
carries nextn tensors will accelerate under `llama-server`, whether or not it
has a registry entry yet.

## Measured results

The following tables describe the legacy direct adapter and its original
seed-inclusive backend block sizes, not the scheduled native runtime. Use a
matched runtime/version when reproducing them; do not compare these decode
figures directly with HTTP end-to-end throughput.

M5 Max (128 GB), 4-bit, batch 1, greedy, 200 tokens, medians of 3 interleaved
reps with nothing else resident.

**Read the control carefully.** The MLX baseline here is the *same provider with
`speculation={"mode": "off"}` per call* — not a separate non-speculating
provider. That matters: turning speculation on also switches the MLX lane from
mlx-lm to mlx-vlm, so comparing against a plain provider measures two changes at
once. The numbers below isolate the drafter.

**MLX, Qwen3.8-27B-4bit:**

| prompt | drafter off | MTP | speedup | output |
|---|---|---|---|---|
| code | 19.9 tok/s | 28.5 tok/s | **1.43x** | identical |
| prose | 20.2 tok/s | 24.0 tok/s | **1.19x** | identical |
| repetitive table | 19.1 tok/s | 28.0 tok/s | **1.47x** | identical |

**MLX, Qwen3.5-4B-4bit — MTP does not pay here:**

| prompt | drafter off | MTP | speedup |
|---|---|---|---|
| code | 106.0 tok/s | 100.5 tok/s | **0.95x** |
| prose | 110.7 tok/s | 80.4 tok/s | **0.73x** |
| repetitive table | 103.6 tok/s | 116.6 tok/s | 1.13x |

A 4B target is cheap enough per decode step that the drafter — a full extra
block plus a pass over the 248k-row LM head per drafted token — barely
amortises. **MTP is a big-model optimisation.** Measure before enabling it on
anything small.

**llama.cpp (`llama-server --spec-type draft-mtp --spec-draft-n-max 3`), same GGUF:**

| prompt | baseline | MTP | speedup | acceptance | mean accepted |
|---|---|---|---|---|---|
| code | 26.3 tok/s | 36.9 tok/s | 1.40x | 0.88 | 3.63 |
| prose | 20.1 tok/s | 24.6 tok/s | 1.22x | 0.50 | 2.51 |
| repetitive table | 19.2 tok/s | 35.7 tok/s | 1.86x | 1.00 | 3.98 |

Speedup tracks the acceptance rate, which is prompt-dependent: predictable,
structured output (code, tables, reformatting) drafts well; discursive prose
drafts worst. Expect roughly 1.2x–1.9x on a 27B, not a fixed number.

### Output fidelity — two different questions

1. **Does the drafter change the answer?** *Algorithmically, no* — mlx-vlm's
   acceptance test is exact (`target token == drafted token`, otherwise reject
   and keep the target's token). *Numerically, sometimes.* Verification
   recomputes logits from a batched hidden state, which changes Metal's
   reduction order, so a near-tie argmax can flip. Measured: both Qwen pairings
   were byte-identical on all three prompts (that is what the gated live test
   pins), while `gemma-4-26B-A4B` diverged on 2 of 3 — deterministically, at
   ~0.87 similarity. Registry entries carry `output_preserving: false` where a
   divergence has been observed, and the response metadata repeats it.
   Do not assume bit-reproducibility on an untested pairing.
2. **Does turning speculation on change the answer?** On MLX it can, because the
   lane switches library. On `Qwen3.5-4B-4bit`, mlx-lm and mlx-vlm produce
   *different* greedy text for the same prompt with no drafter involved; on
   `Qwen3.8-27B-4bit` they agreed. Both are deterministic run to run.
   The llama.cpp lane likewise differed from its own baseline on the prose
   prompt (batched verification changes matmul shapes, so a near-tie flips).

### Tuning `num_draft_tokens`

The drafter's config declares a `block_size`, but that is an architecture
property, not a tuned runtime value — and following it blindly can cost you the
whole win. Measured sweeps (200 tokens, code prompt):

| backend block size (seed included) | Qwen3.5-4B | Qwen3.8-27B |
|---|---|---|
| 1 | **0.14x** | — |
| 2 | **1.16x** | 1.39x |
| 3 | 1.09x | 1.56x |
| 4 | 0.98x *(its config's value)* | **1.57x** |
| 5 | 0.89x | 1.49x |

Backend block size 1 in that measurement has **zero proposals**. The public
`num_draft_tokens=1` means one proposal and maps to backend block size 2; it is
not that pathological setting. Registry defaults are starting points, not
universal optima. Sweep proposal depth with your prompts and concurrency.

## Native backend boundaries

- The Qwen3.8-27B separate-head native path and Flash-Next embedded-head path
  both accept images and native automatic prefix reuse. Full histories remain
  required. Native caches cannot consume ordinary mlx-lm durable append-cache
  artifacts; see [native runtime caching](native-mlx-runtime.md).
- **mlx-lm's own `draft_model` does not work here** and is deliberately not
  offered: Qwen3.5/3.8 are hybrids with linear-attention layers, and mlx-lm
  raises `Speculative decoding requires a trimmable prompt cache
  (got {'ArraysCache'})`.
