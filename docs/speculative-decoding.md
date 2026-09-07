# Speculative decoding (native MTP)

Multi-token prediction (MTP) makes a model draft several of its own next tokens
and verify them in one pass. It is a pure speed trade: a drafted token is kept
only when the target model would have produced it anyway, so the answer does not
change — it just arrives sooner.

AbstractCore exposes this through one optional `speculation` block, and it will
never pretend to have applied it. If a request cannot be honored you get a named
reason, in the log and in `response.metadata["speculation"]`.

## Quick start

```python
from abstractcore import create_llm

llm = create_llm(
    "mlx",
    model="mlx-community/Qwen3.8-27B-4bit",
    speculation={"mode": "native_mtp"},
)

r = llm.generate("Write an LRU cache in Python.", max_tokens=400, thinking=False)
print(r.metadata["speculation"])
# {'requested': True, 'mode': 'native_mtp', 'used': True,
#  'drafter': 'mlx-community/Qwen3.8-27B-MTP-4bit', 'num_draft_tokens': 3,
#  'runtime': 'mlx_vlm', 'draft_kind': 'mtp'}
```

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
| `qwen3.6-27b` | `mlx-community/Qwen3.6-27B-MTP-4bit` | drafter config verified, not run here |
| `qwen3.6-35b-a3b` | `mlx-community/Qwen3.6-35B-A3B-MTP-4bit` | drafter config verified, not run here |
| `qwen3.5-9b` | `mlx-community/Qwen3.5-9B-MTP-4bit` | drafter config verified, not run here |
| `qwen3.5-4b` | `mlx-community/Qwen3.5-4B-MTP-4bit` | tested; **too small to benefit** |
| `gemma-4-26b-a4b-it` | `mlx-community/gemma-4-26B-A4B-it-qat-assistant-4bit` | tested, **1.12–1.48x**; not output-preserving |
| `qwen3.6-27b-mtp-gguf` | — (GGUF only) | llama.cpp / LM Studio |
| `qwen3.6-35b-a3b-mtp-gguf` | — (GGUF only) | llama.cpp / LM Studio |

"drafter config verified" means the repo's `config.json` was fetched and reports
`model_type: qwen3_5_mtp` with a `text_config.hidden_size` matching its target —
artifact evidence, but not a timing measurement on this machine.

Any drafter can also be pointed at explicitly, whether or not it is registered:

```python
create_llm("mlx", model="<target>",
           speculation={"mode": "native_mtp", "drafter": "<drafter repo>"})
```

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

| block size | Qwen3.5-4B | Qwen3.8-27B |
|---|---|---|
| 1 | **0.14x** | — |
| 2 | **1.16x** | 1.39x |
| 3 | 1.09x | 1.56x |
| 4 | 0.98x *(its config's value)* | **1.57x** |
| 5 | 0.89x | 1.49x |

`num_draft_tokens=1` is pathological — a ~7x slowdown — and the provider warns
if you ask for it. The registry ships the measured optimum per model (2 for the
4B, 3 for the 27B); omit `num_draft_tokens` to take it.

## Limitations of the MLX lane

- **Text only.** Enabling speculation swaps the whole runtime to mlx-vlm, whose
  speculative loop does not take the vision add-on's precomputed embeddings. An
  image request under speculation raises rather than silently answering without
  the picture. Build a second provider with `speculation={"mode": "off"}` for
  media.
- **No warm prompt-cache reuse.** The keyed snapshot cache is built from mlx-lm
  cache objects; feeding one into the speculative loop risks a silent desync, so
  the lane declines it and warns once. Prompts are prefilled fresh.
- **mlx-lm's own `draft_model` does not work here** and is deliberately not
  offered: Qwen3.5/3.8 are hybrids with linear-attention layers, and mlx-lm
  raises `Speculative decoding requires a trimmable prompt cache
  (got {'ArraysCache'})`.

## Running the tests

```bash
# hermetic (no models, no network)
python -m pytest -q tests/providers/test_mtp_adv_*.py

# live, needs the 15 GB target and its drafter cached
ABSTRACTCORE_RUN_MLX_TESTS=1 ABSTRACTCORE_RUN_MTP_TESTS=1 \
  python -m pytest -q tests/providers/test_mtp_mlx_live.py
```
