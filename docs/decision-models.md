# Decision Models

Decision models answer typed questions about a document with calibrated probabilities instead of
generating text. AbstractCore surfaces them as the `decision` output modality — **text in,
decisions out** — behind one abstraction that does not depend on any single model or vendor:

- **Hosted**: TypeSafe's Jev (`typesafe` provider) and any System One-compatible endpoint, that is
  any server speaking the public System One decision API (`systemone` provider: a self-hosted Kev or
  Lev process, another AbstractCore server, or another host). Nothing but `httpx`; the light install
  is enough.
- **Local**: the Kev family (pointer head on Qwen3.5), Lev (label logits on Qwen3.5-4B) and Laya
  (ModernBERT encoder with a marker scorer), all Apache-2.0, on the `mlx` provider (Apple silicon)
  and the `huggingface` provider (transformers, or a GGUF through llama.cpp). OpenJev-style
  checkpoints run through the same label-logit engine (their weights are CC-BY-NC-4.0).

The request, the answers and the HTTP surface (`POST /v1/decisions`) are the same for every one of
them, so you switch models and providers without changing your code. Use a decision model when you
classify, route, triage, score, or check documents and need probabilities you can threshold —
send the uncertain cases to a human, automate the rest. Kev-4B answers several questions about a
support ticket in about 100 ms on Apple silicon; Laya in about 50 ms on a laptop CPU/GPU; no prompt
engineering and no output parsing.

## What a decision model returns

One request carries one **state** (the document: a string, or a JSON object/array rendered as
labelled text) and any number of named **questions**. Each question has one of three types:

| Type | You supply | You get |
|---|---|---|
| `choice` | 1–255 named options, each with an optional description | the most likely option (`choice`), `probabilities` per option, `confidence` |
| `noul` | a statement in `instructions` (optional `criteria` describing `true` / `false`) | `noul`: the probability that the statement is true |
| `score` | 1–255 ordered levels, lowest first | `score`: the expected level index, `probabilities` per level, `legend`, `confidence` |

Questions never influence one another: every question is scored against the state alone. The model
only ever scores the options you give it; it never writes an option of its own. Confidence is
derived from the distribution (0 at uniform, 1 at certainty) and follows the System One reference
conventions, so answers are comparable across hosted and local decision models.

## Quick start

Download the model once (no silent downloads: AbstractCore loads only what is on disk):

```bash
# Apple silicon (MLX) or transformers: the adapter + its base model
abstractcore models download mlx jaredpalmer/kev-4b
abstractcore models download mlx Qwen/Qwen3.5-4B-Base
# or the self-contained GGUF (llama.cpp), 3 GB
abstractcore models download huggingface "ggml-org/Kev-4B-GGUF:Q4_K_M"
```

Then ask:

```python
from abstractcore import create_llm

llm = create_llm("mlx", model="jaredpalmer/kev-4b")   # or ("huggingface", model="ggml-org/Kev-4B-GGUF:Q4_K_M")

ticket = "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card."
result = llm.generate(ticket, output="decision", questions={
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                "shipping": "Delivery status, delays, lost packages",
                                "billing": "Charges, invoices, payment problems"}},
    "escalate":   {"type": "noul", "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]},
})

result.decisions["department"].choice          # "returns"
result.decisions["department"].probabilities   # {"returns": 0.51, "shipping": 0.36, "billing": 0.13}
result.decisions["escalate"].noul              # 0.72
result.decisions["frustration"].score          # 1.30  (between "Frustrated" and "Very angry")
result.decision.to_systemone()                 # the System One response body
```

The document is read once per request and reused for every question in it, and the last four
documents stay cached, so consecutive requests about the same document only pay for their questions
(Kev-4B 8-bit on an M5 Max: a 6,300-token contract costs 1.4 s the first time and 0.1 s again; a
100-token ticket 95 ms then 45 ms). `create_llm(..., decision_prefix_cache_size=0)` disables the cache,
a larger value keeps more documents; every `decision_<option>=` kwarg reaches the decision loader.

`generate(..., output="decision")` returns a `MultimodalGenerateResponse` whose `decision` is a
`DecisionResponse` (`answers`, `usage`, `latency_ms`) and whose `decisions` property is the answers
keyed by question name. `result.content` is the answers as JSON, so sessions and logs that read
text still see something meaningful. The state is the prompt; pass `state=` (any JSON) to render an
object instead.

Python shorthand is accepted for questions: a string is a `noul` statement, a list of names is a
`choice` with no descriptions, `{"instructions": ..., "levels": [...]}` is a `score`:

```python
llm.generate(ticket, output="decision", questions={
    "team": ["returns", "shipping", "billing"],
    "urgent": "Does this need urgent human attention?",
    "mood": {"instructions": "How frustrated is the customer?", "levels": ["Calm", "Frustrated", "Very angry"]},
})
```

A decision model has no text lane: `generate("hello")` on it raises with a pointer to
`output="decision"`. Text generation on other models is unchanged.

### From any provider, through the default route

Set the `output.decision` capability default once and every provider instance can answer
`output="decision"`, the way `output="voice"` runs the configured voice engine:

```bash
abstractcore config set-default output.decision --provider mlx --model jaredpalmer/kev-4b
```

```python
llm = create_llm("ollama", model="qwen3.5:9b")           # a text model
result = llm.generate(ticket, output="decision", questions={...})   # answered by the configured Kev
```

The route resolver puts the configured provider and model on the output spec; the decision lane
runs them through the process-wide pool (`abstractcore.decision.pool`): at most two decision
models stay loaded, the least recently used is unloaded, and the server's `/v1/decisions` shares
the same pool, and a model a request is using is never unloaded under it. An explicit route on the
spec wins over the default: `output={"modality": "decision", "provider": "mlx", "model":
"jaredpalmer/kev-0.8b"}`, or a `provider/model` handle in `model` (`"huggingface/ggml-org/Kev-4B-GGUF:Q4_K_M"`),
while a bare `model` picks the local engine (a GGUF on huggingface, else mlx when mlx-lm is installed)
and `kev-latest` reuses the most recently used resident model. `decision_*` route options
(`decision_prefix_cache_size`, `decision_n_ctx`, ...) reach the loader. With no route and no
default, a provider that is not itself a decision model raises an error that names the
`set-default` command. The answers' `model` is always `provider/model`.

### Without a provider object

```python
from abstractcore.decision import load_decision_model

model = load_decision_model("mlx", "jaredpalmer/kev-4b")
response = model.decide(ticket, {"urgent": "Does this need urgent human attention?"})
response["urgent"].noul
```

`load_decision_model(provider, model)` is cache-only; pass `allow_download=True` to fetch the
snapshot (and the base model of an adapter) first.

## Over HTTP: `POST /v1/decisions`

The AbstractCore server answers decisions at `POST /v1/decisions` with the System One request and
answer shape (the format every decision-model host shares; `/{provider}/v1/decisions` takes an
unprefixed model). The Python shorthand works on this route too (a string question is a noul, a
list of option names a choice); hosted System One servers take only the typed form:

```bash
curl -s http://localhost:8000/v1/decisions -H 'content-type: application/json' -d '{
  "model": "mlx/jaredpalmer/kev-4b",
  "state": "Shoes arrived two weeks late and in the wrong size. Also I see two charges on my card.",
  "questions": {
    "department": {"type": "choice", "instructions": "Which team should handle this?",
                   "criteria": {"returns": "Exchanges, refunds, wrong or damaged items",
                                "shipping": "Delivery status, delays, lost packages",
                                "billing": "Charges, invoices, payment problems"}},
    "escalate":   {"type": "noul", "instructions": "Does this need urgent human attention?"},
    "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                    "criteria": ["Calm", "Frustrated", "Very angry"]}
  }}'
```

```json
{"model": "mlx/jaredpalmer/kev-4b",
 "answers": {
   "department": {"type": "choice", "choice": "returns", "confidence": 0.2681,
                  "probabilities": {"returns": 0.5121, "shipping": 0.3554, "billing": 0.1326}},
   "escalate": {"type": "noul", "noul": 0.7242},
   "frustration": {"type": "score", "score": 1.3023, "confidence": 0.422,
                   "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                   "probabilities": {"0": 0.0415, "1": 0.6146, "2": 0.3438}}},
 "usage": {"input_tokens": 101, "output_tokens": 181}, "latency_ms": 43.4}
```

`GET /v1/models?output_type=decision` lists the decision models the server can reach; each entry
carries `capabilities` (`input` / `output`) and a `decision` object naming this endpoint. A chat
completion against a decision model is refused with a 400 that points here: OpenAI has no shape for
calibrated typed decisions, and the System One format is the one every decision-model host shares.

`model` takes `provider/model` (`mlx/jaredpalmer/kev-4b`, `huggingface/ggml-org/Kev-4B-GGUF:Q4_K_M`,
`huggingface/jaredpalmer/kev-4b`). A bare handle picks the local engine (mlx on Apple silicon when
mlx-lm is installed, otherwise huggingface); `kev-latest` and `jev-latest` reuse the decision model
this server loaded most recently, else the configured `output.decision` default. The first request
loads the model, later ones reuse it, and the server keeps the two most recently used decision
models resident (the same pool `generate(output="decision")` uses in that process). The pool is part of
the model residency API: `POST /acore/models/load` with `{"task": "decision", "model": "mlx/jaredpalmer/kev-4b"}`
loads a decision model ahead of the first request, `GET /acore/models/loaded?task=decision` lists the
resident ones (`process:decision:<provider>:<model>` rows), and `POST /acore/models/unload` releases one. Errors: 400 for an invalid
request, 404 when the model is not downloaded, 422 when the state is over the context limit.

## Providers, models and formats

| Provider | Model handle | Kind | Engine | Needs |
|---|---|---|---|---|
| `typesafe` | `jev-latest` (also `jev-preview`) | hosted System One | HTTP | `TYPESAFE_API_KEY` |
| `systemone` | any id the host serves (`base_url=` required) | hosted System One | HTTP | the host's key, if any |
| `mlx` | `AbstractFramework/kev-4b-mlx-8bit`, `AbstractFramework/kev-4b-mlx-4bit` (ready-to-run conversions) or `jaredpalmer/kev-4b` (+ base `Qwen/Qwen3.5-4B-Base`) | kev (pointer head) | mlx-lm | `abstractcore[apple]` |
| `mlx` | `AbstractFramework/kev-0.8b-mlx-8bit`, `AbstractFramework/kev-0.8b-mlx-4bit` or `jaredpalmer/kev-0.8b` (+ base `Qwen/Qwen3.5-0.8B-Base`) | kev | mlx-lm | the small, fast member of the family |
| `huggingface` | `jaredpalmer/kev-4b` (+ base), a conversion | kev | transformers | `abstractcore[apple]` / `[gpu]` |
| `huggingface` | `ggml-org/Kev-4B-GGUF:Q4_K_M` / `Q8_0` | kev | llama.cpp | any released llama-cpp-python (see below) |
| `huggingface` | `ggml-org/lev-GGUF:Q8_0` / `Q4_K_M` | lev (label logits) | llama.cpp | any released llama-cpp-python |
| `mlx` / `huggingface` | `interfaze-ai/lev` (+ base `Qwen/Qwen3.5-4B`) | lev | mlx-lm / transformers | the base model on disk |
| `huggingface` | `convaiinnovations/laya` (also `.../multilingual`, `.../typed-decisions`) | laya (marker scorer) | transformers | `abstractcore[apple]` / `[gpu]`; long states scanned in overlapping 512-token windows |
| `mlx` / `huggingface` | `openjev/openjev-MLX-4bit`, `ggml-org/OpenJev-GGUF:Q4_K_M`, `openjev/openjev` | openjev (letter logits) | mlx-lm / llama.cpp / transformers | weights are CC-BY-NC-4.0 |

```python
# Hosted Jev: the light install, nothing local.
llm = create_llm("typesafe", model="jev-latest")           # TYPESAFE_API_KEY in the environment
result = llm.generate(ticket, output="decision", questions=questions)

# Any System One-compatible endpoint (here a self-hosted Kev server on this machine).
llm = create_llm("systemone", model="kev-latest", base_url="http://localhost:8009")

# Local models: the same call.
llm = create_llm("huggingface", model="convaiinnovations/laya")
llm = create_llm("huggingface", model="ggml-org/lev-GGUF:Q8_0")
```

The hosted route also serves `POST /v1/decisions` on the AbstractCore server with
`"model": "typesafe/jev-latest"`, so one server can front local and hosted decision models.

### Decision kinds

A *kind* is a readout mechanism; every checkpoint of a kind shares its prompt layout and how the
probabilities are read, and differs only in weights and calibration:

- **kev** — five delimiter tokens pack the state and each question; a pointer head scores every
  option's end token against the question's decide token. The state is computed once and cached;
  each question only pays for its own tokens.
- **lev** — a chat prompt lists the options under letters; the answer is read from the next-token
  logits of the label tokens. A choice is read in two option orders and averaged; a yes/no is a
  0–8 rating. The adapter runs at a quarter of its trained strength, as released. The tokens every
  prompt of a request shares (system text and state) run once and are reused for every question
  and option order, on transformers, mlx-lm and llama.cpp alike.
- **openjev** — the same mechanism with one letter per option (up to 52 options) and fixed
  temperatures.
- **laya** — one bidirectional ModernBERT pass per question (`[CLS] question [SEP] [MASK] option …
  [SEP] state [SEP]`, 512 tokens), a two-layer head and a scorer at each `[MASK]`. A state longer
  than the room the questions leave is scanned in overlapping windows (half-window stride) and the
  windows are combined as Laya's own runtime does: a yes/no is the maximum P(true) over windows, a
  choice or score is the most confident window's distribution (`long_state="truncate"` reads the
  start only). The number read is the deciding window's, not a calibration over the whole document.
- **systemone** — a remote endpoint; the host decides.

### Artifact layouts

- **Adapter** (`jaredpalmer/kev-4b`, `interfaze-ai/lev`): the published LoRA; needs its base model
  on disk (download both). AbstractCore folds the adapter into the base at load (about a minute
  for a 4B model).
- **Merged** (an AbstractCore conversion, or a complete checkpoint such as `convaiinnovations/laya`):
  loads directly, in transformers or mlx-lm layout, bf16 or quantized.
- **GGUF** (`ggml-org/Kev-4B-GGUF`, `ggml-org/lev-GGUF`, `ggml-org/OpenJev-GGUF`): llama.cpp's
  conversions. Lev and OpenJev read ordinary logits and run on every released llama-cpp-python.
  Kev's GGUF carries its pointer head inside the file. On a llama-cpp-python that bundles
  llama.cpp from 2026-10-02 or later (ggml-org/llama.cpp#29818) it loads as is; on every
  current release AbstractCore derives once, next to its model cache, a backbone copy of the file
  without the head (every other tensor copied byte for byte) and applies the head itself, with
  the same numbers. Nothing to install beyond `abstractcore[apple]` / `[gpu]`; the first load of a
  file takes a few seconds longer. (This bridge goes away with abetlen/llama-cpp-python#2398.)
  Laya's GGUF needs llama.cpp's decision graph and is not run by AbstractCore yet; use the
  transformers checkpoint.

The AbstractFramework conversions on the Hub (collection *Decision models*) are these recipes applied
to Kev-4B and Kev-0.8B: `kev-4b-mlx-8bit` (4.2 GB), `kev-4b-mlx-4bit` (3.0 GB), `kev-0.8b-mlx-8bit`
(0.8 GB), `kev-0.8b-mlx-4bit`. Convert a Kev adapter yourself with:

```bash
python -m abstractcore.decision.convert jaredpalmer/kev-4b --runtime mlx --bits 8 --out ./kev-4b-mlx-8bit
python -m abstractcore.decision.convert jaredpalmer/kev-4b --runtime mlx --bits 4 --group-size 32 \
    --upgrade '{"linear_attn.in_proj_qkv": 8, "linear_attn.in_proj_z": 8, "linear_attn.in_proj_a": 8, "linear_attn.in_proj_b": 8, "linear_attn.out_proj": 8}' \
    --out ./kev-4b-mlx-4bit
python -m abstractcore.decision.convert jaredpalmer/kev-4b --runtime transformers --out ./kev-4b
# Kev-0.8B is more sensitive: keep the MLP gate/up projections at 8-bit as well
python -m abstractcore.decision.convert jaredpalmer/kev-0.8b --runtime mlx --bits 4 --group-size 32 \
    --upgrade '{"linear_attn.in_proj_qkv": 8, "linear_attn.in_proj_z": 8, "linear_attn.in_proj_a": 8, "linear_attn.in_proj_b": 8, "linear_attn.out_proj": 8, "mlp.gate_proj": 8, "mlp.up_proj": 8}' \
    --out ./kev-0.8b-mlx-4bit
```

The second command is the measured 4-bit recipe (group size 32, the Gated DeltaNet projections at
8 bits): a plain 4-bit conversion of a decision model loses accuracy and shifts probabilities;
this one keeps the accuracy and most of the calibration (table below). When calibration matters
more than size, use 8-bit: it is indistinguishable from bf16. A converted folder is a plain model handle:
`create_llm("mlx", model="./kev-4b-mlx-4bit")`.

## Measured quality and speed

Public development set of the Kev evaluation suite (`public-pool-v6`: 1,328 records, 1,592
questions over banking77, AG News, MNLI, TREC, DBpedia, ARC, OpenBookQA, CommonsenseQA, BoolQ,
SST-5 and Yelp), Kev-4B, Apple M5 Max. Accuracy is argmax accuracy over all questions; Brier and
NLL are per question (lower is better); ECE uses 10 bins. "Agreement" and "max |Δp|" compare each
run with the bf16 MLX reference on the same questions.

| Runtime | Precision | Size | Accuracy | Brier | ECE | Agreement with bf16 | max / mean |Δp| | ms per record |
|---|---|---|---|---|---|---|---|---|
| mlx | bf16 (adapter merged at load) | 7.9 GB | 0.846 | 0.227 | 0.024 | — | — | 117 |
| mlx | bf16 (merged checkpoint) | 7.9 GB | 0.846 | 0.227 | 0.024 | 1.000 | 0.000 / 0.000 | 169 |
| mlx | 8-bit (group 64) | 4.2 GB | 0.846 | 0.227 | 0.025 | 0.997 | 0.07 / 0.004 | 136 |
| mlx | 4-bit (group 64, every linear) | 2.2 GB | 0.837 | 0.232 | 0.026 | 0.954 | 0.92 / 0.038 | 117 |
| llama.cpp | Q4_K_M | 3.0 GB | 0.845 | 0.225 | 0.025 | 0.981 | 0.30 / 0.018 | 398 |
| mlx | 4-bit mixed (group 32, DeltaNet projections 8-bit) | 3.0 GB | 0.847 | 0.226 | 0.024 | 0.973 | 0.34 / 0.023 | 113 |
| llama.cpp | Q8_0 | 4.5 GB | 0.844 | 0.227 | 0.022 | 0.995 | 0.07 / 0.003 | 225 |
| transformers | bf16 (MPS, 300 records) | 7.9 GB | 0.909* | 0.154* | 0.019* | 1.000* | 0.06 / 0.002* | 755 |
| Kev-0.8B mlx | bf16 | 1.6 GB | 0.763 | 0.321 | 0.030 | — | — | 30 |
| Kev-0.8B mlx | 8-bit | 0.8 GB | 0.761 | 0.321 | 0.025 | 0.990 (vs its bf16) | 0.006 / — | 24 |
| Kev-0.8B mlx | 4-bit mixed (group 32, DeltaNet and MLP gate/up at 8-bit) | 0.65 GB | 0.754 | 0.331 | 0.024 | 0.960 (vs its bf16) | 0.028 / — | 24 |

Rows marked * were measured on the first 300 records (460 questions) of the same set; on that
subset the full-set runs read: bf16 0.909 accuracy; 8-bit 0.909 / agreement 1.000; plain 4-bit
0.902 / 0.972 / mean |Δp| 0.030; Q4_K_M 0.904 / 0.991 / 0.014.

"ms per record" is the whole request for a new state (its tokens plus one to three questions);
a repeated state only pays for its questions (Kev-4B on MLX: about 45 ms for three questions).
Latency on transformers reflects PyTorch's reference Gated DeltaNet kernels on MPS; on CUDA with
the fused kernels installed it is several times faster.

### Small benchmark: Kev, Lev, Laya and hosted Jev on the same questions

A source-balanced slice of the same public pool: the first 10 records of each of its 13 sources
(130 records, 160 questions: 115 choice, 30 noul, 15 score; banking77, BoolQ, AG News, MNLI, SST-5,
Yelp, TREC, DBpedia, Amazon, IMDb, ARC, OpenBookQA, CommonsenseQA). States are short: median 30
tokens, mean 59, maximum 313 (see the length ladder below for long documents). Every model
answered the same requests through AbstractCore; Jev through the `typesafe` provider (model `jev-1.14.0` on
2026-11-01). Accuracy is argmax accuracy; Brier, NLL and ECE (10 bins) are per question, lower is
better. Keep in mind that several of these sources are in Kev's and Lev's training data and that
130 records is a small sample (one question is 0.6 points).

| Model | Provider / precision | Accuracy | Brier | NLL | ECE | choice / noul / score accuracy | ms per record |
|---|---|---|---|---|---|---|---|
| Jev (hosted) | typesafe, `jev-1.14.0` | 0.844 | 0.267 | 1.076 | 0.099 | 0.86 / 0.88 / 0.73 | 164 (network included) |
| Kev-4B | mlx 4-bit mixed | 0.825 | 0.255 | 0.549 | 0.066 | 0.88 / 0.92 / 0.53 | 113 |
| Lev | llama.cpp Q4_K_M | 0.825 | 0.295 | 0.701 | 0.098 | 0.88 / 0.88 / 0.60 | 180 |
| Kev-4B | llama.cpp Q4_K_M | 0.812 | 0.257 | 0.554 | 0.064 | 0.85 / 0.92 / 0.53 | 398 |
| Lev | mlx bf16 | 0.812 | 0.304 | 0.719 | 0.104 | 0.85 / 0.86 / 0.63 | 711 |
| Kev-4B | mlx bf16 | 0.806 | 0.259 | 0.564 | 0.070 | 0.85 / 0.90 / 0.53 | 117 |
| Kev-4B | mlx 8-bit | 0.806 | 0.260 | 0.564 | 0.076 | 0.85 / 0.90 / 0.53 | 136 |
| Kev-0.8B | mlx bf16 | 0.762 | 0.326 | 0.643 | 0.053 | 0.75 / 0.92 / 0.53 | 30 |
| Laya | transformers fp32 (start of the state only) | 0.656 | 0.553 | 1.905 | 0.188 | 0.55 / 0.86 / 0.60 | 49 |

What the numbers say: the hosted model is the most accurate and the strongest on ordinal
(`score`) questions; the 4B local models are within two points of it on choice and yes/no questions
and are better calibrated (lower Brier and NLL); Laya is six times faster and much smaller but
weakens on long option lists (its 512-token sequence caps each option at a few tokens when a
question has dozens of them). On a slice dominated by 77-way intent questions (the pool's first 100
records) Laya drops to 0.46 while Kev and Lev score 0.89 and Jev 0.84.

The same support ticket, every model:

| Model | department (returns / shipping / billing) | escalate | frustration (expected level) |
|---|---|---|---|
| Jev (hosted) | returns 0.66 / 0.11 / 0.23 | 0.68 | 1.01 |
| Kev-4B (mlx bf16) | returns 0.51 / 0.36 / 0.13 | 0.72 | 1.30 |
| Lev (llama.cpp Q4_K_M) | shipping 0.25 / 0.72 / 0.03 | 0.99 | 1.78 |
| Lev (mlx bf16, full strength) | returns 0.51 / 0.45 / 0.04 | 0.99 | 1.81 |
| Laya (transformers) | shipping 0.28 / 0.40 / 0.32 | 0.12 | 1.10 |

Lev's release manifest applies its adapter at a quarter of its trained strength (the author's
benchmark setting) while llama.cpp's GGUF conversion merges it at full strength; on this pool the
full-strength weights score higher (0.89 vs 0.80 on the first 100 records), so the `mlx` /
`huggingface` adapter path accepts `adapter_scale=` (default: the manifest's 0.25; pass 1.0 to
match the GGUF). Parity checks against the authors' own runtimes: Kev within 0.006 on the example
and 0.028 max over 40 questions; Laya identical on the example and 0.056 max over 30 questions.

### Length ladder: 1k to 16k-token documents

Two document types, each with its own questions, bucketed by how long the document is:

- **Contracts** (CUAD: real commercial contracts with expert clause annotations): "does this contract
  contain a clause of type X?" (yes/no, one present and one absent per contract) and "which of these
  four clause types is present?" (choice). Buckets of 1k, 2k, 4k, 8k and 16k tokens.
- **Reading** (QuALITY: long articles with multiple-choice comprehension questions written for them):
  two 4-option questions per article. Buckets of 2k, 4k and 8k tokens.

The summary first (mean accuracy over the buckets of each task; seconds per document, all its questions
in one request; Jev over the network, local models on an M5 Max). Every local row runs with the state
cache described above: within a request the document is read once for all its questions, on every
engine and backend (Kev on transformers, mlx-lm and llama.cpp; Lev on the same three); the ladder
measures first requests, so these are cold costs, and a second request about the same document costs
only its questions. Laya is a bidirectional encoder with nothing to cache; its long-state mode is the
windowed scan.

| Model | Contracts, accuracy | Reading, accuracy | Seconds per document at 2k / 8k / 16k |
|---|---|---|---|
| Jev hosted (jev-1.14.0) | 0.88 | 0.94 | 0.2 / 0.2 / 0.3 |
| Kev-4B (mlx bf16) | 0.88 | 0.78 | 0.7 / 3.6 / 5.2 |
| Kev-4B (mlx 4-bit) | 0.87 | 0.78 | 1.1 / 3.4 / 7.6 |
| Kev-4B (llama.cpp Q4_K_M) | 0.87 | 0.80 | 1.5 / 4.9 / 11.1 |
| Lev (llama.cpp Q4_K_M) | 0.84 | 0.78 | 1.3 / 4.2 / 8.7 |
| Kev-0.8B (mlx bf16) | 0.77 | 0.51 | 0.1 / 0.6 / 1.3 |
| Kev-0.8B (mlx 4-bit) | 0.76 | 0.48 | 0.1 / 0.4 / 0.9 |
| Laya (windowed) | 0.48 | 0.26 | 3.4 / 10.0 / 26.1 |
| Laya (start of the document only) | 0.40 | 0.26 | 0.1 / 0.1 / 0.2 |

In one sentence each: the hosted model is the only one that reads long articles well and it answers in
a fraction of a second whatever the length; the 4B local models equal it on contract clauses up to 16k
tokens but pay one to eleven seconds to read a long document (once per document, see the cache below);
Kev-0.8B is the fast local option, good on contract clauses, weak on reading comprehension; Laya does not
handle long documents. The detailed per-bucket tables follow.


The small benchmark above uses short states (median 30 tokens, maximum 313). This one uses whole
documents bucketed by their natural token length, nothing padded or truncated: CUAD contracts
(CC-BY-4.0, expert clause annotations) at 1k, 2k, 4k, 8k and 16k tokens with, per contract, one
present-clause and one absent-clause yes/no question and a 4-way "which clause type is present"
question; and QuALITY articles (CC-BY-4.0, Project Gutenberg texts) at 2k, 4k and 8k tokens with
two 4-option comprehension questions each. 87 documents, 229 questions (7 to 12 documents per
bucket; bucket bands 500–1.3k, 1.3k–2.6k, 3k–4.3k, 6.2k–8.6k, 12.5k–17k state tokens). Accuracy /
Brier / milliseconds per document (every question of the document in one request; Jev over the
network). Laya is listed in both of its modes: reading the start of the document only (512 tokens), and
scanning it in overlapping windows as its runtime does; neither reaches the decoder models on
these tasks, and the windowed scan costs about one forward pass per 150 tokens per question.

Contract clause questions (CUAD):

| Model | 1k (21 q) | 2k (36 q) | 4k (36 q) | 8k (36 q) | 16k (36 q) |
|---|---|---|---|---|---|
| Jev hosted (jev-1.14.0) | 0.95 / 0.05 / 155 | 0.89 / 0.14 / 157 | 0.89 / 0.19 / 184 | 0.75 / 0.33 / 223 | 0.94 / 0.09 / 307 |
| Kev-4B (mlx bf16) | 0.95 / 0.14 / 301 | 0.86 / 0.20 / 643 | 0.83 / 0.25 / 1236 | 0.83 / 0.27 / 2236 | 0.94 / 0.18 / 5222 |
| Laya (transformers, start of the state only) | 0.48 / 0.70 / 190 | 0.47 / 0.65 / 149 | 0.36 / 0.72 / 151 | 0.36 / 0.75 / 153 | 0.31 / 0.74 / 248 |
| Lev (llama.cpp Q4_K_M) | 0.90 / 0.19 / 843 | 0.92 / 0.14 / 1204 | 0.75 / 0.41 / 1972 | 0.78 / 0.36 / 3584 | 0.86 / 0.16 / 8747 |
| Kev-4B (mlx 4-bit) | 0.95 / 0.13 / 594 | 0.86 / 0.20 / 1039 | 0.81 / 0.26 / 1923 | 0.81 / 0.29 / 3446 | 0.94 / 0.17 / 7583 |
| Kev-0.8B (mlx bf16) | 0.86 / 0.31 / 121 | 0.75 / 0.33 / 119 | 0.81 / 0.39 / 308 | 0.64 / 0.44 / 697 | 0.81 / 0.40 / 1342 |
| Kev-0.8B (mlx 4-bit) | 0.81 / 0.32 / 74 | 0.78 / 0.32 / 112 | 0.81 / 0.38 / 193 | 0.61 / 0.45 / 375 | 0.78 / 0.40 / 945 |
| Laya (transformers, windowed) | 0.52 / 0.52 / 1916 | 0.50 / 0.68 / 4034 | 0.47 / 0.71 / 7010 | 0.50 / 0.77 / 11898 | 0.42 / 0.88 / 26119 |
| Kev-4B (llama.cpp Q4_K_M, released binding) | 0.86 / 0.15 / 757 | 0.92 / 0.19 / 1474 | 0.83 / 0.25 / 2593 | 0.81 / 0.29 / 4570 | 0.92 / 0.18 / 11060 |

Reading comprehension (QuALITY):

| Model | 2k (16 q) | 4k (24 q) | 8k (24 q) |
|---|---|---|---|
| Jev hosted (jev-1.14.0) | 1.00 / 0.01 / 233 | 0.92 / 0.13 / 218 | 0.92 / 0.09 / 188 |
| Kev-4B (mlx bf16) | 0.94 / 0.18 / 787 | 0.71 / 0.34 / 1923 | 0.71 / 0.48 / 5540 |
| Laya (transformers, start of the state only) | 0.19 / 0.85 / 133 | 0.38 / 0.76 / 105 | 0.21 / 0.79 / 108 |
| Lev (llama.cpp Q4_K_M) | 0.88 / 0.17 / 1453 | 0.75 / 0.44 / 2574 | 0.71 / 0.49 / 5160 |
| Kev-4B (mlx 4-bit) | 0.94 / 0.17 / 1142 | 0.71 / 0.36 / 1709 | 0.71 / 0.48 / 3353 |
| Kev-0.8B (mlx bf16) | 0.69 / 0.48 / 203 | 0.50 / 0.61 / 268 | 0.33 / 0.77 / 544 |
| Kev-0.8B (mlx 4-bit) | 0.69 / 0.52 / 150 | 0.42 / 0.63 / 207 | 0.33 / 0.77 / 430 |
| Laya (transformers, windowed) | 0.19 / 0.90 / 2003 | 0.29 / 0.90 / 3088 | 0.29 / 0.93 / 7113 |
| Kev-4B (llama.cpp Q4_K_M, released binding) | 0.94 / 0.17 / 1639 | 0.75 / 0.32 / 2676 | 0.71 / 0.51 / 5286 |

What the ladder says: the hosted model holds its accuracy across the whole range and stays flat in
latency; the 4B local models match it on clause presence up to 16k tokens (within the noise of a
few dozen questions) but lose reading comprehension beyond 4k, where their authors' training states
stopped. Kev-0.8B is four times faster than Kev-4B at every length (1.3 s for a 16k-token contract)
and keeps most of the clause-presence accuracy, but not reading comprehension past 2k. On every
local engine the state is read once per request and reused for all its questions and option orders
(the pointer-head cache for Kev, the shared prompt prefix for Lev), so a second request about the
same document only pays for its questions; Kev-4B's 4-bit checkpoint keeps the bf16 accuracy at a
somewhat higher prefill cost on MLX. Local latency grows linearly with the document: a 16k-token contract costs
about five seconds on an M5 Max, once per document, after which more questions about it cost tens
of milliseconds.

### How to read the numbers

- Argmax agreement with the bf16 run is what a classifier or router sees; |Δp| is what a
  threshold sees. 8-bit (MLX or Q8_0) is indistinguishable from bf16. The official Q4_K_M GGUF
  and the mixed MLX 4-bit recipe keep the accuracy; a few percent of their answers move by more
  than 0.10 against bf16, so prefer 8-bit when you threshold probabilities. The plain 4-bit
  recipe (every linear at 4 bits, group 64) loses a point of accuracy and moves a tenth of the
  answers by more than 0.10; it is not offered.
- The Kev model card reports its own figures for Kev-4B (new-source accuracy 0.82–0.84,
  trained-source 0.87) on the author's evaluation suites; they are not reproduced here. The tables
  above are measured on the public pool, which mixes trained and new sources.

## Limits

- Kev and Lev are English only (Laya ships a multilingual checkpoint); knowledge questions are bounded by the 4B base (the author reports MMLU-Pro 0.59
  for Kev-9B against 0.84 for Jev).
- A state may hold up to 65,536 tokens and each question row 8,192 tokens more; an over-long state
  is refused (`ContextOverflow`, HTTP 422) unless you ask to `truncate`. The author trained on
  states up to 7,552 tokens, so accuracy on much longer documents is not characterised.
- Changing the order of a question's options can change its answer; keep option order stable
  when you compare runs.
- Calibration is the author's: one temperature per checkpoint for Kev (2.41 for Kev-4B), per
  question type and option count for Lev and Laya; a temperature never changes which option ranks
  first. A hosted model's calibration is the host's.
- Decision models never generate text: `response_model=`, tools, streaming and media inputs do
  not apply to `output="decision"`.

## Related

- [Capabilities](capabilities.md) — the other `generate(..., output=...)` modalities
- [API (Python)](api.md) — `generate`, `MultimodalGenerateResponse`
- [Server](server.md) — `POST /v1/decisions`
- [Local Models](models.md) — downloading and listing weights
- `abstractcore/assets/model_capabilities.json` — the `kev-*` entries (`capability_routes.output: ["decision"]`)
