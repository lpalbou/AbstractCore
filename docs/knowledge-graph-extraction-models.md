# Choosing a model for knowledge-graph extraction

This page helps you pick a model and a prompting strategy when you extract entities and typed
relationships from documents under a fixed ontology (for example with
[BasicExtractor](apps/basic-extractor.md) or your own prompt). It summarizes a measured comparison
of local and hosted models run through AbstractCore, the settings that changed the results, and
the limits of that measurement.

Related pages: [Reasoning Control](reasoning-control.md) (the `thinking=` parameter used below),
[Structured Output](structured-output.md), [Recommended Models](recommended-models.md) and
[Hugging Face model compatibility](huggingface-model-compatibility.md) (Hy-Memory-30B-A3B setup).

## How the comparison was run

| Setting | Value |
|---|---|
| Task | Extract entities and relationships from one 21 KB Markdown document (a structured note about version control and digital preservation), in one call per pass |
| Ontology | 15 entity types and 12 relationship types (`has_component`, `part_of`, `requires`, `causes`, `located_at`, `occurs_at`, `performs`, `measures`, `warns_about`, `precedes`, `produced_by`, `related_to`), each with a description and examples |
| Output | JSON with `entities` and `relationships` (`source`, `target`, `type`, `description`) |
| Sampling | `temperature=0`, the same output budget for every model (12,000 tokens; 32,000 where noted) |
| Local hardware | Apple M5 Max, 128 GB, one model loaded at a time |
| Local engines | MLX (`mlx` provider) and LM Studio (`lmstudio` provider), 4-bit weights |
| Hosted models | Through an OpenAI-compatible relay (`openai-compatible` provider) |
| Scoring | Every distinct typed relationship (605 in total) judged once against the document by a single blind reviewer: correct, inverted (true only the other way round), contradicts the text, or unsupported |

**Precision** is correct typed relationships divided by all typed relationships a run produced
(`related_to` links are counted separately). **Sections covered** counts how many of the
document's 15 later subsections appear among the extracted entities. Times include every pass and
exclude model loading.

## Results

The rows below are the best-performing configurations per model and the configurations that
illustrate each effect. Thinking "off" means `thinking=False`; "on" means the model reasoned
(see the note on Claude below).

| Model | Where it ran | Method | Thinking | Time | Correct / typed | Precision | Sections covered |
|---|---|---|---|---|---|---|---|
| Claude Haiku 5.5 | hosted | 2 passes, add missing | on | 134 s | 79 / 86 | 92% | 13 / 15 |
| Claude Haiku 5.5 | hosted | 1 pass, improved prompt | on | 36 s | 24 / 27 | 89% | 9 / 15 |
| GPT-6 Luna | hosted | 2 passes, review and correct | off | 85 s | 27 / 30 | 90% | 7 / 15 |
| GPT-6 Luna | hosted | 1 pass, improved prompt | off | 20 s | 15 / 25 | 60% | 3 / 15 |
| GPT-5.6 Luna | hosted | lint + per-edge verification | off | 160 s | 47 / 68 | 69% | 8 / 15 |
| GPT-5.6 Luna | hosted | 2 passes, add missing | off | 94 s | 59 / 104 | 57% | 12 / 15 |
| Hy-Memory 30B-A3B | local (MLX, 4-bit) | 1 pass, original prompt | high | 299 s | 20 / 24 | 83% | 11 / 15 |
| Hy-Memory 30B-A3B | local (MLX, 4-bit) | 1 pass, improved prompt | off | 28 s | 9 / 11 | 82% | 6 / 15 |
| Hy-Memory 30B-A3B | local (MLX, 4-bit) | 1 pass, original prompt | off | 25 s | 2 / 11 | 18% | 2 / 15 |
| Qwen3.6 35B-A3B | local (LM Studio, 4-bit) | lint + per-edge verification | off | 151–232 s | 9–14 / 9–23 | 39–100% | 5–8 / 15 |
| Qwen3.6 35B-A3B | local (LM Studio, 4-bit) | 1 pass, improved prompt | on | 207 s | 12 / 19 | 63% | 12 / 15 |
| Qwen3.8 27B | local (LM Studio, 4-bit) | 1 pass, original prompt | off | 148 s | 11 / 15 | 73% | 6 / 15 |
| Gemma 4 12B QAT | local (LM Studio, GGUF) | 1 pass, improved prompt | off | 57 s | 7 / 10 | 70% | 5 / 15 |
| Qwen3.5 9B | local (LM Studio, 4-bit) | 1 pass, improved prompt | off | 35 s | 4 / 7 | 57% | 9 / 15 |
| Gemma 4 E4B | local (LM Studio, 4-bit) | 1 pass, improved prompt | off | 29 s | 2 / 12 | 17% | 4 / 15 |

The Qwen3.6 pipeline row is a range over three runs of the same configuration: its extraction
changes from run to run even at `temperature=0`, so a single run of that model is not
representative. Hy-Memory-30B-A3B produced byte-identical output on repeated runs.

Claude Haiku 5.5 was reached through a relay that drives the Claude command-line client on a
subscription. On that route its adaptive thinking cannot be switched off, so every Claude row
includes reasoning; compare it with the other models' thinking-on rows. On that route a long
extraction also needs an output budget of about 32,000 tokens.

## What changes the results

### The prompt rules

Two generic rules, added to the ontology prompt, changed results more than any model swap
between neighbours:

- **List the allowed labels as a hard constraint** ("`type` MUST be exactly one of: …"). Without
  it, Qwen3.6, Qwen3.5 and Gemma 4 invent relationship types (`uses`, `enables`, `informs`, …) that
  a typed graph cannot load.
- **State the direction rule**: every relationship must read as the true sentence
  "`<source> <type> <target>`" in the same word order as the ontology's examples. This took
  Hy-Memory-30B-A3B from 2 to 9 correct edges and from 6 inverted edges to 1.

The same rules can hurt a model that already follows the ontology: Qwen3.8 27B produced more
edges with them but its precision fell from 73% to 64%. Test prompt changes per model.

### Reasoning

Reasoning raised quality on every model that could use it within the budget:

- Hy-Memory-30B-A3B: 9 → 20 correct edges at the same precision (82% → 83%), ten times slower.
- Qwen3.6 35B-A3B: 50% → 63% precision with the improved prompt.
- GPT-6 Luna: 60% → 69% precision with the improved prompt.

Reasoning needs output budget. GPT-5.6 Luna at `high` and Claude Haiku 5.5 at the default
12,000-token budget ran out before finishing the JSON. Use [Reasoning Control](reasoning-control.md)
to set the level and size `max_tokens` for reasoning plus the answer.

### A second pass

A second call that receives the document and the first extraction helps only when it gets
something it can act on:

- **"Add what is missing"** grows the graph. It works best on strong models (Claude Haiku 5.5: 24 →
  79 correct edges at 92%); on weaker models it adds a few edges at lower precision.
- **"Review and correct" without specific feedback** rarely removes errors: most models return their
  own mistakes unchanged. GPT-6 Luna is the exception (60% → 90%).
- **Machine checks plus per-edge verification** (extract with an evidence quote per relationship,
  check labels, endpoints and quotes in code, then ask for a verdict per relationship and apply the
  verdicts in code) lifted Qwen3.6 to 93% in its best run and GPT-5.6 Luna from 58% to 69%. It did
  not improve Hy-Memory-30B-A3B, which keeps its own edges when asked to verify them.

Apply verdicts in code rather than asking the model to rewrite the whole graph, and treat a
malformed verification response as "no change" rather than as a reason to drop edges.

## Recommendations

| Goal | Model and method |
|---|---|
| Best quality | Claude Haiku 5.5, two passes ("add missing"), 32,000-token budget |
| Fast hosted, high precision | GPT-6 Luna with a "review and correct" second pass |
| Most relationships, then review | GPT-5.6 Luna with lint + per-edge verification |
| Local, precise and reproducible | Hy-Memory-30B-A3B, one pass, improved prompt, `thinking=False` |
| Local, more coverage | Hy-Memory-30B-A3B with `thinking="high"`, or Qwen3.8 27B with the original prompt |

For any of them:

- Put the allowed labels and the direction rule in the prompt.
- Ask for an evidence quote per relationship and check it in code; it is the cheapest reliable
  signal that a relationship is grounded.
- Validate the JSON and the labels before loading a graph; keep `related_to` as an explicit
  "linked, untyped" fallback rather than forcing a specific label.
- Split long documents by section: no single-pass run covered more than 13 of the 15 later
  subsections.

## Limits of this comparison

- One document and one ontology; rankings can change with domain, language and document length.
- Each relationship was judged once by one reviewer; borderline verdicts are possible.
- Local and hosted models ran on different stacks, so times compare end-to-end latency, not raw
  model speed. Hosted latency also depends on the relay's load.
- Models with run-to-run variation (Qwen3.6 in particular) need several runs per configuration
  before you compare them.
