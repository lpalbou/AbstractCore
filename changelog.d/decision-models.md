### Added

- **Decision models: text in, calibrated typed decisions out** (`abstractcore/decision/`, `generate(..., output="decision")`,
  `POST /v1/decisions`). One abstraction over every decision model AbstractCore can reach: hosted Jev through the new
  `typesafe` provider and any System One-compatible host through `systemone` (HTTP only, light install), and local
  checkpoints on `mlx` and `huggingface` (transformers or GGUF): the Kev family (pointer head on Qwen3.5), Lev (label
  logits on Qwen3.5-4B) and Laya (ModernBERT encoder with a marker scorer), all Apache-2.0; OpenJev-style checkpoints
  run through the same label-logit engine. Questions are `choice` (1–255 named options), `noul` (the probability a
  statement is true) and `score` (ordered levels); answers carry calibrated probabilities and confidence in the public
  System One request/answer shape, the format every decision-model host shares. The Kev state prefix is cached across calls.
  `MultimodalGenerateResponse.decision` / `.decisions` carry the answers; a decision model refuses text generation with
  a pointer to `output="decision"`. `python -m abstractcore.decision.convert` folds a Kev adapter into a merged
  transformers or mlx-lm checkpoint (bf16, 8-bit, or the measured mixed 4-bit recipe). Registry: `kev-0.8b/4b/9b/27b`,
  `lev`, `laya`, `openjev` in `model_capabilities.json` (`capability_routes.output: ["decision"]`, the new `decision`
  route modality, a `decision.kind` block), the `kev` / `lev` / `laya` / `openjev` architectures, download-catalog rows.
  A pointer-head GGUF runs on every released llama-cpp-python: when the bundled llama.cpp predates
  ggml-org/llama.cpp#29818, AbstractCore derives a backbone copy once and applies the head itself (`gguf_compat.py`,
  removed with abetlen/llama-cpp-python#2398). `/v1/models` entries carry additive `capabilities` (input / output
  modalities) and, for decision models, a `decision` object naming `/v1/decisions`; a chat completion against a
  decision model is a 400 (`code: decision_model`) with that pointer (`DecisionModelError`). The label-logit engine reuses
  the prompt prefix every question of a request shares (the state runs once; transformers, mlx-lm, llama.cpp); Laya scans a
  long state in overlapping windows as its runtime does. Kev-0.8B joins the registry and catalog; the AbstractFramework Hub
  collection *Decision models* carries `kev-4b-mlx-8bit`, `kev-4b-mlx-4bit`, `kev-0.8b-mlx-8bit`, `kev-0.8b-mlx-4bit`.
  Docs: `docs/decision-models.md`.
- **`output.decision` capability default and a shared decision pool.** `abstractcore config set-default
  output.decision --provider mlx --model jaredpalmer/kev-4b` names the decision model every provider
  instance uses for `output="decision"` (the route resolver puts it on the spec; an explicit route on the
  spec wins). `abstractcore.decision.pool` keeps at most two decision models resident per process, shared
  by `generate(output="decision")` and `/v1/decisions`; `model: "kev-latest"` on the server falls back to
  the configured default when nothing is loaded yet. `output_has_generated_media` is False for a decision
  (typed answers, never bytes), so hosts need no artifact store for it.
  Decision models also have residency: `POST /acore/models/load` / `unload` with `task: "decision"` (listed as
  `process:decision:<provider>:<model>` rows of `/acore/models/loaded`); catalog rows carry a `decision` capability
  flag; `/v1/decisions` accepts the question shorthand; the decision provider is recorded as
  `metadata.decision_provider`.
  The mlx provider lists the decision models it can run from the local caches (Kev / Lev adapters whose base
  checkpoint is on disk, merged decision conversions), so pickers filtered on `output.decision` offer them.
