# 0877 — Decision models: `output="decision"` and `/v1/decisions`

## Metadata
- Created: 2026-11-01
- Status: Completed
- Completed: 2026-11-01
- Owner: core

## Goal

Run Jev-like decision models (the Kev family: a LoRA adapter plus a pointer head on a Qwen3.5
base, Apache-2.0) as a first-class AbstractCore modality — text input, calibrated typed decisions
output — on the local engines (MLX, transformers, GGUF via llama.cpp), with the public System One
wire format on the server, relying only on AbstractCore and the low-level engines (no `kev`,
`peft` or `typesafe` package).

## Delivered

- `abstractcore/decision/`: typed questions and answers (`Choice`, `Noul`, `Score`,
  `DecisionRequest`, `DecisionResponse`), request rendering, token packing with the five reused
  Qwen delimiter tokens, the pointer head in numpy (`head.safetensors` + `decision.json` format,
  `head.pt` reader), tokenizer adapters, the `DecisionModel` runtime with a state-prefix LRU cache,
  layout detection and loading (`adapter` / `merged` / `gguf`), and `convert.py` (adapter -> merged
  transformers or mlx-lm checkpoint, bf16 or 4/8-bit, optional per-module upgrades).
- Backends: `backends/hf.py` (transformers, adapter folded in at load, replicated DynamicCache
  per question row), `backends/mlx.py` (mlx-lm Qwen3.5 hybrid, merged cache copies),
  `backends/gguf.py` (llama.cpp embeddings mode with the projected `[q|k]` head output,
  `llama_memory_seq_cp` per question).
- Providers: the `decision` output modality (`output_specs`, `generate_contract`,
  `CAPABILITY_MODALITIES`, `ModelOutputCapability.DECISION`, schema enum),
  `BaseProvider._run_decision_output` / `decision_model()`, the decision lane in the
  HuggingFace and MLX providers (`model_type == "decision"`, text generation refused with a
  pointer), `MultimodalGenerateResponse.decision` / `.decisions`.
- Server: `POST /v1/decisions` (System One request/response, provider/model handles, aliases,
  two resident decision models, 400/404/422 mapping).
- Registry: `kev-0.8b/4b/9b/27b` in `model_capabilities.json` (routes `input.text` /
  `output.decision`, `decision` block), the `kev` architecture in `architecture_formats.json`,
  `ArchitectureFamily.KEV`, `ModelType.DECISION`, a `kev-4b` download-catalog row.
- Docs: `docs/decision-models.md`, README, docs index, capabilities, api, server, assets README,
  agent skill, `llms.txt` / `llms-full.txt`, changelog fragment `changelog.d/decision-models.md`.

- Engines beyond Kev: `engines/remote.py` (hosted System One: TypeSafe Jev through the `typesafe`
  provider, any host through `systemone`), `engines/labels.py` (label-logit kinds: Lev, OpenJev-style,
  on transformers / mlx-lm / llama.cpp plain logits), `engines/laya.py` (ModernBERT + marker scorer,
  transformers), `kinds.py` (the prompt/readout facts per kind, mirroring llama.cpp's conversion).
  The loader detects the kind from the checkpoint files, the GGUF metadata or the registry's
  `decision.kind` and builds the matching engine behind the same `DecisionEngine` surface.

- Follow-through (same day): endpoint renamed to `POST /v1/decisions` (+ `/{provider}/v1/decisions`);
  `/v1/models` entries annotated (`capabilities`, `decision.endpoint`); `DecisionModelError` -> HTTP 400;
  state-prefix reuse in the label-logit engine on all three local backends; Laya sliding-window long
  states; Kev-0.8B registered and measured; `gguf_compat.py` for the released llama-cpp-python;
  conversions published to the AbstractFramework Hub collection *Decision models*.

## Validation

- `tests/decision/` (18 tests): rendering, packing, delimiter escaping, context limits, the
  pointer head (projected form, temperature, safetensors round trip), confidence formulas,
  `DecisionModel` end to end with a fake backend (prefix cache hits and eviction),
  `generate(..., output="decision")` through `BaseProvider` with a fake runtime, output-spec
  vocabulary; `tests/providers/test_model_capabilities.py`, `tests/config/test_model_catalog.py`
  green with the new entries.
- Real weights (Kev-4B, Apple M5 Max): the transformers bf16, MLX bf16 and llama.cpp Q4_K_M
  paths agree with each other within 0.01 on the author's README example and with each other
  across the public development set; evaluation reports under `untracked/decision-eval/`.
- GGUF: verified both ways. A local llama-cpp-python build at llama.cpp b12032 runs the official
  file with the in-graph head; the released 0.3.35/0.3.36 run it through `gguf_compat.py` (derived
  backbone + numpy head) with answers equal to the fourth decimal. The bindings bump is
  abetlen/llama-cpp-python#2398; when it ships, delete `gguf_compat.py`, the `compat` branch of
  `backends/gguf.py` and the `gguf` dependency.

## Follow-ups

- Publish AbstractCore conversions (`AbstractFramework/kev-4b-mlx-4bit|8bit`) once the
  per-precision measurements are accepted; add their catalog rows.
- CLI surface (`abstractcore decide`) and `GET /v1/models?output_type=decision` listing check.
- Kev-27B (full-weight checkpoint, no adapter) load path on the `merged`-style layout.
