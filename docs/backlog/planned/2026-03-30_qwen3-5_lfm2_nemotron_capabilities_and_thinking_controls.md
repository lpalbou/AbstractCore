# 2026-03-30 — Qwen3.5/LFM2/Nemotron capability audit + thinking controls

## Goal

1) Ensure `abstractcore/assets/model_capabilities.json` and `abstractcore/assets/architecture_formats.json` are accurate and internally consistent for:
- native tool calling support
- native structured output / JSON mode support
- reasoning/thinking controls + supported levels
- max context (`max_tokens`) and max output (`max_output_tokens`)

2) Verify these claims against *upstream model cards* (Hugging Face and/or official docs) for all models in scope.

3) Run real local experiments (LM Studio + Ollama) for Qwen3.5 thinking control across multiple sizes.

## Scope (models)

### Qwen3.5
- Qwen3.5-0.8B
- Qwen3.5-2B
- Qwen3.5-4B
- Qwen3.5-9B
- Qwen3.5-27B
- Qwen3.5-35B-A3B
- Qwen3.5-122B-A10B
- (Optional) Qwen3.5-397B-A17B if present in assets/local inventory

### LiquidAI
- LFM2: 1.2B, 2.6B, 8B-A1B, 24B-A2B (and any instruct variants if they exist upstream)
- LFM2.5: 1.2B base/instruct/thinking/JP, VL-1.6B, Audio-1.5B

### NVIDIA
- Nemotron-3-Nano-30B-A3B
- Nemotron-Cascade-2-30B-A3B
- Nemotron-3-Super-120B-A12B
- NVIDIA GPT-OSS puzzle: `nvidia/gpt-oss-puzzle-88B`

### Other cited models
- `mistralai/Mistral-Small-4-119B-2603`
- `Tesslate/OmniCoder-9B-GGUF`
- `MiniMaxAI/MiniMax-M2.5`

## Deliverables

- Updated assets with corrected `tool_support`, `structured_output`, `max_tokens`, `max_output_tokens`, and (where applicable) `reasoning_levels` / thinking controls.
- Local experiment results for Qwen3.5 thinking controls on:
  - LM Studio (OpenAI-compatible endpoint)
  - Ollama
- Documentation updates describing the verified control surface and known limitations.
- All existing unit/schema tests remain green.

## Acceptance criteria

- Model card verification:
  - For each model, we have a short note in the final report describing what the model card claims about:
    - context window
    - output limits (if stated)
    - tool/function calling (if any)
    - structured output / JSON mode (if any)
    - how to enable/disable and/or scale thinking (if supported)
- Local experiments:
  - For each required Qwen3.5 size on **LM Studio** and **Ollama**, we can demonstrate:
    - `thinking="off"` suppresses `<think>...</think>` (or whatever reasoning channel exists)
    - `thinking="on"` enables reasoning (when supported)
    - `thinking="low|medium|high|xhigh"` maps to a real control knob *if available*; otherwise the behavior is documented and the API degrades gracefully (warning + best-effort).
- No regressions:
  - `pytest -q tests/assets/test_model_capabilities_schema.py tests/assets/test_architecture_formats_schema.py`
  - `pytest -q tests/providers/test_thinking_mode_control_unit.py tests/providers/test_qwen3_5_model_support_unit.py`

## Notes / risks

- Some “native tool calling” is **server/API-dependent** (OpenAI-compatible gateways vary). If upstream model cards only describe *special-token tool calling*, we may need to classify support as `prompted` (or refine the registry semantics) to avoid over-claiming.
- Qwen3.5 thinking controls are known to be template/serving-stack dependent; we may need multiple control paths:
  - `chat_template_kwargs.enable_thinking`
  - potentially a budget/effort knob if the template exposes it
  - prompt-level “thinking_control” tokens only as a fallback

## Report (to be filled on completion)

TBD.

