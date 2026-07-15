# Proposed: native tool declaration for template-capable models (Tier 2)

## Metadata
- Created: 2026-07-15
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: use-full-capability; no-silent-degradation (ADR 0001)
- ADR impact: none new; this aligns local lanes with the native/prompted split
  the OpenAI-compatible + Anthropic lanes already implement.

## Why this is PROPOSED, not planned
The operator-reported abstractcode issue (a "dynamically injected second system
prompt" on Ornith GGUF ReAct) is ALREADY FIXED by the Tier-1 wave (2026-07-15):
tool declarations now render as ONE system turn (`merge_tools_into_system`), and
MLX renders ChatML by the registry `message_format` instead of a `"qwen"` name
substring (Ornith rendered as plain text before). So this item is a QUALITY/
correctness improvement for native-capable models, not a fix the current issue
requires. It carries real regression risk and must be gated on a live A/B before
it becomes the default — hence proposed, not committed.

## Context
For models whose chat template has a native tools slot (Qwen3.5/3.6, Ornith 1.0,
Gemma-4 — all `tool_support: "native"` in `model_capabilities.json`), AbstractCore
currently still injects its OWN hand-rolled prompted tool block on the local
in-process lanes (MLX/GGUF/Ollama), because the injection gate is
`supports_prompted` (true for native models) rather than `not supports_native`,
and GGUF native tools are hard-disabled (`if False and self.llm.chat_format in …`,
huggingface_provider.py). The injected block teaches a call syntax that
CONTRADICTS the model's trained protocol (registry `qwen3_5` says
`tool_format: special_token` / `<|tool_call|>`, while the family emits
`<tool_call>`+JSON or `<function=…>` XML), plus a "no other text" rule the
template relaxes. The OpenAI-compatible + Anthropic lanes already route native
tools correctly via `tools=`; only the local lanes hand-roll.

Five fable5 design adversaries (2026-07-15) converged on a native-first hybrid:
native-capable models declare tools as structured DATA through their own
template; non-native models keep the prompted layer (one merged system turn).

## Direction (native-first hybrid)
1. Routing: gate prompted injection on `not supports_native` (the OpenAI-
   compatible lane's existing condition), so native models stop getting the
   hand-rolled block. Delete the GGUF `if False` native-tools disable — the
   original break was llama.cpp's grammar/tool_choice FUNCTION handler, which is
   separable from plain template rendering.
2. Native render: pass `tools=` into the model's own template — GGUF via
   `Jinja2ChatFormatter(... tools=...)` (currently hardcodes `tools=None`), MLX
   via `tokenizer.apply_chat_template(..., tools=...)`. Keep AbstractCore's own
   output parsing (arch formats already parse `<tool_call>`/`<function=…>`).
3. Registry fix: `architecture_formats.json` `qwen3_5` `tool_format` is wrong
   (`special_token`/`<|tool_call|>`); the family emits `<tool_call>`+JSON. (Note
   `qwen3_5_agentic` and `qwen3_6` are already correct.)
4. One renderer per lane, used by direct-generate + control-plane-append +
   prepare_modules, so prompt-cache byte-parity holds.

## Break modes to gate on (from the S5 robustness adversary)
- **History replay divergence (correctness, the substrate-biter):** turn 2+ —
  the hand-rolled ChatML replays assistant tool-calls as `functions.name:` and
  drops `tool`-role messages; the native template wants `<function=…>` replay +
  `<tool_response>` in a user turn. Mixed-distribution history regenerates the
  fabrication class the native work aims to kill. DECLARATIONS and REPLAY are
  separate migration surfaces — replay must be aligned in the same wave.
- **Streaming drops (correctness):** streamed `<function=…>` with zero args or
  multiple calls per block are dropped by `_convert_to_openai_format`
  (JSON-only) and `_parse_tool_json` (requires `<parameter`).
- **Arg typing (correctness/security):** `<parameter=recursive>true` arrives as
  a string; no schema coercion — the 2026-02-20 truthy-string risk.
- **Cache byte-parity (silent cost/wrong-KV):** native render on only some lanes
  diverges the delta/snapshot lanes.

## Fail-safe + gate (required)
- Detect the template tools slot behaviorally (render twice, with/without a
  sentinel tool; differ ⇒ slot exists), cached by template hash. No name
  allowlists.
- Loud per-request downshift to the prompted-merged fallback when the template
  render raises or the slot is absent; downshift cache mode to `keyed` on parity
  failure.
- Live A/B on Ornith/Qwen3.5 (MLX + GGUF + LMStudio-served), 5 prompts ×
  stream/non-stream (typed-arg call, zero-arg call, 2-call batch, 2-turn
  replay-with-result, no-tool answer): native ≥ prompted-baseline on structured
  recovery, arg-type fidelity, content cleanliness, turn-2 non-reissue, and
  cache LCP — before flipping the default.

## Scope / non-goals
- Do NOT enable llama.cpp's grammar-constrained function handler (the original
  bug); template rendering only.
- Keep the prompted-merged path as the permanent fallback for template-less /
  non-native models — do not delete it.

## References
- Tier-1 fixes: CHANGELOG 2026-07-15 (`merge_tools_into_system`; MLX
  message_format rendering).
- Design reports: five fable5 adversaries, 2026-07-15 (abstraction / placement /
  instruction / DRY / robustness).
- Related: 0821 (GGUF control-plane embedded-ChatML template detection) is a
  prerequisite-adjacent piece for the GGUF native render.
