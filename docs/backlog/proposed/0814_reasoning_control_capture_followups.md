# Proposed: Reasoning control/capture follow-ups (async no-op, OpenRouter mapping, streaming default, vLLM gating)

## Metadata
- Created: 2026-07-07
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: `docs/adr/0001-engineering-guardrails-and-no-silent-degradation.md`
- ADR impact: None (applies ADR-0001; no new decision required)

## Context
The typed thinking-control surfaces wave (see CHANGELOG Unreleased) fixed the
`thinking_control` conflation (template-variable names appended as prompt text with
`handled=True`), made `thinking_effective` honest, wired LM Studio's native REST
`reasoning` control to `thinking=`, auto-closed truncated thinking blocks into
`metadata["reasoning"]`, and preserved `usage.completion_tokens_details`
(invisible-reasoning billing evidence, e.g. grok-4).

Three adversarial reviews (2026-07-07) surfaced additional defects that were
deliberately deferred to keep the wave focused. They are recorded here so they are
not lost.

## Deferred items

### 1. `agenerate(thinking=...)` is a silent no-op on native-async providers (HIGH)
`_apply_thinking_request` has exactly two call sites: sync `generate_with_telemetry`
and `prompt_cache_update`. `agenerate` passes `thinking` through `**kwargs` into
`_agenerate_internal`; providers overriding it natively (openai_compatible incl.
LM Studio, openai, ollama, anthropic) never consume it. Anthropic's async path even
expects a `thinking` dict, so the string is discarded. Only providers using the base
`to_thread -> sync generate` fallback apply thinking. Proven live (2026-07-07):
`agenerate(thinking="off")` on gemma-4 produced a clean payload with no control, no
metadata, no warning.

### 2. Native-async responses bypass response normalization (MEDIUM-HIGH)
`normalize_assistant_text` (wrapper stripping, Harmony final extraction, think-tag
extraction) and the streaming thinking stripper live inside sync
`generate_with_telemetry` only. Async callers of thinking-tag models get raw
`<think>` blocks in `content` and no `metadata["reasoning"]` from inline tags; async
streaming yields raw unstripped chunks. Fix direction: route async through the same
post-processing seam (or extract normalization into a shared helper invoked by both).

### 3. Streaming buffering when thinking state is unknown (MEDIUM)
`IncrementalThinkingTagStripper` starts in `searching` state and buffers ALL content
until a tag appears (closing-only capture support), releasing everything in the final
chunk when the response has no tags. The wave fixed the `thinking_effective == "off"`
case and the LM Studio native-route case (`reasoning` kwarg present ⇒ reasoning is
pre-separated by typed SSE events ⇒ `assume_visible_start=True`), but the default case
(thinking not requested at all on a server that strips tags server-side) still
degrades streaming into an end-of-stream burst. Fix direction: declare
closing-only-prone templates in assets (template injects the opening tag: Qwen3
thinking variants) and default everything else to eager visible mode.

### 4. OpenRouter reasoning mapping (see BUG_REPORT_openrouter_reasoning_controls.md)
`thinking=` / `reasoning_effort=` remain silent no-ops for OpenRouter routes. With
typed surfaces + honest handling, the failure is now LOUD (RuntimeWarning) instead of
silent, but the control still is not applied. Fix direction: map thinking to
OpenRouter's unified `reasoning: {enabled, effort, exclude}` request object (NOT a
blanket top-level `reasoning_effort`, which hard-fails on models like `x-ai/grok-4`
whose route rejects effort outright), consult per-model reasoning metadata from
`GET /api/v1/models` (`supported_efforts`, `mandatory`), and serialize the object into
the POST payload (nothing does today).

### 5. vLLM blanket `enable_thinking` (decision needed)
`vllm_provider.py` sends `extra_body.chat_template_kwargs.enable_thinking` for EVERY
architecture and claims handled, even for models declaring no template surface
(`test_thinking_xhigh_is_accepted` pins gpt-5.2-through-vLLM behavior). Gating on
`surfaces.template_kwarg` would be honest but changes pinned behavior; also verify
whether vLLM actually reads the nested `extra_body` key in raw JSON POSTs at all.

### 6. Closing-only heuristic false positive (LOW)
A model that merely mentions the literal end tag (e.g. `</think>` in a code answer on
a qwen-family model) has all preceding content silently reclassified as reasoning
(both strippers). Bounding closing-only mode to asset-declared templates (item 3)
also fixes this.

## Evidence notes (2026-07-07 probes, LM Studio 0.4.x)
Recorded so future work does not re-litigate these:
- `/api/v1/chat` REJECTS custom `tools` (400 `unrecognized_keys`), role-based message
  arrays (400 `invalid_union`; `input` parts accept only `text`/`image`), and
  `response_format` (400). Statefulness is via `response_id`, not request-side replay.
  This matches the official endpoint comparison table (Custom tools ❌, assistant
  messages ❌).
- `/api/v1/chat` SUPPORTS streaming (SSE events incl. `reasoning.delta`,
  `message.delta`, `chat.end` with `stats.reasoning_output_tokens`) and image input
  parts (`{"type": "image", "data_url": "data:image/png;base64,..."}`) — both now
  consumed by `LMStudioProvider`.
- LM Studio `/v1/responses` accepts custom tools, assistant history, tool-result
  replay (`function_call`/`function_call_output` items), and a
  `reasoning: {effort: minimal|low|medium|high}` object — but the effort control was a
  NO-OP for gemma-4-26b-a4b (zero reasoning items/tokens at every effort, no
  off/none enum), so it is not a usable thinking-control surface for the
  tools-with-thinking case. If LM Studio later honors it, `/v1/responses` would be the
  natural route for tools+history+reasoning in one request.

## Verification sketch
- Async parity: unit tests asserting `agenerate(thinking=...)` produces the same
  payload artifacts and metadata as `generate(thinking=...)` per provider; async
  content normalization asserted with a fake `<think>` response.
- OpenRouter: payload-capture unit tests for `reasoning` object mapping incl. the
  grok-4 effort-rejection case; live smoke behind `ABSTRACTCORE_RUN_LIVE_API_TESTS`.
- Streaming default: chunk-timing test that tagless streams flow incrementally for a
  non-closing-only architecture while Qwen thinking streams still capture reasoning.
