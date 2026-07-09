# Bug report: `thinking` / `reasoning_effort` are silent no-ops for the OpenRouter provider

Reported: 2026-07-06 (ai-psychology repo, gpt-5-mini reasoning-condition audit)
AbstractCore version: 2.13.38

## Summary

For `provider="openrouter"` (subclass of `OpenAICompatibleProvider`), neither the unified
`thinking=` control nor a `reasoning_effort=` kwarg reaches the HTTP payload. Requests are
sent with no reasoning control at all, so reasoning-capable models run with the server-side
default (reasoning active), while the caller believes reasoning was disabled. No warning is
emitted.

## Where

- `abstractcore/providers/openai_compatible_provider.py::_apply_provider_thinking_kwargs`
  returns unchanged kwargs unless `provider_id in {"lmstudio", "openai-compatible"}` —
  `"openrouter"` falls through, so `thinking=False` is dropped silently.
- The payload builder in `OpenAICompatibleProvider.generate` never serializes
  `reasoning_effort` (only the `openai` provider maps it, via `call_params["reasoning_effort"]`).
- `extra_body` passthrough in `_mutate_payload` nests the dict under a literal `"extra_body"`
  key in the raw JSON POST body; for OpenRouter (raw HTTP, not the OpenAI SDK) the contents
  are therefore ignored upstream, so it cannot be used as a workaround either.

## Observed impact (real runs)

- `openai/gpt-5-mini` via OpenRouter with config `thinking: false` → 83–92% of completion
  tokens were hidden reasoning across 12 measurement suites (March 2026 batch).
- Qwen3.5 / GLM hosted endpoints via OpenRouter with `thinking: false` (May 2026 configs)
  → server-side thinking stayed active (hundreds of k of reasoning chars in archives).

## Expected behavior

OpenRouter accepts OpenAI-style reasoning controls on `POST /chat/completions`:

- `"reasoning_effort": "minimal" | "low" | ...` (top level), and/or
- `"reasoning": {"effort": "...", "exclude": true|false, "enabled": true|false}`.

Verified empirically (2026-07-06): `reasoning_effort: "minimal"` on `openai/gpt-5-mini`
returns `usage.completion_tokens_details.reasoning_tokens == 0`; omitting it yields
~100+ reasoning tokens on the same prompt. Note `"none"` is rejected by the upstream for
gpt-5-mini ("Reasoning is mandatory for this endpoint"), so the provider's
thinking=off fallback ladder should try `none` → `minimal` → `low` per model.

## Suggested fix

In `OpenRouterProvider` (or the compatible base, gated on provider id):

1. Map the normalized thinking control (enabled/level) to a top-level
   `reasoning_effort` payload key (or `reasoning: {effort, exclude}`), mirroring
   `OpenAIProvider._apply_provider_thinking_kwargs` including the
   "cannot fully disable → nearest supported minimum + RuntimeWarning" path.
2. Forward an explicit `reasoning_effort=` generate-kwarg into the payload.
3. Emit a warning when a thinking control is requested but cannot be applied,
   instead of dropping it silently (this is what made the failure invisible).

## Workaround in use (ai-psychology repo)

The v8 runner's pinned-effort chat.completions shim (`_OpenAIChatCompletionsReasoningLLM`)
was generalized to accept any explicit `reasoning_effort` + OpenRouter `base_url` +
`OPENROUTER_API_KEY`, bypassing AbstractCore for those calls.
