# Completed: output-token cap should not impose a silent budget

## Metadata
- Created: 2026-07-15
- Status: Completed
- Completed: 2026-07-15 (flow, at maintainer's request while core was occupied; reviewed by fable5 adversaries)

## Resolution
Shipped in the same-day fix (see CHANGELOG "Output-token cap no longer imposes
a SILENT budget"). Mechanism: `_max_output_tokens_explicit` sentinel in
`core/interface.py`; `_prepare_generation_kwargs` stops fabricating the
registry default; `_resolve_output_token_cap()` returns None (omit) when
unspecified unless `_requires_output_cap()`; openai/openai-compatible call
sites omit on None; Anthropic/Ollama/LMStudio keep a required bound; and
`_annotate_output_truncation()` warns + annotates on `finish_reason=length`
(no silent truncation). Live-verified on OVH `gpt-oss-120b`; regression pins in
`tests/providers/test_output_token_no_silent_budget.py`. Acceptance criteria
below all met. The gpt-oss-120b registry-value re-check remains a separate
follow-up.

## ADR status
- Governing ADRs: use-full-capability / no-silent-budget (abstractruntime/abstractflow lane)
- ADR impact: abstractcore currently violates it for output tokens

## Context (flow investigation, maintainer-flagged, 2026-07-15)
abstractcore ALWAYS writes the output-token parameter on the wire, and when the
caller imposes no cap it defaults to the model's registry `max_output_tokens`.
That is a silent per-call output budget: for a model whose registry cap is below
the endpoint's true ceiling, calls needing more output are truncated with only
`finish_reason=length` as the signal.

Measured (flow, co-scientist run, gpt-oss-120b, 582 llm_calls): max observed
output 6926 tokens, zero `finish_reason=length` — so it has not bitten yet, but
`max_output_tokens=8192` is shipped on every call even when the flow imposed no
budget, and gpt-oss-120b's context is 128000 (the 8192 cap is an under-cap).

## Current code reality
- `openai_provider.py:356` (and the openai-compatible + base payload builders)
  unconditionally set `call_params[self._get_token_param_name()] = max_output_tokens`.
- `_get_provider_max_tokens_param` → `kwargs.get("max_output_tokens", self.max_output_tokens)`
  (`openai_provider.py:1017`, `base.py:4584`).
- `self.max_output_tokens` is never None: `_initialize_token_limits`
  (`base.py:4340-4350`) promotes the 2048 default to the registry cap.
- There is no "caller passed nothing → omit / use full capability" path.

## Direction (scope carefully — behavior change on every OpenAI/compatible call)
- When the caller imposes NO output cap, do not inject the registry cap: either
  omit the token param (true full-capability) or send the endpoint's real max.
- Keep EXPLICIT caps honored, and keep the reasoning-model `max_completion_tokens`
  path intact (those models require the param).
- Verify per-server behavior first: some servers require a bound; confirm vLLM /
  OVH omit-behavior before flipping the default. A per-provider "requires a bound"
  flag may be needed rather than a blanket omit.
- Distinguish "caller specified none" from "registry default" internally
  (`self.max_output_tokens` needs a sentinel for unspecified vs resolved).

## Separate follow-up
Re-check the gpt-oss-120b registry `max_output_tokens=8192` against the true OVH
vLLM output ceiling; correct the registry number if it is an artificial under-cap.

## Acceptance
- A caller that passes no output cap does not silently receive the registry cap.
- Explicit caps and reasoning-model token-param requirements still hold.
- Servers that require a bound still get one (documented per-provider).
