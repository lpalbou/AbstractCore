# Proposed: System-message handling follow-ups (Opus 4.8 native mid-stream system, developer-role position, CachedSession replay)

## Metadata
- Created: 2026-07-08
- Status: Proposed
- Completed: N/A

## ADR status
- Governing ADRs: `docs/adr/0001-engineering-guardrails-and-no-silent-degradation.md`
- ADR impact: None (applies ADR-0001)

## Context
The mid-stream system-message fix (see CHANGELOG Unreleased) stopped the native
OpenAI/Anthropic providers from silently dropping `role:"system"` messages inside
`messages`: OpenAI passes them through verbatim; Anthropic hoists a leading run into the
top-level `system` parameter and converts non-leading ones into position-preserving
`<system_instruction>`-wrapped user turns (deferred past tool_result runs, counted in
`metadata["system_role_user_wrapped"]`). BasicSession now delivers non-system-prompt
system messages (compaction summaries), and the HF native-video history collapse includes
SYSTEM turns.

Two adversarial reviews (2026-07-08) identified deliberate deferrals, recorded here.

## Deferred items

### 1. Registry-gated native mid-conversation `system` for Claude Opus 4.8 (MEDIUM value)
As of the Opus 4.8 release, Anthropic's Messages API accepts `role:"system"` mid-
conversation on that model only (SDK `MessageParam.role` includes "system"; other models
return 400). Native passthrough would preserve operator authority AND position without
the user-wrap conversion. Constraints that make this a careful follow-up, not part of the
minimal fix: placement validation required (not `messages[0]`; must follow a user turn;
no consecutive system messages; never between `tool_use` and `tool_result`; must be last
or followed by an assistant turn), and support is route-dependent (Anthropic platform
docs say not available on Bedrock/Vertex while an AWS docs page claims Bedrock support —
contradictory as of 2026-07-08). Shape: a `mid_stream_system_role: true` capability in
`model_capabilities.json` (assets own model knowledge) consumed by
`_build_anthropic_history`, falling back to the wrap conversion when absent.

### 2. `developer`-role position asymmetry (LOW-MEDIUM)
After the fix, mid-stream `system` keeps its position on both native providers, but
mid-stream `developer` messages are still hoisted position-lossily into `system_prompt`
by `BaseProvider._normalize_developer_messages` (base.py ~1602) for every provider except
OpenAI. CAUTION (maintainer note): some systems deliberately use `developer` — the
current hoist is load-bearing and must not be removed; any change should only make the
conversion position-aware, keep `_supports_developer_messages()` semantics intact, and
ship with parity tests (`tests/providers/test_system_prompt_alias.py` covers the current
behavior).

### 3. CachedSession transcript replay excludes system messages (LOW)
`cached_session.py` rebuild/replay paths historically treat system content as
prefix-only. With BasicSession now delivering compaction summaries in-stream, verify the
KV-mode transcript replay renders mid-stream system messages consistently with the live
path (the transcript now includes them via the shared formatter; the fragment renderers
do render `system` roles — confirm cache-prefix stability expectations still hold).

### 4. Anthropic unknown-role demotion is silent (LOW)
`_build_anthropic_history` delivers any role other than user/assistant/tool/system as
plain user content (Anthropic's API accepts only user/assistant). This predates the fix
and is the only representable option, but it is unlabeled. If it ever matters, count it
in metadata like `system_role_user_wrapped`.

## Evidence notes (2026-07-08)
- OpenAI Chat Completions accepts `system`/`developer` at arbitrary positions; reasoning
  models auto-treat `system` as `developer`; only deprecated `o1-mini`/`o1-preview`
  reject both (they already failed on the leading `system_prompt` message before the fix).
- Anthropic consecutive same-role user turns are combined server-side (alternation is not
  a constraint for the wrap conversion). The one hard placement rule: nothing between
  `tool_use` and its `tool_result` — encoded in `_build_anthropic_history` via deferred
  emission.
- Prior art: LiteLLM hoists all in-array system content into the top-level `system`
  (position lost); LangChain and Vercel AI SDK raise errors on non-consecutive mid-stream
  system messages; nobody drops silently.
- Server path: `/v1/chat/completions` passes client messages verbatim with no
  `system_prompt` extraction (`server/app.py` ~8403), so the provider layer is the only
  place that can deliver client system prompts on native backends. Pinned by
  `tests/server/test_server_system_message_passthrough.py`.
