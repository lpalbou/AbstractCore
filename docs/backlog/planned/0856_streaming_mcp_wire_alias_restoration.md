# 0856-core: [BUG] Restore original MCP tool names in streaming responses

> Package: abstractcore
> Type: bug
> Status: planned
> Created: 2026-10-22
> Priority: P1
> Labels: mcp, streaming, tool-calling, gateway

## Summary

Restore provider-safe native tool aliases to the declared original tool names in streaming responses, with the same allowlist checks as non-streaming responses. A gateway-hosted Assistant MCP call currently reaches execution as `mcp_RAGnarok_query_b6ff8949` instead of `mcp::RAGnarok::query` and is rejected before reaching RAGnarok. The user reports that disabling model streaming makes the same integration work. This item authorizes investigation and a targeted Core fix, not weakening gateway permissions.

## Incident and reproduction

On 2026-10-22 an admin registered RAGnarok 0.5.0 at `/mcp/`, using a valid MCP-purpose bearer. The gateway handshake discovered 14 tools. Desktop, iPhone and Watch all displayed the catalog; non-streaming calls successfully listed workspaces and queried `epar`.

1. Register/test/enable an authenticated MCP server; allow its tools in an Assistant run (Ask is sufficient).
2. Use a native tool-calling OpenAI-compatible provider and enable model streaming.
3. Ask `give me a short summary on gazyvaro using the "epar" workspace of RAGnarok`.
4. Observe the failed tool event: `Tool 'mcp_RAGnarok_query_b6ff8949' is not allowed for this node or run`.
5. Disable streaming and repeat with the same model, tool policy and credentials; the operator reports success.

Evidence supplied by the operator: run `d737d3f3-c7c6-412e-8f02-9d6465f2d39d`, model `gpt-5.6-luna`, call id `call_d8d0ee3fa3fe4d86cb2802250e52848`, arguments include `workspace=epar`, `mode=quick`. Tool settings list the canonical `mcp::RAGnarok::query`. Screenshots are evidence of the user-visible failure; the original ledger/provider trace has NOT been inspected. Rehydrate that run if still available before treating the diagnosis as complete. Do not copy tokens into a fixture, log or prompt.

Earlier runs `ba9e00de-5fe5-4344-a9ae-0433d0d77221` and `e0d93b6b-903b-4387-aeb7-6703f9597514` failed with `Tool-call format repair failed: no executable tool calls after bounded retries` (the latter used Qwen3.8-27B). Those failures may have a separate model/parser cause; do not claim this fix resolves them without evidence.

## Current code reality

- `abstractcore/tools/wire_naming.py`: `wire_safe_tool_name` encodes names unsuitable for provider-native declarations; `resolve_wire_tool_name` maps only against the current allowed original names. Keep this deterministic, collision-resistant contract.
- `abstractcore/providers/base.py`: `_normalize_tool_calls_payload` restores wire aliases against `allowed_tool_names`; `_normalize_tool_calls_passthrough` uses it on the non-streaming pass-through path. The `unified_stream` closure routes streaming through `UnifiedStreamProcessor`; inspection found no corresponding per-request alias restoration at that boundary.
- `abstractcore/providers/streaming.py`: `UnifiedStreamProcessor` preserves provider-emitted native `tool_calls` in pass-through chunks. Inspect complete-call versus partial-delta behavior before adding normalization.
- `abstractcore/providers/openai_provider.py`: sync/async native stream assembly; inspect the OpenAI-compatible provider's inherited path as well.
- `../abstractruntime/src/abstractruntime/integrations/abstractcore/llm_client.py`: `_normalize_local_streaming_response` aggregates streamed calls, preserving their names and deduplicating by call id. It has no allowed-name parameter, so it cannot safely restore aliases on its own today.
- `../abstractgateway/src/abstractgateway/mcp_run_tools.py` and approval/runtime executors use canonical `mcp::<server>::<tool>` names. Treat this refusal as correct enforcement, not a reason to grant aliases permission.
- Existing tests: `tests/tools/test_wire_safe_tool_names.py` exercise declaration and normalization helpers; `tests/streaming/test_stream_tool_calling.py`, `test_stream_envelope_settlement.py`, and `test_unified_streaming.py` exercise streaming. Add a full public-generate streaming/non-streaming parity regression, not only a resolver unit test.

## Scope and suggested approach

Reproduce without a live model using a stub provider yielding complete native calls with deterministic wire names. Find the earliest shared boundary where both the complete call and the request's allowed tool declarations are available. Reuse the existing resolver there, including sync and async generate paths. Keep on-wire aliases for strict providers and canonical names in host-facing responses and durable history. Preserve original call ids, arguments, usage, reasoning and cancellation behavior. Normalization must not turn partial streamed JSON into empty arguments, drop valid calls or emit duplicates.

Coordinate a Runtime follow-up only if its aggregation contract actually needs changing; Core should normally own provider-name normalization. Never patch every client, blindly strip hash suffixes, turn off streaming globally, or relax run allowlists to hide this bug.

## Acceptance Criteria

- [ ] Deterministic failing regression demonstrates a native streaming alias escaping through public Core `generate` before the fix.
- [ ] Sync and async streaming/non-streaming return identical canonical names, arguments and call ids for an MCP tool; two concurrent requests cannot share another request's alias map.
- [ ] Strict native declarations/history still carry wire-safe names. Already-safe tool names are unchanged.
- [ ] Unknown aliases remain rejected; collisions, long names, multiple calls, split argument deltas, repeated terminal envelopes and cancellation preserve existing behavior.
- [ ] Runtime aggregation receives the canonical name; gateway Ask reaches approval then the registered MCP executor, Auto works, and Off remains rejected.
- [ ] A real gateway streaming run is verified when credentials/service are available, or the missing live check is stated explicitly. No claim that separate Qwen repair failures are fixed without reproduction.

## Testing

From AbstractCore: `python -m pytest tests/tools/test_wire_safe_tool_names.py tests/streaming/test_stream_tool_calling.py tests/streaming/test_stream_envelope_settlement.py tests/streaming/test_unified_streaming.py`.
Add a focused fake-provider regression using both stream modes and async iteration; broaden only for shared code paths actually changed. From AbstractGateway: `python -m pytest tests/test_gateway_mcp_runs.py`. Include a Runtime aggregation regression if its code changes.

## ADR status

No new policy intended: restore existing wire-name/allowlist semantics consistently across transport modes. Consult `docs/adr` and `tools/wire_naming.py` before implementation; any change to allowlist interpretation requires separate design review.

## Non-goals

Per-user MCP credentials (Gateway planned 0936), remote authentication, tool-risk classification, provider/model-specific parser repair, and UI styling. Streaming off remains the temporary workaround.

## Receipts

- 2026-10-22: code-path audit and operator screenshots; diagnosis strongly supported by the escaped alias, but no deterministic full-generate reproduction or live trace read performed yet. Implementation deferred by operator instruction.
