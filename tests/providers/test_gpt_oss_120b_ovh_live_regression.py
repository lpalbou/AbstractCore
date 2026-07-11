"""C2 regression harness (live half) — native tool-call round-trip on the OVH path.

Consensus plan (plans/entity-topology-consensus-plan.md, Parallel section): the
gpt-oss-120b native-tool-call probe is demoted to a REGRESSION HARNESS — one live
test pinning the native round-trip on the exact provider path summoned entities use
(`endpoint:ovh-provider` / gpt-oss-120b), kept because the Harmony header race lives
on this path: the shipped retry carve-out (see
tests/providers/test_harmony_transient_artifact_regression.py for the offline pins)
absorbs a mid-run 400 `unexpected tokens remaining in message header` by resampling,
so a single race occurrence must NOT fail this test — that absorption is itself part
of the pinned behavior.

Gating (never runs in the offline suite):
- ABSTRACTCORE_RUN_LIVE_API_TESTS=1
- the `endpoint:ovh-provider` profile configured (abstractcore config), or
  OVH_BASE_URL + OVH_AI_API_KEY env vars as a fallback construction path.

Fallback substrate note (laurent c157): on persistent OVH issues the operator-declared
fallback is LMStudio `qwen/qwen3.6-35b-a3b` — engaged via substrate config, never a
code default. Its registry readiness is pinned offline; this file deliberately does
NOT auto-fall-back (a live regression harness that silently switches endpoints would
hide the regression it exists to catch).
"""

from __future__ import annotations

import os

import pytest

from abstractcore import create_llm

pytestmark = pytest.mark.skipif(
    os.getenv("ABSTRACTCORE_RUN_LIVE_API_TESTS") != "1",
    reason="Live API tests disabled (set ABSTRACTCORE_RUN_LIVE_API_TESTS=1)",
)

MODEL = os.getenv("OVH_GPT_OSS_MODEL", "gpt-oss-120b")

ADD_TOOL = {
    "name": "add_numbers",
    "description": "Add two integers and return their sum.",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {"type": "integer", "description": "First addend"},
            "b": {"type": "integer", "description": "Second addend"},
        },
        "required": ["a", "b"],
    },
}


def _ovh_llm():
    """The exact path entities ride: the endpoint profile; env construction as fallback."""
    try:
        from abstractcore.config import get_config_manager

        profile = get_config_manager().resolve_provider_profile("endpoint:ovh-provider")
    except Exception:
        profile = None

    if profile is not None:
        return create_llm("endpoint:ovh-provider", model=MODEL)

    base_url = os.getenv("OVH_BASE_URL")
    api_key = os.getenv("OVH_AI_API_KEY")
    if not base_url or not api_key:
        pytest.skip(
            "OVH path unavailable: configure the endpoint:ovh-provider profile "
            "(abstractcore config set-provider ovh-provider ...) or set "
            "OVH_BASE_URL + OVH_AI_API_KEY"
        )
    return create_llm("openai-compatible", model=MODEL, base_url=base_url, api_key=api_key)


def test_native_tool_call_round_trip_live():
    llm = _ovh_llm()

    # Leg 1: the model must elect the tool through NATIVE function calling
    # (payload `tools` + structured `tool_calls`, no fenced-text convention).
    first = llm.generate(
        "Use the add_numbers tool to compute 37 + 5. Call the tool; do not answer directly.",
        tools=[ADD_TOOL],
        temperature=0,
    )

    assert first.tool_calls, (
        f"no native tool_calls returned (content={first.content!r}) — either the model "
        "regressed on native tools for this route or the request never reached it"
    )
    call = first.tool_calls[0]
    assert call["name"] == "add_numbers"
    args = call["arguments"]
    assert {int(args["a"]), int(args["b"])} == {37, 5}
    call_id = call.get("call_id") or "call_0"

    # Leg 2: feed the tool result back; the final answer must use it.
    second = llm.generate(
        "",
        messages=[
            {"role": "user", "content": "Use the add_numbers tool to compute 37 + 5."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": "add_numbers", "arguments": '{"a": 37, "b": 5}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call_id, "content": "42"},
        ],
        tools=[ADD_TOOL],
        temperature=0,
    )

    assert "42" in (second.content or ""), (
        f"final answer did not use the tool result: {second.content!r}"
    )
