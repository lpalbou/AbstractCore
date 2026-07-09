"""Strict OpenAI-compatible servers (vLLM/OVH) reject non-leading system messages
("System message must be at the beginning.", HTTP 400 — live incident 2026-07-09: the
runtime's tail-appended attachment index failed a production assistant's FIRST message).

The provider must normalize at the transport boundary, mirroring the shipped Anthropic
behavior: leading system run merged to one message, non-leading system messages converted to
<system_instruction>-wrapped user messages, deferred past tool-result runs so assistant
tool_calls / tool adjacency is never broken.
"""

from __future__ import annotations

from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

_norm = OpenAICompatibleProvider._normalize_system_messages_for_strict_servers


def test_tail_system_message_becomes_wrapped_user():
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "system", "content": "Attached files: a.txt"},  # the incident shape
    ]
    out = _norm(msgs)
    assert [m["role"] for m in out] == ["system", "user", "assistant", "user"]
    assert out[-1]["content"] == "<system_instruction>\nAttached files: a.txt\n</system_instruction>"


def test_mid_stream_system_message_wrapped_in_place():
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "q1"},
        {"role": "system", "content": "[CONVERSATION HISTORY] compacted summary"},
        {"role": "user", "content": "q2"},
    ]
    out = _norm(msgs)
    assert [m["role"] for m in out] == ["system", "user", "user", "user"]
    assert "compacted summary" in out[2]["content"] and out[2]["content"].startswith("<system_instruction>")


def test_system_between_tool_calls_and_results_defers_past_tool_run():
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "do it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "id": "c1", "function": {"name": "f", "arguments": "{}"}},
            {"type": "function", "id": "c2", "function": {"name": "g", "arguments": "{}"}},
        ]},
        {"role": "system", "content": "index update"},  # must NOT split the tool run
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
        {"role": "user", "content": "next"},
    ]
    out = _norm(msgs)
    roles = [m["role"] for m in out]
    # tool results stay adjacent to the assistant tool_calls turn; the wrapped system lands after.
    assert roles == ["system", "user", "assistant", "tool", "tool", "user", "user"]
    assert out[5]["content"].startswith("<system_instruction>")
    assert "index update" in out[5]["content"]


def test_multiple_leading_system_messages_merge_to_one():
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
    ]
    out = _norm(msgs)
    assert [m["role"] for m in out] == ["system", "user"]
    assert "persona" in out[0]["content"] and "rules" in out[0]["content"]


def test_clean_lists_pass_through_unchanged():
    msgs = [
        {"role": "system", "content": "persona"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    assert _norm(list(msgs)) == msgs
    assert _norm([]) == []


def test_trailing_system_after_final_tool_run_flushes_at_end():
    msgs = [
        {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "id": "c1", "function": {"name": "f", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "system", "content": "tail index"},
    ]
    out = _norm(msgs)
    assert [m["role"] for m in out] == ["assistant", "tool", "user"]
    assert out[-1]["content"].startswith("<system_instruction>")


class _CaptureClient:
    """Captures the payload at the HTTP boundary — pins that the request PIPELINE invokes
    normalization (the incident's gap class was exactly unit-tested-helper + unverified wiring)."""

    def __init__(self):
        self.payloads = []

    def post(self, url, json=None, headers=None):  # noqa: A002
        self.payloads.append(dict(json or {}))

        class _R:
            status_code = 200

            @staticmethod
            def json():
                return {
                    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }

        return _R()


def test_wire_path_generate_normalizes_tail_system_message():
    """END-TO-END wiring pin: a real `generate()` call with a tail system message must reach
    the HTTP boundary already normalized. (The incident's gap class was exactly a unit-tested
    helper with unverified wiring — this test fails if the `_generate_internal` call site is
    ever dropped.)"""
    p = OpenAICompatibleProvider(model="stub-model", base_url="http://127.0.0.1:9/v1", api_key="x")
    p.client = _CaptureClient()

    p.generate(
        "",
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "again"},
            {"role": "system", "content": "Attached files: a.txt"},  # the incident shape
        ],
        system_prompt="persona",
        max_output_tokens=8,
    )

    sent = p.client.payloads[-1]["messages"]
    roles = [m["role"] for m in sent]
    assert roles[0] == "system", roles
    assert "system" not in roles[1:], f"non-leading system reached the wire: {roles}"
    wrapped = [m for m in sent if str(m.get("content") or "").startswith("<system_instruction>")]
    assert wrapped and "Attached files: a.txt" in wrapped[-1]["content"]


def test_structured_content_parts_are_not_python_reprs():
    # OpenAI content-part lists must extract text, never stringify to a repr.
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": "persona"}]},
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": [{"type": "text", "text": "tail note"}]},
    ]
    out = _norm(msgs)
    assert out[0]["content"] == "persona\n\nrules"
    assert "{'type'" not in out[0]["content"]
    assert out[-1]["content"] == "<system_instruction>\ntail note\n</system_instruction>"


def test_empty_leading_systems_do_not_emit_empty_message():
    msgs = [
        {"role": "system", "content": ""},
        {"role": "user", "content": "hi"},
        {"role": "system", "content": "tail"},
    ]
    out = _norm(msgs)
    assert out[0]["role"] == "user"  # no empty leading system emitted
    assert out[-1]["content"].startswith("<system_instruction>")
