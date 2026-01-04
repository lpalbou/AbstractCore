from __future__ import annotations

from types import SimpleNamespace

import pytest

from abstractcore.providers.anthropic_provider import AnthropicProvider


def _fake_anthropic_response(*, tool_use: bool) -> SimpleNamespace:
    content = []
    if tool_use:
        content.append(
            SimpleNamespace(
                type="tool_use",
                id="call_1",
                name="write_file",
                input={"file_path": "x.txt", "content": "hello"},
            )
        )
    else:
        content.append(SimpleNamespace(type="text", text="ok"))

    return SimpleNamespace(
        content=content,
        model="claude-haiku-4-5-20251001",
        stop_reason="tool_use" if tool_use else "end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def test_anthropic_provider_sends_native_tools_payload_and_does_not_prompt_tools(monkeypatch):
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_anthropic_response(tool_use=False)

    # Avoid any real HTTP calls
    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    tools = [
        {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "file_path": {"type": "string"},
                "start_line": {"type": "integer", "default": 1},
            },
        }
    ]

    resp = provider.generate(prompt="hi", system_prompt="SYS", tools=tools, temperature=0, max_output_tokens=16)

    call_params = captured.get("call_params") or {}
    assert call_params.get("system") == "SYS"
    assert "tools" in call_params, "Anthropic native tools must be sent in the request payload"
    assert call_params["tools"][0]["name"] == "read_file"
    assert "input_schema" in call_params["tools"][0]
    assert call_params["tools"][0]["input_schema"]["type"] == "object"
    assert call_params["tools"][0]["input_schema"]["required"] == ["file_path"]

    # This response is plain text; tool_calls should remain empty/None
    assert resp.tool_calls in (None, [])


def test_anthropic_provider_parses_tool_use_blocks_into_canonical_tool_calls(monkeypatch):
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    def fake_create(**_call_params):
        return _fake_anthropic_response(tool_use=True)

    # Avoid any real HTTP calls
    monkeypatch.setattr(provider.client.messages, "create", fake_create)
    # Ensure provider does not try to auto-execute tools during the unit test.
    monkeypatch.setattr(provider, "_handle_tool_execution", lambda resp, _tools: resp)

    tools = [
        {
            "name": "write_file",
            "description": "Write a file",
            "parameters": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
        }
    ]

    resp = provider.generate(prompt="call the tool", tools=tools, temperature=0, max_output_tokens=64)

    assert resp.tool_calls == [
        {"name": "write_file", "arguments": {"file_path": "x.txt", "content": "hello"}, "call_id": "call_1"}
    ]


def test_anthropic_provider_converts_role_tool_messages_to_tool_result_blocks(monkeypatch):
    """Anthropic Messages API does not accept role='tool'; tool outputs must be tool_result blocks."""
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_anthropic_response(tool_use=False)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    messages = [
        {"role": "assistant", "content": "ok"},
        {
            "role": "tool",
            "content": "[write_file]: Error: Invalid tool call: write_file requires content",
            "metadata": {"name": "write_file", "call_id": "toolu_123"},
        },
    ]

    provider.generate(prompt="continue", messages=messages, temperature=0, max_output_tokens=16)

    api_messages = (captured.get("call_params") or {}).get("messages") or []
    assert isinstance(api_messages, list)

    found = False
    for m in api_messages:
        if not isinstance(m, dict):
            continue
        if m.get("role") != "user":
            continue
        blocks = m.get("content")
        if not isinstance(blocks, list):
            continue
        for b in blocks:
            if not isinstance(b, dict):
                continue
            if b.get("type") != "tool_result":
                continue
            if b.get("tool_use_id") == "toolu_123":
                found = True
                assert "write_file" in str(b.get("content") or "")
    assert found, "Expected a tool_result content block for role='tool' messages"


