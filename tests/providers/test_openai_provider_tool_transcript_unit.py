from __future__ import annotations

from types import SimpleNamespace

from abstractcore.providers.openai_provider import OpenAIProvider


def _fake_openai_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="stop",
            )
        ],
        model="stubbed-openai",
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def _install_fake_openai(monkeypatch) -> None:
    import abstractcore.providers.openai_provider as openai_provider_module

    class _FakeOpenAIClient:
        def __init__(self, **_kwargs):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **_k: _fake_openai_response()))
            self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[]))

    fake_openai = SimpleNamespace(OpenAI=_FakeOpenAIClient, AsyncOpenAI=_FakeOpenAIClient)
    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True, raising=False)
    monkeypatch.setattr(openai_provider_module, "openai", fake_openai, raising=False)


def _make_provider(monkeypatch, *, model: str) -> OpenAIProvider:
    _install_fake_openai(monkeypatch)

    # Avoid network and keep this Level A test deterministic.
    monkeypatch.setattr(OpenAIProvider, "_validate_model_exists", lambda self: None)

    return OpenAIProvider(model=model, api_key="test")


def test_openai_provider_preserves_tool_calls_and_tool_call_id_in_messages(monkeypatch):
    provider = _make_provider(monkeypatch, model="gpt-3.5-turbo")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_openai_response()

    monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)

    messages = [
        {"role": "user", "content": "List files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "list_files", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "content": "file1.txt\nfile2.txt", "tool_call_id": "call_1"},
    ]

    provider.generate(prompt="", messages=messages, max_output_tokens=16)

    api_messages = (captured.get("call_params") or {}).get("messages") or []
    assert isinstance(api_messages, list)
    assert api_messages[1].get("tool_calls") == messages[1]["tool_calls"]
    assert api_messages[2].get("tool_call_id") == "call_1"

