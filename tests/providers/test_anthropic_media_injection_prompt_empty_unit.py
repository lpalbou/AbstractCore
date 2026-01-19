from __future__ import annotations

from types import SimpleNamespace

from abstractcore.media.types import ContentFormat, MediaContent, MediaType
from abstractcore.providers.anthropic_provider import AnthropicProvider


def _fake_anthropic_response() -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        model="claude-haiku-4-5-20251001",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def test_anthropic_provider_attaches_media_when_prompt_empty(monkeypatch):
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_anthropic_response()

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    media = [
        MediaContent(
            media_type=MediaType.IMAGE,
            content=b"not-a-real-jpeg-but-bytes-are-fine",
            content_format=ContentFormat.BASE64,
            mime_type="image/jpeg",
            file_path="x.jpg",
        )
    ]

    provider.generate(
        prompt="",
        messages=[{"role": "user", "content": "what are those images ?"}],
        media=media,
        temperature=0.0,
        max_output_tokens=16,
    )

    api_messages = (captured.get("call_params") or {}).get("messages") or []
    assert isinstance(api_messages, list) and api_messages, "Expected Anthropic call to include messages"

    last_user = None
    for m in reversed(api_messages):
        if isinstance(m, dict) and m.get("role") == "user":
            last_user = m
            break
    assert last_user is not None, "Expected at least one user message"

    content = last_user.get("content")
    assert isinstance(content, list), "Expected multimodal content list for user message"
    assert any(isinstance(b, dict) and b.get("type") == "image" for b in content), "Expected an image content block"
    assert any(isinstance(b, dict) and b.get("type") == "text" for b in content), "Expected a text content block"

