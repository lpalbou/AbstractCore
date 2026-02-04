from __future__ import annotations

from types import SimpleNamespace

import pytest

from abstractcore.media.types import ContentFormat, MediaContent, MediaType
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

    # Avoid importing media-processing extras for this unit test; MediaContent is already normalized.
    monkeypatch.setattr(OpenAIProvider, "_process_media_content", lambda self, media, *a, **k: list(media))

    return OpenAIProvider(model=model, api_key="test")


def test_openai_provider_non_vision_does_not_silently_drop_images(monkeypatch):
    # Force "text-only" path deterministically.
    import abstractcore.architectures.detection as detection_module

    monkeypatch.setattr(detection_module, "supports_vision", lambda _name: False)

    provider = _make_provider(monkeypatch, model="gpt-3.5-turbo")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_openai_response()

    monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)

    media = [
        MediaContent(
            media_type=MediaType.IMAGE,
            content=b"not-a-real-jpeg-but-bytes-are-fine",
            content_format=ContentFormat.BASE64,
            mime_type="image/jpeg",
            file_path="x.jpg",
            metadata={"file_name": "x.jpg"},
        )
    ]

    resp = provider.generate(
        prompt="what are those images ?",
        media=media,
        temperature=0.0,
        max_output_tokens=16,
    )

    api_messages = (captured.get("call_params") or {}).get("messages") or []
    assert isinstance(api_messages, list) and api_messages, "Expected OpenAI call to include messages"

    last_user = None
    for m in reversed(api_messages):
        if isinstance(m, dict) and m.get("role") == "user":
            last_user = m
            break
    assert last_user is not None, "Expected at least one user message"

    content = last_user.get("content")
    assert isinstance(content, str), "Expected text-embedded fallback message for non-vision model"
    assert "[Image 1:" in content, "Expected a non-silent placeholder for attached image(s)"

    enrich = (resp.metadata or {}).get("media_enrichment")
    assert isinstance(enrich, list) and enrich, "Expected media enrichment metadata for fallback transparency"
    assert any(isinstance(i, dict) and i.get("input_modality") == "image" for i in enrich)


def test_openai_provider_attaches_media_when_prompt_empty(monkeypatch):
    import abstractcore.architectures.detection as detection_module

    monkeypatch.setattr(detection_module, "supports_vision", lambda _name: False)

    provider = _make_provider(monkeypatch, model="gpt-3.5-turbo")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_openai_response()

    monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)

    media = [
        MediaContent(
            media_type=MediaType.IMAGE,
            content=b"not-a-real-jpeg-but-bytes-are-fine",
            content_format=ContentFormat.BASE64,
            mime_type="image/jpeg",
            file_path="x.jpg",
            metadata={"file_name": "x.jpg"},
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
    assert isinstance(api_messages, list) and api_messages, "Expected OpenAI call to include messages"
    assert len(api_messages) == 1, "Expected last user message to be replaced (no duplication)"

    last_user = api_messages[-1]
    assert isinstance(last_user, dict) and last_user.get("role") == "user"
    content = last_user.get("content")
    assert isinstance(content, str), "Expected text-embedded fallback message for non-vision model"
    assert "what are those images ?" in content
    assert "[Image 1:" in content


def test_openai_provider_uses_native_multimodal_for_vision_models(monkeypatch):
    import abstractcore.architectures.detection as detection_module

    monkeypatch.setattr(detection_module, "supports_vision", lambda _name: True)

    provider = _make_provider(monkeypatch, model="gpt-4o")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_openai_response()

    monkeypatch.setattr(provider.client.chat.completions, "create", fake_create)

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
        prompt="what are those images ?",
        media=media,
        temperature=0.0,
        max_output_tokens=16,
    )

    api_messages = (captured.get("call_params") or {}).get("messages") or []
    assert isinstance(api_messages, list) and api_messages, "Expected OpenAI call to include messages"

    last_user = api_messages[-1]
    assert isinstance(last_user, dict) and last_user.get("role") == "user"

    content = last_user.get("content")
    assert isinstance(content, list), "Expected native multimodal content list for vision-capable model"
    assert any(isinstance(b, dict) and b.get("type") == "image_url" for b in content), "Expected an image_url block"
    assert any(isinstance(b, dict) and b.get("type") == "text" for b in content), "Expected a text block"

