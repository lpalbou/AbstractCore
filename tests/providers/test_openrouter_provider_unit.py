import pytest

from abstractcore.providers.openrouter_provider import OpenRouterProvider


def test_openrouter_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(OpenRouterProvider, "_get_api_key_from_config", lambda self: None)
    monkeypatch.setattr(OpenRouterProvider, "_validate_model", lambda self: None)

    with pytest.raises(ValueError, match="OpenRouter API key required"):
        OpenRouterProvider(model="openai/gpt-4o-mini")


def test_openrouter_uses_config_api_key_when_env_missing(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(OpenRouterProvider, "_get_api_key_from_config", lambda self: "sk-test-from-config")
    monkeypatch.setattr(OpenRouterProvider, "_validate_model", lambda self: None)

    provider = OpenRouterProvider(model="openai/gpt-4o-mini")
    assert provider.api_key == "sk-test-from-config"


def test_openrouter_headers_include_auth_and_metadata(monkeypatch):
    monkeypatch.setenv("OPENROUTER_SITE_URL", "https://example.com")
    monkeypatch.setenv("OPENROUTER_APP_NAME", "abstractframework-test")
    monkeypatch.setattr(OpenRouterProvider, "_validate_model", lambda self: None)

    provider = OpenRouterProvider(model="openai/gpt-4o-mini", api_key="sk-test-key")
    headers = provider._get_headers()

    assert headers["Authorization"] == "Bearer sk-test-key"
    assert headers["HTTP-Referer"] == "https://example.com"
    assert headers["X-Title"] == "abstractframework-test"


def test_openrouter_transcription_uses_json_base64_payload(monkeypatch):
    monkeypatch.setattr(OpenRouterProvider, "_validate_model", lambda self: None)

    provider = OpenRouterProvider(model="openai/whisper-large-v3", api_key="sk-test-key")
    captured = {}

    class _Response:
        status_code = 200
        content = b'{"text":"hello"}'
        headers = {"content-type": "application/json"}

        def json(self):
            return {"text": "hello"}

    def fake_post(url, *, json=None, headers=None, **kwargs):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["kwargs"] = kwargs
        return _Response()

    monkeypatch.setattr(provider.client, "post", fake_post)

    content, content_type = provider.transcribe_audio(
        b"abc",
        filename="speech.mp3",
        content_type="audio/mpeg",
        language="en",
        temperature=0,
    )

    assert content == b'{"text":"hello"}'
    assert content_type == "application/json"
    assert captured["url"] == "https://openrouter.ai/api/v1/audio/transcriptions"
    assert captured["headers"]["Authorization"] == "Bearer sk-test-key"
    assert captured["json"] == {
        "model": "openai/whisper-large-v3",
        "input_audio": {"data": "YWJj", "format": "mp3"},
        "language": "en",
        "temperature": 0,
    }
