from __future__ import annotations

import types


def test_openai_model_discovery_uses_base_url(monkeypatch) -> None:
    from abstractcore.providers.openai_provider import OpenAIProvider

    captured: dict[str, object] = {}

    class FakeModels:
        def list(self):
            item = types.SimpleNamespace(id="gpt-5-test")
            return types.SimpleNamespace(data=[item])

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.models = FakeModels()

    monkeypatch.setitem(__import__("sys").modules, "openai", types.SimpleNamespace(OpenAI=FakeOpenAI))

    models = OpenAIProvider.list_available_models(api_key="openai-key", base_url="https://openai-proxy.example/v1")

    assert models == ["gpt-5-test"]
    assert captured["api_key"] == "openai-key"
    assert captured["base_url"] == "https://openai-proxy.example/v1"


def test_anthropic_model_discovery_uses_base_url(monkeypatch) -> None:
    from abstractcore.providers.anthropic_provider import AnthropicProvider

    captured: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"data": [{"id": "claude-haiku-test"}]}

    def fake_get(url, *, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    fake_httpx = types.SimpleNamespace(get=fake_get)
    monkeypatch.setitem(__import__("sys").modules, "httpx", fake_httpx)

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.api_key = "anthropic-key"
    provider.base_url = "https://anthropic-proxy.example/v1"
    provider.logger = types.SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None)

    models = provider.list_available_models()

    assert models == ["claude-haiku-test"]
    assert captured["url"] == "https://anthropic-proxy.example/v1/models"
    assert captured["headers"]["x-api-key"] == "anthropic-key"
