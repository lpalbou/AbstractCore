from __future__ import annotations

from typing import Any, Dict, List

from abstractcore.core.types import GenerateResponse
from abstractcore.providers import openai_provider as openai_provider_module
from abstractcore.providers.openai_provider import OpenAIProvider


class _DummyChatCompletions:
    def __init__(self, capture: List[Dict[str, Any]]):
        self._capture = capture

    def create(self, **kwargs):
        self._capture.append(dict(kwargs))
        return object()


class _DummyChat:
    def __init__(self, capture: List[Dict[str, Any]]):
        self.completions = _DummyChatCompletions(capture)


class _DummyOpenAIClient:
    def __init__(self, capture: List[Dict[str, Any]], **kwargs):
        _ = kwargs
        self.chat = _DummyChat(capture)


def test_openai_prompt_cache_key_is_forwarded(monkeypatch):
    capture: List[Dict[str, Any]] = []

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(OpenAIProvider, "_validate_model_exists", lambda self: None)
    monkeypatch.setattr(
        openai_provider_module.openai,
        "OpenAI",
        lambda **kwargs: _DummyOpenAIClient(capture, **kwargs),
    )

    def _format_response(self, _response):
        return GenerateResponse(content="ok", model=self.model, finish_reason="stop")

    monkeypatch.setattr(OpenAIProvider, "_format_response", _format_response)

    llm = OpenAIProvider(model="gpt-5-mini")
    assert llm.prompt_cache_set("cache-123") is True

    resp = llm.generate("hello")
    assert isinstance(resp, GenerateResponse)
    assert capture, "expected OpenAI SDK call to be captured"
    assert capture[-1].get("prompt_cache_key") == "cache-123"


def test_openai_prompt_cache_key_explicit_overrides_default(monkeypatch):
    capture: List[Dict[str, Any]] = []

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(OpenAIProvider, "_validate_model_exists", lambda self: None)
    monkeypatch.setattr(
        openai_provider_module.openai,
        "OpenAI",
        lambda **kwargs: _DummyOpenAIClient(capture, **kwargs),
    )

    monkeypatch.setattr(
        OpenAIProvider,
        "_format_response",
        lambda self, _response: GenerateResponse(content="ok", model=self.model, finish_reason="stop"),
    )

    llm = OpenAIProvider(model="gpt-5-mini")
    assert llm.prompt_cache_set("cache-123") is True

    llm.generate("hello", prompt_cache_key="cache-xyz")
    assert capture[-1].get("prompt_cache_key") == "cache-xyz"

