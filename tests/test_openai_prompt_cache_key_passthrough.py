from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

from pydantic import BaseModel

from abstractcore.core.types import GenerateResponse
from abstractcore.providers import openai_provider as openai_provider_module
from abstractcore.providers.openai_provider import OpenAIProvider


def _install_fake_openai(monkeypatch, capture: List[Dict[str, Any]]) -> None:
    class _DummyChatCompletions:
        def __init__(self, _capture: List[Dict[str, Any]]):
            self._capture = _capture

        def create(self, **kwargs):
            self._capture.append(dict(kwargs))
            return object()

    class _DummyChat:
        def __init__(self, _capture: List[Dict[str, Any]]):
            self.completions = _DummyChatCompletions(_capture)

    class _DummyOpenAIClient:
        def __init__(self, _capture: List[Dict[str, Any]], **_kwargs):
            self.chat = _DummyChat(_capture)
            self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[]))

    fake_openai = SimpleNamespace(OpenAI=lambda **kwargs: _DummyOpenAIClient(capture, **kwargs))
    monkeypatch.setattr(openai_provider_module, "OPENAI_AVAILABLE", True, raising=False)
    monkeypatch.setattr(openai_provider_module, "openai", fake_openai, raising=False)


def test_openai_prompt_cache_key_is_forwarded(monkeypatch):
    capture: List[Dict[str, Any]] = []

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(OpenAIProvider, "_validate_model_exists", lambda self: None)
    _install_fake_openai(monkeypatch, capture)

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
    _install_fake_openai(monkeypatch, capture)

    monkeypatch.setattr(
        OpenAIProvider,
        "_format_response",
        lambda self, _response: GenerateResponse(content="ok", model=self.model, finish_reason="stop"),
    )

    llm = OpenAIProvider(model="gpt-5-mini")
    assert llm.prompt_cache_set("cache-123") is True

    llm.generate("hello", prompt_cache_key="cache-xyz")
    assert capture[-1].get("prompt_cache_key") == "cache-xyz"


def test_openai_prompt_cache_retention_is_forwarded(monkeypatch):
    capture: List[Dict[str, Any]] = []

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(OpenAIProvider, "_validate_model_exists", lambda self: None)
    _install_fake_openai(monkeypatch, capture)
    monkeypatch.setattr(
        OpenAIProvider,
        "_format_response",
        lambda self, _response: GenerateResponse(content="ok", model=self.model, finish_reason="stop"),
    )

    llm = OpenAIProvider(model="gpt-5-mini")
    llm.generate("hello", prompt_cache_retention="24h")
    assert capture[-1].get("prompt_cache_retention") == "24h"


def test_openai_native_structured_output_sets_response_format(monkeypatch):
    capture: List[Dict[str, Any]] = []

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(OpenAIProvider, "_validate_model_exists", lambda self: None)
    _install_fake_openai(monkeypatch, capture)
    monkeypatch.setattr(
        OpenAIProvider,
        "_format_response",
        lambda self, _response: GenerateResponse(content='{"a": 1}', model=self.model, finish_reason="stop"),
    )

    class _Result(BaseModel):
        a: int

    llm = OpenAIProvider(model="gpt-5.4")
    out = llm.generate("hello", response_model=_Result)
    assert isinstance(out, _Result)
    assert capture, "expected OpenAI SDK call to be captured"
    assert isinstance(capture[-1].get("response_format"), dict)
