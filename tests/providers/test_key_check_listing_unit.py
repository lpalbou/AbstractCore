"""A key check is a listing with `raise_on_error=True` (the gateway's "Check the key").

It must be honest about the KEY, offline (every HTTP call is stubbed):
- OpenRouter's GET /models is public (200 for any key), so a wrong key must be caught by
  GET /key first, never answered with the whole catalogue;
- Anthropic answers a wrong key with 401: the check raises AuthenticationError with
  Anthropic's reason instead of returning an empty list with no reason.
"""

from __future__ import annotations

import httpx
import pytest

from abstractcore.exceptions import AuthenticationError
from abstractcore.providers.anthropic_provider import AnthropicProvider
from abstractcore.providers.openrouter_provider import OpenRouterProvider
from abstractcore.providers.registry import get_available_models_for_provider

GOOD = "sk-or-v1-" + "a" * 16
BAD = "sk-or-v1-" + "b" * 16
CATALOGUE = {"data": [{"id": "openai/gpt-4o-mini"}, {"id": "anthropic/claude-haiku-4-5"}]}


def _response(url: str, status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request("GET", url))


@pytest.fixture()
def openrouter_http(monkeypatch):
    """OpenRouter as measured on 2026-10-29: /models answers every key, /key only the real one."""
    calls: list[str] = []
    key_status = {"value": None}

    def fake_get(self, url, headers=None, **kwargs):
        url = str(url)
        calls.append(url)
        auth = dict(headers or {}).get("Authorization", "")
        if url.endswith("/key"):
            if key_status["value"] is not None:
                return _response(url, key_status["value"], {"error": {"message": "Not Found"}})
            if auth == f"Bearer {GOOD}":
                return _response(url, 200, {"data": {"label": "sk-or-v1-aaa...aaa", "usage": 0}})
            return _response(url, 401, {"error": {"message": "User not found.", "code": 401}})
        if url.endswith("/models"):
            return _response(url, 200, CATALOGUE)
        raise AssertionError(f"unexpected GET {url}")

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(OpenRouterProvider, "_get_api_key_from_config", lambda self: None)
    return calls, key_status


def test_openrouter_wrong_key_is_refused_although_models_are_public(openrouter_http) -> None:
    calls, _ = openrouter_http
    with pytest.raises(AuthenticationError) as exc:
        get_available_models_for_provider("openrouter", api_key=BAD, raise_on_error=True)
    assert "401" in str(exc.value) and "User not found" in str(exc.value)
    assert BAD not in str(exc.value)
    # Without raise_on_error: nothing listed for a refused key (never the public catalogue).
    assert get_available_models_for_provider("openrouter", api_key=BAD) == []


def test_openrouter_real_key_lists_the_catalogue(openrouter_http) -> None:
    calls, _ = openrouter_http
    models = get_available_models_for_provider("openrouter", api_key=GOOD, raise_on_error=True)
    assert models == ["anthropic/claude-haiku-4-5", "openai/gpt-4o-mini"]
    key_calls = [c for c in calls if c.endswith("/key")]
    assert key_calls and key_calls[0] == "https://openrouter.ai/api/v1/key"


def test_openrouter_proxy_without_a_key_route_still_lists(openrouter_http) -> None:
    _, key_status = openrouter_http
    key_status["value"] = 404
    provider = OpenRouterProvider(model="default", api_key=BAD, base_url="https://proxy.example/v1", validate_model=False)
    assert provider.list_available_models(raise_on_error=True) == ["anthropic/claude-haiku-4-5", "openai/gpt-4o-mini"]


def test_anthropic_wrong_key_raises_its_reason(monkeypatch) -> None:
    def fake_get(url, headers=None, timeout=None, **kwargs):
        return _response(str(url), 401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})

    monkeypatch.setattr(httpx, "get", fake_get)
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.api_key = "sk-ant-wrong"
    provider.base_url = None
    import logging

    provider.logger = logging.getLogger("test")
    with pytest.raises(AuthenticationError) as exc:
        provider.list_available_models(api_key="sk-ant-wrong", raise_on_error=True)
    assert str(exc.value) == "Anthropic API error (401): invalid x-api-key"
    assert getattr(exc.value, "status_code", None) == 401
    # The listing path without raise_on_error keeps its old answer.
    assert provider.list_available_models(api_key="sk-ant-wrong") == []


def test_anthropic_real_key_lists(monkeypatch) -> None:
    def fake_get(url, headers=None, timeout=None, **kwargs):
        assert headers["x-api-key"] == "sk-ant-good"
        return _response(str(url), 200, {"data": [{"id": "claude-haiku-4-5"}, {"id": "claude-sonnet-4-5"}]})

    monkeypatch.setattr(httpx, "get", fake_get)
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.api_key = None
    provider.base_url = None
    import logging

    provider.logger = logging.getLogger("test")
    assert provider.list_available_models(api_key="sk-ant-good", raise_on_error=True) == ["claude-sonnet-4-5", "claude-haiku-4-5"]
