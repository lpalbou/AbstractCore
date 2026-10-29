"""A key check on an OpenAI-compatible endpoint whose GET /models is PUBLIC (offline: HTTP stubbed).

OVH AI Endpoints, measured 2026-10-29: GET /v1/models answers 200 with no key and with a wrong one;
POST /v1/chat/completions with a wrong key answers 403 "authentication failed" BEFORE looking at the
model, and with no key "model_not_found" (404). So listing alone said "Connected · N models" to a wrong
key; the check now asks for a model that does not exist (nothing is generated) and trusts a 401/403.
"""

from __future__ import annotations

import httpx
import pytest

from abstractcore.exceptions import AuthenticationError
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider
from abstractcore.providers.registry import get_available_models_for_provider

BASE = "https://oai.endpoints.example-ovh.test/v1"
GOOD = "ovh-" + "g" * 24
BAD = "ovh-" + "b" * 24
CATALOGUE = {"object": "list", "data": [{"id": "Qwen3.5-9B"}, {"id": "gpt-oss-20b"}]}


def _resp(method: str, url: str, status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, json=body, request=httpx.Request(method, url))


@pytest.fixture()
def endpoint(monkeypatch):
    """public_models: GET /models answers without a key (OVH); else it needs the key (Mistral-like)."""
    state = {"public_models": True, "calls": []}

    def fake_get(self, url, headers=None, **kwargs):
        url = str(url)
        auth = dict(headers or {}).get("Authorization", "")
        state["calls"].append(("GET", url, auth))
        if not url.endswith("/models"):
            raise AssertionError(url)
        if state["public_models"] or auth == f"Bearer {GOOD}":
            return _resp("GET", url, 200, CATALOGUE)
        return _resp("GET", url, 401, {"error": {"message": "Unauthorized"}})

    def fake_post(self, url, headers=None, json=None, **kwargs):
        url = str(url)
        auth = dict(headers or {}).get("Authorization", "")
        state["calls"].append(("POST", url, auth, (json or {}).get("model")))
        if auth and auth != f"Bearer {GOOD}":
            return _resp("POST", url, 403, {"message": "Forbidden: authentication failed."})
        return _resp("POST", url, 404, {"error": {"code": "model_not_found", "message": "The model does not exist."}})

    monkeypatch.setattr(httpx.Client, "get", fake_get)
    monkeypatch.setattr(httpx.Client, "post", fake_post)
    return state


def test_a_wrong_key_is_refused_although_models_are_public(endpoint) -> None:
    with pytest.raises(AuthenticationError) as exc:
        get_available_models_for_provider("openai-compatible", base_url=BASE, api_key=BAD, raise_on_error=True)
    assert "403" in str(exc.value) and BAD not in str(exc.value)
    probe = [c for c in endpoint["calls"] if c[0] == "POST"]
    assert probe and probe[0][1] == f"{BASE}/chat/completions" and probe[0][3] == OpenAICompatibleProvider.KEY_PROBE_MODEL


def test_a_valid_key_lists_and_runs_nothing(endpoint) -> None:
    assert get_available_models_for_provider("openai-compatible", base_url=BASE, api_key=GOOD, raise_on_error=True) == ["Qwen3.5-9B", "gpt-oss-20b"]
    # Only the nonexistent model was asked for.
    assert all(c[3] == OpenAICompatibleProvider.KEY_PROBE_MODEL for c in endpoint["calls"] if c[0] == "POST")


def test_no_probe_when_models_needs_the_key(endpoint) -> None:
    endpoint["public_models"] = False
    assert get_available_models_for_provider("openai-compatible", base_url=BASE, api_key=GOOD, raise_on_error=True)
    assert not any(c[0] == "POST" for c in endpoint["calls"])


def test_no_probe_outside_a_key_check(endpoint) -> None:
    """Plain listing (no raise_on_error) keeps its single request."""
    assert get_available_models_for_provider("openai-compatible", base_url=BASE, api_key=BAD) == ["Qwen3.5-9B", "gpt-oss-20b"]
    assert not any(c[0] == "POST" for c in endpoint["calls"])
