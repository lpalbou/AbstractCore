"""Ollama `num_ctx` forwarding (abstractagent find, 2026-07-13).

`llm_kwargs={"num_ctx": ...}` landed in interface config and was never read —
the options builder sent only temperature/num_predict/top_p/top_k/seed, so the
stack could not request a context window per call and Ollama silently
truncated long prompts to the model default. Pins: per-call kwarg wins,
constructor kwarg is honored, absence sends NO num_ctx (the model default is
never second-guessed), and invalid values raise loudly.
"""

import json

import pytest

from abstractcore.providers.ollama_provider import OllamaProvider


class _FakeResponse:
    status_code = 200

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "message": {"role": "assistant", "content": "ok"},
            "model": "test-model",
            "done": True,
            "prompt_eval_count": 1,
            "eval_count": 1,
        }


def _capture_payload(provider):
    captured = {}

    def fake_post(url, json=None, **kwargs):
        captured["url"] = url
        captured["payload"] = json
        return _FakeResponse(json)

    provider.client.post = fake_post
    return captured


def test_per_call_num_ctx_is_forwarded_into_options():
    provider = OllamaProvider(model="test-model")
    captured = _capture_payload(provider)
    provider._generate_internal("hello", num_ctx=32768)
    assert captured["payload"]["options"]["num_ctx"] == 32768


def test_constructor_num_ctx_is_honored():
    provider = OllamaProvider(model="test-model", num_ctx=16384)
    captured = _capture_payload(provider)
    provider._generate_internal("hello")
    assert captured["payload"]["options"]["num_ctx"] == 16384


def test_per_call_overrides_constructor():
    provider = OllamaProvider(model="test-model", num_ctx=16384)
    captured = _capture_payload(provider)
    provider._generate_internal("hello", num_ctx=8192)
    assert captured["payload"]["options"]["num_ctx"] == 8192


def test_absent_num_ctx_sends_nothing():
    """Never second-guess the model's default context window."""
    provider = OllamaProvider(model="test-model")
    captured = _capture_payload(provider)
    provider._generate_internal("hello")
    assert "num_ctx" not in captured["payload"]["options"]


def test_invalid_num_ctx_raises_loudly():
    provider = OllamaProvider(model="test-model")
    _capture_payload(provider)
    with pytest.raises(ValueError):
        provider._generate_internal("hello", num_ctx=0)
    with pytest.raises((ValueError, TypeError)):
        provider._generate_internal("hello", num_ctx="not-a-number")
