from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse
from abstractcore.endpoint.app import create_app


class _FakeProvider:
    def __init__(self):
        self.model = "fake-model"
        self.calls: List[Dict[str, Any]] = []

    def generate(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        **kwargs,
    ):
        _ = (prompt, messages, system_prompt, tools, stream)
        self.calls.append(dict(kwargs))
        return GenerateResponse(
            content="hi",
            model=self.model,
            finish_reason="stop",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )


def test_abstractendpoint_constructs_provider_once_and_forwards_prompt_cache_key():
    calls = {"n": 0}
    provider_holder: Dict[str, Any] = {}

    def factory():
        calls["n"] += 1
        provider_holder["provider"] = _FakeProvider()
        return provider_holder["provider"]

    app = create_app(provider_factory=factory)
    assert calls["n"] == 1

    client = TestClient(app)

    payload = {
        "model": "fake-model",
        "messages": [{"role": "system", "content": "You are helpful"}, {"role": "user", "content": "Hi"}],
        "prompt_cache_key": "cache-abc",
    }

    r1 = client.post("/v1/chat/completions", json=payload)
    assert r1.status_code == 200
    body1 = r1.json()
    assert body1["object"] == "chat.completion"
    assert body1["model"] == "fake-model"
    assert body1["choices"][0]["message"]["content"] == "hi"
    assert body1["usage"]["total_tokens"] == 2

    r2 = client.post("/v1/chat/completions", json=payload)
    assert r2.status_code == 200
    assert calls["n"] == 1

    provider = provider_holder["provider"]
    assert provider.calls[-1].get("prompt_cache_key") == "cache-abc"

