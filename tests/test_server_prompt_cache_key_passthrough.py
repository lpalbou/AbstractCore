from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse


class _StubLLM:
    def __init__(self) -> None:
        self.last_kwargs: Dict[str, Any] = {}

    def generate(self, **kwargs: Any) -> GenerateResponse:
        self.last_kwargs = dict(kwargs)
        return GenerateResponse(
            content="ok",
            model="stub",
            finish_reason="stop",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    def unload_model(self, model_name: str) -> None:
        _ = model_name


def test_server_forwards_prompt_cache_key(monkeypatch) -> None:
    import importlib

    server_app = importlib.import_module("abstractcore.server.app")

    created: List[_StubLLM] = []

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        _ = (args, kwargs)
        llm = _StubLLM()
        created.append(llm)
        return llm

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-5-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "prompt_cache_key": "cache-abc",
        },
    )
    assert resp.status_code == 200
    assert created, "expected server to create an LLM instance"
    assert created[-1].last_kwargs.get("prompt_cache_key") == "cache-abc"

