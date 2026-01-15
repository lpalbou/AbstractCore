from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.e2e


class _StubGenerateResponse:
    def __init__(self, *, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.content = content
        self.tool_calls = None
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.metadata = metadata or {}


class _StubLLM:
    def __init__(self) -> None:
        self.unload_model_calls: List[str] = []

    def unload_model(self, model_name: str) -> None:
        self.unload_model_calls.append(str(model_name))

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        return _StubGenerateResponse(content="stub-response")


def test_e2e_asgi_chat_completions_unload_model_after(monkeypatch: pytest.MonkeyPatch) -> None:
    if os.getenv("ABSTRACTCORE_E2E_UNLOAD_MODEL") != "1":
        pytest.skip("Set ABSTRACTCORE_E2E_UNLOAD_MODEL=1 to run this test")

    import importlib

    server_app = importlib.import_module("abstractcore.server.app")

    created: List[_StubLLM] = []

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        llm = _StubLLM()
        created.append(llm)
        return llm

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    r = client.get("/health")
    assert r.status_code == 200

    r = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "unload_after": True,
        },
    )
    assert r.status_code == 200
    assert created and created[0].unload_model_calls == ["gpt-4o-mini"]
