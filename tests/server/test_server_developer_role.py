from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider


class _CaptureProvider(BaseProvider):
    def __init__(self, model: str = "stub") -> None:
        super().__init__(model=model, enable_tracing=False)
        self.last_messages: Optional[List[Dict[str, Any]]] = None
        self.last_system_prompt: Optional[str] = None

    def get_capabilities(self) -> List[str]:
        return ["chat"]

    def list_available_models(self, **kwargs: Any) -> List[str]:
        _ = kwargs
        return [self.model]

    def unload_model(self, model_name: str) -> None:
        _ = model_name

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> GenerateResponse:
        _ = (prompt, tools, media, stream, kwargs)
        self.last_messages = messages
        self.last_system_prompt = system_prompt
        return GenerateResponse(
            content="ok",
            model=self.model,
            finish_reason="stop",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )


def test_chat_completions_accepts_developer_role_and_normalizes_for_provider(monkeypatch) -> None:
    import importlib

    for name in (
        "ABSTRACTCORE_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "PORTKEY_API_KEY",
        "VLLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    server_app = importlib.import_module("abstractcore.server.app")
    created: List[_CaptureProvider] = []

    def fake_create_llm(provider: str, model: str, **kwargs: Any) -> _CaptureProvider:
        _ = (provider, kwargs)
        llm = _CaptureProvider(model=model)
        created.append(llm)
        return llm

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    resp = TestClient(server_app.app).post(
        "/v1/chat/completions",
        json={
            "model": "stub/test-model",
            "messages": [
                {"role": "developer", "content": "Always answer with ok."},
                {"role": "user", "content": "hello"},
            ],
        },
    )

    assert resp.status_code == 200, resp.text
    assert created
    assert created[0].last_system_prompt == "Always answer with ok."
    assert created[0].last_messages == [{"role": "user", "content": "hello"}]
