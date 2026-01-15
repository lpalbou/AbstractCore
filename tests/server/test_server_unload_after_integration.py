from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import pytest
from fastapi.testclient import TestClient


pytestmark = pytest.mark.integration


class _StubGenerateResponse:
    def __init__(self, *, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.content = content
        self.tool_calls = None
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.metadata = metadata or {}


class _StubLLM:
    def __init__(self, *, stream_chunks: Optional[List[str]] = None) -> None:
        self._stream_chunks = stream_chunks or []
        self.unload_model_calls: List[str] = []

    def unload_model(self, model_name: str) -> None:
        self.unload_model_calls.append(str(model_name))

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        if kwargs.get("stream"):
            return self._stream()
        return _StubGenerateResponse(content="stub-response")

    def _stream(self) -> Iterator[_StubGenerateResponse]:
        for chunk in self._stream_chunks:
            yield _StubGenerateResponse(content=chunk)


def test_chat_completions_unload_model_after_default_false_does_not_unload(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    server_app = importlib.import_module("abstractcore.server.app")

    created: List[_StubLLM] = []

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        llm = _StubLLM()
        created.append(llm)
        return llm

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
        },
    )
    assert resp.status_code == 200
    assert created and created[0].unload_model_calls == []


def test_chat_completions_unload_model_after_true_unloads(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    server_app = importlib.import_module("abstractcore.server.app")

    created: List[_StubLLM] = []

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        llm = _StubLLM()
        created.append(llm)
        return llm

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Hello"}],
            "unload_after": True,
        },
    )
    assert resp.status_code == 200
    assert created and created[0].unload_model_calls == ["gpt-4o-mini"]


def test_chat_completions_streaming_unload_model_after_runs_after_stream(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    server_app = importlib.import_module("abstractcore.server.app")

    llm = _StubLLM(stream_chunks=["Hello", " world"])
    gen_kwargs: Dict[str, Any] = {
        "prompt": "",
        "messages": [],
        "temperature": 0.7,
        "max_tokens": 16,
        "stream": True,
        "tools": None,
        "tool_choice": None,
        "execute_tools": False,
    }
    syntax_rewriter = server_app.create_passthrough_rewriter()

    iterator = server_app.generate_streaming_response(
        llm,
        gen_kwargs,
        "openai",
        "gpt-4o-mini",
        syntax_rewriter,
        "test-request-id",
        [],
        unload_after=True,
    )

    next(iterator)
    assert llm.unload_model_calls == []
    for _ in iterator:
        pass
    assert llm.unload_model_calls == ["gpt-4o-mini"]


def test_unload_model_after_ollama_forbidden_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    server_app = importlib.import_module("abstractcore.server.app")

    monkeypatch.delenv("ABSTRACTCORE_ALLOW_UNSAFE_UNLOAD_AFTER", raising=False)

    def fail_create_llm(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("create_llm should not be called when unload_after is forbidden")

    monkeypatch.setattr(server_app, "create_llm", fail_create_llm)

    client = TestClient(server_app.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "ollama/qwen3:4b-instruct",
            "messages": [{"role": "user", "content": "Hello"}],
            "unload_after": True,
        },
    )
    assert resp.status_code == 403


def test_unload_model_after_ollama_allowed_with_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    server_app = importlib.import_module("abstractcore.server.app")

    monkeypatch.setenv("ABSTRACTCORE_ALLOW_UNSAFE_UNLOAD_AFTER", "1")

    created: List[_StubLLM] = []

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        llm = _StubLLM()
        created.append(llm)
        return llm

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "ollama/qwen3:4b-instruct",
            "messages": [{"role": "user", "content": "Hello"}],
            "unload_after": True,
        },
    )
    assert resp.status_code == 200
    assert created and created[0].unload_model_calls == ["qwen3:4b-instruct"]


def test_responses_api_unload_model_after_true_unloads(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    server_app = importlib.import_module("abstractcore.server.app")

    created: List[_StubLLM] = []

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        llm = _StubLLM()
        created.append(llm)
        return llm

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "openai/gpt-4o-mini",
            "input": [{"role": "user", "content": [{"type": "input_text", "text": "Hello"}]}],
            "unload_after": "true",
        },
    )
    assert resp.status_code == 200
    assert created and created[0].unload_model_calls == ["gpt-4o-mini"]
