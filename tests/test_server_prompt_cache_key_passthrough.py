from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi.testclient import TestClient

from abstractcore.core.types import GenerateResponse


class _StubLLM:
    def __init__(self) -> None:
        self.last_kwargs: Dict[str, Any] = {}
        self.unload_calls: List[str] = []

    def generate(self, **kwargs: Any) -> GenerateResponse:
        self.last_kwargs = dict(kwargs)
        return GenerateResponse(
            content="ok",
            model="stub",
            finish_reason="stop",
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    def unload_model(self, model_name: str) -> None:
        self.unload_calls.append(model_name)


def test_server_forwards_prompt_cache_key(monkeypatch) -> None:
    import importlib

    server_app = importlib.import_module("abstractcore.server.app")
    for name in (
        "ABSTRACTCORE_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "PORTKEY_API_KEY",
        "OPENAI_BASE_URL",
        "VLLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

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
            "prompt_cache_retention": "24h",
        },
    )
    assert resp.status_code == 200
    assert created, "expected server to create an LLM instance"
    assert created[-1].last_kwargs.get("prompt_cache_key") == "cache-abc"
    assert created[-1].last_kwargs.get("prompt_cache_retention") == "24h"


def test_server_responses_openai_format_forwards_shared_chat_controls(monkeypatch) -> None:
    import importlib

    server_app = importlib.import_module("abstractcore.server.app")
    for name in (
        "ABSTRACTCORE_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "PORTKEY_API_KEY",
        "OPENAI_BASE_URL",
        "VLLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    created: List[_StubLLM] = []
    provider_kwargs_seen: List[Dict[str, Any]] = []

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        _ = args
        provider_kwargs_seen.append(dict(kwargs))
        llm = _StubLLM()
        created.append(llm)
        return llm

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "openai/gpt-5-mini",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ],
            "tools": [
                {"type": "web_search", "external_web_access": True},
            ],
            "tool_choice": "auto",
            "stop": ["DONE"],
            "seed": 7,
            "frequency_penalty": 0.25,
            "presence_penalty": 0.5,
            "base_url": "http://127.0.0.1:1234/v1",
            "agent_format": "codex",
            "thinking": "high",
            "prompt_cache_key": "cache-openai-format",
            "prompt_cache_retention": "24h",
            "timeout_s": 12.5,
            "unload_after": True,
        },
    )
    assert resp.status_code == 200
    assert created, "expected server to create an LLM instance"
    assert provider_kwargs_seen[-1]["base_url"] == "http://127.0.0.1:1234/v1"
    assert provider_kwargs_seen[-1]["timeout"] == 12.5
    assert created[-1].last_kwargs.get("thinking") == "high"
    assert created[-1].last_kwargs.get("stop") == ["DONE"]
    assert created[-1].last_kwargs.get("seed") == 7
    assert created[-1].last_kwargs.get("frequency_penalty") == 0.25
    assert created[-1].last_kwargs.get("presence_penalty") == 0.5
    assert created[-1].last_kwargs.get("prompt_cache_key") == "cache-openai-format"
    assert created[-1].last_kwargs.get("prompt_cache_retention") == "24h"
    assert created[-1].last_kwargs.get("execute_tools") is False
    assert created[-1].last_kwargs.get("tool_choice") == "auto"
    assert created[-1].last_kwargs.get("tools") == [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "recency": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    assert created[-1].unload_calls == ["gpt-5-mini"]


def test_convert_openai_responses_preserves_shared_chat_extensions() -> None:
    from abstractcore.server.app import OpenAIResponsesRequest, convert_openai_responses_to_chat_completion

    openai_request = OpenAIResponsesRequest(
        model="openai/gpt-5-mini",
        input=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": "Hello"}],
            }
        ],
        stop=["DONE"],
        seed=7,
        frequency_penalty=0.25,
        presence_penalty=0.5,
        base_url="http://127.0.0.1:1234/v1",
        agent_format="codex",
        thinking="high",
        prompt_cache_key="cache-openai-format",
        prompt_cache_retention="24h",
        timeout_s=12.5,
        unload_after=True,
    )

    chat_request = convert_openai_responses_to_chat_completion(openai_request)

    assert chat_request.stop == ["DONE"]
    assert chat_request.seed == 7
    assert chat_request.frequency_penalty == 0.25
    assert chat_request.presence_penalty == 0.5
    assert chat_request.base_url == "http://127.0.0.1:1234/v1"
    assert chat_request.agent_format == "codex"
    assert chat_request.thinking == "high"
    assert chat_request.prompt_cache_key == "cache-openai-format"
    assert chat_request.prompt_cache_retention == "24h"
    assert chat_request.timeout_s == 12.5
    assert chat_request.unload_after is True


def test_convert_openai_responses_normalizes_responses_tools_to_chat_tools() -> None:
    from abstractcore.server.app import OpenAIResponsesRequest, convert_openai_responses_to_chat_completion

    openai_request = OpenAIResponsesRequest(
        model="openai/gpt-5-mini",
        input="Hello",
        tools=[{"type": "web_search", "external_web_access": False}],
        tool_choice={"type": "web_search"},
    )

    chat_request = convert_openai_responses_to_chat_completion(openai_request)

    assert chat_request.tools == [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "recency": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        }
    ]
    assert chat_request.tool_choice == {"type": "function", "function": {"name": "web_search"}}


def test_server_responses_openai_format_returns_openai_responses_object(monkeypatch) -> None:
    import importlib

    from openai.types import responses as openai_responses

    server_app = importlib.import_module("abstractcore.server.app")
    for name in (
        "ABSTRACTCORE_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "PORTKEY_API_KEY",
        "OPENAI_BASE_URL",
        "VLLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    created: List[_StubLLM] = []

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StubLLM:
        _ = (args, kwargs)
        llm = _StubLLM()
        created.append(llm)
        return llm

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "openai/gpt-5-mini",
            "input": [{"type": "message", "role": "user", "content": "What is 2+2?"}],
            "stream": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    parsed = openai_responses.Response.model_validate(data)
    assert parsed.object == "response"
    assert parsed.output, "expected at least one output item"
    assert parsed.output[0].type == "message"
    assert parsed.output[0].content, "expected message content"
    assert parsed.output[0].content[0].type == "output_text"
    assert parsed.output[0].content[0].text == "ok"


def test_server_responses_openai_format_stream_emits_openai_responses_events(monkeypatch) -> None:
    import importlib
    import json

    from pydantic import TypeAdapter
    from openai.types import responses as openai_responses

    from abstractcore.core.types import GenerateResponse

    class _StreamingStubLLM(_StubLLM):
        def generate(self, **kwargs: Any):  # type: ignore[override]
            self.last_kwargs = dict(kwargs)
            if kwargs.get("stream"):
                def _gen():
                    yield GenerateResponse(content="o", model="stub", finish_reason=None, usage=None)
                    yield GenerateResponse(content="k", model="stub", finish_reason=None, usage=None)
                return _gen()
            return super().generate(**kwargs)

    server_app = importlib.import_module("abstractcore.server.app")
    for name in (
        "ABSTRACTCORE_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "OPENROUTER_API_KEY",
        "PORTKEY_API_KEY",
        "OPENAI_BASE_URL",
        "VLLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)

    def fake_create_llm(*args: Any, **kwargs: Any) -> _StreamingStubLLM:
        _ = (args, kwargs)
        return _StreamingStubLLM()

    monkeypatch.setattr(server_app, "create_llm", fake_create_llm)

    client = TestClient(server_app.app)
    resp = client.post(
        "/v1/responses",
        json={
            "model": "openai/gpt-5-mini",
            "input": "Say ok.",
            "stream": True,
        },
    )
    assert resp.status_code == 200

    adapter = TypeAdapter(openai_responses.ResponseStreamEvent)
    seen_completed = False
    seen_created = False
    done = False
    last_completed_response = None

    for line in resp.iter_lines():
        if not line:
            continue
        assert line.startswith("data: ")
        payload = line[len("data: "):]
        if payload.startswith("[DONE]"):
            done = True
            break
        event = json.loads(payload)
        parsed_event = adapter.validate_python(event)
        if getattr(parsed_event, "type", None) == "response.created":
            seen_created = True
        if getattr(parsed_event, "type", None) == "response.completed":
            seen_completed = True
            last_completed_response = parsed_event.response

    assert done is True
    assert seen_created is True
    assert seen_completed is True
    assert last_completed_response is not None
    assert last_completed_response.object == "response"
    assert last_completed_response.output
    assert last_completed_response.output[0].type == "message"
