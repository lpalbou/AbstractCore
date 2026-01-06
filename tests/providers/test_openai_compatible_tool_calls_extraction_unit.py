import pytest

from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider
from abstractcore.providers.vllm_provider import VLLMProvider


class _StubHttpResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.mark.parametrize(
    ("provider_cls", "base_url"),
    [
        (OpenAICompatibleProvider, "http://localhost:8080/v1"),
        (VLLMProvider, "http://localhost:8000/v1"),
    ],
)
def test_openai_compatible_providers_extract_tool_calls_from_response_json(provider_cls, base_url, monkeypatch):
    # Avoid network during provider init (model discovery).
    monkeypatch.setattr(provider_cls, "_validate_model", lambda self: None)

    provider = provider_cls(model="nvidia/nemotron-3-nano", base_url=base_url)

    tools = [
        {
            "name": "write_file",
            "description": "Write a file",
            "parameters": {
                "file_path": {"type": "string"},
                "content": {"type": "string"},
            },
        }
    ]

    response_json = {
        "id": "chatcmpl_test",
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "write_file",
                                "arguments": '{"file_path":"x.txt","content":"hello"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    def _fake_post(*args, **kwargs):
        return _StubHttpResponse(response_json)

    monkeypatch.setattr(provider.client, "post", _fake_post)

    resp = provider.generate("hi", tools=tools, temperature=0)

    assert resp is not None
    assert resp.tool_calls == [
        {
            "name": "write_file",
            "arguments": {"file_path": "x.txt", "content": "hello"},
            "call_id": "call_123",
        }
    ]

