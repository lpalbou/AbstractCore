import pytest

from abstractcore.architectures.detection import get_model_capabilities
from abstractcore.providers.lmstudio_provider import LMStudioProvider
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
        (LMStudioProvider, "http://127.0.0.1:1234/v1"),
        (OpenAICompatibleProvider, "http://127.0.0.1:1234/v1"),
        (VLLMProvider, "http://127.0.0.1:8000/v1"),
    ],
)
def test_gpt_oss_20b_native_tools_send_tools_and_extract_tool_calls(provider_cls, base_url, monkeypatch):
    """
    `openai/gpt-oss-20b` is configured as a native-tools model.

    This unit test verifies two critical integration points for OpenAI-compatible servers:
    1) Request: provider includes `payload["tools"]` / `payload["tool_choice"]` when tools are passed.
    2) Response: provider+BaseProvider normalize OpenAI `choice.message.tool_calls` into canonical
       `GenerateResponse.tool_calls` with JSON `arguments` parsed into a dict.
    """
    caps = get_model_capabilities("openai/gpt-oss-20b")
    assert caps.get("tool_support") == "native"

    # Avoid any dependency on a running server during provider init.
    monkeypatch.setattr(provider_cls, "_validate_model", lambda self: None, raising=False)

    provider = provider_cls(model="openai/gpt-oss-20b", base_url=base_url)

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

    captured = {}

    def _fake_post(*args, **kwargs):
        captured["payload"] = kwargs.get("json")
        return _StubHttpResponse(response_json)

    monkeypatch.setattr(provider.client, "post", _fake_post)

    resp = provider.generate("hi", tools=tools, temperature=0)

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["model"] == "openai/gpt-oss-20b"
    assert "tools" in payload
    assert payload["tool_choice"] == "auto"
    assert payload["tools"][0]["function"]["name"] == "write_file"

    assert resp.tool_calls == [
        {
            "name": "write_file",
            "arguments": {"file_path": "x.txt", "content": "hello"},
            "call_id": "call_123",
        }
    ]

