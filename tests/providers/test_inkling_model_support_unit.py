"""Thinking Machines Inkling family (Inkling, Inkling Small, Inkling 2 Small, Inkling 2 Mini) support.

Covers the registry (every published repo name and common spellings resolve to the `inkling`
architecture and to the right model entry), native tool calling on the OpenAI-compatible
provider (the `tools` field is sent; vLLM/SGLang's `--tool-call-parser inkling` only extracts
calls for declared tools, so a tool list written into the system prompt would yield none), and
the thinking control: `reasoning_effort` levels, with thinking="off" mapped to the lowest level
because no Inkling effort value switches reasoning off.
"""

from __future__ import annotations

import pytest

from abstractcore.architectures import detect_architecture, get_model_capabilities
from abstractcore.core.retry import RetryConfig
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider
from abstractcore.tools.handler import UniversalToolHandler

# ------------------------------------------------------------------ registry

SPELLINGS = [
    ("thinkingmachines/Inkling-2-Mini", "26.4B"),
    ("Inkling-2-Mini", "26.4B"),
    ("inkling-2-mini", "26.4B"),
    ("inkling2-mini", "26.4B"),
    ("vllm/inkling-2-mini", "26.4B"),
    ("unsloth/Inkling-2-Mini-GGUF", "26.4B"),
    ("thinkingmachines/Inkling-2-Small", "594B"),
    ("thinkingmachines/Inkling-2-Small-MXFP8", "594B"),
    ("inkling-2-small", "594B"),
    ("thinkingmachines/Inkling", "975B"),
    ("thinkingmachines/Inkling-NVFP4", "975B"),
    ("inkling", "975B"),
    ("thinkingmachines/Inkling-Small", "276B"),
    ("thinkingmachines/Inkling-Small-NVFP4", "276B"),
    ("inkling-small", "276B"),
]


@pytest.mark.parametrize("name,total", SPELLINGS)
def test_every_spelling_resolves_to_its_inkling_entry(name: str, total: str) -> None:
    caps = get_model_capabilities(name)
    assert caps["architecture"] == "inkling"
    # The architecture patterns need the publisher or the "2": the bare first-generation names
    # (`inkling`, `inkling-small`) resolve through their aliases only.
    if "/" in name or "inkling-2" in name.lower() or "inkling2" in name.lower():
        assert detect_architecture(name) == "inkling"
    assert caps["total_parameters"] == total
    assert caps["tool_support"] == "native"
    assert caps["thinking_support"] is True
    assert caps["thinking_output_field"] == "reasoning"
    assert caps["max_tokens"] == 1048576
    assert UniversalToolHandler(name).supports_native is True


@pytest.mark.parametrize(
    "name",
    [
        "twinkling-7b",
        "sprinkling-llm",
        "tinkling",
        "wrinkling-coder-33b",
        "org/Twinkling-Star-GGUF",
        "twinkling-small",
        "acme/twinkling-nvfp4",
        "my-inkling-finetune",
    ],
)
def test_unrelated_names_containing_the_word_are_not_inkling(name: str) -> None:
    assert detect_architecture(name) == "generic"
    assert get_model_capabilities(name).get("total_parameters") is None
    assert UniversalToolHandler(name).supports_native is False


def test_an_unlisted_inkling_release_still_gets_native_tools() -> None:
    assert detect_architecture("thinkingmachines/Inkling-3-Large") == "inkling"
    assert UniversalToolHandler("thinkingmachines/Inkling-3-Large").supports_native is True


def test_inputs_follow_the_model_cards() -> None:
    assert get_model_capabilities("inkling-2-mini")["capability_routes"]["input"] == ["text", "image"]
    assert get_model_capabilities("inkling-2-small")["audio_support"] is False
    assert get_model_capabilities("inkling")["capability_routes"]["input"] == ["text", "image", "voice", "sound"]
    assert get_model_capabilities("inkling-small")["audio_support"] is True


def test_reasoning_levels_follow_each_chat_template() -> None:
    # Inkling 2 templates map xhigh; the Inkling (v1) templates do not.
    assert get_model_capabilities("inkling-2-mini")["reasoning_levels"] == ["minimal", "low", "medium", "high", "xhigh"]
    assert get_model_capabilities("inkling-2-small")["reasoning_levels"][-1] == "xhigh"
    assert get_model_capabilities("inkling")["reasoning_levels"] == ["minimal", "low", "medium", "high"]
    # "none" (effort 0.0) still reasons, so it is not declared as a level.
    assert "none" not in get_model_capabilities("inkling-2-mini")["reasoning_levels"]


# ------------------------------------------------------------------ provider wire format

_TOOL_CALL_BODY = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": None,
                "reasoning": "The user wants today's news; search the web.",
                "tool_calls": [
                    {
                        "id": "chatcmpl-tool-1",
                        "type": "function",
                        "function": {"name": "web_search", "arguments": '{"query": "top news today"}'},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 120, "completion_tokens": 40, "total_tokens": 160},
}

_TOOLS = [
    {
        "name": "web_search",
        "description": "Search the public internet.",
        "parameters": {"query": {"type": "string", "description": "what to search for"}},
        "required_args": ["query"],
    }
]


class _Resp:
    def __init__(self, body):
        self.status_code = 200
        self._body = dict(body)
        self.text = ""

    def json(self):
        return dict(self._body)

    def read(self):
        return b""


class _CapturingClient:
    def __init__(self, body):
        self._body = body
        self.payloads = []

    def post(self, url, json=None, headers=None):  # noqa: A002 - httpx signature
        self.payloads.append(json)
        return _Resp(self._body)


def _provider() -> tuple[OpenAICompatibleProvider, _CapturingClient]:
    provider = OpenAICompatibleProvider(
        model="inkling-2-mini",
        base_url="http://127.0.0.1:9/v1",  # unreachable on purpose
        api_key="x",
        validate_model=False,
        retry_config=RetryConfig(max_attempts=1, initial_delay=0.0, use_jitter=False),
    )
    client = _CapturingClient(_TOOL_CALL_BODY)
    provider.client = client
    return provider, client


def test_tools_are_sent_natively_and_the_call_is_parsed() -> None:
    provider, client = _provider()
    response = provider.generate("What is in the news today?", tools=_TOOLS)

    payload = client.payloads[-1]
    assert [t["function"]["name"] for t in payload["tools"]] == ["web_search"]
    assert payload["tool_choice"] == "auto"
    system = " ".join(m["content"] for m in payload["messages"] if m["role"] == "system" and isinstance(m["content"], str))
    assert "You have access to the following tools" not in system

    assert response.tool_calls == [
        {"name": "web_search", "arguments": {"query": "top news today"}, "call_id": "chatcmpl-tool-1"}
    ]


@pytest.mark.parametrize("thinking,effort", [("low", "low"), ("high", "high"), ("off", "minimal")])
def test_thinking_is_sent_as_reasoning_effort(thinking: str, effort: str) -> None:
    provider, client = _provider()
    provider.generate("hello", tools=_TOOLS, thinking=thinking)
    assert client.payloads[-1].get("reasoning_effort") == effort
    assert "chat_template_kwargs" not in client.payloads[-1]
