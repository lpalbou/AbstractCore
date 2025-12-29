from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional, Union

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider
from abstractcore.tools.handler import UniversalToolHandler


class _DummyProvider(BaseProvider):
    """A minimal provider used to exercise BaseProvider normalization logic."""

    def __init__(self, *, model: str, response: GenerateResponse):
        super().__init__(model)
        self.tool_handler = UniversalToolHandler(model)
        self._response = response

    def get_capabilities(self) -> List[str]:
        return []

    def list_available_models(self, **kwargs) -> List[str]:
        return [self.model]

    def generate(self, *args, **kwargs):
        return self.generate_with_telemetry(*args, **kwargs)

    def _generate_internal(
        self,
        prompt: str,
        messages: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        media: Optional[List[Any]] = None,
        stream: bool = False,
        response_model: Any = None,
        execute_tools: Optional[bool] = None,
        tool_call_tags: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
        assert stream is False
        return self._response


def test_non_streaming_prompted_tool_calls_are_parsed_and_content_is_cleaned() -> None:
    provider = _DummyProvider(
        model="qwen/qwen3-next-80b",
        response=GenerateResponse(
            content=(
                "Let me check.\n\n"
                "<|tool_call|>\n"
                '{"name":"list_files","arguments":{"directory":"."}}\n'
                "</|tool_call|>\n"
            ),
            tool_calls=None,
            model="qwen/qwen3-next-80b",
        ),
    )

    tools = [{"name": "list_files", "description": "List files", "parameters": {"directory": {"type": "string"}}}]
    resp = provider.generate(prompt="list files", tools=tools)

    assert resp.tool_calls == [{"name": "list_files", "arguments": {"directory": "."}, "call_id": None}]
    assert "<|tool_call|>" not in (resp.content or "")
    assert resp.content.strip() == "Let me check."


def test_non_streaming_native_tool_calls_are_normalized_and_echoed_markup_is_cleaned() -> None:
    provider = _DummyProvider(
        model="openai/gpt-4o-mini",
        response=GenerateResponse(
            content=(
                "I will read the file.\n\n"
                "<|tool_call|>\n"
                '{"name":"read_file","arguments":{"path":"README.md"}}\n'
                "</|tool_call|>\n"
            ),
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": '{"path":"README.md"}'},
                }
            ],
            model="openai/gpt-4o-mini",
        ),
    )

    tools = [{"name": "read_file", "description": "Read a file", "parameters": {"path": {"type": "string"}}}]
    resp = provider.generate(prompt="read README.md", tools=tools)

    assert resp.tool_calls == [{"name": "read_file", "arguments": {"path": "README.md"}, "call_id": "call_1"}]
    assert "<|tool_call|>" not in (resp.content or "")
    assert "I will read the file." in (resp.content or "")


def test_non_streaming_tool_call_tags_request_preserves_tags_and_rewrites_format() -> None:
    provider = _DummyProvider(
        model="qwen/qwen3-next-80b",
        response=GenerateResponse(
            content=(
                "<|tool_call|>\n"
                '{"name":"list_files","arguments":{"directory":"."}}\n'
                "</|tool_call|>\n"
            ),
            tool_calls=None,
            model="qwen/qwen3-next-80b",
        ),
    )

    tools = [{"name": "list_files", "description": "List files", "parameters": {"directory": {"type": "string"}}}]
    resp = provider.generate(prompt="list files", tools=tools, tool_call_tags="llama3")

    assert resp.tool_calls == [{"name": "list_files", "arguments": {"directory": "."}, "call_id": None}]
    assert "<function_call>" in (resp.content or "")
    assert "<|tool_call|>" not in (resp.content or "")


def test_non_streaming_harmony_tool_calls_are_parsed_and_content_is_cleaned() -> None:
    provider = _DummyProvider(
        model="openai/gpt-oss-20b",
        response=GenerateResponse(
            content=(
                "Okay.\n\n"
                "<|channel|>commentary to=list_files <|constrain|>json<|message|>"
                '{"directory_path":"./gpt20b-rtype","recursive":true,"pattern":"*"}'
            ),
            tool_calls=None,
            model="openai/gpt-oss-20b",
        ),
    )

    tools = [
        {
            "name": "list_files",
            "description": "List files",
            "parameters": {
                "directory_path": {"type": "string"},
                "recursive": {"type": "boolean"},
                "pattern": {"type": "string"},
            },
        }
    ]
    resp = provider.generate(prompt="list files", tools=tools)

    assert resp.tool_calls == [
        {
            "name": "list_files",
            "arguments": {"directory_path": "./gpt20b-rtype", "recursive": True, "pattern": "*"},
            "call_id": None,
        }
    ]
    assert "<|channel|>" not in (resp.content or "")
    assert resp.content.strip() == "Okay."


def test_non_streaming_harmony_wrapper_tool_calls_are_parsed_and_content_is_cleaned() -> None:
    provider = _DummyProvider(
        model="openai/gpt-oss-20b",
        response=GenerateResponse(
            content=(
                "Okay.\n\n"
                "<|channel|>commentary to=list_files <|constrain|>json<|message|>"
                '{"name":"list_files","arguments":{"directory_path":"./gpt20b-rtype","recursive":true,"pattern":"*"},"call_id":null}'
            ),
            tool_calls=None,
            model="openai/gpt-oss-20b",
        ),
    )

    tools = [
        {
            "name": "list_files",
            "description": "List files",
            "parameters": {
                "directory_path": {"type": "string"},
                "recursive": {"type": "boolean"},
                "pattern": {"type": "string"},
            },
        }
    ]
    resp = provider.generate(prompt="list files", tools=tools)

    assert resp.tool_calls == [
        {
            "name": "list_files",
            "arguments": {"directory_path": "./gpt20b-rtype", "recursive": True, "pattern": "*"},
            "call_id": None,
        }
    ]
    assert "<|channel|>" not in (resp.content or "")
    assert resp.content.strip() == "Okay."
