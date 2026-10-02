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

    def unload_model(self, model_name: str) -> None:
        return None

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


def test_non_streaming_xmlish_direct_tool_tag_is_parsed_and_content_is_cleaned() -> None:
    provider = _DummyProvider(
        model="nvidia/nemotron-3-nano",
        response=GenerateResponse(
            content=(
                "Searching now.\n\n"
                "<tool_call>\n"
                "<web_search>\n"
                "<parameter=query>Coolizi arnaque scam avis 2025 2026</parameter>\n"
                "<parameter=num_results>10</parameter>\n"
                "</function>\n"
                "</tool_call>\n"
            ),
            tool_calls=None,
            model="nvidia/nemotron-3-nano",
        ),
    )

    tools = [
        {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {"query": {"type": "string"}, "num_results": {"type": "integer"}},
        }
    ]
    resp = provider.generate(prompt="search", tools=tools)

    assert resp.tool_calls == [
        {
            "name": "web_search",
            "arguments": {"query": "Coolizi arnaque scam avis 2025 2026", "num_results": "10"},
            "call_id": None,
        }
    ]
    assert "<tool_call>" not in (resp.content or "")
    assert "<web_search>" not in (resp.content or "")
    assert resp.content.strip() == "Searching now."


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


def test_non_streaming_nested_wrapper_arguments_are_unwrapped() -> None:
    provider = _DummyProvider(
        model="openai/gpt-oss-20b",
        response=GenerateResponse(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "arguments": {
                        "name": "write_file",
                        "arguments": {"file_path": "oss-rtype/main.py", "content": "hi"},
                    },
                }
            ],
            model="openai/gpt-oss-20b",
        ),
    )

    tools = [{"name": "write_file", "description": "Write file", "parameters": {"file_path": {"type": "string"}}}]
    resp = provider.generate(prompt="write file", tools=tools)

    assert resp.tool_calls == [
        {"name": "write_file", "arguments": {"file_path": "oss-rtype/main.py", "content": "hi"}, "call_id": None}
    ]


def test_non_streaming_wrapper_arguments_with_extras_are_merged() -> None:
    provider = _DummyProvider(
        model="openai/gpt-oss-20b",
        response=GenerateResponse(
            content="",
            tool_calls=[
                {
                    "name": "write_file",
                    "arguments": {
                        "name": "write_file",
                        "arguments": {"content": "hi"},
                        # Some models mistakenly place real kwargs alongside wrapper fields.
                        "file_path": "oss-rtype/main.py",
                    },
                }
            ],
            model="openai/gpt-oss-20b",
        ),
    )

    tools = [{"name": "write_file", "description": "Write file", "parameters": {"file_path": {"type": "string"}}}]
    resp = provider.generate(prompt="write file", tools=tools)

    assert resp.tool_calls == [
        {"name": "write_file", "arguments": {"file_path": "oss-rtype/main.py", "content": "hi"}, "call_id": None}
    ]


def test_non_streaming_duplicate_tool_calls_are_deduped() -> None:
    provider = _DummyProvider(
        model="openai/gpt-oss-20b",
        response=GenerateResponse(
            content="",
            tool_calls=[
                {"name": "list_files", "arguments": {"directory_path": ".", "recursive": True, "pattern": "*"}},
                {"name": "list_files", "arguments": {"pattern": "*", "recursive": True, "directory_path": "."}},
            ],
            model="openai/gpt-oss-20b",
        ),
    )

    tools = [{"name": "list_files", "description": "List files", "parameters": {"directory_path": {"type": "string"}}}]
    resp = provider.generate(prompt="list files", tools=tools)

    assert resp.tool_calls == [
        {"name": "list_files", "arguments": {"directory_path": ".", "recursive": True, "pattern": "*"}, "call_id": None}
    ]


def test_native_tool_call_arguments_are_canonicalized() -> None:
    provider = _DummyProvider(
        model="openai/gpt-4o-mini",
        response=GenerateResponse(
            content="",
            tool_calls=[
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": {"file_path": "README.md", "start_line_one_indexed": 2, "end_line_one_indexed_inclusive": 2},
                    },
                }
            ],
            model="openai/gpt-4o-mini",
        ),
    )

    tools = [{"name": "read_file", "description": "Read a file", "parameters": {"file_path": {"type": "string"}}}]
    resp = provider.generate(prompt="read line 2", tools=tools)

    assert resp.tool_calls == [
        {
            "name": "read_file",
            "arguments": {"file_path": "README.md", "start_line": 2, "end_line": 2},
            "call_id": "call_1",
        }
    ]


def test_native_tool_calls_with_wrapped_function_names_are_mapped_to_allowed_tools() -> None:
    """Some OpenAI-compatible servers return wrapped function names like '{function-name: write_file}'."""
    provider = _DummyProvider(
        model="glm-4.6v",
        response=GenerateResponse(
            content="I will now call the function.",
            tool_calls=[
                {
                    "type": "function",
                    "id": "call_x",
                    "function": {
                        "name": "{function-name: write_file}",
                        "arguments": '{"file_path":"x.txt","content":"hello"}',
                    },
                }
            ],
            model="glm-4.6v",
        ),
    )

    tools = [{"name": "write_file", "description": "Write file", "parameters": {"file_path": {"type": "string"}}}]
    resp = provider.generate(prompt="write file", tools=tools)

    assert resp.tool_calls == [
        {"name": "write_file", "arguments": {"file_path": "x.txt", "content": "hello"}, "call_id": "call_x"}
    ]


def test_malformed_tool_call_has_structured_repair_diagnostic() -> None:
    provider = _DummyProvider(model="qwen3", response=GenerateResponse(
        content='<tool_call>{"name":"web_search","arguments": BROKEN}</tool_call>', model="qwen3"))
    response = provider.generate(prompt="Search", tools=[{"name": "web_search", "parameters": {"query": {"type": "string"}}}])
    assert not response.tool_calls
    assert response.metadata["tool_call_error"] == {
        "code": "invalid_tool_syntax", "available_tools": ["web_search"]}


def test_operator_nine_xml_calls_require_an_offered_tool() -> None:
    queries = ["Seagate Exos 12TB HDD price 2026", "Seagate Exos 16TB HDD price 2026",
               "Seagate Exos 18TB HDD price 2026", "Seagate Exos 24TB HDD price 2026",
               "Mac mini M6 32GB price availability 2026", "Mac mini M5 Pro 48GB price availability 2026",
               "Mac Studio M5 36GB price availability 2026", "Mac Studio M5 Ultra 96GB price availability 2026",
               "Mac Studio M5 256GB price availability 2026"]
    text = "I'll research current prices.\n" + "\n".join(
        f"<tool_call>\n<function=web_search>\n<parameter=query>\n{q}\n</parameter>\n</function>\n</tool_call>" for q in queries)
    for offered in (False, True):
        provider = _DummyProvider(model="Jundot/Qwen3.8-27B-oQ4e-mtp",
                                  response=GenerateResponse(content=text, model="qwen3"))
        response = provider.generate(prompt="Search", tools=[{"name": "web_search", "parameters": {"query": {"type": "string"}}}] if offered else [])
        if offered:
            assert [call["arguments"]["query"] for call in response.tool_calls] == queries
            assert "<tool_call>" not in response.content
        else:
            assert not response.tool_calls
