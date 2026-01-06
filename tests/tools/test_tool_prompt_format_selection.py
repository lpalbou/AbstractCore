from __future__ import annotations

from abstractcore.tools.core import ToolDefinition
from abstractcore.tools.parser import format_tool_prompt, parse_tool_calls


def _echo(value: str) -> str:
    return value


def test_prompted_tool_format_selection_prefers_architecture_specific_syntax() -> None:
    tool = ToolDefinition.from_function(_echo)

    # Qwen2-VL is configured as prompted tools + im_start_end message format.
    # It should prefer the Qwen <|tool_call|> syntax (not <function_call>).
    qwen_prompt = format_tool_prompt([tool], model_name="qwen2-vl")
    assert "<|tool_call|>" in qwen_prompt
    assert "<function_call>" not in qwen_prompt

    # Mistral is configured as JSON tool format (raw JSON tool calls).
    mistral_prompt = format_tool_prompt([tool], model_name="mistral-7b")
    assert '{"name": "tool_name"' in mistral_prompt
    assert "<|tool_call|>" not in mistral_prompt
    assert "<function_call>" not in mistral_prompt

    # OpenAI GPT models use native tools; we should not instruct prompted <function_call> tags.
    openai_prompt = format_tool_prompt([tool], model_name="gpt-4o-mini")
    assert "<function_call>" not in openai_prompt


def test_parse_tool_calls_prefers_raw_json_for_json_architectures() -> None:
    content = '{"name":"list_files","arguments":{"directory_path":"."}}'
    calls = parse_tool_calls(content, model_name="mistral-7b")
    assert len(calls) == 1
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {"directory_path": "."}


def _optional_none(path: str, end: int | None = None) -> str:
    _ = (path, end)
    return ""


def test_tool_prompt_does_not_render_default_null_for_optional_none() -> None:
    tool = ToolDefinition.from_function(_optional_none)
    prompt = format_tool_prompt([tool], model_name="mistral-7b")
    assert "default null" not in prompt
    assert "default None" not in prompt
    assert "(optional)" in prompt


def test_tool_prompt_can_omit_tool_list_but_keep_protocol() -> None:
    tool = ToolDefinition.from_function(_echo)
    prompt = format_tool_prompt([tool], model_name="mistral-7b", include_tool_list=False)
    assert tool.name not in prompt
    assert '{"name": "tool_name"' in prompt
    assert "CRITICAL RULES FOR TOOL USAGE" in prompt


def test_tool_prompt_explicitly_allows_multiple_tool_calls_per_response() -> None:
    tool = ToolDefinition.from_function(_echo)

    qwen_prompt = format_tool_prompt([tool], model_name="qwen2-vl", include_tool_list=False)
    assert "repeat the block once per call" in qwen_prompt
    assert "batch multiple tool calls" in qwen_prompt

    mistral_prompt = format_tool_prompt([tool], model_name="mistral-7b", include_tool_list=False)
    assert "multiple JSON objects" in mistral_prompt
    assert "batch multiple tool calls" in mistral_prompt


def test_parse_tool_calls_supports_multiple_raw_json_objects() -> None:
    content = '\n'.join(
        [
            '{"name":"list_files","arguments":{"directory_path":"."}}',
            '{"name":"read_file","arguments":{"file_path":"README.md"}}',
        ]
    )
    calls = parse_tool_calls(content, model_name="mistral-7b")
    assert [c.name for c in calls] == ["list_files", "read_file"]
