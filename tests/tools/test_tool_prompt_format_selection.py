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

