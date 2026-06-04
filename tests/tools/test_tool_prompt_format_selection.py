from __future__ import annotations

from abstractcore.tools.core import ToolDefinition
from abstractcore.tools.parser import clean_tool_syntax, detect_tool_calls, format_tool_prompt, parse_tool_calls


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


def test_gemma4_tool_prompt_uses_gemma4_special_token_syntax() -> None:
    tool = ToolDefinition.from_function(_echo)
    prompt = format_tool_prompt([tool], model_name="google/gemma-4-31B-it", include_tool_list=False, include_examples=False)
    assert "<|tool_call>" in prompt
    assert "<tool_call|>" in prompt
    assert "call:tool_name" in prompt


def test_parse_tool_calls_supports_gemma4_call_syntax() -> None:
    content = '<|tool_call>call:list_files{"directory_path":"."}<tool_call|>'
    calls = parse_tool_calls(content, model_name="google/gemma-4-31B-it")
    assert len(calls) == 1
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {"directory_path": "."}


def test_lfm2_tool_prompt_uses_liquid_special_token_syntax() -> None:
    tool = ToolDefinition.from_function(_echo)
    prompt = format_tool_prompt(
        [tool],
        model_name="LiquidAI/LFM2.5-8B-A1B",
        include_tool_list=False,
        include_examples=False,
    )
    assert "<|tool_call_start|>" in prompt
    assert "<|tool_call_end|>" in prompt
    assert '[tool_name(param1="value1", param2="value2")]' in prompt
    assert "<|tool_call|>" not in prompt


def test_parse_tool_calls_supports_lfm2_pythonic_special_token_syntax() -> None:
    content = '<|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|>'
    assert detect_tool_calls(content, model_name="LiquidAI/LFM2.5-8B-A1B")

    calls = parse_tool_calls(content, model_name="LiquidAI/LFM2.5-8B-A1B")
    assert len(calls) == 1
    assert calls[0].name == "get_candidate_status"
    assert calls[0].arguments == {"candidate_id": "12345"}


def test_clean_tool_syntax_removes_lfm2_special_token_blocks() -> None:
    content = 'Before <|tool_call_start|>[get_candidate_status(candidate_id="12345")]<|tool_call_end|> After'
    calls = parse_tool_calls(content, model_name="LiquidAI/LFM2.5-8B-A1B")
    cleaned = clean_tool_syntax(content, calls)
    assert cleaned == "Before  After"


def test_incremental_detector_supports_lfm2_pythonic_special_token_syntax() -> None:
    from abstractcore.providers.streaming import IncrementalToolDetector

    detector = IncrementalToolDetector(model_name="LiquidAI/LFM2.5-8B-A1B", rewrite_tags=False)
    chunks = [
        "Before ",
        "<|tool_call_start|>",
        '[get_candidate_status(candidate_id="12345")]',
        "<|tool_call_end|>",
        " After",
    ]

    streamable_parts = []
    tools = []
    for chunk in chunks:
        streamable, parsed = detector.process_chunk(chunk)
        streamable_parts.append(streamable)
        tools.extend(parsed)

    assert "".join(streamable_parts) == "Before  After"
    assert len(tools) == 1
    assert tools[0].name == "get_candidate_status"
    assert tools[0].arguments == {"candidate_id": "12345"}


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
