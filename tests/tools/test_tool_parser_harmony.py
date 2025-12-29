from __future__ import annotations

from abstractcore.tools.parser import clean_tool_syntax, detect_tool_calls, parse_tool_calls


def test_detect_and_parse_harmony_tool_prefix_json_args() -> None:
    content = (
        '<|channel|>commentary to=list_files <|constrain|>json<|message|>'
        '{"directory_path":"./gpt20b-rtype","recursive":true,"pattern":"*"}'
    )
    assert detect_tool_calls(content, model_name="openai/gpt-oss-20b") is True
    calls = parse_tool_calls(content, model_name="openai/gpt-oss-20b")
    assert len(calls) == 1
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {"directory_path": "./gpt20b-rtype", "recursive": True, "pattern": "*"}


def test_detect_and_parse_harmony_tool_prefix_wrapper_payload() -> None:
    content = (
        '<|channel|>commentary to=list_files <|constrain|>json<|message|>'
        '{"name":"list_files","arguments":{"directory_path":"./gpt20b-rtype","recursive":true,"pattern":"*"},"call_id":null}'
    )
    assert detect_tool_calls(content, model_name="openai/gpt-oss-20b") is True
    calls = parse_tool_calls(content, model_name="openai/gpt-oss-20b")
    assert len(calls) == 1
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {"directory_path": "./gpt20b-rtype", "recursive": True, "pattern": "*"}
    assert calls[0].call_id is None


def test_parse_harmony_tool_prefix_wrapper_with_unescaped_newlines() -> None:
    content = (
        '<|channel|>commentary to=write_file <|constrain|>json<|message|>'
        '{"name":"write_file","arguments":{"file_path":"oss-rtype/main.py","content":"line1\nline2\n","mode":"w"},"call_id":null}'
    )
    assert detect_tool_calls(content, model_name="openai/gpt-oss-20b") is True
    calls = parse_tool_calls(content, model_name="openai/gpt-oss-20b")
    assert len(calls) == 1
    assert calls[0].name == "write_file"
    assert calls[0].arguments["file_path"] == "oss-rtype/main.py"
    assert calls[0].arguments["content"] == "line1\nline2\n"
    assert calls[0].arguments["mode"] == "w"


def test_clean_tool_syntax_removes_harmony_tool_prefix_block() -> None:
    content = (
        "I'll list the directory.\n\n"
        '<|channel|>commentary to=list_files <|constrain|>json<|message|>'
        '{"directory_path":"./gpt20b-rtype","recursive":true,"pattern":"*"}\n'
    )
    calls = parse_tool_calls(content, model_name="openai/gpt-oss-20b")
    assert calls and calls[0].name == "list_files"
    cleaned = clean_tool_syntax(content, calls)
    assert "<|channel|>" not in cleaned
    assert "<|message|>" not in cleaned
    assert "I'll list the directory." in cleaned
