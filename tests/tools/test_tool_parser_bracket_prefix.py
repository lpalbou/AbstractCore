from __future__ import annotations

from abstractcore.tools.parser import clean_tool_syntax, detect_tool_calls, parse_tool_calls


def test_detect_and_parse_bracket_prefix_json_args() -> None:
    content = 'tool: [list_files]: {"directory_path":"rtype","recursive":true}\n'
    assert detect_tool_calls(content) is True
    calls = parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {"directory_path": "rtype", "recursive": True}


def test_detect_and_parse_bracket_prefix_python_literal_args() -> None:
    content = "tool: [list_files]: {'directory_path': 'rtype', 'recursive': True}\n"
    assert detect_tool_calls(content) is True
    calls = parse_tool_calls(content)
    assert len(calls) == 1
    assert calls[0].name == "list_files"
    assert calls[0].arguments == {"directory_path": "rtype", "recursive": True}


def test_clean_tool_syntax_removes_orphan_closing_tag() -> None:
    content = (
        "I'll list files now.\n\n"
        "tool: [list_files]: {'directory_path': 'rtype', 'recursive': True}\n"
        "</tool_call>\n"
    )
    calls = parse_tool_calls(content)
    assert calls and calls[0].name == "list_files"
    cleaned = clean_tool_syntax(content, calls)
    assert "tool:" not in cleaned
    assert "</tool_call>" not in cleaned
    assert "I'll list files now." in cleaned

