from __future__ import annotations

from abstractcore.tools.parser import parse_tool_calls


def test_parse_tool_calls_canonicalizes_read_file_range_aliases() -> None:
    response = (
        "<|tool_call|>\n"
        '{"name":"read_file","arguments":{"file_path":"README.md","start_line_one_indexed":2,"end_line_one_indexed_inclusive":3}}\n'
        "</|tool_call|>\n"
    )

    calls = parse_tool_calls(response, model_name="qwen/qwen3-next-80b")
    assert len(calls) == 1
    args = calls[0].arguments

    assert args["file_path"] == "README.md"
    assert args["start_line"] == 2
    assert args["end_line"] == 3
    assert "should_read_entire_file" not in args
    assert "start_line_one_indexed" not in args
    assert "end_line_one_indexed_inclusive" not in args
