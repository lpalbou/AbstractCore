from __future__ import annotations


def test_tool_prompt_renders_when_to_use_for_guidance() -> None:
    from abstractcore.tools.common_tools import edit_file, fetch_url, skim_url, write_file
    from abstractcore.tools.parser import format_tool_prompt

    tool_defs = [
        edit_file._tool_definition,
        write_file._tool_definition,
        skim_url._tool_definition,
        fetch_url._tool_definition,
    ]

    prompt = format_tool_prompt(tool_defs, model_name="mistral-7b")
    for tool_def in tool_defs:
        assert tool_def.when_to_use
        assert tool_def.when_to_use in prompt
