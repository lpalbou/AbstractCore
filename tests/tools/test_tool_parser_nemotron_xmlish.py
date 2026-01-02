from abstractcore.tools.parser import parse_tool_calls


def test_parse_tool_call_nemotron_xmlish_function_and_parameters():
    content = """
<tool_call>
<function=write_file>
<parameter=file_path>
pcr2/main.py
</parameter>
<parameter=content>
import pygame
print("hi")
</parameter>
</function>
</tool_call>
""".strip()

    calls = parse_tool_calls(content, model_name="nvidia/nemotron-3-nano")
    assert len(calls) == 1
    assert calls[0].name == "write_file"
    assert calls[0].arguments["file_path"] == "pcr2/main.py"
    assert calls[0].arguments["content"] == 'import pygame\nprint("hi")'

