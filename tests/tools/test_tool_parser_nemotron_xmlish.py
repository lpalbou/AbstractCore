from abstractcore.tools.parser import clean_tool_syntax, detect_tool_calls, parse_tool_calls


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


def test_parse_tool_call_direct_tool_tag_with_mismatched_function_close():
    content = """
<tool_call>
<web_search>
<parameter=query>
Coolizi arnaque scam avis 2025 2026
</parameter>
<parameter=num_results>
10
</parameter>
</function>
</tool_call>
<tool_call>
<web_search>
<parameter=query>
producttrendreport.com avis fiable légitime
</parameter>
<parameter=num_results>
10
</parameter>
</function>
</tool_call>
<tool_call>
<skim_url>
<parameter=url>
https://producttrendreport.com/coolizi-testbericht-fr/
</parameter>
<parameter=max_preview_chars>
2000
</parameter>
</function>
</tool_call>
""".strip()

    assert detect_tool_calls(content, model_name="nvidia/nemotron-3-nano") is True

    calls = parse_tool_calls(content, model_name="nvidia/nemotron-3-nano")
    assert [call.name for call in calls] == ["web_search", "web_search", "skim_url"]
    assert calls[0].arguments == {
        "query": "Coolizi arnaque scam avis 2025 2026",
        "num_results": "10",
    }
    assert calls[1].arguments == {
        "query": "producttrendreport.com avis fiable légitime",
        "num_results": "10",
    }
    assert calls[2].arguments == {
        "url": "https://producttrendreport.com/coolizi-testbericht-fr/",
        "max_preview_chars": "2000",
    }

    cleaned = clean_tool_syntax(content, calls)
    assert cleaned == ""


def test_parse_tool_call_xmlish_missing_outer_close_is_recovered_and_cleaned():
    content = """
I will search.
<tool_call>
<function=web_search>
<parameter=query>Coolizi arnaque scam avis</parameter>
<parameter=num_results>10</parameter>
</function>
""".strip()

    calls = parse_tool_calls(content, model_name="nvidia/nemotron-3-nano")
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "Coolizi arnaque scam avis", "num_results": "10"}

    cleaned = clean_tool_syntax(content, calls)
    assert cleaned.strip() == "I will search."


def test_parse_tool_call_direct_tool_tag_same_close_missing_outer_is_cleaned():
    content = """
I will search.
<tool_call>
<web_search>
<parameter=query>Coolizi avis</parameter>
</web_search>
""".strip()

    calls = parse_tool_calls(content, model_name="nvidia/nemotron-3-nano")
    assert len(calls) == 1
    assert calls[0].name == "web_search"
    assert calls[0].arguments == {"query": "Coolizi avis"}

    cleaned = clean_tool_syntax(content, calls)
    assert cleaned.strip() == "I will search."
