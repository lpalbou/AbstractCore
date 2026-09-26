"""Qwen3-Coder "XML function" tool calls (Qwen3.8 chat template), non-streamed parser.

Grammar (the model's chat template, `tool_calling_format: qwen3_coder`):

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    value (may span lines)
    </parameter>
    </function>
    </tool_call>

Values stay raw strings at parse time (backlog 039): they are coerced to the
tool's declared schema type at dispatch (`tools/arg_coercion.py`), so a string
field whose value happens to read `42` or `true` is never mangled.
"""

from pathlib import Path

import pytest

from abstractcore.tools.arg_coercion import coerce_arguments
from abstractcore.tools.parser import (
    clean_tool_syntax,
    detect_tool_calls,
    detect_unparsed_tool_intent,
    parse_tool_calls,
)

MODEL = "Jundot/Qwen3.8-27B-oQ4e-mtp"
OPERATOR_TEXT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "qwen3_coder_three_fetch_url_calls.txt"
).read_text()
URLS = [
    "https://headsupai.io/ai-news-and-updates/today",
    "https://aiweekly.co/ai-news-today",
    "https://www.businesstoday.in/technology/news/story/"
    "blackrock-sees-ai-agents-creating-a-new-economy-for-stablecoins-blockchain-and-compute-557945-2026-09-26",
]
FETCH_URL_SCHEMA = {
    "url": {"type": "string"},
    "include_full_content": {"type": "boolean", "default": True},
}


def _call(name, params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params)
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


@pytest.mark.parametrize("model", [MODEL, None])
def test_operator_text_three_consecutive_calls(model):
    assert detect_tool_calls(OPERATOR_TEXT, model)
    calls = parse_tool_calls(OPERATOR_TEXT, model)
    assert [c.name for c in calls] == ["fetch_url"] * 3
    assert [c.arguments["url"] for c in calls] == URLS
    # Surrounding newlines trimmed; Python-style literal kept raw for dispatch.
    assert all(c.arguments["include_full_content"] == "False" for c in calls)
    assert clean_tool_syntax(OPERATOR_TEXT, calls).strip() == ""


def test_operator_text_coerces_at_dispatch_to_the_declared_type():
    call = parse_tool_calls(OPERATOR_TEXT, MODEL)[0]
    coerced, _warnings = coerce_arguments(FETCH_URL_SCHEMA, call.arguments)
    assert coerced == {"url": URLS[0], "include_full_content": False}


def test_single_call():
    calls = parse_tool_calls(_call("fetch_url", [("url", "https://a.example")]), MODEL)
    assert [(c.name, c.arguments) for c in calls] == [("fetch_url", {"url": "https://a.example"})]


def test_multiline_value_keeps_inner_newlines_and_trims_wrapper_newlines():
    text = _call("write_file", [("path", "a.txt"), ("content", "line 1\n\n  line 3\n")])
    (call,) = parse_tool_calls(text, MODEL)
    assert call.arguments == {"path": "a.txt", "content": "line 1\n\n  line 3\n"}


@pytest.mark.parametrize(
    "raw,schema_type,expected",
    [
        ("False", "boolean", False),
        ("True", "boolean", True),
        ("false", "boolean", False),
        ("3", "integer", 3),
        ("2.5", "number", 2.5),
        ('["a", "b"]', "array", ["a", "b"]),
        ('{"k": 1}', "object", {"k": 1}),
        ("42", "string", "42"),
        ("True", "string", "True"),
    ],
)
def test_values_parse_raw_and_coerce_by_schema(raw, schema_type, expected):
    (call,) = parse_tool_calls(_call("t", [("v", raw)]), MODEL)
    assert call.arguments == {"v": raw}
    coerced, _ = coerce_arguments({"v": {"type": schema_type}}, call.arguments)
    assert coerced == {"v": expected}


def test_missing_parameter_is_absent_not_invented():
    (call,) = parse_tool_calls(_call("fetch_url", [("url", "https://a.example")]), MODEL)
    assert "include_full_content" not in call.arguments


def test_call_without_parameters():
    calls = parse_tool_calls("<tool_call>\n<function=list_models>\n</function>\n</tool_call>", MODEL)
    assert [(c.name, c.arguments) for c in calls] == [("list_models", {})]


def test_several_functions_in_one_envelope():
    text = (
        "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n"
        "<function=b>\n<parameter=y>\n2\n</parameter>\n</function>\n</tool_call>"
    )
    assert [(c.name, c.arguments) for c in parse_tool_calls(text, MODEL)] == [
        ("a", {"x": "1"}),
        ("b", {"y": "2"}),
    ]


def test_unknown_function_name_is_still_parsed_roster_is_the_provider_s_job():
    # The parser never knows the offered tools; BaseProvider maps/drops names
    # (see tests/providers/test_tool_calls_inside_reasoning.py).
    (call,) = parse_tool_calls(_call("not_a_tool", [("q", "x")]), MODEL)
    assert call.name == "not_a_tool"


def test_output_cut_inside_a_parameter_value_is_not_a_call():
    cut = "<tool_call>\n<function=fetch_url>\n<parameter=url>\nhttps://aiweekly.co/ai-"
    assert parse_tool_calls(cut, MODEL) == []
    assert detect_unparsed_tool_intent(cut)


def test_missing_closing_wrapper_after_a_complete_function_is_still_a_call():
    text = "<tool_call>\n<function=fetch_url>\n<parameter=url>\nhttps://a.example\n</parameter>\n</function>\n"
    assert [c.arguments for c in parse_tool_calls(text, MODEL)] == [{"url": "https://a.example"}]
