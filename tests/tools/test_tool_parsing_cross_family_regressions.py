"""Cross-family pins for the shared tool parser / cleaner / stream detector.

The HY V3 (Hy-Memory) work taught `_parse_arg_kv_tool_call` a `<tool_sep>` name
separator, truncation rejection and an argument-less call, and taught
`clean_tool_syntax` to drop an unclosed trailing call. Those lanes are shared by
every family, so the corpus below pins the results of the pre-HY parser
(83d5f6d4) for GLM-4.6, Laguna, Qwen3, Qwen3-Coder, Nemotron and an unknown model.

Regressions found in review and pinned here:
  S1  a GLM/Laguna value that MENTIONS `<tool_sep>` or `<arg_key>` dropped the call;
  S2  prose after a parsed call that mentions `<tool_call>` was truncated;
  S3  a HY call cut right after `name<tool_sep>` ran with no arguments;
  S4  a call cut inside a value and then closed vanished from the stream untraced;
  S5  the `<tool_calls>` envelope was scanned for every glm_xml family.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractcore.providers.streaming import IncrementalToolDetector
from abstractcore.tools.parser import clean_tool_syntax, parse_tool_calls

MODELS = [
    "zai-org/GLM-4.6",
    "poolside/Laguna-XS",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
    "nvidia/Nemotron-3-Nano-30B-A3B",
    None,
]

HY = "tencent/Hy-Memory-30B-A3B"

WEATHER = [("get_weather", {"city": "Paris"})]

# case -> (payload, parse result, clean_tool_syntax(payload, calls) or None when no
# call parsed, stream calls), all as produced by 83d5f6d4 for every model in MODELS.
BASELINE: Dict[str, Tuple[str, List[Tuple[str, Dict[str, Any]]], Optional[str], List[Tuple[str, Dict[str, Any]]]]] = {
    "glm_basic": (
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>",
        WEATHER, "", WEATHER,
    ),
    "glm_two": (
        "ok\n"
        "<tool_call>a<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>\n"
        '<tool_call>b<arg_key>y</arg_key><arg_value>{"k":[1,2,{"z":null}]}</arg_value></tool_call>',
        [("a", {"x": 1}), ("b", {"y": {"k": [1, 2, {"z": None}]}})],
        "ok",
        [("a", {"x": 1}), ("b", {"y": {"k": [1, 2, {"z": None}]}})],
    ),
    "glm_nested_json": (
        '<tool_call>write<arg_key>data</arg_key><arg_value>{"a":{"b":[1,{"c":"<x>"}]}}</arg_value>'
        "<arg_key>n</arg_key><arg_value>3</arg_value></tool_call>",
        [("write", {"data": {"a": {"b": [1, {"c": "<x>"}]}}, "n": 3})],
        "",
        [("write", {"data": {"a": {"b": [1, {"c": "<x>"}]}}, "n": 3})],
    ),
    # S1: the value mentions the HY separator.
    "glm_value_contains_tool_sep": (
        "<tool_call>write_file<arg_key>path</arg_key><arg_value>p.py</arg_value>"
        '<arg_key>content</arg_key><arg_value>SEP = "<tool_sep>"</arg_value></tool_call>',
        [("write_file", {"content": 'SEP = "<tool_sep>"', "path": "p.py"})],
        "",
        [("write_file", {"content": 'SEP = "<tool_sep>"', "path": "p.py"})],
    ),
    # S1: the value mentions a complete <arg_key> pair.
    "glm_value_contains_arg_key": (
        "<tool_call>write_file<arg_key>path</arg_key><arg_value>p.md</arg_value>"
        "<arg_key>content</arg_key><arg_value>use <arg_key>k</arg_key> tags</arg_value></tool_call>",
        [("write_file", {"content": "use <arg_key>k</arg_key> tags", "path": "p.md"})],
        "",
        [("write_file", {"content": "use <arg_key>k</arg_key> tags", "path": "p.md"})],
    ),
    "glm_empty_args_no_sep": ("<tool_call>list_files</tool_call>", [], None, []),
    "glm_newlines": (
        "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>",
        WEATHER, "", WEATHER,
    ),
    "preamble_as_name": (
        "Let me check the weather for you.\n"
        "<tool_call>I will now call get_weather\n<arg_key>city</arg_key>\n<arg_value>Paris</arg_value>\n</tool_call>",
        [], None, [],
    ),
    "prose_literal_tool_call": (
        "In this format you write <tool_call> followed by the name. That is all.", [], None, [],
    ),
    "prose_literal_tool_calls": (
        "The model emits <tool_calls> then </tool_calls> around calls.", [], None, [],
    ),
    "json_call": (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
        WEATHER, "", WEATHER,
    ),
    # S2: prose after the call mentions the tag.
    "json_call_then_prose_mention": (
        '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n'
        "Next I may use <tool_call> again for the forecast; please confirm.",
        WEATHER,
        "Next I may use <tool_call> again for the forecast; please confirm.",
        WEATHER,
    ),
    "qwen3_coder": (
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
        WEATHER, "", WEATHER,
    ),
    "qwen3_coder_no_args": (
        "<tool_call>\n<function=list_files>\n</function>\n</tool_call>",
        [("list_files", {})], "", [("list_files", {})],
    ),
    "special_token": (
        '<|tool_call|>{"name": "get_weather", "arguments": {"city": "Paris"}}</|tool_call|>',
        WEATHER, "", WEATHER,
    ),
    "raw_json": (
        '{"name": "get_weather", "arguments": {"city": "Paris"}}',
        WEATHER, '{"name": "get_weather", "arguments": {"city": "Paris"}}', [],
    ),
    "function_call": (
        '<function_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</function_call>',
        WEATHER, "", WEATHER,
    ),
    "arg_key_uppercase": (
        "<tool_call>f<ARG_KEY>a</ARG_KEY><ARG_VALUE>1</ARG_VALUE></tool_call>",
        [("f", {"a": 1})], "", [("f", {"a": 1})],
    ),
}


def _calls(calls) -> List[Tuple[str, Dict[str, Any]]]:
    return [(c.name, c.arguments) for c in calls]


def _stream(model: Optional[str], text: str, step: int):
    detector = IncrementalToolDetector(model_name=model)
    content, calls = "", []
    for i in range(0, len(text), step):
        out, done = detector.process_chunk(text[i:i + step])
        content += out
        calls += done
    owed, done, unparsed = detector.drain()
    return content + owed, calls + list(done), unparsed


@pytest.mark.parametrize("model", MODELS, ids=lambda m: str(m))
@pytest.mark.parametrize("case", sorted(BASELINE))
def test_parse_and_clean_match_the_pre_hy_parser(case: str, model: Optional[str]) -> None:
    payload, expected_calls, expected_clean, _ = BASELINE[case]
    calls = parse_tool_calls(payload, model)
    assert _calls(calls) == expected_calls
    assert (clean_tool_syntax(payload, calls) if calls else None) == expected_clean


@pytest.mark.parametrize("step", [1, 7, 10_000])
@pytest.mark.parametrize("model", MODELS, ids=lambda m: str(m))
@pytest.mark.parametrize("case", sorted(BASELINE))
def test_stream_calls_match_the_pre_hy_detector(case: str, model: Optional[str], step: int) -> None:
    payload, _, _, expected_stream = BASELINE[case]
    _, calls, _ = _stream(model, payload, step)
    assert _calls(calls) == expected_stream


def test_gemma4_call_parses_for_every_family() -> None:
    payload = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
    for model in MODELS:
        assert _calls(parse_tool_calls(payload, model)) == [("get_weather", {})]


# --- S2: the unclosed-trailing cleaner only removes a body that starts like a call ---


def test_prose_mentioning_tool_call_survives_cleaning() -> None:
    assert (
        clean_tool_syntax("In this format you write <tool_call> followed by the name. That is all.", None)
        == "In this format you write <tool_call> followed by the name. That is all."
    )


@pytest.mark.parametrize(
    "trailing",
    [
        '<tool_call>\n{"name": "save_memory", "arguments": {"fact": "x"}}',  # live Hy-Memory GGUF shape + EOS
        "<tool_call>\n<function=save_memory>\n<parameter=fact>\nx\n</parameter>\n</function>",
        "<tool_call>save_memory<tool_sep>\n<arg_key>fact</arg_key>\n<arg_value>x</arg_value>\n",
        "<tool_call>save_memory<arg_key>fact</arg_key><arg_value>x</arg_value>",
    ],
)
def test_unclosed_trailing_call_payload_is_still_removed(trailing: str) -> None:
    text = "Saving that.\n" + trailing
    calls = parse_tool_calls(text, HY)
    assert [c.name for c in calls] == ["save_memory"]
    assert clean_tool_syntax(text, calls) == "Saving that."


# --- S3: a HY call cut right after its separator is not a call ---


@pytest.mark.parametrize(
    "text",
    [
        "<tool_calls>\n<tool_call>save_memory<tool_sep>\n",
        "<tool_calls>\n<tool_call>save_memory<tool_sep>",
        "<tool_call>save_memory<tool_sep>\n",
    ],
)
@pytest.mark.parametrize("model", [HY, None, "poolside/Laguna-XS"], ids=str)
def test_call_cut_after_the_separator_is_not_executed(text: str, model: Optional[str]) -> None:
    assert parse_tool_calls(text, model) == []


@pytest.mark.parametrize("step", [1, 5, 10_000])
def test_streamed_call_cut_after_the_separator_is_reported_not_executed(step: int) -> None:
    content, calls, unparsed = _stream(HY, "<tool_calls>\n<tool_call>save_memory<tool_sep>\n", step)
    assert calls == []
    assert content == ""
    assert unparsed is not None and unparsed["reason"] == "unclosed"


@pytest.mark.parametrize("model", [HY, None, "poolside/Laguna-XS"], ids=str)
def test_closed_argument_less_call_is_a_call(model: Optional[str]) -> None:
    text = "<tool_calls>\n<tool_call>list_memories<tool_sep>\n</tool_call>\n</tool_calls>"
    assert _calls(parse_tool_calls(text, model)) == [("list_memories", {})]


# --- S4: a call cut inside a value is rejected, and a CLOSED one is reported ---

CUT = (
    "<tool_call>write_file<arg_key>path</arg_key><arg_value>a.txt</arg_value>"
    "<arg_key>content</arg_key><arg_value>hello wor"
)


@pytest.mark.parametrize("model", MODELS, ids=lambda m: str(m))
@pytest.mark.parametrize("closed", [False, True])
def test_call_cut_inside_a_value_is_not_executed(model: Optional[str], closed: bool) -> None:
    assert parse_tool_calls(CUT + ("</tool_call>" if closed else ""), model) == []


@pytest.mark.parametrize("step", [1, 5, 10_000])
@pytest.mark.parametrize("model", ["poolside/Laguna-XS", "zai-org/GLM-4.6", "Qwen/Qwen3-8B", HY], ids=str)
def test_closed_unparseable_envelope_is_reported_by_drain(model: str, step: int) -> None:
    content, calls, unparsed = _stream(model, "pre " + CUT + "</tool_call> post", step)
    assert calls == []
    assert content == "pre  post"
    assert unparsed is not None
    assert unparsed["reason"] == "unparseable"
    assert "hello wor" in unparsed["text"] and unparsed["text"].startswith("<tool_call>")


def test_unparseable_report_does_not_leak_into_the_next_stream() -> None:
    detector = IncrementalToolDetector(model_name="poolside/Laguna-XS")
    detector.process_chunk(CUT + "</tool_call>")
    assert detector.drain()[2] is not None
    streamed, calls = detector.process_chunk("plain answer")
    owed, more, unparsed = detector.drain()
    assert (streamed + owed, calls + list(more), unparsed) == ("plain answer", [], None)


# --- S5: the `<tool_calls>` envelope is HY V3's alone ---


@pytest.mark.parametrize("step", [1, 7, 10_000])
@pytest.mark.parametrize(
    "model", ["poolside/Laguna-XS", "zai-org/GLM-5.1", "zai-org/GLM-4.6V", None], ids=str
)
def test_prose_mentioning_tool_calls_is_kept_outside_hy(model: Optional[str], step: int) -> None:
    text = "The model emits <tool_calls> then </tool_calls> around calls."
    content, calls, unparsed = _stream(model, text, step)
    assert (content, calls, unparsed) == (text, [], None)


@pytest.mark.parametrize("step", [1, 7, 10_000])
def test_hy_stream_still_strips_the_wrapper(step: int) -> None:
    text = (
        "Saving.\n<tool_calls>\n<tool_call>save_memory<tool_sep>\n<arg_key>fact</arg_key>\n"
        "<arg_value>Alex is allergic to peanuts</arg_value>\n</tool_call>\n</tool_calls>"
    )
    content, calls, unparsed = _stream(HY, text, step)
    assert _calls(calls) == [("save_memory", {"fact": "Alex is allergic to peanuts"})]
    assert content.strip() == "Saving."
    assert unparsed is None
