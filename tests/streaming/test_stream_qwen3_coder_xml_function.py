"""Streamed Qwen3-Coder "XML function" tool calls (UnifiedStreamProcessor).

The envelope is withheld from content chunks, parsed when it closes, and an
envelope that never completes is reported as `unparsed_tool_call` (never
printed as the answer).
"""

from pathlib import Path

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.streaming import UnifiedStreamProcessor

MODEL = "Jundot/Qwen3.8-27B-oQ4e-mtp"
OPERATOR_TEXT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "qwen3_coder_three_fetch_url_calls.txt"
).read_text()
TOOLS = [{"name": "fetch_url", "description": "Fetch a URL", "parameters": {}}]
CUTS = [1, 7, 1000]


def _run(text, cut, *, model=MODEL, tools=TOOLS):
    chunks = [GenerateResponse(content=text[i : i + cut], model=model) for i in range(0, len(text), cut)]
    chunks.append(GenerateResponse(content="", model=model, finish_reason="stop"))
    content, calls, unparsed, warnings = "", [], [], []
    for out in UnifiedStreamProcessor(model_name=model).process_stream(iter(chunks), tools):
        content += out.content or ""
        calls.extend(out.tool_calls or [])
        meta = out.metadata if isinstance(out.metadata, dict) else {}
        if "unparsed_tool_call" in meta:
            unparsed.append(meta["unparsed_tool_call"])
        warnings.extend(meta.get("warnings") or [])
    return content, calls, unparsed, warnings


@pytest.mark.parametrize("cut", CUTS)
@pytest.mark.parametrize("model", [MODEL, None])
def test_operator_text_streams_three_calls_and_no_markup(cut, model):
    content, calls, unparsed, _ = _run(OPERATOR_TEXT, cut, model=model)
    assert [c["name"] for c in calls] == ["fetch_url"] * 3
    assert calls[1]["arguments"] == {"url": "https://aiweekly.co/ai-news-today", "include_full_content": "False"}
    assert "<tool_call>" not in content and "<function=" not in content and "<parameter" not in content
    assert content.strip() == ""
    assert unparsed == []


@pytest.mark.parametrize("cut", CUTS)
def test_prose_before_the_calls_streams_as_content(cut):
    content, calls, _, _ = _run("Fetching three sources.\n\n" + OPERATOR_TEXT, cut)
    assert content.strip() == "Fetching three sources."
    assert len(calls) == 3


@pytest.mark.parametrize("cut", CUTS)
def test_call_without_parameters(cut):
    _, calls, unparsed, _ = _run("<tool_call>\n<function=fetch_url>\n</function>\n</tool_call>", cut)
    assert [(c["name"], c["arguments"]) for c in calls] == [("fetch_url", {})]
    assert unparsed == []


@pytest.mark.parametrize("cut", CUTS)
def test_several_functions_in_one_envelope_are_all_calls(cut):
    text = (
        "<tool_call>\n<function=fetch_url>\n<parameter=url>\nhttps://a.example\n</parameter>\n</function>\n"
        "<function=fetch_url>\n<parameter=url>\nhttps://b.example\n</parameter>\n</function>\n</tool_call>"
    )
    _, calls, _, _ = _run(text, cut)
    assert [c["arguments"]["url"] for c in calls] == ["https://a.example", "https://b.example"]


@pytest.mark.parametrize("cut", CUTS)
def test_output_cut_inside_a_value_is_reported_unparsed_not_printed_or_run(cut):
    text = "Checking.\n<tool_call>\n<function=fetch_url>\n<parameter=url>\nhttps://aiweekly.co/ai-"
    content, calls, unparsed, warnings = _run(text, cut)
    assert calls == []
    assert content.strip() == "Checking."
    assert len(unparsed) == 1 and unparsed[0]["reason"] == "unclosed"
    assert "<function=fetch_url>" in unparsed[0]["text"]
    assert any("Unparsed tool call" in w for w in warnings)


@pytest.mark.parametrize("cut", CUTS)
def test_unknown_name_is_kept_with_a_warning_when_tools_are_offered(cut):
    text = "<tool_call>\n<function=web_fetch>\n<parameter=url>\nhttps://a.example\n</parameter>\n</function>\n</tool_call>"
    _, calls, _, warnings = _run(text, cut)
    assert [c["name"] for c in calls] == ["web_fetch"]
    assert any("does not match any available tool" in w for w in warnings)


@pytest.mark.parametrize("cut", CUTS)
def test_calls_written_inside_the_thinking_block_are_still_calls(cut):
    # The processor runs on the raw stream before inline thinking is split.
    content, calls, _, _ = _run("<think>\nI need news.\n" + OPERATOR_TEXT + "\n</think>\n", cut)
    assert len(calls) == 3
    assert "<tool_call>" not in content
