"""Namespaced tool-call names and unrecognized-tool-syntax warnings.

Live incident (2026-08-28, session acode-912712976c0c, Qwen3.8-Flash-Next
UD-Q3_K_XL): at cycle 14 the model emitted a perfectly formed qwen3_coder
tool call whose function name carried an OpenAI-style namespace prefix —
`<function=functions.browser_probe>`. The `<function=...>` name pattern
rejected the dot, zero calls parsed, and the gateway concluded the ReAct run
with the tool call as the final answer.

Pinned here:
1. Dotted names parse inside the explicit `<function=...>` context.
2. The existing roster mapping recovers `functions.X` -> `X` end-to-end.
3. Tool-call syntax that yields no usable structured calls raises a
   `metadata["warnings"]` entry instead of silently becoming a final answer.
4. False-positive guards: bare `<ns.tag>` prose stays unparsed; prose without
   tool syntax gets no warning.
"""

import logging

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider
from abstractcore.providers.huggingface_provider import HuggingFaceProvider
from abstractcore.tools.handler import create_handler
from abstractcore.tools.parser import detect_unparsed_tool_intent, parse_tool_calls

MODEL = "qwen3.8-flash-next"

# Byte-shape of the incident payload (ledger msg_d2ca938e3ff1462796cebdb5392749a1).
INCIDENT_PAYLOAD = (
    "<tool_call>\n"
    "<function=functions.browser_probe>\n"
    "<parameter=target>\n"
    "file:///Users/albou/test-qwen38flash/test.html\n"
    "</parameter>\n"
    "<parameter=expect_text>\n"
    "ALL TESTS PASSED\n"
    "</parameter>\n"
    "<parameter=timeout_s>\n"
    "30\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>"
)

TOOLS = [
    {"name": "browser_probe", "description": "probe", "parameters": {"type": "object", "properties": {}}},
    {"name": "write_file", "description": "write", "parameters": {"type": "object", "properties": {}}},
]


def _passthrough_provider() -> HuggingFaceProvider:
    """A provider shell exercising only BaseProvider's passthrough machinery."""
    p = object.__new__(HuggingFaceProvider)
    p.model = MODEL
    p.logger = logging.getLogger("test.namespaced")
    p.tool_handler = create_handler(MODEL)
    return p


def _passthrough(content: str, tools=TOOLS) -> GenerateResponse:
    provider = _passthrough_provider()
    response = GenerateResponse(content=content, model=MODEL)
    return BaseProvider._normalize_tool_calls_passthrough(
        provider, response=response, tools=tools, tool_call_tags=None
    )


def test_dotted_function_name_parses():
    calls = parse_tool_calls(INCIDENT_PAYLOAD, MODEL)
    assert len(calls) == 1
    assert calls[0].name == "functions.browser_probe"
    assert calls[0].arguments == {
        "target": "file:///Users/albou/test-qwen38flash/test.html",
        "expect_text": "ALL TESTS PASSED",
        "timeout_s": "30",
    }


def test_incident_payload_recovers_to_roster_name_end_to_end():
    response = _passthrough(INCIDENT_PAYLOAD)
    assert response.tool_calls, "the incident payload must yield a structured tool call"
    assert response.tool_calls[0]["name"] == "browser_probe"
    assert response.tool_calls[0]["arguments"]["expect_text"] == "ALL TESTS PASSED"
    # Recovered calls carry no warning.
    assert not (response.metadata or {}).get("warnings")


def test_plain_name_still_parses():
    payload = INCIDENT_PAYLOAD.replace("functions.browser_probe", "browser_probe")
    response = _passthrough(payload)
    assert response.tool_calls and response.tool_calls[0]["name"] == "browser_probe"


def test_unknown_dotted_name_warns_and_names_the_call():
    payload = INCIDENT_PAYLOAD.replace("functions.browser_probe", "acme.no_such_tool")
    response = _passthrough(payload)
    assert not response.tool_calls
    warnings = (response.metadata or {}).get("warnings") or []
    assert warnings, "dropping every parsed call must leave a warning"
    assert "acme.no_such_tool" in warnings[0]
    # When no call was recovered the original content must survive: stripping
    # the syntax here would leave an EMPTY final answer on top of the lost call.
    assert response.content == payload


def test_malformed_block_warns():
    # A name the parser cannot accept even with the widened charset.
    payload = "<tool_call>\n<function=9bad name>\n</function>\n</tool_call>"
    response = _passthrough(payload)
    assert not response.tool_calls
    warnings = (response.metadata or {}).get("warnings") or []
    assert warnings, "unparseable tool-call syntax must leave a warning"


def test_prose_without_tool_syntax_gets_no_warning():
    response = _passthrough("The capital of Australia is Canberra.")
    assert not response.tool_calls
    assert not (response.metadata or {}).get("warnings")


def test_bare_dotted_tag_is_not_a_tool_call_even_wrapped():
    # The permissive charset applies ONLY inside `<function=...>`; the bare-tag
    # recovery keeps the strict pattern so `<ns.tag>` stays unparsed — INSIDE a
    # wrapper (where direct_open_re actually runs) as well as in plain prose.
    wrapped = (
        "<tool_call>\n<web.search>\n<parameter=query>\nhello\n</parameter>\n"
        "</web.search>\n</tool_call>"
    )
    assert parse_tool_calls(wrapped, MODEL) == []
    # ...but it IS unmistakable tool intent, so the warning machinery must see it
    # (this exact shape silently ended a run pre-fix).
    assert detect_unparsed_tool_intent(wrapped) is True
    response = _passthrough(wrapped)
    assert not response.tool_calls
    assert (response.metadata or {}).get("warnings")

    prose = "Use <config.option>value</config.option> in your settings file."
    assert parse_tool_calls(prose, MODEL) == []
    assert detect_unparsed_tool_intent(prose) is False


def test_detect_unparsed_tool_intent_shapes():
    assert detect_unparsed_tool_intent("<tool_call>\n<function=9bad>\n</tool_call>") is True
    assert detect_unparsed_tool_intent('<tool_call>{"name": "x", "arguments": {}}</tool_call>') is True
    # Wrapper-less qwen3_coder block: function + parameter = intent.
    assert detect_unparsed_tool_intent(
        "<function=functions.browser_probe>\n<parameter=target>\nx\n</parameter>\n</function>"
    ) is True
    # A bare mention of either marker alone stays prose.
    assert detect_unparsed_tool_intent("just prose mentioning <function=") is False
    assert detect_unparsed_tool_intent("docs mention <parameter=key> blocks") is False
    assert detect_unparsed_tool_intent("no syntax at all") is False


def test_namespaced_mapper_prefers_whole_name_readings():
    from abstractcore.tools.wire_naming import map_namespaced_tool_name

    both = {"probe", "browser_probe"}
    # Whole-name dots->underscores outranks the shorter token match.
    assert map_namespaced_tool_name("browser.probe", both) == "browser_probe"
    # Namespace stripping still recovers the short name when that IS the intent.
    assert map_namespaced_tool_name("functions.probe", both) == "probe"
    assert map_namespaced_tool_name("functions.browser_probe", both) == "browser_probe"
    # Only-short-name roster: stripping is the best faithful reading.
    assert map_namespaced_tool_name("browser.probe", {"probe"}) == "probe"
    assert map_namespaced_tool_name("acme.nothing", {"probe"}) is None


def test_streaming_lane_maps_namespaced_name():
    from abstractcore.providers.streaming import UnifiedStreamProcessor

    def _chunks(text, size=17):
        for i in range(0, len(text), size):
            yield GenerateResponse(content=text[i:i + size], model=MODEL)

    processor = UnifiedStreamProcessor(model_name=MODEL)
    out = list(processor.process_stream(_chunks(INCIDENT_PAYLOAD), converted_tools=TOOLS))
    calls = [c for chunk in out for c in (chunk.tool_calls or [])]
    assert calls, "streaming must surface the structured tool call"
    assert calls[0]["name"] == "browser_probe"


def test_streaming_lane_warns_on_unknown_name():
    from abstractcore.providers.streaming import UnifiedStreamProcessor

    payload = INCIDENT_PAYLOAD.replace("functions.browser_probe", "acme.no_such_tool")

    def _chunks(text, size=23):
        for i in range(0, len(text), size):
            yield GenerateResponse(content=text[i:i + size], model=MODEL)

    processor = UnifiedStreamProcessor(model_name=MODEL)
    out = list(processor.process_stream(_chunks(payload), converted_tools=TOOLS))
    calls = [c for chunk in out for c in (chunk.tool_calls or [])]
    assert calls and calls[0]["name"] == "acme.no_such_tool"  # verbatim, host decides
    warnings = [w for chunk in out for w in ((chunk.metadata or {}).get("warnings") or [])]
    assert any("acme.no_such_tool" in w for w in warnings)
