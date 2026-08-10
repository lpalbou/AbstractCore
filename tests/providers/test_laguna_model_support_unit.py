from abstractcore.architectures import detect_architecture, get_architecture_format, get_model_capabilities
from abstractcore.architectures.detection import format_messages
from abstractcore.architectures.response_postprocessing import normalize_assistant_text
from abstractcore.providers.streaming import IncrementalToolDetector
from abstractcore.tools.parser import parse_tool_calls

MODEL = "poolside/Laguna-S-2.1"


def test_laguna_architecture_detected_for_common_variants() -> None:
    variants = [
        "poolside/Laguna-S-2.1",
        "Laguna-S-2.1",
        "laguna-s-2.1",
        "models--poolside--Laguna-S-2.1",
        "poolside/Laguna-S-2.1-FP8",
        "poolside/Laguna-S-2.1-NVFP4",
        "poolside/Laguna-S-2.1-INT4",
        "poolside/Laguna-S-2.1-GGUF",
        "laguna-s-2.1:q4_k_m",
    ]

    for model in variants:
        assert detect_architecture(model) == "laguna", model
        caps = get_model_capabilities(model)
        assert caps.get("architecture") == "laguna", model
        assert caps.get("max_tokens") == 1048576, model
        assert caps.get("tool_support") == "native", model
        assert caps.get("thinking_support") is True, model


def test_laguna_siblings_inherit_architecture_tool_support() -> None:
    # Only Laguna-S-2.1 is in the capability registry; the rest of the family must
    # still resolve to the laguna architecture and its native tool support default.
    for model in ("poolside/Laguna-XS-2.1", "poolside/Laguna-XS.2", "poolside/Laguna-M.1"):
        assert detect_architecture(model) == "laguna", model
        assert get_model_capabilities(model).get("tool_support") == "native", model


def test_laguna_transcript_matches_chat_template_tags() -> None:
    rendered = format_messages(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ],
        "laguna",
    )
    assert rendered == "<system>S</system>\n<user>U</user>\n<assistant>A</assistant>\n"

    arch_fmt = get_architecture_format("laguna")
    assert arch_fmt["output_wrappers"]["end"] == "</assistant>"
    assert arch_fmt["thinking_control"]["template_kwarg"] == "enable_thinking"


def test_laguna_thinking_tags_are_stripped_and_reasoning_returned() -> None:
    arch_fmt = get_architecture_format(detect_architecture(MODEL))
    caps = get_model_capabilities(MODEL)

    cleaned, reasoning = normalize_assistant_text(
        "<think>weighing options</think>Final answer.",
        architecture_format=arch_fmt,
        model_capabilities=caps,
    )
    assert cleaned == "Final answer."
    assert reasoning == "weighing options"


def test_laguna_arg_key_value_tool_calls_are_parsed() -> None:
    calls = parse_tool_calls(
        "<tool_call>read_file<arg_key>path</arg_key><arg_value>src/main.py</arg_value></tool_call>",
        MODEL,
    )
    assert [(c.name, c.arguments) for c in calls] == [("read_file", {"path": "src/main.py"})]


def test_laguna_tool_call_values_keep_emitter_types() -> None:
    # The chat template renders strings verbatim and JSON-encodes everything else.
    calls = parse_tool_calls(
        "<tool_call>search\n"
        "<arg_key>query</arg_key>\n<arg_value>moe routing</arg_value>\n"
        "<arg_key>limit</arg_key>\n<arg_value>5</arg_value>\n"
        "<arg_key>strict</arg_key>\n<arg_value>true</arg_value>\n"
        "</tool_call>",
        MODEL,
    )
    assert len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments == {"query": "moe routing", "limit": 5, "strict": True}


def test_laguna_parallel_tool_calls_are_parsed() -> None:
    calls = parse_tool_calls(
        "<tool_call>a<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
        "<tool_call>b<arg_key>y</arg_key><arg_value>hi</arg_value></tool_call>",
        MODEL,
    )
    assert [(c.name, c.arguments) for c in calls] == [("a", {"x": 1}), ("b", {"y": "hi"})]


def _stream(text: str, chunk_size: int):
    detector = IncrementalToolDetector(model_name=MODEL)
    streamed, calls = "", []
    for index in range(0, len(text), chunk_size):
        content, tool_calls = detector.process_chunk(text[index:index + chunk_size])
        streamed += content
        calls.extend(tool_calls)
    calls.extend(detector.finalize() or [])
    return streamed, calls


def test_laguna_streaming_extracts_tool_call_and_keeps_prose() -> None:
    streamed, calls = _stream(
        "<think>read it</think>Reading now."
        "<tool_call>read_file<arg_key>path</arg_key><arg_value>src/main.py</arg_value></tool_call>",
        7,
    )
    assert streamed == "<think>read it</think>Reading now."
    assert [(c.name, c.arguments) for c in calls] == [("read_file", {"path": "src/main.py"})]


def test_laguna_streaming_recovers_truncated_tool_call() -> None:
    _, calls = _stream("<tool_call>search<arg_key>q</arg_key><arg_value>moe</arg_value>", 9)
    assert [(c.name, c.arguments) for c in calls] == [("search", {"q": "moe"})]


def test_laguna_calls_recovered_when_server_returns_empty_tool_calls() -> None:
    """The production failure: a server that leaves tool syntax in `content`.

    When no `tools` field reaches llama.cpp it applies no tool grammar, so it
    answers `tool_calls: []` with the Laguna syntax sitting in `content`.
    AbstractCore's documented passthrough contract is to recover those calls.
    """
    from abstractcore.core.interface import GenerateResponse
    from abstractcore.providers.base import BaseProvider
    from abstractcore.tools.handler import create_handler

    class _Probe(BaseProvider):
        def _generate_internal(self, *args, **kwargs):
            raise NotImplementedError

        def get_capabilities(self):
            return {}

        def list_available_models(self):
            return []

        def unload_model(self):
            return None

    provider = _Probe(model=MODEL)
    provider.tool_handler = create_handler(MODEL)

    content = (
        "Let me read the identity files."
        "<tool_call>read_file<arg_key>file_path</arg_key>"
        "<arg_value>Core/Self_Model.md</arg_value>"
        "<arg_key>start_line</arg_key><arg_value>1</arg_value></tool_call>"
        "<tool_call>read_file<arg_key>file_path</arg_key>"
        "<arg_value>Core/Values.md</arg_value>"
        "<arg_key>start_line</arg_key><arg_value>1</arg_value></tool_call>"
    )
    tools = [{
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "start_line": {"type": "integer"},
                },
            },
        },
    }]

    normalized = provider._normalize_tool_calls_passthrough(
        response=GenerateResponse(
            content=content, model=MODEL, finish_reason="stop", tool_calls=[]
        ),
        tools=tools,
    )

    recovered = normalized.tool_calls or []
    assert len(recovered) == 2
    assert [c["name"] for c in recovered] == ["read_file", "read_file"]
    assert recovered[0]["arguments"] == {"file_path": "Core/Self_Model.md", "start_line": 1}
    # The markup must not leak into user-visible content.
    assert "<tool_call>" not in (normalized.content or "")
    assert "<arg_key>" not in (normalized.content or "")
