"""LM Studio tool-call HISTORY argument stringification (minja `safe`-filter fix).

Root cause (live-proven on the operator's LM Studio, Ornith-1.0-35B GGUF,
2026-07-15): GGUF chat templates of the Qwen3-Coder XML convention render
replayed assistant tool-call arguments per entry as

    {%- set args_value = args_value | string if args_value is string
                         else args_value | tojson | safe %}

LM Studio's template engine (minja-class) lacks the ``safe`` filter, so the
first request whose conversation carries a tool call with any NON-STRING
argument value (e.g. ``{"query": "news", "num_results": 10}``) fails template
rendering with HTTP 400 ``Unknown StringValue filter: safe`` — cycle 2 of every
ReAct loop.

The fix (`stringify_tool_call_history_argument_values`, applied via
`LMStudioProvider._mutate_payload`) JSON-stringifies each non-string argument
VALUE so the template takes its ``| string`` branch, which renders the exact
bytes ``| tojson`` would have produced (byte-equivalence pinned below against
the REAL Ornith template). LM Studio-only: OpenAI/vLLM-class servers render
with real Jinja2 (which has ``safe``) and must keep the untouched shared wire.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import httpx
import jinja2
import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.lmstudio_provider import (
    LMStudioProvider,
    stringify_tool_call_history_argument_values,
)
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

FIXTURE_TEMPLATE = Path(__file__).resolve().parent.parent / "fixtures" / "ornith_chat_template.jinja"


def _assistant_history(arguments):
    return [
        {"role": "user", "content": "what are the news today"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "id": "call_1",
                    "function": {"name": "web_search", "arguments": arguments},
                }
            ],
        },
        {"role": "tool", "content": "1. Example headline.", "tool_call_id": "call_1"},
    ]


# ---------------------------------------------------------------------------
# Transform unit behavior
# ---------------------------------------------------------------------------


def test_wire_json_string_arguments_values_stringified_container_stays_string():
    history = _assistant_history(json.dumps({"query": "news", "num_results": 10}))
    out = stringify_tool_call_history_argument_values(history)

    args_raw = out[1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args_raw, str)
    assert json.loads(args_raw) == {"query": "news", "num_results": "10"}


def test_dict_arguments_values_stringified_container_stays_dict():
    history = _assistant_history(
        {
            "s": "keep me",
            "i": 10,
            "f": 1.5,
            "t": True,
            "none": None,
            "d": {"a": 1, "nested": {"b": [1, "two"]}},
            "l": [1, "two", {"x": 3}],
        }
    )
    out = stringify_tool_call_history_argument_values(history)

    args = out[1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, dict)
    assert args["s"] == "keep me"  # strings untouched
    assert args["i"] == "10"
    assert args["f"] == "1.5"
    assert args["t"] == "true"
    assert args["none"] == "null"
    assert args["d"] == json.dumps({"a": 1, "nested": {"b": [1, "two"]}}, ensure_ascii=False)
    assert args["l"] == json.dumps([1, "two", {"x": 3}], ensure_ascii=False)
    # keys stay
    assert set(args.keys()) == {"s", "i", "f", "t", "none", "d", "l"}


def test_all_string_arguments_are_a_no_op_returning_same_list():
    history = _assistant_history(json.dumps({"query": "news"}))
    assert stringify_tool_call_history_argument_values(history) is history


def test_non_json_and_non_object_arguments_pass_through():
    for raw in ("not json at all", "[1, 2, 3]", "42", "", None):
        history = _assistant_history(raw)
        out = stringify_tool_call_history_argument_values(history)
        assert out[1]["tool_calls"][0]["function"]["arguments"] == raw


def test_caller_history_is_never_mutated_and_untouched_messages_keep_identity():
    history = _assistant_history({"query": "news", "num_results": 10})
    snapshot = copy.deepcopy(history)

    out = stringify_tool_call_history_argument_values(history)

    assert history == snapshot  # copy-on-write: caller-owned session history intact
    assert out is not history
    assert out[0] is history[0]  # user message untouched by identity
    assert out[2] is history[2]  # tool message untouched by identity
    assert out[1] is not history[1]
    assert out[1]["tool_calls"][0]["function"]["arguments"]["num_results"] == "10"


def test_canonical_no_wrapper_shape_and_legacy_function_call_are_handled():
    history = [
        {
            "role": "assistant",
            "content": "",
            # canonical AbstractCore replay shape (no "function" wrapper)
            "tool_calls": [{"name": "web_search", "arguments": {"num_results": 10}, "call_id": "c1"}],
            # legacy single-call field
            "function_call": {"name": "web_search", "arguments": json.dumps({"limit": 5})},
        }
    ]
    out = stringify_tool_call_history_argument_values(history)

    assert out[0]["tool_calls"][0]["arguments"] == {"num_results": "10"}
    assert out[0]["tool_calls"][0]["call_id"] == "c1"
    assert json.loads(out[0]["function_call"]["arguments"]) == {"limit": "5"}


def test_non_assistant_roles_are_never_touched():
    history = [
        {"role": "user", "content": "x", "tool_calls": [{"function": {"name": "t", "arguments": {"n": 1}}}]},
        {"role": "tool", "content": json.dumps({"n": 1}), "tool_call_id": "c"},
    ]
    out = stringify_tool_call_history_argument_values(history)
    assert out is history  # nothing changed -> same object
    assert history[0]["tool_calls"][0]["function"]["arguments"] == {"n": 1}


# ---------------------------------------------------------------------------
# Byte-equivalence property against the REAL Ornith GGUF chat template
# ---------------------------------------------------------------------------


def _raise_exception(message):
    raise ValueError(message)


def _render(template, messages):
    return template.render(
        messages=messages,
        eos_token="<|im_end|>",
        bos_token="",
        raise_exception=_raise_exception,
        add_generation_prompt=True,
        functions=None,
        function_call=None,
        tools=None,
        tool_choice=None,
    )


def _conversation(args_dict):
    return [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "what are the news today"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "id": "call_1", "function": {"name": "web_search", "arguments": args_dict}}],
        },
        {"role": "tool", "content": "result text"},
    ]


def _stringified(args_dict):
    return {k: (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for k, v in args_dict.items()}


def test_byte_equivalence_real_template_transformers_tojson_parity():
    """Full adversarial payload under the training-time rendering convention.

    HF transformers overrides Jinja's ``tojson`` with
    ``json.dumps(x, ensure_ascii=False, sort_keys=False)`` (no HTML escaping) —
    that is what Qwen3-Coder-convention models saw during training and exactly
    the convention `stringify_tool_call_history_argument_values` emits. Under
    it, original-typed args via the template's tojson branch and pre-stringified
    args via its string branch must render byte-identically.
    """
    env = ImmutableSandboxedEnvironment(loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
    )
    template = env.from_string(FIXTURE_TEMPLATE.read_text())

    args = {
        "query": "keep <this> & 'quotes' héllo",  # string branch: untouched either way
        "num_results": 10,
        "threshold": 0.25,
        "strict": False,
        "nothing": None,
        "unsorted": {"zeta": 1, "alpha": {"deep": [1, "two", None]}},
        "mixed_list": [1, "two", {"z": 9, "a": "<&>"}],
        "unicode": {"clé": "héllo"},
    }

    rendered_typed = _render(template, _conversation(dict(args)))
    rendered_stringified = _render(template, _conversation(_stringified(args)))

    assert rendered_typed == rendered_stringified
    assert "<parameter=num_results>\n10\n</parameter>" in rendered_typed


def test_byte_equivalence_real_template_llama_cpp_default_env():
    """llama-cpp-python parity env (jinja2 defaults: tojson sorts keys, HTML-escapes).

    Payload constrained to the intersection where the default-jinja2 tojson and
    ``json.dumps(v, ensure_ascii=False)`` agree byte-for-byte: ASCII text without
    HTML-escapable characters and dicts whose insertion order is already sorted.
    (The transformers-parity test above carries the adversarial payload.)
    """
    env = ImmutableSandboxedEnvironment(loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True)
    template = env.from_string(FIXTURE_TEMPLATE.read_text())

    args = {
        "query": "plain text",
        "num_results": 10,
        "ratio": 2.5,
        "flag": True,
        "empty": None,
        "ids": [1, 2, 3],
        "sorted_dict": {"a": 1, "b": [2, "three"]},
    }

    rendered_typed = _render(template, _conversation(dict(args)))
    rendered_stringified = _render(template, _conversation(_stringified(args)))

    assert rendered_typed == rendered_stringified
    assert "<parameter=flag>\ntrue\n</parameter>" in rendered_typed


# ---------------------------------------------------------------------------
# Wire payload: LM Studio transforms; shared OpenAI-compatible wire untouched;
# live RESPONSE tool_calls parsing untouched.
# ---------------------------------------------------------------------------


def _lmstudio(monkeypatch) -> LMStudioProvider:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    return LMStudioProvider(model="ornith-1.0-35b", base_url="http://127.0.0.1:9/v1")


def test_lmstudio_payload_history_args_stringified_and_response_tool_calls_untouched(monkeypatch):
    provider = _lmstudio(monkeypatch)

    captured = {}

    def _fake_post(url, json=None, headers=None, **kwargs):
        captured["url"] = url
        captured["payload"] = json
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            # LIVE tool call in the RESPONSE: int must stay int after parsing.
                            "tool_calls": [
                                {
                                    "id": "call_live",
                                    "type": "function",
                                    "function": {
                                        "name": "web_search",
                                        "arguments": "{\"query\": \"news\", \"num_results\": 10}",
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(provider.client, "post", _fake_post)

    history = _assistant_history(json.dumps({"query": "news", "num_results": 10}))
    response = provider.generate(
        "Summarize.",
        messages=history,
        tools=[{"name": "web_search", "description": "d", "parameters": {"query": {"type": "string"}}}],
        max_output_tokens=32,
    )

    # Request side: history tool_call arguments carry stringified VALUES on the wire.
    wire_messages = captured["payload"]["messages"]
    wire_assistant = [m for m in wire_messages if m.get("role") == "assistant" and m.get("tool_calls")]
    assert len(wire_assistant) == 1
    wire_args = wire_assistant[0]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(wire_args) == {"query": "news", "num_results": "10"}

    # Caller-owned history is never mutated.
    assert json.loads(history[1]["tool_calls"][0]["function"]["arguments"]) == {"query": "news", "num_results": 10}

    # Response side: live tool_calls parsing keeps typed arguments (int stays int).
    assert response.tool_calls, "expected live tool_calls on the response"
    live_args = response.tool_calls[0]["arguments"]
    if isinstance(live_args, str):  # tolerate raw passthrough shape
        live_args = json.loads(live_args)
    assert live_args["num_results"] == 10
    assert not isinstance(live_args["num_results"], str)


def test_shared_openai_compatible_wire_is_not_transformed(monkeypatch):
    """vLLM/OpenAI-class servers render with real Jinja2 (has `safe`): shared wire stays typed."""
    provider = OpenAICompatibleProvider(
        model="test-model", base_url="http://127.0.0.1:9/v1", api_key="x", validate_model=False
    )

    captured = {}

    def _single(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _single)

    history = _assistant_history(json.dumps({"query": "news", "num_results": 10}))
    provider._generate_internal(prompt="Summarize.", messages=history, stream=False)

    wire_assistant = [m for m in captured["payload"]["messages"] if m.get("role") == "assistant" and m.get("tool_calls")]
    assert json.loads(wire_assistant[0]["tool_calls"][0]["function"]["arguments"]) == {
        "query": "news",
        "num_results": 10,
    }


def test_lmstudio_stream_payload_also_transformed(monkeypatch):
    provider = _lmstudio(monkeypatch)

    captured = {}

    def _stream(payload):
        captured["payload"] = payload
        yield GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_stream_generate", _stream)

    history = _assistant_history(json.dumps({"query": "news", "num_results": 10}))
    chunks = provider._generate_internal(prompt="Summarize.", messages=history, stream=True)
    list(chunks)

    wire_assistant = [m for m in captured["payload"]["messages"] if m.get("role") == "assistant" and m.get("tool_calls")]
    assert json.loads(wire_assistant[0]["tool_calls"][0]["function"]["arguments"]) == {
        "query": "news",
        "num_results": "10",
    }
