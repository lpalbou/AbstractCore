"""Pin: GGUF tool-call history must not crash dict-expecting embedded chat
templates on replay.

Regression guard for the 2026-07-15 live crash on Ornith-1.0-35B GGUF
(`Error: Can only get item pairs from a mapping.`): the OpenAI/llama-cpp wire
contract carries `tool_calls[].function.arguments` as a JSON STRING, but
embedded GGUF chat templates (Qwen-Agent / Hermes / Ornith family) iterate it
with `arguments|items`, and Jinja's `items` filter raises on a string. The
provider now parses string arguments to a dict for the create_chat_completion
fallback lane. These tests pin the pure normalization contract (no model load).
"""

from __future__ import annotations

import copy

from abstractcore.providers.huggingface_provider import HuggingFaceProvider

_norm = HuggingFaceProvider._gguf_normalize_tool_call_arguments_for_template


def test_json_string_arguments_become_dict() -> None:
    msgs = [
        {"role": "user", "content": "list files"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "list_files", "arguments": '{"path": ".", "pattern": "*.py"}'},
                }
            ],
        },
    ]
    out = _norm(copy.deepcopy(msgs))
    args = out[1]["tool_calls"][0]["function"]["arguments"]
    assert args == {"path": ".", "pattern": "*.py"}  # dict now, template-safe


def test_already_dict_arguments_untouched() -> None:
    msgs = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "x", "arguments": {"a": 1}}}
            ],
        }
    ]
    out = _norm(copy.deepcopy(msgs))
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {"a": 1}


def test_unparseable_string_left_as_is() -> None:
    """A non-JSON arguments string is NOT coerced — a template that genuinely
    wants a string still gets one, and we never throw during normalization."""
    msgs = [{"role": "assistant", "tool_calls": [{"function": {"name": "x", "arguments": "not-json"}}]}]
    out = _norm(copy.deepcopy(msgs))
    assert out[0]["tool_calls"][0]["function"]["arguments"] == "not-json"


def test_json_scalar_string_not_object_left_as_is() -> None:
    """`arguments` that parses to a non-object (e.g. a bare JSON string/number)
    must stay a string — only dict objects are the `|items` target."""
    msgs = [{"role": "assistant", "tool_calls": [{"function": {"name": "x", "arguments": '"hello"'}}]}]
    out = _norm(copy.deepcopy(msgs))
    assert out[0]["tool_calls"][0]["function"]["arguments"] == '"hello"'


def test_top_level_arguments_shape_also_coerced() -> None:
    """Some tool-call shapes carry name/arguments at the top level (no nested
    `function`)."""
    msgs = [{"role": "assistant", "tool_calls": [{"name": "x", "arguments": '{"k": "v"}'}]}]
    out = _norm(copy.deepcopy(msgs))
    assert out[0]["tool_calls"][0]["arguments"] == {"k": "v"}


def test_legacy_function_call_coerced() -> None:
    msgs = [{"role": "assistant", "function_call": {"name": "x", "arguments": '{"k": 1}'}}]
    out = _norm(copy.deepcopy(msgs))
    assert out[0]["function_call"]["arguments"] == {"k": 1}


def test_non_list_and_malformed_messages_never_raise() -> None:
    assert _norm(None) is None
    assert _norm("nonsense") == "nonsense"
    # message without tool_calls, and non-dict entries, pass through untouched
    weird = [{"role": "user", "content": "hi"}, "junk", 42]
    assert _norm(copy.deepcopy(weird)) == weird


def test_ornith_template_renders_after_normalization() -> None:
    """End-to-end against the exact failing template shape: a Jinja template
    that iterates `arguments|items` crashes on a string and renders after
    normalization. Uses a minimal ChatML-ish template (no model load)."""
    jinja2 = __import__("jinja2")
    tmpl_src = (
        "{% for m in messages %}"
        "{% if m.tool_calls %}{% for tc in m.tool_calls %}"
        "<function={{ tc.function.name }}>"
        "{% for k, v in tc.function.arguments|items %}<p={{k}}>{{v}}</p>{% endfor %}"
        "</function>{% endfor %}{% else %}{{ m.content }}{% endif %}"
        "{% endfor %}"
    )
    env = jinja2.Environment()
    tmpl = env.from_string(tmpl_src)
    hist = [
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "list_files", "arguments": '{"path": "."}'}}
            ],
        }
    ]
    # BEFORE: string arguments crash the |items filter.
    import pytest

    with pytest.raises(Exception):
        tmpl.render(messages=hist)
    # AFTER: normalization makes it render.
    out = tmpl.render(messages=_norm(copy.deepcopy(hist)))
    assert out == "<function=list_files><p=path>.</p></function>"
