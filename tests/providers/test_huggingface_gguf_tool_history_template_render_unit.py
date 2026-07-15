"""HF-GGUF lane: tool-call history template rendering (root cause ≠ LM Studio's).

Diagnosed from scratch (2026-07-15, no 21GB weights needed): llama-cpp-python
renders GGUF embedded chat templates through
``jinja2.sandbox.ImmutableSandboxedEnvironment`` (see
``llama_cpp.llama_chat_format.Jinja2ChatFormatter``), which DOES implement the
``safe`` filter — so LM Studio's minja failure mode does not exist here.

The HF failure is upstream of ``safe``: the OpenAI wire convention carries
``tool_calls[].function.arguments`` as a JSON STRING (exactly what callers
replay and what ``create_chat_completion`` emits), while the Ornith/Qwen3-Coder
template iterates ``tool_call.arguments|items``. Jinja's ``items`` filter on a
string raises ``TypeError: Can only get item pairs from a mapping.`` — the
whole request dies at render on the second ReAct cycle, 100% of tool loops.

The fix is the provider's fallback-lane bridge
``HuggingFaceProvider._gguf_normalize_tool_call_arguments_for_template``
(JSON-string arguments parsed to dicts before ``create_chat_completion``).
These tests pin the failure AND the fix against the REAL Ornith template in
the exact jinja2 environment llama-cpp-python uses.
"""
from __future__ import annotations

import json
from pathlib import Path

import jinja2
import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

from abstractcore.providers.huggingface_provider import HuggingFaceProvider

FIXTURE_TEMPLATE = Path(__file__).resolve().parent.parent / "fixtures" / "ornith_chat_template.jinja"


def _llama_cpp_parity_template():
    """The environment llama_cpp.llama_chat_format.Jinja2ChatFormatter builds."""
    env = ImmutableSandboxedEnvironment(loader=jinja2.BaseLoader(), trim_blocks=True, lstrip_blocks=True)
    return env.from_string(FIXTURE_TEMPLATE.read_text())


def _render(template, messages):
    def raise_exception(message):
        raise ValueError(message)

    return template.render(
        messages=messages,
        eos_token="<|im_end|>",
        bos_token="",
        raise_exception=raise_exception,
        add_generation_prompt=True,
        functions=None,
        function_call=None,
        tools=None,
        tool_choice=None,
    )


def _tool_turn_conversation(arguments):
    """Shaped exactly as HuggingFaceProvider._generate_gguf passes messages to
    create_chat_completion: merged single system turn, then replayed history."""
    return [
        {"role": "system", "content": "You are helpful.\n\n## Tools (session)\n- web_search(query, num_results)"},
        {"role": "user", "content": "what are the news today"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "id": "call_1", "function": {"name": "web_search", "arguments": arguments}}
            ],
        },
        {"role": "tool", "content": "1. Example headline: markets rally."},
    ]


def test_wire_string_arguments_crash_the_real_template_in_llama_cpp_env():
    """Pins the HF-GGUF root cause: `arguments|items` on a JSON STRING."""
    template = _llama_cpp_parity_template()
    messages = _tool_turn_conversation(json.dumps({"query": "news", "num_results": 10}))

    with pytest.raises(TypeError, match="Can only get item pairs from a mapping"):
        _render(template, messages)


def test_safe_filter_exists_in_llama_cpp_env_so_int_args_render_after_normalization():
    """Pins that HF needs NO `safe` workaround: dict args with an int value render
    fine through `tojson | safe` in real Jinja2 (unlike LM Studio's minja)."""
    template = _llama_cpp_parity_template()
    messages = _tool_turn_conversation({"query": "news", "num_results": 10})

    rendered = _render(template, messages)
    assert "<parameter=num_results>\n10\n</parameter>" in rendered
    assert "<function=web_search>" in rendered


def test_provider_normalizer_bridges_wire_string_to_dict_and_render_succeeds():
    """End-to-end at the render layer: the exact provider bridge applied to the
    exact wire shape makes the exact template render."""
    messages = _tool_turn_conversation(json.dumps({"query": "news", "num_results": 10}))

    normalized = HuggingFaceProvider._gguf_normalize_tool_call_arguments_for_template(messages)

    args = normalized[2]["tool_calls"][0]["function"]["arguments"]
    assert args == {"query": "news", "num_results": 10}  # parsed, types preserved

    rendered = _render(_llama_cpp_parity_template(), normalized)
    assert "<parameter=query>\nnews\n</parameter>" in rendered
    assert "<parameter=num_results>\n10\n</parameter>" in rendered
    # Tool result rides as a <tool_response> user turn per the template.
    assert "<tool_response>" in rendered


def test_provider_normalizer_leaves_non_json_and_dict_shapes_untouched():
    messages = [
        {"role": "assistant", "tool_calls": [{"function": {"name": "a", "arguments": "not json"}}]},
        {"role": "assistant", "tool_calls": [{"function": {"name": "b", "arguments": "[1, 2]"}}]},
        {"role": "assistant", "tool_calls": [{"function": {"name": "c", "arguments": {"k": 1}}}]},
        # canonical no-wrapper replay shape
        {"role": "assistant", "tool_calls": [{"name": "d", "arguments": json.dumps({"n": 2}), "call_id": "x"}]},
    ]

    normalized = HuggingFaceProvider._gguf_normalize_tool_call_arguments_for_template(messages)

    assert normalized[0]["tool_calls"][0]["function"]["arguments"] == "not json"
    assert normalized[1]["tool_calls"][0]["function"]["arguments"] == "[1, 2]"  # non-object stays
    assert normalized[2]["tool_calls"][0]["function"]["arguments"] == {"k": 1}
    assert normalized[3]["tool_calls"][0]["arguments"] == {"n": 2}


def test_lmstudio_stringified_values_also_render_fine_on_the_hf_lane():
    """Cross-lane compatibility: a conversation already carrying the LM Studio
    lane's pre-stringified values renders identically on HF (string branch)."""
    template = _llama_cpp_parity_template()
    typed = _render(template, _tool_turn_conversation({"query": "news", "num_results": 10}))
    stringified = _render(template, _tool_turn_conversation({"query": "news", "num_results": "10"}))
    assert typed == stringified
