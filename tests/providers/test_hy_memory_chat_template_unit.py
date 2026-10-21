"""Hy-Memory-30B-A3B: abstractcore's assumptions checked against the model's OWN chat template.

Renders tencent/Hy-Memory-30B-A3B chat_template.jinja from the operator's local
Hugging Face cache (read-only). Skipped, never passed, when it is absent.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List, Optional

import pytest

from abstractcore.tools.parser import parse_tool_calls

pytestmark = pytest.mark.real_home(
    "reads tencent/Hy-Memory-30B-A3B chat_template.jinja from the operator's Hugging Face cache (read-only)"
)

MODEL = "tencent/Hy-Memory-30B-A3B"
ASSISTANT = "<｜hy_Assistant｜>"


def _template() -> Optional[str]:
    home = os.environ.get("ABSTRACT_TEST_REAL_HOME") or os.path.expanduser("~")
    root = os.path.join(home, ".cache/huggingface/hub/models--tencent--Hy-Memory-30B-A3B/snapshots")
    hits = sorted(glob.glob(os.path.join(root, "*", "chat_template.jinja")))
    return open(hits[0], encoding="utf-8").read() if hits else None


def _render(messages: List[Dict[str, Any]], **kwargs) -> str:
    tpl = _template()
    if tpl is None:
        pytest.skip("tencent/Hy-Memory-30B-A3B chat_template.jinja is not in the local HF cache")
    from transformers.utils.chat_template_utils import render_jinja_template

    out, _ = render_jinja_template(conversations=[messages], chat_template=tpl, **kwargs)
    return out[0]


def test_parser_round_trips_what_the_template_renders() -> None:
    calls = [
        {"type": "function", "function": {"name": "search", "arguments": {"query": "moe routing", "limit": 5, "strict": True}}},
        {"type": "function", "function": {"name": "list_memories", "arguments": {}}},
    ]
    rendered = _render([
        {"role": "user", "content": "U"},
        {"role": "assistant", "content": "", "tool_calls": calls},
    ])
    assistant_turn = rendered.split(ASSISTANT, 1)[1]
    parsed = parse_tool_calls(assistant_turn, MODEL)
    assert [(c.name, c.arguments) for c in parsed] == [
        ("search", {"query": "moe routing", "limit": 5, "strict": True}),
        ("list_memories", {}),
    ]


@pytest.mark.parametrize(
    "effort,gen_suffix,marker",
    [
        ("no_think", f"{ASSISTANT}<think></think>", "reasoning_effort:no_think"),
        ("high", f"{ASSISTANT}<think>", "reasoning_effort:high"),
        ("low", f"{ASSISTANT}<think>", "reasoning_effort:low"),
    ],
)
def test_effort_values_drive_the_templates_generation_prompt(effort, gen_suffix, marker) -> None:
    rendered = _render([{"role": "user", "content": "U"}], add_generation_prompt=True, reasoning_effort=effort)
    assert rendered.endswith(gen_suffix)
    assert marker in rendered


def test_unset_effort_renders_no_think() -> None:
    # Why `thinking="on"` must send a level: the template defaults to OFF.
    rendered = _render([{"role": "user", "content": "U"}], add_generation_prompt=True)
    assert rendered.endswith(f"{ASSISTANT}<think></think>")
    rendered = _render([{"role": "user", "content": "U"}], add_generation_prompt=True, enable_thinking=True)
    assert rendered.endswith(f"{ASSISTANT}<think></think>")


def test_transformers_lane_renders_a_tool_round_trip_through_the_template() -> None:
    """OpenAI-wire tool_calls (arguments as a JSON string) must reach the template as a
    dict: the Hy template iterates `arguments.items()`, and a raise there made the
    transformers lane fall back to a plain render WITHOUT reasoning_effort (live
    find, Hy-Memory BF16 tool round trip, 2026-10-21)."""
    import warnings

    tpl = _template()
    if tpl is None:
        pytest.skip("tencent/Hy-Memory-30B-A3B chat_template.jinja is not in the local HF cache")
    from abstractcore.architectures import get_architecture_format, get_model_capabilities
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    class _Tok:
        chat_template = tpl

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
            return _render(messages, add_generation_prompt=add_generation_prompt, **kwargs)

    p = HuggingFaceProvider.__new__(HuggingFaceProvider)
    p.model, p.model_type, p.architecture = MODEL, "transformers", "hy_v3"
    p.architecture_config = get_architecture_format("hy_v3")
    p.model_capabilities = get_model_capabilities(MODEL)
    p.tokenizer, p.tool_handler = _Tok(), None
    messages = [
        {"role": "user", "content": "Remember: I am allergic to peanuts."},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_0", "type": "function",
            "function": {"name": "save_memory", "arguments": '{"fact": "allergic to peanuts", "importance": 5}'},
        }]},
        {"role": "tool", "tool_call_id": "call_0", "content": '{"status": "saved"}'},
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the plain-format fallback warns
        text = p._build_input_text_transformers("", messages, None, reasoning_effort="no_think")
    assert "<tool_call>save_memory<tool_sep>\n<arg_key>fact</arg_key>\n<arg_value>allergic to peanuts</arg_value>" in text
    assert "<arg_key>importance</arg_key>\n<arg_value>5</arg_value>" in text
    assert "reasoning_effort:no_think" in text
    # the turn after a tool result is the assistant's, not an empty user turn
    assert text.endswith("</tool_responses><｜hy_Assistant｜><think></think>")
