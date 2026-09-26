"""The MLX prompt is rendered by the model's OWN chat template (2026-09-26).

Root cause pinned here (XP report, 2026-09-26): the hand-built MLX renderer
(1) dropped the `tool_calls` of earlier assistant turns, so every past turn
looked like one prose sentence and, at iteration 3, the model copied that shape
(one sentence, no call, the run "completed" with it); (2) sent tool results as
`<|im_start|>tool` instead of the template's `<tool_response>` inside a user
turn; (3) never ended the prompt with the template's `<think>\\n` opener.

Two layers:
- GOLDEN tests against the real `Jundot/Qwen3.8-27B-oQ4e-mtp` tokenizer, read
  offline from the local HF cache (skipped, never passed, when it is absent);
- hermetic tests with a small in-file Jinja template of the same shape, so the
  logic is covered on a box with no models.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from abstractcore.providers.mlx_provider import MLXProvider

QWEN38 = "Jundot/Qwen3.8-27B-oQ4e-mtp"


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


# The golden tokenizer is an INSTALLED model in the operator's real home, read
# offline and read-only; the test conftest moves HOME/HF_HOME to tmp and hands
# the real home over as ABSTRACT_TEST_REAL_HOME, which this marker declares.
pytestmark = pytest.mark.real_home("reads the installed Qwen3.8 tokenizer (HF cache) read-only")


def _hf_snapshot(repo: str) -> Optional[Path]:
    real_home = os.environ.get("ABSTRACT_TEST_REAL_HOME") or os.path.expanduser("~")
    base = Path(real_home) / ".cache" / "huggingface" / "hub" / ("models--" + repo.replace("/", "--"))
    ref = base / "refs" / "main"
    if ref.is_file():
        snap = base / "snapshots" / ref.read_text().strip()
        if (snap / "tokenizer_config.json").is_file():
            return snap
    for snap in sorted((base / "snapshots").glob("*")) if (base / "snapshots").is_dir() else []:
        if (snap / "tokenizer_config.json").is_file():
            return snap
    return None


@pytest.fixture(scope="module")
def qwen38_tokenizer():
    snap = _hf_snapshot(QWEN38)
    if snap is None:
        pytest.skip(f"{QWEN38} is not in the local HF cache (golden template tests need it)")
    transformers = pytest.importorskip("transformers")
    tok = transformers.AutoTokenizer.from_pretrained(str(snap), local_files_only=True)
    assert isinstance(tok.chat_template, str) and "<tool_response>" in tok.chat_template
    return tok


def _qwen38_provider(tokenizer: Any) -> MLXProvider:
    from abstractcore.architectures import detect_architecture, get_architecture_format
    from abstractcore.tools import UniversalToolHandler

    p = MLXProvider.__new__(MLXProvider)
    p.model = QWEN38
    p.tokenizer = tokenizer
    p.tool_handler = UniversalToolHandler(QWEN38)
    p.architecture = detect_architecture(QWEN38)
    p.architecture_config = get_architecture_format(p.architecture)
    p.model_capabilities = {}
    return p


TOOLS = [
    {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {
            "query": {"type": "string"},
            "num_results": {"type": "integer", "default": 5},
        },
    },
    {
        "name": "fetch_url",
        "description": "Fetch a URL.",
        "parameters": {
            "url": {"type": "string"},
            "include_full_content": {"type": "boolean", "default": True},
        },
    },
]

SYSTEM = "You are an autonomous ReAct agent."


def _call(cid: str, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    # Arguments as a JSON STRING: that is how the runtime sends them.
    return {"type": "function", "id": cid, "function": {"name": name, "arguments": json.dumps(args)}}


def _iterations(n: int) -> List[Dict[str, Any]]:
    """The first `n` iterations of a tool loop as the runtime sends them."""
    msgs: List[Dict[str, Any]] = [{"role": "user", "content": "Write a digest of this week's news."}]
    turns = [
        (
            "I'll research the week's developments.",
            [
                _call("call_1", "web_search", {"query": "world news this week", "num_results": "10"}),
                _call("call_2", "web_search", {"query": "markets this week"}),
            ],
            ['[web_search]: {"results": ["a", "b"]}', '[web_search]: {"results": ["c"]}'],
        ),
        (
            "Let me open the two strongest sources.",
            [
                _call("call_1", "fetch_url", {"url": "https://example.com/a", "include_full_content": False}),
                _call("call_2", "fetch_url", {"url": "https://example.com/b"}),
            ],
            ["[fetch_url]: page A", "[fetch_url]: page B"],
        ),
        (
            "",
            [_call("call_1", "web_search", {"query": "oil price today"})],
            ['[web_search]: {"results": ["oil"]}'],
        ),
    ]
    for i in range(n - 1):
        content, calls, results = turns[i]
        msgs.append({"role": "assistant", "content": content, "tool_calls": calls})
        for call, result in zip(calls, results):
            msgs.append({"role": "tool", "content": result, "tool_call_id": call["id"]})
        msgs.append({"role": "user", "content": f"[loop] iteration {i + 2} of 20."})
    return msgs


def _canonical(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """What the template expects: system first, dict arguments."""
    out: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM}]
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [
                {
                    "type": "function",
                    "id": c["id"],
                    "function": {
                        "name": c["function"]["name"],
                        "arguments": json.loads(c["function"]["arguments"]),
                    },
                }
                for c in m["tool_calls"]
            ]
        out.append(m)
    return out


# --------------------------------------------------------------------------
# GOLDEN: the real Qwen3.8 template
# --------------------------------------------------------------------------


def test_golden_three_iteration_tool_conversation_matches_apply_chat_template(qwen38_tokenizer):
    p = _qwen38_provider(qwen38_tokenizer)
    msgs = _iterations(3)
    rendered = p._build_prompt("", msgs, SYSTEM, TOOLS)
    expected = qwen38_tokenizer.apply_chat_template(
        _canonical(msgs),
        tools=p.tool_handler.format_tools_for_chat_template(TOOLS),
        tokenize=False,
        add_generation_prompt=True,
    )
    assert rendered == expected
    # The three divergences of the hand-built renderer, pinned by name:
    # (1) earlier assistant turns carry their calls
    assert "<tool_call>\n<function=web_search>\n<parameter=query>\nworld news this week\n</parameter>" in rendered
    assert "<tool_call>\n<function=fetch_url>\n<parameter=url>\nhttps://example.com/a\n</parameter>" in rendered
    # (2) tool results are <tool_response> blocks inside a user turn
    assert "<|im_start|>tool" not in rendered
    assert (
        "<|im_start|>user\n<tool_response>\n[fetch_url]: page A\n</tool_response>\n"
        "<tool_response>\n[fetch_url]: page B\n</tool_response><|im_end|>\n"
    ) in rendered
    # (3) the generation prompt opens thinking exactly as the template does
    assert rendered.endswith("<|im_start|>user\n[loop] iteration 3 of 20.<|im_end|>\n<|im_start|>assistant\n<think>\n")
    # and the prompt-opened detector sees it (reasoning split relies on it)
    assert p._prompt_opened_thinking(rendered) is True


def test_golden_thinking_disabled_matches_template(qwen38_tokenizer):
    p = _qwen38_provider(qwen38_tokenizer)
    msgs = _iterations(2)
    rendered = p._build_prompt("", msgs, SYSTEM, TOOLS, enable_thinking=False)
    expected = qwen38_tokenizer.apply_chat_template(
        _canonical(msgs),
        tools=p.tool_handler.format_tools_for_chat_template(TOOLS),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    assert rendered == expected
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "Reasoning effort is set to" not in rendered
    assert p._prompt_opened_thinking(rendered) is False


def test_golden_thinking_enabled_explicitly_equals_default(qwen38_tokenizer):
    p = _qwen38_provider(qwen38_tokenizer)
    msgs = _iterations(2)
    on = p._build_prompt("", msgs, SYSTEM, TOOLS, enable_thinking=True)
    default = p._build_prompt("", msgs, SYSTEM, TOOLS)
    assert on == default
    assert on.endswith("<|im_start|>assistant\n<think>\n")


def test_golden_reasoning_effort_reaches_the_template(qwen38_tokenizer):
    p = _qwen38_provider(qwen38_tokenizer)
    msgs = _iterations(1)
    low = p._build_prompt("", msgs, SYSTEM, None, enable_thinking=True, reasoning_effort="low")
    expected = qwen38_tokenizer.apply_chat_template(
        _canonical(msgs), tokenize=False, add_generation_prompt=True,
        enable_thinking=True, reasoning_effort="low",
    )
    assert low == expected
    assert low.startswith("<|im_start|>system\nReasoning effort is set to low.")
    medium = p._build_prompt("", msgs, SYSTEM, None, enable_thinking=True, reasoning_effort="medium")
    assert medium.startswith(f"<|im_start|>system\n{SYSTEM}<|im_end|>")


def test_golden_prefix_is_byte_stable_across_turns(qwen38_tokenizer):
    """Turn N's prompt (generation prompt included) is a prefix of turn N+1's.

    That is what keeps APC / prefix caches hitting across the turns of a session.
    """
    p = _qwen38_provider(qwen38_tokenizer)
    for thinking in (None, False):
        prev = None
        for n in (1, 2, 3):
            text = p._build_prompt("", _iterations(n), SYSTEM, TOOLS, enable_thinking=thinking)
            if prev is not None:
                assert text.startswith(prev), (thinking, n)
            prev = text
        # Token view. Everything before the generation prompt is a token prefix
        # of the next turn; inside the generation prompt the template's own
        # `<think>\n` becomes `<think>\n\n</think>` in history, and BPE merges
        # "\n" + "\n" into one token, so the LAST token may differ (the
        # generation-prompt holdback exists for exactly this seam).
        turn2 = p._build_prompt("", _iterations(2), SYSTEM, TOOLS, enable_thinking=thinking)
        turn3 = p._build_prompt("", _iterations(3), SYSTEM, TOOLS, enable_thinking=thinking)
        literal = next(t for t in p._generation_prompt_literals() if turn2.endswith(t))
        head = qwen38_tokenizer.encode(turn2[: -len(literal)])
        a = qwen38_tokenizer.encode(turn2)
        b = qwen38_tokenizer.encode(turn3)
        assert b[: len(head)] == head
        lcp = p._token_lcp_len(a, b)
        assert lcp >= len(a) - 1, (thinking, len(a), lcp)


def test_golden_stable_head_is_a_prefix_of_the_prompt(qwen38_tokenizer):
    p = _qwen38_provider(qwen38_tokenizer)
    msgs = _iterations(3)
    full = p._build_prompt("", msgs, SYSTEM, TOOLS)
    head = p._build_prompt_fragment(messages=msgs[:-1], system_prompt=SYSTEM, tools=TOOLS)
    assert head and full.startswith(head)
    # system-only head (no user turn yet): the template cannot render it alone
    system_only = p._build_prompt_fragment(system_prompt=SYSTEM, tools=TOOLS)
    assert system_only.startswith("<|im_start|>system\n") and system_only.endswith("<|im_end|>\n")
    assert full.startswith(system_only)


def test_golden_generation_prompt_literals_come_from_the_template(qwen38_tokenizer):
    p = _qwen38_provider(qwen38_tokenizer)
    assert p._generation_prompt_literals() == [
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        "<|im_start|>assistant\n<think>\n",
    ]


def test_golden_continuation_fragment(qwen38_tokenizer):
    p = _qwen38_provider(qwen38_tokenizer)
    frag = p._build_prompt_fragment(
        messages=[{"role": "user", "content": "next question"}],
        add_generation_prompt=True,
        include_bos=False,
    )
    assert frag == "<|im_start|>user\nnext question<|im_end|>\n<|im_start|>assistant\n<think>\n"


def test_golden_serializer_version_names_the_template(qwen38_tokenizer):
    p = _qwen38_provider(qwen38_tokenizer)
    frag = p.prompt_cache_render_fragment(messages=[{"role": "user", "content": "hi"}])
    assert frag is not None
    assert frag.serializer_version.startswith("mlx-prompt-fragment/v2:chat-template:")


# --------------------------------------------------------------------------
# Hermetic: a small template of the same shape (no model needed)
# --------------------------------------------------------------------------

# Every byte is emitted by an expression (the Qwen style), so no Jinja
# whitespace control can eat a newline.
MINI_TEMPLATE = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\\n# Tools' }}"
    "{%- for t in tools %}{{- '\\n' + (t | tojson) }}{%- endfor %}"
    "{%- if messages[0].role == 'system' %}{{- '\\n\\n' + messages[0].content }}{%- endif %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- elif messages[0].role == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- set ns = namespace(user=false) %}"
    "{%- for m in messages %}{%- if m.role == 'user' %}{%- set ns.user = true %}{%- endif %}{%- endfor %}"
    "{%- if not ns.user %}{{- raise_exception('No user query found in messages.') }}{%- endif %}"
    "{%- for m in messages %}"
    "{%- if m.role == 'system' %}"
    "{%- if not loop.first %}{{- raise_exception('System message must be at the beginning.') }}{%- endif %}"
    "{%- elif m.role == 'user' %}"
    "{{- '<|im_start|>user\\n' + (m.content | trim) + '<|im_end|>\\n' }}"
    "{%- elif m.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' + (m.content | trim) }}"
    "{%- for c in (m.tool_calls or []) %}"
    "{{- '\\n<tool_call>\\n<function=' + c.function.name + '>\\n' }}"
    "{%- for k, v in c.function.arguments | items %}"
    "{{- '<parameter=' + k + '>\\n' + (v | string) + '\\n</parameter>\\n' }}"
    "{%- endfor %}"
    "{{- '</function>\\n</tool_call>' }}"
    "{%- endfor %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- elif m.role == 'tool' %}"
    "{%- if loop.previtem and loop.previtem.role != 'tool' %}{{- '<|im_start|>user' }}{%- endif %}"
    "{{- '\\n<tool_response>\\n' + (m.content | trim) + '\\n</tool_response>' }}"
    "{%- if loop.last or loop.nextitem.role != 'tool' %}{{- '<|im_end|>\\n' }}{%- endif %}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\\n\\n</think>\\n\\n' }}"
    "{%- else %}{{- '<think>\\n' }}{%- endif %}"
    "{%- endif %}"
)


class _JinjaTokenizer:
    """Minimal tokenizer double: a real Jinja render of `chat_template`."""

    bos_token = None

    def __init__(self, template: Optional[str] = MINI_TEMPLATE, fail: bool = False):
        self.chat_template = template
        self.fail = fail
        self.calls = 0

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False, **kw):
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        self.calls += 1
        if self.fail:
            raise RuntimeError("template exploded")

        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.filters["tojson"] = lambda x: json.dumps(x, ensure_ascii=False)
        env.globals["raise_exception"] = raise_exception
        return env.from_string(self.chat_template).render(
            messages=messages, tools=tools, add_generation_prompt=add_generation_prompt, **kw
        )

    def encode(self, text, add_special_tokens=True):
        return [ord(c) for c in text]


class _FakeToolHandler:
    supports_prompted = True

    def format_tools_prompt(self, tools, *, include_tool_list=True):
        return "## Tools (fake prompted block)"

    def format_tools_for_chat_template(self, tools):
        return [{"type": "function", "function": {"name": t["name"]}} for t in tools]


def _provider(tokenizer: Any) -> MLXProvider:
    p = MLXProvider.__new__(MLXProvider)
    p.model = "mlx-community/Qwen3-mini-test"
    p.tokenizer = tokenizer
    p.tool_handler = _FakeToolHandler()
    p.architecture_config = {"message_format": "im_start_end", "user_prefix": "<|im_start|>user\n"}
    p.model_capabilities = {}
    return p


def test_tool_role_maps_to_tool_response_inside_one_user_turn():
    p = _provider(_JinjaTokenizer())
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c1", "a", {"x": 1}), _call("c2", "b", {})]},
        {"role": "tool", "content": "R1", "tool_call_id": "c1"},
        {"role": "function", "content": "R2", "tool_call_id": "c2"},
        {"role": "user", "content": "[loop] iteration 2 of 20."},
    ]
    out = p._build_prompt("", msgs, None, None)
    assert "<|im_start|>tool" not in out
    assert "<|im_start|>user\n<tool_response>\nR1\n</tool_response>\n<tool_response>\nR2\n</tool_response><|im_end|>\n" in out
    assert "<tool_call>\n<function=a>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>" in out
    assert out.endswith("<|im_start|>assistant\n<think>\n")


def test_tool_call_shapes_are_normalized_to_dict_arguments():
    calls = MLXProvider._template_tool_calls(
        [
            {"id": "1", "function": {"name": "f", "arguments": '{"a": 1}'}},
            {"name": "g", "arguments": {"b": "x"}},
            {"function": {"name": "h", "arguments": ""}},
            {"function": {"name": ""}},  # nameless: dropped
        ]
    )
    assert [c["function"]["name"] for c in calls] == ["f", "g", "h"]
    assert calls[0]["function"]["arguments"] == {"a": 1} and calls[0]["id"] == "1"
    assert calls[1]["function"]["arguments"] == {"b": "x"}
    assert calls[2]["function"]["arguments"] == {}


def test_tools_go_to_the_template_not_the_prompted_block():
    p = _provider(_JinjaTokenizer())
    out = p._build_prompt("hi", None, "SYS", [{"name": "web_search"}])
    assert out.startswith('<|im_start|>system\n# Tools\n{"type": "function", "function": {"name": "web_search"}}\n\nSYS<|im_end|>\n')
    assert "fake prompted block" not in out


def test_template_without_tool_support_gets_the_prompted_block_in_the_system_turn():
    tmpl = MINI_TEMPLATE.replace("{%- if tools %}", "{%- if false %}").replace("for t in tools", "for t in []")
    assert "tools" not in tmpl
    p = _provider(_JinjaTokenizer(tmpl))
    out = p._build_prompt("hi", None, "SYS", [{"name": "web_search"}])
    assert out.startswith("<|im_start|>system\nSYS\n\n## Tools (fake prompted block)<|im_end|>\n")


def test_system_turns_merge_and_mid_conversation_system_becomes_a_user_instruction():
    p = _provider(_JinjaTokenizer())
    msgs = [
        {"role": "system", "content": "LEAD"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c1", "a", {})]},
        {"role": "system", "content": "GUIDANCE"},  # arrives inside a tool run
        {"role": "tool", "content": "R1", "tool_call_id": "c1"},
        {"role": "user", "content": "next"},
    ]
    out = p._build_prompt("", msgs, "SYS", None)
    assert out.startswith("<|im_start|>system\nSYS\n\nLEAD<|im_end|>\n")
    assert out.count("<|im_start|>system") == 1
    # deferred past the tool-result run, never splitting call/result adjacency
    assert "</tool_response><|im_end|>\n<|im_start|>user\n<system_instruction>\nGUIDANCE\n</system_instruction><|im_end|>\n" in out


def test_long_tool_content_is_kept_verbatim():
    p = _provider(_JinjaTokenizer())
    big = "x" * 200_000 + " END"
    msgs = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [_call("c1", "a", {})]},
        {"role": "tool", "content": big, "tool_call_id": "c1"},
    ]
    out = p._build_prompt("", msgs, None, None)
    assert f"<tool_response>\n{big}\n</tool_response>" in out


def test_image_parts_in_messages_do_not_become_image_placeholders_on_a_text_lane():
    p = _provider(_JinjaTokenizer())
    content = [{"type": "text", "text": "look"}, {"type": "image_url", "image_url": {"url": "data:..."}}]
    out = p._build_prompt("", [{"role": "user", "content": content}], None, None)
    assert "<|image_pad|>" not in out
    assert json.dumps(content, ensure_ascii=False) in out  # unchanged behaviour: JSON text


def test_native_vision_placeholders_in_the_prompt_survive():
    p = _provider(_JinjaTokenizer())
    prompt = "<|vision_start|><|image_pad|><|vision_end|>\nwhat is this?"
    out = p._build_prompt(prompt, None, None, None)
    assert f"<|im_start|>user\n{prompt}<|im_end|>\n" in out


def test_system_only_head_is_cut_before_the_sentinel_turn():
    p = _provider(_JinjaTokenizer())
    head = p._build_prompt_fragment(system_prompt="SYS")
    assert head == "<|im_start|>system\nSYS<|im_end|>\n"
    assert "SENTINEL" not in head
    # generation prompt on a head with no user turn
    opened = p._build_prompt_fragment(system_prompt="SYS", add_generation_prompt=True)
    assert opened == "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>assistant\n<think>\n"


def test_head_cut_without_a_declared_user_prefix_stays_a_true_prefix():
    p = _provider(_JinjaTokenizer())
    p.architecture_config = {}
    head = p._build_prompt_fragment(system_prompt="SYS")
    full = p._build_prompt("q", None, "SYS", None)
    assert full.startswith(head) and "SYS" in head and "SENTINEL" not in head


def test_prefix_stable_across_turns_hermetic():
    p = _provider(_JinjaTokenizer())
    prev = None
    for n in (1, 2, 3):
        text = p._build_prompt("", _iterations(n), SYSTEM, None)
        if prev is not None:
            assert text.startswith(prev)
        prev = text


def test_no_chat_template_falls_back_to_the_hand_built_renderer(caplog):
    p = _provider(_JinjaTokenizer(template=None))
    with caplog.at_level(logging.INFO):
        out = p._build_prompt("hi", None, "SYS", None)
        p._build_prompt("again", None, "SYS", None)
    assert out == "<|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    assert p.tokenizer.calls == 0
    notes = [r for r in caplog.records if "hand-built renderer" in r.getMessage()]
    assert len(notes) == 1  # once per model, not per call
    assert notes[0].levelno == logging.INFO  # no template = nothing to diverge from


def test_a_template_that_raises_falls_back_loudly(caplog):
    p = _provider(_JinjaTokenizer(fail=True))
    with caplog.at_level(logging.WARNING):
        out = p._build_prompt("hi", None, "SYS", None)
    assert out.endswith("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
    assert any(
        "template exploded" in r.getMessage() and r.levelno == logging.WARNING and "#FALLBACK" in r.getMessage()
        for r in caplog.records
    )


def test_template_path_is_logged_once(caplog):
    p = _provider(_JinjaTokenizer())
    with caplog.at_level(logging.INFO):
        p._build_prompt("hi", None, "SYS", None)
        p._build_prompt("hi", None, "SYS", None)
    notes = [r for r in caplog.records if "own chat template" in r.getMessage()]
    assert len(notes) == 1


def test_a_mock_tokenizer_never_switches_the_renderer():
    from unittest.mock import MagicMock

    p = _provider(MagicMock())
    out = p._build_prompt("hi", None, None, None)
    assert out == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"


def test_tools_for_the_chat_template_keep_names_defaults_and_guidance():
    from abstractcore.tools import UniversalToolHandler

    handler = UniversalToolHandler(QWEN38)
    tools = [
        {
            "name": "mcp::srv::fetch",
            "description": "Fetch a URL.",
            "when_to_use": "After you know the URL is worth opening.",
            "parameters": {
                "url": {"type": "string"},
                "include_full_content": {"type": "boolean", "default": True},
            },
        }
    ]
    [schema] = handler.format_tools_for_chat_template(tools)
    fn = schema["function"]
    assert schema["type"] == "function"
    assert fn["name"] == "mcp::srv::fetch"  # the local parser checks calls against this name
    assert fn["description"] == "Fetch a URL.\n\nWhen to use: After you know the URL is worth opening."
    assert fn["parameters"] == {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "include_full_content": {"type": "boolean", "default": True},
        },
        "required": ["url"],
    }
