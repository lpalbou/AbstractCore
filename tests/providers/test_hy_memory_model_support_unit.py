"""Tencent Hy-Memory-30B-A3B (HY V3, model_type `hy_v3`) support.

Covers the registry (every artifact spelling must resolve to `hy_v3`, never to the
295B Hy3 `hunyuan3` entry whose `hy3` pattern is a substring of the GGUF file stem
`hy3a3b-...`), the template tokens, thinking separation, the `<tool_calls>` /
`<tool_sep>` tool-call encoding on the parse, clean and streaming lanes, and the
`effort_disable_value` thinking surface (one `reasoning_effort` variable that is
also the on/off switch: `no_think | low | high`).

Template-backed checks live in test_hy_memory_chat_template_unit.py (real_home).
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from abstractcore.architectures import detect_architecture, get_architecture_format, get_model_capabilities
from abstractcore.architectures.detection import format_messages
from abstractcore.architectures.response_postprocessing import normalize_assistant_text
from abstractcore.architectures.thinking_controls import (
    effort_switch_value,
    resolve_thinking_control_surfaces,
)
from abstractcore.providers.streaming import IncrementalToolDetector
from abstractcore.tools.parser import clean_tool_syntax, parse_tool_calls

MODEL = "tencent/Hy-Memory-30B-A3B"
BOS = "<｜hy_begin▁of▁sentence｜>"
USER = "<｜hy_User｜>"
ASSISTANT = "<｜hy_Assistant｜>"
EOS = "<eos:6124c78e>"

# Exactly what the chat template renders for an assistant tool-call turn
# (chat_template.jinja @ tencent/Hy-Memory-30B-A3B 64a460a9), minus the eos.
TEMPLATE_TOOL_CALLS = (
    "<tool_calls>\n"
    "<tool_call>search<tool_sep>\n"
    "<arg_key>query</arg_key>\n"
    "<arg_value>moe routing</arg_value>\n"
    "<arg_key>limit</arg_key>\n"
    "<arg_value>5</arg_value>\n"
    "<arg_key>strict</arg_key>\n"
    "<arg_value>true</arg_value>\n"
    "</tool_call>\n"
    "</tool_calls>"
)


# ------------------------------------------------------------------ registry

ALIASES_AND_SPELLINGS = [
    "tencent/Hy-Memory-30B-A3B",
    "Hy-Memory-30B-A3B",
    "hy-memory-30b-a3b",
    "models--tencent--Hy-Memory-30B-A3B",
    "tencent/Hy-Memory-30B-A3B-FP8",
    "tencent/Hy-Memory-30B-A3B-GGUF",
    "tencent/Hy-Memory-30B-A3B-GGUF:Q4_K_M",
    "huggingface/tencent/Hy-Memory-30B-A3B",
    "hy3a3b-memrl1009-step250-Q4_K_M-imat.gguf",
    "hy3a3b-memrl1009-step250-Q6_K-imat.gguf",
    "Hy-Memory-Step250-Q8_0.gguf",
    "/models/gguf/hy3a3b-memrl1009-step250-Q4_K_M-imat.gguf",
    "hy3a3b-memrl1009-step250-Q5_K_M.gguf",
    "mlx-community/Hy-Memory-30B-A3B-4bit",
    "mlx-community/Hy-Memory-30B-A3B-8bit",
    "/tmp/out/Hy-Memory-30B-A3B-4bit",
    "/tmp/out/Hy-Memory-30B-A3B-8bit",
    "hermitdave/Hy-Memory-30B-A3B-MLX-8bit",
    "lmstudio/tencent/hy-memory-30b-a3b",
]


@pytest.mark.parametrize("model", ALIASES_AND_SPELLINGS)
def test_every_hy_memory_spelling_resolves_to_hy_v3_and_its_model_entry(model: str) -> None:
    assert detect_architecture(model) == "hy_v3", model
    caps = get_model_capabilities(model)
    assert caps.get("architecture") == "hy_v3", model
    assert caps.get("max_tokens") == 262144, model
    assert caps.get("tool_support") == "native", model
    assert caps.get("thinking_support") is True, model
    # Never the 295B Hy3 entry (its ':opensource' think tags would mis-split output).
    assert caps.get("thinking_tags") == ["<think>", "</think>"], model


def test_hy3_295b_is_untouched_by_the_hy_v3_patterns() -> None:
    for model in ("tencent/Hy3", "tencent/Hy3-FP8", "hy3"):
        assert detect_architecture(model) == "hunyuan3", model
        assert get_model_capabilities(model)["thinking_tags"] == ["<think:opensource>", "</think:opensource>"]


def test_model_entry_carries_the_specialty_and_checkpoint_facts() -> None:
    caps = get_model_capabilities(MODEL)
    notes = caps["notes"].lower()
    assert "specialty seems to be a long-term memory writer" in notes
    assert "expert_hidden_dim" in notes  # the 768-vs-1536 checkpoint quirk is documented
    assert caps["experts"] == 128 and caps["experts_activated"] == 8 and caps["shared_experts"] == 1
    assert caps["license"] == "Apache-2.0"
    assert caps["release_date"] == "2026-10-15"
    assert "2026-10-21" in caps["source"]
    assert caps["reasoning_levels"] == ["low", "high"]


def test_transcript_matches_chat_template_tokens() -> None:
    rendered = format_messages(
        [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
            {"role": "assistant", "content": "A"},
        ],
        "hy_v3",
    )
    assert rendered == f"{BOS}S{USER}U{ASSISTANT}A{EOS}"
    arch_fmt = get_architecture_format("hy_v3")
    assert arch_fmt["output_wrappers"]["end"] == EOS


# ------------------------------------------------------------------ thinking output

def test_think_block_is_split_into_reasoning() -> None:
    arch_fmt = get_architecture_format(detect_architecture(MODEL))
    caps = get_model_capabilities(MODEL)
    cleaned, reasoning = normalize_assistant_text(
        "<think>weighing what to retain</think>{\"memories\": [\"Alex lives in Seattle\"]}",
        architecture_format=arch_fmt,
        model_capabilities=caps,
    )
    assert cleaned == '{"memories": ["Alex lives in Seattle"]}'
    assert reasoning == "weighing what to retain"


def test_prompt_opened_think_block_is_split_into_reasoning() -> None:
    # reasoning_effort=low|high ends the generation prompt with `<think>`, so the
    # model's output carries only the closing tag.
    arch_fmt = get_architecture_format("hy_v3")
    cleaned, reasoning = normalize_assistant_text(
        "the user stated a preference</think>Noted.",
        architecture_format=arch_fmt,
        model_capabilities=get_model_capabilities(MODEL),
        thinking_opened_by_prompt=True,
    )
    assert cleaned == "Noted."
    assert reasoning == "the user stated a preference"


# ------------------------------------------------------------------ tool calls (parse)

def test_template_shaped_tool_call_is_parsed() -> None:
    calls = parse_tool_calls(TEMPLATE_TOOL_CALLS, MODEL)
    assert [(c.name, c.arguments) for c in calls] == [
        ("search", {"query": "moe routing", "limit": 5, "strict": True})
    ]


def test_tool_name_never_contains_the_separator() -> None:
    calls = parse_tool_calls("<tool_calls><tool_call>get_time<tool_sep><arg_key>tz</arg_key><arg_value>UTC</arg_value></tool_call></tool_calls>", MODEL)
    assert [c.name for c in calls] == ["get_time"]
    assert all("<tool_sep>" not in c.name for c in calls)


def test_argument_less_call_with_separator_is_a_call() -> None:
    calls = parse_tool_calls("<tool_calls>\n<tool_call>list_memories<tool_sep>\n</tool_call>\n</tool_calls>", MODEL)
    assert [(c.name, c.arguments) for c in calls] == [("list_memories", {})]


def test_parallel_calls_inside_one_wrapper_are_all_parsed() -> None:
    calls = parse_tool_calls(
        "<tool_calls>\n"
        "<tool_call>a<tool_sep>\n<arg_key>x</arg_key>\n<arg_value>1</arg_value>\n</tool_call>\n"
        "<tool_call>b<tool_sep>\n<arg_key>y</arg_key>\n<arg_value>hi</arg_value>\n</tool_call>\n"
        "</tool_calls>",
        MODEL,
    )
    assert [(c.name, c.arguments) for c in calls] == [("a", {"x": 1}), ("b", {"y": "hi"})]


def test_missing_separator_never_returns_preamble_as_the_name() -> None:
    # mlx-lm PR #1211 review: with <tool_sep> omitted, a parser returned the whole
    # preamble as the tool name. A garbled name must be rejected, not returned.
    for text in (
        "<tool_calls><tool_call>I will now look this up for you, calling search</arg_value></tool_call></tool_calls>",
        "<tool_calls><tool_call>Let me check the memory bank first.\nsearch<arg_key>q</arg_key><arg_value>x</arg_value></tool_call></tool_calls>",
        "<tool_calls><tool_call>Sure! search<tool_sep><arg_key>q</arg_key><arg_value>x</arg_value></tool_call></tool_calls>",
    ):
        for call in parse_tool_calls(text, MODEL):
            assert call.name.isidentifier(), (text, call.name)


def test_call_cut_inside_a_value_is_not_executed_with_a_missing_argument() -> None:
    calls = parse_tool_calls(
        "<tool_calls>\n<tool_call>write_memory<tool_sep>\n"
        "<arg_key>key</arg_key>\n<arg_value>diet</arg_value>\n"
        "<arg_key>value</arg_key>\n<arg_value>vegetar",
        MODEL,
    )
    assert calls == []


def test_tool_markup_does_not_leak_into_cleaned_content() -> None:
    calls = parse_tool_calls("Checking the bank.\n" + TEMPLATE_TOOL_CALLS, MODEL)
    cleaned = clean_tool_syntax("Checking the bank.\n" + TEMPLATE_TOOL_CALLS, calls)
    assert cleaned == "Checking the bank."


def test_unclosed_trailing_call_payload_does_not_stay_as_content() -> None:
    # Live (Q4_K_M GGUF, abstractcore's JSON tool prompt): the model wrote
    # `<tool_call>\n{json}` and then its EOS, never `</tool_call>`. The parser
    # accepts that call, so the cleaner must remove the same span.
    text = 'Saving it.\n<tool_call>\n{"name": "save_memory", "arguments": {"fact": "Allergic to peanuts", "importance": 5}}'
    calls = parse_tool_calls(text, MODEL)
    assert [(c.name, c.arguments) for c in calls] == [("save_memory", {"fact": "Allergic to peanuts", "importance": 5})]
    assert clean_tool_syntax(text, calls) == "Saving it."
    # A closed call followed by prose keeps the prose.
    closed = '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\nDone.'
    assert clean_tool_syntax(closed, parse_tool_calls(closed, MODEL)) == "Done."


def test_calls_recovered_when_server_returns_empty_tool_calls() -> None:
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
    tools = [{
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search memories",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string"}, "limit": {"type": "integer"}, "strict": {"type": "boolean"}}},
        },
    }]
    normalized = provider._normalize_tool_calls_passthrough(
        response=GenerateResponse(
            content="Looking it up.\n" + TEMPLATE_TOOL_CALLS, model=MODEL, finish_reason="stop", tool_calls=[]
        ),
        tools=tools,
    )
    recovered = normalized.tool_calls or []
    assert [(c["name"], c["arguments"]) for c in recovered] == [
        ("search", {"query": "moe routing", "limit": 5, "strict": True})
    ]
    for marker in ("<tool_calls>", "</tool_calls>", "<tool_call>", "<tool_sep>", "<arg_key>"):
        assert marker not in (normalized.content or ""), marker


# ------------------------------------------------------------------ tool calls (streaming)

def _stream(text: str, chunk_size: int):
    detector = IncrementalToolDetector(model_name=MODEL)
    streamed, calls = "", []
    for index in range(0, len(text), chunk_size):
        content, tool_calls = detector.process_chunk(text[index:index + chunk_size])
        streamed += content
        calls.extend(tool_calls)
    owed, tail_calls, unparsed = detector.drain()
    streamed += owed
    calls.extend(tail_calls)
    return streamed, calls, unparsed


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 64])
def test_streaming_extracts_wrapped_call_and_keeps_only_prose(chunk_size: int) -> None:
    streamed, calls, unparsed = _stream("Checking the bank." + TEMPLATE_TOOL_CALLS + "\nDone.", chunk_size)
    assert [(c.name, c.arguments) for c in calls] == [
        ("search", {"query": "moe routing", "limit": 5, "strict": True})
    ]
    assert unparsed is None
    assert "<tool_calls>" not in streamed and "</tool_calls>" not in streamed
    assert streamed.startswith("Checking the bank.")
    assert streamed.rstrip().endswith("Done.")


def test_streaming_recovers_truncated_wrapped_call() -> None:
    _, calls, _ = _stream("<tool_calls>\n<tool_call>search<tool_sep>\n<arg_key>q</arg_key>\n<arg_value>moe</arg_value>\n", 5)
    assert [(c.name, c.arguments) for c in calls] == [("search", {"q": "moe"})]


def test_streaming_parallel_calls_in_one_wrapper() -> None:
    _, calls, _ = _stream(
        "<tool_calls>\n<tool_call>a<tool_sep>\n<arg_key>x</arg_key>\n<arg_value>1</arg_value>\n</tool_call>\n"
        "<tool_call>b<tool_sep>\n</tool_call>\n</tool_calls>",
        4,
    )
    assert [(c.name, c.arguments) for c in calls] == [("a", {"x": 1}), ("b", {})]


# ------------------------------------------------------------------ effort switch surface

def _surfaces():
    return resolve_thinking_control_surfaces(
        model_capabilities=get_model_capabilities(MODEL),
        architecture_format=get_architecture_format("hy_v3"),
    )


def test_registry_declares_an_effort_switch() -> None:
    s = _surfaces()
    assert s.effort_is_switch()
    assert s.effort_template_kwarg == "reasoning_effort"
    assert s.effort_disable_value == "no_think"
    assert s.template_kwarg is None  # the template has NO boolean switch


@pytest.mark.parametrize(
    "enabled,level,expected",
    [(False, None, "no_think"), (True, None, "high"), (True, "low", "low"), (True, "high", "high"), (True, "medium", None)],
)
def test_effort_switch_values(enabled, level, expected) -> None:
    assert effort_switch_value(_surfaces(), enabled=enabled, level=level, declared_levels=["low", "high"]) == expected


def _mlx_provider():
    from abstractcore.providers.mlx_provider import MLXProvider

    p = MLXProvider.__new__(MLXProvider)
    p.model = MODEL
    p.architecture = "hy_v3"
    p.architecture_config = get_architecture_format("hy_v3")
    p.model_capabilities = get_model_capabilities(MODEL)
    return p


def _hf_provider(model_type: str):
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    p = HuggingFaceProvider.__new__(HuggingFaceProvider)
    p.model = MODEL
    p.model_type = model_type
    p.architecture = "hy_v3"
    p.architecture_config = get_architecture_format("hy_v3")
    p.model_capabilities = get_model_capabilities(MODEL)
    return p


@pytest.mark.parametrize("thinking,expected", [(False, "no_think"), (True, "high"), ("low", "low")])
def test_mlx_hook_sends_the_effort_value_not_a_boolean(thinking, expected) -> None:
    p = _mlx_provider()
    enabled, level = p._normalize_thinking_request(thinking)
    kwargs, handling = p._apply_provider_thinking_kwargs(enabled=enabled, level=level, kwargs={})
    assert kwargs.get("_acore_mlx_reasoning_effort") == expected
    assert "_acore_mlx_enable_thinking" not in kwargs
    assert handling.handled_enable_disable is True


def test_mlx_hook_declines_when_the_system_bloc_is_prefilled() -> None:
    p = _mlx_provider()
    kwargs, handling = p._apply_provider_thinking_kwargs(
        enabled=True, level="high", kwargs={"prompt_cache_prefilled_modules": ["system"]}
    )
    assert "_acore_mlx_reasoning_effort" not in kwargs
    assert not handling.handled_enable_disable and not handling.handled_level


@pytest.mark.parametrize("thinking,expected", [(False, "no_think"), (True, "high"), ("high", "high")])
def test_hf_transformers_hook_sends_the_effort_value(thinking, expected) -> None:
    p = _hf_provider("transformers")
    enabled, level = p._normalize_thinking_request(thinking)
    kwargs, handling = p._apply_provider_thinking_kwargs(enabled=enabled, level=level, kwargs={})
    assert kwargs.get("_acore_hf_transformers_reasoning_effort") == expected
    assert "_acore_hf_transformers_enable_thinking" not in kwargs
    assert handling.handled_enable_disable is True
    assert p._hf_thinking_template_kwargs(kwargs) == {"reasoning_effort": expected}


def test_hf_transformers_cached_lane_declines() -> None:
    p = _hf_provider("transformers")
    kwargs, handling = p._apply_provider_thinking_kwargs(
        enabled=False, level=None, kwargs={"prompt_cache_key": "k"}
    )
    assert "_acore_hf_transformers_reasoning_effort" not in kwargs
    assert not handling.handled_enable_disable


def _openai_compatible(monkeypatch, *, template_kwargs: bool, provider_id: str = "openai-compatible"):
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None, raising=False)
    provider = OpenAICompatibleProvider(model=MODEL, base_url="http://127.0.0.1:9/v1")
    provider.provider = provider_id
    provider._supports_chat_template_kwargs = template_kwargs
    return provider


@pytest.mark.parametrize(
    "template_kwargs,thinking,top_level,ctk,handled",
    [
        # no_think is not an OpenAI enum value: never top-level (a validating server 400s)
        (True, False, None, "no_think", True),
        (False, False, None, None, False),  # no transport for off: decline honestly
        (True, "high", "high", "high", True),
        (False, "high", "high", None, True),
        (False, True, "high", None, True),  # bare on -> preferred declared level
        (True, "low", "low", "low", True),
    ],
)
def test_openai_compatible_effort_switch_transport(monkeypatch, template_kwargs, thinking, top_level, ctk, handled) -> None:
    provider = _openai_compatible(monkeypatch, template_kwargs=template_kwargs)
    enabled, level = provider._normalize_thinking_request(thinking)
    kwargs, handling = provider._apply_provider_thinking_kwargs(enabled=enabled, level=level, kwargs={})
    assert kwargs.get("reasoning_effort") == top_level
    assert (kwargs.get("chat_template_kwargs") or {}).get("reasoning_effort") == ctk
    assert "enable_thinking" not in (kwargs.get("chat_template_kwargs") or {})
    assert handling.handled_enable_disable is handled


def test_lmstudio_effort_switch_mirrors_chat_template_kwargs(monkeypatch) -> None:
    provider = _openai_compatible(monkeypatch, template_kwargs=True, provider_id="lmstudio")
    kwargs, handling = provider._apply_provider_thinking_kwargs(enabled=False, level=None, kwargs={})
    assert kwargs["chat_template_kwargs"] == {"reasoning_effort": "no_think"}
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"reasoning_effort": "no_think"}
    assert "reasoning_effort" not in kwargs


def test_openai_compatible_high_reaches_the_wire_and_off_is_not_misreported(monkeypatch) -> None:
    from abstractcore.core.types import GenerateResponse

    provider = _openai_compatible(monkeypatch, template_kwargs=False)
    captured: Dict[str, Any] = {}

    def _cap(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _cap)
    with _no_warning_named("cannot be fully disabled"):
        resp = provider.generate("hi", thinking="high", temperature=0)
    assert captured["payload"].get("reasoning_effort") == "high"
    assert resp.metadata.get("thinking_effective") == "high"
    resp = provider.generate("hi", thinking="off", temperature=0)
    assert "reasoning_effort" not in captured["payload"]
    assert resp.metadata.get("thinking_effective") is None  # no control sent: not claimed


@pytest.mark.parametrize("thinking,expected", [(False, "no_think"), (True, "high"), ("low", "low"), ("high", "high")])
def test_vllm_effort_switch_sends_only_the_effort_value(thinking, expected) -> None:
    from abstractcore.providers.vllm_provider import VLLMProvider

    p = VLLMProvider.__new__(VLLMProvider)
    p.model, p.provider, p.architecture = MODEL, "vllm", "hy_v3"
    p.architecture_config = get_architecture_format("hy_v3")
    p.model_capabilities = get_model_capabilities(MODEL)
    enabled, level = p._normalize_thinking_request(thinking)
    kwargs, handling = p._apply_provider_thinking_kwargs(enabled=enabled, level=level, kwargs={})
    assert kwargs["extra_body"]["chat_template_kwargs"] == {"reasoning_effort": expected}
    assert handling.handled_enable_disable is True


def test_vllm_effort_switch_declines_an_undeclared_level() -> None:
    from abstractcore.providers.vllm_provider import VLLMProvider

    p = VLLMProvider.__new__(VLLMProvider)
    p.model, p.provider, p.architecture = MODEL, "vllm", "hy_v3"
    p.architecture_config = get_architecture_format("hy_v3")
    p.model_capabilities = get_model_capabilities(MODEL)
    kwargs, handling = p._apply_provider_thinking_kwargs(enabled=True, level="medium", kwargs={})
    assert kwargs == {} and not handling.handled_enable_disable


class _no_warning_named:
    def __init__(self, fragment: str) -> None:
        self.fragment = fragment

    def __enter__(self):
        import warnings

        self._cm = warnings.catch_warnings(record=True)
        self._records = self._cm.__enter__()
        warnings.simplefilter("always")
        return self

    def __exit__(self, *exc):
        self._cm.__exit__(*exc)
        bad = [str(r.message) for r in self._records if self.fragment in str(r.message)]
        assert not bad, bad
        return False


# ------------------------------------------------------------------ provider load/lane fixes

def test_transformers_config_fix_sizes_experts_from_expert_hidden_dim() -> None:
    from types import SimpleNamespace

    p = _hf_provider("transformers")
    import logging

    p.logger = logging.getLogger("test")
    cfg = SimpleNamespace(model_type="hy_v3", moe_intermediate_size=1536, expert_hidden_dim=768, num_shared_experts=1)
    assert p._fix_transformers_checkpoint_config(cfg) is True
    assert cfg.moe_intermediate_size == 768
    assert p._fix_transformers_checkpoint_config(cfg) is False  # already consistent
    other = SimpleNamespace(model_type="qwen3_moe", moe_intermediate_size=1536, expert_hidden_dim=768)
    assert p._fix_transformers_checkpoint_config(other) is False and other.moe_intermediate_size == 1536
    assert p._fix_transformers_checkpoint_config(None) is False


@pytest.mark.parametrize(
    "rendered,expected",
    [
        (f"{BOS}PROBE-SYSTEM{USER}PROBE-USER{ASSISTANT}<think></think>PROBE-ASSISTANT{EOS}{USER}PROBE-FOLLOWUP{ASSISTANT}<think></think>", True),
        # role tokens the registry does not declare: stays off the control plane
        ("<|im_start|>system\nPROBE-SYSTEM<|im_end|>\n<|im_start|>user\nPROBE-FOLLOWUP<|im_end|>\n<|im_start|>assistant\n", False),
        # no generation prompt after the last user turn
        (f"{BOS}PROBE-SYSTEM{USER}PROBE-FOLLOWUP", False),
    ],
)
def test_gguf_control_plane_admits_the_embedded_template_by_declared_role_tokens(monkeypatch, rendered, expected) -> None:
    p = _hf_provider("gguf")
    monkeypatch.setattr(p, "_gguf_render_llama_cpp_chat_template_prompt", lambda **kw: rendered)
    assert p._gguf_embedded_template_probe_renders_declared_turns("TEMPLATE-" + rendered[:8]) is expected


@pytest.mark.parametrize("thinking,expected", [(False, "no_think"), ("high", "high")])
def test_gguf_hook_sends_the_effort_value_on_the_embedded_template_lane(monkeypatch, thinking, expected) -> None:
    p = _hf_provider("gguf")
    monkeypatch.setattr(p, "_gguf_prompt_cache_supports_local_control_plane", lambda: True)
    monkeypatch.setattr(p, "_gguf_prompt_cache_control_plane_chat_format", lambda: "llama-cpp-chat-template")
    enabled, level = p._normalize_thinking_request(thinking)
    kwargs, handling = p._apply_provider_thinking_kwargs(
        enabled=enabled, level=level, kwargs={}, request_shape={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert kwargs.get("_acore_gguf_reasoning_effort") == expected
    assert "_acore_gguf_enable_thinking" not in kwargs
    assert handling.handled_enable_disable is True


def test_gguf_hook_declines_off_the_embedded_template_lane(monkeypatch) -> None:
    p = _hf_provider("gguf")
    monkeypatch.setattr(p, "_gguf_prompt_cache_supports_local_control_plane", lambda: True)
    monkeypatch.setattr(p, "_gguf_prompt_cache_control_plane_chat_format", lambda: "")
    monkeypatch.setattr(p, "_gguf_prompt_cache_chat_format", lambda: "chat_template.default")
    kwargs, handling = p._apply_provider_thinking_kwargs(enabled=False, level=None, kwargs={}, request_shape={})
    assert "_acore_gguf_reasoning_effort" not in kwargs
    assert not handling.handled_enable_disable


@pytest.mark.parametrize(
    "path",
    [
        "/Users/u/.cache/huggingface/hub/models--tencent--Hy-Memory-30B-A3B/snapshots/64a460a94d5498b7316fb02395c72d01c47a7347",
        "/Users/u/.cache/huggingface/hub/models--tencent--Hy-Memory-30B-A3B/snapshots/64a460a94d5498b7316fb02395c72d01c47a7347/",
        "/Users/u/.cache/huggingface/hub/models--tencent--Hy-Memory-30B-A3B-GGUF/snapshots/7778309f/hy3a3b-memrl1009-step250-Q4_K_M-imat.gguf",
    ],
)
def test_hf_cache_snapshot_paths_resolve_to_the_model_entry(path: str) -> None:
    # The tail of a snapshot path is a revision hash; without resolving the
    # `models--org--name` segment a request through the local path fell back to
    # architecture defaults (16384 context, no effort switch).
    caps = get_model_capabilities(path)
    assert caps["architecture"] == "hy_v3"
    assert caps["max_tokens"] == 262144
    assert caps["thinking_control"]["effort_disable_value"] == "no_think"


# ------------------------------------------------------------------ GGUF control-plane sampler (review N3/S8)

def test_gguf_sampler_keeps_explicit_zeros_and_registry_defaults() -> None:
    p = _hf_provider("gguf")
    p.top_k = None
    explicit = p._gguf_control_plane_sampler({"top_k": 0, "min_p": 0.0, "repeat_penalty": 1.0}, {"temperature": 0.0, "top_p": 1.0})
    assert explicit["temperature"] == 0.0  # greedy, not the lane default 0.2 (04957608)
    assert explicit["top_k"] == 0 and explicit["min_p"] == 0.0 and explicit["repeat_penalty"] == 1.0
    defaults = p._gguf_control_plane_sampler({}, {"temperature": None, "top_p": None})
    assert defaults["temperature"] == 0.2 and defaults["top_p"] == 0.95 and defaults["top_k"] == 40
    assert defaults["repeat_penalty"] == 1.0  # hy_v3 registry inference_parameters (GGUF card)
    assert defaults["min_p"] == 0.05
    p.top_k = 0  # provider-level setting (e.g. create_llm(..., top_k=0))
    assert p._gguf_control_plane_sampler({}, {})["top_k"] == 0
