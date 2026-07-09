"""Typed thinking-control surfaces: resolver semantics, honest handling, and capture.

These tests pin the fix for the untyped `thinking_control` conflation:
- template-variable names (e.g. `enable_thinking`) must NEVER be appended as prompt text;
- `thinking_handled_enable_disable` / `thinking_effective` must be honest (ADR-0001);
- LM Studio maps `thinking=` to its documented native REST `reasoning` control for models
  whose only declared surface is a chat-template kwarg (e.g. Gemma 4);
- unterminated thinking blocks are auto-closed into reasoning (#TRUNCATION) instead of
  leaking into visible content;
- OpenAI-compatible usage extraction preserves `completion_tokens_details`
  (the only billing evidence of invisible reasoning, e.g. grok-4).
"""

from __future__ import annotations

import warnings

import pytest

from abstractcore.architectures import detect_architecture, get_architecture_format, get_model_capabilities
from abstractcore.architectures.response_postprocessing import strip_thinking_tags
from abstractcore.architectures.thinking_controls import (
    ThinkingControlSurfaces,
    resolve_thinking_control_surfaces,
)
from abstractcore.core.types import GenerateResponse
from abstractcore.providers.lmstudio_provider import LMStudioProvider
from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider


# ---------------------------------------------------------------------------
# Resolver semantics
# ---------------------------------------------------------------------------


def test_resolver_merges_per_key_with_model_caps_overriding_architecture() -> None:
    arch = {"thinking_control": {"prompt_disable_token": "/no_think", "template_kwarg": "enable_thinking"}}
    caps = {"thinking_control": {"template_kwarg": "enable_reasoning"}}

    surfaces = resolve_thinking_control_surfaces(model_capabilities=caps, architecture_format=arch)

    assert surfaces.prompt_disable_token == "/no_think"  # inherited from architecture
    assert surfaces.template_kwarg == "enable_reasoning"  # model override wins
    assert surfaces.any_declared() is True


def test_resolver_legacy_string_prompt_token_is_tolerated_with_warning() -> None:
    with pytest.warns(RuntimeWarning, match="#FALLBACK.*prompt_disable_token"):
        surfaces = resolve_thinking_control_surfaces(
            model_capabilities={"thinking_control": "/nothink"},
            architecture_format=None,
        )
    assert surfaces.prompt_disable_token == "/nothink"


def test_resolver_legacy_string_non_token_is_ignored_with_warning() -> None:
    # `enable_thinking` is a template variable name and cannot be applied as a prompt token.
    with pytest.warns(RuntimeWarning, match="#FALLBACK.*cannot be applied safely"):
        surfaces = resolve_thinking_control_surfaces(
            model_capabilities={"thinking_control": "enable_thinking"},
            architecture_format=None,
        )
    assert surfaces.any_declared() is False


def test_shipped_assets_declare_expected_surfaces() -> None:
    gemma = resolve_thinking_control_surfaces(
        model_capabilities=get_model_capabilities("google/gemma-4-26b-a4b"),
        architecture_format=get_architecture_format(detect_architecture("google/gemma-4-26b-a4b")),
    )
    assert gemma.template_kwarg == "enable_thinking"
    assert gemma.prompt_disable_token is None

    glm = resolve_thinking_control_surfaces(
        model_capabilities=get_model_capabilities("glm-4.6v"),
        architecture_format=get_architecture_format(detect_architecture("glm-4.6v")),
    )
    assert glm.prompt_disable_token == "/nothink"

    qwen = resolve_thinking_control_surfaces(
        model_capabilities=get_model_capabilities("qwen3.5-4b"),
        architecture_format=get_architecture_format(detect_architecture("qwen3.5-4b")),
    )
    assert qwen.template_kwarg == "enable_thinking"
    assert qwen.assistant_prefill_disable == "<think>\n\n</think>\n\n"


# ---------------------------------------------------------------------------
# Gemma 4: no prompt pollution + LM Studio native reasoning mapping
# ---------------------------------------------------------------------------


def _make_lmstudio(monkeypatch, model: str) -> LMStudioProvider:
    monkeypatch.setattr(LMStudioProvider, "_validate_model", lambda self: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # token-window config warnings on init are unrelated
        return LMStudioProvider(model=model, base_url="http://localhost:1234/v1")


def test_gemma4_thinking_off_no_longer_pollutes_prompt(monkeypatch) -> None:
    provider = _make_lmstudio(monkeypatch, "google/gemma-4-26b-a4b")

    prompt, messages, system_prompt, kwargs, meta = provider._apply_thinking_request(
        thinking="off",
        prompt="Hello",
        messages=None,
        system_prompt=None,
        kwargs={},
    )

    assert prompt == "Hello"  # previously: "Hello\nenable_thinking"
    assert "enable_thinking" not in str(prompt)
    assert kwargs.get("reasoning") == "off"  # LM Studio native REST control
    assert kwargs.get("chat_template_kwargs", {}).get("enable_thinking") is False
    assert meta is not None
    assert meta["thinking_effective"] == "off"
    assert meta["thinking_handled_enable_disable"] is True


def test_gemma4_lmstudio_thinking_routes_to_native_rest(monkeypatch) -> None:
    provider = _make_lmstudio(monkeypatch, "google/gemma-4-26b-a4b")

    captured = {}

    def _fake_native(*, prompt, system_prompt, stream, **kwargs):
        captured["reasoning"] = kwargs.get("reasoning")
        return GenerateResponse(
            content="final",
            model=provider.model,
            finish_reason="stop",
            metadata={"reasoning": "native reasoning text"},
        )

    monkeypatch.setattr(provider, "_native_rest_chat_generate", _fake_native)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resp = provider.generate("hi", thinking="on", temperature=0)

    assert captured["reasoning"] == "on"
    assert resp.content == "final"
    assert resp.metadata.get("reasoning") == "native reasoning text"
    assert resp.metadata.get("thinking_effective") == "on"
    # The old "does not implement a thinking control mapping" warning must be gone.
    mapping_warnings = [w for w in caught if "does not implement a thinking control" in str(w.message)]
    assert not mapping_warnings


def test_gemma4_lmstudio_thinking_level_clamps_to_on_with_warning(monkeypatch) -> None:
    provider = _make_lmstudio(monkeypatch, "google/gemma-4-26b-a4b")

    captured = {}

    def _fake_native(*, prompt, system_prompt, stream, **kwargs):
        captured["reasoning"] = kwargs.get("reasoning")
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_native_rest_chat_generate", _fake_native)

    with pytest.warns(RuntimeWarning, match="cannot enforce effort scaling"):
        provider.generate("hi", thinking="high", temperature=0)

    # Gemma supports only on/off on LM Studio native REST; clamping avoids HTTP 400.
    assert captured["reasoning"] == "on"


def test_gemma4_lmstudio_native_route_ineligible_warns_and_falls_back(monkeypatch) -> None:
    provider = _make_lmstudio(monkeypatch, "google/gemma-4-26b-a4b")

    def _fail_native(**_kwargs):  # must not be called for tool requests
        raise AssertionError("native REST must not be used when tools are present")

    monkeypatch.setattr(provider, "_native_rest_chat_generate", _fail_native)

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    tools = [{"type": "function", "function": {"name": "noop", "parameters": {"type": "object", "properties": {}}}}]
    with pytest.warns(RuntimeWarning, match="does not accept custom tools"):
        provider.generate("hi", thinking="off", tools=tools, temperature=0)

    payload = captured["payload"]
    # The native reasoning control was dropped, but the best-effort template artifact rides along.
    assert "reasoning" not in payload
    assert payload.get("chat_template_kwargs", {}).get("enable_thinking") is False


def test_gemma4_lmstudio_streaming_routes_native_with_separated_reasoning(monkeypatch) -> None:
    # Native REST streams typed SSE events: message deltas -> content, reasoning deltas ->
    # metadata["reasoning"], chat.end -> usage (incl. reasoning token details).
    provider = _make_lmstudio(monkeypatch, "google/gemma-4-26b-a4b")

    def _fake_stream(*, prompt, system_prompt, media=None, **kwargs):
        assert kwargs.get("reasoning") == "on"
        yield GenerateResponse(content="Hel", model=provider.model)
        yield GenerateResponse(content="", model=provider.model, metadata={"reasoning": "because"})
        yield GenerateResponse(content="lo", model=provider.model)
        yield GenerateResponse(
            content="",
            model=provider.model,
            finish_reason="stop",
            usage={
                "input_tokens": 5,
                "output_tokens": 30,
                "total_tokens": 35,
                "prompt_tokens": 5,
                "completion_tokens": 30,
                "completion_tokens_details": {"reasoning_tokens": 20},
            },
        )

    monkeypatch.setattr(provider, "_native_rest_chat_stream", _fake_stream)

    chunks = list(provider.generate("hi", thinking="on", stream=True, temperature=0))

    text = "".join(c.content or "" for c in chunks)
    assert text == "Hello"
    # Content must flow incrementally (eager visible mode), not burst in one final chunk.
    content_chunks = [c.content for c in chunks if c.content]
    assert content_chunks == ["Hel", "lo"]
    reasoning_deltas = [c.metadata.get("reasoning") for c in chunks if isinstance(c.metadata, dict) and c.metadata.get("reasoning")]
    assert "because" in reasoning_deltas
    usage_chunks = [c.usage for c in chunks if c.usage]
    assert usage_chunks and usage_chunks[-1]["completion_tokens_details"] == {"reasoning_tokens": 20}


def test_lmstudio_native_sse_event_parsing() -> None:
    lines = [
        "event: chat.start",
        'data: {"type":"chat.start","model_instance_id":"m"}',
        "",
        "event: reasoning.delta",
        'data: {"type":"reasoning.delta","content":"think"}',
        "event: message.delta",
        'data: {"type":"message.delta","content":"answer"}',
        "event: chat.end",
        'data: {"type":"chat.end","result":{"output":[],"stats":{"input_tokens":3,"total_output_tokens":9,"reasoning_output_tokens":4}}}',
    ]

    events = list(LMStudioProvider._iter_native_sse_events(lines))
    types = [e.get("type") for e in events]
    assert types == ["chat.start", "reasoning.delta", "message.delta", "chat.end"]
    assert events[1]["content"] == "think"
    assert events[2]["content"] == "answer"

    usage = LMStudioProvider._native_rest_usage_from_stats(events[3]["result"]["stats"])
    assert usage["input_tokens"] == 3
    assert usage["output_tokens"] == 9
    assert usage["completion_tokens_details"] == {"reasoning_tokens": 4}


def test_lmstudio_native_image_parts_and_payload(monkeypatch) -> None:
    from abstractcore.media.types import ContentFormat, MediaContent, MediaType

    provider = _make_lmstudio(monkeypatch, "google/gemma-4-26b-a4b")

    image = MediaContent(
        media_type=MediaType.IMAGE,
        content=b"\x89PNG",
        content_format=ContentFormat.BASE64,
        mime_type="image/png",
    )
    parts = LMStudioProvider._native_rest_image_parts([image])
    assert parts is not None and len(parts) == 1
    assert parts[0]["type"] == "image"
    assert parts[0]["data_url"].startswith("data:image/png;base64,")

    payload = provider._native_rest_build_chat_payload(
        prompt="What color?",
        system_prompt=None,
        stream=False,
        media=[image],
        reasoning="off",
    )
    assert isinstance(payload["input"], list)
    assert payload["input"][0] == {"type": "text", "content": "What color?"}
    assert payload["input"][1]["type"] == "image"
    assert payload["reasoning"] == "off"

    # Non-image media cannot ride the native route.
    audio = MediaContent(
        media_type=MediaType.AUDIO,
        content=b"RIFF",
        content_format=ContentFormat.BASE64,
        mime_type="audio/wav",
    )
    assert LMStudioProvider._native_rest_image_parts([image, audio]) is None


def test_gemma4_strict_openai_compatible_thinking_off_is_honest_noop(monkeypatch) -> None:
    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None, raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        provider = OpenAICompatibleProvider(model="google/gemma-4-26b-a4b", base_url="http://127.0.0.1:9/v1")

    captured = {}

    def _capture_single_generate(payload):
        captured["payload"] = payload
        return GenerateResponse(content="ok", model=provider.model, finish_reason="stop")

    monkeypatch.setattr(provider, "_single_generate", _capture_single_generate)

    with pytest.warns(RuntimeWarning, match="default thinking behavior remains in effect"):
        resp = provider.generate("Say hello.", thinking="off", temperature=0)

    payload = captured["payload"]
    payload_str = str(payload)
    assert "enable_thinking" not in payload_str  # no prompt pollution, no unsupported kwargs
    # Honesty: the request was NOT handled, so effective state must not claim "off".
    assert resp.metadata.get("thinking_effective") is None
    assert resp.metadata.get("thinking_handled_enable_disable") is False


def test_glm_prompt_disable_token_still_appended(monkeypatch) -> None:
    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None, raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        provider = OpenAICompatibleProvider(model="glm-4.6v", base_url="http://127.0.0.1:9/v1")

    prompt, _messages, _system_prompt, _kwargs, meta = provider._apply_thinking_request(
        thinking="off",
        prompt="Describe the picture.",
        messages=None,
        system_prompt=None,
        kwargs={},
    )

    assert prompt.endswith("/nothink")
    assert meta is not None
    assert meta["thinking_effective"] == "off"
    assert meta["thinking_handled_enable_disable"] is True


# ---------------------------------------------------------------------------
# Truncation auto-close (non-streaming path; streaming is covered in
# tests/test_incremental_thinking_tag_stripper.py)
# ---------------------------------------------------------------------------


def test_strip_thinking_tags_auto_closes_unterminated_block() -> None:
    arch = get_architecture_format("gemma4")
    caps = get_model_capabilities("google/gemma-4-26b-a4b")

    cleaned, reasoning = strip_thinking_tags(
        "<|channel>thought\nlong hidden reasoning that got truncated",
        architecture_format=arch,
        model_capabilities=caps,
    )

    assert cleaned == ""
    assert reasoning == "long hidden reasoning that got truncated (...)"


def test_strip_thinking_tags_auto_closes_trailing_block_after_complete_pairs() -> None:
    arch = {"thinking_tags": ["<think>", "</think>"]}

    cleaned, reasoning = strip_thinking_tags(
        "<think>first</think>Answer.<think>second unterminated",
        architecture_format=arch,
        model_capabilities=None,
    )

    assert cleaned == "Answer."
    assert reasoning == "first\n\nsecond unterminated (...)"


# ---------------------------------------------------------------------------
# Usage details passthrough (invisible reasoning billing evidence)
# ---------------------------------------------------------------------------


def test_openai_compatible_usage_preserves_reasoning_token_details() -> None:
    usage = OpenAICompatibleProvider._build_usage_dict(
        {
            "prompt_tokens": 10,
            "completion_tokens": 170,
            "total_tokens": 180,
            "completion_tokens_details": {"reasoning_tokens": 160},
            "prompt_tokens_details": {"cached_tokens": 4},
        }
    )

    assert usage["input_tokens"] == 10
    assert usage["output_tokens"] == 170
    assert usage["completion_tokens_details"] == {"reasoning_tokens": 160}
    assert usage["prompt_tokens_details"] == {"cached_tokens": 4}


def test_registry_entries_for_problem_models_resolve() -> None:
    nemotron = get_model_capabilities("nvidia/nemotron-3-nano-4b")
    assert nemotron.get("max_tokens") == 262144
    assert nemotron.get("tool_support") == "native"
    assert nemotron.get("thinking_support") is True
    assert detect_architecture("nvidia/nemotron-3-nano-4b") == "nemotron_hybrid_moe"

    grok = get_model_capabilities("x-ai/grok-4")
    assert grok.get("thinking_support") is True
    assert grok.get("reasoning_output") is False
    assert grok.get("reasoning_levels") is None  # never send reasoning_effort to grok-4
