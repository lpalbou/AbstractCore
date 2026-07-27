from __future__ import annotations

from types import SimpleNamespace

import pytest

from abstractcore.providers.anthropic_provider import AnthropicProvider


def _fake_response(*, blocks: list[SimpleNamespace], model: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=blocks,
        model=model,
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def test_anthropic_opus_4_6_thinking_high_maps_to_adaptive_effort(monkeypatch) -> None:
    provider = AnthropicProvider(model="claude-opus-4-6", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    provider.generate(prompt="hi", thinking="high", temperature=0, max_output_tokens=16)

    call_params = captured["call_params"]
    assert call_params["thinking"] == {"type": "adaptive"}
    assert call_params["output_config"]["effort"] == "high"


def test_anthropic_sampling_knobs_are_forwarded_only_when_requested(monkeypatch) -> None:
    provider = AnthropicProvider(model="claude-opus-4-6", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    provider.generate(prompt="hi", temperature=0, max_output_tokens=16)
    call_params = captured["call_params"]
    assert "top_p" not in call_params
    assert "top_k" not in call_params

    provider.generate(prompt="hi", temperature=0, top_p=0.8, top_k=20, max_output_tokens=16)
    call_params = captured["call_params"]
    assert call_params["top_p"] == 0.8
    assert call_params["top_k"] == 20


def test_anthropic_opus_4_6_thinking_xhigh_passes_through(monkeypatch) -> None:
    # xhigh is a declared effort level for this model; the provider must send the
    # requested level verbatim, never silently escalate to the pricier "max".
    provider = AnthropicProvider(model="claude-opus-4-6", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    provider.generate(prompt="hi", thinking="xhigh", temperature=0, max_output_tokens=16)

    call_params = captured["call_params"]
    assert call_params["thinking"] == {"type": "adaptive"}
    assert call_params["output_config"]["effort"] == "xhigh"


def test_anthropic_sonnet_4_6_thinking_xhigh_passes_through(monkeypatch) -> None:
    provider = AnthropicProvider(model="claude-sonnet-4-6", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    provider.generate(prompt="hi", thinking="xhigh", temperature=0, max_output_tokens=16)

    call_params = captured["call_params"]
    assert call_params["thinking"] == {"type": "adaptive"}
    assert call_params["output_config"]["effort"] == "xhigh"


def test_anthropic_thinking_on_without_level_omits_effort(monkeypatch) -> None:
    # thinking='on' must not silently pick an effort; adaptive alone lets the API
    # default apply.
    provider = AnthropicProvider(model="claude-opus-4-6", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    provider.generate(prompt="hi", thinking=True, temperature=0, max_output_tokens=16)

    call_params = captured["call_params"]
    assert call_params["thinking"] == {"type": "adaptive"}
    assert "output_config" not in call_params


def test_anthropic_legacy_thinking_level_maps_to_budget_tokens(monkeypatch) -> None:
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    provider.generate(prompt="hi", thinking="high", temperature=0, max_output_tokens=5000)

    call_params = captured["call_params"]
    assert call_params["thinking"]["type"] == "enabled"
    # High maps to 8192, but Anthropic requires max_tokens STRICTLY greater than
    # the budget: clamp to max_output_tokens - 1.
    assert call_params["thinking"]["budget_tokens"] == 4999


def test_anthropic_budget_thinking_refused_when_max_tokens_too_small(monkeypatch) -> None:
    # budget_tokens has a hard minimum of 1024 and max_tokens must exceed it;
    # a 1024-token cap cannot satisfy both, so thinking is refused loudly
    # instead of sending a request the API is documented to reject.
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    with pytest.warns(RuntimeWarning):
        provider.generate(prompt="hi", thinking="high", temperature=0, max_output_tokens=1024)

    call_params = captured["call_params"]
    assert "thinking" not in call_params


def test_anthropic_budget_floor_stays_at_1024(monkeypatch) -> None:
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    # low maps to 1024 which already sits at the API floor; a 1200-token cap
    # keeps budget at the floor (not clamped below it).
    provider.generate(prompt="hi", thinking="low", temperature=0, max_output_tokens=1200)

    call_params = captured["call_params"]
    assert call_params["thinking"]["budget_tokens"] == 1024


def test_anthropic_adaptive_only_model_thinking_off_omits_parameter(monkeypatch) -> None:
    # Opus 4.7 rejects thinking type "disabled" (HTTP 400); off is expressed by
    # omitting the thinking parameter, with a warning.
    provider = AnthropicProvider(model="claude-opus-4-7", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    with pytest.warns(RuntimeWarning):
        provider.generate(prompt="hi", thinking="off", temperature=0, max_output_tokens=16)

    call_params = captured["call_params"]
    assert "thinking" not in call_params


class _FakeAnthropicStream:
    def __init__(self, events):
        self._events = list(events)

    def __enter__(self):
        return iter(self._events)

    def __exit__(self, *_exc):
        return False


def test_anthropic_streaming_captures_thinking_deltas(monkeypatch) -> None:
    provider = AnthropicProvider(model="claude-opus-4-6", api_key="test")

    events = [
        SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="thinking")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(thinking="Let me ")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(thinking="think.")),
        # signature_delta events carry no renderable text and must be skipped.
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(signature="sig")),
        SimpleNamespace(type="content_block_stop"),
        SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="text")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="Answer")),
        SimpleNamespace(type="content_block_stop"),
        SimpleNamespace(type="message_stop"),
    ]

    monkeypatch.setattr(
        provider.client.messages, "stream", lambda **_kw: _FakeAnthropicStream(events)
    )

    chunks = list(
        provider.generate(prompt="hi", thinking="high", temperature=0, max_output_tokens=64, stream=True)
    )

    visible = "".join(c.content or "" for c in chunks)
    assert visible == "Answer"

    deltas = [
        c.metadata.get("reasoning_delta")
        for c in chunks
        if isinstance(c.metadata, dict) and c.metadata.get("reasoning_delta")
    ]
    assert deltas == ["Let me ", "think."]

    # BaseProvider trailing aggregate: complete reasoning for stream folds.
    final = chunks[-1]
    assert isinstance(final.metadata, dict)
    assert final.metadata.get("reasoning") == "Let me think."


def test_anthropic_thinking_off_sets_disabled(monkeypatch) -> None:
    provider = AnthropicProvider(model="claude-opus-4-6", api_key="test")

    captured = {}

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_response(blocks=[SimpleNamespace(type="text", text="ok")], model=provider.model)

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    provider.generate(prompt="hi", thinking="off", temperature=0, max_output_tokens=16)

    call_params = captured["call_params"]
    assert call_params["thinking"] == {"type": "disabled"}
    assert "output_config" not in call_params


def test_anthropic_thinking_blocks_are_captured_as_reasoning(monkeypatch) -> None:
    provider = AnthropicProvider(model="claude-opus-4-6", api_key="test")

    def fake_create(**_call_params):
        return _fake_response(
            blocks=[
                SimpleNamespace(type="thinking", thinking="r"),
                SimpleNamespace(type="text", text="final"),
            ],
            model=provider.model,
        )

    monkeypatch.setattr(provider.client.messages, "create", fake_create)

    resp = provider.generate(prompt="hi", thinking="high", temperature=0, max_output_tokens=16)
    assert resp.content == "final"
    assert isinstance(resp.metadata, dict)
    assert resp.metadata.get("reasoning") == "r"
