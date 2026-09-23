"""Pins for live TEXT phase feedback (prefill vs generation).

"Thinking…" cannot distinguish a 40-second prefill from a stalled call. Providers
that can observe the boundary report it through a `progress_callback`; the host
(AbstractRuntime) turns each event into one durable ledger record. These tests
cover the three places that contract can silently rot:

1. the emitter's cost discipline — a bounded number of events per call, a
   minimum interval between cadence events, and a first-token event that is
   never dropped by either;
2. the provider boundary — a host callback must NEVER reach a provider that
   cannot honour it (an unknown callable in a strict SDK's kwargs is a 400 at
   best), and must reach one that can under exactly one canonical name;
3. `MLXProvider._observed_generate` — the lane-agnostic seam that replaces
   `generate_fn` when someone is listening. It must return byte-identical text
   (mlx-lm's `generate` IS `"".join(r.text for r in stream_generate(...))`) and
   report the first sampled token as the end of prefill.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.generation_progress import (
    TextProgressEmitter,
    pop_text_progress_callback,
)


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _emitter(events: List[Dict[str, Any]], clock: _Clock, **kwargs: Any) -> TextProgressEmitter:
    return TextProgressEmitter(events.append, provider="mlx", model="m", clock=clock, **kwargs)


# --------------------------------------------------------------------------
# 1. cost discipline
# --------------------------------------------------------------------------


def test_generation_events_are_rate_limited_but_first_token_is_never_dropped():
    events: List[Dict[str, Any]] = []
    clock = _Clock()
    em = _emitter(events, clock, min_interval_s=0.5)
    em.prefill(prompt_tokens=5899, cached_tokens=4096)
    for i in range(1, 21):  # 20 tokens, 0.1 s apart -> 2.0 s of decode
        clock.t += 0.1
        em.generation(generated_tokens=i)
    em.complete(generated_tokens=20, finish_reason="stop")

    phases = [e["phase"] for e in events]
    assert phases[0] == "prefill"
    assert phases[-1] == "complete"
    assert sum(1 for e in events if e.get("first_token")) == 1
    assert events[1]["first_token"] is True and events[1]["phase"] == "generate"
    # 2.0 s of decode at a 0.5 s floor: the first token plus 3 cadence events.
    assert len([e for e in events if e["phase"] == "generate"]) == 4
    cadence = [e for e in events if e["phase"] == "generate" and not e.get("first_token")]
    for event in cadence:
        previous = events[events.index(event) - 1]
        assert event["elapsed_s"] - previous["elapsed_s"] >= 0.5 - 1e-9


def test_updates_keep_flowing_for_the_whole_call_by_default():
    """Progress events are never capped by count (operator, 2026-09-23; ADR-0026).

    The first version capped a call at 64 events; at the 0.5 s cadence that
    went dark after ~60 s while a multi-minute decode was still running, and
    the status bar looked like a hang. A 10-minute call must keep reporting
    at the cadence to the very end.
    """
    events: List[Dict[str, Any]] = []
    clock = _Clock()
    em = _emitter(events, clock, min_interval_s=0.5)
    em.prefill(prompt_tokens=10)
    for i in range(1, 1201):  # 600 s of decode at two ticks per second
        clock.t += 0.5
        em.generation(generated_tokens=i)
    em.complete(generated_tokens=1200, finish_reason="stop")

    assert not hasattr(em, "max_events"), "a count-based cap must not exist (ADR-0026)"
    cadence = [e for e in events if e["phase"] == "generate"]
    assert len(cadence) == 1200, "every 0.5 s tick over the whole 10-minute call must be reported"
    assert cadence[-1]["generated_tokens"] == 1200
    assert events[-1]["phase"] == "complete" and events[-1]["final"] is True
    assert [e["event_index"] for e in events] == list(range(len(events)))


def test_a_single_token_has_no_rate_and_prompt_size_never_shrinks():
    events: List[Dict[str, Any]] = []
    clock = _Clock()
    em = _emitter(events, clock)
    em.prefill(prompt_tokens=779, cached_tokens=690)
    clock.t += 0.2
    # mlx-vlm's first response on a warm cache: one token at a near-zero
    # interval (16,609 tok/s) and `prompt_tokens: 1` for the fed suffix only.
    em.generation(generated_tokens=1, prompt_tokens=1, tokens_per_second=16609.1,
                  prompt_tokens_per_second=10.03)

    first = events[1]
    assert "tokens_per_second" not in first
    assert first["prompt_tokens"] == 779, "a narrower later measurement overwrote the prefill total"
    assert "prompt_tokens_per_second" not in first
    assert first["fed_tokens"] == 89  # derived from 779 - 690


def test_a_raising_callback_is_dropped_and_never_reaches_generation():
    def boom(_event: Dict[str, Any]) -> None:
        raise RuntimeError("consumer exploded")

    em = TextProgressEmitter(boom, provider="mlx", model="m")
    assert em.active is True
    em.prefill(prompt_tokens=10)  # must not raise
    assert em.active is False
    assert em.generation(generated_tokens=1) is False


def test_an_inactive_emitter_costs_nothing():
    em = TextProgressEmitter(None)
    assert em.active is False
    assert em.prefill(prompt_tokens=10) is False
    assert em.generation(generated_tokens=5) is False
    assert em.complete(generated_tokens=5) is False


# --------------------------------------------------------------------------
# 2. the provider boundary
# --------------------------------------------------------------------------


def test_pop_text_progress_callback_removes_every_alias():
    def cb(_event):
        return None

    kwargs = {"on_progress": cb, "progress_callback": cb, "progress_event_callback": cb, "temperature": 0.2}
    assert pop_text_progress_callback(kwargs) is cb
    assert kwargs == {"temperature": 0.2}
    assert pop_text_progress_callback({"on_progress": "not callable"}) is None


class _RecordingProvider:
    """StaticProvider + a `_generate_internal` that records what reached it."""

    @staticmethod
    def build(*, supports: bool):
        from tests.provider_stubs import StaticProvider

        class _Provider(StaticProvider):
            def __init__(self) -> None:
                super().__init__("static-model")
                self.seen: Dict[str, Any] = {}

            def supports_text_progress_events(self) -> bool:
                return supports

            def _generate_internal(self, prompt, **kwargs):  # type: ignore[override]
                self.seen = dict(kwargs)
                return GenerateResponse(content="ok", model="static-model", finish_reason="stop")

        return _Provider()


@pytest.mark.parametrize("alias", ["on_progress", "progress_callback", "progress_event_callback"])
def test_a_provider_without_a_real_signal_never_receives_the_callback(alias):
    provider = _RecordingProvider.build(supports=False)
    calls: List[Any] = []
    provider.generate(prompt="hi", **{alias: calls.append})
    for key in ("on_progress", "progress_callback", "progress_event_callback", "_text_progress_callback"):
        assert key not in provider.seen, f"{key} leaked into provider kwargs"
    assert calls == []


def test_a_provider_that_reports_phases_receives_exactly_one_canonical_kwarg():
    from abstractcore.providers.generation_progress import PROGRESS_KWARG

    provider = _RecordingProvider.build(supports=True)
    calls: List[Any] = []
    provider.generate(prompt="hi", on_progress=calls.append)
    assert callable(provider.seen.get(PROGRESS_KWARG))
    assert "on_progress" not in provider.seen


def test_base_providers_default_to_silence():
    from tests.provider_stubs import StaticProvider

    assert StaticProvider("static-model").supports_text_progress_events() is False


# --------------------------------------------------------------------------
# 3. MLXProvider._observed_generate (covers the mlx-lm lane without weights)
# --------------------------------------------------------------------------


def _mlx_lm_response(text: str, token: int, generated: int, *, finish=None, prompt_tokens=6906):
    """mlx_lm.generate.GenerationResponse shape (no `cached_tokens` on this lane)."""

    return SimpleNamespace(
        text=text, token=token, from_draft=False, logprobs=None,
        prompt_tokens=prompt_tokens, prompt_tps=4390.4,
        generation_tokens=generated, generation_tps=45.0 if generated > 1 else 0.0,
        peak_memory=4.1, finish_reason=finish,
    )


def _provider_with_stream(chunks, clock):
    from abstractcore.providers.mlx_provider import MLXProvider

    provider = MLXProvider.__new__(MLXProvider)
    provider.llm = object()
    provider.tokenizer = object()
    seen: Dict[str, Any] = {}

    def stream_generate_fn(model, tokenizer, prompt, **kwargs):
        seen["prompt"] = prompt
        seen["kwargs"] = dict(kwargs)
        for chunk in chunks:
            clock.t += 0.25
            yield chunk

    provider.stream_generate_fn = stream_generate_fn
    return provider, seen


def test_observed_generate_returns_the_same_text_and_reports_the_prefill_boundary():
    clock = _Clock()
    events: List[Dict[str, Any]] = []
    words = ["Hello", " there", ",", " ledger", "."]
    chunks = [
        _mlx_lm_response(word, 100 + i, i + 1, finish="stop" if i == len(words) - 1 else None)
        for i, word in enumerate(words)
    ]
    provider, seen = _provider_with_stream(chunks, clock)
    emitter = _emitter(events, clock, min_interval_s=0.5)
    emitter.prefill(prompt_tokens=6906)

    text = provider._observed_generate(
        emitter, prompt="rendered prompt", max_tokens=64, prompt_cache=None,
        sampler_kwargs={"sampler": "s"}, embed_kwargs={},
    )

    assert text == "Hello there, ledger."
    assert seen["prompt"] == "rendered prompt"
    assert seen["kwargs"]["max_tokens"] == 64 and seen["kwargs"]["sampler"] == "s"
    assert "verbose" not in seen["kwargs"], "stream_generate has no verbose parameter"

    assert [e["phase"] for e in events][0] == "prefill"
    assert events[1]["phase"] == "generate" and events[1]["first_token"] is True
    assert events[1]["ttft_s"] == pytest.approx(0.25, abs=1e-6)
    assert events[-1]["phase"] == "complete" and events[-1]["final"] is True
    assert events[-1]["generated_tokens"] == 5
    assert events[-1]["finish_reason"] == "stop"
    counts = [e.get("generated_tokens", 0) for e in events]
    assert counts == sorted(counts)
    # `cached_tokens` is native-only: this lane must report nothing, not zero.
    assert all("cached_tokens" not in e for e in events)


def test_observed_generate_still_reports_a_terminal_event_when_the_stream_raises():
    clock = _Clock()
    events: List[Dict[str, Any]] = []

    def exploding(model, tokenizer, prompt, **kwargs):
        clock.t += 0.25
        yield _mlx_lm_response("partial", 1, 1)
        raise RuntimeError("metal fault")

    from abstractcore.providers.mlx_provider import MLXProvider

    provider = MLXProvider.__new__(MLXProvider)
    provider.llm = object()
    provider.tokenizer = object()
    provider.stream_generate_fn = exploding
    emitter = _emitter(events, clock)
    emitter.prefill(prompt_tokens=10)

    with pytest.raises(RuntimeError):
        provider._observed_generate(emitter, prompt="p", max_tokens=8, prompt_cache=None,
                                    sampler_kwargs={}, embed_kwargs={})
    assert events[-1]["phase"] == "complete", "a failed call left the UI showing a live phase forever"


def test_mlx_declares_it_can_report_phases():
    from abstractcore.providers.mlx_provider import MLXProvider

    provider = MLXProvider.__new__(MLXProvider)
    assert provider.supports_text_progress_events() is True


def _single_generate_provider(chunks, clock):
    """A `_single_generate`-capable provider with BOTH call shapes stubbed."""

    from abstractcore.providers.mlx_provider import MLXProvider

    provider = MLXProvider.__new__(MLXProvider)
    provider.llm = object()
    provider.tokenizer = SimpleNamespace(encode=lambda text: list(str(text).split()))
    provider.model = "mlx-community/Qwen3.5-4B-4bit"
    provider.architecture_config = None
    provider.model_capabilities = None
    provider.logger = SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None)
    provider._mtp_processor = object()  # skip the mlx sampler build; no mlx import needed
    provider._native_sampling_kwargs = {}
    provider._native_runtime = None
    provider._mtp_last_result = None
    used: Dict[str, int] = {"generate_fn": 0, "stream_generate_fn": 0}

    def generate_fn(model, tokenizer, prompt=None, **kwargs):
        used["generate_fn"] += 1
        return "".join(c.text for c in chunks)

    def stream_generate_fn(model, tokenizer, prompt, **kwargs):
        used["stream_generate_fn"] += 1
        for chunk in chunks:
            clock.t += 0.25
            yield chunk

    provider.generate_fn = generate_fn
    provider.stream_generate_fn = stream_generate_fn
    return provider, used


def test_single_generate_uses_the_observed_stream_only_when_someone_subscribed():
    """The dispatch itself, not just `_observed_generate`.

    Without a subscriber the call must stay on `generate_fn` byte for byte —
    the whole point of making the streamed lane opt-in — and with one it must
    switch, or the phase events never happen in production.
    """

    clock = _Clock()
    chunks = [
        _mlx_lm_response("Hel", 1, 1),
        _mlx_lm_response("lo", 2, 2),
        _mlx_lm_response("!", 3, 3, finish="stop"),
    ]

    quiet, used_quiet = _single_generate_provider(chunks, clock)
    silent = quiet._single_generate("p", 16, 0.2, 0.9, None, None, None, usage_prompt="p",
                                    progress=TextProgressEmitter(None))
    assert used_quiet == {"generate_fn": 1, "stream_generate_fn": 0}

    events: List[Dict[str, Any]] = []
    loud, used_loud = _single_generate_provider(chunks, clock)
    watched = loud._single_generate("p", 16, 0.2, 0.9, None, None, None, usage_prompt="p",
                                    progress=_emitter(events, clock))
    assert used_loud == {"generate_fn": 0, "stream_generate_fn": 1}

    assert watched.content == silent.content == "Hello!"
    assert watched.finish_reason == silent.finish_reason
    assert [e["phase"] for e in events] == ["generate", "generate", "complete"]
    assert events[0]["first_token"] is True
