"""Pins for host cancellation of an IN-FLIGHT generation (Stop must stop the model).

Incident 2026-09-22: a run was cancelled from the UI, the runtime marked it
CANCELLED within seconds, and the model kept decoding at full CPU for an hour
because nothing handed the cancel to the call that was already running. The
provider boundary consumed `cancel_event` for retry backoff only.

These tests pin every hop that lives in AbstractCore (contract:
`abstractcore/providers/generation_cancel.py`):

1. the BaseProvider boundary forwards the host's event under ONE canonical
   kwarg to providers that declare `supports_generation_cancel()`, never to
   others, refuses to start when the event is already set, and stops a
   stream between chunks for EVERY provider (closing the upstream stream);
2. the MLX mlx-lm / mlx-vlm lanes check the event per sampled token, close
   their generator, and raise `GenerationCancelledError` — never an
   "Error: ..." content chunk, never a truncated answer dressed as complete;
3. the MLX native-runtime lane binds the caller's event to the request view
   and hands THAT event to the scheduler (which checks it per token);
4. a cancel is request-local: the retry layer never retries it.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.exceptions import GenerationCancelledError
from abstractcore.providers.generation_cancel import CANCEL_KWARG
from abstractcore.providers.generation_progress import TextProgressEmitter


# --------------------------------------------------------------------------
# 1. BaseProvider boundary
# --------------------------------------------------------------------------


def _recording_provider(*, supports: bool, chunks: List[str] | None = None):
    from tests.provider_stubs import StaticProvider

    class _Provider(StaticProvider):
        def __init__(self) -> None:
            super().__init__("static-model")
            self.seen: Dict[str, Any] = {}
            self.calls = 0
            self.pulled = 0
            self.closed = False

        def supports_generation_cancel(self) -> bool:
            return supports

        def _generate_internal(self, prompt, **kwargs):  # type: ignore[override]
            self.calls += 1
            self.seen = dict(kwargs)
            if not kwargs.get("stream"):
                return GenerateResponse(content="ok", model="static-model", finish_reason="stop")
            owner = self

            class _Upstream:
                def __init__(self) -> None:
                    self._it = iter(chunks or [])

                def __iter__(self):
                    return self

                def __next__(self):
                    text = next(self._it)
                    owner.pulled += 1
                    return GenerateResponse(content=text, model="static-model")

                def close(self) -> None:
                    owner.closed = True

            return _Upstream()

    return _Provider()


def test_a_cancel_capable_provider_receives_the_same_event_under_one_kwarg():
    provider = _recording_provider(supports=True)
    event = threading.Event()
    provider.generate(prompt="hi", cancel_event=event)
    assert provider.seen.get(CANCEL_KWARG) is event
    assert "cancel_event" not in provider.seen


def test_a_provider_without_cancel_support_never_receives_the_event():
    provider = _recording_provider(supports=False)
    provider.generate(prompt="hi", cancel_event=threading.Event())
    assert CANCEL_KWARG not in provider.seen
    assert "cancel_event" not in provider.seen


def test_base_providers_default_to_no_cancel_support():
    from tests.provider_stubs import StaticProvider

    assert StaticProvider("static-model").supports_generation_cancel() is False


def test_an_event_set_before_the_call_never_reaches_the_model():
    provider = _recording_provider(supports=True)
    event = threading.Event()
    event.set()
    with pytest.raises(GenerationCancelledError):
        provider.generate(prompt="hi", cancel_event=event)
    assert provider.calls == 0


@pytest.mark.parametrize("supports", [True, False])
def test_the_base_stream_loop_stops_between_chunks_and_closes_upstream(supports):
    """Every provider's stream (HTTP included) stops within one chunk."""

    provider = _recording_provider(supports=supports, chunks=["a", "b", "c", "d", "e"])
    event = threading.Event()
    stream = provider.generate(prompt="hi", stream=True, cancel_event=event)
    received = []
    with pytest.raises(GenerationCancelledError):
        for chunk in stream:
            received.append(chunk.content)
            if len(received) == 2:
                event.set()
    assert received == ["a", "b"]
    assert provider.pulled <= 3, "the stream kept pulling chunks after the cancel"
    assert provider.closed is True, "the upstream stream (HTTP response) was not closed"


def test_a_cancel_is_request_local_and_never_retried():
    from abstractcore.core.retry import RetryManager

    attempts = []

    def call():
        attempts.append(1)
        raise GenerationCancelledError("stopped")

    with pytest.raises(GenerationCancelledError):
        RetryManager().execute_with_retry(call, provider_key="p")
    assert len(attempts) == 1


# --------------------------------------------------------------------------
# 2. MLX mlx-lm / mlx-vlm lanes (no weights, stubbed generators)
# --------------------------------------------------------------------------


def _resp(text: str, n: int, *, finish=None):
    return SimpleNamespace(
        text=text, token=n, from_draft=False, logprobs=None, prompt_tokens=100,
        prompt_tps=1000.0, generation_tokens=n, generation_tps=20.0, peak_memory=1.0,
        finish_reason=finish,
    )


class _Decoder:
    """A stream_generate stand-in that counts pulls and observes close()."""

    def __init__(self, total: int = 1000, *, set_after: int | None = None, event=None) -> None:
        self.total = total
        self.set_after = set_after
        self.event = event
        self.pulled = 0
        self.closed = False
        self.kwargs: Dict[str, Any] = {}

    def __call__(self, model, tokenizer, prompt, **kwargs):
        self.kwargs = dict(kwargs)
        try:
            for i in range(1, self.total + 1):
                self.pulled += 1
                if self.set_after is not None and i == self.set_after + 1 and self.event is not None:
                    self.event.set()  # the Stop arrives while token i is being decoded
                yield _resp("w ", i, finish="length" if i == self.total else None)
        finally:
            self.closed = True


def _mlx(decoder, *, mtp: bool = False):
    from abstractcore.providers.mlx_provider import MLXProvider

    provider = MLXProvider.__new__(MLXProvider)
    provider.llm = object()
    provider.tokenizer = SimpleNamespace(encode=lambda text: list(str(text).split()))
    provider.model = "mlx-community/Qwen3.5-4B-4bit"
    provider.architecture_config = None
    provider.model_capabilities = None
    provider.logger = SimpleNamespace(debug=lambda *a, **k: None, warning=lambda *a, **k: None)
    provider._mtp_processor = object() if mtp else None
    provider._native_sampling_kwargs = {}
    provider._native_runtime = None
    provider._mtp_last_result = None
    provider.stream_generate_fn = decoder
    provider.generate_fn = lambda *a, **k: pytest.fail("the uninterruptible generate_fn was used")
    provider._build_mlx_sampler = lambda *a, **k: None
    return provider


def test_mlx_declares_generation_cancel_support():
    from abstractcore.providers.mlx_provider import MLXProvider

    assert MLXProvider.__new__(MLXProvider).supports_generation_cancel() is True


@pytest.mark.parametrize("mtp", [False, True])
def test_observed_generate_stops_within_one_token_and_closes_the_generator(mtp):
    event = threading.Event()
    decoder = _Decoder(set_after=5, event=event)
    provider = _mlx(decoder, mtp=mtp)
    events: List[Dict[str, Any]] = []
    with pytest.raises(GenerationCancelledError) as caught:
        provider._observed_generate(
            TextProgressEmitter(events.append, provider="mlx", model="m"),
            prompt="p", max_tokens=100000, prompt_cache=None,
            sampler_kwargs={}, embed_kwargs={}, cancel_event=event,
        )
    assert decoder.pulled == 6, f"decoded {decoder.pulled} tokens; the cancel landed during token 6"
    assert decoder.closed is True
    assert caught.value.generated_tokens == 5
    assert caught.value.partial_text == "w " * 5
    assert events[-1]["phase"] == "complete" and events[-1]["finish_reason"] == "cancelled"


def test_single_generate_with_an_event_uses_the_cancellable_lane_even_unobserved():
    event = threading.Event()
    decoder = _Decoder(set_after=3, event=event)
    provider = _mlx(decoder, mtp=True)
    with pytest.raises(GenerationCancelledError):
        provider._single_generate("p", 100000, 0.2, 0.9, None, None, None, usage_prompt="p",
                                  progress=TextProgressEmitter(None), cancel_event=event)
    assert decoder.pulled == 4 and decoder.closed


def test_plain_mlx_lm_lane_aborts_prefill_through_its_progress_hook():
    event = threading.Event()
    provider = _mlx(_Decoder(), mtp=False)
    # Capture the hook the lane hands mlx-lm, then drive it like mlx-lm's prefill loop.
    captured: Dict[str, Any] = {}

    def capturing(model, tokenizer, prompt, **kwargs):
        captured.update(kwargs)
        hook = kwargs["prompt_progress_callback"]
        hook(0, 8192)
        hook(2048, 8192)
        event.set()
        hook(4096, 8192)  # must raise: prefill stops at this chunk boundary
        yield _resp("never", 1)

    provider.stream_generate_fn = capturing
    with pytest.raises(GenerationCancelledError, match="during prefill"):
        provider._observed_generate(
            TextProgressEmitter(None), prompt="p", max_tokens=10, prompt_cache=None,
            sampler_kwargs={}, embed_kwargs={}, cancel_event=event,
        )
    assert callable(captured.get("prompt_progress_callback"))


def test_mtp_lane_is_not_handed_an_mlx_lm_only_prefill_hook():
    event = threading.Event()
    decoder = _Decoder(total=3)
    provider = _mlx(decoder, mtp=True)
    provider._observed_generate(TextProgressEmitter(None), prompt="p", max_tokens=10, prompt_cache=None,
                                sampler_kwargs={}, embed_kwargs={}, cancel_event=event)
    assert "prompt_progress_callback" not in decoder.kwargs


def test_without_an_event_the_lane_is_byte_identical_to_before():
    # Nobody listening and nothing to cancel: no prefill hook at all.
    decoder = _Decoder(total=3)
    provider = _mlx(decoder, mtp=False)
    text = provider._observed_generate(TextProgressEmitter(None), prompt="p", max_tokens=10,
                                       prompt_cache=None, sampler_kwargs={}, embed_kwargs={})
    assert text == "w w w "
    assert "prompt_progress_callback" not in decoder.kwargs

    # A phase subscriber (Mission J, mid-prefill progress) gets mlx-lm's hook,
    # and without a cancel event that hook only reports — it never aborts.
    decoder = _Decoder(total=3)
    provider = _mlx(decoder, mtp=False)
    text = provider._observed_generate(TextProgressEmitter(lambda e: None), prompt="p", max_tokens=10,
                                       prompt_cache=None, sampler_kwargs={}, embed_kwargs={})
    assert text == "w w w "
    decoder.kwargs["prompt_progress_callback"](128, 1000)  # must not raise


def test_stream_lane_raises_the_cancel_instead_of_an_error_chunk_and_closes_the_source():
    event = threading.Event()
    decoder = _Decoder(set_after=2, event=event)
    provider = _mlx(decoder, mtp=True)
    seen = []
    with pytest.raises(GenerationCancelledError):
        for chunk in provider._stream_generate("p", 1000, 0.2, 0.9, cancel_event=event):
            seen.append(chunk.content)
    assert seen == ["w ", "w "]
    assert all(not str(c).startswith("Error") for c in seen)
    assert decoder.closed is True


def test_stream_generate_with_tools_accepts_the_progress_the_core_passes():
    """Regression: `_generate_core` passed `progress=` to a signature without it,
    so every stream=True call on the mlx-lm/mlx-vlm lanes raised TypeError."""

    decoder = _Decoder(total=2)
    provider = _mlx(decoder, mtp=True)
    events: List[Dict[str, Any]] = []
    chunks = list(provider._stream_generate_with_tools(
        "p", 10, 0.2, 0.9, progress=TextProgressEmitter(events.append, provider="mlx", model="m"),
        cancel_event=None,
    ))
    assert "".join(c.content or "" for c in chunks) == "w w "
    assert events and events[-1]["phase"] == "complete"


# --------------------------------------------------------------------------
# 3. MLX native-runtime lane: the CALLER's event is the one the scheduler sees
# --------------------------------------------------------------------------


def _native_provider():
    from abstractcore.providers.mlx_provider import MLXProvider
    from abstractcore.providers.speculation import SpeculationRequest

    value = MLXProvider.__new__(MLXProvider)
    value.model = "local/native"
    value._mtp_processor = object()
    value._mtp_drafter = object()
    value._mtp_kind = "mtp"
    value._mtp_drafter_id = "local/head"
    value._speculation_request = SpeculationRequest(mode="native_mtp", num_draft_tokens=2)
    value._mtp_block_size = 2
    value._mlx_cache_scope = "test-owner"
    value._native_owner_id = "lease"
    value._native_runtime = object()
    value._apply_per_call_speculation(None)
    return value


def test_native_generate_internal_binds_the_callers_event_to_the_request_view(monkeypatch):
    from abstractcore.providers.mlx_provider import MLXProvider

    seen: Dict[str, Any] = {}

    def fake_unlocked(self, prompt, **kwargs):
        seen["event"] = self._native_cancel_event
        seen["kwargs"] = kwargs
        return GenerateResponse(content="ok", model="m", finish_reason="stop")

    monkeypatch.setattr(MLXProvider, "_generate_internal_unlocked", fake_unlocked)
    monkeypatch.setattr(MLXProvider, "_publish_native_request_status", lambda self: None)
    event = threading.Event()
    _native_provider()._generate_internal("p", **{CANCEL_KWARG: event})
    assert seen["event"] is event
    assert CANCEL_KWARG not in seen["kwargs"]


def test_native_decode_hands_the_view_event_to_the_scheduler():
    from abstractcore.providers.mlx_runtime import NativeResult

    event = threading.Event()
    view = _native_provider()._native_request_view(event)
    handed: Dict[str, Any] = {}

    class Handle:
        def __iter__(self):
            yield NativeResult(text="done", finish_reason="stop")

        def close(self):
            pass

    def stream(request, **kwargs):
        handed.update(kwargs)
        return Handle()

    view._native_runtime = SimpleNamespace(stream=stream)
    view._observe_native_runtime_result = lambda result: None
    list(view._mtp_stream_generate_fn(None, None, "prompt", max_tokens=5))
    assert handed.get("cancel_event") is event


def test_generate_core_propagates_the_cancel_instead_of_an_error_answer():
    """Live find (mission H hermetic proof): the non-scheduled lane turned every
    exception into `GenerateResponse(content="Error: ...", finish_reason="error")`,
    so a stopped call reached the runtime as a COMPLETED answer whose text was
    the cancel message. A host cancel must surface as the cancel itself."""

    event = threading.Event()
    provider = _mlx(_Decoder(set_after=3, event=event), mtp=False)
    provider.max_tokens, provider.max_output_tokens, provider.max_input_tokens = 4096, 2048, 2048
    provider.temperature, provider.top_p, provider.top_k, provider.seed = 0.7, 0.9, None, None
    with pytest.raises(GenerationCancelledError):
        provider._generate_core("hello", messages=None, system_prompt=None, **{CANCEL_KWARG: event})
