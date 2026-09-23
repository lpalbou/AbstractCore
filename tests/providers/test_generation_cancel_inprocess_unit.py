"""Pins: host cancel + eject safety on the IN-PROCESS HuggingFace lanes.

Before 2026-09-23 `HuggingFaceProvider` did not declare cancel support: a Stop
reached it only between (simulated) stream chunks, i.e. after the whole
generation, and every lane turned exceptions into an "Error: ..." answer —
so even a stop would have been recorded as a completed reply. Contract now:

- transformers: a StoppingCriteria RAISES `GenerationCancelledError` on the
  first decode step after the event is set (the pipeline forwards it to
  `model.generate()`); chunked prefill checks between chunks; the typed error
  is never converted to an "Error:" answer;
- llama-cpp: a logits processor (called once per sampled token) raises on the
  fallback `create_chat_completion` lane; the control-plane token loop checks
  per token;
- eject: `unload_model` cancels the instance's in-flight calls FIRST and waits
  for them (bounded); a call still running at the deadline makes it RAISE
  (nothing freed under a running decode — `Llama.close()` there crashes);
- `load_model` re-warms an unloaded instance.

Real-model cases use `sshleifer/tiny-gpt2` from the local HF cache (skipped
when absent; nothing is downloaded).
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import pytest

from abstractcore.exceptions import GenerationCancelledError, ProviderAPIError

TINY = "sshleifer/tiny-gpt2"


def _tiny_cached() -> bool:
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path.home() / ".cache" / "huggingface" / "hub")
    snap = hub / "models--sshleifer--tiny-gpt2" / "snapshots"
    return snap.is_dir() and any(snap.iterdir())


class CountingEvent(threading.Event):
    """A threading.Event that reports SET after `n` checks (deterministic
    'the host cancelled while the model was decoding')."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.calls = 0

    def is_set(self) -> bool:  # type: ignore[override]
        self.calls += 1
        return self.calls > self.n or super().is_set()


@pytest.fixture(scope="module")
def tiny():
    if not _tiny_cached():
        pytest.skip("sshleifer/tiny-gpt2 not in the local HF cache")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    return HuggingFaceProvider(model=TINY, device="cpu")


def _count_generate_steps(provider) -> List[int]:
    """Wrap model_instance.forward to count decode steps."""
    steps: List[int] = []
    model = provider.model_instance
    orig = model.forward

    def forward(*a, **k):
        steps.append(1)
        return orig(*a, **k)

    model.forward = forward
    return steps


@pytest.mark.parametrize("stream", [False, True])
def test_transformers_cancel_stops_within_one_step_and_is_typed(tiny, stream):
    assert tiny.supports_generation_cancel() is True
    steps = _count_generate_steps(tiny)
    ev = CountingEvent(3)
    try:
        with pytest.raises(GenerationCancelledError):
            out = tiny.generate("hello there", max_output_tokens=400, cancel_event=ev, temperature=0.0, stream=stream)
            if stream:
                list(out)
    finally:
        tiny.model_instance.forward = tiny.model_instance.__class__.forward.__get__(tiny.model_instance)
    # 400 tokens were allowed; the stop landed within a handful of steps.
    assert 0 < len(steps) <= 5, len(steps)


def test_transformers_uncancelled_call_still_answers(tiny):
    r = tiny.generate("hello there", max_output_tokens=8, temperature=0.0)
    assert r.finish_reason != "error" and isinstance(r.content, str)


def test_transformers_attach_appends_to_existing_criteria():
    pytest.importorskip("transformers")
    from transformers import StoppingCriteria, StoppingCriteriaList

    from abstractcore.providers.huggingface_provider import _transformers_attach_host_cancel

    class Mine(StoppingCriteria):
        def __call__(self, input_ids, scores, **kw):
            return False

    kwargs: dict = {"stopping_criteria": StoppingCriteriaList([Mine()])}
    ev = threading.Event()
    assert _transformers_attach_host_cancel(kwargs, ev, model="m") is True
    crits = list(kwargs["stopping_criteria"])
    assert isinstance(crits[0], Mine) and len(crits) == 2
    assert crits[1](None, None) is False
    ev.set()
    with pytest.raises(GenerationCancelledError):
        crits[1](None, None)
    assert _transformers_attach_host_cancel({}, None, model="m") is False


def test_transformers_chunked_prefill_checks_between_chunks(tiny, monkeypatch):
    from abstractcore.providers.huggingface_provider import _TransformersPromptCacheValue

    monkeypatch.setattr(tiny, "_transformers_prefill_step_cached", 4, raising=False)
    state = _TransformersPromptCacheValue(cache=tiny._transformers_empty_native_cache())
    ev = CountingEvent(2)  # two chunks fed, then the cancel lands
    with pytest.raises(GenerationCancelledError):
        tiny._transformers_prefill_cache(state, list(range(1, 30)), cancel_event=ev)
    # State holds exactly the chunks already fed: consistent and reusable.
    assert len(state.prompt_tokens) == 8


def test_llama_cpp_logits_processor_forces_eos_then_caller_raises():
    """llama-cpp-python runs logits processors in a ctypes callback that
    swallows exceptions (measured: a raising processor did NOT stop a
    non-streaming call), so the processor forces EOS and records `fired`;
    the caller raises the typed error afterwards."""
    np = pytest.importorskip("numpy")
    from abstractcore.providers.huggingface_provider import _llama_cpp_host_cancel_logits_processor

    ev = threading.Event()
    proc = _llama_cpp_host_cancel_logits_processor(ev, model="m", eos_token_id=2)
    scores = np.array([0.5, 1.5, -1.0, 3.0], dtype=np.float32)
    out = proc(None, scores.copy())
    assert np.array_equal(out, scores) and proc.fired is False
    proc.raise_if_fired()  # not fired: no-op
    ev.set()
    out = proc(None, scores.copy())
    assert proc.fired is True
    assert int(np.argmax(out)) == 2 and np.isneginf(out[[0, 1, 3]]).all()
    with pytest.raises(GenerationCancelledError):
        proc.raise_if_fired()


def test_gguf_fallback_nonstream_raises_after_a_forced_eos(monkeypatch):
    """The create_chat_completion lane: the call returns normally (EOS forced),
    the provider raises the typed stop instead of returning the cut answer."""
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider, _LlamaCppHostCancel

    p = object.__new__(HuggingFaceProvider)
    p.model = "fake.gguf"
    seen = {}

    class FakeLlama:
        chat_format = "chatml"

        def token_eos(self):
            return 2

        def set_cache(self, c):
            pass

        def create_chat_completion(self, **kw):
            procs = [x for x in kw.get("logits_processor") or [] if isinstance(x, _LlamaCppHostCancel)]
            seen["procs"] = procs
            procs[0].cancel_event.set()
            procs[0](None, __import__("numpy").zeros(4, dtype="float32"))
            return {"choices": [{"message": {"content": "partial"}, "finish_reason": "stop"}]}

    p.llm = FakeLlama()
    for name, fn in {
        "_gguf_build_chat_messages": lambda **k: [{"role": "user", "content": "hi"}],
        "_prepare_generation_kwargs": lambda **k: {},
        "_get_provider_max_tokens_param": lambda k: 16,
        "_gguf_prompt_cache_supports_local_control_plane": lambda: False,
        "_thinking_disable_prefill": lambda x: "",
        "_gguf_normalize_tool_call_arguments_for_template": lambda m: m,
    }.items():
        monkeypatch.setattr(p, name, fn, raising=False)
    p.temperature = 0.0
    with pytest.raises(GenerationCancelledError):
        p._generate_gguf("hi", None, None, None, None, False, None, cancel_event=threading.Event())
    assert seen["procs"], "no host-cancel logits processor reached create_chat_completion"


def test_gguf_control_plane_token_loop_checks_every_token(monkeypatch):
    """The control-plane lane's own token loop stops within one token, raises
    the typed error (never an "Error:" chunk)."""
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    p = object.__new__(HuggingFaceProvider)
    p.model = "fake.gguf"
    produced: List[int] = []

    class FakeLlama:
        def tokenize(self, b, add_bos=False, special=True):
            return [99999]

        def detokenize(self, toks):
            return b"x"

        def set_seed(self, s):
            pass

        def generate(self, tokens, **kw):
            i = 0
            while True:
                i += 1
                produced.append(i)
                yield 1

    p.llm = FakeLlama()
    monkeypatch.setattr(p, "_gguf_control_plane_stop_strings", lambda: [], raising=False)
    monkeypatch.setattr(p, "_gguf_render_prompt_tokens", lambda **k: ("p", (1, 2, 3)), raising=False)
    monkeypatch.setattr(p, "_gguf_compose_cached_prompt_tokens",
                        lambda **k: (k["live_prompt_text"], k["live_prompt_tokens"], {}), raising=False)
    monkeypatch.setattr(p, "_gguf_generation_prompt_boundary", lambda **k: None, raising=False)
    monkeypatch.setattr(p, "_gguf_prefill_prompt_cache", lambda *a, **k: True, raising=False)
    ev = CountingEvent(10)
    gen = p._gguf_control_plane_stream_generate(
        chat_messages=[{"role": "user", "content": "hi"}], cache_obj=None, max_output_tokens=10_000,
        temperature=0.0, top_p=1.0, top_k=1, min_p=0.0, typical_p=1.0, repeat_penalty=1.0,
        presence_penalty=0.0, frequency_penalty=0.0, tfs_z=1.0, mirostat_mode=0, mirostat_tau=5.0,
        mirostat_eta=0.1, seed=None, cancel_event=ev,
    )
    with pytest.raises(GenerationCancelledError):
        list(gen)
    assert len(produced) <= 11, len(produced)


def test_eject_cancels_inflight_then_unloads(tiny):
    """unload_model under a running decode: the decode stops, THEN weights go."""
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    p = HuggingFaceProvider(model=TINY, device="cpu")
    started = threading.Event()
    orig_forward = p.model_instance.forward

    def slow_forward(*a, **k):
        started.set()
        time.sleep(0.02)
        return orig_forward(*a, **k)

    p.model_instance.forward = slow_forward
    out: dict = {}

    def call():
        try:
            p.generate("hello", max_output_tokens=900, temperature=0.0)  # NO host event: private one
            out["result"] = "completed"
        except BaseException as e:  # noqa: BLE001
            out["error"] = e

    th = threading.Thread(target=call, daemon=True)
    th.start()
    assert started.wait(10)
    t0 = time.monotonic()
    p.unload_model(TINY)
    th.join(10)
    assert isinstance(out.get("error"), GenerationCancelledError), out
    assert p._last_unload_inflight["cancelled"] == 1 and p._last_unload_inflight["drained"] is True
    assert time.monotonic() - t0 < 5.0
    assert p.get_model_residency()["loaded"] is False
    # load_model re-warms the SAME instance, which answers again.
    res = p.load_model()
    assert res["action"] == "loaded" and p.get_model_residency()["loaded"] is True
    assert p.load_model()["action"] == "already_loaded"
    r = p.generate("hello", max_output_tokens=4, temperature=0.0)
    assert r.finish_reason != "error"


def test_eject_refuses_when_a_call_does_not_stop(tiny, monkeypatch):
    """A call still running at the drain deadline -> unload RAISES, nothing freed."""
    import abstractcore.providers.generation_cancel as gc_mod

    monkeypatch.setattr(gc_mod, "DEFAULT_EJECT_DRAIN_TIMEOUT_S", 0.2)
    p = tiny
    reg = p._inflight_generations()
    stuck = reg.begin(threading.Event())  # never ends: a call blocked in one native op
    try:
        with pytest.raises(ProviderAPIError, match="Refusing to unload"):
            p.unload_model(TINY)
        assert stuck.event.is_set()  # it WAS told to stop
        assert p.get_model_residency()["loaded"] is True  # and nothing was freed
    finally:
        reg.end(stuck)


def test_load_model_rejects_another_model(tiny):
    with pytest.raises(ValueError):
        tiny.load_model("some/other-model")


def _fake_gguf_provider(monkeypatch, *, n_batch: int = 4):
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    p = object.__new__(HuggingFaceProvider)
    p.model = "fake.gguf"
    calls = {"eval": [], "reset": 0}

    class L:
        def __init__(self):
            self.n_batch = n_batch

        def eval(self, toks):
            calls["eval"].append(list(toks))

        def reset(self):
            calls["reset"] += 1

    p.llm = L()
    monkeypatch.setattr(p, "_gguf_live_context_prefix_len", lambda toks: 0, raising=False)
    monkeypatch.setattr(p, "_gguf_prompt_cache_prefix_state", lambda c, t: (0, None), raising=False)
    import logging

    p.logger = logging.getLogger("test")
    return p, calls


def test_gguf_prefill_is_cancellable_between_batches(monkeypatch):
    p, calls = _fake_gguf_provider(monkeypatch, n_batch=4)
    with pytest.raises(GenerationCancelledError):
        p._gguf_eval_cancellable(p.llm, list(range(20)), CountingEvent(2))
    assert [len(c) for c in calls["eval"]] == [4, 4]
    # No event: ONE eval call, exactly as before.
    calls["eval"].clear()
    p._gguf_eval_cancellable(p.llm, list(range(20)), None)
    assert [len(c) for c in calls["eval"]] == [20]


def test_gguf_prefill_cancel_is_never_retried_cold(monkeypatch):
    """A cancel mid-prefill must not trip the failure recovery (reset + cold
    retry + engine rebuild): it raises straight through."""
    p, calls = _fake_gguf_provider(monkeypatch, n_batch=4)
    with pytest.raises(GenerationCancelledError):
        p._gguf_prefill_prompt_cache(
            None, tuple(range(1, 30)), save_state=False, set_cache=False, cancel_event=CountingEvent(1),
        )
    assert calls["reset"] == 1  # the cold-start reset only; no recovery reset
    assert len(calls["eval"]) == 1


def test_generate_after_eject_reloads_instead_of_answering_an_error(tiny):
    """Measured on the gateway before 2026-09-23: a run after an eject
    'completed' with the answer "Error: MLX model not loaded" (success=true).
    The in-process providers now reload on demand (logged) instead."""
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    p = HuggingFaceProvider(model=TINY, device="cpu")
    p.unload_model(TINY)
    assert p.get_model_residency()["loaded"] is False
    r = p.generate("hello", max_output_tokens=4, temperature=0.0)
    assert r.finish_reason != "error" and not str(r.content).startswith("Error")
    assert p.get_model_residency()["loaded"] is True
