"""Pins: eject (unload) safety and memory release around host cancels.

- `InflightGenerations`: every cancel-evented call is registered while it
  runs; `cancel_all` sets each event and waits (bounded, explicit) for it to
  end; a call that does not stop is REPORTED (and in-process unloads refuse).
- BaseProvider registers non-streaming calls for their whole duration and
  streams while their body runs; an unstarted stream is not "running".
- A cancelled call's traceback no longer pins the aborted decode's tensors
  (measured live: 1.1 GB of MPS pool retained across an in-flight eject
  before `_release_cancelled_frames`, 6.5 MB after).
- HTTP providers: unload cancels OUR in-flight requests first (LM Studio would
  otherwise answer "Model unloaded" mid-stream and the retry layer would
  JIT-reload the model), and REPLACES the sync httpx client instead of leaving
  a closed one that raises on every later request of the pooled instance.
"""

from __future__ import annotations

import gc
import threading
import time
import weakref
from typing import Any, Dict, List

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.exceptions import GenerationCancelledError
from abstractcore.providers.generation_cancel import InflightGenerations, cancelled_error


def test_registry_cancel_all_drains_running_calls():
    reg = InflightGenerations()
    ev = threading.Event()
    call = reg.begin(ev)
    assert reg.active() == 1

    def worker():
        ev.wait(5)
        reg.end(call)

    threading.Thread(target=worker, daemon=True).start()
    out = reg.cancel_all(reason="test", drain_timeout_s=5)
    assert out["cancelled"] == 1 and out["drained"] is True and out["still_running"] == 0
    assert reg.active() == 0


def test_registry_reports_a_call_that_does_not_stop():
    reg = InflightGenerations()
    stuck = reg.begin(threading.Event())
    t0 = time.monotonic()
    out = reg.cancel_all(reason="test", drain_timeout_s=0.2)
    assert out["drained"] is False and out["still_running"] == 1
    assert 0.15 <= time.monotonic() - t0 < 2.0
    assert stuck.event.is_set()
    reg.end(stuck)


def test_registry_forgets_garbage_collected_calls():
    reg = InflightGenerations()
    call = reg.begin(threading.Event())
    assert reg.active() == 1
    del call
    gc.collect()
    assert reg.active() == 0
    assert reg.begin(None) is None


def _provider(stream_hold: threading.Event | None = None):
    from tests.provider_stubs import StaticProvider

    class _P(StaticProvider):
        def supports_generation_cancel(self) -> bool:
            return True

        def _generate_internal(self, prompt, **kwargs):  # type: ignore[override]
            ev = kwargs["_cancel_event"]
            if not kwargs.get("stream"):
                ev.wait(5)
                raise cancelled_error(provider="p", model="m", where="test")

            def gen():
                while True:
                    if ev.is_set():
                        raise cancelled_error(provider="p", model="m", where="test")
                    yield GenerateResponse(content="x", model="m")
                    time.sleep(0.01)

            return gen()

    return _P("static-model")


def test_base_registers_a_nonstream_call_and_eject_stops_it():
    p = _provider()
    out: Dict[str, Any] = {}

    def call():
        try:
            p.generate("x", cancel_event=threading.Event())
        except BaseException as e:  # noqa: BLE001
            out["error"] = e

    th = threading.Thread(target=call, daemon=True)
    th.start()
    deadline = time.monotonic() + 5
    while p._inflight_generations().active() == 0 and time.monotonic() < deadline:
        time.sleep(0.005)
    res = p.cancel_inflight_generations(reason="test eject", drain_timeout_s=5)
    th.join(5)
    assert res["cancelled"] == 1 and res["drained"] is True
    assert isinstance(out.get("error"), GenerationCancelledError)
    assert p._inflight_generations().active() == 0


def test_base_registers_a_stream_only_while_it_runs():
    p = _provider()
    stream = p.generate("x", stream=True, cancel_event=threading.Event())
    assert p._inflight_generations().active() == 0  # created, not started
    it = iter(stream)
    next(it)
    assert p._inflight_generations().active() == 1
    res = p.cancel_inflight_generations(reason="test eject", drain_timeout_s=0.2)
    assert res["cancelled"] == 1
    with pytest.raises(GenerationCancelledError):
        for _ in it:
            pass
    assert p._inflight_generations().active() == 0


def test_cancelled_call_does_not_pin_its_frames():
    """The aborted call's locals (a stand-in for the decode's KV tensors) are
    released as soon as the cancel propagates out of generate()."""
    from tests.provider_stubs import StaticProvider

    holder: List[Any] = []

    class Big:
        pass

    class _P(StaticProvider):
        def supports_generation_cancel(self) -> bool:
            return True

        def _generate_internal(self, prompt, **kwargs):  # type: ignore[override]
            kv_cache = Big()  # noqa: F841 - pinned by this frame
            holder.append(weakref.ref(kv_cache))
            raise cancelled_error(provider="p", model="m", where="test")

    p = _P("static-model")
    try:
        p.generate("x", cancel_event=threading.Event())
    except GenerationCancelledError as exc:
        kept = exc  # the caller still HOLDS the exception (the runtime does)
        assert holder[0]() is None, "the cancelled call's frame still pins its tensors"
        del kept


@pytest.mark.parametrize("kind", ["openai-compatible", "ollama"])
def test_http_unload_replaces_the_sync_client(kind):
    if kind == "openai-compatible":
        from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

        p = OpenAICompatibleProvider(model="m", base_url="http://127.0.0.1:9/v1")
    else:
        from abstractcore.providers.ollama_provider import OllamaProvider

        p = OllamaProvider(model="m", base_url="http://127.0.0.1:9")
        p._running_model_entries = lambda: []  # type: ignore[method-assign]
        p.client.post = lambda *a, **k: type("R", (), {"raise_for_status": lambda self: None})()  # type: ignore[assignment]
    before = p.client
    p.unload_model("m")
    assert before.is_closed is True
    assert p.client is not before and p.client.is_closed is False


def test_lmstudio_unload_cancels_our_requests_before_the_server_unload(monkeypatch):
    from abstractcore.providers.lmstudio_provider import LMStudioProvider

    p = LMStudioProvider(model="m", base_url="http://127.0.0.1:9/v1")
    ev = threading.Event()
    call = p._inflight_generations().begin(ev)
    seen: Dict[str, Any] = {}

    def fake_unload(target):
        seen["event_set_before_server_unload"] = ev.is_set()

    monkeypatch.setattr(p, "_native_rest_unload_model", fake_unload)

    def finish():
        ev.wait(5)
        p._inflight_generations().end(call)

    threading.Thread(target=finish, daemon=True).start()
    p.unload_model("m")
    assert seen == {"event_set_before_server_unload": True}
    assert p._last_unload_inflight["drained"] is True


def test_mlx_load_model_reloads_only_when_unloaded(monkeypatch):
    pytest.importorskip("mlx_lm")
    from abstractcore.providers.mlx_provider import MLXProvider

    p = object.__new__(MLXProvider)
    p.model = "mlx-community/some-model"
    calls: List[int] = []

    def fake_load():
        calls.append(1)
        p.llm, p.tokenizer = object(), object()

    monkeypatch.setattr(p, "_load_model", fake_load, raising=False)
    p.llm = p.tokenizer = None
    assert p.load_model()["action"] == "loaded" and calls == [1]
    assert p.load_model()["action"] == "already_loaded" and calls == [1]
    with pytest.raises(ValueError):
        p.load_model("mlx-community/another-model")


def test_mlx_unload_refuses_while_a_call_does_not_stop(monkeypatch):
    """The first thing an MLX unload does is stop what runs on it; a call that
    does not stop by the deadline makes it raise BEFORE anything is freed."""
    pytest.importorskip("mlx_lm")
    import abstractcore.providers.generation_cancel as gc_mod
    from abstractcore.exceptions import ProviderAPIError
    from abstractcore.providers.mlx_provider import MLXProvider

    monkeypatch.setattr(gc_mod, "DEFAULT_EJECT_DRAIN_TIMEOUT_S", 0.2)
    p = object.__new__(MLXProvider)
    p.model = "mlx-community/some-model"
    p.llm, p.tokenizer = object(), object()
    freed: List[int] = []
    monkeypatch.setattr(p, "_unload_model_unlocked", lambda name: freed.append(1), raising=False)
    stuck = p._inflight_generations().begin(threading.Event())
    try:
        with pytest.raises(ProviderAPIError, match="Refusing to unload"):
            p.unload_model(p.model)
        assert freed == [] and stuck.event.is_set()
    finally:
        p._inflight_generations().end(stuck)
