"""Collected owners retire on safe worker boundaries without loading MLX."""
import gc
import sys
import threading
import time
import weakref
from types import SimpleNamespace

import pytest

from abstractcore.providers import mlx_runtime as native
from abstractcore.providers.mlx_native_session import NativeSession, release_native_owner
from abstractcore.providers.mlx_provider import MLXProvider


@pytest.fixture
def harness(monkeypatch):
    gate, entered = threading.Event(), threading.Event()
    calls, closes, unraisable, runtimes = [], [], [], []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)

    def generate(model, processor, prompt, **kwargs):
        calls.append(prompt)
        entered.set()
        assert gate.wait(3), "CPU backend gate must be released"
        yield SimpleNamespace(text="OK", token=1, generation_tokens=1,
                              prompt_tokens=1, finish_reason="stop")

    monkeypatch.setattr(native, "_load_backend", lambda: SimpleNamespace(
        stream_generate=generate, mx=SimpleNamespace(get_peak_memory=lambda: 0)))

    def make(on_close=None):
        session = NativeSession()
        session.model, session.processor = object(), object()
        def close():
            closes.append(threading.get_ident())
            if on_close is not None:
                on_close()
        session.runtime = native.NativeRuntime(session.model, session.processor,
                                               on_close=close)
        runtimes.append(session.runtime)
        return session

    yield SimpleNamespace(make=make, gate=gate, entered=entered, calls=calls,
                          closes=closes, unraisable=unraisable)
    gate.set()
    for runtime in runtimes:
        try:
            runtime.close()
        except native.NativeRuntimeError:
            assert runtime._closed.is_set(), "test leaked a live CPU worker"
    assert not unraisable


def attach(session):
    # Actual provider objects/finalizer contract; only artifact loading is absent.
    value = MLXProvider.__new__(MLXProvider)
    value._native_session = session
    value._native_runtime = session.runtime
    value._native_owner_id = session.runtime.acquire()
    value.llm, value._mtp_processor = session.model, session.processor
    session.holders.add(value)
    value._native_finalizer = weakref.finalize(
        value, release_native_owner, session, value._native_owner_id)
    return value


def request(owner, prompt="blocked"):
    return native.NativeRequest(prompt, max_tokens=2, temperature=.4, owner_id=owner)


@pytest.mark.parametrize("queued", [False, True])
def test_collected_last_owner_closes_after_cancelled_backend_returns(harness, queued):
    session = harness.make()
    holder = attach(session)
    runtime, owner = session.runtime, holder._native_owner_id
    handle = runtime.stream(request(owner))
    assert harness.entered.wait(1)
    pending = runtime.stream(request(owner, "queued")) if queued else None
    with pytest.raises(native.NativeRuntimeError) as busy:
        runtime.release(owner)
    assert busy.value.code == "owner_busy", "explicit unload must stay strict"
    handle.close()
    session_ref, holder_ref = weakref.ref(session), weakref.ref(holder)
    del session, holder, busy
    before = time.monotonic()
    gc.collect()
    assert time.monotonic() - before < .5, "GC callback blocked waiting for backend"
    assert holder_ref() is None and not harness.unraisable
    assert session_ref() is not None, "weak registry session must survive live cancelled tensors"
    assert runtime.stats()["owners"] == runtime.stats()["retiring_owners"] == 1
    assert harness.closes == [] and runtime._thread.is_alive()
    with pytest.raises(native.NativeRuntimeError) as unavailable:
        runtime.stream(request(owner, "must-not-enter"))
    assert unavailable.value.code == "missing_lease"
    del unavailable

    harness.gate.set()
    runtime._thread.join(1)
    assert not runtime._thread.is_alive() and runtime.stats()["closed"]
    assert runtime.stats()["owners"] == runtime.stats()["retiring_owners"] == 0
    assert runtime.stats()["cancelled"] == (2 if queued else 1)
    assert harness.calls == ["blocked"] and harness.closes == [runtime._thread.ident]
    assert runtime._retirement_keepalive == {} and session_ref() is None
    if pending is not None:
        pending.close()


def test_retiring_one_owner_does_not_cancel_or_close_sibling(harness):
    session = harness.make()
    first, second = attach(session), attach(session)
    runtime, retired_owner = session.runtime, first._native_owner_id
    blocked = runtime.stream(request(retired_owner))
    assert harness.entered.wait(1)
    healthy = runtime.stream(request(second._native_owner_id, "sibling"))
    del first
    gc.collect()
    assert blocked._job.cancelled.is_set()
    assert not healthy._job.cancelled.is_set() and harness.closes == []
    harness.gate.set()
    assert "".join(item.text for item in healthy) == "OK"
    runtime.control(lambda: None)
    assert runtime.stats()["owners"] == 1 and runtime.stats()["retiring_owners"] == 0
    assert not runtime.stats()["closing"] and runtime._thread.is_alive()
    assert runtime.generate(request(second._native_owner_id, "following")).text == "OK"
    blocked.close()
    del second, session
    gc.collect()
    runtime._thread.join(1)
    assert runtime.stats()["owners"] == 0 and runtime.stats()["closed"]


def test_idle_gc_starts_shutdown_without_waiting_for_close_hook(harness):
    close_entered, finish_close = threading.Event(), threading.Event()
    def close_hook():
        close_entered.set()
        assert finish_close.wait(3)
    session = harness.make(close_hook)
    holder = attach(session)
    runtime, session_ref = session.runtime, weakref.ref(session)
    del holder, session
    before = time.monotonic()
    gc.collect()
    assert time.monotonic() - before < .5
    try:
        assert close_entered.wait(1)
        assert runtime.stats()["owners"] == 0 and runtime._thread.is_alive()
        assert session_ref() is not None, "close hook still owns shared resources"
    finally:
        finish_close.set()
    runtime._thread.join(1)
    assert runtime.stats()["closed"] and session_ref() is None
    assert harness.calls == [], "idle retirement must never initialize the backend"


def test_gc_retirement_adopts_pending_owner_from_explicit_close_timeout(harness):
    close_entered, finish_close = threading.Event(), threading.Event()
    def close_hook():
        close_entered.set()
        assert finish_close.wait(3)
    session = harness.make(close_hook)
    holder = attach(session)
    runtime, owner = session.runtime, holder._native_owner_id
    worker, join = runtime._thread, runtime._thread.join
    worker.join = lambda timeout=None: join(timeout=.01)
    try:
        with pytest.raises(native.NativeRuntimeError, match="worker has not stopped"):
            runtime.release(owner)
        assert close_entered.wait(1)
        assert runtime._pending_release_owner == owner
        del holder, session
        gc.collect()
        assert runtime._pending_release_owner is None
        assert runtime.stats()["owners"] == 0 and not harness.unraisable
        assert runtime.retire(owner) is False, "repeated retirement must be harmless"
    finally:
        finish_close.set()
        worker.join = join
        join(1)
    assert runtime.stats()["closed"] and runtime._retirement_keepalive == {}


def test_deferred_close_failure_is_visible_without_unraisable_gc_error(harness, caplog):
    def fail_close():
        raise OSError("GC cache flush failed")
    session = harness.make(fail_close)
    holder = attach(session)
    runtime = session.runtime
    del holder, session
    gc.collect()
    runtime._thread.join(1)
    assert runtime.stats()["closed"] and runtime.stats()["owners"] == 0
    assert runtime.stats()["close_error"] == "GC cache flush failed"
    assert "Native MLX deferred close hook failed: GC cache flush failed" in caplog.text
    assert runtime._retirement_keepalive == {} and not harness.unraisable
    with pytest.raises(native.NativeRuntimeError, match="GC cache flush failed"):
        runtime.close()
