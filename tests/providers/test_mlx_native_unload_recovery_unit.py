"""CPU-only public unload recovery, including allocator ordering after failure."""
import sys
import threading
import types
import weakref
from types import SimpleNamespace

import pytest

from abstractcore.providers.mlx_native_session import (
    NativeSession, load_native_session, release_native_owner,
)
from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.providers.mlx_runtime import NativeRuntime, NativeRuntimeError


@pytest.fixture
def allocator(monkeypatch):
    state = {"active": 0, "cached": 0, "flushes": []}

    class Weight:
        def __init__(self, cyclic=False):
            state["active"] += 1
            if cyclic:
                self.cycle = self

        def __del__(self):
            state["active"] -= 1
            state["cached"] += 1

    def clear_cache():
        state["flushes"].append((state["active"], state["cached"]))
        state["cached"] = 0

    package, core = types.ModuleType("mlx"), types.ModuleType("mlx.core")
    package.core, core.clear_cache = core, clear_cache
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return state, Weight


def attach(session):
    """Use production runtime/release/unload; replace only loading/allocator."""
    value = MLXProvider.__new__(MLXProvider)
    value.model = "fake/native-recovery"
    value.logger = SimpleNamespace(info=lambda *a: None, warning=lambda *a: None)
    value._release_shared_model = lambda: False
    value._clear_prompt_caches_for_unload = lambda: None
    value._native_owner_id = session.runtime.acquire()
    session.holders.add(value)
    value._native_session = value._native_qwen4 = session
    value._native_runtime = session.runtime
    value._mtp_generation_lock = session.lock
    value.llm, value._mtp_drafter = session.model, session.drafter
    value.tokenizer = session.processor.tokenizer
    value._mtp_processor = session.processor
    value._native_finalizer = weakref.finalize(
        value, release_native_owner, session, value._native_owner_id)
    return value


def session_with_weights(Weight, *, cyclic=False, on_close=None):
    session = NativeSession()
    session.model, session.drafter = Weight(cyclic), Weight(cyclic)
    session.processor = SimpleNamespace(tokenizer=object())
    session.execution_config = "test"
    session.runtime = NativeRuntime(session.model, session.processor, session.drafter,
                                    "mtp", on_close=on_close)
    assert session.runtime._backend is None
    return session


def assert_detached(value):
    assert value._native_runtime is None
    assert value._native_session is None
    assert value._native_owner_id is None
    assert value._native_qwen4 is None
    assert not value._native_finalizer.alive
    assert value.llm is value.tokenizer is None
    assert value._mtp_processor is value._mtp_drafter is None


@pytest.mark.parametrize("cyclic", [False, True])
def test_stopped_flush_failure_detaches_and_frees_before_raising(allocator, cyclic):
    state, Weight = allocator
    def fail_close():
        raise OSError("disk flush failed visibly")
    session = session_with_weights(Weight, cyclic=cyclic, on_close=fail_close)
    value = attach(session)
    session_ref, runtime_ref = weakref.ref(session), weakref.ref(session.runtime)
    target_ref, head_ref = weakref.ref(session.model), weakref.ref(session.drafter)
    worker = session.runtime._thread
    del session

    with pytest.raises(NativeRuntimeError, match="disk flush failed visibly") as caught:
        value.unload_model(value.model)
    assert caught.value.code == "backend_error"
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert_detached(value)
    assert not worker.is_alive()
    # No compensating collect/clear: the actual public method must have freed
    # even cyclic weights before the allocator flush despite the close error.
    assert state["flushes"] == [(0, 2)]
    assert state["active"] == state["cached"] == 0
    assert session_ref() is runtime_ref() is target_ref() is head_ref() is None
    value.unload_model(value.model)  # Already detached, not a missing-lease trap.


def test_sibling_release_does_not_flush_or_destroy_shared_model(allocator):
    state, Weight = allocator
    closed = []
    def fail_close():
        closed.append(True)
        raise OSError("final owner flush failed")
    session = session_with_weights(Weight, on_close=fail_close)
    first, second = attach(session), attach(session)
    worker = session.runtime._thread
    session_ref = weakref.ref(session)
    del session
    first.unload_model(first.model)
    assert_detached(first)
    # The shared weights are NOT destroyed while `second` still holds them
    # (active stays 2), and the final-owner close hook has NOT run (closed==[]).
    # `mx.clear_cache()` DOES run now on every unload, harmlessly: it returns
    # only unreferenced buffers, so with the weights still live it frees
    # nothing (flush recorded (active=2, cached=0)). This is the 2026-09-25
    # fix: the old guard skipped the clear whenever a sibling remained, and
    # when that sibling was then GC'd its freed buffers stayed in MLX's
    # allocator cache with no later clear -- the process held gigabytes while
    # `get_active_memory()` read 0.
    assert closed == [] and state["flushes"] == [(2, 0)] and state["active"] == 2
    assert session_ref().runtime.stats()["owners"] == 1
    assert list(session_ref().holders) == [second]
    with pytest.raises(NativeRuntimeError, match="final owner flush failed"):
        second.unload_model(second.model)
    assert_detached(second)
    assert closed == [True] and not worker.is_alive()
    # Final owner: weights freed (active 2->0, cached +2), then cleared.
    assert session_ref() is None and state["flushes"] == [(2, 0), (0, 2)]


def test_live_worker_close_timeout_retains_provider_and_same_owner_can_retry(allocator):
    state, Weight = allocator
    entered, finish = threading.Event(), threading.Event()
    def slow_close():
        entered.set()
        assert finish.wait(3), "test must release the CPU close hook"
    session = session_with_weights(Weight, on_close=slow_close)
    value = attach(session)
    owner = value._native_owner_id
    worker = session.runtime._thread
    runtime_ref, session_ref = weakref.ref(session.runtime), weakref.ref(session)
    del session
    original_join = worker.join
    # Exercise the production timeout branch without waiting its normal 10 s.
    worker.join = lambda timeout=None: original_join(timeout=0.01)
    try:
        with pytest.raises(NativeRuntimeError, match="worker has not stopped"):
            value.unload_model(value.model)
        assert entered.wait(1)
        assert value._native_runtime is runtime_ref() and value._native_session is session_ref()
        assert value._native_finalizer.alive and value.llm is not None
        assert state["active"] == 2 and state["flushes"] == []
        assert runtime_ref()._pending_release_owner == owner
        assert runtime_ref().stats()["owners"] == 0
        assert list(session_ref().holders) == [value]
        with pytest.raises(NativeRuntimeError) as unknown:
            runtime_ref().release("not-the-pending-owner")
        assert unknown.value.code == "missing_lease"
        with pytest.raises(NativeRuntimeError) as admission:
            runtime_ref().acquire()
        assert admission.value.code == "runtime_closed"
    finally:
        finish.set()
        original_join(timeout=1)
        worker.join = original_join
    assert not worker.is_alive()
    # ExceptionInfo retains traceback frames/runtime if kept across this check.
    del unknown, admission
    value.unload_model(value.model)
    assert_detached(value)
    assert runtime_ref() is session_ref() is None
    assert state["flushes"] == [(0, 2)]


def test_final_owner_retry_clears_pending_identity_after_success():
    runtime = NativeRuntime(object(), object())
    owner = runtime.acquire()
    real_close = runtime.close
    def first_timeout():
        raise NativeRuntimeError("simulated live worker timeout")
    runtime.close = first_timeout
    with pytest.raises(NativeRuntimeError, match="timeout"):
        runtime.release(owner)
    assert runtime._pending_release_owner == owner
    runtime.close = real_close
    runtime.release(owner)
    assert runtime._pending_release_owner is None
    with pytest.raises(NativeRuntimeError) as unknown:
        runtime.release(owner)
    assert unknown.value.code == "missing_lease"


def test_stopped_runtime_retry_reports_original_flush_failure_not_missing_lease():
    calls = []
    def fail_close():
        calls.append(True)
        raise OSError("persistent cache flush failure")
    runtime = NativeRuntime(object(), object(), on_close=fail_close)
    owner = runtime.acquire()
    for _ in range(2):
        with pytest.raises(NativeRuntimeError, match="persistent cache flush failure") as caught:
            runtime.release(owner)
        assert caught.value.code == "backend_error"
    assert runtime.stats()["closed"] and not runtime._thread.is_alive()
    assert runtime._pending_release_owner == owner
    assert calls == [True], "releasing must not rerun a stopped worker's cleanup hook"


def test_fresh_same_target_session_reloads_after_failed_final_flush(allocator, monkeypatch, tmp_path):
    state, Weight = allocator
    loads = []
    package = types.ModuleType("mlx_vlm")
    def load(target):
        loads.append(target)
        return Weight(), SimpleNamespace(tokenizer=object())
    package.load = load
    monkeypatch.setitem(sys.modules, "mlx_vlm", package)
    session = load_native_session(str(tmp_path))
    def fail_close():
        raise OSError("old session flush failed")
    session.runtime = NativeRuntime(session.model, session.processor, on_close=fail_close)
    first = attach(session)
    old_ref = weakref.ref(session)
    del session
    with pytest.raises(NativeRuntimeError, match="old session flush failed"):
        first.unload_model(first.model)
    assert old_ref() is None and state["active"] == 0

    replacement = load_native_session(str(tmp_path))
    replacement.runtime = NativeRuntime(replacement.model, replacement.processor)
    second = attach(replacement)
    assert len(loads) == 2 and state["active"] == 1
    assert second.llm is not None and second._native_runtime.stats()["owners"] == 1
    del replacement
    second.unload_model(second.model)
    assert_detached(second)
    assert state["active"] == state["cached"] == 0
