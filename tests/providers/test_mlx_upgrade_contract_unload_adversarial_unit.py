"""Final native unload must release weights before flushing the MLX allocator.

No MLX import, allocation, or model load is required: a fake allocator records
when Python model destruction would transfer active buffers into its free pool.
"""
import gc
import sys
import types
import weakref
from types import SimpleNamespace

import pytest

from abstractcore.providers.mlx_native_session import NativeSession, release_native_owner
from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.providers.mlx_runtime import NativeRuntime


@pytest.mark.parametrize("scheduled", [False, True], ids=["direct", "scheduled"])
@pytest.mark.parametrize("cyclic", [False, True], ids=["acyclic-model", "cyclic-model"])
def test_final_unload_destroys_native_weights_before_allocator_flush(monkeypatch, scheduled, cyclic):
    allocator = {"active": 0, "cached": 0, "flushes": []}

    class Weight:
        def __init__(self):
            allocator["active"] += 1
            if cyclic:
                self.cycle = self

        def __del__(self):
            allocator["active"] -= 1
            allocator["cached"] += 1

    def flush():
        allocator["flushes"].append({"active": allocator["active"], "cached": allocator["cached"]})
        allocator["cached"] = 0

    fake_mlx = types.ModuleType("mlx")
    fake_core = types.ModuleType("mlx.core")
    fake_core.clear_cache = flush
    fake_mlx.core = fake_core
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_core)

    value = MLXProvider.__new__(MLXProvider)
    value.model = "fake/native"
    value.logger = SimpleNamespace(info=lambda *args: None, warning=lambda *args: None)
    value._release_shared_model = lambda: False
    value._clear_prompt_caches_for_unload = lambda: None

    session = NativeSession()
    session.model, session.drafter = Weight(), Weight()
    session.processor = SimpleNamespace(tokenizer=object())
    session.holders.add(value)
    session.execution_config = "test"
    if scheduled:
        session.runtime = NativeRuntime(session.model, session.processor, session.drafter, "mtp")
        value._native_owner_id = session.runtime.acquire()
        worker = session.runtime._thread
        runtime_ref = weakref.ref(session.runtime)
        assert session.runtime._backend is None, "CPU harness must never load an MLX backend"
    else:
        value._native_owner_id = None
        worker = None
        runtime_ref = lambda: None

    value._native_session = session
    value._native_runtime = session.runtime
    value._native_qwen4 = session
    value._mtp_generation_lock = session.lock
    value.llm = session.model
    value.tokenizer = session.processor.tokenizer
    value._mtp_processor = session.processor
    value._mtp_drafter = session.drafter
    value._native_finalizer = weakref.finalize(value, release_native_owner, session, value._native_owner_id)
    session_ref = weakref.ref(session)
    target_ref, drafter_ref = weakref.ref(session.model), weakref.ref(session.drafter)
    del session

    try:
        value.unload_model(value.model)
        # Do not perform a compensating gc/clear before checking the public API.
        assert allocator["flushes"] == [{"active": 0, "cached": 2}], allocator
        assert allocator["active"] == allocator["cached"] == 0, allocator
        assert session_ref() is None and runtime_ref() is None
        assert target_ref() is None and drafter_ref() is None
        if worker is not None:
            assert not worker.is_alive()
    finally:
        # Failure must not leave a CPU worker running or cyclic fake buffers.
        surviving_runtime = runtime_ref()
        if surviving_runtime is not None:
            surviving_runtime.close()
        gc.collect()
