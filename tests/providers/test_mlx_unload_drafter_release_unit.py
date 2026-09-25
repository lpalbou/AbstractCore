"""An MTP drafter (the companion head loaded beside the target through the
native mlx-vlm session) must leave memory with the model on unload, and MLX's
allocator cache must be cleared AFTER it is freed (mission M2, 2026-09-25).

The drafter is referenced twice: `provider._mtp_drafter` and the shared
`NativeSession.drafter`. The session lives in a weak registry, so it (and its
drafter) goes away only when the LAST provider drops `_native_session`. This
drives the public `MLXProvider.unload_model` on the drafter-less-runtime
native lane with a fake MLX allocator and a real `NativeSession`, and asserts:
the drafter array is collected, `clear_cache` ran with nothing left active,
and the session is gone. MUTANTS that turn it RED: keep `_native_session`
(session + drafter stay alive), keep `_mtp_drafter`, skip `clear_cache`.
CPU only; no weights, no Metal.
"""
from __future__ import annotations

import sys
import types
import weakref
from types import SimpleNamespace

import pytest

from abstractcore.providers.mlx_native_session import NativeSession
from abstractcore.providers.mlx_provider import MLXProvider


@pytest.fixture
def allocator(monkeypatch):
    state = {"active": 0, "cache": 0, "clears": []}

    class Array:
        def __init__(self, nbytes):
            self.nbytes = int(nbytes)
            state["active"] += self.nbytes

        def __del__(self):
            state["active"] -= self.nbytes
            state["cache"] += self.nbytes

    def clear_cache():
        state["clears"].append((state["active"], state["cache"]))
        state["cache"] = 0

    core = types.ModuleType("mlx.core")
    core.clear_cache = clear_cache
    core.get_active_memory = lambda: state["active"]
    core.get_cache_memory = lambda: state["cache"]
    core.get_peak_memory = lambda: state["active"]
    package = types.ModuleType("mlx")
    package.core = core
    monkeypatch.setitem(sys.modules, "mlx", package)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    return state, Array


def _provider_on_native_session(session):
    p = MLXProvider.__new__(MLXProvider)
    p.model = "fake/target-mtp"
    p.logger = SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None,
                               debug=lambda *a, **k: None)
    p._stop_inflight_before_unload = lambda *a, **k: None
    p._release_shared_model = lambda: False
    p._clear_prompt_caches_for_unload = lambda: None
    p._native_runtime = None
    p._native_session = session
    p._native_qwen4 = None
    p._native_finalizer = None
    p._mtp_generation_lock = session.lock
    p.llm = session.model
    p.tokenizer = session.processor.tokenizer
    p._mtp_processor = session.processor
    p._mtp_drafter = session.drafter
    p._mtp_last_result = None
    p._vision_addon = None
    p._outlines_model = None
    p.generate_fn = p.stream_generate_fn = None
    session.holders.add(p)
    return p


def test_unload_releases_the_drafter_and_clears_the_allocator_after_it(allocator):
    state, Array = allocator
    session = NativeSession()
    session.model = Array(1_000_000)
    session.drafter = Array(200_000)          # the MTP companion head
    session.processor = SimpleNamespace(tokenizer=object())
    provider = _provider_on_native_session(session)
    session_ref = weakref.ref(session)
    drafter_ref = weakref.ref(session.drafter)
    del session
    assert state["active"] == 1_200_000

    provider.unload_model(provider.model)

    assert provider._native_session is None and provider._mtp_drafter is None
    assert drafter_ref() is None, "the drafter must be collected with the session"
    assert session_ref() is None, "no provider holds the native session any more"
    assert state["active"] == 0
    assert state["clears"], "clear_cache must run on unload"
    assert state["clears"][-1][0] == 0, "clear_cache ran after the drafter and target were freed"
    assert state["cache"] == 0, "the allocator cache is returned (drafter bytes included)"


def test_a_sibling_on_the_same_session_keeps_the_drafter_until_it_unloads(allocator):
    state, Array = allocator
    session = NativeSession()
    session.model = Array(1_000_000)
    session.drafter = Array(200_000)
    session.processor = SimpleNamespace(tokenizer=object())
    a = _provider_on_native_session(session)
    b = _provider_on_native_session(session)
    drafter_ref = weakref.ref(session.drafter)
    del session

    a.unload_model(a.model)
    assert drafter_ref() is not None, "the sibling still serves with the drafter"
    assert state["active"] == 1_200_000
    b.unload_model(b.model)
    assert drafter_ref() is None and state["active"] == 0 and state["cache"] == 0
