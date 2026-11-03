"""Idle unload (core ADR 0012, backlog 1115): a model no call has used for the idle-unload time is
unloaded; never a locked one, never one in use or still loading; a call is never cancelled for it.
Fakes only: no weights, no gateway, no network."""
from __future__ import annotations

import gc
import threading
import time
import types
from typing import Any, Dict, List

import pytest

import abstractcore.providers.huggingface_provider  # noqa: F401 - the HF backend is "loaded"
from abstractcore.providers import hf_residency, mlx_residency, process_residency as pr
from abstractcore.providers.generation_cancel import InflightGenerations

GB = 1000 ** 3


class _Holder:
    def __init__(self, model: str, size: int = GB, last_used: float | None = None):
        self.model = model
        self.llm = object()
        self.size = size
        self._reg = InflightGenerations()
        self._reg.last_used_at = last_used

    def _inflight_generations(self):
        return self._reg


class _Claimant:
    def __init__(self, provider: str, model: str, *, locked: bool, kind: str):
        self.claim = {"provider": provider, "model": model, "locked": locked, "kind": kind,
                      "owner_label": f"the {kind} of a test"}

    def residency_claims(self):
        return [self.claim]


@pytest.fixture
def hf(monkeypatch):
    """HuggingFace holders the residency module sees; ejects recorded, never real."""
    gc.collect()
    holders: List[_Holder] = []
    ejected: List[Dict[str, Any]] = []
    monkeypatch.setattr(hf_residency, "registered_providers", lambda: list(holders))
    monkeypatch.setattr(hf_residency, "_holder_alive", lambda h: h.llm is not None)
    monkeypatch.setattr(hf_residency, "_holder_row", lambda h: {"held_bytes": h.size})
    monkeypatch.setattr(hf_residency, "_holders_for", lambda model, path: [h for h in holders if h.model == model])

    def fake_eject(backend, model, reason="eject"):
        cancelled = 0
        for h in holders:
            if h.model == model:
                cancelled += h._reg.cancel_all(reason="test eject", drain_timeout_s=0.5)["cancelled"]
                h.llm = None
        ejected.append({"backend": backend, "model": model, "reason": reason, "cancelled": cancelled})
        return {"ok": True}

    monkeypatch.setattr(pr, "eject", fake_eject)
    monkeypatch.setattr(pr, "_mlx_model_states", lambda: [])
    monkeypatch.setattr(pr, "_embedding_model_states", lambda: [])
    monkeypatch.setattr(pr, "_FIRST_SEEN", {})
    pr.set_idle_unload(pr.DEFAULT_IDLE_UNLOAD_S)
    ns = types.SimpleNamespace(holders=holders, ejected=ejected)
    yield ns
    pr.set_idle_unload(pr.DEFAULT_IDLE_UNLOAD_S)


def test_the_default_is_60_minutes_and_none_or_0_turn_it_off():
    try:
        assert pr.DEFAULT_IDLE_UNLOAD_S == 3600.0 and pr.get_idle_unload() == 3600.0
        assert pr.set_idle_unload(0) is None and pr.get_idle_unload() is None
        assert pr.set_idle_unload(None) is None
        assert pr.set_idle_unload(90) == 90.0
        for bad in (-1, "soon", float("nan")):
            with pytest.raises(ValueError):
                pr.set_idle_unload(bad)
    finally:
        pr.set_idle_unload(pr.DEFAULT_IDLE_UNLOAD_S)


def test_a_model_idle_past_the_window_is_unloaded_and_a_recent_one_is_kept(hf, caplog):
    now = time.time()
    hf.holders += [_Holder("old/A", 4 * GB, now - 3700), _Holder("recent/B", 2 * GB, now - 60)]
    pr._FIRST_SEEN.update({("huggingface", "old/A"): now - 7200, ("huggingface", "recent/B"): now - 7200})
    with caplog.at_level("INFO", logger="abstractcore.residency"):
        out = pr.reap_idle(now=now)
    assert [e["model"] for e in hf.ejected] == ["old/A"], "MUTANT: no reaper / wrong window -> RED"
    assert hf.ejected[0]["reason"] == "idle"
    assert out == [{"model": "old/A", "backend": "huggingface", "idle_s": 3700.0, "freed_bytes": 4 * GB, "ok": True}]
    assert "unloaded old/A after 62 min idle; freed 4.0 GB" in caplog.text


def test_off_unloads_nothing(hf):
    hf.holders.append(_Holder("old/A", GB, time.time() - 10 * 3600))
    pr.set_idle_unload(0)
    assert pr.reap_idle(now=time.time()) == [] and hf.ejected == []


def test_a_locked_model_never_unloads(hf):
    now = time.time()
    hf.holders.append(_Holder("kept/A", GB, now - 10 * 3600))
    pr._FIRST_SEEN[("huggingface", "kept/A")] = now - 10 * 3600
    lock = _Claimant("huggingface", "kept/A", locked=True, kind="lock")
    pr.register_claimant(lock)
    try:
        assert pr.reap_idle(now=now) == []
        state = next(s for s in pr.resident_model_states() if s["model"] == "kept/A")
    finally:
        pr.unregister_claimant(lock)
    assert hf.ejected == [], "MUTANT: lock ignored -> unloaded -> RED"
    assert state["locked_by"] and state["idle_unload_at"] is None


def test_a_pool_claim_does_not_keep_a_model_loaded(hf):
    """ADR 0010: pools and configured defaults do not lock; only a lock or use does."""
    now = time.time()
    hf.holders.append(_Holder("pooled/A", GB, now - 10 * 3600))
    pr._FIRST_SEEN[("huggingface", "pooled/A")] = now - 10 * 3600
    pool = _Claimant("huggingface", "pooled/A", locked=False, kind="pool")
    pr.register_claimant(pool)
    try:
        pr.reap_idle(now=now)
    finally:
        pr.unregister_claimant(pool)
    assert [e["model"] for e in hf.ejected] == ["pooled/A"]


def test_a_model_still_loading_counts_as_in_use(hf):
    now = time.time()
    hf.holders.append(_Holder("building/A", GB, now - 10 * 3600))
    pr._FIRST_SEEN[("huggingface", "building/A")] = now - 10 * 3600
    build = _Claimant("huggingface", "building/A", locked=False, kind="building")
    pr.register_claimant(build)
    try:
        assert pr.reap_idle(now=now) == []
    finally:
        pr.unregister_claimant(build)
    assert hf.ejected == [], "MUTANT: a building claim not counted in use -> RED"


def test_a_running_call_is_never_cancelled_and_its_model_is_kept(hf):
    now = time.time()
    holder = _Holder("busy/A", GB, now - 10 * 3600)
    hf.holders.append(holder)
    pr._FIRST_SEEN[("huggingface", "busy/A")] = now - 10 * 3600
    event = threading.Event()
    call = holder._reg.begin(event)
    try:
        assert pr.reap_idle(now=now) == []
    finally:
        holder._reg.end(call)
    assert hf.ejected == [] and not event.is_set(), "MUTANT: in-flight ignored -> cancelled -> RED"


def test_a_call_that_starts_after_the_check_keeps_the_model_and_is_never_cancelled(hf, monkeypatch):
    """The race the idle gate closes: the state said idle, a call begins before the eject. The gate
    refuses to close while it runs, so the model is kept and the call is not cancelled."""
    now = time.time()
    holder = _Holder("raced/A", GB, now - 10 * 3600)
    hf.holders.append(holder)
    pr._FIRST_SEEN[("huggingface", "raced/A")] = now - 10 * 3600
    event = threading.Event()
    real_states = pr.resident_model_states
    started: list = []

    def states_then_a_call_begins():
        out = real_states()
        if not started:
            started.append(holder._reg.begin(event))  # lands between the reaper's check and its eject
        return out

    monkeypatch.setattr(pr, "resident_model_states", states_then_a_call_begins)
    assert pr.reap_idle(now=now) == []
    assert hf.ejected == [] and not event.is_set(), "MUTANT: no gate -> the call is cancelled -> RED"
    holder._reg.end(started[0])


def test_a_call_that_begins_during_the_idle_eject_waits_then_runs(hf, monkeypatch):
    """While the gate is closed a new call waits (it is not refused, not cancelled); it runs once the
    eject is done and the gate reopens. A deadlock fails this test by its timeout."""
    now = time.time()
    holder = _Holder("slow/A", GB, now - 10 * 3600)
    hf.holders.append(holder)
    pr._FIRST_SEEN[("huggingface", "slow/A")] = now - 10 * 3600
    in_eject, release, began = threading.Event(), threading.Event(), threading.Event()
    real_eject = pr.eject

    def slow_eject(backend, model, reason="eject"):
        in_eject.set()
        assert release.wait(5)
        return real_eject(backend, model, reason=reason)

    monkeypatch.setattr(pr, "eject", slow_eject)
    reaper = threading.Thread(target=pr.reap_idle, kwargs={"now": now})
    reaper.start()
    assert in_eject.wait(5)
    caller = threading.Thread(target=lambda: (holder._reg.end(holder._reg.begin(threading.Event())), began.set()))
    caller.start()
    time.sleep(0.2)
    assert not began.is_set(), "MUTANT: no gate -> a call runs during the eject -> RED"
    release.set()
    reaper.join(5)
    caller.join(5)
    assert began.is_set() and not reaper.is_alive() and not caller.is_alive()
    assert [e["model"] for e in hf.ejected] == ["slow/A"] and hf.ejected[0]["cancelled"] == 0


def test_a_tick_is_skipped_while_a_model_loads(hf):
    now = time.time()
    hf.holders.append(_Holder("old/A", GB, now - 10 * 3600))
    pr._FIRST_SEEN[("huggingface", "old/A")] = now - 10 * 3600
    holding, done = threading.Event(), threading.Event()

    def a_load():
        with pr._LOAD_LOCK:
            holding.set()
            done.wait(5)

    loader = threading.Thread(target=a_load)
    loader.start()
    assert holding.wait(5)
    try:
        assert pr.reap_idle(now=now) == [], "MUTANT: no load-lock check -> RED"
        assert hf.ejected == []
    finally:
        done.set()
        loader.join(5)
    pr.reap_idle(now=now)
    assert [e["model"] for e in hf.ejected] == ["old/A"]


def test_a_model_never_called_is_timed_from_when_it_was_first_seen(hf):
    t0 = time.time()
    hf.holders.append(_Holder("loaded/A", GB, None))  # loaded, no call yet
    assert pr.reap_idle(now=t0) == []                 # first seen now
    assert pr.reap_idle(now=t0 + 3599) == []
    assert [e["model"] for e in pr.reap_idle(now=t0 + 3600)] == ["loaded/A"]


def test_a_call_without_a_cancel_event_is_counted_and_stamps_last_use():
    """The last-used gap: `begin(None)` used to return before stamping."""
    reg = InflightGenerations()
    before = time.time()
    call = reg.begin(None)
    assert reg.active() == 1 and reg.last_used_at is not None and reg.last_used_at >= before
    assert reg.cancel_all(reason="test", drain_timeout_s=0.1)["still_running"] == 1  # waited for, never "stopped"
    reg.end(call)
    assert reg.active() == 0


def test_mlx_models_are_reaped_with_their_registries(monkeypatch):
    now = time.time()
    holder = _Holder("mlx-community/Tiny-4bit", GB, now - 7200)
    monkeypatch.setattr(pr, "_backend_loaded", lambda b: b == "mlx")
    monkeypatch.setattr(mlx_residency, "resident_models", lambda: [
        {"lane": "mlx_lm", "models": ["mlx-community/Tiny-4bit"], "model_path": "/p", "holders": 1,
         "held_bytes": GB, "weights_alive": holder.llm is not None}])
    monkeypatch.setattr(mlx_residency, "_holders_for", lambda model, path: [holder])
    ejected = []
    monkeypatch.setattr(pr, "eject", lambda b, m, reason="eject": (ejected.append((b, m, reason)),
                                                                     setattr(holder, "llm", None), {"ok": True})[-1])
    monkeypatch.setattr(pr, "_FIRST_SEEN", {("mlx", "mlx-community/Tiny-4bit"): now - 7200})
    state = pr.resident_model_states()[0]
    assert (state["backend"], state["in_use"], state["last_used_at"]) == ("mlx", False, now - 7200)
    pr.reap_idle(now=now)
    assert ejected == [("mlx", "mlx-community/Tiny-4bit", "idle")], "MUTANT: MLX missing from the states -> RED"


def test_embedding_models_are_reaped_and_a_running_encode_keeps_them(monkeypatch):
    now = time.time()

    class _Manager:
        _inflight_encodes = 0
        _last_used_at = now - 7200

        def get_residency(self):
            return {"model": "BAAI/bge-small", "loaded": True}

    manager = _Manager()
    module = types.SimpleNamespace(
        registered_embedding_managers=lambda: [manager],
        resident_embedding_models=lambda: [{"models": ["BAAI/bge-small"], "held_bytes": GB, "holders": 1}])
    monkeypatch.setitem(pr.sys.modules, pr._EMBEDDINGS_MODULE, module)
    monkeypatch.setattr(pr, "_backend_loaded", lambda b: b == "embeddings")
    ejected = []
    monkeypatch.setattr(pr, "eject", lambda b, m, reason="eject": (ejected.append((b, m)), {"ok": True})[-1])
    monkeypatch.setattr(pr, "_FIRST_SEEN", {("embeddings", "BAAI/bge-small"): now - 7200})
    manager._inflight_encodes = 1
    assert pr.reap_idle(now=now) == [] and ejected == []
    manager._inflight_encodes = 0
    pr.reap_idle(now=now)
    assert ejected == [("embeddings", "BAAI/bge-small")], "MUTANT: embeddings missing from the states -> RED"


def test_the_reaper_thread_starts_with_the_first_load(monkeypatch):
    monkeypatch.setattr(pr, "_REAPER", None)
    started = []
    monkeypatch.setattr(pr._threading, "Thread", lambda **kw: types.SimpleNamespace(
        start=lambda: started.append(kw["name"]), is_alive=lambda: bool(started)))
    with pr.load_lock():
        pass
    with pr.load_lock():
        pass
    assert started == ["abstractcore-idle-unload"], "MUTANT: no lazy start / started twice -> RED"


def test_a_thread_that_is_no_longer_the_reaper_stops(monkeypatch):
    ticks = []
    monkeypatch.setattr(pr, "reap_idle", lambda now=None: ticks.append(1) or [])
    thread = threading.Thread(target=pr._reaper_loop, name="a superseded reaper", daemon=True)
    monkeypatch.setattr(pr, "_REAPER", None)  # the registered reaper is not this thread
    thread.start()
    pr._REAPER_WAKE.set()
    thread.join(5)
    assert not thread.is_alive() and ticks == [], "MUTANT: two reapers tick in one process -> RED"
