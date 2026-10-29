"""The per-model slot pool (ADR 0011): concurrent generations on one weights instance, bounded,
FIFO, cancellable, with the queued state visible. Fake engines (no llama.cpp)."""
from __future__ import annotations

import gc
import threading
import time

import pytest

from abstractcore.exceptions import GenerationCancelledError
from abstractcore.providers import gguf_slots
from abstractcore.providers.gguf_slots import (ModelBusyError, ModelQueueCancelled, ModelQueueFullError,
                                                ModelQueueTimeoutError, SlotScheduler)


class Engine:
    n = 0

    def __init__(self):
        Engine.n += 1
        self.id = Engine.n
        self.closed = False

    def close(self):
        self.closed = True


def _pool(max_concurrent=2, fits=lambda: None, **kw):
    return SlotScheduler(key="k|n_ctx=4096", model="m", first_llm=Engine(), n_ctx=4096, make_slot=Engine,
                         slot_fits=fits, max_concurrent=max_concurrent, **kw)


def _parallel(fns):
    out, threads = {}, []
    for i, fn in enumerate(fns):
        def run(i=i, fn=fn):
            try:
                out[i] = fn()
            except BaseException as exc:  # noqa: BLE001
                out[i] = exc
        t = threading.Thread(target=run)
        t.start()
        threads.append(t)
        time.sleep(0.02)  # admission order = start order
    for t in threads:
        t.join(10)
    return out


def test_concurrency_is_bounded_and_slots_are_built_lazily():
    pool = _pool(max_concurrent=2)
    peak, now, lock = [0], [0], threading.Lock()

    def work():
        with lock:
            now[0] += 1
            peak[0] = max(peak[0], now[0])
        time.sleep(0.15)
        with lock:
            now[0] -= 1
        return gguf_slots.current_llm(pool).id

    out = _parallel([lambda: pool.run(work) for _ in range(5)])
    assert peak[0] == 2 and pool.state()["slots"] == 2
    assert len(set(out.values())) == 2


def test_each_call_sees_its_own_slot_and_a_callback_stays_on_it():
    """The contextvar resolves `llm` to the borrowed slot inside the worker; a call back into the
    pool from the worker runs inline on the same slot. MUTANT: resolve to slot 0 -> RED."""
    pool = _pool(max_concurrent=2)
    barrier = threading.Barrier(2)

    def work():
        mine = gguf_slots.current_llm(pool)
        barrier.wait(5)
        inner = pool.run(lambda: gguf_slots.current_llm(pool))
        return mine, inner

    out = _parallel([lambda: pool.run(work), lambda: pool.run(work)])
    (a, a_inner), (b, b_inner) = out[0], out[1]
    assert a is not b and a is a_inner and b is b_inner


def test_the_provider_llm_is_the_borrowed_slot():
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    pool = _pool(max_concurrent=2)
    prov = object.__new__(HuggingFaceProvider)
    prov.__dict__["_gguf_scheduler"] = pool
    barrier = threading.Barrier(2)

    def work():
        got = prov.llm
        barrier.wait(5)
        return got is gguf_slots.current_llm(pool), got

    out = _parallel([lambda: pool.run(work), lambda: pool.run(work)])
    assert out[0][0] and out[1][0] and out[0][1] is not out[1][1]
    assert prov.llm is pool.first_llm  # outside a call: slot 0


def test_fifo_order_and_queue_full():
    pool = _pool(max_concurrent=1, max_queue=2)
    order, gate = [], threading.Event()
    first = threading.Thread(target=lambda: pool.run(lambda: gate.wait(5)))
    first.start()
    time.sleep(0.05)
    results = {}

    def req(i):
        try:
            pool.run(lambda: order.append(i))
        except BaseException as exc:  # noqa: BLE001
            results[i] = exc

    ts = []
    for i in range(3):
        t = threading.Thread(target=req, args=(i,))
        t.start()
        ts.append(t)
        time.sleep(0.05)
    gate.set()
    for t in ts + [first]:
        t.join(5)
    assert order == [0, 1]
    assert isinstance(results[2], ModelQueueFullError) and results[2].status_code == 503
    assert getattr(results[2], "request_local", False) and results[2].retry_after_s


def test_a_cancel_while_queued_leaves_the_queue_and_is_a_cancel():
    pool = _pool(max_concurrent=1)
    gate, cancel = threading.Event(), threading.Event()
    t1 = threading.Thread(target=lambda: pool.run(lambda: gate.wait(5)))
    t1.start()
    time.sleep(0.05)
    out = {}
    t2 = threading.Thread(target=lambda: out.setdefault("r", _catch(lambda: pool.run(lambda: "ran", cancel_event=cancel))))
    t2.start()
    time.sleep(0.1)
    assert pool.state()["queued"] == 1
    cancel.set()
    t2.join(5)
    assert isinstance(out["r"], ModelQueueCancelled) and isinstance(out["r"], GenerationCancelledError)
    assert pool.state()["queued"] == 0
    gate.set()
    t1.join(5)
    assert pool.run(lambda: "next") == "next"


def _catch(fn):
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001
        return exc


def test_a_slot_that_does_not_fit_is_not_built_and_the_request_waits_with_the_reason():
    reason = "another slot of m needs 2.0 GiB of GPU memory for its 4,096-token context, 0.5 GiB free (5 % kept free)"
    pool = _pool(max_concurrent=4, fits=lambda: reason)
    gate = threading.Event()
    seen = []
    t1 = threading.Thread(target=lambda: pool.run(lambda: gate.wait(5)))
    t1.start()
    time.sleep(0.05)
    t2 = threading.Thread(target=lambda: pool.run(lambda: "ok", on_queued=lambda pos, st: seen.append((pos, st))))
    t2.start()
    time.sleep(0.4)
    st = pool.state()
    assert st["slots"] == 1 and st["queued"] == 1 and st["slot_refusal_reason"] == reason
    assert seen and seen[-1][0] == 1 and seen[-1][1]["slot_refusal_reason"] == reason
    assert len(seen) >= 2, "the queued state is reported on every wake, not only on a position change"
    gate.set()
    t1.join(5)
    t2.join(5)


def test_queue_timeout_is_typed():
    pool = _pool(max_concurrent=1, queue_timeout_s=0.3)
    gate = threading.Event()
    t1 = threading.Thread(target=lambda: pool.run(lambda: gate.wait(5)))
    t1.start()
    time.sleep(0.05)
    err = _catch(lambda: pool.run(lambda: "x"))
    gate.set()
    t1.join(5)
    assert isinstance(err, ModelQueueTimeoutError) and err.status_code == 504 and err.request_local


def test_close_refuses_while_a_slot_generates():
    """A context freed under a running decode crashes the process. MUTANT: close without the check -> RED."""
    pool = _pool(max_concurrent=1)
    gate = threading.Event()
    t1 = threading.Thread(target=lambda: pool.run(lambda: gate.wait(5)))
    t1.start()
    time.sleep(0.05)
    with pytest.raises(ModelBusyError):
        pool.close(wait_s=0.2)
    assert not pool.first_llm.closed
    gate.set()
    t1.join(5)
    pool.close()
    assert pool.closed


def test_idle_extra_slots_are_reaped_slot_zero_stays():
    pool = _pool(max_concurrent=3, idle_ttl_s=0.0)
    barrier = threading.Barrier(3)
    _parallel([lambda: pool.run(lambda: barrier.wait(5)) for _ in range(3)])
    assert pool.state()["slots"] == 1, "extra slots freed at once with a 0 s TTL"


def test_a_dropped_holder_detaches_and_the_last_frees_the_pool():
    """MUTANT: no finalizer -> the pool lives forever with no holder -> RED."""
    class Holder:
        pass

    pool = gguf_slots.register("w", _pool())
    a, b = Holder(), Holder()
    gguf_slots.attach(pool, a)
    gguf_slots.attach(pool, b)
    assert pool.state()["holders"] == 2
    first = pool.first_llm
    del a
    gc.collect()
    assert pool.state()["holders"] == 1 and not pool.closed
    del b
    gc.collect()
    assert pool.closed and first.closed and pool not in gguf_slots.all_schedulers()


def test_progress_emitter_reports_queued_then_prefill():
    from abstractcore.providers.generation_progress import TextProgressEmitter

    events = []
    em = TextProgressEmitter(events.append, provider="huggingface", model="m", min_interval_s=0.0)
    em.queued(position=2, slots=1, slots_max=1, active=1, queued=2)
    em.queued(position=1, slots=1, slots_max=1, active=1, queued=1)
    assert em.prefill(prompt_tokens=10)
    assert [e["phase"] for e in events] == ["queued", "queued", "prefill"]
    assert events[0]["queue_position"] == 2 and events[1]["queue_position"] == 1


def _holder(n_gpu_layers, explicit):
    import logging

    from abstractcore.providers.huggingface_provider import HuggingFaceProvider, _InstanceGenerationWorker

    prov = object.__new__(HuggingFaceProvider)
    prov.model = "m"
    prov.n_gpu_layers = n_gpu_layers
    prov._n_gpu_layers_explicit = explicit
    prov.max_output_tokens = 1024
    prov.logger = logging.getLogger("test")
    prov._generation_worker = _InstanceGenerationWorker("t")
    prov._gguf_prompt_cache_lock = threading.Lock()
    return prov


def test_attach_adopts_the_resident_gpu_layers_unless_a_person_set_them():
    """A resolution accident (torch imported -> CPU) adopts the resident GPU model, flagged; a
    person's explicit n_gpu_layers is refused in words, never silently overridden. MUTANT: adopt
    always -> RED."""
    from abstractcore.exceptions import InvalidRequestError

    pool = _pool()
    pool.attach_state = {"n_gpu_layers": -1}
    accident = _holder(0, explicit=False)
    accident._gguf_attach(pool)
    assert accident.n_gpu_layers == -1 and accident._adopted_gpu_layers_from_resident_pool
    pinned = _holder(0, explicit=True)
    with pytest.raises(InvalidRequestError, match="never loaded"):
        pinned._gguf_attach(pool)
    assert pool.state()["holders"] == 1


def test_a_holder_dropped_while_its_pool_works_frees_the_pool_when_the_work_ends():
    """The last holder leaves while a generation runs: nothing is freed under it, and the pool
    (slots, weights) is freed when that generation returns. MUTANT: swallow the busy refusal and
    keep the attachment -> the pool leaks -> RED."""
    class Holder:
        pass

    pool = gguf_slots.register("w2", _pool())
    a = Holder()
    gguf_slots.attach(pool, a)
    gate, started = threading.Event(), threading.Event()
    t = threading.Thread(target=lambda: pool.run(lambda: (started.set(), gate.wait(5))))
    t.start()
    started.wait(5)
    first = pool.first_llm
    t0 = time.time()
    del a
    gc.collect()
    assert time.time() - t0 < 1.0, "a dropped holder never waits inside garbage collection"
    assert not first.closed and pool not in gguf_slots.all_schedulers()
    gate.set()
    t.join(5)
    assert pool.closed and first.closed


def test_two_streams_interleave_on_a_two_slot_pool():
    """ADR 0011 test (c). MUTANT: one worker for both streams -> one stream ends before the other starts -> RED."""
    pool = _pool(max_concurrent=2)
    order = []
    lock = threading.Lock()

    def chunks(name):
        for i in range(5):
            time.sleep(0.03)
            yield f"{name}{i}"

    def consume(name):
        for c in pool.stream(lambda: chunks(name)):
            with lock:
                order.append(c)

    ts = [threading.Thread(target=consume, args=(n,)) for n in ("a", "b")]
    [t.start() for t in ts]
    [t.join(10) for t in ts]
    first_a, last_a = order.index("a0"), order.index("a4")
    first_b, last_b = order.index("b0"), order.index("b4")
    assert first_b < last_a and first_a < last_b, order


def test_cancelling_one_of_four_active_requests_stops_only_it():
    """ADR 0011 test (d): the cancelled call stops within one token and returns its slot; the other
    three complete in full. MUTANT: the slot is not returned on an error -> the next request hangs -> RED."""
    pool = _pool(max_concurrent=4)
    events = [threading.Event() for _ in range(4)]
    started = threading.Barrier(5)

    def decode(i):
        def call():
            started.wait(5)
            out = []
            for tok in range(20):
                if events[i].is_set():
                    raise GenerationCancelledError("cancelled", generated_tokens=len(out))
                out.append(tok)
                time.sleep(0.02)
            return out
        return call

    results = {}
    ts = [threading.Thread(target=lambda i=i: results.__setitem__(i, _catch(lambda: pool.run(decode(i), cancel_event=events[i]))))
          for i in range(4)]
    [t.start() for t in ts]
    started.wait(5)
    time.sleep(0.05)
    t_cancel = time.time()
    events[1].set()
    ts[1].join(5)
    assert time.time() - t_cancel < 0.2, "stopped within about one token"
    [t.join(10) for t in ts]
    assert isinstance(results[1], GenerationCancelledError)
    assert all(results[i] == list(range(20)) for i in (0, 2, 3))
    assert pool.state()["active"] == 0 and pool.run(lambda: "slot returned") == "slot returned"
