"""Every generation of one HuggingFace provider instance runs on one worker thread, one at a time
(backlog 0859).

The server's `/v1` route lends a resident instance (a host's pool) instead of building a second
copy, so a pool's runs and `/v1` calls share ONE llama.cpp context. Round-2 review: a lock held
across a stream's yields deadlocked (a non-stream call blocking the event loop while a stream
waited for it; two streams advanced by one thread ran at once). The worker produces a stream into
a queue, so it never needs its consumer to finish.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Iterator, List

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers import huggingface_provider as hf


class _Overlap:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0
        self.threads: set = set()

    def __enter__(self):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.threads.add(threading.get_ident())

    def __exit__(self, *a):
        with self.lock:
            self.active -= 1


def _provider(seen: _Overlap, *, chunks: int = 3, fail_at_start: bool = False) -> Any:
    p = hf.HuggingFaceProvider.__new__(hf.HuggingFaceProvider)
    p.model = "unsloth/Qwen3.5-4B-GGUF"
    p.model_type = "gguf"
    p.llm = object()
    p._generation_worker = hf._InstanceGenerationWorker("test-hf")
    p.produced = 0

    def lane(prompt: Any, messages: Any, system_prompt: Any, tools: Any, media: Any, stream: bool, *a: Any, **k: Any):
        if fail_at_start:
            raise RuntimeError("model file missing")
        if not stream:
            with seen:
                time.sleep(0.03)
            return GenerateResponse(content="ok", model="m")

        def gen() -> Iterator[GenerateResponse]:
            with seen:
                for i in range(chunks):
                    time.sleep(0.01)
                    p.produced += 1
                    yield GenerateResponse(content=str(i), model="m")
        return gen()

    p._generate_gguf = lane
    return p


def _within(seconds: float, fn) -> Any:
    box: List[Any] = []
    t = threading.Thread(target=lambda: box.append(fn()), daemon=True)
    t.start()
    t.join(seconds)
    assert not t.is_alive(), "deadlocked"
    return box[0] if box else None


def test_calls_from_many_threads_run_one_at_a_time_on_one_worker_thread() -> None:
    seen = _Overlap()
    p = _provider(seen)
    threads = [threading.Thread(target=lambda: p._generate_internal("hi")) for _ in range(4)]
    [t.start() for t in threads]
    [t.join(5) for t in threads]
    assert seen.peak == 1 and len(seen.threads) == 1


def test_a_paused_stream_never_blocks_a_call_waiting_for_the_instance() -> None:
    """Round-2 deadlock: stream A paused by its consumer, call B blocking its thread (the event
    loop in the server) until the instance is free. A is produced to the end without its
    consumer, so B runs."""
    seen = _Overlap()
    p = _provider(seen, chunks=5)
    a = p._generate_internal("a", stream=True)
    assert next(a).content == "0"
    assert _within(5, lambda: p._generate_internal("b")).content == "ok"
    assert [c.content for c in a] == ["1", "2", "3", "4"]
    assert seen.peak == 1


def test_two_streams_advanced_by_one_thread_never_overlap() -> None:
    seen = _Overlap()
    p = _provider(seen)

    def interleave() -> List[str]:
        s1 = p._generate_internal("1", stream=True)
        first = next(s1).content
        s2 = p._generate_internal("2", stream=True)
        out = [first, next(s2).content]
        out += [c.content for c in s1] + [c.content for c in s2]
        return out

    assert sorted(_within(5, interleave)) == sorted(["0", "1", "2"] * 2)
    assert seen.peak == 1


def test_an_early_close_stops_the_producer_and_frees_the_instance() -> None:
    seen = _Overlap()
    p = _provider(seen, chunks=200)
    s = p._generate_internal("x", stream=True)
    next(s)
    s.close()
    assert _within(5, lambda: p._generate_internal("y")).content == "ok"
    assert p.produced < 200


def test_errors_at_the_start_of_a_stream_reach_the_caller_at_the_call() -> None:
    p = _provider(_Overlap(), fail_at_start=True)
    with pytest.raises(RuntimeError, match="model file missing"):
        p._generate_internal("x", stream=True)


def test_a_generation_that_calls_back_into_its_instance_runs_inline() -> None:
    worker = hf._InstanceGenerationWorker("test-reentry")
    assert _within(5, lambda: worker.run(lambda: worker.run(lambda: "inner"))) == "inner"


# --- Round-2 review findings (worker design) ---------------------------------------------------


def test_every_operation_that_touches_the_model_runs_on_the_worker() -> None:
    """Prompt-cache prefills / save / load / set / clear and the unload's free step drive the same
    llama.cpp context as a generation: they are queued on the instance's worker too."""
    for name in ("_prompt_cache_backend_create", "_prompt_cache_backend_clone", "_prompt_cache_backend_append",
                 "prompt_cache_set", "prompt_cache_clear", "prompt_cache_save", "prompt_cache_load",
                 "prompt_cache_prepare_modules", "load_model", "_free_model_on_unload"):
        assert getattr(getattr(hf.HuggingFaceProvider, name), "_abstractcore_on_model_thread", False), name

    seen = _Overlap()
    p = _provider(seen, chunks=5)

    @hf._on_model_thread
    def prefill(self: Any) -> int:
        with seen:
            time.sleep(0.02)
        return threading.get_ident()

    a = p._generate_internal("a", stream=True)
    next(a)
    worker_thread = _within(5, lambda: prefill(p))
    assert list(a) and seen.peak == 1 and seen.threads == {worker_thread}


def test_a_stream_is_in_flight_from_the_call_until_its_production_ends() -> None:
    """An unload cancels and waits for in-flight calls: a stream produces before its first next()."""
    gate = threading.Event()
    p = _provider(_Overlap(), chunks=2)
    inner = p._generate_gguf

    def lane(*a: Any, **k: Any):
        chunks = inner(*a, **k)

        def held() -> Iterator[GenerateResponse]:
            gate.wait(5)  # produced on the worker, paused until the test has looked
            yield from chunks
        return held()

    p._generate_gguf = lane
    s = p._generate_internal("x", stream=True, _cancel_event=threading.Event())  # base.generate gives one
    assert p._inflight_generations().active() == 1
    gate.set()
    assert [c.content for c in s] == ["0", "1"]
    assert _within(5, lambda: _wait_for(lambda: p._inflight_generations().active() == 0))


def _wait_for(pred) -> bool:
    deadline = time.time() + 5
    while not pred():
        if time.time() > deadline:
            return False
        time.sleep(0.01)
    return True


def test_an_eject_while_a_call_waits_is_seen_on_the_worker() -> None:
    """The ejected-reload check runs on the worker right before the lane, not when the call queued."""
    seen = _Overlap()
    p = _provider(seen, chunks=1)
    reloads: List[int] = []
    p.logger = logging.getLogger("test-hf-reload")

    def load_model(*a: Any, **k: Any) -> None:
        reloads.append(1)
        p.llm = object()

    p.load_model = load_model
    busy = threading.Event()
    release = threading.Event()
    p._generation_worker.run(lambda: None)  # start the worker

    t = threading.Thread(target=lambda: p._generation_worker.run(lambda: (busy.set(), release.wait(5))))
    t.start()
    busy.wait(5)
    box: List[Any] = []
    q = threading.Thread(target=lambda: box.append(p._generate_internal("queued")))
    q.start()
    time.sleep(0.05)
    p.llm = None  # ejected while the call waits
    release.set()
    t.join(5)
    q.join(5)
    assert box and box[0].content == "ok" and reloads == [1]


def test_the_callers_contextvars_reach_the_lane() -> None:
    import contextvars

    var: "contextvars.ContextVar[str]" = contextvars.ContextVar("hf_worker_test", default="unset")
    worker = hf._InstanceGenerationWorker("test-ctx")
    var.set("caller")
    assert worker.run(lambda: var.get()) == "caller"
    assert list(worker.stream(lambda: iter([var.get()]))) == ["caller"]


def test_an_interrupted_caller_cancels_its_generation() -> None:
    import os
    import signal

    worker = hf._InstanceGenerationWorker("test-sigint")
    cancel = threading.Event()
    started = threading.Event()

    def generation() -> str:
        started.set()
        cancel.wait(5)
        return "stopped" if cancel.is_set() else "ran to the end"

    if threading.current_thread() is not threading.main_thread():
        pytest.skip("signals are delivered to the main thread only")
    # A shell that started this process in the background may have it ignore SIGINT: install the default.
    previous = signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        threading.Thread(target=lambda: (started.wait(5), time.sleep(0.05), os.kill(os.getpid(), signal.SIGINT)), daemon=True).start()
        with pytest.raises(KeyboardInterrupt):
            worker.run(generation, cancel_event=cancel)
    finally:
        signal.signal(signal.SIGINT, previous)
    assert cancel.is_set()
