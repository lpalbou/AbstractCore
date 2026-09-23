"""Independent CPU scheduler tests using a deliberately controllable backend."""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from abstractcore.providers import mlx_runtime as native


class Array:
    def __init__(self, value):
        self.value = value

    def tolist(self):
        return self.value


class Detokenizer:
    def reset(self):
        self.text, self.offset = "", 0

    def add_token(self, token):
        self.text += chr(token)

    def finalize(self):
        pass

    @property
    def last_segment(self):
        text = self.text[self.offset:]
        self.offset = len(self.text)
        return text


class Backend:
    """The fake returns exact token/terminal events, not canned runtime results."""

    def __init__(self):
        self.scripts = {}
        self.engines = []
        self.thread_ids = []
        self.entered = threading.Event()
        self.prefilled = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self.exclusive_error = None
        self.exclusive_terminal = "length"
        self.exclusive_calls = []
        self.generator_closes = 0
        self.prefill_cancel = False

    def stream_generate(self, model, processor, prompt, **kwargs):
        self.exclusive_calls.append(kwargs)
        self.thread_ids.append(threading.get_ident())
        try:
            self.entered.set()
            assert self.gate.wait(2), "fake backend gate was not released"
            if self.exclusive_error is not None:
                raise self.exclusive_error
            text = self.scripts.get(prompt, "AB")[:kwargs["max_tokens"]]
            for i, char in enumerate(text, 1):
                yield SimpleNamespace(
                    text=char, token=ord(char), generation_tokens=i,
                    prompt_tokens=len(prompt), cached_tokens=0, prompt_tps=1.,
                    generation_tps=2., peak_memory=.1,
                    finish_reason=self.exclusive_terminal if i == len(text) else None,
                )
        finally:
            self.generator_closes += 1

    def batch_class(self):
        backend = self

        class Batch:
            def __init__(self, model, processor, **kwargs):
                self.kwargs = kwargs
                self.rows = {}
                self.next_uid = 0
                self.removals = []
                self.closed = False
                self.steps = 0
                self.peak_rows = 0
                backend.engines.append(self)

            def insert(self, prompts, max_tokens, prompt_kwargs):
                backend.thread_ids.append(threading.get_ident())
                uids = []
                for ids, cap in zip(prompts, max_tokens):
                    prompt = "".join(chr(i) for i in ids)
                    output = backend.scripts.get(prompt, "AB")[:cap]
                    uid = self.next_uid
                    self.next_uid += 1
                    self.rows[uid] = [output, 0]
                    uids.append(uid)
                self.peak_rows = max(self.peak_rows, len(self.rows))
                return uids

            def next(self):
                backend.thread_ids.append(threading.get_ident())
                backend.entered.set()
                assert backend.gate.wait(2), "fake backend gate was not released"
                self.steps += 1
                if backend.prefill_cancel and self.steps == 1:
                    backend.gate.clear()
                    backend.prefilled.set()
                    assert backend.gate.wait(2), "fake prefill was not released"
                    return [], []
                responses = []
                for uid, (text, offset) in list(self.rows.items()):
                    offset += 1
                    finish = "length" if offset >= len(text) else None
                    responses.append(SimpleNamespace(uid=uid, token=ord(text[offset - 1]), finish_reason=finish))
                    if finish:
                        self.rows.pop(uid)
                    else:
                        self.rows[uid][1] = offset
                return [], responses

            def remove(self, uid):
                allowed = not backend.prefill_cancel or self.steps >= 2
                self.removals.append((uid, allowed))
                if not allowed:
                    return False
                return self.rows.pop(uid, None) is not None

            @property
            def has_work(self):
                return bool(self.rows)

            def close(self):
                self.closed = True

        return Batch


@pytest.fixture
def runtime_factory(monkeypatch):
    backend = Backend()
    backend.mx = SimpleNamespace(array=Array, get_peak_memory=lambda: 10_000)
    monkeypatch.setattr(native, "_load_backend", lambda: SimpleNamespace(
        mx=backend.mx, BatchGenerator=backend.batch_class(), stream_generate=backend.stream_generate,
    ))
    detokenizer = Detokenizer()
    detokenizer.reset()
    processor = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda prompt, **_: list(map(ord, prompt))),
        detokenizer=detokenizer,
    )

    def embed(*args, **kwargs):
        backend.thread_ids.append(threading.get_ident())
        return SimpleNamespace(to_dict=lambda: {})

    model = SimpleNamespace(get_input_embeddings=embed)
    runtimes = []

    def make(**kwargs):
        runtime = native.NativeRuntime(model, processor, SimpleNamespace(), "mtp", **kwargs)
        runtimes.append(runtime)
        owner = runtime.acquire()
        return runtime, owner, backend

    yield make
    backend.gate.set()
    for runtime in runtimes:
        runtime.close()


def request(owner, prompt="p", **kwargs):
    return native.NativeRequest(prompt, max_tokens=kwargs.pop("max_tokens", 2), owner_id=owner, **kwargs)


def test_request_snapshots_nested_sampling_and_image_bytes():
    sampling = {"logit_bias": {"42": 3.0}, "limits": [1, 2]}
    pixels = bytearray(b"pixels")
    value = native.NativeRequest("p", 2, sampling=sampling, media=[(0, pixels)])
    sampling["logit_bias"]["42"] = 99
    sampling["limits"].append(3)
    pixels[0] = 0
    assert value.sampling["logit_bias"]["42"] == 3.0
    assert value.sampling["limits"] == (1, 2)
    assert value.media == ((0, b"pixels"),)
    with pytest.raises(TypeError):
        value.sampling["logit_bias"]["42"] = 10


def test_batch_execution_and_control_share_one_noncaller_worker(runtime_factory):
    runtime, owner, backend = runtime_factory(batch_wait_ms=25)
    first = runtime.stream(request(owner, "a"))
    second = runtime.stream(request(owner, "b"))
    a, b = list(first), list(second)
    assert "".join(c.text for c in a) == "AB"
    assert "".join(c.text for c in b) == "AB"
    assert a[-1].metadata["execution"]["peak_batch_size"] == 2
    assert a[-1].metadata["execution"]["mode"] == "continuous"
    worker = runtime.control(threading.get_ident)
    assert worker != threading.get_ident()
    assert set(backend.thread_ids) == {worker}


def test_mixed_depths_never_share_one_drafter_configuration(runtime_factory):
    runtime, owner, backend = runtime_factory(batch_wait_ms=25)
    streams = [runtime.stream(request(owner, str(depth), draft_tokens=depth)) for depth in (1, 3, 0)]
    results = [list(stream)[-1] for stream in streams]
    assert [engine.kwargs["draft_block_size"] for engine in backend.engines] == [2, 4, None]
    assert [result.metadata["execution"]["mode"] for result in results] == ["cohort", "cohort", "continuous"]


def test_stop_across_token_boundaries_does_not_leak_into_text(runtime_factory):
    runtime, owner, backend = runtime_factory()
    backend.scripts["p"] = "aaSTOPignored"
    chunks = list(runtime.stream(request(owner, stop=("STOP",), max_tokens=13)))
    assert "".join(c.text for c in chunks) == "aa"
    assert chunks[-1].finish_reason == "stop"
    assert len([c for c in chunks if c.finish_reason is not None]) == 1


def test_exclusive_emits_exactly_one_terminal_and_preserves_backend_eos(runtime_factory):
    runtime, owner, backend = runtime_factory()
    backend.exclusive_terminal = "stop"
    chunks = list(runtime.stream(request(owner, temperature=.7)))
    terminals = [chunk for chunk in chunks if chunk.finish_reason is not None]
    assert len(terminals) == 1, "one sampled request emitted multiple terminal chunks"
    assert terminals[0].finish_reason == "stop", "EOS at output cap was overwritten as length"
    assert backend.generator_closes == 1


def test_exclusive_failure_remains_original_error_and_runtime_recovers(runtime_factory):
    runtime, owner, backend = runtime_factory()
    failure = RuntimeError("injected verifier failure")
    backend.exclusive_error = failure
    with pytest.raises(RuntimeError, match="injected verifier failure") as caught:
        list(runtime.stream(request(owner, temperature=.7)))
    assert caught.value is failure
    backend.exclusive_error = None
    assert runtime.generate(request(owner)).text == "AB"
    assert runtime.stats()["failed"] == 1


def test_queue_full_and_close_do_not_destroy_another_owner(runtime_factory):
    runtime, first_owner, backend = runtime_factory(max_queue_size=1, batch_wait_ms=0)
    second_owner = runtime.acquire()
    backend.gate.clear()
    running = runtime.stream(request(first_owner, temperature=.7))
    assert backend.entered.wait(1)
    queued = runtime.stream(request(second_owner))
    with pytest.raises(native.NativeRuntimeError, match="queue is full"):
        runtime.stream(request(second_owner))
    with pytest.raises(native.NativeRuntimeError, match="active or queued"):
        runtime.release(first_owner)
    queued.close()
    backend.gate.set()
    list(running)
    runtime.control(lambda: None)
    runtime.release(first_owner)
    assert runtime.generate(request(second_owner)).text == "AB"


def test_slow_consumer_failure_isolated_from_next_request(runtime_factory):
    runtime, owner, backend = runtime_factory(output_queue_size=1, batch_wait_ms=0)
    stalled = runtime.stream(request(owner, max_tokens=4))
    assert stalled._job.done.wait(1)
    with pytest.raises(native.NativeRuntimeError, match="consumer is too slow"):
        list(stalled)
    # A one-token request fits the same queue and proves the worker survived.
    result = runtime.generate(request(owner, max_tokens=1))
    assert result.text == "A"
    assert result.finish_reason == "length"


def test_multirow_prefill_cancellation_keeps_slot_until_backend_can_remove(runtime_factory):
    runtime, owner, backend = runtime_factory(max_batch_size=2, batch_wait_ms=25)
    backend.prefill_cancel = True
    cancelled = runtime.stream(request(owner, "a"))
    surviving = runtime.stream(request(owner, "b"))
    assert backend.prefilled.wait(1)
    cancelled.close()
    following = runtime.stream(request(owner, "c"))
    backend.gate.set()
    assert "".join(c.text for c in surviving) == "AB"
    assert "".join(c.text for c in following) == "AB"
    engine = backend.engines[0]
    assert (0, False) in engine.removals, "test did not exercise unremovable multirow prefill"
    assert (0, True) in engine.removals, "tombstoned backend row was never removed"
    assert engine.peak_rows <= 2, "cancelled-but-live row was reused before backend removal"
    assert list(cancelled) == []
    assert runtime.stats()["cancelled"] == 1


def test_external_cancel_event_prevents_admission_before_any_backend_work(runtime_factory):
    runtime, owner, backend = runtime_factory()
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(native.NativeRuntimeError, match="cancel"):
        runtime.stream(request(owner), cancel_event=cancelled)
    assert backend.engines == []
    assert runtime.stats()["submitted"] == 0


def test_cache_partition_changes_for_scope_key_depth_and_layout():
    one = native.NativeRequest("p", 2, cache_scope="alpha", cache_key="session", draft_tokens=2)
    baseline = native.NativeRuntime._tenant(one, "batch")
    variants = [
        native.NativeRequest("p", 2, cache_scope="beta", cache_key="session", draft_tokens=2),
        native.NativeRequest("p", 2, cache_scope="alpha", cache_key="other", draft_tokens=2),
        native.NativeRequest("p", 2, cache_scope="alpha", cache_key="session", draft_tokens=3),
    ]
    assert all(native.NativeRuntime._tenant(value, "batch") != baseline for value in variants)
    assert native.NativeRuntime._tenant(one, "exclusive") != baseline


def test_late_target_request_prefills_before_running_request_finishes(runtime_factory):
    """Faithfully model VLM's free-slots >= prefill_batch_size admission gate."""
    runtime, owner, backend = runtime_factory(max_batch_size=2, batch_wait_ms=0)
    events = []
    first_decode = threading.Event()
    backend.gate.clear()

    class AdmissionGatedBatch:
        def __init__(self, model, processor, **kwargs):
            self.prefill_batch_size = kwargs["prefill_batch_size"]
            self.completion_batch_size = kwargs["completion_batch_size"]
            self._generation_batch = []
            self.waiting, self.live = {}, {}
            self.counter = 0
            self.paused = False

        def insert(self, prompts, max_tokens, prompt_kwargs):
            ids = []
            for prompt, cap in zip(prompts, max_tokens):
                uid = self.counter
                self.counter += 1
                self.waiting[uid] = [cap, 0]
                ids.append(uid)
            return ids

        def next(self):
            responses = []
            if self.live and not self.paused:
                self.paused = True
                first_decode.set()
                assert backend.gate.wait(2)
            for uid, row in list(self.live.items()):
                row[1] += 1
                finish = "length" if row[1] >= row[0] else None
                responses.append(SimpleNamespace(uid=uid, token=65 + uid, finish_reason=finish))
                if finish:
                    self.live.pop(uid)
                    events.append(("finished", uid))
            slots = self.completion_batch_size - len(self.live)
            if self.waiting and slots >= self.prefill_batch_size:
                for uid in list(self.waiting)[:self.prefill_batch_size]:
                    self.live[uid] = self.waiting.pop(uid)
                    events.append(("prefilled", uid))
            self._generation_batch[:] = self.live
            return [], responses

        def remove(self, uid):
            result = self.waiting.pop(uid, None) is not None or self.live.pop(uid, None) is not None
            self._generation_batch[:] = self.live
            return result

        @property
        def has_work(self):
            return bool(self.waiting or self.live)

        def close(self):
            pass

    backend.batch_class = lambda: AdmissionGatedBatch
    first = runtime.stream(request(owner, "first", max_tokens=8))
    assert first_decode.wait(1)
    second = runtime.stream(request(owner, "later", max_tokens=2))
    backend.gate.set()
    assert "".join(chunk.text for chunk in first) == "A" * 8
    assert "".join(chunk.text for chunk in second) == "B" * 2
    assert events.index(("prefilled", 1)) < events.index(("finished", 0)), (
        "advertised continuous mode drained the first request before admitting the second"
    )
