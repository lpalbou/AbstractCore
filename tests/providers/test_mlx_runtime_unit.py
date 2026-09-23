"""CPU-only native runtime contract tests; no MLX import or model allocation."""

import threading
import time
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from abstractcore.providers import mlx_runtime as module
from abstractcore.providers.mlx_runtime import NativeRequest, NativeRuntime, NativeRuntimeError


class Array:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class Detokenizer:
    def __init__(self):
        self.reset()

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
    def __init__(self):
        self.calls, self.engines, self.exclusive = [], [], []
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()
        self.mx = SimpleNamespace(array=Array, get_peak_memory=lambda: 1024)
        self.semantic_calls = []
        self.model = SimpleNamespace(language_model=object(), get_input_embeddings=self.embeddings)
        self.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda text, **kw: [ord(c) for c in text]),
            detokenizer=Detokenizer(),
        )
        self.drafter = SimpleNamespace(speculative_total_rounds=0, speculative_total_accepted=0,
                                       speculative_total_drafted=0)
        backend = self

        class Engine:
            def __init__(self, model, processor, **kwargs):
                self.kwargs, self.rows, self.uid, self.closed = kwargs, {}, 0, False
                self.inserted, self.removed, self.next_calls = [], [], 0
                backend.engines.append(self)

            def insert(self, prompts, max_tokens, prompt_kwargs):
                backend.calls.append(("insert", threading.get_ident()))
                result = []
                for prompt, limit, kwargs in zip(prompts, max_tokens, prompt_kwargs):
                    uid = self.uid
                    self.uid += 1
                    self.rows[uid] = [prompt, limit, 0, kwargs]
                    self.inserted.append((uid, prompt, kwargs))
                    result.append(uid)
                return result

            def next(self):
                backend.calls.append(("next", threading.get_ident()))
                backend.entered.set()
                assert backend.gate.wait(3)
                self.next_calls += 1
                time.sleep(0.002)
                progress, responses = [], []
                for uid, row in list(self.rows.items()):
                    prompt, limit, count, kwargs = row
                    if not count:
                        progress.append(SimpleNamespace(uid=uid, prompt_tokens=len(prompt),
                                                        prompt_tps=100, cached_tokens=0))
                    token = ord("A") + count
                    row[2] += 1
                    finish = "length" if row[2] >= limit else None
                    responses.append(SimpleNamespace(uid=uid, token=token, finish_reason=finish))
                    if finish:
                        del self.rows[uid]
                if self.kwargs["draft_model"]:
                    backend.drafter.speculative_total_rounds += 1
                    backend.drafter.speculative_total_accepted += len(responses)
                    backend.drafter.speculative_total_drafted += len(responses) * 2
                return progress, responses

            def remove(self, uid):
                self.removed.append(uid)
                return self.rows.pop(uid, None) is not None

            @property
            def has_work(self):
                return bool(self.rows)

            def close(self):
                self.closed = True

        self.BatchGenerator = Engine

    def semantic_extra_hash(self, **kwargs):
        self.semantic_calls.append(kwargs)
        data = json.dumps([kwargs["tenant"], kwargs["image_hash"]]).encode()
        return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")

    def embeddings(self, ids, pixels, **kwargs):
        self.calls.append(("embeddings", threading.get_ident()))
        return SimpleNamespace(to_dict=lambda: {"inputs_embeds": ids, "ignored": None})

    def stream_generate(self, model, processor, prompt, **kwargs):
        self.exclusive.append(kwargs)
        self.calls.append(("exclusive", threading.get_ident()))
        if prompt == "fail":
            raise ValueError("backend failed")
        for n in range(kwargs["max_tokens"]):
            yield SimpleNamespace(text=chr(ord("a") + n), token=n + 1,
                                  prompt_tokens=len(prompt), generation_tokens=n + 1,
                                  cached_tokens=0, prompt_tps=100, generation_tps=100,
                                  peak_memory=0.1)


@pytest.fixture
def setup_runtime(monkeypatch):
    backend, runtimes = Backend(), []
    monkeypatch.setattr(module, "_load_backend", lambda: backend)

    def make(**kwargs):
        runtime = NativeRuntime(backend.model, backend.processor, backend.drafter, "mtp", **kwargs)
        runtimes.append(runtime)
        owner = runtime.acquire()
        return runtime, owner, backend

    yield make
    backend.gate.set()
    for runtime in runtimes:
        runtime.close()


def request(owner, **kwargs):
    return NativeRequest("hello", 4, owner_id=owner, **kwargs)


def test_request_is_deeply_immutable():
    sampling = {"logit_bias": {1: 2}, "x": [1, 2]}
    media = bytearray(b"png")
    req = NativeRequest("x", 4, sampling=sampling, media=[(0, media)], stop=["STOP"])
    sampling["logit_bias"][1] = 7
    media[0] = 0
    assert req.sampling["logit_bias"][1] == 2
    assert req.sampling["x"] == (1, 2)
    assert req.media == ((0, b"png"),)
    with pytest.raises(TypeError):
        req.sampling["logit_bias"][1] = 8


@pytest.mark.parametrize("kwargs", [{"max_tokens": 0}, {"draft_tokens": True},
                                   {"temperature": float("nan")}, {"top_p": 2},
                                   {"seed": True}, {"stop": [""]}])
def test_request_validation(kwargs):
    params = {"prompt": "x", "max_tokens": 3, **kwargs}
    with pytest.raises((ValueError, TypeError)):
        NativeRequest(**params)


def test_runtime_requires_lease(setup_runtime):
    runtime, owner, _ = setup_runtime()
    with pytest.raises(NativeRuntimeError, match="lease"):
        runtime.generate(NativeRequest("hi", 4))
    runtime.release(owner)
    assert runtime.stats()["closed"]
    with pytest.raises(NativeRuntimeError, match="closed"):
        runtime.acquire()


def test_generate_routes_all_work_to_worker(setup_runtime):
    runtime, owner, backend = setup_runtime()
    result = runtime.generate(request(owner))
    assert result.text == "ABCD"
    assert result.generation_tokens == 4
    assert result.prompt_tokens == 5
    assert result.finish_reason == "length"
    assert result.metadata["execution"]["mode"] == "continuous"
    assert result.metadata["execution"]["ttft_s"] > 0
    assert all(ident == runtime._thread.ident for _, ident in backend.calls)
    assert backend.engines[0].inserted[0][2]["inputs_embeds"].tolist() == [[104, 101, 108, 108, 111]]


def test_compatible_requests_really_share_cohort(setup_runtime):
    runtime, owner, backend = setup_runtime(batch_wait_ms=30)
    a = runtime.stream(request(owner, draft_tokens=2))
    b = runtime.stream(request(owner, draft_tokens=2))
    left, right = list(a), list(b)
    assert "".join(x.text for x in left) == "ABCD"
    assert "".join(x.text for x in right) == "ABCD"
    assert len(backend.engines) == 1
    assert backend.engines[0].kwargs["draft_block_size"] == 3
    assert left[-1].metadata["execution"]["peak_batch_size"] == 2
    assert left[-1].metadata["speculation"]["used"]
    assert left[-1].metadata["speculation"]["stats"]["accounting"] == "cohort"
    assert left[0].metadata is not right[0].metadata


def test_mixed_depths_have_separate_engines(setup_runtime):
    runtime, owner, backend = setup_runtime(batch_wait_ms=30)
    handles = [runtime.stream(request(owner, draft_tokens=n)) for n in (1, 3)]
    results = [list(h)[-1] for h in handles]
    assert [e.kwargs["draft_block_size"] for e in backend.engines] == [2, 4]
    assert all(r.metadata["execution"]["peak_batch_size"] == 1 for r in results)


def test_no_cache_and_cache_requests_do_not_share_engine(setup_runtime):
    cache = object()
    runtime, owner, backend = setup_runtime(cache_factory=lambda req: cache, batch_wait_ms=30)
    a = runtime.stream(request(owner))
    b = runtime.stream(request(owner, cache_key="thread", cache_scope="workspace"))
    list(a)
    list(b)
    assert [e.kwargs["apc_manager"] for e in backend.engines] == [None, cache]
    tenant = backend.engines[1].inserted[0][2]["_apc_tenant"]
    assert '"workspace","thread",0,"batch"' in tenant


def test_stream_stop_matching_across_token_boundaries(setup_runtime):
    runtime, owner, _ = setup_runtime()
    result = runtime.generate(request(owner, stop=("BC",)))
    assert result.text == "A"
    assert result.finish_reason == "stop"
    assert result.generation_tokens == 3


def test_exclusive_preserves_sampler_and_seed(setup_runtime):
    runtime, owner, backend = setup_runtime()
    result = runtime.generate(request(owner, temperature=0.7, top_p=0.9, top_k=20, seed=42,
                                      sampling={"presence_penalty": 0.4}))
    assert result.text == "abcd"
    assert result.metadata["execution"]["mode"] == "exclusive"
    assert backend.exclusive[0]["seed"] == 42
    assert backend.exclusive[0]["presence_penalty"] == 0.4
    assert not backend.engines


def test_exclusive_errors_reach_consumer(setup_runtime):
    runtime, owner, _ = setup_runtime()
    with pytest.raises(ValueError, match="backend failed"):
        runtime.generate(NativeRequest("fail", 4, temperature=0.2, owner_id=owner))


def test_mtp_penalty_fail_closed(setup_runtime):
    runtime, owner, _ = setup_runtime()
    with pytest.raises(NativeRuntimeError, match="cannot honor"):
        runtime.generate(request(owner, draft_tokens=2, sampling={"presence_penalty": 0.4}))


def test_unknown_controls_not_silently_ignored(setup_runtime):
    runtime, owner, _ = setup_runtime()
    with pytest.raises(NativeRuntimeError, match="Unsupported"):
        runtime.generate(request(owner, sampling={"imaginary_penalty": 7}))


def test_bounded_admission_and_release_barrier(setup_runtime):
    runtime, owner, backend = setup_runtime(max_queue_size=1, batch_wait_ms=0)
    backend.gate.clear()
    a = runtime.stream(request(owner))
    assert backend.entered.wait(1)
    b = runtime.stream(request(owner, draft_tokens=2))
    with pytest.raises(NativeRuntimeError, match="full"):
        runtime.stream(request(owner))
    with pytest.raises(NativeRuntimeError, match="active or queued"):
        runtime.release(owner)
    backend.gate.set()
    list(a)
    list(b)
    runtime.release(owner)
    assert runtime.stats()["closed"]


def test_control_runs_after_earlier_generation_on_same_worker(setup_runtime):
    runtime, owner, backend = setup_runtime(batch_wait_ms=0)
    backend.gate.clear()
    handle = runtime.stream(request(owner))
    assert backend.entered.wait(1)
    with ThreadPoolExecutor() as pool:
        result = pool.submit(runtime.control, lambda: (threading.get_ident(), backend.engines[0].closed))
        assert not result.done()
        backend.gate.set()
        list(handle)
        assert result.result(2) == (runtime._thread.ident, True)


def test_queue_deadline_does_not_expire_running_job(setup_runtime):
    runtime, owner, backend = setup_runtime(batch_wait_ms=0, queue_timeout_s=0.04)
    backend.gate.clear()
    running = runtime.stream(request(owner))
    assert backend.entered.wait(1)
    waiting = runtime.stream(request(owner, draft_tokens=2))
    time.sleep(0.06)
    backend.gate.set()
    assert list(running)[-1].finish_reason == "length"
    with pytest.raises(NativeRuntimeError, match="timed out"):
        list(waiting)


def test_slow_consumer_does_not_block_other_requests(setup_runtime):
    runtime, owner, backend = setup_runtime(output_queue_size=1, batch_wait_ms=0)
    slow = runtime.stream(request(owner))
    assert slow._job.done.wait(1)
    with pytest.raises(NativeRuntimeError, match="too slow"):
        list(slow)
    assert runtime.generate(NativeRequest("ok", 1, owner_id=owner)).text == "A"


def test_cancel_can_close_blocked_consumer_from_another_thread(setup_runtime):
    runtime, owner, backend = setup_runtime(batch_wait_ms=0)
    backend.gate.clear()
    stream = runtime.stream(request(owner))
    assert backend.entered.wait(1)
    with ThreadPoolExecutor() as pool:
        consumer = pool.submit(lambda: next(stream, "cancelled"))
        stream.close()
        assert consumer.result(1) == "cancelled"
    backend.gate.set()
    assert stream._job.done.wait(1)
    assert runtime.stats()["cancelled"] == 1


def test_release_one_owner_keeps_other_owner_alive(setup_runtime):
    runtime, owner, _ = setup_runtime()
    other = runtime.acquire()
    runtime.release(owner)
    assert not runtime.stats()["closing"]
    assert runtime.generate(request(other)).text == "ABCD"
    runtime.release(other)
    assert runtime.stats()["closed"]


def test_target_admits_staggered_request_but_mtp_waits_for_cohort(setup_runtime):
    for depth in (0, 2):
        runtime, owner, backend = setup_runtime(batch_wait_ms=0)
        backend.engines.clear()
        backend.entered.clear()
        backend.gate.clear()
        first = runtime.stream(request(owner, draft_tokens=depth))
        assert backend.entered.wait(1)
        later = runtime.stream(request(owner, draft_tokens=depth))
        backend.gate.set()
        list(first)
        result = list(later)[-1]
        assert len(backend.engines) == (1 if depth == 0 else 2)
        assert result.metadata["execution"]["mode"] == ("continuous" if depth == 0 else "cohort")
        runtime.release(owner)


def test_on_close_executes_once_on_worker(setup_runtime):
    calls = []
    runtime, owner, _ = setup_runtime(on_close=lambda: calls.append(threading.get_ident()))
    runtime.release(owner)
    runtime.close()
    assert calls == [runtime._thread.ident]


def test_control_exception_is_delivered_and_worker_recovers(setup_runtime):
    runtime, owner, _ = setup_runtime()

    def failed():
        raise ValueError("bad cache")

    with pytest.raises(ValueError, match="bad cache"):
        runtime.control(failed)
    assert runtime.generate(request(owner)).text == "ABCD"


def test_external_cancel_event_is_installed_before_admission(setup_runtime):
    runtime, owner, backend = setup_runtime(batch_wait_ms=0)
    backend.gate.clear()
    cancellation = threading.Event()
    stream = runtime.stream(request(owner), cancel_event=cancellation)
    assert stream._job.cancelled is cancellation
    assert backend.entered.wait(1)
    with ThreadPoolExecutor() as pool:
        consumer = pool.submit(lambda: next(stream, "cancelled"))
        cancellation.set()
        assert consumer.result(1) == "cancelled"
    backend.gate.set()
    assert stream._job.done.wait(1)
    with pytest.raises(NativeRuntimeError, match="cancelled before admission"):
        runtime.stream(request(owner), cancel_event=cancellation)
    with pytest.raises(TypeError, match="threading.Event"):
        runtime.stream(request(owner), cancel_event=object())


def test_cache_semantic_hash_ignores_text_suffix_and_preserves_scope_depth(setup_runtime):
    cache = object()
    runtime, owner, backend = setup_runtime(cache_factory=lambda req: cache)
    requests = [
        NativeRequest("prefix one", 1, cache_key="chat", owner_id=owner),
        NativeRequest("prefix two", 1, cache_key="chat", owner_id=owner),
        NativeRequest("prefix two", 1, cache_key="chat", cache_scope="other", owner_id=owner),
        NativeRequest("prefix two", 1, cache_key="chat", draft_tokens=2, owner_id=owner),
    ]
    for req in requests:
        runtime.generate(req)
    hashes = [engine.inserted[0][2]["_apc_semantic_hash"] for engine in backend.engines]
    assert hashes[0] == hashes[1]
    assert hashes[1] != hashes[2] and hashes[1] != hashes[3]
    assert all(call["media"] is None for call in backend.semantic_calls)
    assert all(call["model"] is backend.model and call["processor"] is backend.processor
               for call in backend.semantic_calls)


def test_cache_semantic_hash_preserves_image_bytes(setup_runtime, monkeypatch):
    from abstractcore.providers import mlx_qwen4

    def prepare(model, processor, prompt, parts):
        records = [{"index": index, "content": part.content, "kind": "image", "tokens": 2,
                    "transport": "mlx_vlm_native"} for index, part in parts]
        return {"input_ids": Array([[1, 2, 3]]), "pixel_values": Array([[1]])}, records

    monkeypatch.setattr(mlx_qwen4, "prepare_images", prepare)
    cache = object()
    runtime, owner, backend = setup_runtime(cache_factory=lambda req: cache)
    first = runtime.generate(NativeRequest("same", 1, cache_key="chat", media=((0, b"red"),), owner_id=owner))
    second = runtime.generate(NativeRequest("same", 1, cache_key="chat", media=((0, b"blue"),), owner_id=owner))
    salts = [engine.inserted[0][2]["_apc_semantic_hash"] for engine in backend.engines]
    assert salts[0] != salts[1]
    assert first.media_records[0]["content"] == b"red"
    assert second.media_records[0]["content"] == b"blue"
