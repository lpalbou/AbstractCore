"""Real weights (ADR 0011): N concurrent generations on ONE llama.cpp weights instance.

Runs only with ABSTRACTCORE_RUN_GGUF_TESTS=1 and unsloth/Qwen3.5-4B-GGUF (Q4_K_M) in the HF cache.
Checks: one weights pointer for every holder and slot; 4 concurrent requests are never slower than
serial (separate contexts interleave, they do not batch weight reads); a short request is not blocked
behind a long one; two streams interleave; RSS grows by the slots' KV only (never a second copy)."""
from __future__ import annotations

import gc
import os
import threading
import time

import pytest

MODEL = os.environ.get("ABSTRACTCORE_GGUF_CONCURRENCY_MODEL", "unsloth/Qwen3.5-4B-GGUF")

pytestmark = pytest.mark.skipif(os.environ.get("ABSTRACTCORE_RUN_GGUF_TESTS") != "1",
                                reason="GGUF tests are heavy; set ABSTRACTCORE_RUN_GGUF_TESTS=1 to run")


def _rss() -> int:
    import psutil

    return int(psutil.Process().memory_info().rss)


def _ask(p, q, **kw):
    return p.generate(q + " /no_think", max_output_tokens=96, temperature=0, **kw)


def test_four_concurrent_generations_on_one_weights_instance():
    from abstractcore.providers import gguf_shared, gguf_slots
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    rss0 = _rss()
    a = HuggingFaceProvider(model=MODEL, max_tokens=4096)
    rss_loaded = _rss()
    b = HuggingFaceProvider(model=MODEL, max_tokens=4096)  # another pool's holder: attaches
    rss_b = _rss()
    assert a.__dict__["_gguf_scheduler"] is b.__dict__["_gguf_scheduler"]
    assert rss_b - rss_loaded < 200 * 2**20, f"a second holder must not load anything ({(rss_b - rss_loaded) / 2**20:.0f} MiB)"
    pool = a.__dict__["_gguf_scheduler"]
    # Equal decode lengths (each answer runs to max_output_tokens), so serial = 4 x one answer.
    qs = [f"Count from {n} upward, one number per line, and never stop." for n in (1, 100, 1000, 5000)]
    _ask(a, "Hi")  # warm

    t = time.time()
    serial = [_ask(a, q).content for q in qs]
    serial_s = time.time() - t

    def concurrent_round():
        out = {}
        t = time.time()
        threads = [threading.Thread(target=lambda i=i: out.__setitem__(i, _ask(a if i % 2 else b, qs[i]).content)) for i in range(4)]
        [x.start() for x in threads]
        [x.join(300) for x in threads]
        return out, time.time() - t

    _, first_round_s = concurrent_round()  # builds the extra slots (lazily, each through the fit check)
    out, concurrent_s = concurrent_round()
    st = pool.state()
    weights = {gguf_shared.weights_identity(llm) for llm in pool.slot_llms()}
    rss_slots = _rss()
    print(f"\nRSS: before {rss0 / 2**30:.2f} GiB, loaded {rss_loaded / 2**30:.2f}, 2 holders {rss_b / 2**30:.2f}, "
          f"{st['slots']} slots {rss_slots / 2**30:.2f}; serial {serial_s:.2f} s, first concurrent round (slots built) "
          f"{first_round_s:.2f} s, concurrent {concurrent_s:.2f} s "
          f"({concurrent_s / serial_s:.2f} x); weights instances {len(weights)}; report {gguf_shared.report()}")
    assert len(weights) == 1 and st["slots"] >= 2
    assert [out[i] for i in range(4)] == serial, "same answers at temperature 0"
    # Separate contexts interleave on the device; they do not batch weight reads, so the wall
    # clock of N equal decodes is about the serial one on a saturated GPU (Metal, measured 0.92 x)
    # -- never worse. What they remove is head-of-line blocking, asserted below.
    assert concurrent_s < 1.1 * serial_s

    # NO HEAD-OF-LINE BLOCKING: a short request sent while a long one decodes finishes long before it.
    done = {}

    def timed(name, p, q, n):
        t0 = time.time()
        p.generate(q + " /no_think", max_output_tokens=n, temperature=0)
        done[name] = (t0, time.time())

    long_t = threading.Thread(target=timed, args=("long", a, qs[0], 400))
    long_t.start()
    time.sleep(0.3)
    timed("short", b, "Say OK.", 4)
    long_t.join(300)
    short_s = done["short"][1] - done["short"][0]
    long_s = done["long"][1] - done["long"][0]
    print(f"head-of-line: short request {short_s:.2f} s while a {long_s:.2f} s request decoded")
    assert done["short"][1] < done["long"][1] and short_s < 0.5 * long_s
    # each extra slot costs its KV, never the weights again
    assert rss_slots - rss_b < (rss_loaded - rss0) * 0.5

    # two streams interleave
    stamps = {0: [], 1: []}

    def stream(i, p):
        for _chunk in p.generate(qs[i] + " /no_think", max_output_tokens=48, temperature=0, stream=True):
            stamps[i].append(time.time())

    s = [threading.Thread(target=stream, args=(0, a)), threading.Thread(target=stream, args=(1, b))]
    [x.start() for x in s]
    [x.join(300) for x in s]
    assert stamps[0] and stamps[1]
    assert stamps[1][0] < stamps[0][-1] and stamps[0][0] < stamps[1][-1], "the two streams overlap in time"

    a.unload_model(MODEL)
    assert not pool.closed and b.llm is not None, "the other holder keeps the model"
    b.unload_model(MODEL)
    gc.collect()
    assert pool.closed and gguf_shared.report() == [] and gguf_slots.all_schedulers() == []
