"""One generation at a time on one HuggingFace provider instance (backlog 0859).

The server's `/v1` route lends a resident instance (a host's pool) instead of building a second
copy; the pool's runs and `/v1` calls then share ONE llama.cpp context, so the instance itself
must serialize its generations, streams included (held across yields, released on close, from
whichever thread resumes the stream).
"""
from __future__ import annotations

import threading
import time
from typing import Any, Iterator, List

from abstractcore.core.types import GenerateResponse
from abstractcore.providers import huggingface_provider as hf


def _bare_provider(lane) -> Any:
    p = hf.HuggingFaceProvider.__new__(hf.HuggingFaceProvider)
    p.model = "unsloth/Qwen3.5-4B-GGUF"
    p.model_type = "gguf"
    p.llm = object()
    p.logger = hf.logging.getLogger("test") if hasattr(hf, "logging") else None
    p._generation_gate = hf._InstanceGenerationGate()
    p._generate_gguf = lane
    return p


class _Overlap:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active = 0
        self.peak = 0

    def __enter__(self):
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    def __exit__(self, *a):
        with self.lock:
            self.active -= 1


def test_two_threads_never_generate_at_once_on_one_instance() -> None:
    seen = _Overlap()

    def lane(*args: Any, **kwargs: Any) -> GenerateResponse:
        with seen:
            time.sleep(0.05)
        return GenerateResponse(content="ok", model="m")

    p = _bare_provider(lane)
    threads = [threading.Thread(target=lambda: p._generate_internal("hi")) for _ in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert seen.peak == 1


def test_a_stream_holds_the_gate_across_yields_and_releases_on_close_from_another_thread() -> None:
    seen = _Overlap()
    started = threading.Event()

    def lane(prompt: Any, messages: Any, system_prompt: Any, tools: Any, media: Any, stream: bool, *a: Any, **k: Any):
        if not stream:
            with seen:
                time.sleep(0.01)
            return GenerateResponse(content="ok", model="m")

        def gen() -> Iterator[GenerateResponse]:
            with seen:
                for i in range(3):
                    started.set()
                    time.sleep(0.02)
                    yield GenerateResponse(content=str(i), model="m")
        return gen()

    p = _bare_provider(lane)
    stream = p._generate_internal("hi", stream=True)
    first = next(stream)  # the gate is taken here (prefill + first chunk)
    assert first.content == "0"
    other_done = threading.Event()
    threading.Thread(target=lambda: (p._generate_internal("x"), other_done.set())).start()
    assert not other_done.wait(0.1), "a second generation ran while the stream held the instance"
    # The stream is resumed and closed on ANOTHER thread (the server's pool does that).
    closer = threading.Thread(target=lambda: (next(stream), stream.close()))
    closer.start()
    closer.join()
    assert other_done.wait(2.0), "closing the stream did not release the instance"
    assert seen.peak == 1


def test_the_gate_is_reentrant_on_the_holding_thread() -> None:
    gate = hf._InstanceGenerationGate()
    out: List[str] = []
    gate.run(lambda: gate.run(lambda: out.append("inner")))
    assert out == ["inner"] and gate._owner is None and not gate._lock.locked()
