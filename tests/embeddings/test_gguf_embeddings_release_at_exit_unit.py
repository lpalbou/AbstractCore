"""A process that embedded with a GGUF model must not abort at exit (2.35.1).

Found on Apple silicon during the 0.14.0 profile proof: after every embedding call succeeded, the
interpreter died in `exit()` with ``ggml-metal-device.m: GGML_ASSERT([rsets->data count] == 0)``
(exit code 134) because the llama.cpp context was freed during module teardown, after the Metal
device. `release_shared_gguf_embeddings_at_exit` closes the resident GGUF encoders first; it is
registered with `atexit` at import. Simulated encoders: nothing from llama.cpp is loaded here.
"""

from __future__ import annotations

import atexit
import threading

import pytest

from abstractcore.embeddings import manager


class _FakeGgufEncoder:
    """Shaped like `_GgufEmbeddingEncoder`: a `_llm` and a `close()`."""

    def __init__(self) -> None:
        self._llm = object()
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


class _FakeSentenceTransformer:
    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:  # no `_llm`: not a llama.cpp encoder, must be left alone
        self.closed += 1


@pytest.fixture
def shared(monkeypatch: pytest.MonkeyPatch):
    import weakref

    table: "weakref.WeakValueDictionary" = weakref.WeakValueDictionary()
    monkeypatch.setattr(manager, "_SHARED_EMBEDDING_MODELS", table)
    monkeypatch.setattr(manager, "_SHARED_EMBEDDING_LOCK", threading.RLock())
    return table


def test_resident_gguf_encoders_are_closed_and_dropped(shared):
    gguf = _FakeGgufEncoder()
    st = _FakeSentenceTransformer()
    entries = [manager._SharedEmbeddingModel(("gguf",), gguf, "org/model-gguf", None),
               manager._SharedEmbeddingModel(("st",), st, "org/minilm", None)]
    for e in entries:
        shared[e.key] = e
    assert manager.release_shared_gguf_embeddings_at_exit() == 1
    assert gguf.closed == 1 and st.closed == 0
    assert entries[0].model is None and entries[1].model is st
    # Idempotent: a second run (another atexit handler, a test teardown) closes nothing twice.
    assert manager.release_shared_gguf_embeddings_at_exit() == 0 and gguf.closed == 1


def test_a_failing_close_does_not_stop_the_others(shared):
    bad = _FakeGgufEncoder()

    def boom():
        raise RuntimeError("already freed")

    bad.close = boom
    good = _FakeGgufEncoder()
    for key, model in (("a",), bad), (("b",), good):
        e = manager._SharedEmbeddingModel(key, model, str(key), None)
        shared[key] = e
        shared[key] = e
    assert manager.release_shared_gguf_embeddings_at_exit() == 1
    assert good.closed == 1


def test_the_release_is_registered_with_atexit():
    # atexit keeps no public registry; re-registering the same function is harmless and proves
    # the module did it (unregister returns silently only for registered callables in CPython,
    # so check through the module's own import-time call instead: the function exists and a
    # fresh registration + unregistration round-trips).
    atexit.register(manager.release_shared_gguf_embeddings_at_exit)
    atexit.unregister(manager.release_shared_gguf_embeddings_at_exit)
    source = open(manager.__file__, encoding="utf-8").read()
    assert "atexit.register(release_shared_gguf_embeddings_at_exit)" in source
