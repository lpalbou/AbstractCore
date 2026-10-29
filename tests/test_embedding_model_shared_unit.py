"""ONE loaded embedding model per (model, device, backend) in a process (residency audit R3,
2026-10-29): two managers of one model (two accounts' runtime services) share its weights; an
unload of one detaches it; the last unload frees them. MUTANT: `_load_model` loading per manager
-> two loads -> RED."""
from __future__ import annotations

import pytest

from abstractcore.embeddings import manager as m


class _Weights:
    def __init__(self):
        self.closed = False


@pytest.fixture()
def fake_load(monkeypatch):
    loads = []

    def load(self):
        w = _Weights()
        loads.append(w)
        self.model = w

    monkeypatch.setattr(m.EmbeddingManager, "_load_model_unshared", load)
    import weakref

    monkeypatch.setattr(m, "_SHARED_EMBEDDING_MODELS", weakref.WeakValueDictionary())
    return loads


def _manager(model_id="org/embed", device=None):
    mgr = object.__new__(m.EmbeddingManager)
    mgr.model_id = model_id
    mgr.provider = "huggingface"
    mgr.backend = None
    mgr.trust_remote_code = False
    mgr.device_preference = device
    mgr.model = None
    return mgr


def test_two_managers_of_one_model_share_its_weights(fake_load):
    a, b = _manager(), _manager()
    a._load_model()
    b._load_model()
    assert len(fake_load) == 1 and a.model is b.model
    assert m.shared_embedding_models() == [{"model": "org/embed", "holders": 2}]


def test_another_device_is_another_model(fake_load):
    a, b = _manager(device="cpu"), _manager(device="gpu")
    a._load_model()
    b._load_model()
    assert len(fake_load) == 2 and a.model is not b.model


def test_unloading_one_keeps_the_model_for_the_other_the_last_frees_it(fake_load):
    a, b = _manager(), _manager()
    a._load_model()
    b._load_model()
    assert a._release_shared_model() is True, "b still holds it"
    a.model = None
    assert m.shared_embedding_models() == [{"model": "org/embed", "holders": 1}]
    assert b._release_shared_model() is False
    b.model = None
    assert m.shared_embedding_models() == []
    c = _manager()
    c._load_model()
    assert len(fake_load) == 2, "after the last unload, the next manager loads again"


def test_a_dropped_manager_frees_the_shared_model(fake_load):
    import gc

    a = _manager()
    a._load_model()
    assert m.shared_embedding_models()
    del a
    gc.collect()
    assert m.shared_embedding_models() == []
