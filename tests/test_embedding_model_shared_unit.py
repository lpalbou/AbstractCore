"""ONE loaded embedding artifact per process (residency audit R3,
2026-10-29): two managers of one model (two accounts' runtime services) share its weights; an
unload of one detaches it; the last unload frees them. MUTANT: `_load_model` loading per manager
-> two loads -> RED."""
from __future__ import annotations

import pytest

from abstractcore.embeddings import manager as m
from abstractcore.exceptions import InvalidRequestError


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


def test_another_device_refuses_a_second_copy_of_the_same_model(fake_load):
    a, b = _manager(device="cpu"), _manager(device="gpu")
    a._load_model()
    with pytest.raises(InvalidRequestError, match="second copy is never loaded"):
        b._load_model()
    assert len(fake_load) == 1 and b.model is None


@pytest.mark.parametrize("setting,value", [("backend", "onnx"), ("trust_remote_code", True)])
def test_incompatible_embedding_profile_is_refused_before_allocation(fake_load, setting, value):
    a, b = _manager(), _manager()
    a._load_model()
    setattr(b, setting, value)
    with pytest.raises(InvalidRequestError, match="second copy is never loaded"):
        b._load_model()
    assert len(fake_load) == 1


def test_automatic_embedding_settings_adopt_an_existing_explicit_profile(fake_load):
    a, b = _manager(device="cpu"), _manager()
    a._load_model()
    b._load_model()
    assert a.model is b.model and len(fake_load) == 1


def test_embedding_artifact_aliases_share_and_new_artifacts_remain_distinct(fake_load, tmp_path, monkeypatch):
    path = tmp_path / "snapshot"
    path.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(path, target_is_directory=True)
    monkeypatch.setattr("abstractcore.utils.model_cache.resolve_hf_load_snapshot", lambda model: path if model == "org/embed" else None)
    a, b = _manager("org/embed"), _manager(str(alias))
    a._load_model()
    b._load_model()
    c = _manager(str(tmp_path / "another-quant"))
    c._load_model()
    assert a.model is b.model and a.model is not c.model and len(fake_load) == 2


def test_embedding_concurrent_first_profiles_allocate_once_and_unload_allows_change(fake_load):
    import threading
    a, b = _manager(device="cpu"), _manager(device="gpu")
    start = threading.Barrier(2)
    results = []
    def load(manager):
        start.wait()
        try:
            manager._load_model()
            results.append(manager)
        except InvalidRequestError as exc:
            results.append(exc)
    threads = [threading.Thread(target=load, args=(manager,)) for manager in (a, b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert len(fake_load) == 1 and sum(isinstance(r, InvalidRequestError) for r in results) == 1
    loaded = next(r for r in results if isinstance(r, m.EmbeddingManager))
    loaded._release_shared_model()
    loaded.model = None
    replacement = _manager(device="gpu" if loaded.device_preference == "cpu" else "cpu")
    replacement._load_model()
    assert replacement.model is not None and len(fake_load) == 2


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
