"""The residency snapshot must not treat a module as loaded while it is still importing.

2026-11-08, model-host.log 02:44: `abstractcore.embeddings.manager` "partially initialized …
has no attribute 'registered_embedding_managers'" from process_residency. The module was in
`sys.modules` from the first line of its import on another thread; the snapshot called a
function that did not exist yet. Loaded means the functions the snapshot calls exist.
"""

from __future__ import annotations

import sys
import types

from abstractcore.providers import process_residency as pr


def test_partially_initialised_embeddings_module_is_not_loaded(monkeypatch):
    partial = types.ModuleType(pr._EMBEDDINGS_MODULE)  # no functions yet: mid-import
    monkeypatch.setitem(sys.modules, pr._EMBEDDINGS_MODULE, partial)
    assert pr._backend_loaded("embeddings") is False
    assert pr._embedding_model_states() == []  # the snapshot reports "nothing resident", no AttributeError


def test_fully_initialised_embeddings_module_is_loaded(monkeypatch):
    ready = types.ModuleType(pr._EMBEDDINGS_MODULE)
    ready.registered_embedding_managers = lambda: []
    ready.resident_embedding_models = lambda: []
    monkeypatch.setitem(sys.modules, pr._EMBEDDINGS_MODULE, ready)
    assert pr._backend_loaded("embeddings") is True
    assert pr._embedding_model_states() == []


def test_absent_embeddings_module_is_not_loaded(monkeypatch):
    monkeypatch.delitem(sys.modules, pr._EMBEDDINGS_MODULE, raising=False)
    assert pr._backend_loaded("embeddings") is False
