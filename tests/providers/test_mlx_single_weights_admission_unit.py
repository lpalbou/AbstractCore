"""MLX first-load admission uses the existing registries, before any target allocation."""
import gc
import logging
import sys
import threading
import types
import weakref
from pathlib import Path

import pytest

from abstractcore.exceptions import InvalidRequestError
from abstractcore.providers import mlx_native_session as native
from abstractcore.providers import mlx_provider as mp


@pytest.fixture()
def loaders(monkeypatch):
    monkeypatch.setattr(native, "_SESSIONS", weakref.WeakValueDictionary())
    monkeypatch.setattr(mp, "_SHARED_MLX_MODELS", weakref.WeakValueDictionary())
    monkeypatch.setattr(native, "resolve_native_drafter_path", lambda p: str(Path(p).resolve()))
    monkeypatch.setattr("abstractcore.providers.weights_fingerprint.weights_fingerprint_for_dir", lambda p: "revision:" + p)
    calls = []
    heads = []
    vlm = types.ModuleType("mlx_vlm")
    def allocate(path):
        calls.append(str(path))
        return object(), object()
    vlm.load = allocate
    drafters = types.ModuleType("mlx_vlm.speculative.drafters")
    def load_drafter(path):
        heads.append(path)
        return types.SimpleNamespace(), "mtp"
    drafters.load_drafter = load_drafter
    monkeypatch.setitem(sys.modules, "mlx_vlm", vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative.drafters", drafters)
    def plain(path, *, sharing=True):
        p = mp.MLXProvider.__new__(mp.MLXProvider)
        p.logger = logging.getLogger("test_mlx_admission")
        p._prompt_cache_store = object()
        p._share_loaded_model = sharing
        p._load_or_adopt_shared_model(str(path), lambda: allocate(path))
        return p
    return plain, calls, heads


def test_native_profiles_refuse_second_target_before_head_or_target_load(loaders, tmp_path):
    _plain, calls, heads = loaders
    target = tmp_path / "target"
    head = tmp_path / "head"
    first = native.load_native_session(str(target), str(head))
    assert native.load_native_session(str(target), str(head)) is first
    for other_head in (None, str(tmp_path / "other-head")):
        with pytest.raises(InvalidRequestError, match="different drafter"):
            native.load_native_session(str(target), other_head)
    assert len(calls) == 1 and len(heads) == 1
    del first
    gc.collect()
    replacement = native.load_native_session(str(target))
    assert replacement.model is not None and len(calls) == 2


@pytest.mark.parametrize("native_first", [False, True])
def test_plain_and_native_lanes_cannot_load_the_same_target(loaders, tmp_path, native_first):
    plain, calls, _heads = loaders
    target = tmp_path / "target"
    if native_first:
        held = native.load_native_session(str(target))
        with pytest.raises(InvalidRequestError, match="already loaded through native"):
            plain(target)
    else:
        held = plain(target)
        with pytest.raises(InvalidRequestError, match="already loaded through mlx-lm"):
            native.load_native_session(str(target))
    assert held is not None and len(calls) == 1


def test_plain_sharing_canonicalizes_aliases_and_refuses_optout_before_loading(loaders, tmp_path):
    plain, calls, _heads = loaders
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    first = plain(target)
    second = plain(alias)
    assert first._shared_model is second._shared_model and len(calls) == 1
    with pytest.raises(InvalidRequestError, match="sharing cannot be disabled"):
        plain(tmp_path / "other", sharing=False)
    assert len(calls) == 1
    first._shared_model.holders.discard(first)
    second._shared_model.holders.discard(second)
    del first, second
    gc.collect()
    replacement = native.load_native_session(str(alias))
    assert replacement.model is not None and len(calls) == 2


def test_native_alias_reuses_target_and_other_artifact_is_allowed(loaders, tmp_path):
    _plain, calls, _heads = loaders
    target = tmp_path / "target"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    first = native.load_native_session(str(target))
    assert native.load_native_session(str(alias)) is first
    other = native.load_native_session(str(tmp_path / "another-quant"))
    assert first.model is not other.model and len(calls) == 2


@pytest.mark.parametrize("same_lane", [False, True])
def test_concurrent_mlx_first_loads_share_or_refuse_before_allocating_twice(loaders, tmp_path, same_lane):
    plain, calls, _heads = loaders
    target = tmp_path / "target"
    start = threading.Barrier(2)
    results = []
    def run(native_lane):
        start.wait()
        try:
            results.append(native.load_native_session(str(target)) if native_lane else plain(target))
        except InvalidRequestError as exc:
            results.append(exc)
    threads = [threading.Thread(target=run, args=(True,)), threading.Thread(target=run, args=(same_lane,))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert len(calls) == 1
    assert sum(isinstance(r, InvalidRequestError) for r in results) == (0 if same_lane else 1)


def test_native_profile_refusal_does_not_fall_back_to_a_plain_load(loaders, monkeypatch, tmp_path):
    _plain, calls, _heads = loaders
    target = str(tmp_path / "target")
    first = native.load_native_session(target)
    p = mp.MLXProvider.__new__(mp.MLXProvider)
    p._speculation_request = object()
    with pytest.raises(InvalidRequestError, match="different drafter"):
        p._enter_mtp_lane(target, {"drafter": str(tmp_path / "head")})
    assert first.model is not None and len(calls) == 1


@pytest.mark.parametrize("reuse_role", ["head", "target", "plain"])
def test_native_drafter_cannot_be_loaded_again_in_another_role(loaders, tmp_path, reuse_role):
    plain, calls, heads = loaders
    target, head = tmp_path / "target", tmp_path / "head"
    resident = native.load_native_session(str(target), str(head))
    with pytest.raises(InvalidRequestError, match="second copy is never loaded"):
        if reuse_role == "head":
            native.load_native_session(str(tmp_path / "other-target"), str(head))
        elif reuse_role == "target":
            native.load_native_session(str(head))
        else:
            plain(head)
    assert resident.model is not None and len(calls) == len(heads) == 1


def test_plain_target_cannot_be_loaded_again_as_a_native_drafter(loaders, tmp_path):
    plain, calls, heads = loaders
    head = tmp_path / "head"
    resident = plain(head)
    with pytest.raises(InvalidRequestError, match="second copy is never loaded"):
        native.load_native_session(str(tmp_path / "target"), str(head))
    assert resident is not None and len(calls) == 1 and heads == []
