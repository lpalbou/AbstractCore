"""Transformers profiles that cannot safely share refuse before another weights allocation."""
import logging
import sys
import threading
import types
import weakref
from pathlib import Path

import pytest

from abstractcore.exceptions import InvalidRequestError
from abstractcore.providers import hf_residency, huggingface_provider as hp


@pytest.fixture()
def load_model(monkeypatch):
    calls = []
    class Weights:
        def to(self, device):
            return self
        def eval(self):
            return self
    def allocate(model, **kwargs):
        calls.append(model)
        return Weights()
    transformers = types.ModuleType("transformers")
    transformers.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *a, **kw: types.SimpleNamespace())
    transformers.AutoTokenizer = transformers.AutoProcessor = types.SimpleNamespace(from_pretrained=lambda *a, **kw: object())
    transformers.AutoModel = transformers.AutoModelForCausalLM = transformers.AutoModelForImageTextToText = types.SimpleNamespace(from_pretrained=allocate)
    transformers.pipeline = lambda *a, **kw: object()
    utils = types.ModuleType("transformers.utils")
    utils.logging = types.SimpleNamespace(set_verbosity_error=lambda: None, disable_progress_bar=lambda: None,
                                          set_verbosity_warning=lambda: None, enable_progress_bar=lambda: None)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.utils", utils)
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setattr(hf_residency, "_HF_PROVIDERS", weakref.WeakSet())
    monkeypatch.setattr(hp, "_resolve_transformers_load_source", lambda model, **kw: hp._TransformersLoadSource(
        model=str(Path(model).resolve()), tokenizer=str(Path(model).resolve())))
    def load(path, *, vision=False):
        p = hp.HuggingFaceProvider.__new__(hp.HuggingFaceProvider)
        p.model = str(path)
        p.model_instance = None
        p.transformers_kwargs = {}
        p.device = "cpu"
        p.debug = False
        p.logger = logging.getLogger("test_transformers_admission")
        p._is_vision_model = lambda model: vision
        p._apply_transformers_experts_fallback = lambda kwargs: False
        p._prepare_bnb_mps_fused_kernel = lambda config: None
        p._warn_if_bnb_mps_fused_kernel_missing = lambda: None
        p._apply_loaded_generation_config_defaults = lambda: None
        hf_residency.register_provider(p)
        p._load_transformers_model()
        return p
    return load, calls


@pytest.mark.parametrize("vision", [False, True])
def test_transformers_second_provider_is_refused_before_from_pretrained(load_model, tmp_path, vision):
    load, calls = load_model
    first = load(tmp_path, vision=vision)
    with pytest.raises(InvalidRequestError, match="second copy is never loaded"):
        load(tmp_path, vision=vision)
    assert first.model_instance is not None and len(calls) == 1


def test_transformers_alias_is_refused_but_unloaded_or_other_artifact_can_load(load_model, tmp_path):
    load, calls = load_model
    path = tmp_path / "snapshot"
    path.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(path, target_is_directory=True)
    first = load(path)
    with pytest.raises(InvalidRequestError, match="already loaded"):
        load(alias)
    first.model_instance = None
    replacement = load(alias)
    other = load(tmp_path / "another-revision")
    assert len(calls) == 3 and replacement.model_instance is not other.model_instance


def test_transformers_concurrent_first_loads_allocate_once(load_model, tmp_path):
    load, calls = load_model
    start = threading.Barrier(2)
    results = []
    def run():
        start.wait()
        try:
            results.append(load(tmp_path))
        except InvalidRequestError as exc:
            results.append(exc)
    threads = [threading.Thread(target=run) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert len(calls) == 1 and sum(isinstance(r, InvalidRequestError) for r in results) == 1


def test_transformers_pipeline_still_holding_weights_prevents_a_reload(load_model, tmp_path):
    load, calls = load_model
    first = load(tmp_path)
    first.pipeline = types.SimpleNamespace(model=first.model_instance)
    first.model_instance = None
    with pytest.raises(InvalidRequestError, match="second copy is never loaded"):
        load(tmp_path)
    assert len(calls) == 1


@pytest.fixture()
def cross_backend_loaders(load_model, monkeypatch):
    from abstractcore.embeddings import manager as em
    from abstractcore.providers import mlx_native_session as native, mlx_provider as mp, mlx_qwen4 as qwen4

    hf_load, hf_calls = load_model
    monkeypatch.setattr(em, "_SHARED_EMBEDDING_MODELS", weakref.WeakValueDictionary())
    monkeypatch.setattr(native, "_SESSIONS", weakref.WeakValueDictionary())
    monkeypatch.setattr(mp, "_SHARED_MLX_MODELS", weakref.WeakValueDictionary())
    monkeypatch.setattr(qwen4, "_SESSIONS", weakref.WeakValueDictionary())
    mlx_calls, embedding_calls = [], []
    vlm = types.ModuleType("mlx_vlm")
    def mlx_allocate(path):
        mlx_calls.append(path)
        return object(), object()
    vlm.load = mlx_allocate
    monkeypatch.setitem(sys.modules, "mlx_vlm", vlm)
    drafters = types.ModuleType("mlx_vlm.speculative.drafters")
    drafters.load_drafter = lambda path: (types.SimpleNamespace(), "mtp")
    monkeypatch.setitem(sys.modules, "mlx_vlm.speculative.drafters", drafters)
    monkeypatch.setattr(native, "resolve_native_drafter_path", lambda path: str(Path(path).resolve()))
    monkeypatch.setattr("abstractcore.providers.weights_fingerprint.weights_fingerprint_for_dir", lambda path: "revision")
    def qwen4_allocate(path, *, mtp, ple_offload):
        session = qwen4.Qwen4Session()
        session.model, session.processor = mlx_allocate(path)
        return session
    monkeypatch.setattr(qwen4, "_create_qwen4_session", qwen4_allocate)
    def plain(path):
        provider = mp.MLXProvider.__new__(mp.MLXProvider)
        provider.logger = logging.getLogger("test_cross_backend")
        provider._prompt_cache_store = object()
        provider._load_or_adopt_shared_model(str(path), lambda: mlx_allocate(path))
        return provider
    def embedding(path):
        manager = em.EmbeddingManager.__new__(em.EmbeddingManager)
        manager.model_id = str(path)
        manager.backend = None
        manager.device_preference = None
        manager.trust_remote_code = False
        manager.model = None
        def allocate():
            embedding_calls.append(path)
            manager.model = types.SimpleNamespace(model_name_or_path=str(path))
        manager._load_model_unshared = allocate
        manager._load_model()
        return manager
    return {
        "hf": hf_load, "embedding": embedding, "mlx": plain,
        "native": lambda path: native.load_native_session(str(path)),
        "nativehead": lambda path: native.load_native_session(
            str(Path(path).parent / ("target-for-" + Path(path).resolve().name)), str(path)),
        "qwen4": lambda path: qwen4.load_qwen4_session(str(path), mtp=False),
    }, lambda: len(hf_calls) + len(mlx_calls) + len(embedding_calls)


_CROSS_PAIRS = [("hf", "embedding"), ("mlx", "embedding"), ("native", "embedding"),
                ("hf", "mlx"), ("hf", "native"), ("hf", "nativehead"), ("embedding", "nativehead")]
_CROSS_PAIRS += [("qwen4", lane) for lane in ("hf", "embedding", "mlx", "native", "nativehead")]


@pytest.mark.parametrize("pair", _CROSS_PAIRS)
@pytest.mark.parametrize("reverse", [False, True])
def test_cross_backend_same_snapshot_refuses_before_second_allocator(
    cross_backend_loaders, tmp_path, pair, reverse
):
    loaders, allocated = cross_backend_loaders
    path = tmp_path / "snapshot"
    path.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(path, target_is_directory=True)
    a, b = pair[::-1] if reverse else pair
    held = loaders[a](path)
    with pytest.raises(InvalidRequestError, match="second copy is never loaded"):
        loaders[b](alias)
    assert allocated() == 1
    other = loaders[b](tmp_path / "different-revision")
    assert other is not None and allocated() == 2
    if a == "hf":
        held.model_instance = None
    elif a == "embedding":
        held._release_shared_model()
        held.model = None
    else:
        del held
        import gc
        gc.collect()
    replacement = loaders[b](alias)
    assert replacement is not None and allocated() == 3


@pytest.mark.parametrize("pair", _CROSS_PAIRS)
def test_cross_backend_concurrent_first_loads_allocate_once(cross_backend_loaders, tmp_path, pair):
    loaders, allocated = cross_backend_loaders
    gate = threading.Barrier(2)
    results = []
    def load(backend):
        gate.wait()
        try:
            results.append(loaders[backend](tmp_path))
        except InvalidRequestError as exc:
            results.append(exc)
    threads = [threading.Thread(target=load, args=(backend,)) for backend in pair]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert allocated() == 1 and sum(isinstance(r, InvalidRequestError) for r in results) == 1


def test_qwen4_same_profile_reuses_and_changed_ple_profile_refuses(cross_backend_loaders, tmp_path):
    from abstractcore.providers import mlx_qwen4 as qwen4
    loaders, allocated = cross_backend_loaders
    first = loaders["qwen4"](tmp_path)
    assert loaders["qwen4"](tmp_path) is first
    with pytest.raises(InvalidRequestError, match="second copy is never loaded"):
        qwen4.load_qwen4_session(str(tmp_path), mtp=False, ple_offload=False)
    assert allocated() == 1
    del first
    import gc
    gc.collect()
    replacement = qwen4.load_qwen4_session(str(tmp_path), mtp=False, ple_offload=False)
    assert replacement is not None and allocated() == 2


@pytest.mark.parametrize("same_profile", [False, True])
def test_qwen4_concurrent_first_profiles_allocate_once(cross_backend_loaders, tmp_path, same_profile):
    from abstractcore.providers import mlx_qwen4 as qwen4
    _loaders, allocated = cross_backend_loaders
    gate = threading.Barrier(2)
    results = []
    def load(ple_offload):
        gate.wait()
        try:
            results.append(qwen4.load_qwen4_session(str(tmp_path), mtp=False, ple_offload=ple_offload))
        except InvalidRequestError as exc:
            results.append(exc)
    threads = [threading.Thread(target=load, args=(True,)), threading.Thread(target=load, args=(same_profile,))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert allocated() == 1
    assert sum(isinstance(r, InvalidRequestError) for r in results) == (0 if same_profile else 1)
