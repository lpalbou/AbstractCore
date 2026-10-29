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
