"""A model that is not downloaded raises `ModelNotDownloadedError`, carrying `model` and `provider`.

Hosts (the gateway's boot sentence for an embedding default nobody fetched) read the attributes
instead of parsing the message. The message keeps its wording, the class stays a
`ModelNotFoundError`, and the attributes survive a pickle round trip (the gateway's model host
child returns exceptions by pickle)."""
from __future__ import annotations

import pickle
from types import SimpleNamespace

import pytest

from abstractcore.exceptions import ModelNotDownloadedError, ModelNotFoundError, ProviderError


def test_the_error_is_a_model_not_found_error_and_survives_pickle():
    err = ModelNotDownloadedError("text for people", model="acme/m", provider="huggingface")
    assert isinstance(err, ModelNotFoundError) and isinstance(err, ProviderError)
    assert str(err) == "text for people"
    back = pickle.loads(pickle.dumps(err))
    assert type(back) is ModelNotDownloadedError
    assert (str(back), back.model, back.provider) == ("text for people", "acme/m", "huggingface")


@pytest.fixture
def empty_cache(tmp_path, monkeypatch):
    from abstractcore.utils import model_cache

    hub = tmp_path / "hub"
    hub.mkdir()
    monkeypatch.setattr(model_cache, "hf_hub_cache_dirs", lambda: [hub])
    monkeypatch.setattr(model_cache, "resolve_hf_load_snapshot", lambda *a, **k: None)
    return hub


def test_the_embedding_path_raises_it_with_the_model_id(empty_cache, monkeypatch):
    import abstractcore.config as config
    from abstractcore.embeddings.manager import EmbeddingManager

    monkeypatch.setattr(config, "get_config_manager", lambda: SimpleNamespace(
        is_offline_first=lambda: True, should_force_local_files_only=lambda: True))
    owner = SimpleNamespace(model_id="acme/never-downloaded-embedder", provider="huggingface")
    with pytest.raises(ModelNotDownloadedError) as info:
        EmbeddingManager._sentence_transformers_source(owner)
    assert (info.value.model, info.value.provider) == ("acme/never-downloaded-embedder", "huggingface")
    assert str(info.value).startswith(
        "Embedding model 'acme/never-downloaded-embedder' is not in the local Hugging Face cache (looked in: ")


def test_the_transformers_generation_path_raises_it_with_the_model_id(empty_cache, monkeypatch):
    pytest.importorskip("transformers", reason="the HuggingFace provider module needs transformers")
    import abstractcore.providers.huggingface_provider as hfp

    monkeypatch.setattr(hfp, "_get_local_model_path", lambda name: None)
    monkeypatch.setattr(hfp, "_hf_load_cache_dirs", lambda: [empty_cache])
    with pytest.raises(ModelNotDownloadedError) as info:
        hfp._resolve_transformers_load_source("acme/never-downloaded", local_only=True)
    assert (info.value.model, info.value.provider) == ("acme/never-downloaded", "huggingface")
    assert str(info.value).startswith("HuggingFace model 'acme/never-downloaded' is not in the local Hugging Face cache")


def test_the_gguf_generation_path_raises_it_with_the_model_id(monkeypatch):
    pytest.importorskip("transformers", reason="the HuggingFace provider module needs transformers")
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    owner = SimpleNamespace(model="acme/never-downloaded-GGUF",
                            _suggest_correct_repo_format=lambda name: name,
                            _find_similar_gguf_models=lambda: [])
    with pytest.raises(ModelNotDownloadedError) as info:
        HuggingFaceProvider._handle_gguf_not_found(owner)
    assert (info.value.model, info.value.provider) == ("acme/never-downloaded-GGUF", "huggingface")
    assert str(info.value).startswith("❌ GGUF model 'acme/never-downloaded-GGUF' not found in local caches")


def test_the_mlx_drafter_path_raises_it_with_the_drafter_id(monkeypatch):
    import abstractcore.config.manager as cfg_manager
    from abstractcore.providers import mlx_native_session
    from abstractcore.utils import model_cache

    monkeypatch.setattr(model_cache, "resolve_hf_snapshot_dir", lambda *a, **k: None)
    monkeypatch.setattr(cfg_manager, "get_config_manager", lambda: SimpleNamespace(is_offline_first=lambda: True))
    with pytest.raises(ModelNotDownloadedError) as info:
        mlx_native_session.resolve_native_drafter_path("acme/never-downloaded-drafter")
    assert (info.value.model, info.value.provider) == ("acme/never-downloaded-drafter", "mlx")
