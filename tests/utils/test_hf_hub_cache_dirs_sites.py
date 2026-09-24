"""Every model-cache scan reads the SAME hub caches (mission EE, 2026-09-24).

Five sites looked only in `~/.cache/huggingface/hub` and ignored HF_HUB_CACHE /
HF_HOME / `cache.huggingface_cache_dir`. Each test below points HF_HUB_CACHE at
a scratch directory OUTSIDE the home and proves the site finds a model there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from abstractcore.utils.model_cache import hf_hub_cache_dirs


def _repo(hub: Path, repo_id: str, files: dict) -> Path:
    model_dir = hub / ("models--" + repo_id.replace("/", "--"))
    snap = model_dir / "snapshots" / ("a" * 40)
    for name, data in files.items():
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).write_bytes(data)
    (model_dir / "refs").mkdir(parents=True, exist_ok=True)
    (model_dir / "refs" / "main").write_text("a" * 40)
    return snap


@pytest.fixture
def relocated_hub(tmp_path, monkeypatch):
    hub = tmp_path / "elsewhere" / "hub"
    hub.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(hub))
    assert not str(hub).startswith(str(Path.home()))
    return hub


def test_helper_lists_hf_hub_cache(relocated_hub):
    assert relocated_hub in hf_hub_cache_dirs()


def test_helper_adds_the_configured_cache_dir(tmp_path, monkeypatch):
    from abstractcore.config import get_config_manager

    configured = tmp_path / "configured-hf"
    (configured / "hub").mkdir(parents=True)
    monkeypatch.setattr(get_config_manager().config.cache, "huggingface_cache_dir", str(configured))
    assert configured / "hub" in hf_hub_cache_dirs()


def test_hf_provider_gguf_lookup(relocated_hub):
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    snap = _repo(relocated_hub, "acme/Tiny-GGUF", {"tiny-Q4_K_M.gguf": b"GGUF"})
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    assert provider._find_gguf_in_cache("acme/Tiny-GGUF") == str(snap / "tiny-Q4_K_M.gguf")


def test_hf_provider_similar_gguf_models(relocated_hub):
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    _repo(relocated_hub, "acme/Other-GGUF", {"x-Q4_K_M.gguf": b"GGUF"})
    provider = HuggingFaceProvider.__new__(HuggingFaceProvider)
    assert "acme/Other-GGUF" in provider._find_similar_gguf_models()


def test_hf_provider_list_available_models(relocated_hub):
    from abstractcore.providers.huggingface_provider import HuggingFaceProvider

    _repo(relocated_hub, "acme/tiny-bert", {"config.json": b"{}", "model.safetensors": b"x"})
    assert "acme/tiny-bert" in HuggingFaceProvider.list_available_models()


def test_mlx_provider_list_available_models(relocated_hub):
    from abstractcore.providers.mlx_provider import MLXProvider

    _repo(relocated_hub, "mlx-community/Tiny-4bit", {"config.json": b"{}", "model.safetensors": b"x"})
    assert "mlx-community/Tiny-4bit" in MLXProvider.list_available_models()


def test_model_materializer_fallback(relocated_hub, monkeypatch):
    import abstractcore.capabilities.vision_catalog as vision_catalog
    from abstractcore.config import model_materializer as mm

    def unavailable():
        raise ImportError("vision catalog unavailable")

    monkeypatch.setattr(vision_catalog, "_default_hf_hub_cache_dirs", unavailable)
    assert relocated_hub in mm._hf_cache_dirs()


def test_embeddings_onnx_probe(relocated_hub):
    from abstractcore.embeddings.manager import EmbeddingManager

    _repo(relocated_hub, "acme/embedder", {"config.json": b"{}", "onnx/model.onnx": b"x"})
    mgr = EmbeddingManager.__new__(EmbeddingManager)
    mgr.model_id = "acme/embedder"
    assert mgr._has_preexported_onnx() is True
