"""Tests for shared capability routing defaults."""

from __future__ import annotations

from abstractcore.config.manager import ConfigurationManager


def test_capability_defaults_persist_provider_model_base_url_and_options(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = ConfigurationManager()
    assert manager.set_capability_default(
        "output.voice",
        provider="supertonic",
        model="supertonic-3",
        base_url="http://127.0.0.1:5000/v1",
        options={"voice": "M1"},
    )

    reloaded = ConfigurationManager()
    route = reloaded.get_capability_default("output", "voice")

    assert route["provider"] == "supertonic"
    assert route["model"] == "supertonic-3"
    assert route["base_url"] == "http://127.0.0.1:5000/v1"
    assert route["options"] == {"voice": "M1"}
    assert route["source"] == "abstractcore.capability_defaults"


def test_global_default_writes_explicit_text_capability_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = ConfigurationManager()
    assert manager.set_global_default_model("lmstudio:qwen/qwen3.6-35b-a3b")

    out_text = manager.get_capability_default("output", "text")
    in_text = manager.get_capability_default("input", "text")

    assert out_text["provider"] == "lmstudio"
    assert out_text["model"] == "qwen/qwen3.6-35b-a3b"
    assert out_text["source"] == "abstractcore.capability_defaults"
    assert in_text["provider"] == "lmstudio"
    assert in_text["model"] == "qwen/qwen3.6-35b-a3b"
    assert in_text["source"] == "abstractcore.capability_defaults"


def test_embeddings_defaults_sync_to_embedding_text_route(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = ConfigurationManager()
    assert manager.set_embeddings_provider("lmstudio")
    assert manager.set_embeddings_model("lmstudio/text-embedding-nomic-embed-text-v1.5")
    assert manager.set_embeddings_base_url("http://127.0.0.1:1234/v1")

    route = ConfigurationManager().get_capability_default("embedding", "text")

    assert route["provider"] == "lmstudio"
    assert route["model"] == "text-embedding-nomic-embed-text-v1.5"
    assert route["base_url"] == "http://127.0.0.1:1234/v1"
    assert route["source"] == "abstractcore.capability_defaults"


def test_status_reports_effective_embedding_text_route(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = ConfigurationManager()
    assert manager.set_capability_default(
        "embedding.text",
        provider="lmstudio",
        model="text-embedding-route",
        base_url="http://127.0.0.1:1234/v1",
    )

    embeddings = ConfigurationManager().get_status()["embeddings"]

    assert embeddings["route"] == "embedding.text"
    assert embeddings["source"] == "abstractcore.capability_defaults"
    assert embeddings["provider"] == "lmstudio"
    assert embeddings["model"] == "text-embedding-route"
    assert embeddings["base_url"] == "http://127.0.0.1:1234/v1"
    assert embeddings["legacy"]["provider"] != embeddings["provider"]


def test_capability_defaults_include_embedding_and_rerank_route_specs(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    rows = ConfigurationManager().list_capability_defaults()
    by_key = {row["key"]: row for row in rows}

    assert by_key["embedding.text"]["kind"] == "embedding"
    assert by_key["embedding.image"]["kind"] == "embedding"
    assert by_key["rerank.text"]["kind"] == "rerank"
    assert by_key["rerank.text"]["configured"] is False


def test_embedding_manager_uses_route_base_url_only_for_matching_route(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    import abstractcore.config.manager as config_manager_module
    from abstractcore.embeddings.manager import EmbeddingManager
    from abstractcore.providers.lmstudio_provider import LMStudioProvider

    monkeypatch.setattr(config_manager_module, "_config_manager", None)
    manager = ConfigurationManager()
    assert manager.set_capability_default(
        "embedding.text",
        provider="lmstudio",
        model="text-embedding-route",
        base_url="http://127.0.0.1:1234/v1",
    )
    monkeypatch.setattr(config_manager_module, "_config_manager", None)

    captured: list[dict[str, object]] = []

    def fake_lmstudio_init(self, model: str = "local-model", base_url: str | None = None, **kwargs) -> None:
        captured.append({"model": model, "base_url": base_url, "kwargs": kwargs})
        self.model = model
        self.base_url = base_url

    monkeypatch.setattr(LMStudioProvider, "__init__", fake_lmstudio_init)

    EmbeddingManager(provider="lmstudio", model="text-embedding-route", cache_dir=tmp_path / "cache-a")
    assert captured[-1]["model"] == "text-embedding-route"
    assert captured[-1]["base_url"] == "http://127.0.0.1:1234/v1"

    EmbeddingManager(provider="lmstudio", model="different-embedding", cache_dir=tmp_path / "cache-b")
    assert captured[-1]["model"] == "different-embedding"
    assert captured[-1]["base_url"] is None
