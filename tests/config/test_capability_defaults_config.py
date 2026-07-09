"""Tests for shared capability routing defaults."""

from __future__ import annotations

import json
import os
import stat

import pytest

from abstractcore.config.main import main as config_main
from abstractcore.config.manager import ConfigurationManager


def test_capability_defaults_persist_provider_model_base_url_and_options(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = ConfigurationManager()
    assert manager.set_capability_default(
        "output.voice",
        provider="supertonic",
        model="supertonic-3",
        base_url="http://127.0.0.1:5000/v1",
        reasoning="low",
        options={"voice": "M1"},
    )

    reloaded = ConfigurationManager()
    route = reloaded.get_capability_default("output", "voice")

    assert route["provider"] == "supertonic"
    assert route["model"] == "supertonic-3"
    assert route["base_url"] == "http://127.0.0.1:5000/v1"
    assert route["reasoning"] == "low"
    assert route["options"] == {"voice": "M1"}
    assert route["source"] == "abstractcore.capability_defaults"


def test_task_specific_generative_capability_defaults_are_distinct(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = ConfigurationManager()
    assert manager.set_capability_default("output.image.text_to_image", provider="mlx-gen", model="AbstractFramework/z-image-turbo-8bit")
    assert manager.set_capability_default("output.image.image_to_image", provider="mlx-gen", model="AbstractFramework/qwen-image-edit-2511-4bit")
    assert manager.set_capability_default(
        "output.image.image_upscale",
        provider="mlx-gen",
        model="AbstractFramework/seedvr2-3b-8bit",
        options={"resolution": "2x", "softness": 0.25},
    )
    assert manager.set_capability_default("output.video.text_to_video", provider="mlx-gen", model="AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit")
    assert manager.set_capability_default("output.video.image_to_video", provider="mlx-gen", model="AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit")

    reloaded = ConfigurationManager()
    t2i = reloaded.get_capability_default("output", "image", "text_to_image")
    i2i = reloaded.get_capability_default("output", "image", "image_to_image")
    upscale = reloaded.get_capability_default("output", "image", "image_upscale")
    t2v = reloaded.get_capability_default("output", "video", "text_to_video")
    i2v = reloaded.get_capability_default("output", "video", "image_to_video")
    broad_image = reloaded.get_capability_default("output", "image")

    assert t2i["key"] == "output.image.text_to_image"
    assert t2i["model"] == "AbstractFramework/z-image-turbo-8bit"
    assert i2i["key"] == "output.image.image_to_image"
    assert i2i["model"] == "AbstractFramework/qwen-image-edit-2511-4bit"
    assert upscale["key"] == "output.image.image_upscale"
    assert upscale["model"] == "AbstractFramework/seedvr2-3b-8bit"
    assert upscale["options"] == {"resolution": "2x", "softness": 0.25}
    assert t2v["key"] == "output.video.text_to_video"
    assert i2v["key"] == "output.video.image_to_video"
    assert broad_image["key"] == "output.image"
    assert broad_image["source"] == "not_configured"

    rows = {row["key"]: row for row in reloaded.list_capability_defaults()}
    assert rows["output.image.text_to_image"]["model"] == "AbstractFramework/z-image-turbo-8bit"
    assert rows["output.image.image_to_image"]["label"] == "Image Edit"
    assert rows["output.image.image_upscale"]["options"] == {"resolution": "2x", "softness": 0.25}
    assert rows["output.video.text_to_video"]["model"] == "AbstractFramework/wan2.2-t2v-a14b-diffusers-8bit"
    assert rows["output.video.image_to_video"]["model"] == "AbstractFramework/wan2.2-i2v-a14b-diffusers-8bit"


def test_configuration_manager_supports_scoped_config_files_without_env_mutation(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)

    alice_file = tmp_path / "alice" / "abstractcore.json"
    bob_file = tmp_path / "bob" / "abstractcore.json"
    alice = ConfigurationManager(config_file=alice_file, apply_env=False)
    bob = ConfigurationManager(config_file=bob_file, apply_env=False)

    alice.config.api_keys.openai = "alice-key"
    alice.config.server.auth_token = "alice-server-token"
    assert alice.set_capability_default("output.text", provider="openai", model="gpt-alice")
    assert bob.set_capability_default("output.text", provider="anthropic", model="claude-bob")

    assert os.environ.get("OPENAI_API_KEY") is None
    assert os.environ.get("ABSTRACTCORE_AUTH_TOKEN") is None
    assert ConfigurationManager(config_file=alice_file, apply_env=False).get_capability_default("output", "text")["model"] == "gpt-alice"
    assert ConfigurationManager(config_file=bob_file, apply_env=False).get_capability_default("output", "text")["model"] == "claude-bob"
    assert stat.S_IMODE(alice_file.stat().st_mode) & stat.S_IRWXG == 0
    assert stat.S_IMODE(alice_file.stat().st_mode) & stat.S_IRWXO == 0


def test_configuration_manager_honors_config_file_env(monkeypatch, tmp_path) -> None:
    config_file = tmp_path / "custom" / "abstractcore.json"
    monkeypatch.setenv("ABSTRACTCORE_CONFIG_FILE", str(config_file))

    manager = ConfigurationManager(apply_env=False)
    assert manager.config_file == config_file
    assert manager.set_capability_default("embedding.text", provider="lmstudio", model="embed")
    data = json.loads(config_file.read_text(encoding="utf-8"))
    assert data["capability_defaults"]["routes"]["embedding.text"]["model"] == "embed"


def test_config_subcommand_set_list_and_clear_defaults(monkeypatch, tmp_path, capsys) -> None:
    config_file = tmp_path / "core" / "abstractcore.json"

    assert config_main(
        [
            "config",
            "--config-file",
            str(config_file),
            "set-default",
            "output.text",
            "--provider",
            "lmstudio",
            "--model",
            "qwen-local",
            "--base-url",
            "http://127.0.0.1:1234/v1",
            "--reasoning",
            "medium",
            "--option",
            "temperature=0.2",
        ]
    ) == 0
    assert "Set capability default for output.text" in capsys.readouterr().out

    assert config_main(["config", "--config-file", str(config_file), "defaults", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    route = next(row for row in payload["routes"] if row["key"] == "output.text")
    assert route["provider"] == "lmstudio"
    assert route["model"] == "qwen-local"
    assert route["base_url"] == "http://127.0.0.1:1234/v1"
    assert route["reasoning"] == "medium"
    assert route["options"] == {"temperature": 0.2}

    assert config_main(["config", "--config-file", str(config_file), "clear-default", "output.text"]) == 0
    assert "Cleared capability default for output.text" in capsys.readouterr().out


def test_top_level_capability_default_flags_are_not_supported() -> None:
    with pytest.raises(SystemExit) as exc_info:
        config_main(
            [
                "--set-capability-default",
                "output.voice",
                "--capability-provider",
                "supertonic",
                "--capability-model",
                "supertonic-3",
            ]
        )
    assert exc_info.value.code == 2


def test_global_default_writes_explicit_text_capability_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = ConfigurationManager()
    assert manager.set_global_default_model("lmstudio:qwen/qwen3.6-35b-a3b")

    out_text = manager.get_capability_default("output", "text")
    in_text = manager.get_capability_default("input", "text")

    assert out_text["provider"] == "lmstudio"
    assert out_text["model"] == "qwen/qwen3.6-35b-a3b"
    assert out_text["source"] == "abstractcore.capability_defaults"
    assert out_text["derived_from"] == "input.text"
    assert out_text["read_only"] is True
    assert in_text["provider"] == "lmstudio"
    assert in_text["model"] == "qwen/qwen3.6-35b-a3b"
    assert in_text["source"] == "abstractcore.capability_defaults"
    assert "output.text" not in manager.config.capability_defaults.routes


def test_output_text_is_canonicalized_to_input_text(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = ConfigurationManager()
    assert manager.set_capability_default("output.text", provider="lmstudio", model="qwen-local")

    assert "input.text" in manager.config.capability_defaults.routes
    assert "output.text" not in manager.config.capability_defaults.routes

    rows = {row["key"]: row for row in ConfigurationManager().list_capability_defaults()}
    assert rows["input.text"]["model"] == "qwen-local"
    assert rows["output.text"]["model"] == "qwen-local"
    assert rows["output.text"]["derived_from"] == "input.text"
    assert rows["output.text"]["read_only"] is True


def test_image_input_is_covered_by_vision_capable_text_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = ConfigurationManager()
    assert manager.set_capability_default("input.text", provider="lmstudio", model="qwen/qwen3.6-35b-a3b")
    assert manager.set_capability_default("input.image", provider="openai", model="gpt-4o")

    rows = {row["key"]: row for row in ConfigurationManager().list_capability_defaults()}

    assert rows["input.image"]["provider"] == "lmstudio"
    assert rows["input.image"]["model"] == "qwen/qwen3.6-35b-a3b"
    assert rows["input.image"]["covered_by"] == "input.text"
    assert rows["input.image"]["read_only"] is True

    image_default = ConfigurationManager().get_capability_default("input", "image")
    assert image_default["provider"] == "lmstudio"
    assert image_default["model"] == "qwen/qwen3.6-35b-a3b"
    assert image_default["covered_by"] == "input.text"
    assert image_default["read_only"] is True


def test_video_input_is_covered_by_text_default_but_overrideable(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = ConfigurationManager()
    assert manager.set_capability_default("input.text", provider="lmstudio", model="qwen/qwen3.6-35b-a3b")

    inherited = ConfigurationManager().get_capability_default("input", "video")
    assert inherited["provider"] == "lmstudio"
    assert inherited["model"] == "qwen/qwen3.6-35b-a3b"
    assert inherited["covered_by"] == "input.text"
    assert inherited["coverage_mode"] == "video_frames"
    assert inherited["overrideable"] is True
    assert inherited["read_only"] is False

    assert manager.set_capability_default("input.video", provider="openrouter", model="video-fallback")

    explicit = ConfigurationManager().get_capability_default("input", "video")
    assert explicit["provider"] == "openrouter"
    assert explicit["model"] == "video-fallback"
    assert explicit["overrideable"] is True
    assert "covered_by" not in explicit


def test_sound_and_music_input_are_covered_by_audio_capable_text_default_without_stt(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = ConfigurationManager()
    assert manager.set_capability_default("input.text", provider="lmstudio", model="qwen3-omni-30b-a3b-instruct")

    rows = {row["key"]: row for row in ConfigurationManager().list_capability_defaults()}

    for key in ("input.sound", "input.music"):
        row = rows[key]
        assert row["provider"] == "lmstudio"
        assert row["model"] == "qwen3-omni-30b-a3b-instruct"
        assert row["covered_by"] == "input.text"
        assert row["overrideable"] is True
        assert row["read_only"] is False

    assert rows["input.voice"]["source"] == "not_configured"
    assert rows["input.voice"]["configured"] is False
    assert "covered_by" not in rows["input.voice"]


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
    assert by_key["input.music"]["task"] == "music_understanding"
    assert by_key["output.sound"]["task"] == "sound_generation"
    assert by_key["output.music"]["task"] == "music_generation"
    assert by_key["output.scene3d"]["task"] == "scene3d_generation"
    assert by_key["output.scene3d.text_to_scene3d"]["task"] == "text_to_scene3d"
    assert by_key["output.scene3d.image_to_scene3d"]["task"] == "image_to_scene3d"
    assert by_key["rerank.text"]["kind"] == "rerank"
    assert by_key["rerank.text"]["configured"] is False


def test_capability_defaults_keep_music_distinct_from_sound(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))

    manager = ConfigurationManager()
    assert manager.set_capability_default("output.music", provider="acemusic", model="ace-step")
    assert manager.set_capability_default("output.sound", provider="stable-audio", model="stabilityai/stable-audio-open-small")

    music = ConfigurationManager().get_capability_default("output", "music")
    sound = ConfigurationManager().get_capability_default("output", "sound")

    assert music["key"] == "output.music"
    assert music["provider"] == "acemusic"
    assert sound["key"] == "output.sound"
    assert sound["provider"] == "stable-audio"


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
