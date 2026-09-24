"""`abstractcore --download-vision-model` honours `cache.local_models_cache_dir`
(mission EE, 2026-09-24): it wrote to ~/.abstractcore/models whatever the setting."""

from __future__ import annotations

from pathlib import Path

import pytest


class _Saved:
    def __init__(self, name):
        self.name = name

    def save_pretrained(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "saved.txt").write_text(self.name)


def test_download_vision_model_writes_under_the_configured_dir(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    import abstractcore.config.manager as manager_module
    from abstractcore.config import main as config_main
    from abstractcore.config.manager import ConfigurationManager

    cfg = ConfigurationManager(config_file=tmp_path / "abstractcore.json", apply_env=False)
    target = tmp_path / "big-disk" / "models"
    assert cfg.set_local_models_cache_dir(str(target))
    monkeypatch.setattr(manager_module, "_config_manager", cfg)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)

    # transformers is a lazy module that rebinds its public names on import, so
    # patch the REAL classes' loaders; nothing is downloaded.
    calls = []

    def fake_from_pretrained(cls, hf_id, **kwargs):
        calls.append((cls.__name__, hf_id, kwargs))
        return _Saved(f"{cls.__name__}:{hf_id}")

    for cls in (transformers.GitProcessor, transformers.GitForCausalLM):
        monkeypatch.setattr(cls, "from_pretrained", classmethod(fake_from_pretrained))

    assert config_main.local_models_cache_dir() == target
    assert config_main.download_vision_model("git-base") is True

    assert (target / "git-base" / "download_complete.txt").is_file()
    assert (target / "git-base" / "model" / "saved.txt").is_file()
    assert {kw["cache_dir"] for _n, _id, kw in calls} == {str(target / "git-base")}
    assert len(calls) == 2
    assert not (Path.home() / ".abstractcore" / "models").exists()
