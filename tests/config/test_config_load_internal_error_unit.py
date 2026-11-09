"""A config load that fails for the LOADER's reasons never touches the file (2026-11-09).

187 byte-identical `abstractcore.json.corrupt-*.bak` copies of a valid config accumulated on the
operator's machine: a logger built during `import abstractcore` re-entered get_config_manager(),
the inner load imported the half-initialized provider registry while validating provider_profiles,
and the loader treated that ImportError as a corrupt file (quarantine + defaults for the session).
"""

import json
import sys
import threading
import types

import pytest

from abstractcore.config import manager as manager_module
from abstractcore.config.manager import ConfigurationManager


def _write_config(tmp_path):
    path = tmp_path / "abstractcore.json"
    path.write_text(json.dumps({
        "provider_profiles": {"profiles": {"lab": {
            "id": "lab", "provider_family": "openai-compatible", "base_url": "http://192.168.1.7:1234/v1",
        }}},
    }))
    return path


def test_internal_error_during_load_leaves_the_file_alone_and_raises(tmp_path, monkeypatch):
    path = _write_config(tmp_path)
    before = path.read_bytes()

    def boom(self, data):
        raise ImportError("cannot import name 'get_provider_registry' from partially initialized module")

    monkeypatch.setattr(ConfigurationManager, "_dict_to_config", boom)
    with pytest.raises(RuntimeError, match="internal error") as info:
        ConfigurationManager(config_dir=tmp_path, config_file=path, apply_env=False)
    assert isinstance(info.value.__cause__, ImportError)
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.corrupt-*")) == [], "a loader failure is not a corrupt file"


def test_schema_error_from_our_own_code_does_not_quarantine_either(tmp_path, monkeypatch):
    """A ValueError inside _dict_to_config is as likely our bug as the file's fault: no copy, no defaults."""
    path = _write_config(tmp_path)
    before = path.read_bytes()
    monkeypatch.setattr(ConfigurationManager, "_dict_to_config", lambda self, data: (_ for _ in ()).throw(ValueError("x")))
    with pytest.raises(RuntimeError, match="internal error"):
        ConfigurationManager(config_dir=tmp_path, config_file=path, apply_env=False)
    assert path.read_bytes() == before
    assert list(tmp_path.glob("*.corrupt-*")) == []


def test_unparseable_file_is_still_quarantined(tmp_path):
    path = tmp_path / "abstractcore.json"
    path.write_text("{ not json")
    mgr = ConfigurationManager(config_dir=tmp_path, config_file=path, apply_env=False)
    assert mgr.config is not None
    assert len(list(tmp_path.glob("abstractcore.json.corrupt-*.bak"))) == 1


def test_provider_family_accepted_while_the_registry_is_half_imported(monkeypatch):
    from abstractcore.config import provider_profiles

    partial = types.ModuleType("abstractcore.providers.registry")  # mid-import: no attributes yet
    monkeypatch.setitem(sys.modules, "abstractcore.providers.registry", partial)
    assert provider_profiles.supported_provider_families() is None
    assert provider_profiles.normalize_provider_family("openai-compatible") == "openai-compatible"
    assert provider_profiles.normalize_provider_family("OpenAI") == "openai"


def test_provider_family_still_validated_when_the_registry_is_importable():
    from abstractcore.config import provider_profiles

    assert "openai-compatible" in provider_profiles.supported_provider_families()
    with pytest.raises(ValueError, match="Unsupported provider family"):
        provider_profiles.normalize_provider_family("no-such-family")


def test_reentrant_get_config_manager_raises_inside_construction_and_the_outer_one_wins(tmp_path, monkeypatch):
    monkeypatch.setattr(manager_module, "_config_manager", None)
    monkeypatch.setattr(manager_module, "_constructing_thread", None)
    seen = {}
    original = ConfigurationManager._load_config

    def reenter(self):
        try:
            manager_module.get_config_manager()
        except RuntimeError as exc:
            seen["inner"] = str(exc)
        return original(self)

    monkeypatch.setattr(ConfigurationManager, "_load_config", reenter)
    monkeypatch.setenv("HOME", str(tmp_path))
    outer = manager_module.get_config_manager()
    assert "re-entered" in seen["inner"]
    assert manager_module.get_config_manager() is outer


def test_another_thread_during_construction_gets_the_same_instance(tmp_path, monkeypatch):
    monkeypatch.setattr(manager_module, "_config_manager", None)
    monkeypatch.setattr(manager_module, "_constructing_thread", None)
    monkeypatch.setenv("HOME", str(tmp_path))
    started = threading.Event()
    release = threading.Event()
    original = ConfigurationManager._load_config

    def slow(self):
        started.set()
        release.wait(5)
        return original(self)

    monkeypatch.setattr(ConfigurationManager, "_load_config", slow)
    results = {}
    t1 = threading.Thread(target=lambda: results.setdefault("a", manager_module.get_config_manager()))
    t1.start()
    assert started.wait(5)
    t2 = threading.Thread(target=lambda: results.setdefault("b", manager_module.get_config_manager()))
    t2.start()
    release.set()
    t1.join(10); t2.join(10)
    assert results["a"] is results["b"]
