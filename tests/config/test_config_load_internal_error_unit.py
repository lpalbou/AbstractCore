"""The settings file is read as it is, or the load stops with the path and the error (2026-11-09).

No copy is ever made of it, no defaults ever stand in for it: the next save would publish those
defaults over the operator's settings. A failure of the loader's own code propagates as itself.
"""

import json
import os
import subprocess
import sys
import types

import pytest

from abstractcore.config.manager import ConfigParseError, ConfigurationManager


def _write_config(tmp_path):
    path = tmp_path / "abstractcore.json"
    path.write_text(json.dumps({
        "provider_profiles": {"profiles": {"lab": {
            "id": "lab", "provider_family": "openai-compatible", "base_url": "http://192.168.1.7:1234/v1",
        }}},
    }))
    return path


def test_an_unparseable_file_stops_the_load_with_its_path_and_is_left_alone(tmp_path):
    path = tmp_path / "abstractcore.json"
    path.write_text("{ not json")
    before = path.read_bytes()
    with pytest.raises(ConfigParseError) as info:
        ConfigurationManager(config_dir=tmp_path, config_file=path, apply_env=False)
    assert str(path) in str(info.value) and "not valid JSON" in str(info.value)
    assert isinstance(info.value.__cause__, json.JSONDecodeError)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["abstractcore.json"]


@pytest.mark.parametrize("text", ["[]", "null", '"x"'])
def test_a_decoded_non_object_top_level_stops_the_load_too(tmp_path, text):
    path = tmp_path / "abstractcore.json"
    path.write_text(text)
    with pytest.raises(ConfigParseError, match="must be a JSON object"):
        ConfigurationManager(config_dir=tmp_path, config_file=path, apply_env=False)
    assert path.read_text() == text
    assert sorted(p.name for p in tmp_path.iterdir()) == ["abstractcore.json"]


def test_a_loader_failure_propagates_as_itself_and_leaves_the_file_alone(tmp_path, monkeypatch):
    path = _write_config(tmp_path)
    before = path.read_bytes()

    def boom(self, data):
        raise ImportError("cannot import name 'get_provider_registry' from partially initialized module")

    monkeypatch.setattr(ConfigurationManager, "_dict_to_config", boom)
    with pytest.raises(ImportError):
        ConfigurationManager(config_dir=tmp_path, config_file=path, apply_env=False)
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["abstractcore.json"]


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


def test_a_registry_that_truly_fails_to_import_is_not_silenced(monkeypatch):
    from abstractcore.config import provider_profiles

    monkeypatch.setitem(sys.modules, "abstractcore.providers.registry", None)  # import now raises
    with pytest.raises(ImportError):
        provider_profiles.supported_provider_families()


def test_the_server_refuses_to_start_on_an_unparseable_file(tmp_path):
    """`import abstractcore.server.app` loads the settings for its auth middleware; on a file that
    cannot be parsed it must stop with the error, never come up with no settings and no message."""
    home = tmp_path / "home"
    (home / ".abstractcore" / "config").mkdir(parents=True)
    bad = home / ".abstractcore" / "config" / "abstractcore.json"
    bad.write_text("{ not json")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ABSTRACTCORE_")}
    env["HOME"] = str(home)
    env["PYTHONWARNINGS"] = "ignore"
    out = subprocess.run([sys.executable, "-c", "import abstractcore.server.app"], env=env,
                         capture_output=True, text=True, timeout=240)
    assert out.returncode != 0
    assert "ConfigParseError" in out.stderr and str(bad) in out.stderr, out.stderr[-2000:]
    assert bad.read_text() == "{ not json"
    assert sorted(p.name for p in bad.parent.iterdir()) == ["abstractcore.json"]


def test_plain_import_then_config_load_in_a_fresh_process_keeps_the_file(tmp_path):
    """End to end: a fresh interpreter, `import abstractcore`, then the config, with a profile-bearing
    file under a scratch HOME. The profile loads and the config directory holds only the file."""
    home = tmp_path / "home"
    (home / ".abstractcore" / "config").mkdir(parents=True)
    _write_config(home / ".abstractcore" / "config")
    env = {k: v for k, v in os.environ.items() if not k.startswith("ABSTRACTCORE_")}
    env["HOME"] = str(home)
    env["PYTHONWARNINGS"] = "ignore"
    code = (
        "import abstractcore\n"
        "from abstractcore.config.manager import get_config_manager\n"
        "m = get_config_manager()\n"
        "print(sorted(m.config.provider_profiles.profiles))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "['lab']" in out.stdout, out.stdout
    assert sorted(p.name for p in (home / ".abstractcore" / "config").iterdir()) == ["abstractcore.json"]
