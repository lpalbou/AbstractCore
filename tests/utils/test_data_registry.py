"""Data-home registry: register-at-first-write, enumeration, safe-purge verbs.

The ruled shape (cache-ownership vote v-gtytw8 SPLIT + operator sign-off
2026-07-13): core owns the machine-level registry primitive; the gateway
console consumes it; safe_to_purge is OWNER-declared and enforced here.
"""

import json
import os
from pathlib import Path

import pytest

from abstractcore.utils.data_registry import (
    DATA_HOME_KINDS,
    DataRegistryError,
    data_home_size,
    get_data_home,
    list_data_homes,
    purge_data_home,
    register_data_home,
    registry_path,
    unregister_data_home,
)


@pytest.fixture()
def registry_env(tmp_path, monkeypatch):
    reg = tmp_path / "registry" / "data_registry.json"
    monkeypatch.setenv("ABSTRACTFRAMEWORK_DATA_REGISTRY", str(reg))
    return reg


def _make_home(tmp_path: Path, name: str = "home") -> Path:
    p = tmp_path / name
    p.mkdir(parents=True, exist_ok=True)
    return p


class TestRegistration:
    def test_register_creates_row_and_file(self, registry_env, tmp_path):
        home = _make_home(tmp_path)
        row = register_data_home(
            "test-home", path=str(home), kind="prompt-cache",
            owner="abstractcore", safe_to_purge=True, description="test",
        )
        assert row.name == "test-home"
        assert row.path == str(home.resolve())
        assert registry_env.exists()
        on_disk = json.loads(registry_env.read_text())
        assert on_disk["homes"]["test-home"]["kind"] == "prompt-cache"

    def test_register_is_idempotent_upsert_preserving_registered_at(self, registry_env, tmp_path):
        home = _make_home(tmp_path)
        first = register_data_home("h", path=str(home), kind="logs", owner="a", safe_to_purge=True)
        second = register_data_home("h", path=str(home), kind="logs", owner="a", safe_to_purge=False)
        assert second.registered_at == first.registered_at
        assert get_data_home("h").safe_to_purge is False

    def test_unknown_kind_refused_naming_the_ruled_set(self, registry_env, tmp_path):
        home = _make_home(tmp_path)
        with pytest.raises(DataRegistryError) as e:
            register_data_home("h", path=str(home), kind="scratch", owner="a", safe_to_purge=True)
        for kind in DATA_HOME_KINDS:
            assert kind in str(e.value)

    def test_empty_name_and_owner_refused(self, registry_env, tmp_path):
        home = _make_home(tmp_path)
        with pytest.raises(DataRegistryError):
            register_data_home("  ", path=str(home), kind="logs", owner="a", safe_to_purge=True)
        with pytest.raises(DataRegistryError):
            register_data_home("h", path=str(home), kind="logs", owner="", safe_to_purge=True)

    def test_home_dir_and_root_refused(self, registry_env):
        with pytest.raises(DataRegistryError):
            register_data_home("h", path=str(Path.home()), kind="logs", owner="a", safe_to_purge=True)
        with pytest.raises(DataRegistryError):
            register_data_home("h", path="/", kind="logs", owner="a", safe_to_purge=True)

    def test_corrupt_registry_refused_never_silently_regenerated(self, registry_env, tmp_path):
        registry_env.parent.mkdir(parents=True, exist_ok=True)
        registry_env.write_text("{not json")
        home = _make_home(tmp_path)
        with pytest.raises(DataRegistryError) as e:
            register_data_home("h", path=str(home), kind="logs", owner="a", safe_to_purge=True)
        assert "not be silently overwritten" in str(e.value)
        assert registry_env.read_text() == "{not json"


class TestEnumeration:
    def test_list_rows_sorted_and_json_ready(self, registry_env, tmp_path):
        register_data_home("b-home", path=str(_make_home(tmp_path, "b")), kind="runs", owner="gw", safe_to_purge=True)
        register_data_home("a-home", path=str(_make_home(tmp_path, "a")), kind="logs", owner="gw", safe_to_purge=False)
        rows = list_data_homes()
        assert [r["name"] for r in rows] == ["a-home", "b-home"]
        json.dumps(rows)  # strictly serializable

    def test_sizes_on_demand_and_missing_path(self, registry_env, tmp_path):
        home = _make_home(tmp_path)
        (home / "f.bin").write_bytes(b"x" * 1024)
        sub = home / "sub"
        sub.mkdir()
        (sub / "g.bin").write_bytes(b"y" * 512)
        register_data_home("h", path=str(home), kind="prompt-cache", owner="core", safe_to_purge=True)
        register_data_home("gone", path=str(tmp_path / "never-made"), kind="logs", owner="core", safe_to_purge=True)

        rows = {r["name"]: r for r in list_data_homes(include_sizes=True)}
        assert rows["h"]["size_bytes"] == 1536
        assert rows["h"]["exists"] is True
        assert rows["gone"]["size_bytes"] is None
        assert rows["gone"]["exists"] is False
        assert data_home_size("h") == 1536

    def test_size_of_unknown_home_refused(self, registry_env):
        with pytest.raises(DataRegistryError):
            data_home_size("nope")


class TestPurge:
    def test_purge_deletes_contents_never_the_home_dir(self, registry_env, tmp_path):
        home = _make_home(tmp_path)
        (home / "f.bin").write_bytes(b"x" * 100)
        (home / "d").mkdir()
        (home / "d" / "g.bin").write_bytes(b"y" * 50)
        register_data_home("h", path=str(home), kind="prompt-cache", owner="core", safe_to_purge=True)

        result = purge_data_home("h")
        assert result["files_deleted"] == 2
        assert result["dirs_deleted"] == 1
        assert result["bytes_freed"] == 150
        assert result["errors"] == []
        assert home.is_dir(), "the home directory itself must survive a purge"
        assert list(home.iterdir()) == []
        assert get_data_home("h") is not None, "registration must survive a purge"

    def test_dry_run_accounts_without_deleting(self, registry_env, tmp_path):
        home = _make_home(tmp_path)
        (home / "f.bin").write_bytes(b"x" * 100)
        register_data_home("h", path=str(home), kind="logs", owner="core", safe_to_purge=True)

        result = purge_data_home("h", dry_run=True)
        assert result["dry_run"] is True
        assert result["files_deleted"] == 1
        assert result["bytes_freed"] == 100
        assert (home / "f.bin").exists()

    def test_owner_protected_row_refused_naming_owner_and_rule(self, registry_env, tmp_path):
        home = _make_home(tmp_path)
        (home / "life.sqlite3").write_bytes(b"z")
        register_data_home(
            "castor-home", path=str(home), kind="sessions",
            owner="abstractgateway", safe_to_purge=False,
        )
        with pytest.raises(DataRegistryError) as e:
            purge_data_home("castor-home")
        msg = str(e.value)
        assert "abstractgateway" in msg
        assert "safe_to_purge" in msg
        assert (home / "life.sqlite3").exists()

    def test_unregistered_name_refused(self, registry_env):
        with pytest.raises(DataRegistryError) as e:
            purge_data_home("ghost")
        assert "not in the data registry" in str(e.value)

    def test_missing_path_is_noop_accounting(self, registry_env, tmp_path):
        register_data_home("gone", path=str(tmp_path / "never"), kind="logs", owner="core", safe_to_purge=True)
        result = purge_data_home("gone")
        assert result["files_deleted"] == 0
        assert result["bytes_freed"] == 0

    def test_symlink_out_of_home_is_unlinked_never_followed(self, registry_env, tmp_path):
        outside = tmp_path / "outside"
        outside.mkdir()
        precious = outside / "precious.txt"
        precious.write_text("keep me")

        home = _make_home(tmp_path)
        (home / "link-dir").symlink_to(outside, target_is_directory=True)
        (home / "link-file").symlink_to(precious)
        register_data_home("h", path=str(home), kind="prompt-cache", owner="core", safe_to_purge=True)

        purge_data_home("h")
        assert precious.exists(), "purge must never follow symlinks out of the home"
        assert precious.read_text() == "keep me"
        assert not (home / "link-dir").exists()
        assert not (home / "link-file").exists()

    def test_unregister_removes_row_never_disk(self, registry_env, tmp_path):
        home = _make_home(tmp_path)
        (home / "f.bin").write_bytes(b"x")
        register_data_home("h", path=str(home), kind="logs", owner="core", safe_to_purge=True)
        assert unregister_data_home("h") is True
        assert get_data_home("h") is None
        assert (home / "f.bin").exists()
        assert unregister_data_home("h") is False


class TestConcurrency:
    def test_cross_process_register_at_first_write(self, registry_env, tmp_path):
        """Two processes registering concurrently must both land (no lost update)."""
        import multiprocessing as mp

        home_a = _make_home(tmp_path, "a")
        home_b = _make_home(tmp_path, "b")
        ctx = mp.get_context("spawn")
        procs = [
            ctx.Process(target=_register_in_subprocess, args=(str(registry_env), "proc-a", str(home_a))),
            ctx.Process(target=_register_in_subprocess, args=(str(registry_env), "proc-b", str(home_b))),
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0
        names = {r["name"] for r in list_data_homes()}
        assert {"proc-a", "proc-b"} <= names

    def test_stale_lock_is_swept(self, registry_env, tmp_path, monkeypatch):
        registry_env.parent.mkdir(parents=True, exist_ok=True)
        lock = registry_env.with_name(registry_env.name + ".lock")
        lock.write_text("999999")
        old = lock.stat()
        os.utime(lock, (old.st_atime - 3600, old.st_mtime - 3600))
        home = _make_home(tmp_path)
        row = register_data_home("h", path=str(home), kind="logs", owner="core", safe_to_purge=True)
        assert row.name == "h"


def _register_in_subprocess(reg_path: str, name: str, home_path: str) -> None:
    os.environ["ABSTRACTFRAMEWORK_DATA_REGISTRY"] = reg_path
    from abstractcore.utils.data_registry import register_data_home as _reg

    _reg(name, path=home_path, kind="runs", owner="test", safe_to_purge=True)


class TestPackageSurface:
    def test_public_exports(self):
        import abstractcore.utils as u

        for symbol in (
            "register_data_home", "list_data_homes", "purge_data_home",
            "get_data_home", "data_home_size", "unregister_data_home",
            "DataHome", "DataRegistryError", "DATA_HOME_KINDS", "registry_path",
            "register_core_data_homes",
        ):
            assert hasattr(u, symbol), f"abstractcore.utils must export {symbol}"

    def test_registry_path_env_override(self, registry_env):
        assert registry_path() == registry_env
