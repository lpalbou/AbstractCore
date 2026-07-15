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

    def test_nested_homes_refused_both_directions(self, registry_env, tmp_path):
        """P0 (adversarial find): an ancestor home purged as 'safe' would eat a
        nested home's data, bypassing the child's owner-declared safe_to_purge —
        the entity-home amputation class. Overlap is refused at registration."""
        parent = _make_home(tmp_path, "data")
        child = parent / "entities" / "castor"
        child.mkdir(parents=True)

        register_data_home("castor-home", path=str(child), kind="sessions",
                           owner="abstractgateway", safe_to_purge=False)
        with pytest.raises(DataRegistryError) as e:
            register_data_home("all-data", path=str(parent), kind="runs",
                               owner="abstractgateway", safe_to_purge=True)
        assert "overlaps" in str(e.value)
        assert "castor-home" in str(e.value)

        # Descendant direction refused too.
        deeper = child / "artifacts"
        deeper.mkdir()
        with pytest.raises(DataRegistryError):
            register_data_home("castor-artifacts", path=str(deeper), kind="logs",
                               owner="abstractgateway", safe_to_purge=True)

    def test_registry_container_dir_refused(self, registry_env, tmp_path):
        """P0 (adversarial find): registering the directory holding the registry
        file itself as safe-to-purge would let a purge destroy the registry."""
        with pytest.raises(DataRegistryError) as e:
            register_data_home("meta", path=str(registry_env.parent), kind="logs",
                               owner="x", safe_to_purge=True)
        assert "registry itself" in str(e.value)

    def test_reregister_same_name_same_path_is_not_overlap(self, registry_env, tmp_path):
        home = _make_home(tmp_path)
        register_data_home("h", path=str(home), kind="logs", owner="a", safe_to_purge=True)
        row = register_data_home("h", path=str(home), kind="logs", owner="a", safe_to_purge=True)
        assert row.name == "h"


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

        result = purge_data_home("h")
        assert precious.exists(), "purge must never follow symlinks out of the home"
        assert precious.read_text() == "keep me"
        assert not (home / "link-dir").exists()
        assert not (home / "link-file").exists()
        assert result["symlinks_removed"] == 2, "symlinks are accounted as symlinks, not files"

    def test_root_swapped_for_symlink_after_registration_refused(self, registry_env, tmp_path):
        """P1 (adversarial find, TOCTOU class): os.walk always scandirs `top`,
        so a home path replaced by a symlink AFTER registration would be
        followed into foreign territory. Purge must refuse the swap loudly."""
        home = _make_home(tmp_path, "real-home")
        register_data_home("h", path=str(home), kind="prompt-cache", owner="core", safe_to_purge=True)

        victim = tmp_path / "victim"
        victim.mkdir()
        (victim / "v.txt").write_text("do not delete")

        import shutil
        shutil.rmtree(home)
        home.symlink_to(victim, target_is_directory=True)

        with pytest.raises(DataRegistryError) as e:
            purge_data_home("h")
        assert "no longer resolves" in str(e.value)
        assert (victim / "v.txt").exists()

    def test_hand_edited_nested_home_is_skipped_at_purge(self, registry_env, tmp_path):
        """Belt for registries edited outside the API: a nested protected home
        inside a purgeable one is SKIPPED (with accounting), never deleted."""
        parent = _make_home(tmp_path, "data")
        child = parent / "entities" / "castor"
        child.mkdir(parents=True)
        (child / "life.sqlite3").write_bytes(b"life")
        (parent / "junk.tmp").write_bytes(b"x" * 10)

        register_data_home("all-data", path=str(parent), kind="runs",
                           owner="abstractgateway", safe_to_purge=True)
        # Simulate a hand-edited registry that bypassed the nesting guard.
        reg = json.loads(registry_env.read_text())
        reg["homes"]["castor-home"] = {
            "name": "castor-home", "path": str(child), "kind": "sessions",
            "owner": "abstractgateway", "safe_to_purge": False,
            "description": "", "registered_at": "x", "updated_at": "x", "meta": {},
        }
        registry_env.write_text(json.dumps(reg))

        result = purge_data_home("all-data")
        assert (child / "life.sqlite3").exists(), "nested protected home must survive"
        assert not (parent / "junk.tmp").exists()
        assert any("castor" in s for s in result["skipped_protected"])

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


class TestEnsureLane:
    """The best-effort register-at-first-write lane hot paths call.

    Contract: never raises into the caller, dedupes per process, warns
    #FALLBACK once per name on refusal but RETRIES the registration on the
    next call (transient failures heal), and honors the kill switch.
    """

    @pytest.fixture(autouse=True)
    def _fresh_process_state(self):
        from abstractcore.utils.data_registry import _reset_ensure_state

        _reset_ensure_state()
        yield
        _reset_ensure_state()

    def test_registers_once_then_dedupes(self, registry_env, tmp_path):
        from abstractcore.utils.data_registry import ensure_data_home_registered

        home = _make_home(tmp_path)
        first = ensure_data_home_registered(
            "ensure-home", path=str(home), kind="logs", owner="abstractcore", safe_to_purge=True,
        )
        assert first is not None and first.name == "ensure-home"
        second = ensure_data_home_registered(
            "ensure-home", path=str(home), kind="logs", owner="abstractcore", safe_to_purge=True,
        )
        assert second is None  # per-process dedup; the row still exists
        assert get_data_home("ensure-home").path == str(home.resolve())

    def test_refusal_never_raises_warns_once_and_retries(self, registry_env, tmp_path, caplog):
        import logging

        from abstractcore.utils.data_registry import ensure_data_home_registered

        outer = _make_home(tmp_path, "outer")
        register_data_home("outer", path=str(outer), kind="runs", owner="x", safe_to_purge=False)
        nested = outer / "nested"
        nested.mkdir()

        with caplog.at_level(logging.WARNING):
            assert ensure_data_home_registered(
                "nested-home", path=str(nested), kind="logs", owner="abstractcore", safe_to_purge=True,
            ) is None
            assert ensure_data_home_registered(
                "nested-home", path=str(nested), kind="logs", owner="abstractcore", safe_to_purge=True,
            ) is None
        warnings = [r for r in caplog.records if "#FALLBACK" in r.getMessage() and "nested-home" in r.getMessage()]
        assert len(warnings) == 1, "refusal must warn exactly once per name"

        # The failure was not latched as success: once the conflict clears,
        # the same name registers on the next call.
        unregister_data_home("outer")
        row = ensure_data_home_registered(
            "nested-home", path=str(nested), kind="logs", owner="abstractcore", safe_to_purge=True,
        )
        assert row is not None and get_data_home("nested-home") is not None

    def test_kill_switch_disables_registration(self, registry_env, tmp_path, monkeypatch):
        from abstractcore.utils.data_registry import ensure_data_home_registered

        monkeypatch.setenv("ABSTRACTFRAMEWORK_DATA_REGISTRY_DISABLE", "1")
        home = _make_home(tmp_path)
        assert ensure_data_home_registered(
            "killed", path=str(home), kind="logs", owner="abstractcore", safe_to_purge=True,
        ) is None
        assert get_data_home("killed") is None

    def test_ensure_core_data_homes_runs_once_per_process(self, registry_env, monkeypatch):
        import abstractcore.utils.data_registry as dr

        calls = {"n": 0}

        def _counting():
            calls["n"] += 1
            return []

        monkeypatch.setattr(dr, "register_core_data_homes", _counting)
        dr.ensure_core_data_homes()
        dr.ensure_core_data_homes()
        assert calls["n"] == 1

    def test_file_logging_registers_the_log_home(self, registry_env, tmp_path):
        """The structured-logging file handler is a real first-write site."""
        import logging as _logging

        from abstractcore.utils.structured_logging import LogConfig

        log_dir = tmp_path / "logs"
        cfg = LogConfig()
        prior_dir = cfg.log_dir
        prior_level = cfg.file_level
        try:
            cfg.configure(log_dir=str(log_dir), file_level=_logging.DEBUG)
            row = get_data_home("abstractcore-logs")
            assert row is not None and row.kind == "logs" and row.safe_to_purge is True
            assert row.path == str(log_dir.resolve())
        finally:
            cfg.log_dir = prior_dir
            cfg.file_level = prior_level
            cfg._setup_structlog()

    def test_bloc_store_upsert_registers_only_the_default_root(self, registry_env, tmp_path, monkeypatch):
        """FileBlocStore.upsert is the write moment for the largest core-owned
        data home (the 500-GB-class bloc/KV store) — first write at the
        machine-level DEFAULT root registers. Custom roots do NOT self-register
        (they live inside their caller's data home and ride that row; live
        incident 2026-07-13: suite-driven custom dirs spammed 372 tmp rows)."""
        import hashlib

        import abstractcore.core.file_blocs as fb

        default_root = tmp_path / "default-blocs"
        monkeypatch.setattr(fb, "default_blocs_root_dir", lambda: default_root)
        payload = "hello bloc"
        sha = hashlib.sha256(payload.encode()).hexdigest()

        fb.FileBlocStore(root_dir=default_root).upsert(
            file_meta={"sha256": sha, "path": "/tmp/x.txt", "size_bytes": len(payload)}, content=payload,
        )
        row = get_data_home("abstractcore-blocs")
        assert row is not None
        assert row.kind == "prompt-cache" and row.safe_to_purge is True
        assert row.path == str(default_root.resolve())

        custom_root = tmp_path / "caller-owned" / "blocs"
        fb.FileBlocStore(root_dir=custom_root).upsert(
            file_meta={"sha256": sha, "path": "/tmp/x.txt", "size_bytes": len(payload)}, content=payload,
        )
        rows = [r["name"] for r in list_data_homes()]
        assert rows.count("abstractcore-blocs") == 1
        assert not any(n.startswith("abstractcore-blocs-") for n in rows), (
            "custom bloc roots must not self-register"
        )


class TestPackageSurface:
    def test_public_exports(self):
        import abstractcore.utils as u

        for symbol in (
            "register_data_home", "list_data_homes", "purge_data_home",
            "get_data_home", "data_home_size", "unregister_data_home",
            "DataHome", "DataRegistryError", "DATA_HOME_KINDS", "registry_path",
            "register_core_data_homes", "ensure_data_home_registered",
            "ensure_core_data_homes",
        ):
            assert hasattr(u, symbol), f"abstractcore.utils must export {symbol}"

    def test_registry_path_env_override(self, registry_env):
        assert registry_path() == registry_env

    def test_kind_set_carries_the_semantics_ruled_entries(self, registry_env, tmp_path):
        """The ruled five + entity-home (semantics pre-ruling c1297) + artifacts
        (semantics ruling c1302), 2026-07-13 + workflow-memory (semantics ruling
        dm:memory--semantics#4, 2026-07-15 — flow's per-workflow memory graphs,
        operator data, safe_to_purge=True default; an entity home or any file
        under one must never register as workflow-memory). Widenings go through
        the semantics registry — this pin makes an ad-hoc widening a conscious
        act."""
        assert DATA_HOME_KINDS == (
            "model-cache", "prompt-cache", "runs", "sessions", "logs", "entity-home", "artifacts",
            "workflow-memory",
        )
        home = tmp_path / "castor"
        home.mkdir()
        row = register_data_home(
            "castor-home", path=str(home), kind="entity-home",
            owner="abstractgateway", safe_to_purge=False,
        )
        assert row.kind == "entity-home"
        store = tmp_path / "artifact-store"
        store.mkdir()
        row = register_data_home(
            "run-media-artifacts", path=str(store), kind="artifacts",
            owner="abstractgateway", safe_to_purge=True,
        )
        assert row.kind == "artifacts"
