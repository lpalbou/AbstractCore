from __future__ import annotations

from pathlib import Path

import pytest

from abstractcore.utils import (
    WorkspacePathError,
    build_workspace_mounts,
    extensions_for_family,
    file_matches_filters,
    guess_file_family,
    resolve_workspace_path,
)


def test_build_workspace_mounts_uses_digest_for_colliding_names(tmp_path: Path) -> None:
    first = tmp_path / "team-a" / "reports"
    second = tmp_path / "team-b" / "reports"
    first.mkdir(parents=True, exist_ok=True)
    second.mkdir(parents=True, exist_ok=True)

    mounts = build_workspace_mounts(allowed_dirs=[first, second], used_names=set())

    assert len(mounts) == 2
    assert all(name.startswith("reports_") for name in mounts)


def test_resolve_workspace_path_accepts_mount_alias_root(tmp_path: Path) -> None:
    base = tmp_path / "workspace"
    mounted = tmp_path / "shared" / "docs"
    base.mkdir(parents=True, exist_ok=True)
    mounted.mkdir(parents=True, exist_ok=True)

    mounts = build_workspace_mounts(allowed_dirs=[mounted], used_names=set())
    alias = next(iter(mounts))

    resolved = resolve_workspace_path(base=base, mounts=mounts, raw_path=alias)

    assert resolved.resolved_path == mounted.resolve()
    assert resolved.virtual_path == alias
    assert resolved.root_path == mounted.resolve()


def test_resolve_workspace_path_rejects_escape(tmp_path: Path) -> None:
    base = tmp_path / "workspace"
    base.mkdir(parents=True, exist_ok=True)

    with pytest.raises(WorkspacePathError, match="path escapes workspace root"):
        resolve_workspace_path(base=base, mounts={}, raw_path="../outside.txt")


def test_file_filters_cover_known_families(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    code = tmp_path / "tool.py"
    archive = tmp_path / "bundle.zip"
    image.write_bytes(b"\x89PNG\r\n\x1a\n")
    code.write_text("print('hi')\n", encoding="utf-8")
    archive.write_bytes(b"PK\x03\x04")

    assert "png" in extensions_for_family("image")
    assert guess_file_family(code) == "code"
    assert guess_file_family(archive) == "archive"
    assert file_matches_filters(code, family="code")
    assert file_matches_filters(image, extensions=[".png"])
    assert not file_matches_filters(image, family="code")
