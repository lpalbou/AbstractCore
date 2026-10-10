"""Round 16 sealing (2.26.0): one sealing-key file, no OS keychain on any OS — pure units.

- owner-only key folder/file on macOS and Linux (0700/0600), and on Windows (best-effort
  owner-only ACL through `icacls`, os.name monkeypatched — no Windows host needed);
- `keyring` is never imported: an AST scan of the whole package, and a trap `keyring` package
  first on sys.path of a fresh interpreter that records any import while sealing, unsealing and
  meeting a keychain-sealed store.
"""

from __future__ import annotations

import ast
import json
import os
import stat
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

import abstractcore
import abstractcore.comms.email.vault as vault_module

pytestmark = pytest.mark.basic

PACKAGE = Path(abstractcore.__file__).resolve().parent


# ------------------------------------------------------------------ permissions per OS


@pytest.mark.parametrize("platform_name", ["posix-darwin", "posix-linux"])
@pytest.mark.skipif(os.name == "nt", reason="POSIX modes")
def test_key_folder_and_file_are_owner_only_on_macos_and_linux(tmp_path: Path, monkeypatch, platform_name) -> None:
    monkeypatch.setattr(sys, "platform", platform_name.split("-")[1])
    calls = []
    monkeypatch.setattr(vault_module, "_restrict_windows", lambda *a, **k: calls.append(a) or True)
    key = tmp_path / "data" / "secrets" / "sealing.key"
    assert len(vault_module.ensure_key_file(key)) == 32
    assert stat.S_IMODE(key.stat().st_mode) == 0o600
    assert stat.S_IMODE(key.parent.stat().st_mode) == 0o700
    assert calls == []  # no ACL tool on POSIX
    # Made once: a second call reads the same key.
    assert vault_module.ensure_key_file(key) == vault_module.read_key_file(key)


def test_windows_gets_the_owner_only_acl_on_folder_and_file(tmp_path: Path, monkeypatch) -> None:
    ran = []

    class Done:
        returncode = 0

    def fake_run(argv, **kwargs):
        ran.append(list(argv))
        return Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("USERNAME", "alice")
    folder = tmp_path / "secrets"
    folder.mkdir()
    key = folder / "sealing.key"
    key.write_bytes(b"x")
    monkeypatch.setattr(os, "name", "nt")
    try:
        vault_module._make_private(folder, folder=True)
        vault_module._make_private(key, folder=False)
    finally:
        monkeypatch.setattr(os, "name", "posix")
    assert ran == [
        ["icacls", str(folder), "/inheritance:r", "/grant:r", "alice:(OI)(CI)F"],
        ["icacls", str(key), "/inheritance:r", "/grant:r", "alice:F"],
    ]


def test_windows_acl_is_best_effort_and_never_raises(tmp_path: Path, monkeypatch) -> None:
    def missing(*a, **k):
        raise FileNotFoundError("icacls")

    monkeypatch.setattr(subprocess, "run", missing)
    monkeypatch.setenv("USERNAME", "alice")
    target = tmp_path / "f"
    target.write_bytes(b"x")
    assert vault_module._restrict_windows(target, folder=False) is False
    monkeypatch.setattr(os, "name", "nt")
    try:
        vault_module._make_private(target, folder=False)  # no exception
    finally:
        monkeypatch.setattr(os, "name", "posix")


# ------------------------------------------------------------------ keyring is gone


def _keyring_imports(path: Path) -> list:
    out = []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        out += [f"{path.relative_to(PACKAGE)}:{node.lineno}" for n in names if n == "keyring" or n.startswith("keyring.")]
        if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant):
            fn = getattr(node.func, "attr", getattr(node.func, "id", ""))
            arg = node.args[0].value
            if fn in ("import_module", "__import__", "find_spec") and isinstance(arg, str) and arg.split(".")[0] == "keyring":
                out.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    return out


def test_no_abstractcore_module_imports_keyring_and_it_is_no_dependency() -> None:
    offenders = [o for p in sorted(PACKAGE.rglob("*.py")) for o in _keyring_imports(p)]
    assert offenders == []
    pyproject = PACKAGE.parent / "pyproject.toml"
    if pyproject.is_file():  # a source checkout
        assert '"keyring' not in pyproject.read_text(encoding="utf-8")


def test_a_trap_keyring_on_sys_path_is_never_imported(tmp_path: Path) -> None:
    trap = tmp_path / "trap"
    (trap / "keyring").mkdir(parents=True)
    hits = tmp_path / "keyring-imports.txt"
    (trap / "keyring" / "__init__.py").write_text(
        "import os, traceback\n"
        f"open({str(hits)!r}, 'a').write(''.join(traceback.format_stack()) + '\\n---\\n')\n"
        "def get_keyring():\n    raise RuntimeError('trap')\n"
        "def get_password(*a):\n    raise RuntimeError('trap')\n"
        "def set_password(*a):\n    raise RuntimeError('trap')\n"
    )
    data = tmp_path / "data"
    script = textwrap.dedent(
        f"""
        import json, sys
        from pathlib import Path
        from abstractcore.comms.email import SecretVault, EmailSecretUnavailable, EmailAccountStore
        from abstractcore.config import email_cli  # noqa: F401
        data = Path({str(data)!r})
        key = data / "secrets" / "sealing.key"
        v = SecretVault(data / "mail", key_file=key)
        v.store({{"password": "p"}})
        assert v.load() == {{"password": "p"}}
        old = SecretVault(data / "old", key_file=key)
        old.directory.mkdir(parents=True)
        old.sealed_path.write_text(json.dumps({{"v": 1, "alg": "AES-256-GCM", "key": "keyring",
                                               "key_id": "0" * 24, "nonce": "AA==", "ct": "AA=="}}))
        try:
            old.load()
        except EmailSecretUnavailable as exc:
            assert exc.code == "email_secret_sealed_with_keychain"
        else:
            raise SystemExit("a keychain-sealed store opened")
        assert old.retire_legacy_keychain()
        v.delete()
        assert "keyring" not in sys.modules, "keyring imported"
        print("ok")
        """
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = os.pathsep.join([str(trap), str(PACKAGE.parent)])
    env["HOME"] = str(tmp_path / "home")
    env["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    done = subprocess.run([sys.executable, "-c", script], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr[-2000:]
    assert not hits.exists(), hits.read_text()[-2000:]
