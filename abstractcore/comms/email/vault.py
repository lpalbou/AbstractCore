"""Credentials encrypted at rest: AES-256-GCM under ONE sealing-key file — never an OS keychain.

The same file, the same code and the same guarantees on macOS, Linux and Windows (2.26.0,
operator ruling 2026-10-10: keychains are a macOS feature, and reading one prompts the user):

    <key root>/secrets/               0700
    <key root>/secrets/sealing.key    0600  base64 of 32 random bytes (os.urandom), made on first use

    <directory>/secret.enc            0600  {"v": 1, "alg": "AES-256-GCM", "key": "sealing-key",
                                             "kid", "nonce", "ct"}

`key_file` names the sealing key; by default it is `<directory's parent>/secrets/sealing.key`,
so AbstractCore's own account store (`<config dir>/email/`) uses `<config dir>/secrets/sealing.key`.
A host passes its own (AbstractGateway: `<data dir>/secrets/sealing.key`, one key for every store
of its data folder). Each seal binds the store's place relative to the key root (the AES-GCM
associated data): a sealed file copied into another store does not open, and a folder moved or
restored elsewhere WITH its `secrets/` keeps working.

What the encryption protects against, plainly: copies of the folder taken WITHOUT
`secrets/sealing.key` (a synced sub-folder, a file-reading tool, a log that captured one file).
Code running as the same OS user can read the key file, exactly as it could read that user's
keychain items before; a copy of the whole folder carries the key. Backups must include
`secrets/`. On Windows the 0600/0700 modes are best effort (Python maps them to the read-only
flag); the folder's NTFS permissions protect the key file there.

Older stores (AbstractCore < 2.26):

- `"key": "keyring"` — the key is in an OS keychain. It is NEVER read (no keychain call, no
  password prompt, no `keyring` import): `load()` raises `EmailSecretUnavailable` with the
  caller's sentence (`legacy_cause` / `legacy_fix` / `legacy_code`), `legacy_keychain()` says
  so, `retire_legacy_keychain()` moves the file aside (`secret.keychain-old.enc`, never read).
- `"key": "file"` — the key is a `secret.key` beside it: opened, then re-sealed under the
  sealing key and the old file deleted (`reseal_legacy=False` reads without writing).

Nothing here logs, prints or raises a secret value.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import stat
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import EmailInvalidSettings, EmailSecretUnavailable

SECRETS_DIRNAME = "secrets"
KEY_FILENAME = "sealing.key"
KEY_BYTES = 32
KEY_LABEL = "sealing-key"
LEGACY_RETIRED_NAME = "secret.keychain-old.enc"
# Accepted for compatibility (the `--key-storage` flag, older callers): both mean the key file.
KEY_BACKENDS = ("auto", "file")
# Kept for compatibility: nothing reads it any more (no keychain).
KEYRING_SERVICE = "abstractcore-email"
# Kept for compatibility (store.py imported it): there is no fallback to warn about any more.
KEY_FILE_WARNING = ""

_AAD_PREFIX = b"abstractcore-sealed-v2|"
_LEGACY_AAD_PREFIX = b"abstractcore-email-secret-v1|"

LEGACY_KEYCHAIN_CAUSE = (
    "The stored credentials were sealed with the old OS keychain key, which AbstractCore no longer "
    "reads; connect the account again — the new key lives in the config folder."
)
LEGACY_KEYCHAIN_FIX = "Connect the account again to store the credentials (the keychain is never opened)."
LEGACY_KEYCHAIN_CODE = "email_secret_sealed_with_keychain"

_KEY_LOCK = threading.Lock()


def _aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except Exception as exc:  # pragma: no cover - cryptography is a base dependency
        raise EmailSecretUnavailable(
            "The `cryptography` package is not installed, so credentials cannot be encrypted.",
            "Reinstall AbstractCore (`pip install -U abstractcore`); cryptography is part of its light install.",
        ) from exc
    return AESGCM


def _chmod(path: Path, mode: int) -> None:
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _write_private(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}-{secrets.token_hex(4)}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    _chmod(path, 0o600)


def _decode_key(raw: bytes, path: Path) -> bytes:
    try:
        key = base64.b64decode(raw.strip(), validate=True)
    except Exception:
        key = b""
    if len(key) != KEY_BYTES:
        raise EmailSecretUnavailable(
            f"The sealing key file {path} is damaged (it must hold 32 bytes, base64).",
            "Restore secrets/sealing.key from a backup of this folder, or delete it and connect the account again.",
        )
    return key


def read_key_file(path: Path) -> Optional[bytes]:
    """The sealing key at `path`, or None when it was never made (nothing is created)."""

    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    return _decode_key(raw, path)


def ensure_key_file(path: Path) -> bytes:
    """The sealing key at `path`, made on first use: its folder 0700, the file 0600."""

    path = Path(path)
    folder = path.parent
    with _KEY_LOCK:
        folder.mkdir(parents=True, exist_ok=True)
        _chmod(folder, 0o700)
        try:
            return _decode_key(path.read_bytes(), path)
        except FileNotFoundError:
            pass
        encoded = base64.b64encode(os.urandom(KEY_BYTES)) + b"\n"
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:  # another process made it first: use theirs
            return _decode_key(path.read_bytes(), path)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(encoded)
                fh.flush()
                os.fsync(fh.fileno())
        except BaseException:
            try:
                path.unlink()
            except OSError:
                pass
            raise
        _chmod(path, 0o600)
        return _decode_key(encoded, path)


def key_fingerprint(key: bytes) -> str:
    """A non-secret fingerprint of a sealing key (which key sealed a file)."""

    return hashlib.sha256(b"abstractcore-sealing-kid|" + key).hexdigest()[:16]


class SecretVault:
    """Seal / unseal one JSON object under `directory` with the sealing key at `key_file`."""

    def __init__(
        self,
        directory: Path,
        *,
        key_file: Optional[Path] = None,
        key_backend: str = "auto",
        keyring_service: Optional[str] = None,
        reseal_legacy: bool = True,
        legacy_cause: str = LEGACY_KEYCHAIN_CAUSE,
        legacy_fix: str = LEGACY_KEYCHAIN_FIX,
        legacy_code: str = LEGACY_KEYCHAIN_CODE,
    ) -> None:
        backend = str(key_backend or "auto").strip().lower()
        if backend not in KEY_BACKENDS:
            raise EmailInvalidSettings(
                f"Unknown key storage {key_backend!r}: AbstractCore no longer uses an OS keychain.",
                "Leave the key storage at auto (the sealing key file in the config folder).",
            )
        del keyring_service  # accepted for compatibility; no keychain is ever used
        self.directory = Path(directory)
        self.key_file = Path(key_file) if key_file is not None else self.directory.parent / SECRETS_DIRNAME / KEY_FILENAME
        self.key_backend = "file"
        self.reseal_legacy = bool(reseal_legacy)
        self.legacy_cause = legacy_cause
        self.legacy_fix = legacy_fix
        self.legacy_code = legacy_code
        self.key_warning = ""

    def __repr__(self) -> str:
        return f"SecretVault(directory={str(self.directory)!r}, key_file={str(self.key_file)!r})"

    @property
    def sealed_path(self) -> Path:
        return self.directory / "secret.enc"

    @property
    def key_path(self) -> Path:
        """The OLD per-store key file (AbstractCore < 2.26 without a keychain)."""

        return self.directory / "secret.key"

    @property
    def key_id(self) -> str:
        """Where this store sits, relative to the key root (`<root>/secrets/sealing.key`)."""

        root = self.key_file.parent.parent
        try:
            here = self.directory.resolve()
        except OSError:
            here = self.directory.absolute()
        try:
            base = root.resolve()
        except OSError:
            base = root.absolute()
        try:
            return here.relative_to(base).as_posix()
        except ValueError:
            return here.as_posix()

    def _ensure_dir(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        _chmod(self.directory, 0o700)

    def _meta(self) -> Optional[Dict[str, Any]]:
        try:
            meta = json.loads(self.sealed_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            return {}
        return meta if isinstance(meta, dict) else {}

    # -- the key ---------------------------------------------------------------------------

    def ensure_key(self) -> bytes:
        """The sealing key, made on first use (0600 file in a 0700 folder)."""

        return ensure_key_file(self.key_file)

    # -- public ----------------------------------------------------------------------------

    def exists(self) -> bool:
        return self.sealed_path.is_file()

    def location(self) -> str:
        """"sealing-key" (sealed here), "keyring" / "file" (an older store), "" when nothing is sealed."""

        meta = self._meta()
        if not meta:
            return ""
        return str(meta.get("key") or "")

    def legacy_keychain(self) -> bool:
        """Sealed by AbstractCore < 2.26 with the key in an OS keychain (never opened here)."""

        meta = self._meta()
        return bool(meta) and str(meta.get("key") or "") == "keyring"

    def store(self, payload: Dict[str, Any], *, reuse_key: bool = False) -> str:
        """Seal `payload` under the sealing key. Returns "sealing-key". `reuse_key` is accepted
        for compatibility: there is one key, so it changes nothing."""

        del reuse_key
        AESGCM = _aesgcm()
        key = self.ensure_key()
        self._ensure_dir()
        nonce = secrets.token_bytes(12)
        aad = _AAD_PREFIX + self.key_id.encode("utf-8")
        ct = AESGCM(key).encrypt(nonce, json.dumps(payload, separators=(",", ":")).encode("utf-8"), aad)
        doc = {
            "v": 1,
            "alg": "AES-256-GCM",
            "key": KEY_LABEL,
            "kid": key_fingerprint(key),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ct": base64.b64encode(ct).decode("ascii"),
        }
        _write_private(self.sealed_path, (json.dumps(doc, indent=2) + "\n").encode("utf-8"))
        try:
            self.key_path.unlink()  # an older per-store key file has nothing left to open
        except OSError:
            pass
        return KEY_LABEL

    def load(self) -> Optional[Dict[str, Any]]:
        meta = self._meta()
        if meta is None:
            return None
        if not meta or meta.get("v") != 1 or meta.get("alg") != "AES-256-GCM":
            raise EmailSecretUnavailable(
                "The sealed credentials file is unreadable or has an unknown format.",
                "Connect the account again to store the credentials.",
            )
        where = str(meta.get("key") or "")
        if where == "keyring":
            raise EmailSecretUnavailable(self.legacy_cause, self.legacy_fix, code=self.legacy_code)
        if where == "file":
            return self._load_legacy_file(meta)
        if where != KEY_LABEL:
            raise EmailSecretUnavailable(
                "The sealed credentials file names a key this version does not know.",
                "Connect the account again to store the credentials.",
            )
        key = read_key_file(self.key_file)
        if key is None:
            raise EmailSecretUnavailable(
                f"The sealing key ({self.key_file}) is missing, so the stored credentials cannot be opened.",
                "Restore secrets/sealing.key from the backup this folder came from, or connect the account again.",
            )
        if str(meta.get("kid") or "") not in ("", key_fingerprint(key)):
            raise EmailSecretUnavailable(
                "The stored credentials were sealed with another sealing key (secrets/sealing.key was replaced).",
                "Restore the matching secrets/sealing.key from a backup, or connect the account again.",
            )
        try:
            aad = _AAD_PREFIX + self.key_id.encode("utf-8")
            plain = _aesgcm()(key).decrypt(base64.b64decode(meta["nonce"]), base64.b64decode(meta["ct"]), aad)
        except Exception:
            raise EmailSecretUnavailable(
                "The stored credentials could not be decrypted (modified file, or copied from another store).",
                "Connect the account again to store the credentials.",
            ) from None
        data = json.loads(plain.decode("utf-8"))
        return data if isinstance(data, dict) else None

    def _load_legacy_file(self, meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """An AbstractCore < 2.26 store whose key is the 0600 `secret.key` beside it."""

        try:
            key = base64.b64decode(self.key_path.read_bytes().strip())
            aad = _LEGACY_AAD_PREFIX + str(meta.get("key_id") or "").encode("ascii")
            plain = _aesgcm()(key).decrypt(base64.b64decode(meta["nonce"]), base64.b64decode(meta["ct"]), aad)
        except Exception:
            raise EmailSecretUnavailable(
                "The stored credentials (older key file beside them) could not be opened.",
                "Connect the account again to store the credentials.",
            ) from None
        data = json.loads(plain.decode("utf-8"))
        data = data if isinstance(data, dict) else None
        if data is not None and self.reseal_legacy:
            self.store(data)  # now under the sealing key; the old key file is deleted
        return data

    def delete(self) -> None:
        """Remove the sealed file (and an older key file beside it). Never touches a keychain;
        the sealing key stays (other stores use it)."""

        for path in (self.sealed_path, self.key_path):
            try:
                path.unlink()
            except OSError:
                pass

    def retire_legacy_keychain(self) -> bool:
        """Move a keychain-sealed file aside (`secret.keychain-old.enc`, never read) so the
        store reads as empty. True when something was moved."""

        if not self.legacy_keychain():
            return False
        try:
            os.replace(self.sealed_path, self.directory / LEGACY_RETIRED_NAME)
        except OSError:
            return False
        return True

    def permissions_ok(self) -> bool:
        """True when the sealed file, an older key file and the sealing key are not readable by group/others."""

        for path in (self.sealed_path, self.key_path, self.key_file):
            if path.exists():
                mode = stat.S_IMODE(path.stat().st_mode)
                if mode & 0o077:
                    return False
        return True
