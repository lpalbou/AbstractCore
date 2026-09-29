"""Credentials encrypted at rest: AES-256-GCM, key in the OS keychain (or a 0600 key file).

Layout inside the store directory (created 0700):

    secret.enc   0600  {"v": 1, "alg": "AES-256-GCM", "key": "keyring" | "file", "key_id", "nonce", "ct"}
    secret.key   0600  base64 key, ONLY when no OS keychain is available

Key storage (`key_backend`):

- `"auto"` (default): the OS keychain through `keyring` (macOS Keychain, Windows Credential
  Manager, Linux Secret Service) when a real backend is available; otherwise the key file,
  and `key_warning` says so.
- `"keyring"`: the OS keychain or an error.
- `"file"`: the key file (headless hosts, tests).

What the encryption protects against, plainly: copies of the store (backups, synced or copied
folders, a file-reading tool, a log that captured the file). With the keychain, the key is not
in the folder at all. With the key file fallback, the key sits next to the sealed file: a copy
of the whole folder carries both, so the protection is limited to copies of `secret.enc`
alone. Code running as the same OS user can reach either key; this is not isolation between
programs of one user.

Nothing here logs, prints or raises a secret value.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import EmailInvalidSettings, EmailSecretUnavailable

KEYRING_SERVICE = "abstractcore-email"
KEY_BACKENDS = ("auto", "keyring", "file")
_AAD_PREFIX = b"abstractcore-email-secret-v1|"

KEY_FILE_WARNING = (
    "No OS keychain is available, so the encryption key is stored in a 0600 file next to the "
    "sealed credentials: a copy of the whole folder carries both. Protect this folder like a password."
)


def _aesgcm():
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except Exception as exc:  # pragma: no cover - cryptography is a base dependency
        raise EmailSecretUnavailable(
            "The `cryptography` package is not installed, so credentials cannot be encrypted.",
            "Reinstall AbstractCore (`pip install -U abstractcore`); cryptography is part of its light install.",
        ) from exc
    return AESGCM


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
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _real_keyring():
    """The keyring module when a usable (non-fail, non-null) backend is configured, else None."""

    try:
        import keyring  # type: ignore
        from keyring.backends import fail as _fail  # type: ignore
    except Exception:
        return None
    try:
        backend = keyring.get_keyring()
    except Exception:
        return None
    try:
        from keyring.backends import null as _null  # type: ignore

        null_cls = _null.Keyring
    except Exception:  # pragma: no cover
        null_cls = ()
    if isinstance(backend, _fail.Keyring) or (null_cls and isinstance(backend, null_cls)):
        return None
    try:
        if float(getattr(backend, "priority", 1)) <= 0:
            return None
    except Exception:
        return None
    return keyring


class SecretVault:
    """Seal / unseal one JSON object under `directory`."""

    def __init__(self, directory: Path, *, key_backend: str = "auto", keyring_service: str = KEYRING_SERVICE) -> None:
        backend = str(key_backend or "auto").strip().lower()
        if backend not in KEY_BACKENDS:
            raise EmailInvalidSettings(
                f"Unknown key storage {key_backend!r}.",
                "Use auto, keyring or file.",
            )
        self.directory = Path(directory)
        self.key_backend = backend
        self.keyring_service = keyring_service
        self.key_warning = ""

    def __repr__(self) -> str:
        return f"SecretVault(directory={str(self.directory)!r}, key_backend={self.key_backend!r})"

    @property
    def sealed_path(self) -> Path:
        return self.directory / "secret.enc"

    @property
    def key_path(self) -> Path:
        return self.directory / "secret.key"

    @property
    def key_id(self) -> str:
        try:
            resolved = str(self.directory.resolve())
        except OSError:
            resolved = str(self.directory.absolute())
        return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:24]

    def _ensure_dir(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.directory, 0o700)
        except OSError:
            pass

    # -- keys ------------------------------------------------------------------------------

    def _new_key_location(self) -> str:
        if self.key_backend == "file":
            return "file"
        kr = _real_keyring()
        if kr is None:
            if self.key_backend == "keyring":
                raise EmailSecretUnavailable(
                    "No OS keychain is available to hold the encryption key.",
                    "Use the key file storage on this host, or install and unlock a keychain (Secret Service on Linux).",
                )
            self.key_warning = KEY_FILE_WARNING
            return "file"
        return "keyring"

    def _store_key(self, location: str, key: bytes) -> str:
        encoded = base64.b64encode(key).decode("ascii")
        if location == "keyring":
            kr = _real_keyring()
            try:
                if kr is None:
                    raise RuntimeError("no keyring")
                kr.set_password(self.keyring_service, self.key_id, encoded)
                return "keyring"
            except Exception:
                if self.key_backend == "keyring":
                    raise EmailSecretUnavailable(
                        "The OS keychain refused to store the encryption key.",
                        "Unlock the keychain and retry, or use the key file storage on this host.",
                    ) from None
                self.key_warning = KEY_FILE_WARNING
                location = "file"
        _write_private(self.key_path, encoded.encode("ascii"))
        return "file"

    def _load_key(self, location: str) -> bytes:
        if location == "keyring":
            kr = _real_keyring()
            value = None
            if kr is not None:
                try:
                    value = kr.get_password(self.keyring_service, self.key_id)
                except Exception:
                    value = None
            if not value:
                raise EmailSecretUnavailable(
                    "The encryption key of the stored credentials is not in the OS keychain (keychain locked, "
                    "or the folder was moved or copied to another machine).",
                    "Unlock the keychain, or connect the account again (the credentials are sealed per folder).",
                )
            return base64.b64decode(value)
        try:
            return base64.b64decode(self.key_path.read_bytes().strip())
        except OSError:
            raise EmailSecretUnavailable(
                "The key file of the stored credentials is missing.",
                "Connect the account again to store the credentials.",
            ) from None

    def _delete_key(self, location: str) -> None:
        if location == "keyring":
            kr = _real_keyring()
            if kr is not None:
                try:
                    kr.delete_password(self.keyring_service, self.key_id)
                except Exception:
                    pass
            return
        try:
            self.key_path.unlink()
        except OSError:
            pass

    # -- public ----------------------------------------------------------------------------

    def exists(self) -> bool:
        return self.sealed_path.is_file()

    def location(self) -> str:
        """"keyring", "file", or "" when nothing is sealed."""

        try:
            meta = json.loads(self.sealed_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return ""
        return str(meta.get("key") or "") if isinstance(meta, dict) else ""

    def store(self, payload: Dict[str, Any], *, reuse_key: bool = False) -> str:
        """Seal `payload`. Returns where the key is ("keyring" | "file").

        New credentials get a fresh key (`reuse_key=False`); a token refresh re-seals under the
        existing key (`reuse_key=True`) so the keychain is not rewritten every hour.
        """

        AESGCM = _aesgcm()
        self._ensure_dir()
        previous = self.location()
        key: Optional[bytes] = None
        location = ""
        if reuse_key and previous:
            try:
                key = self._load_key(previous)
                location = previous
            except EmailSecretUnavailable:
                key = None
        if key is None:
            key = AESGCM.generate_key(bit_length=256)
            location = self._store_key(self._new_key_location(), key)
            if previous and previous != location:
                self._delete_key(previous)
        nonce = secrets.token_bytes(12)
        aad = _AAD_PREFIX + self.key_id.encode("ascii")
        ct = AESGCM(key).encrypt(nonce, json.dumps(payload, separators=(",", ":")).encode("utf-8"), aad)
        doc = {
            "v": 1,
            "alg": "AES-256-GCM",
            "key": location,
            "key_id": self.key_id,
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ct": base64.b64encode(ct).decode("ascii"),
        }
        _write_private(self.sealed_path, (json.dumps(doc, indent=2) + "\n").encode("utf-8"))
        return location

    def load(self) -> Optional[Dict[str, Any]]:
        if not self.exists():
            return None
        AESGCM = _aesgcm()
        try:
            meta = json.loads(self.sealed_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise EmailSecretUnavailable(
                "The sealed credentials file is unreadable.",
                "Connect the account again to store the credentials.",
            ) from None
        if not isinstance(meta, dict) or meta.get("v") != 1 or meta.get("alg") != "AES-256-GCM":
            raise EmailSecretUnavailable(
                "The sealed credentials file has an unknown format.",
                "Connect the account again to store the credentials.",
            )
        key = self._load_key(str(meta.get("key") or "file"))
        try:
            nonce = base64.b64decode(meta["nonce"])
            ct = base64.b64decode(meta["ct"])
            aad = _AAD_PREFIX + str(meta.get("key_id") or "").encode("ascii")
            plain = AESGCM(key).decrypt(nonce, ct, aad)
        except Exception:
            raise EmailSecretUnavailable(
                "The stored credentials could not be decrypted (wrong key or modified file).",
                "Connect the account again to store the credentials.",
            ) from None
        data = json.loads(plain.decode("utf-8"))
        return data if isinstance(data, dict) else None

    def delete(self) -> None:
        location = self.location()
        if location == "keyring":
            self._delete_key("keyring")
        self._delete_key("file")
        try:
            self.sealed_path.unlink()
        except OSError:
            pass

    def permissions_ok(self) -> bool:
        """True when the sealed file (and key file, if any) are not readable by group/others."""

        for path in (self.sealed_path, self.key_path):
            if path.exists():
                mode = stat.S_IMODE(path.stat().st_mode)
                if mode & 0o077:
                    return False
        return True
