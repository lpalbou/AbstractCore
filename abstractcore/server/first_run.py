"""First run of ``abstractcore serve`` on a loopback address.

A server bound to 127.0.0.1 / localhost / ::1 with no ``ABSTRACTCORE_AUTH_TOKEN``
and no ``ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED`` still requires a bearer
token, but it creates one for you:

- the token is generated once and kept in ``<config dir>/server-token``
  (mode 0600), then loaded as the server's auth token on every start;
- each start mints a single-use claim code (10-minute lifetime) kept only as
  a SHA-256 hash under ``<config dir>/claims/``;
- the banner prints ``http://127.0.0.1:<port>/console#claim=<code>``. The web
  console redeems the code through ``POST /acore/session/claim`` (accepted
  only from a loopback peer, never on the word of a forwarding header) and
  keeps the returned token in the tab's session storage.

A non-loopback bind keeps the explicit requirement: set
``ABSTRACTCORE_AUTH_TOKEN`` (or opt out with
``ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED=1``).

This module has no FastAPI dependency so the CLI (``abstractcore serve
--print-token`` / ``--claim-url``) can use it without the server extra.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import re
import secrets
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO

AUTH_TOKEN_ENV = "ABSTRACTCORE_AUTH_TOKEN"
ALLOW_UNAUTHENTICATED_ENV = "ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED"
# Non-secret marker set by `serve` when the token came from the token file:
# only then does `POST /acore/session/claim` hand the token out.
TOKEN_SOURCE_ENV = "ABSTRACTCORE_SERVER_TOKEN_SOURCE"
TOKEN_SOURCE_FIRST_RUN = "first_run"

TOKEN_FILENAME = "server-token"
CLAIMS_DIRNAME = "claims"
CLAIM_TTL_SECONDS = 600

_CLAIM_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_TRUTHY = {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------


def server_config_dir() -> Path:
    """The abstractcore config directory (``ABSTRACTCORE_CONFIG_DIR``,
    ``ABSTRACTCORE_CONFIG_FILE``'s folder, else ``~/.abstractcore/config``)."""

    from ..config.manager import resolve_config_file

    return resolve_config_file().parent


def token_path(config_dir: Optional[Path] = None) -> Path:
    return Path(config_dir or server_config_dir()) / TOKEN_FILENAME


def claims_dir(config_dir: Optional[Path] = None) -> Path:
    return Path(config_dir or server_config_dir()) / CLAIMS_DIRNAME


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------


def is_loopback_host(host: Optional[str]) -> bool:
    """True for ``localhost``, ``127.0.0.0/8`` and ``::1`` (brackets allowed)."""

    raw = str(host or "").strip().strip("[]").lower()
    if not raw:
        return False
    if raw == "localhost":
        return True
    try:
        return ipaddress.ip_address(raw).is_loopback
    except ValueError:
        return False


def env_token() -> str:
    return str(os.getenv(AUTH_TOKEN_ENV) or "").strip()


def allows_unauthenticated() -> bool:
    return str(os.getenv(ALLOW_UNAUTHENTICATED_ENV) or "").strip().lower() in _TRUTHY


def first_run_applies(bind_host: str) -> bool:
    """Loopback bind and no explicit auth choice (token or opt-out)."""

    return is_loopback_host(bind_host) and not env_token() and not allows_unauthenticated()


def claims_enabled() -> bool:
    """True when this server process loaded its token from the token file."""

    return os.getenv(TOKEN_SOURCE_ENV) == TOKEN_SOURCE_FIRST_RUN and bool(env_token())


# ---------------------------------------------------------------------------
# Token
# ---------------------------------------------------------------------------


def _write_private(path: Path, text: str) -> None:
    """Create ``path`` with mode 0600 (fails if it already exists)."""

    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(text)


def _tighten(path: Path) -> None:
    if os.name != "posix":
        return
    try:
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            os.chmod(path, 0o600)
    except OSError:
        pass


def load_or_create_token(config_dir: Optional[Path] = None) -> tuple[str, bool]:
    """Return ``(token, created)``. The file is created once (0600) and reused.

    Editing the file (one line) sets a fixed token; an empty file is refused
    rather than silently replaced.
    """

    path = token_path(config_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        token = secrets.token_urlsafe(32)
        try:
            _write_private(path, token + "\n")
            return token, True
        except FileExistsError:
            pass  # another process won the race: read theirs
    _tighten(path)
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise RuntimeError(f"{path} is empty; delete it to generate a new token or write one token on one line")
    return token, False


# ---------------------------------------------------------------------------
# Claim codes (single use, hashed at rest)
# ---------------------------------------------------------------------------


def _hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _prune(directory: Path, now: float) -> None:
    try:
        entries = list(directory.glob("*.json"))
    except OSError:
        return
    for entry in entries:
        try:
            data = json.loads(entry.read_text(encoding="utf-8"))
            if float(data.get("expires_at", 0)) < now:
                entry.unlink()
        except (OSError, ValueError, TypeError):
            try:
                entry.unlink()
            except OSError:
                pass


def mint_claim(config_dir: Optional[Path] = None, *, ttl_seconds: int = CLAIM_TTL_SECONDS, now: Optional[float] = None) -> str:
    """Create a single-use claim code; only its SHA-256 is written to disk."""

    now = time.time() if now is None else now
    directory = claims_dir(config_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        try:
            os.chmod(directory, 0o700)
        except OSError:
            pass
    _prune(directory, now)
    code = secrets.token_urlsafe(24)
    record = {"created_at": now, "expires_at": now + int(ttl_seconds)}
    _write_private(directory / f"{_hash_code(code)}.json", json.dumps(record))
    return code


@dataclass(frozen=True)
class ClaimResult:
    ok: bool
    reason: str = ""  # invalid | expired | used_or_unknown


def redeem_claim(code: str, config_dir: Optional[Path] = None, *, now: Optional[float] = None) -> ClaimResult:
    """Consume ``code`` once. The record is renamed away before it is read,
    so two concurrent redeems can never both succeed."""

    if not isinstance(code, str) or not _CLAIM_CODE_RE.match(code):
        return ClaimResult(False, "invalid")
    now = time.time() if now is None else now
    directory = claims_dir(config_dir)
    record_path = directory / f"{_hash_code(code)}.json"
    taken = directory / f".{record_path.stem}.{os.getpid()}.{secrets.token_hex(4)}.redeeming"
    try:
        os.replace(record_path, taken)
    except FileNotFoundError:
        return ClaimResult(False, "used_or_unknown")
    except OSError:
        return ClaimResult(False, "used_or_unknown")
    try:
        data = json.loads(taken.read_text(encoding="utf-8"))
        expires_at = float(data.get("expires_at", 0))
    except (OSError, ValueError, TypeError):
        expires_at = 0.0
    finally:
        try:
            taken.unlink()
        except OSError:
            pass
    if expires_at < now:
        return ClaimResult(False, "expired")
    return ClaimResult(True)


# ---------------------------------------------------------------------------
# Server start
# ---------------------------------------------------------------------------


def display_host(bind_host: str) -> str:
    raw = str(bind_host or "").strip()
    if raw.strip("[]") == "::1":
        return "[::1]"
    if raw.lower() == "localhost":
        return "127.0.0.1"
    return raw


def claim_url(host: str, port: int, code: str) -> str:
    return f"http://{display_host(host)}:{int(port)}/console#claim={code}"


@dataclass(frozen=True)
class FirstRunState:
    active: bool
    config_dir: Path
    token_created: bool = False
    claim_code: str = ""


def prepare_server_auth(bind_host: str, config_dir: Optional[Path] = None) -> FirstRunState:
    """Called by ``serve`` before the app starts.

    On a loopback bind with no explicit auth choice, load (or create) the token
    file into ``ABSTRACTCORE_AUTH_TOKEN`` for this process, mark the process as
    claim-enabled and mint a claim code. Otherwise change nothing.
    """

    directory = Path(config_dir or server_config_dir())
    if not first_run_applies(bind_host):
        return FirstRunState(active=False, config_dir=directory)
    token, created = load_or_create_token(directory)
    os.environ[AUTH_TOKEN_ENV] = token
    os.environ[TOKEN_SOURCE_ENV] = TOKEN_SOURCE_FIRST_RUN
    code = mint_claim(directory)
    return FirstRunState(active=True, config_dir=directory, token_created=created, claim_code=code)


def print_banner(state: FirstRunState, bind_host: str, port: int, *, out: Optional[TextIO] = None) -> None:
    """The start banner: where state lives, the URL, and how to authenticate."""

    stream = out or sys.stdout
    local_url = f"http://{display_host(bind_host) if is_loopback_host(bind_host) else '127.0.0.1'}:{int(port)}"
    lines = ["", "AbstractCore server", f"  Config dir: {state.config_dir}", f"  URL:        {local_url}"]
    if state.active:
        lines += [
            f"  Token:      {token_path(state.config_dir)}" + (" (created)" if state.token_created else ""),
            "",
            "  Open the console (one-time link, valid 10 minutes):",
            f"    {claim_url(bind_host, port, state.claim_code)}",
            "",
            "  From a terminal:",
            "    abstractcore serve --print-token        # the bearer token for API clients",
            f"    abstractcore serve --claim-url --port {int(port)}   # a fresh console link",
            "    ABSTRACTCORE_AUTH_TOKEN=<token> abstractcore serve   # use a fixed token instead",
        ]
    elif env_token():
        lines += ["  Auth:       bearer token from ABSTRACTCORE_AUTH_TOKEN", f"  Console:    {local_url}/console"]
    elif allows_unauthenticated():
        lines += [
            "  Auth:       OFF (ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED is set)",
            f"  Console:    {local_url}/console",
        ]
    else:
        lines += [
            f"  Auth:       NOT CONFIGURED for a non-loopback bind ({bind_host}).",
            "              Every API call will answer server_auth_not_configured until you",
            "              set ABSTRACTCORE_AUTH_TOKEN=<token>, or bind to loopback:",
            "                abstractcore serve --host 127.0.0.1",
        ]
    print("\n".join(lines) + "\n", file=stream, flush=True)
