"""TDLib (tdjson) wrapper used for Telegram Secret Chats (E2EE).

Why this exists:
- Telegram Bot API does *not* support Secret Chats (end-to-end encryption).
- TDLib is Telegram's official client library that supports Secret Chats and handles
  encryption, re-keying, sequencing, and media download/upload.

Design constraints (framework-wide):
- Keep Python dependencies minimal (stdlib-only here; TDLib itself is an external binary).
- Keep the interface JSON-safe for durable tool execution.
- Avoid side effects at import time: the tdjson shared library is loaded lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import json
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional


class TdlibNotAvailable(RuntimeError):
    pass


def _as_bool(raw: Any, default: bool) -> bool:
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    s = str(raw).strip().lower()
    if not s:
        return default
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _load_tdjson() -> ctypes.CDLL:
    """Load TDLib tdjson shared library.

    Users can override the location via:
    - ABSTRACT_TELEGRAM_TDJSON_PATH=/absolute/path/to/libtdjson.{so|dylib|dll}
    """

    override = str(os.getenv("ABSTRACT_TELEGRAM_TDJSON_PATH", "") or "").strip()
    candidates: list[str] = []
    if override:
        candidates.append(override)

    # Common names (platform dependent). We try them all and let ctypes resolve.
    candidates.extend(
        [
            "libtdjson.dylib",
            "libtdjson.so",
            "tdjson.dll",
            "libtdjson.dll",
        ]
    )

    last_err: Optional[Exception] = None
    for name in candidates:
        try:
            return ctypes.CDLL(name)
        except Exception as e:
            last_err = e

    hint = (
        "TDLib (tdjson) is required for Telegram Secret Chats. "
        "Install/build TDLib and set ABSTRACT_TELEGRAM_TDJSON_PATH to the tdjson shared library."
    )
    raise TdlibNotAvailable(f"Failed loading tdjson library. {hint} (last error: {last_err})")


class _TdJsonBindings:
    def __init__(self, lib: ctypes.CDLL):
        self._lib = lib

        self.td_json_client_create = lib.td_json_client_create
        self.td_json_client_create.restype = ctypes.c_void_p

        self.td_json_client_destroy = lib.td_json_client_destroy
        self.td_json_client_destroy.argtypes = [ctypes.c_void_p]
        self.td_json_client_destroy.restype = None

        self.td_json_client_send = lib.td_json_client_send
        self.td_json_client_send.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.td_json_client_send.restype = None

        self.td_json_client_receive = lib.td_json_client_receive
        self.td_json_client_receive.argtypes = [ctypes.c_void_p, ctypes.c_double]
        self.td_json_client_receive.restype = ctypes.c_char_p

        # NOTE: td_json_client_execute is available but only supports synchronous methods.
        # We keep it for completeness; main flow uses send/receive.
        self.td_json_client_execute = getattr(lib, "td_json_client_execute", None)
        if self.td_json_client_execute is not None:
            self.td_json_client_execute.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            self.td_json_client_execute.restype = ctypes.c_char_p


@dataclass(frozen=True)
class TdlibConfig:
    api_id: int
    api_hash: str
    phone: str
    database_directory: str
    files_directory: str
    database_encryption_key: str = ""
    use_secret_chats: bool = True
    # Optional: provide non-interactive auth for initial bootstrap (not recommended for production).
    login_code: Optional[str] = None
    two_factor_password: Optional[str] = None

    @staticmethod
    def from_env() -> "TdlibConfig":
        def _require(name: str) -> str:
            v = str(os.getenv(name, "") or "").strip()
            if not v:
                raise ValueError(f"Missing env var {name}")
            return v

        api_id_raw = _require("ABSTRACT_TELEGRAM_API_ID")
        try:
            api_id = int(api_id_raw)
        except Exception as e:
            raise ValueError("ABSTRACT_TELEGRAM_API_ID must be an integer") from e

        api_hash = _require("ABSTRACT_TELEGRAM_API_HASH")
        phone = _require("ABSTRACT_TELEGRAM_PHONE_NUMBER")

        db_dir = _require("ABSTRACT_TELEGRAM_DB_DIR")
        files_dir = str(os.getenv("ABSTRACT_TELEGRAM_FILES_DIR", "") or "").strip() or db_dir

        db_key = str(os.getenv("ABSTRACT_TELEGRAM_DB_ENCRYPTION_KEY", "") or "")
        use_secret = _as_bool(os.getenv("ABSTRACT_TELEGRAM_USE_SECRET_CHATS"), True)

        login_code = str(os.getenv("ABSTRACT_TELEGRAM_LOGIN_CODE", "") or "").strip() or None
        two_factor = str(os.getenv("ABSTRACT_TELEGRAM_2FA_PASSWORD", "") or "").strip() or None

        return TdlibConfig(
            api_id=api_id,
            api_hash=api_hash,
            phone=phone,
            database_directory=db_dir,
            files_directory=files_dir,
            database_encryption_key=db_key,
            use_secret_chats=bool(use_secret),
            login_code=login_code,
            two_factor_password=two_factor,
        )


class TdlibClient:
    """A single-process TDLib client with a background receive loop.

    TDLib requires that only one client instance accesses the database directory.
    This class is therefore designed to be used as a process-wide singleton.
    """

    def __init__(self, *, config: TdlibConfig):
        self._config = config
        self._td = _TdJsonBindings(_load_tdjson())
        self._client = self._td.td_json_client_create()
        if not self._client:
            raise TdlibNotAvailable("td_json_client_create() returned NULL")

        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._ready = threading.Event()
        self._closed = threading.Event()
        self._last_error: Optional[str] = None

        self._pending_lock = threading.Lock()
        self._pending: Dict[str, Dict[str, Any]] = {}

        self._handlers_lock = threading.Lock()
        self._update_handlers: list[Callable[[Dict[str, Any]], None]] = []

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def add_update_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        with self._handlers_lock:
            self._update_handlers.append(handler)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="tdlib-recv", daemon=True)
        self._thread.start()

        # Kick TDLib to emit initial auth state.
        self.send({"@type": "getAuthorizationState"})

    def stop(self) -> None:
        self._running = False
        try:
            self.send({"@type": "close"})
        except Exception:
            pass
        if self._thread is not None:
            try:
                self._thread.join(timeout=3.0)
            except Exception:
                pass
        try:
            self._td.td_json_client_destroy(self._client)
        except Exception:
            pass

    def wait_until_ready(self, *, timeout_s: float = 30.0) -> bool:
        return self._ready.wait(timeout=max(0.0, float(timeout_s)))

    def send(self, req: Dict[str, Any]) -> None:
        data = json.dumps(req, ensure_ascii=False).encode("utf-8")
        self._td.td_json_client_send(self._client, ctypes.c_char_p(data))

    def request(self, req: Dict[str, Any], *, timeout_s: float = 15.0) -> Dict[str, Any]:
        extra = uuid.uuid4().hex
        req2 = dict(req)
        req2["@extra"] = extra
        done = threading.Event()
        slot: Dict[str, Any] = {"done": done, "response": None}
        with self._pending_lock:
            self._pending[extra] = slot
        self.send(req2)
        ok = done.wait(timeout=max(0.0, float(timeout_s)))
        with self._pending_lock:
            self._pending.pop(extra, None)
        if not ok:
            raise TimeoutError(f"TDLib request timed out after {timeout_s}s (@type={req.get('@type')})")
        resp = slot.get("response")
        return dict(resp) if isinstance(resp, dict) else {"@type": "error", "message": "Invalid response"}

    # ---------------------------------------------------------------------
    # Internal loop + authorization
    # ---------------------------------------------------------------------

    def _loop(self) -> None:
        # Best-effort: reduce noise if supported.
        try:
            self.send({"@type": "setLogVerbosityLevel", "new_verbosity_level": 1})
        except Exception:
            pass

        while self._running and not self._closed.is_set():
            try:
                raw = self._td.td_json_client_receive(self._client, ctypes.c_double(1.0))
            except Exception as e:
                self._last_error = str(e)
                time.sleep(0.25)
                continue

            if not raw:
                continue
            try:
                text = raw.decode("utf-8", errors="replace")
                msg = json.loads(text)
            except Exception:
                continue
            if not isinstance(msg, dict):
                continue

            # Route explicit responses to pending waiters.
            extra = msg.get("@extra")
            if isinstance(extra, str) and extra:
                with self._pending_lock:
                    slot = self._pending.get(extra)
                if slot is not None:
                    slot["response"] = msg
                    done = slot.get("done")
                    if isinstance(done, threading.Event):
                        done.set()
                    continue

            typ = msg.get("@type")
            if typ == "updateAuthorizationState":
                try:
                    self._handle_auth_update(msg)
                except Exception as e:
                    self._last_error = str(e)
                continue

            if typ == "updateConnectionState":
                # Useful for debugging; no-op.
                continue

            # Broadcast other updates.
            if isinstance(typ, str) and typ.startswith("update"):
                with self._handlers_lock:
                    handlers = list(self._update_handlers)
                for h in handlers:
                    try:
                        h(msg)
                    except Exception:
                        continue

    def _handle_auth_update(self, update: Dict[str, Any]) -> None:
        state = update.get("authorization_state")
        if not isinstance(state, dict):
            return
        st = str(state.get("@type") or "").strip()
        if not st:
            return

        if st == "authorizationStateReady":
            self._ready.set()
            return

        if st == "authorizationStateClosed":
            self._closed.set()
            return

        if st == "authorizationStateWaitTdlibParameters":
            cfg = self._config
            params = {
                "@type": "setTdlibParameters",
                "parameters": {
                    "database_directory": cfg.database_directory,
                    "files_directory": cfg.files_directory,
                    "use_message_database": True,
                    "use_secret_chats": bool(cfg.use_secret_chats),
                    "use_file_database": True,
                    "use_chat_info_database": True,
                    "api_id": int(cfg.api_id),
                    "api_hash": str(cfg.api_hash),
                    "system_language_code": "en",
                    "device_model": "AbstractFramework",
                    "system_version": "0.0",
                    "application_version": "0.0",
                    "enable_storage_optimizer": True,
                },
            }
            self.send(params)
            return

        if st == "authorizationStateWaitEncryptionKey":
            # If the database is encrypted, we must provide the key.
            self.send(
                {
                    "@type": "checkDatabaseEncryptionKey",
                    "encryption_key": self._config.database_encryption_key or "",
                }
            )
            return

        if st == "authorizationStateWaitPhoneNumber":
            self.send({"@type": "setAuthenticationPhoneNumber", "phone_number": self._config.phone})
            return

        if st == "authorizationStateWaitCode":
            code = self._config.login_code
            if not code:
                self._last_error = (
                    "TDLib needs a login code (authorizationStateWaitCode) but ABSTRACT_TELEGRAM_LOGIN_CODE is not set. "
                    "Run the one-time bootstrap flow described in docs/guide/telegram-integration.md to create the session."
                )
                return
            self.send({"@type": "checkAuthenticationCode", "code": code})
            return

        if st == "authorizationStateWaitPassword":
            pw = self._config.two_factor_password
            if not pw:
                self._last_error = (
                    "TDLib requires a 2FA password (authorizationStateWaitPassword) but ABSTRACT_TELEGRAM_2FA_PASSWORD is not set."
                )
                return
            self.send({"@type": "checkAuthenticationPassword", "password": pw})
            return


_GLOBAL_TDLIB: Optional[TdlibClient] = None
_GLOBAL_TDLIB_LOCK = threading.Lock()


def get_global_tdlib_client(*, start: bool = True) -> TdlibClient:
    global _GLOBAL_TDLIB
    with _GLOBAL_TDLIB_LOCK:
        if _GLOBAL_TDLIB is None:
            cfg = TdlibConfig.from_env()
            _GLOBAL_TDLIB = TdlibClient(config=cfg)
            if start:
                _GLOBAL_TDLIB.start()
        return _GLOBAL_TDLIB


def stop_global_tdlib_client() -> None:
    global _GLOBAL_TDLIB
    with _GLOBAL_TDLIB_LOCK:
        if _GLOBAL_TDLIB is None:
            return
        try:
            _GLOBAL_TDLIB.stop()
        finally:
            _GLOBAL_TDLIB = None

