"""Communication tools (email + WhatsApp).

Design goals:
- Durable-tool friendly: JSON-safe inputs/outputs; no callables persisted in run state.
- Secrets-safe by default: resolve credentials from env vars at execution time (avoid ledger leaks).
- Minimal dependencies: email uses stdlib (imaplib/smtplib/email); WhatsApp uses `requests` (already used by common tools).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import email
from email.header import decode_header
from email.message import EmailMessage, Message
from email.utils import formatdate, make_msgid
import imaplib
import json
import os
from pathlib import Path
import re
import smtplib
from typing import Any, Dict, List, Optional, Tuple

from abstractcore.tools.core import tool
from abstractcore.utils.truncation import preview_text


_MONTH_ABBR = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def _coerce_str_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if isinstance(v, str) and v.strip()]
    if isinstance(value, tuple):
        return [str(v).strip() for v in value if isinstance(v, str) and v.strip()]
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return []
        parts = [p.strip() for p in re.split(r"[;,]+", raw) if p.strip()]
        return parts
    text = str(value).strip()
    return [text] if text else []


def _decode_mime_header(value: Optional[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        return ""
    try:
        chunks = decode_header(value)
    except Exception:
        return value.strip()

    out: list[str] = []
    for part, charset in chunks:
        if isinstance(part, bytes):
            enc = charset or "utf-8"
            try:
                out.append(part.decode(enc, errors="replace"))
            except Exception:
                out.append(part.decode("utf-8", errors="replace"))
        else:
            out.append(str(part))
    return "".join(out).strip()


def _resolve_required_env(env_var: str, *, label: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve a secret reference.

    The input is usually an env var *name* (e.g. EMAIL_PASSWORD). For pragmatic operator ergonomics,
    we also accept literal secrets (common when the "env var name" contains characters that shells
    can't export, or when operators intentionally place the secret directly in the reference).

    Resolution rules:
    1) If an env var exists with that name, use its value.
    2) If the reference looks like a conventional env var identifier, fail fast (clear config error).
    3) Otherwise treat the reference itself as the secret.
    """
    ref = str(env_var or "").strip()
    if not ref:
        return None, f"Missing {label} env var name"

    value = os.getenv(ref)
    if value is not None and str(value).strip():
        return str(value), None

    # If it looks like a normal env var name, missing should be an error (don't silently use a name as a password).
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ref):
        return None, f"Missing env var {ref} for {label}"

    # Otherwise: treat the reference itself as the secret.
    return ref, None


def _env_str(name: str) -> Optional[str]:
    v = os.getenv(name)
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _env_bool(name: str) -> Optional[bool]:
    v = _env_str(name)
    if v is None:
        return None
    s = v.strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return None


def _env_int(name: str) -> Optional[int]:
    v = _env_str(name)
    if v is None:
        return None
    try:
        return int(v)
    except Exception:
        return None


def _load_smtp_defaults() -> Dict[str, Any]:
    """Load default SMTP settings from AbstractCore config + env overrides.

    Precedence:
    1) env vars (ABSTRACT_EMAIL_SMTP_*)
    2) AbstractCore config system (`config.email.*`)
    3) hardcoded fallbacks
    """
    out: Dict[str, Any] = {
        "smtp_host": "",
        "smtp_port": 587,
        "username": "",
        "password_env_var": "EMAIL_PASSWORD",
        "use_starttls": True,
        "from_email": "",
        "reply_to": "",
    }

    # AbstractCore config (best-effort).
    try:
        from abstractcore.config.manager import get_config_manager  # type: ignore

        cfg = get_config_manager().config
        email_cfg = getattr(cfg, "email", None)
        if email_cfg is not None:
            out["smtp_host"] = str(getattr(email_cfg, "smtp_host", "") or "")
            try:
                out["smtp_port"] = int(getattr(email_cfg, "smtp_port", 587) or 587)
            except Exception:
                out["smtp_port"] = 587
            out["username"] = str(getattr(email_cfg, "smtp_username", "") or "")
            out["password_env_var"] = str(getattr(email_cfg, "smtp_password_env_var", "") or "") or "EMAIL_PASSWORD"
            out["use_starttls"] = bool(getattr(email_cfg, "smtp_use_starttls", True))
            out["from_email"] = str(getattr(email_cfg, "from_email", "") or "")
            out["reply_to"] = str(getattr(email_cfg, "reply_to", "") or "")
    except Exception:
        pass

    # Env overrides (framework-native).
    out["smtp_host"] = _env_str("ABSTRACT_EMAIL_SMTP_HOST") or str(out.get("smtp_host") or "")
    out["username"] = _env_str("ABSTRACT_EMAIL_SMTP_USERNAME") or str(out.get("username") or "")
    out["password_env_var"] = _env_str("ABSTRACT_EMAIL_SMTP_PASSWORD_ENV_VAR") or str(out.get("password_env_var") or "EMAIL_PASSWORD")
    out["from_email"] = _env_str("ABSTRACT_EMAIL_FROM") or str(out.get("from_email") or "")
    out["reply_to"] = _env_str("ABSTRACT_EMAIL_REPLY_TO") or str(out.get("reply_to") or "")

    port = _env_int("ABSTRACT_EMAIL_SMTP_PORT")
    if isinstance(port, int) and port > 0:
        out["smtp_port"] = int(port)

    starttls = _env_bool("ABSTRACT_EMAIL_SMTP_STARTTLS")
    if isinstance(starttls, bool):
        out["use_starttls"] = bool(starttls)
    else:
        # If STARTTLS isn't explicitly configured, infer a safe default from common ports.
        # - 465 => implicit TLS (SMTP_SSL)
        # - 587/25 => STARTTLS (when supported by the server)
        try:
            port_i = int(out.get("smtp_port") or 587)
        except Exception:
            port_i = 587
        if port_i == 465:
            out["use_starttls"] = False

    return out


def _load_imap_defaults() -> Dict[str, Any]:
    """Load default IMAP settings from AbstractCore config + env overrides.

    Precedence:
    1) env vars (ABSTRACT_EMAIL_IMAP_*)
    2) AbstractCore config system (`config.email.*`)
    3) hardcoded fallbacks
    """
    out: Dict[str, Any] = {
        "imap_host": "",
        "imap_port": 993,
        "username": "",
        "password_env_var": "EMAIL_PASSWORD",
    }

    # AbstractCore config (best-effort).
    try:
        from abstractcore.config.manager import get_config_manager  # type: ignore

        cfg = get_config_manager().config
        email_cfg = getattr(cfg, "email", None)
        if email_cfg is not None:
            out["imap_host"] = str(getattr(email_cfg, "imap_host", "") or "")
            try:
                out["imap_port"] = int(getattr(email_cfg, "imap_port", 993) or 993)
            except Exception:
                out["imap_port"] = 993
            out["username"] = str(getattr(email_cfg, "imap_username", "") or "")
            out["password_env_var"] = str(getattr(email_cfg, "imap_password_env_var", "") or "") or "EMAIL_PASSWORD"
    except Exception:
        pass

    # Env overrides (framework-native).
    out["imap_host"] = _env_str("ABSTRACT_EMAIL_IMAP_HOST") or str(out.get("imap_host") or "")
    out["username"] = _env_str("ABSTRACT_EMAIL_IMAP_USERNAME") or str(out.get("username") or "")
    out["password_env_var"] = _env_str("ABSTRACT_EMAIL_IMAP_PASSWORD_ENV_VAR") or str(out.get("password_env_var") or "EMAIL_PASSWORD")

    port = _env_int("ABSTRACT_EMAIL_IMAP_PORT")
    if isinstance(port, int) and port > 0:
        out["imap_port"] = int(port)

    return out


@dataclass(frozen=True)
class _EmailImapConfig:
    host: str
    port: int
    username: str
    password_env_var: str
    mailbox: str


@dataclass(frozen=True)
class _EmailSmtpConfig:
    host: str
    port: int
    username: str
    password_env_var: str
    use_starttls: bool
    from_email: str
    reply_to: str


@dataclass(frozen=True)
class _EmailAccountConfig:
    name: str
    allow_read: bool
    allow_send: bool
    imap: Optional[_EmailImapConfig]
    smtp: Optional[_EmailSmtpConfig]

    def email_address(self) -> str:
        if self.smtp is not None and self.smtp.username:
            return self.smtp.username
        if self.imap is not None and self.imap.username:
            return self.imap.username
        return ""

    def from_email(self) -> str:
        if self.smtp is None:
            return ""
        return self.smtp.from_email or self.smtp.username


@dataclass(frozen=True)
class _EmailAccountsConfig:
    accounts: Dict[str, _EmailAccountConfig]
    default_account: Optional[str]
    source: str  # "env" | "yaml"
    config_path: Optional[str] = None


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if int(value) == 1:
            return True
        if int(value) == 0:
            return False
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"1", "true", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _as_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return int(s)
        except Exception:
            return None
    try:
        return int(value)
    except Exception:
        return None


_ENV_INTERP_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _interpolate_env_in_str(text: str, *, missing: set[str]) -> str:
    raw = str(text or "")
    if "${" not in raw:
        return raw

    def repl(match: re.Match[str]) -> str:
        name = str(match.group(1) or "").strip()
        default = match.group(2)
        value = os.getenv(name)
        if value is not None and str(value).strip():
            return str(value)
        if default is not None:
            return str(default)
        missing.add(name)
        return ""

    return _ENV_INTERP_RE.sub(repl, raw)


def _interpolate_env(obj: Any) -> Tuple[Any, set[str]]:
    missing: set[str] = set()

    def walk(v: Any) -> Any:
        if isinstance(v, str):
            return _interpolate_env_in_str(v, missing=missing)
        if isinstance(v, list):
            return [walk(x) for x in v]
        if isinstance(v, tuple):
            return [walk(x) for x in v]
        if isinstance(v, dict):
            return {k: walk(val) for k, val in v.items()}
        return v

    return walk(obj), missing


def _parse_email_imap_config(raw: Any) -> Tuple[Optional[_EmailImapConfig], Optional[str]]:
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, "imap must be a mapping"

    host = _as_str(raw.get("host") or raw.get("imap_host"))
    username = _as_str(raw.get("username") or raw.get("user"))
    port = _as_int(raw.get("port") or raw.get("imap_port")) or 993
    password_env_var = _as_str(raw.get("password_env_var") or raw.get("passwordEnvVar")) or "EMAIL_PASSWORD"
    mailbox = _as_str(raw.get("mailbox") or raw.get("folder")) or "INBOX"

    if not host:
        return None, "imap.host is required"
    if not username:
        return None, "imap.username is required"
    if port <= 0:
        return None, "imap.port must be a positive integer"
    if not password_env_var:
        return None, "imap.password_env_var is required"

    return (
        _EmailImapConfig(
            host=host,
            port=int(port),
            username=username,
            password_env_var=password_env_var,
            mailbox=mailbox,
        ),
        None,
    )


def _parse_email_smtp_config(raw: Any) -> Tuple[Optional[_EmailSmtpConfig], Optional[str]]:
    if raw is None:
        return None, None
    if not isinstance(raw, dict):
        return None, "smtp must be a mapping"

    host = _as_str(raw.get("host") or raw.get("smtp_host"))
    username = _as_str(raw.get("username") or raw.get("user"))
    port = _as_int(raw.get("port") or raw.get("smtp_port")) or 587
    password_env_var = _as_str(raw.get("password_env_var") or raw.get("passwordEnvVar")) or "EMAIL_PASSWORD"

    use_starttls = _as_bool(raw.get("use_starttls"))
    if use_starttls is None:
        use_starttls = _as_bool(raw.get("starttls"))
    if use_starttls is None:
        use_starttls = False if int(port) == 465 else True

    from_email = _as_str(raw.get("from_email") or raw.get("from"))
    reply_to = _as_str(raw.get("reply_to") or raw.get("replyTo"))

    if not host:
        return None, "smtp.host is required"
    if not username:
        return None, "smtp.username is required"
    if port <= 0:
        return None, "smtp.port must be a positive integer"
    if not password_env_var:
        return None, "smtp.password_env_var is required"

    return (
        _EmailSmtpConfig(
            host=host,
            port=int(port),
            username=username,
            password_env_var=password_env_var,
            use_starttls=bool(use_starttls),
            from_email=from_email,
            reply_to=reply_to,
        ),
        None,
    )


def _parse_email_account(name: str, raw: Any) -> Tuple[Optional[_EmailAccountConfig], Optional[str]]:
    if not isinstance(raw, dict):
        return None, "account entry must be a mapping"

    allow_read_raw = _as_bool(raw.get("allow_read"))
    allow_send_raw = _as_bool(raw.get("allow_send"))

    imap_cfg, imap_err = _parse_email_imap_config(raw.get("imap"))
    if imap_err is not None:
        return None, imap_err

    smtp_cfg, smtp_err = _parse_email_smtp_config(raw.get("smtp"))
    if smtp_err is not None:
        return None, smtp_err

    allow_read = bool(imap_cfg is not None) if allow_read_raw is None else bool(allow_read_raw)
    allow_send = bool(smtp_cfg is not None) if allow_send_raw is None else bool(allow_send_raw)

    if allow_read and imap_cfg is None:
        return None, "allow_read=true but imap is missing"
    if allow_send and smtp_cfg is None:
        return None, "allow_send=true but smtp is missing"

    return (
        _EmailAccountConfig(
            name=name,
            allow_read=allow_read,
            allow_send=allow_send,
            imap=imap_cfg,
            smtp=smtp_cfg,
        ),
        None,
    )


def _load_email_accounts_from_file(path: str) -> Tuple[Optional[_EmailAccountsConfig], Optional[str]]:
    raw_path = str(path or "").strip()
    if not raw_path:
        return None, "Missing email accounts config path"

    expanded = os.path.expanduser(os.path.expandvars(raw_path))
    p = Path(expanded)
    if not p.is_file():
        return None, f"ABSTRACT_EMAIL_ACCOUNTS_CONFIG is set but file not found: {expanded}"

    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        return None, f"Failed to read email accounts config ({expanded}): {e}"

    try:
        if p.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            try:
                import yaml  # type: ignore
            except Exception as e:
                return None, f"PyYAML is required to parse {expanded}: {e}"
            data = yaml.safe_load(text)
    except Exception as e:
        return None, f"Failed to parse email accounts config ({expanded}): {e}"

    if not isinstance(data, dict):
        return None, "Email accounts config must be a mapping (YAML/JSON object)"

    data2, missing = _interpolate_env(data)
    if missing:
        missing_list = ", ".join(sorted(missing))
        return None, f"Email accounts config references missing env vars: {missing_list}"
    data = data2

    accounts_raw = data.get("accounts")
    if not isinstance(accounts_raw, dict) or not accounts_raw:
        return None, "Email accounts config must contain a non-empty 'accounts' mapping"

    accounts: Dict[str, _EmailAccountConfig] = {}
    errors: list[str] = []
    for k, v in accounts_raw.items():
        name = _as_str(k)
        if not name:
            errors.append("account name must be a non-empty string")
            continue
        acc, err = _parse_email_account(name, v)
        if err is not None:
            errors.append(f"{name}: {err}")
            continue
        if acc is not None:
            accounts[name] = acc

    if errors:
        return None, "Invalid email accounts config: " + "; ".join(errors)

    default_from_env = _env_str("ABSTRACT_EMAIL_DEFAULT_ACCOUNT")
    default_from_file = _as_str(data.get("default_account"))
    default_account = (default_from_env or default_from_file or "").strip() or None
    if default_account is None and len(accounts) == 1:
        default_account = next(iter(accounts.keys()))
    if default_account is not None and default_account not in accounts:
        return None, f"Default email account '{default_account}' not found in accounts"

    return _EmailAccountsConfig(accounts=accounts, default_account=default_account, source="yaml", config_path=str(p)), None


def _load_email_accounts_from_env() -> Tuple[_EmailAccountsConfig, Optional[str]]:
    name = _env_str("ABSTRACT_EMAIL_ACCOUNT_NAME") or "default"

    smtp_defaults = _load_smtp_defaults()
    smtp_host = str(smtp_defaults.get("smtp_host") or "").strip()
    smtp_user = str(smtp_defaults.get("username") or "").strip()
    smtp_cfg: Optional[_EmailSmtpConfig] = None
    if smtp_host and smtp_user:
        smtp_cfg = _EmailSmtpConfig(
            host=smtp_host,
            port=int(smtp_defaults.get("smtp_port") or 587),
            username=smtp_user,
            password_env_var=str(smtp_defaults.get("password_env_var") or "EMAIL_PASSWORD").strip() or "EMAIL_PASSWORD",
            use_starttls=bool(smtp_defaults.get("use_starttls")),
            from_email=str(smtp_defaults.get("from_email") or "").strip(),
            reply_to=str(smtp_defaults.get("reply_to") or "").strip(),
        )

    imap_defaults = _load_imap_defaults()
    imap_host = str(imap_defaults.get("imap_host") or "").strip()
    imap_user = str(imap_defaults.get("username") or "").strip()
    imap_cfg: Optional[_EmailImapConfig] = None
    if imap_host and imap_user:
        mailbox = _env_str("ABSTRACT_EMAIL_IMAP_FOLDER") or "INBOX"
        imap_cfg = _EmailImapConfig(
            host=imap_host,
            port=int(imap_defaults.get("imap_port") or 993),
            username=imap_user,
            password_env_var=str(imap_defaults.get("password_env_var") or "EMAIL_PASSWORD").strip() or "EMAIL_PASSWORD",
            mailbox=str(mailbox or "").strip() or "INBOX",
        )

    accounts: Dict[str, _EmailAccountConfig] = {}
    if smtp_cfg is not None or imap_cfg is not None:
        accounts[name] = _EmailAccountConfig(
            name=name,
            allow_read=bool(imap_cfg is not None),
            allow_send=bool(smtp_cfg is not None),
            imap=imap_cfg,
            smtp=smtp_cfg,
        )

    default_from_env = _env_str("ABSTRACT_EMAIL_DEFAULT_ACCOUNT")
    default_account = (default_from_env or "").strip() or (name if accounts else None)
    if default_account is not None and default_account not in accounts:
        default_account = name if accounts else None

    return _EmailAccountsConfig(accounts=accounts, default_account=default_account, source="env", config_path=None), None


def _load_email_accounts_config() -> Tuple[Optional[_EmailAccountsConfig], Optional[str]]:
    path = _env_str("ABSTRACT_EMAIL_ACCOUNTS_CONFIG")
    if path:
        return _load_email_accounts_from_file(path)
    cfg, err = _load_email_accounts_from_env()
    return cfg, err


def _available_email_accounts(cfg: _EmailAccountsConfig) -> str:
    names = sorted(cfg.accounts.keys())
    return ", ".join(names)


def _select_email_account(cfg: _EmailAccountsConfig, requested: Optional[str]) -> Tuple[Optional[_EmailAccountConfig], Optional[str]]:
    if not cfg.accounts:
        return (
            None,
            "No email accounts configured. Set ABSTRACT_EMAIL_ACCOUNTS_CONFIG (YAML) or "
            "ABSTRACT_EMAIL_{IMAP,SMTP}_* env vars.",
        )

    name = _as_str(requested)
    if not name:
        if cfg.default_account:
            name = cfg.default_account
        elif len(cfg.accounts) == 1:
            name = next(iter(cfg.accounts.keys()))
        else:
            return None, f"Multiple email accounts configured; specify account. Available: {_available_email_accounts(cfg)}"

    acc = cfg.accounts.get(name)
    if acc is None:
        return None, f"Unknown email account '{name}'. Available: {_available_email_accounts(cfg)}"
    return acc, None


def _select_email_account_for(
    requested: Optional[str], *, capability: str
) -> Tuple[Optional[_EmailAccountConfig], Optional[str]]:
    cfg, err = _load_email_accounts_config()
    if err is not None:
        return None, err
    if cfg is None:
        return None, "Email account configuration is unavailable"

    acc, err2 = _select_email_account(cfg, requested)
    if err2 is not None:
        return None, err2
    if acc is None:
        return None, "Email account selection failed"

    cap = str(capability or "").strip().lower()
    if cap == "read":
        if not acc.allow_read or acc.imap is None:
            return None, f"Email account '{acc.name}' is not configured for reading (IMAP)."
    elif cap == "send":
        if not acc.allow_send or acc.smtp is None:
            return None, f"Email account '{acc.name}' is not configured for sending (SMTP)."
    else:
        return None, f"Unknown capability: {capability}"

    return acc, None


def _parse_since(value: Optional[str]) -> Tuple[Optional[datetime], Optional[str]]:
    if value is None:
        return None, None
    raw = str(value).strip()
    if not raw:
        return None, None

    # Convenience: "7" or "7d" => now - 7 days (UTC).
    m = re.fullmatch(r"(\d+)\s*d?", raw.lower())
    if m:
        days = int(m.group(1))
        return datetime.now(timezone.utc) - timedelta(days=days), None

    try:
        # Accept ISO 8601. If timezone-naive, treat as UTC to keep behavior deterministic.
        dt = datetime.fromisoformat(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt, None
    except Exception:
        return None, "Invalid since datetime; expected ISO8601 (or '7d')"


def _imap_date(dt: datetime) -> str:
    # IMAP expects English month abbreviations.
    month = _MONTH_ABBR[dt.month - 1]
    return f"{dt.day:02d}-{month}-{dt.year:04d}"


def _normalize_imap_flag(flag: str) -> str:
    raw = str(flag or "").strip()
    if not raw:
        return ""
    # Some servers/bridges may double-escape backslash-prefixed flags (e.g. "\\Seen").
    if raw.startswith("\\"):
        return "\\" + raw.lstrip("\\")
    return raw


def _imap_has_seen(flags: List[str]) -> bool:
    for f in flags:
        name = _normalize_imap_flag(f)
        if not name:
            continue
        if name.lstrip("\\").lower() == "seen":
            return True
    return False


def _extract_text_bodies(msg: Message) -> Tuple[str, str]:
    """Return (text/plain, text/html) bodies, best-effort decoded."""
    if msg is None:
        return "", ""

    text_parts: list[str] = []
    html_parts: list[str] = []

    def _decode_part(part: Message) -> str:
        payload = part.get_payload(decode=True)
        if payload is None:
            return ""
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except Exception:
            return payload.decode("utf-8", errors="replace")

    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue
            disp = part.get_content_disposition()
            if disp == "attachment":
                continue
            ctype = str(part.get_content_type() or "")
            if ctype == "text/plain":
                text = _decode_part(part).strip()
                if text:
                    text_parts.append(text)
            elif ctype == "text/html":
                html = _decode_part(part).strip()
                if html:
                    html_parts.append(html)
    else:
        ctype = str(msg.get_content_type() or "")
        if ctype == "text/plain":
            text_parts.append(_decode_part(msg).strip())
        elif ctype == "text/html":
            html_parts.append(_decode_part(msg).strip())

    return ("\n\n".join([t for t in text_parts if t]).strip(), "\n\n".join([h for h in html_parts if h]).strip())


@tool(
    description="List configured email accounts (what the runtime host has enabled).",
    tags=["comms", "email"],
    when_to_use="Use before reading/sending email to discover which accounts are configured and allowed.",
    examples=[{"description": "List available accounts", "arguments": {}}],
)
def list_email_accounts() -> Dict[str, Any]:
    cfg, err = _load_email_accounts_config()
    if err is not None:
        return {"success": False, "error": err}
    if cfg is None:
        return {"success": False, "error": "Email account configuration is unavailable"}

    accounts: list[Dict[str, Any]] = []

    def secret_available(secret_ref: str) -> bool:
        ref = str(secret_ref or "").strip()
        if not ref:
            return False
        v = os.getenv(ref)
        if v is not None and str(v).strip():
            return True
        return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", ref) is None

    for name in sorted(cfg.accounts.keys()):
        acc = cfg.accounts.get(name)
        if acc is None:
            continue
        imap_password_set = False
        smtp_password_set = False
        if acc.imap is not None:
            imap_password_set = secret_available(str(acc.imap.password_env_var or ""))
        if acc.smtp is not None:
            smtp_password_set = secret_available(str(acc.smtp.password_env_var or ""))
        accounts.append(
            {
                "account": acc.name,
                "email": acc.email_address(),
                "from_email": acc.from_email() or None,
                "can_read": bool(acc.allow_read and acc.imap is not None),
                "can_send": bool(acc.allow_send and acc.smtp is not None),
                "imap_password_set": imap_password_set if acc.imap is not None else None,
                "smtp_password_set": smtp_password_set if acc.smtp is not None else None,
            }
        )

    return {
        "success": True,
        "source": cfg.source,
        "config_path": cfg.config_path,
        "default_account": cfg.default_account,
        "accounts": accounts,
    }


@tool(
    description="Send an email from a configured account (SMTP).",
    tags=["comms", "email"],
    when_to_use=(
        "Use to send an email notification or report. The sender account is restricted to the operator-configured "
        "email accounts (it cannot be overridden by tool arguments)."
    ),
    examples=[
        {
            "description": "Send a simple text email from the default configured account",
            "arguments": {"to": "you@example.com", "subject": "Hello", "body_text": "Hi there!"},
        },
        {
            "description": "Send from a named account (multi-account config)",
            "arguments": {"account": "work", "to": "you@example.com", "subject": "Report", "body_text": "Done."},
        },
    ],
)
def send_email(
    to: Any,
    subject: str,
    *,
    account: Optional[str] = None,
    body_text: Optional[str] = None,
    body_html: Optional[str] = None,
    cc: Any = None,
    bcc: Any = None,
    timeout_s: float = 30.0,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Send an email via SMTP using only operator-configured account settings."""
    acc, err = _select_email_account_for(account, capability="send")
    if err is not None:
        return {"success": False, "error": err}
    if acc is None or acc.smtp is None:
        return {"success": False, "error": "SMTP account configuration is unavailable"}

    smtp_cfg = acc.smtp

    password, err2 = _resolve_required_env(smtp_cfg.password_env_var, label="SMTP password")
    if err2 is not None:
        return {"success": False, "error": err2, "account": acc.name}

    subject_s = str(subject or "").strip()
    if not subject_s:
        return {"success": False, "error": "subject is required", "account": acc.name}

    to_list = _coerce_str_list(to)
    cc_list = _coerce_str_list(cc)
    bcc_list = _coerce_str_list(bcc)
    if not to_list and not cc_list and not bcc_list:
        return {"success": False, "error": "At least one recipient is required (to/cc/bcc)", "account": acc.name}

    body_text_s = (body_text or "").strip()
    body_html_s = (body_html or "").strip()
    if not body_text_s and not body_html_s:
        return {"success": False, "error": "Provide body_text and/or body_html", "account": acc.name}

    sender = smtp_cfg.from_email or smtp_cfg.username
    reply_to = smtp_cfg.reply_to.strip() or None

    msg = EmailMessage()
    msg["Subject"] = subject_s
    msg["From"] = sender
    if to_list:
        msg["To"] = ", ".join(to_list)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    if reply_to:
        msg["Reply-To"] = reply_to
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()

    extra_headers = headers if isinstance(headers, dict) else None
    if extra_headers:
        for k, v in extra_headers.items():
            if not isinstance(k, str) or not k.strip():
                continue
            msg[str(k).strip()] = str(v)

    if body_text_s and body_html_s:
        msg.set_content(body_text_s)
        msg.add_alternative(body_html_s, subtype="html")
    elif body_html_s:
        msg.set_content(body_text_s or "(This email contains HTML content.)")
        msg.add_alternative(body_html_s, subtype="html")
    else:
        msg.set_content(body_text_s)

    timeout = float(timeout_s) if isinstance(timeout_s, (int, float)) else 30.0
    if timeout <= 0:
        timeout = 30.0

    try:
        if smtp_cfg.use_starttls:
            client: Any = smtplib.SMTP(smtp_cfg.host, int(smtp_cfg.port), timeout=timeout)
            try:
                client.ehlo()
                client.starttls()
                client.ehlo()
                client.login(smtp_cfg.username, password)
                client.send_message(msg, from_addr=sender, to_addrs=to_list + cc_list + bcc_list)
            finally:
                try:
                    client.quit()
                except Exception:
                    pass
        else:
            client2: Any = smtplib.SMTP_SSL(smtp_cfg.host, int(smtp_cfg.port), timeout=timeout)
            try:
                client2.login(smtp_cfg.username, password)
                client2.send_message(msg, from_addr=sender, to_addrs=to_list + cc_list + bcc_list)
            finally:
                try:
                    client2.quit()
                except Exception:
                    pass

        msg_id = str(msg.get("Message-ID") or "")
        recipients = to_list + cc_list + bcc_list
        rendered = (
            f"Sent email (account={acc.name}) from {sender} to {', '.join(recipients)} "
            f"subject={subject_s!r} message_id={msg_id}"
        )

        return {
            "success": True,
            "account": acc.name,
            "message_id": msg_id,
            "from": sender,
            "to": list(to_list),
            "cc": list(cc_list),
            "bcc": list(bcc_list),
            "rendered": rendered,
            "smtp": {
                "host": smtp_cfg.host,
                "port": int(smtp_cfg.port),
                "username": smtp_cfg.username,
                "starttls": bool(smtp_cfg.use_starttls),
            },
        }
    except Exception as e:
        return {
            "success": False,
            "account": acc.name,
            "error": str(e),
            "rendered": (
                f"Failed to send email (account={acc.name}) from {sender} to {', '.join(to_list + cc_list + bcc_list)} "
                f"subject={subject_s!r}"
            ),
            "smtp": {
                "host": smtp_cfg.host,
                "port": int(smtp_cfg.port),
                "username": smtp_cfg.username,
                "starttls": bool(smtp_cfg.use_starttls),
            },
        }


@tool(
    description="List recent emails from a configured IMAP mailbox (supports since + read/unread filters).",
    tags=["comms", "email"],
    when_to_use="Use to fetch a digest of recent emails (subject/from/date/flags) for review or routing.",
    examples=[
        {
            "description": "List unread emails from the default configured account (last 7 days)",
            "arguments": {"since": "7d", "status": "unread", "limit": 10},
        },
        {
            "description": "List unread emails from a named account (multi-account config)",
            "arguments": {"account": "work", "since": "7d", "status": "unread", "limit": 10},
        },
    ],
)
def list_emails(
    *,
    account: Optional[str] = None,
    mailbox: Optional[str] = None,
    since: Optional[str] = None,
    status: str = "all",
    limit: int = 20,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """List email headers from an IMAP mailbox using only operator-configured account settings."""
    acc, err = _select_email_account_for(account, capability="read")
    if err is not None:
        return {"success": False, "error": err}
    if acc is None or acc.imap is None:
        return {"success": False, "error": "IMAP account configuration is unavailable"}

    imap_cfg = acc.imap

    password, err2 = _resolve_required_env(imap_cfg.password_env_var, label="IMAP password")
    if err2 is not None:
        return {"success": False, "error": err2, "account": acc.name}

    mailbox2 = str(mailbox or "").strip() or str(imap_cfg.mailbox or "").strip() or "INBOX"

    try:
        limit_i = int(limit or 0)
    except Exception:
        limit_i = 0
    if limit_i <= 0:
        limit_i = 20

    if int(imap_cfg.port) <= 0:
        return {"success": False, "error": "imap_port must be a positive integer", "account": acc.name}

    dt_since, dt_err = _parse_since(since)
    if dt_err is not None:
        return {"success": False, "error": dt_err, "account": acc.name}

    status_norm = str(status or "").strip().lower() or "all"
    if status_norm not in {"all", "unread", "read"}:
        return {"success": False, "error": "status must be one of: all, unread, read", "account": acc.name}

    timeout = float(timeout_s) if isinstance(timeout_s, (int, float)) else 30.0
    if timeout <= 0:
        timeout = 30.0

    client: Optional[imaplib.IMAP4_SSL] = None
    try:
        client = imaplib.IMAP4_SSL(imap_cfg.host, int(imap_cfg.port))
        try:
            if getattr(client, "sock", None) is not None:
                client.sock.settimeout(timeout)  # type: ignore[attr-defined]
        except Exception:
            pass

        client.login(imap_cfg.username, password)
        typ, _ = client.select(mailbox2, readonly=True)
        if typ != "OK":
            return {"success": False, "error": f"Failed to select mailbox: {mailbox2}", "account": acc.name}

        criteria: list[str] = []
        if dt_since is not None:
            criteria.append(f"SINCE { _imap_date(dt_since) }")
        if status_norm == "unread":
            criteria.append("UNSEEN")
        elif status_norm == "read":
            criteria.append("SEEN")
        search_query = " ".join(criteria) if criteria else "ALL"

        typ2, data = client.uid("search", None, search_query)
        if typ2 != "OK" or not data:
            return {"success": False, "error": "IMAP search failed", "account": acc.name, "mailbox": mailbox2}

        raw_uids = data[0] if isinstance(data, list) and data else b""
        if not isinstance(raw_uids, (bytes, bytearray)):
            raw_uids = str(raw_uids).encode("utf-8", errors="replace")
        uids = [u.decode("utf-8", errors="replace") for u in raw_uids.split() if u]

        # IMAP SEARCH tends to return ascending order; return newest first.
        uids = list(reversed(uids))[:limit_i]

        messages: list[Dict[str, Any]] = []
        for uid in uids:
            typ3, fetched = client.uid(
                "fetch",
                uid,
                "(FLAGS RFC822.SIZE BODY.PEEK[HEADER.FIELDS (FROM TO CC BCC SUBJECT DATE MESSAGE-ID)])",
            )
            if typ3 != "OK" or not fetched:
                continue
            # fetched is typically a list of tuples + trailing b')' entry.
            header_bytes: Optional[bytes] = None
            flags: list[str] = []
            size: Optional[int] = None
            for item in fetched:
                if not isinstance(item, tuple) or len(item) < 2:
                    continue
                meta, payload = item[0], item[1]
                if isinstance(payload, (bytes, bytearray)) and payload:
                    header_bytes = bytes(payload)
                if isinstance(meta, (bytes, bytearray)):
                    try:
                        flags_bytes = imaplib.ParseFlags(meta)
                        flags = [_normalize_imap_flag(fb.decode("utf-8", errors="replace")) for fb in flags_bytes]
                    except Exception:
                        flags = []
                    m = re.search(rb"RFC822\.SIZE\s+(\d+)", meta)
                    if m:
                        try:
                            size = int(m.group(1))
                        except Exception:
                            size = None
            if header_bytes is None:
                continue

            msg = email.message_from_bytes(header_bytes)
            subject_v = _decode_mime_header(msg.get("Subject"))
            from_v = _decode_mime_header(msg.get("From"))
            to_v = _decode_mime_header(msg.get("To"))
            date_v = _decode_mime_header(msg.get("Date"))
            message_id = _decode_mime_header(msg.get("Message-ID"))

            messages.append(
                {
                    "uid": str(uid),
                    "message_id": message_id,
                    "subject": subject_v,
                    "from": from_v,
                    "to": to_v,
                    "date": date_v,
                    "flags": flags,
                    "seen": _imap_has_seen(flags),
                    "size": size,
                }
            )

        unread = sum(1 for m in messages if not bool(m.get("seen")))
        read = len(messages) - unread

        return {
            "success": True,
            "account": acc.name,
            "mailbox": mailbox2,
            "filter": {"since": dt_since.isoformat() if dt_since else None, "status": status_norm, "limit": limit_i},
            "counts": {"returned": len(messages), "unread": unread, "read": read},
            "messages": messages,
        }
    except Exception as e:
        return {"success": False, "account": acc.name, "error": str(e), "mailbox": mailbox2}
    finally:
        if client is not None:
            try:
                client.logout()
            except Exception:
                pass


@tool(
    description="Read a specific email by IMAP UID (returns decoded subject/headers and text/html body).",
    tags=["comms", "email"],
    when_to_use="Use after list_emails to fetch the full content of a specific email.",
    examples=[
        {
            "description": "Read an email by UID from the default configured account",
            "arguments": {"uid": "12345"},
        },
        {"description": "Read an email by UID from a named account", "arguments": {"account": "work", "uid": "12345"}},
    ],
)
def read_email(
    *,
    uid: str,
    account: Optional[str] = None,
    mailbox: Optional[str] = None,
    timeout_s: float = 30.0,
    max_body_chars: int = 20000,
) -> Dict[str, Any]:
    """Read a single email by UID from an IMAP mailbox (best-effort; does not mark as read)."""
    acc, err = _select_email_account_for(account, capability="read")
    if err is not None:
        return {"success": False, "error": err}
    if acc is None or acc.imap is None:
        return {"success": False, "error": "IMAP account configuration is unavailable"}

    imap_cfg = acc.imap

    password, err2 = _resolve_required_env(imap_cfg.password_env_var, label="IMAP password")
    if err2 is not None:
        return {"success": False, "error": err2, "account": acc.name}

    mailbox2 = str(mailbox or "").strip() or str(imap_cfg.mailbox or "").strip() or "INBOX"
    uid2 = str(uid or "").strip()
    if not uid2:
        return {"success": False, "error": "uid is required", "account": acc.name}

    if int(imap_cfg.port) <= 0:
        return {"success": False, "error": "imap_port must be a positive integer", "account": acc.name}

    try:
        max_chars = int(max_body_chars or 0)
    except Exception:
        max_chars = 0
    if max_chars <= 0:
        max_chars = 20000

    timeout = float(timeout_s) if isinstance(timeout_s, (int, float)) else 30.0
    if timeout <= 0:
        timeout = 30.0

    client: Optional[imaplib.IMAP4_SSL] = None
    try:
        client = imaplib.IMAP4_SSL(imap_cfg.host, int(imap_cfg.port))
        try:
            if getattr(client, "sock", None) is not None:
                client.sock.settimeout(timeout)  # type: ignore[attr-defined]
        except Exception:
            pass

        client.login(imap_cfg.username, password)
        typ, _ = client.select(mailbox2, readonly=True)
        if typ != "OK":
            return {"success": False, "error": f"Failed to select mailbox: {mailbox2}", "account": acc.name}

        typ2, fetched = client.uid("fetch", uid2, "(FLAGS BODY.PEEK[])")
        if typ2 != "OK" or not fetched:
            return {"success": False, "error": f"Email not found for uid={uid2}", "account": acc.name, "mailbox": mailbox2}

        raw_bytes: Optional[bytes] = None
        flags: list[str] = []
        for item in fetched:
            if not isinstance(item, tuple) or len(item) < 2:
                continue
            meta, payload = item[0], item[1]
            if isinstance(payload, (bytes, bytearray)) and payload:
                raw_bytes = bytes(payload)
            if isinstance(meta, (bytes, bytearray)):
                try:
                    flags_bytes = imaplib.ParseFlags(meta)
                    flags = [_normalize_imap_flag(fb.decode("utf-8", errors="replace")) for fb in flags_bytes]
                except Exception:
                    flags = []

        if raw_bytes is None:
            return {"success": False, "error": f"Failed to fetch email bytes for uid={uid2}", "account": acc.name, "mailbox": mailbox2}

        msg = email.message_from_bytes(raw_bytes)
        subject_v = _decode_mime_header(msg.get("Subject"))
        from_v = _decode_mime_header(msg.get("From"))
        to_v = _decode_mime_header(msg.get("To"))
        cc_v = _decode_mime_header(msg.get("Cc"))
        date_v = _decode_mime_header(msg.get("Date"))
        message_id = _decode_mime_header(msg.get("Message-ID"))

        body_text, body_html = _extract_text_bodies(msg)
        if len(body_text) > max_chars:
            body_text = body_text[:max_chars] + "…"
        if len(body_html) > max_chars:
            body_html = body_html[:max_chars] + "…"

        attachments: list[Dict[str, Any]] = []
        if msg.is_multipart():
            for part in msg.walk():
                if part.is_multipart():
                    continue
                disp = part.get_content_disposition()
                filename = part.get_filename()
                if disp == "attachment" or filename:
                    attachments.append(
                        {
                            "filename": _decode_mime_header(filename),
                            "content_type": str(part.get_content_type() or ""),
                        }
                    )

        return {
            "success": True,
            "account": acc.name,
            "mailbox": mailbox2,
            "uid": uid2,
            "message_id": message_id,
            "subject": subject_v,
            "from": from_v,
            "to": to_v,
            "cc": cc_v,
            "date": date_v,
            "flags": flags,
            "seen": _imap_has_seen(flags),
            "body_text": body_text,
            "body_html": body_html,
            "attachments": attachments,
        }
    except Exception as e:
        return {"success": False, "account": acc.name, "error": str(e), "mailbox": mailbox2, "uid": uid2}
    finally:
        if client is not None:
            try:
                client.logout()
            except Exception:
                pass


def _whatsapp_prefix(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    return raw if raw.lower().startswith("whatsapp:") else f"whatsapp:{raw}"


@dataclass(frozen=True)
class _TwilioCreds:
    account_sid: str
    auth_token: str


def _twilio_creds(*, account_sid_env_var: str, auth_token_env_var: str) -> Tuple[Optional[_TwilioCreds], Optional[str]]:
    sid, err1 = _resolve_required_env(account_sid_env_var, label="Twilio account SID")
    if err1 is not None:
        return None, err1
    tok, err2 = _resolve_required_env(auth_token_env_var, label="Twilio auth token")
    if err2 is not None:
        return None, err2
    return _TwilioCreds(account_sid=sid, auth_token=tok), None


def _twilio_base_url(account_sid: str) -> str:
    sid = str(account_sid or "").strip()
    return f"https://api.twilio.com/2010-04-01/Accounts/{sid}"


@tool(
    description="Send a WhatsApp message via a provider API (default: Twilio REST).",
    tags=["comms", "whatsapp"],
    when_to_use="Use to send a WhatsApp message notification (credentials resolved from env vars).",
    examples=[
        {
            "description": "Send via Twilio WhatsApp",
            "arguments": {
                "to": "+15551234567",
                "from_number": "+15557654321",
                "body": "Hello from AbstractFramework",
            },
        }
    ],
)
def send_whatsapp_message(
    to: str,
    from_number: str,
    body: str,
    *,
    provider: str = "twilio",
    account_sid_env_var: str = "TWILIO_ACCOUNT_SID",
    auth_token_env_var: str = "TWILIO_AUTH_TOKEN",
    timeout_s: float = 30.0,
    media_urls: Any = None,
) -> Dict[str, Any]:
    """Send a WhatsApp message (Twilio-backed v1)."""
    provider_norm = str(provider or "").strip().lower() or "twilio"
    if provider_norm != "twilio":
        return {"success": False, "error": f"Unsupported WhatsApp provider: {provider_norm} (v1 supports: twilio)"}

    creds, err = _twilio_creds(account_sid_env_var=account_sid_env_var, auth_token_env_var=auth_token_env_var)
    if err is not None:
        return {"success": False, "error": err}

    to_norm = _whatsapp_prefix(to)
    from_norm = _whatsapp_prefix(from_number)
    body_norm = str(body or "").strip()
    if not to_norm:
        return {"success": False, "error": "to is required"}
    if not from_norm:
        return {"success": False, "error": "from_number is required"}
    if not body_norm:
        return {"success": False, "error": "body is required"}

    timeout = float(timeout_s) if isinstance(timeout_s, (int, float)) else 30.0
    if timeout <= 0:
        timeout = 30.0

    media_list = _coerce_str_list(media_urls)

    try:
        import requests  # type: ignore
    except Exception as e:
        return {"success": False, "error": f"requests is required for WhatsApp tools: {e}"}

    url = f"{_twilio_base_url(creds.account_sid)}/Messages.json"
    data: Dict[str, Any] = {"To": to_norm, "From": from_norm, "Body": body_norm}
    # Twilio supports multiple MediaUrl fields (repeated keys); requests supports sequences of tuples.
    request_data: Any = data
    if media_list:
        pairs: list[tuple[str, str]] = [(k, str(v)) for k, v in data.items()]
        for mu in media_list:
            pairs.append(("MediaUrl", mu))
        request_data = pairs

    try:
        resp = requests.post(url, data=request_data, auth=(creds.account_sid, creds.auth_token), timeout=timeout)
    except Exception as e:
        return {"success": False, "error": str(e), "provider": provider_norm}

    try:
        payload = resp.json()
    except Exception:
        payload = {"raw": (resp.text or "").strip()}

    if not getattr(resp, "ok", False):
        return {
            "success": False,
            "error": str(payload.get("message") or payload.get("raw") or f"HTTP {resp.status_code}"),
            "status_code": int(getattr(resp, "status_code", 0) or 0),
            "provider": provider_norm,
        }

    sid = str(payload.get("sid") or "")
    return {
        "success": True,
        "provider": provider_norm,
        "sid": sid,
        "status": payload.get("status"),
        "to": payload.get("to") or to_norm,
        "from": payload.get("from") or from_norm,
    }


@tool(
    description="List recent WhatsApp messages via a provider API (default: Twilio REST).",
    tags=["comms", "whatsapp"],
    when_to_use="Use to fetch a digest of recent WhatsApp messages for review (since + direction filters).",
    examples=[
        {
            "description": "List inbound messages since 7 days ago",
            "arguments": {"since": "7d", "direction": "inbound"},
        }
    ],
)
def list_whatsapp_messages(
    *,
    provider: str = "twilio",
    account_sid_env_var: str = "TWILIO_ACCOUNT_SID",
    auth_token_env_var: str = "TWILIO_AUTH_TOKEN",
    to: Optional[str] = None,
    from_number: Optional[str] = None,
    since: Optional[str] = None,
    limit: int = 20,
    direction: str = "all",
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """List recent WhatsApp messages (Twilio-backed v1)."""
    provider_norm = str(provider or "").strip().lower() or "twilio"
    if provider_norm != "twilio":
        return {"success": False, "error": f"Unsupported WhatsApp provider: {provider_norm} (v1 supports: twilio)"}

    creds, err = _twilio_creds(account_sid_env_var=account_sid_env_var, auth_token_env_var=auth_token_env_var)
    if err is not None:
        return {"success": False, "error": err}

    try:
        limit_i = int(limit or 0)
    except Exception:
        limit_i = 0
    if limit_i <= 0:
        limit_i = 20

    dt_since, dt_err = _parse_since(since)
    if dt_err is not None:
        return {"success": False, "error": dt_err}

    direction_norm = str(direction or "").strip().lower() or "all"
    if direction_norm not in {"all", "inbound", "outbound"}:
        return {"success": False, "error": "direction must be one of: all, inbound, outbound"}

    timeout = float(timeout_s) if isinstance(timeout_s, (int, float)) else 30.0
    if timeout <= 0:
        timeout = 30.0

    to_norm = _whatsapp_prefix(to or "") if to else None
    from_norm = _whatsapp_prefix(from_number or "") if from_number else None

    try:
        import requests  # type: ignore
    except Exception as e:
        return {"success": False, "error": f"requests is required for WhatsApp tools: {e}"}

    url = f"{_twilio_base_url(creds.account_sid)}/Messages.json"
    params: Dict[str, Any] = {"PageSize": limit_i}
    if to_norm:
        params["To"] = to_norm
    if from_norm:
        params["From"] = from_norm
    if dt_since is not None:
        # Twilio supports DateSent> filters (date-only).
        params["DateSent>"] = dt_since.date().isoformat()

    try:
        resp = requests.get(url, params=params, auth=(creds.account_sid, creds.auth_token), timeout=timeout)
    except Exception as e:
        return {"success": False, "error": str(e), "provider": provider_norm}

    try:
        payload = resp.json()
    except Exception:
        payload = {"raw": (resp.text or "").strip()}

    if not getattr(resp, "ok", False):
        return {
            "success": False,
            "error": str(payload.get("message") or payload.get("raw") or f"HTTP {resp.status_code}"),
            "status_code": int(getattr(resp, "status_code", 0) or 0),
            "provider": provider_norm,
        }

    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raw_messages = []

    out: list[Dict[str, Any]] = []
    for m in raw_messages[:limit_i]:
        if not isinstance(m, dict):
            continue
        direction_val = str(m.get("direction") or "")
        if direction_norm == "inbound" and not direction_val.startswith("inbound"):
            continue
        if direction_norm == "outbound" and not direction_val.startswith("outbound"):
            continue

        body_text = str(m.get("body") or "")
        body_text = preview_text(body_text, max_chars=500)

        out.append(
            {
                "sid": str(m.get("sid") or ""),
                "status": m.get("status"),
                "direction": direction_val,
                "from": m.get("from"),
                "to": m.get("to"),
                "date_sent": m.get("date_sent"),
                "date_created": m.get("date_created"),
                "body": body_text,
            }
        )

    return {
        "success": True,
        "provider": provider_norm,
        "filter": {
            "since": dt_since.isoformat() if dt_since else None,
            "direction": direction_norm,
            "to": to_norm,
            "from": from_norm,
            "limit": limit_i,
        },
        "messages": out,
        "counts": {"returned": len(out)},
    }


@tool(
    description="Read a specific WhatsApp message by provider message id (default: Twilio SID).",
    tags=["comms", "whatsapp"],
    when_to_use="Use after list_whatsapp_messages to fetch full details of one message.",
    examples=[
        {"description": "Read a message by SID", "arguments": {"message_id": "SMxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}},
    ],
)
def read_whatsapp_message(
    message_id: str,
    *,
    provider: str = "twilio",
    account_sid_env_var: str = "TWILIO_ACCOUNT_SID",
    auth_token_env_var: str = "TWILIO_AUTH_TOKEN",
    timeout_s: float = 30.0,
    max_body_chars: int = 2000,
) -> Dict[str, Any]:
    """Read a WhatsApp message by id (Twilio-backed v1)."""
    provider_norm = str(provider or "").strip().lower() or "twilio"
    if provider_norm != "twilio":
        return {"success": False, "error": f"Unsupported WhatsApp provider: {provider_norm} (v1 supports: twilio)"}

    creds, err = _twilio_creds(account_sid_env_var=account_sid_env_var, auth_token_env_var=auth_token_env_var)
    if err is not None:
        return {"success": False, "error": err}

    mid = str(message_id or "").strip()
    if not mid:
        return {"success": False, "error": "message_id is required"}

    timeout = float(timeout_s) if isinstance(timeout_s, (int, float)) else 30.0
    if timeout <= 0:
        timeout = 30.0

    try:
        max_chars = int(max_body_chars or 0)
    except Exception:
        max_chars = 0
    if max_chars <= 0:
        max_chars = 2000

    try:
        import requests  # type: ignore
    except Exception as e:
        return {"success": False, "error": f"requests is required for WhatsApp tools: {e}"}

    url = f"{_twilio_base_url(creds.account_sid)}/Messages/{mid}.json"
    try:
        resp = requests.get(url, auth=(creds.account_sid, creds.auth_token), timeout=timeout)
    except Exception as e:
        return {"success": False, "error": str(e), "provider": provider_norm}

    try:
        payload = resp.json()
    except Exception:
        payload = {"raw": (resp.text or "").strip()}

    if not getattr(resp, "ok", False):
        return {
            "success": False,
            "error": str(payload.get("message") or payload.get("raw") or f"HTTP {resp.status_code}"),
            "status_code": int(getattr(resp, "status_code", 0) or 0),
            "provider": provider_norm,
        }

    body_text = str(payload.get("body") or "")
    if len(body_text) > max_chars:
        body_text = body_text[:max_chars] + "…"

    return {
        "success": True,
        "provider": provider_norm,
        "sid": str(payload.get("sid") or mid),
        "status": payload.get("status"),
        "direction": payload.get("direction"),
        "from": payload.get("from"),
        "to": payload.get("to"),
        "date_sent": payload.get("date_sent"),
        "date_created": payload.get("date_created"),
        "body": body_text,
        "raw": payload,
    }
