"""Typed email settings: the account (non-secret), its secret, limits and cursors.

Serialised shapes are plain JSON objects (the AbstractCore config store's `email` section
and the gateway's per-user settings use the same `to_dict()` / `from_dict()`).

Secrets (`EmailSecret`) never serialise through these helpers: they are sealed by
`vault.SecretVault`, and their `repr()` redacts every value.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .errors import EmailInvalidSettings
from .policy import normalize_address

SECURITY_MODES = ("ssl", "starttls")
AUTH_KINDS = ("password", "oauth2")
OAUTH_PROVIDERS = ("google", "microsoft", "custom")

DEFAULT_PORTS = {
    ("imap", "ssl"): 993,
    ("imap", "starttls"): 143,
    ("smtp", "ssl"): 465,
    ("smtp", "starttls"): 587,
}

DEFAULT_PER_HOUR = 20
DEFAULT_PER_DAY = 100
MAX_LIMIT = 100_000


def _invalid(what: str, fix: str) -> EmailInvalidSettings:
    return EmailInvalidSettings(what, fix)


def _check_host(host: str, label: str) -> str:
    h = str(host or "").strip()
    if not h:
        raise _invalid(f"The {label} host is missing.", f"Give the {label} host name (for example imap.example.test).")
    if any(c.isspace() for c in h) or "/" in h or "@" in h:
        raise _invalid(
            f"The {label} host {h!r} is not a host name.",
            f"Give only the {label} host name, without a scheme, path, user or spaces.",
        )
    return h


def _check_port(port: Any, label: str) -> int:
    try:
        p = int(port)
    except (TypeError, ValueError):
        raise _invalid(f"The {label} port {port!r} is not a number.", f"Give the {label} port as a number (1-65535).")
    if not 1 <= p <= 65535:
        raise _invalid(f"The {label} port {p} is out of range.", f"Give the {label} port as a number (1-65535).")
    return p


def _check_security(security: Any, label: str) -> str:
    s = str(security or "").strip().lower()
    if s not in SECURITY_MODES:
        raise _invalid(
            f"The {label} security mode {security!r} is not one of: ssl, starttls.",
            f"Use ssl (implicit TLS) or starttls for {label}; unencrypted connections are not supported.",
        )
    return s


def _check_ca_file(ca_file: Any, label: str) -> str:
    path = str(ca_file or "").strip()
    if not path:
        return ""
    if not Path(os.path.expanduser(path)).is_file():
        raise _invalid(
            f"The {label} CA file {path!r} does not exist.",
            "Give the path of the PEM file of the CA that signed the server certificate, or leave it empty "
            "to use the system trust store.",
        )
    return path


def _check_address(address: Any, label: str) -> str:
    a = str(address or "").strip()
    try:
        normalize_address(a)
    except ValueError:
        raise _invalid(f"The {label} {a!r} is not a valid email address.", f"Give the {label} as name@example.test.")
    return a


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    security: str
    ca_file: str = ""


@dataclass(frozen=True)
class ImapSettings(ServerSettings):
    folder: str = "INBOX"

    @classmethod
    def build(cls, host: str, *, port: Any = None, security: str = "ssl", folder: str = "INBOX", ca_file: str = "") -> "ImapSettings":
        sec = _check_security(security, "IMAP")
        p = _check_port(port if port not in (None, "", 0) else DEFAULT_PORTS[("imap", sec)], "IMAP")
        f = str(folder or "").strip() or "INBOX"
        return cls(host=_check_host(host, "IMAP"), port=p, security=sec, ca_file=_check_ca_file(ca_file, "IMAP"), folder=f)

    def to_dict(self) -> Dict[str, Any]:
        return {"host": self.host, "port": self.port, "security": self.security, "folder": self.folder, "ca_file": self.ca_file}

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["ImapSettings"]:
        if not isinstance(raw, dict) or not str(raw.get("host") or "").strip():
            return None
        return cls.build(
            str(raw.get("host") or ""),
            port=raw.get("port"),
            security=str(raw.get("security") or "ssl"),
            folder=str(raw.get("folder") or "INBOX"),
            ca_file=str(raw.get("ca_file") or ""),
        )


@dataclass(frozen=True)
class SmtpSettings(ServerSettings):
    @classmethod
    def build(cls, host: str, *, port: Any = None, security: str = "ssl", ca_file: str = "") -> "SmtpSettings":
        sec = _check_security(security, "SMTP")
        p = _check_port(port if port not in (None, "", 0) else DEFAULT_PORTS[("smtp", sec)], "SMTP")
        return cls(host=_check_host(host, "SMTP"), port=p, security=sec, ca_file=_check_ca_file(ca_file, "SMTP"))

    def to_dict(self) -> Dict[str, Any]:
        return {"host": self.host, "port": self.port, "security": self.security, "ca_file": self.ca_file}

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["SmtpSettings"]:
        if not isinstance(raw, dict) or not str(raw.get("host") or "").strip():
            return None
        return cls.build(
            str(raw.get("host") or ""),
            port=raw.get("port"),
            security=str(raw.get("security") or "ssl"),
            ca_file=str(raw.get("ca_file") or ""),
        )


@dataclass(frozen=True)
class OAuthSettings:
    """Non-secret OAuth2 client settings. The client secret and tokens live in `EmailSecret`."""

    provider: str
    client_id: str
    token_endpoint: str
    scopes: Tuple[str, ...]
    authorization_endpoint: str = ""
    device_authorization_endpoint: str = ""
    tenant: str = ""
    client_source: str = "own"  # "own" (bring your own client) | "builtin" (AbstractFramework's)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "client_id": self.client_id,
            "client_source": self.client_source,
            "token_endpoint": self.token_endpoint,
            "authorization_endpoint": self.authorization_endpoint,
            "device_authorization_endpoint": self.device_authorization_endpoint,
            "scopes": list(self.scopes),
            "tenant": self.tenant,
        }

    @classmethod
    def build(
        cls,
        provider: str,
        client_id: str,
        *,
        token_endpoint: str = "",
        authorization_endpoint: str = "",
        device_authorization_endpoint: str = "",
        scopes: Optional[List[str]] = None,
        tenant: str = "",
        client_source: str = "own",
    ) -> "OAuthSettings":
        from .oauth import provider_preset

        prov = str(provider or "").strip().lower()
        if prov not in OAUTH_PROVIDERS:
            raise _invalid(
                f"The OAuth provider {provider!r} is not one of: google, microsoft, custom.",
                "Use --oauth google, --oauth microsoft, or --oauth custom with explicit endpoints.",
            )
        cid = str(client_id or "").strip()
        if not cid:
            raise _invalid(
                "The OAuth client id is missing.",
                "Give the client id of the OAuth client registered with the provider (--client-id <value>).",
            )
        preset = provider_preset(prov, tenant=tenant) if prov != "custom" else {}
        tok = str(token_endpoint or preset.get("token_endpoint") or "").strip()
        auth = str(authorization_endpoint or preset.get("authorization_endpoint") or "").strip()
        dev = str(device_authorization_endpoint or preset.get("device_authorization_endpoint") or "").strip()
        sc = tuple(str(s).strip() for s in (scopes or preset.get("scopes") or []) if str(s).strip())
        for name, url in (("token", tok), ("authorization", auth), ("device authorization", dev)):
            if url and not url.lower().startswith("https://"):
                raise _invalid(
                    f"The OAuth {name} endpoint must use https.",
                    "Give the provider's https endpoint; tokens are never sent over plain HTTP.",
                )
        if not tok:
            raise _invalid("The OAuth token endpoint is missing.", "Give --token-endpoint <https URL> for a custom provider.")
        if not sc:
            raise _invalid("The OAuth scopes are missing.", "Give --scope <scope> for a custom provider (repeatable).")
        source = str(client_source or "own").strip().lower()
        if source not in ("own", "builtin"):
            raise _invalid(f"The OAuth client source {client_source!r} is not one of: own, builtin.", "Use own or builtin.")
        return cls(
            provider=prov,
            client_id=cid,
            token_endpoint=tok,
            scopes=sc,
            authorization_endpoint=auth,
            device_authorization_endpoint=dev,
            tenant=str(tenant or preset.get("tenant") or "").strip(),
            client_source=source,
        )

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["OAuthSettings"]:
        if not isinstance(raw, dict) or not raw:
            return None
        scopes = raw.get("scopes")
        return cls.build(
            str(raw.get("provider") or ""),
            str(raw.get("client_id") or ""),
            token_endpoint=str(raw.get("token_endpoint") or ""),
            authorization_endpoint=str(raw.get("authorization_endpoint") or ""),
            device_authorization_endpoint=str(raw.get("device_authorization_endpoint") or ""),
            scopes=[str(s) for s in scopes] if isinstance(scopes, (list, tuple)) else None,
            tenant=str(raw.get("tenant") or ""),
            client_source=str(raw.get("client_source") or "own"),
        )


@dataclass(frozen=True)
class EmailAccount:
    """The non-secret account: who sends, where to connect, how to sign in."""

    address: str
    username: str
    imap: Optional[ImapSettings]
    smtp: Optional[SmtpSettings]
    display_name: str = ""
    auth_kind: str = "password"
    oauth: Optional[OAuthSettings] = None

    @classmethod
    def build(
        cls,
        *,
        address: str,
        username: str = "",
        imap: Optional[ImapSettings] = None,
        smtp: Optional[SmtpSettings] = None,
        display_name: str = "",
        auth_kind: str = "password",
        oauth: Optional[OAuthSettings] = None,
    ) -> "EmailAccount":
        addr = _check_address(address, "sender address")
        user = str(username or "").strip() or addr
        if any(c in "\r\n" for c in user):
            raise _invalid("The user name contains a line break.", "Give the user name on one line.")
        name = str(display_name or "").strip()
        if any(c in "\r\n" for c in name):
            raise _invalid("The display name contains a line break.", "Give the display name on one line.")
        kind = str(auth_kind or "password").strip().lower()
        if kind not in AUTH_KINDS:
            raise _invalid(
                f"The sign-in method {auth_kind!r} is not one of: password, oauth2.",
                "Use a password (app password) or OAuth2 (--oauth google|microsoft).",
            )
        if kind == "oauth2" and oauth is None:
            raise _invalid("OAuth2 sign-in needs the OAuth client settings.", "Connect with --oauth <provider> --client-id <value>.")
        if imap is None and smtp is None:
            raise _invalid(
                "The account has neither IMAP (receive) nor SMTP (send) settings.",
                "Give at least --imap-host (to read mail) or --smtp-host (to send mail).",
            )
        return cls(
            address=addr,
            username=user,
            imap=imap,
            smtp=smtp,
            display_name=name,
            auth_kind=kind,
            oauth=oauth if kind == "oauth2" else None,
        )

    @property
    def can_read(self) -> bool:
        return self.imap is not None

    @property
    def can_send(self) -> bool:
        return self.smtp is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "display_name": self.display_name,
            "username": self.username,
            "auth_kind": self.auth_kind,
            "imap": self.imap.to_dict() if self.imap else {},
            "smtp": self.smtp.to_dict() if self.smtp else {},
            "oauth": self.oauth.to_dict() if self.oauth else {},
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["EmailAccount"]:
        if not isinstance(raw, dict) or not str(raw.get("address") or "").strip():
            return None
        return cls.build(
            address=str(raw.get("address") or ""),
            username=str(raw.get("username") or ""),
            imap=ImapSettings.from_dict(raw.get("imap")),
            smtp=SmtpSettings.from_dict(raw.get("smtp")),
            display_name=str(raw.get("display_name") or ""),
            auth_kind=str(raw.get("auth_kind") or "password"),
            oauth=OAuthSettings.from_dict(raw.get("oauth")),
        )


class EmailSecret:
    """The secret half of an account. Held in memory only; `repr()`/`str()` redact.

    Password sign-in: `password`. OAuth2 sign-in: `refresh_token` (+ cached `access_token`
    and `expires_at`) and the OAuth client's `client_secret` (empty for public clients).
    """

    __slots__ = ("password", "refresh_token", "access_token", "expires_at", "client_secret")

    def __init__(
        self,
        password: str = "",
        *,
        refresh_token: str = "",
        access_token: str = "",
        expires_at: float = 0.0,
        client_secret: str = "",
    ) -> None:
        self.password = str(password or "")
        self.refresh_token = str(refresh_token or "")
        self.access_token = str(access_token or "")
        try:
            self.expires_at = float(expires_at or 0.0)
        except (TypeError, ValueError):
            self.expires_at = 0.0
        self.client_secret = str(client_secret or "")

    def __repr__(self) -> str:
        parts = [f"{name}={'«set»' if getattr(self, name) else '«empty»'}" for name in ("password", "refresh_token", "access_token", "client_secret")]
        return f"EmailSecret({', '.join(parts)})"

    __str__ = __repr__

    def __reduce__(self):  # pragma: no cover - defensive: never pickle a secret by accident
        raise TypeError("EmailSecret is not picklable")

    @property
    def is_set(self) -> bool:
        return bool(self.password or self.refresh_token or self.access_token)

    def access_token_valid(self, *, skew_s: float = 60.0, now: Optional[float] = None) -> bool:
        t = time.time() if now is None else now
        return bool(self.access_token) and self.expires_at > t + skew_s

    def sealed_payload(self) -> Dict[str, Any]:
        """The dict `SecretVault` seals. Never log or return it."""

        return {
            "password": self.password,
            "refresh_token": self.refresh_token,
            "access_token": self.access_token,
            "expires_at": self.expires_at,
            "client_secret": self.client_secret,
        }

    @classmethod
    def from_sealed_payload(cls, raw: Dict[str, Any]) -> "EmailSecret":
        raw = raw if isinstance(raw, dict) else {}
        return cls(
            str(raw.get("password") or ""),
            refresh_token=str(raw.get("refresh_token") or ""),
            access_token=str(raw.get("access_token") or ""),
            expires_at=raw.get("expires_at") or 0.0,
            client_secret=str(raw.get("client_secret") or ""),
        )


@dataclass(frozen=True)
class SendLimits:
    """Messages per rolling hour and per rolling day. 0 means no sending in that window."""

    per_hour: int = DEFAULT_PER_HOUR
    per_day: int = DEFAULT_PER_DAY

    @classmethod
    def build(cls, per_hour: Any = DEFAULT_PER_HOUR, per_day: Any = DEFAULT_PER_DAY) -> "SendLimits":
        out = []
        for label, value in (("per-hour", per_hour), ("per-day", per_day)):
            try:
                v = int(value)
            except (TypeError, ValueError):
                raise _invalid(f"The {label} send limit {value!r} is not a number.", f"Give the {label} limit as a whole number (0-{MAX_LIMIT}).")
            if not 0 <= v <= MAX_LIMIT:
                raise _invalid(f"The {label} send limit {v} is out of range.", f"Give the {label} limit as a whole number (0-{MAX_LIMIT}).")
            out.append(v)
        return cls(per_hour=out[0], per_day=out[1])

    @classmethod
    def from_dict(cls, raw: Any) -> "SendLimits":
        if not isinstance(raw, dict) or not raw:
            return cls()
        return cls.build(raw.get("per_hour", DEFAULT_PER_HOUR), raw.get("per_day", DEFAULT_PER_DAY))

    def to_dict(self) -> Dict[str, int]:
        return {"per_hour": self.per_hour, "per_day": self.per_day}


@dataclass(frozen=True)
class MailCursor:
    """Incremental-fetch position in one folder: the UIDVALIDITY epoch and the last UID seen.

    `last_internaldate` (ISO date of the newest message seen) lets a watcher resynchronise by
    date when the server resets UIDVALIDITY.
    """

    uidvalidity: int
    last_uid: int
    folder: str = "INBOX"
    last_internaldate: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uidvalidity": self.uidvalidity,
            "last_uid": self.last_uid,
            "folder": self.folder,
            "last_internaldate": self.last_internaldate,
        }

    @classmethod
    def from_dict(cls, raw: Any) -> Optional["MailCursor"]:
        if not isinstance(raw, dict):
            return None
        try:
            return cls(
                uidvalidity=int(raw.get("uidvalidity")),
                last_uid=int(raw.get("last_uid")),
                folder=str(raw.get("folder") or "INBOX"),
                last_internaldate=str(raw.get("last_internaldate") or ""),
            )
        except (TypeError, ValueError):
            return None


@dataclass(frozen=True)
class Attachment:
    """An outgoing attachment: bytes in memory or a file path read at send time."""

    filename: str
    content_type: str = "application/octet-stream"
    data: Optional[bytes] = None
    path: str = ""

    def read(self) -> bytes:
        if self.data is not None:
            return bytes(self.data)
        if self.path:
            return Path(os.path.expanduser(self.path)).read_bytes()
        return b""


@dataclass(frozen=True)
class OutgoingMessage:
    to: Tuple[str, ...] = ()
    subject: str = ""
    text: str = ""
    html: str = ""
    cc: Tuple[str, ...] = ()
    bcc: Tuple[str, ...] = ()
    attachments: Tuple[Attachment, ...] = ()
    in_reply_to: str = ""
    references: Tuple[str, ...] = ()

    @property
    def recipients(self) -> List[str]:
        return list(self.to) + list(self.cc) + list(self.bcc)

    def with_recipients(self, *, to=None, cc=None, bcc=None) -> "OutgoingMessage":
        return replace(
            self,
            to=tuple(to) if to is not None else self.to,
            cc=tuple(cc) if cc is not None else self.cc,
            bcc=tuple(bcc) if bcc is not None else self.bcc,
        )


@dataclass(frozen=True)
class AttachmentInfo:
    """An attachment as listed from the message structure (its content is not fetched).

    `size` is the part's size on the wire as the server reports it (encoded: a base64 part is
    about 4/3 of the file); the saved file's size is in the download result.
    """

    index: int
    filename: str
    content_type: str
    size: int
    disposition: str = ""
    content_id: str = ""
    encoding: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "filename": self.filename,
            "content_type": self.content_type,
            "size": self.size,
            "encoding": self.encoding,
            "disposition": self.disposition,
            "content_id": self.content_id,
        }


@dataclass(frozen=True)
class MessageSummary:
    uid: int
    folder: str
    uidvalidity: int
    message_id: str
    subject: str
    from_: str
    from_address: str
    to: str
    cc: str
    date: str
    internaldate: str
    flags: Tuple[str, ...]
    size: Optional[int]
    # From the message structure: None when the server's BODYSTRUCTURE could not be read.
    has_attachments: Optional[bool] = None
    reply_to: str = ""
    in_reply_to: str = ""
    # Typed header values, exactly as defined (anything else is None; no guessing):
    # Importance low|normal|high, X-Priority 1 (highest)..5 (lowest), Priority normal|urgent|non-urgent.
    importance: Optional[str] = None
    x_priority: Optional[int] = None
    priority: Optional[str] = None
    list_unsubscribe: bool = False

    @property
    def seen(self) -> bool:
        return any(f.lstrip("\\").lower() == "seen" for f in self.flags)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": str(self.uid),
            "folder": self.folder,
            "uidvalidity": self.uidvalidity,
            "message_id": self.message_id,
            "subject": self.subject,
            "from": self.from_,
            "from_address": self.from_address,
            "to": self.to,
            "cc": self.cc,
            "date": self.date,
            "internaldate": self.internaldate,
            "flags": list(self.flags),
            "seen": self.seen,
            "size": self.size,
            "has_attachments": self.has_attachments,
            "reply_to": self.reply_to,
            "in_reply_to": self.in_reply_to,
            "importance": self.importance,
            "x_priority": self.x_priority,
            "priority": self.priority,
            "list_unsubscribe": self.list_unsubscribe,
        }


@dataclass(frozen=True)
class MessageDetail:
    summary: MessageSummary
    reply_to: str
    in_reply_to: str
    references: Tuple[str, ...]
    text: str
    html: str
    attachments: Tuple[AttachmentInfo, ...]
    raw: bytes = field(default=b"", repr=False)
    # Set when the bodies were NOT fetched because they exceed the reading limit
    # (`max_message_bytes`): `{code: "email_message_too_large", cause, fix, uid, folder, size,
    # limit}`. The bodies are then absent (None in `to_dict()`), never cut.
    skipped: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out = self.summary.to_dict()
        out.update(
            {
                "reply_to": self.reply_to,
                "in_reply_to": self.in_reply_to,
                "references": list(self.references),
                "body_text": None if self.skipped else self.text,
                "body_html": None if self.skipped else self.html,
                "attachments": [a.to_dict() for a in self.attachments],
            }
        )
        if self.skipped:
            out["body_skipped"] = dict(self.skipped)
        return out


@dataclass(frozen=True)
class FetchResult:
    messages: Tuple[MessageSummary, ...]
    cursor: MailCursor
    reset: bool = False
    baseline: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "cursor": self.cursor.to_dict(),
            "reset": self.reset,
            "baseline": self.baseline,
        }


@dataclass(frozen=True)
class SendResult:
    message_id: str
    accepted: Tuple[str, ...]
    refused: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"message_id": self.message_id, "accepted": list(self.accepted), "refused": dict(self.refused)}
