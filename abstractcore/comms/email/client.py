"""`EmailClient`: IMAP (read-only) and SMTP for one account.

Guarantees, each enforced in code:

- **Verified TLS, always.** Every connection uses `ssl.create_default_context()` (certificate
  chain and host name checked), plus the account's CA file when one is set. There is no
  plaintext mode and no switch that turns verification off: an explicit `ssl_context` is
  accepted (private CAs, tests) only if it verifies certificates and host names.
- **Read-only mailbox.** Folders are opened with EXAMINE, bodies are fetched with BODY.PEEK,
  and the IMAP connection itself refuses any command outside a read-only allowlist (no
  SELECT, STORE, COPY, MOVE, EXPUNGE, APPEND, DELETE, ...) before it reaches the wire.
- **No truncation.** Bodies are returned whole (ADR-0026).
- **Typed errors** (`errors.py`), never a raw exception with a password in it.

Sign-in: `auth_kind="password"` uses LOGIN (AUTHENTICATE PLAIN for non-ASCII passwords) on
IMAP and AUTH on SMTP; `auth_kind="oauth2"` uses SASL XOAUTH2 on both with an access token
from an `OAuthTokenProvider`.
"""

from __future__ import annotations

import base64
import email
import email.policy
import imaplib
import mimetypes
import os
import smtplib
import ssl
from contextlib import contextmanager
import time
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage, Message
from email.utils import format_datetime, formataddr, getaddresses, make_msgid, parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from . import imap_codec
from .errors import (
    EmailAttachmentNotFound,
    EmailError,
    EmailInvalidMessage,
    EmailInvalidSettings,
    EmailMessageNotFound,
    EmailNotConfigured,
    EmailProtocolError,
    EmailReadOnlyViolation,
    EmailRecipientRefused,
    EmailSecretUnavailable,
    EmailServerError,
    classify_connection_error,
    classify_imap_failure,
    classify_smtp_code,
)
from .models import (
    Attachment,
    AttachmentInfo,
    EmailAccount,
    EmailSecret,
    FetchResult,
    MailCursor,
    MessageDetail,
    MessageSummary,
    OutgoingMessage,
    SendResult,
    ServerSettings,
)
from .oauth import OAuthTokenProvider, xoauth2_string
from .policy import normalize_address, parse_recipients
from .search import SearchCriteria, imap_date

DEFAULT_TIMEOUT_S = 30.0

_SUMMARY_HEADERS = "FROM TO CC SUBJECT DATE MESSAGE-ID"
_SUMMARY_ITEMS = f"(UID FLAGS RFC822.SIZE INTERNALDATE BODY.PEEK[HEADER.FIELDS ({_SUMMARY_HEADERS})])"
_FULL_ITEMS = "(UID FLAGS RFC822.SIZE INTERNALDATE BODY.PEEK[])"


# ---------------------------------------------------------------------------------------
# TLS
# ---------------------------------------------------------------------------------------


def tls_context(ca_file: str = "") -> ssl.SSLContext:
    """Certificate chain + host name verified; `ca_file` ADDS a private CA to the system store."""

    ctx = ssl.create_default_context()
    if ca_file:
        ctx.load_verify_locations(cafile=os.path.expanduser(ca_file))
    return ctx


def _require_verifying(ctx: ssl.SSLContext) -> ssl.SSLContext:
    if not isinstance(ctx, ssl.SSLContext) or ctx.verify_mode != ssl.CERT_REQUIRED or not ctx.check_hostname:
        raise EmailInvalidSettings(
            "An SSL context that does not verify certificates and host names was given.",
            "Pass a context from ssl.create_default_context() (load a private CA into it if needed).",
        )
    return ctx


# ---------------------------------------------------------------------------------------
# Read-only IMAP
# ---------------------------------------------------------------------------------------

READ_ONLY_COMMANDS = frozenset(
    {
        "CAPABILITY",
        "NOOP",
        "LOGOUT",
        "LOGIN",
        "AUTHENTICATE",
        "STARTTLS",
        "EXAMINE",
        "LIST",
        "LSUB",
        "STATUS",
        "ID",
        "NAMESPACE",
        "UID",
        "SEARCH",
        "FETCH",
    }
)
_READ_ONLY_UID_SUBCOMMANDS = frozenset({"SEARCH", "FETCH"})


def check_read_only_fetch(items: str) -> None:
    """Refuse FETCH items that set \\Seen (BODY[...] without .PEEK, RFC822, RFC822.TEXT, BINARY[...])."""

    upper = str(items or "").upper()
    stripped = upper.replace("BODY.PEEK[", "").replace("BINARY.PEEK[", "")
    if "BODY[" in stripped or "BINARY[" in stripped:
        raise EmailReadOnlyViolation(
            "A FETCH without .PEEK would mark the message as read; refused.",
            "Fetch bodies with BODY.PEEK[...] (the mailbox is read-only).",
        )
    for tok in stripped.replace("(", " ").replace(")", " ").split():
        if tok in {"RFC822", "RFC822.TEXT"}:
            raise EmailReadOnlyViolation(
                f"FETCH {tok} would mark the message as read; refused.",
                "Fetch bodies with BODY.PEEK[...] (the mailbox is read-only).",
            )


class _ReadOnlyGuard:
    """Refuses every IMAP command that could change the mailbox, before it is sent."""

    def _command(self, name, *args):  # type: ignore[override]
        cmd = str(name or "").upper()
        if cmd not in READ_ONLY_COMMANDS:
            raise EmailReadOnlyViolation(
                f"The IMAP command {cmd} could change the mailbox; refused (the mailbox is read-only).",
                "Only read operations are available: list folders, search, read, download attachments.",
            )
        if cmd == "UID":
            sub = str(args[0] if args else "").upper()
            if sub not in _READ_ONLY_UID_SUBCOMMANDS:
                raise EmailReadOnlyViolation(
                    f"The IMAP command UID {sub} could change the mailbox; refused (the mailbox is read-only).",
                    "Only read operations are available: list folders, search, read, download attachments.",
                )
            if sub == "FETCH":
                check_read_only_fetch(" ".join(str(a) for a in args[2:]))
        elif cmd == "FETCH":
            check_read_only_fetch(" ".join(str(a) for a in args[1:]))
        return super()._command(name, *args)  # type: ignore[misc]


class ReadOnlyIMAP4(_ReadOnlyGuard, imaplib.IMAP4):
    pass


class ReadOnlyIMAP4_SSL(_ReadOnlyGuard, imaplib.IMAP4_SSL):
    pass


# ---------------------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------------------


def _header(msg: Message, name: str) -> str:
    value = msg.get(name)
    if value is None:
        return ""
    return str(value).strip()


def _first_address(header_value: str) -> str:
    for _name, addr in getaddresses([header_value or ""]):
        if addr:
            return addr
    return ""


def _flags(value: Any) -> Tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(f) for f in value if f is not None)
    return ()


def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("ascii", errors="replace")
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _internaldate_iso(value: Any) -> str:
    if not value:
        return ""
    try:
        tt = imaplib.Internaldate2tuple(f'INTERNALDATE "{value}"'.encode("ascii"))
    except Exception:
        tt = None
    if not tt:
        return ""
    try:
        # Internaldate2tuple answers in local time; store UTC with its offset so the value
        # means the same instant on every host.
        return datetime.fromtimestamp(time.mktime(tt), tz=timezone.utc).isoformat()
    except Exception:
        return ""


def _decode_part(part: Message) -> str:
    try:
        content = part.get_content()  # type: ignore[attr-defined]
        if isinstance(content, str):
            return content
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")
    except Exception:
        pass
    payload = part.get_payload(decode=True)
    if payload is None:
        return ""
    charset = part.get_content_charset() or "utf-8"
    try:
        return payload.decode(charset, errors="replace")
    except LookupError:
        return payload.decode("utf-8", errors="replace")


def _is_attachment(part: Message) -> bool:
    if part.is_multipart():
        return False
    disp = part.get_content_disposition()
    if disp == "attachment":
        return True
    if part.get_filename():
        return True
    return False


def _attachment_parts(msg: Message) -> List[Message]:
    return [p for p in msg.walk() if _is_attachment(p)]


def extract_bodies(msg: Message) -> Tuple[str, str]:
    """(text/plain, text/html) bodies of the non-attachment parts, whole (no truncation)."""

    texts: List[str] = []
    htmls: List[str] = []
    for part in msg.walk():
        if part.is_multipart() or _is_attachment(part):
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain":
            t = _decode_part(part)
            if t.strip():
                texts.append(t.strip("\r\n"))
        elif ctype == "text/html":
            h = _decode_part(part)
            if h.strip():
                htmls.append(h.strip("\r\n"))
    return "\n\n".join(texts), "\n\n".join(htmls)


def safe_filename(name: str, *, fallback: str = "attachment") -> str:
    """One safe path component from a sender-chosen filename."""

    base = str(name or "").replace("\\", "/").split("/")[-1]
    cleaned = "".join(c for c in base if c.isprintable() and c not in '<>:"|?*\x00')
    cleaned = cleaned.strip().strip(".")
    if not cleaned or cleaned in {".", ".."}:
        cleaned = fallback
    return cleaned[:200]


def _split_ids(value: str) -> Tuple[str, ...]:
    out: List[str] = []
    for tok in str(value or "").split():
        tok = tok.strip()
        if tok.startswith("<") and tok.endswith(">") and len(tok) > 2:
            out.append(tok)
    return tuple(out)


# ---------------------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------------------


class EmailClient:
    """One account's mail operations. Cheap to build; each operation opens its own connection."""

    def __init__(
        self,
        account: EmailAccount,
        secret: Optional[EmailSecret] = None,
        *,
        token_provider: Optional[OAuthTokenProvider] = None,
        timeout: float = DEFAULT_TIMEOUT_S,
        ssl_context: Optional[ssl.SSLContext] = None,
    ) -> None:
        self.account = account
        self._secret = secret
        self._timeout = float(timeout) if timeout and float(timeout) > 0 else DEFAULT_TIMEOUT_S
        self._ssl_context = _require_verifying(ssl_context) if ssl_context is not None else None
        if account.auth_kind == "oauth2":
            if token_provider is None:
                if secret is None or not secret.refresh_token and not secret.access_token:
                    raise EmailSecretUnavailable(
                        "No OAuth2 tokens are stored for this account.",
                        "Sign in again: connect the account with OAuth2.",
                    )
                token_provider = OAuthTokenProvider(account.oauth, secret, verify=self._ssl_context)
            self._tokens = token_provider
        else:
            self._tokens = None
            if secret is None or not secret.password:
                raise EmailSecretUnavailable(
                    "No password is stored for this account.",
                    "Connect the account again with --password <value>.",
                )

    def __repr__(self) -> str:
        return f"EmailClient(address={self.account.address!r})"

    # -- connections ---------------------------------------------------------------------

    def _tls(self, server: ServerSettings) -> ssl.SSLContext:
        if self._ssl_context is not None:
            return self._ssl_context
        try:
            return tls_context(server.ca_file)
        except (OSError, ssl.SSLError):
            raise EmailInvalidSettings(
                "The CA file of the account could not be loaded.",
                "Give the path of a readable PEM file, or clear the CA file to use the system trust store.",
            ) from None

    def _imap_settings(self):
        if self.account.imap is None:
            raise EmailNotConfigured(
                "This account has no IMAP (receive) settings.",
                "Connect the account again with --imap-host to read mail.",
            )
        return self.account.imap

    def _smtp_settings(self):
        if self.account.smtp is None:
            raise EmailNotConfigured(
                "This account has no SMTP (send) settings.",
                "Connect the account again with --smtp-host to send mail.",
            )
        return self.account.smtp

    @contextmanager
    def _imap(self) -> Iterator[imaplib.IMAP4]:
        s = self._imap_settings()
        conn: Optional[imaplib.IMAP4] = None
        try:
            if s.security == "ssl":
                conn = ReadOnlyIMAP4_SSL(s.host, s.port, ssl_context=self._tls(s), timeout=self._timeout)
            else:
                conn = ReadOnlyIMAP4(s.host, s.port, timeout=self._timeout)
                try:
                    conn.starttls(ssl_context=self._tls(s))
                except imaplib.IMAP4.abort as exc:
                    raise classify_connection_error(exc, protocol="imap", host=s.host, port=s.port) from None
                except imaplib.IMAP4.error:
                    raise EmailServerError(
                        "The IMAP server does not offer STARTTLS on this port.",
                        "Use the SSL security mode (usually port 993).",
                        details={"protocol": "imap", "host": s.host, "port": s.port},
                    ) from None
        except EmailError:
            self._close_imap(conn)
            raise
        except Exception as exc:
            self._close_imap(conn)
            raise classify_connection_error(exc, protocol="imap", host=s.host, port=s.port) from None
        try:
            self._imap_login(conn, s)
            yield conn
        except EmailError:
            raise
        except imaplib.IMAP4.abort as exc:
            raise classify_connection_error(exc, protocol="imap", host=s.host, port=s.port) from None
        except imaplib.IMAP4.error as exc:
            raise EmailProtocolError(
                "The IMAP server sent an answer this client could not use.",
                "Retry; if it persists, check that the server supports IMAP4rev1.",
                details={"protocol": "imap", "host": s.host, "port": s.port},
            ) from exc
        except (OSError, ssl.SSLError) as exc:
            raise classify_connection_error(exc, protocol="imap", host=s.host, port=s.port) from None
        finally:
            self._close_imap(conn)

    @staticmethod
    def _close_imap(conn: Optional[imaplib.IMAP4]) -> None:
        if conn is None:
            return
        try:
            conn.logout()
        except Exception:
            try:
                conn.shutdown()
            except Exception:
                pass

    def _imap_login(self, conn: imaplib.IMAP4, s) -> None:
        user = self.account.username
        try:
            if self._tokens is not None:
                token = self._tokens.access_token()
                sent = {"done": False}

                def xoauth(_challenge: bytes) -> bytes:
                    if sent["done"]:
                        return b""  # answer the server's error challenge with an empty line
                    sent["done"] = True
                    return xoauth2_string(user, token).encode("utf-8")

                conn.authenticate("XOAUTH2", xoauth)
            else:
                password = self._secret.password if self._secret else ""
                if password.isascii() and user.isascii():
                    conn.login(user, password)
                else:
                    creds = b"\0" + user.encode("utf-8") + b"\0" + password.encode("utf-8")
                    conn.authenticate("PLAIN", lambda _c: creds)
        except imaplib.IMAP4.abort as exc:
            raise classify_connection_error(exc, protocol="imap", host=s.host, port=s.port) from None
        except imaplib.IMAP4.error as exc:
            data = exc.args[0] if exc.args else b""
            err = classify_imap_failure("LOGIN", data, host=s.host, port=s.port)
            if self._tokens is not None and err.code == "email_auth_failed":
                err = type(err)(
                    "The IMAP server rejected the OAuth2 sign-in for this account.",
                    "Sign in again: connect the account with OAuth2 once more (the access may have been revoked).",
                    details=err.details,
                )
            raise err from None

    @contextmanager
    def _smtp(self) -> Iterator[smtplib.SMTP]:
        s = self._smtp_settings()
        conn: Optional[smtplib.SMTP] = None
        try:
            if s.security == "ssl":
                conn = smtplib.SMTP_SSL(s.host, s.port, timeout=self._timeout, context=self._tls(s))
                conn.ehlo()
            else:
                conn = smtplib.SMTP(s.host, s.port, timeout=self._timeout)
                conn.ehlo()
                if not conn.has_extn("starttls"):
                    raise EmailServerError(
                        "The SMTP server does not offer STARTTLS on this port.",
                        "Use the SSL security mode (usually port 465); mail is never sent unencrypted.",
                        details={"protocol": "smtp", "host": s.host, "port": s.port},
                    )
                conn.starttls(context=self._tls(s))
                conn.ehlo()
        except EmailError:
            self._close_smtp(conn)
            raise
        except smtplib.SMTPResponseException as exc:
            self._close_smtp(conn)
            raise classify_smtp_code(exc.smtp_code, host=s.host, port=s.port, stage="connect") from None
        except Exception as exc:
            self._close_smtp(conn)
            raise classify_connection_error(exc, protocol="smtp", host=s.host, port=s.port) from None
        try:
            self._smtp_login(conn, s)
            yield conn
        except EmailError:
            raise
        except smtplib.SMTPServerDisconnected as exc:
            raise classify_connection_error(ConnectionError(str(type(exc).__name__)), protocol="smtp", host=s.host, port=s.port) from None
        except (OSError, ssl.SSLError) as exc:
            raise classify_connection_error(exc, protocol="smtp", host=s.host, port=s.port) from None
        finally:
            self._close_smtp(conn)

    @staticmethod
    def _close_smtp(conn: Optional[smtplib.SMTP]) -> None:
        if conn is None:
            return
        try:
            conn.quit()
        except Exception:
            try:
                conn.close()
            except Exception:
                pass

    def _smtp_login(self, conn: smtplib.SMTP, s) -> None:
        user = self.account.username
        try:
            if self._tokens is not None:
                token = self._tokens.access_token()

                def xoauth(challenge: Optional[bytes] = None) -> str:
                    # Initial response carries the token; a 334 error challenge gets an empty reply.
                    return xoauth2_string(user, token) if challenge is None else ""

                conn.auth("XOAUTH2", xoauth, initial_response_ok=True)
            else:
                conn.login(user, self._secret.password if self._secret else "")
        except smtplib.SMTPAuthenticationError as exc:
            err = classify_smtp_code(exc.smtp_code, host=s.host, port=s.port, stage="auth")
            if self._tokens is not None and err.code == "email_auth_failed":
                err = type(err)(
                    "The SMTP server rejected the OAuth2 sign-in for this account.",
                    "Sign in again: connect the account with OAuth2 once more (the access may have been revoked).",
                    details=err.details,
                )
            raise err from None
        except smtplib.SMTPNotSupportedError:
            raise EmailServerError(
                "The SMTP server does not offer authentication on this connection.",
                "Check the SMTP host, port and security mode (submission is usually 587 STARTTLS or 465 SSL).",
                details={"protocol": "smtp", "host": s.host, "port": s.port},
            ) from None
        except smtplib.SMTPException as exc:
            code = getattr(exc, "smtp_code", None)
            if isinstance(code, int):
                raise classify_smtp_code(code, host=s.host, port=s.port, stage="auth") from None
            raise EmailServerError(
                "The SMTP server refused the sign-in.",
                "Check the user name, password and security mode.",
                details={"protocol": "smtp", "host": s.host, "port": s.port},
            ) from None
        except UnicodeEncodeError:
            raise EmailInvalidSettings(
                "The SMTP sign-in contains non-ASCII characters this server connection cannot send.",
                "Use an ASCII app password for this account.",
            ) from None

    # -- IMAP helpers ---------------------------------------------------------------------

    def _examine(self, conn: imaplib.IMAP4, folder: Optional[str]) -> Dict[str, Any]:
        s = self._imap_settings()
        name = str(folder or "").strip() or s.folder or "INBOX"
        typ, data = conn.select(imap_codec.quote(imap_codec.encode_mailbox(name)), readonly=True)
        if typ != "OK":
            raise classify_imap_failure("EXAMINE", data, host=s.host, port=s.port, folder=name)
        state = {"folder": name, "exists": _int_or_none(data[0] if data else None) or 0}
        for key in ("UIDVALIDITY", "UIDNEXT"):
            _t, vals = conn.response(key)
            value = vals[0] if vals and vals[0] is not None else None
            state[key.lower()] = _int_or_none(value.decode() if isinstance(value, bytes) else value)
        if state.get("uidvalidity") is None:
            raise EmailProtocolError(
                "The IMAP server did not report UIDVALIDITY for the folder.",
                "This server cannot be used for incremental reading; check that it supports IMAP4rev1.",
                details={"protocol": "imap", "host": s.host, "port": s.port, "folder": name},
            )
        return state

    def _uid_search(self, conn: imaplib.IMAP4, keys: Sequence[str]) -> List[int]:
        s = self._imap_settings()
        typ, data = conn.uid("SEARCH", *keys)
        if typ != "OK":
            raise classify_imap_failure("SEARCH", data, host=s.host, port=s.port)
        uids: List[int] = []
        for chunk in data or []:
            if isinstance(chunk, (bytes, bytearray)):
                for tok in bytes(chunk).split():
                    if tok.isdigit():
                        uids.append(int(tok))
        return sorted(set(uids))

    def _fetch(self, conn: imaplib.IMAP4, uids: Sequence[int], items: str) -> Dict[int, dict]:
        s = self._imap_settings()
        if not uids:
            return {}
        uid_set = ",".join(str(u) for u in uids)
        typ, data = conn.uid("FETCH", uid_set, items)
        if typ != "OK":
            raise classify_imap_failure("FETCH", data, host=s.host, port=s.port)
        out: Dict[int, dict] = {}
        for group in imap_codec.segments_from_response(data):
            try:
                attrs = imap_codec.parse_fetch_group(group)
            except ValueError:
                continue
            uid = _int_or_none(attrs.get("UID"))
            if uid is None:
                continue
            out.setdefault(uid, {}).update(attrs)
        return out

    @staticmethod
    def _body_of(attrs: dict) -> bytes:
        for key, value in attrs.items():
            if key.startswith("BODY[") and isinstance(value, (bytes, bytearray)):
                return bytes(value)
            if key.startswith("BODY[") and isinstance(value, str):
                return value.encode("utf-8", errors="replace")
        return b""

    def _summary(self, uid: int, attrs: dict, folder: str, uidvalidity: int) -> MessageSummary:
        msg = email.message_from_bytes(self._body_of(attrs), policy=email.policy.default)
        from_h = _header(msg, "From")
        return MessageSummary(
            uid=uid,
            folder=folder,
            uidvalidity=uidvalidity,
            message_id=_header(msg, "Message-ID"),
            subject=_header(msg, "Subject"),
            from_=from_h,
            from_address=_first_address(from_h),
            to=_header(msg, "To"),
            cc=_header(msg, "Cc"),
            date=_header(msg, "Date"),
            internaldate=_internaldate_iso(attrs.get("INTERNALDATE")),
            flags=_flags(attrs.get("FLAGS")),
            size=_int_or_none(attrs.get("RFC822.SIZE")),
        )

    # -- public read API ------------------------------------------------------------------

    def test_imap(self) -> Dict[str, Any]:
        with self._imap() as conn:
            state = self._examine(conn, None)
        return {"ok": True, "folder": state["folder"], "messages": state["exists"]}

    def test_smtp(self) -> Dict[str, Any]:
        with self._smtp():
            pass
        return {"ok": True}

    def test(self) -> Dict[str, Any]:
        """Sign in on each configured leg. Never raises: each leg is `{ok, ...}` or a typed error."""

        out: Dict[str, Any] = {}
        for leg, configured, fn in (
            ("imap", self.account.imap is not None, self.test_imap),
            ("smtp", self.account.smtp is not None, self.test_smtp),
        ):
            if not configured:
                out[leg] = {"ok": None, "skipped": "not configured"}
                continue
            try:
                out[leg] = fn()
            except EmailError as err:
                out[leg] = {"ok": False, **err.to_dict()}
        return out

    def capabilities(self) -> List[str]:
        with self._imap() as conn:
            return sorted(str(c) for c in (conn.capabilities or ()))

    def list_folders(self) -> List[Dict[str, Any]]:
        s = self._imap_settings()
        with self._imap() as conn:
            typ, data = conn.list()
            if typ != "OK":
                raise classify_imap_failure("LIST", data, host=s.host, port=s.port)
            out = []
            for group in imap_codec.segments_from_response(data):
                try:
                    row = imap_codec.parse_list_line(group)
                except ValueError:
                    row = None
                if row:
                    out.append({"name": row["name"], "delimiter": row["delimiter"], "flags": row["flags"]})
            return out

    def folder_state(self, folder: Optional[str] = None) -> Dict[str, Any]:
        with self._imap() as conn:
            return self._examine(conn, folder)

    def search(
        self,
        criteria: Optional[SearchCriteria] = None,
        *,
        folder: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Matching messages, newest first: `{folder, uidvalidity, total, messages}`."""

        crit = criteria or SearchCriteria()
        with self._imap() as conn:
            state = self._examine(conn, folder)
            uids = self._uid_search(conn, crit.imap_keys())
            uids.sort(reverse=True)
            matched: List[MessageSummary] = []
            # Fetch in pages, newest first, until `limit` exact matches are found.
            page = 200
            for start in range(0, len(uids), page):
                batch = uids[start : start + page]
                fetched = self._fetch(conn, batch, _SUMMARY_ITEMS)
                for uid in batch:
                    attrs = fetched.get(uid)
                    if attrs is None:
                        continue
                    summary = self._summary(uid, attrs, state["folder"], state["uidvalidity"])
                    if crit.matches(
                        from_header=summary.from_,
                        to_header=summary.to,
                        cc_header=summary.cc,
                        subject=summary.subject,
                        seen=summary.seen,
                    ):
                        matched.append(summary)
                        if limit and len(matched) >= limit:
                            break
                if limit and len(matched) >= limit:
                    break
        return {
            "folder": state["folder"],
            "uidvalidity": state["uidvalidity"],
            "candidates": len(uids),
            "messages": matched,
        }

    def get(self, uid: Any, *, folder: Optional[str] = None, include_raw: bool = False) -> MessageDetail:
        uid_i = _int_or_none(uid)
        if uid_i is None or uid_i <= 0:
            raise EmailInvalidMessage("The message UID must be a positive number.", "Use a uid from list/search results.")
        with self._imap() as conn:
            state = self._examine(conn, folder)
            fetched = self._fetch(conn, [uid_i], _FULL_ITEMS)
        attrs = fetched.get(uid_i)
        raw = self._body_of(attrs) if attrs else b""
        if not attrs or not raw:
            raise EmailMessageNotFound(
                f"No message with uid {uid_i} in {state['folder']!r}.",
                "List or search the folder again; UIDs change when the server resets the folder.",
                details={"uid": uid_i, "folder": state["folder"]},
            )
        return self._detail(uid_i, attrs, raw, state, include_raw=include_raw)

    def _detail(self, uid: int, attrs: dict, raw: bytes, state: dict, *, include_raw: bool) -> MessageDetail:
        msg = email.message_from_bytes(raw, policy=email.policy.default)
        summary = self._summary(uid, {**attrs, "BODY[]": raw}, state["folder"], state["uidvalidity"])
        text, html = extract_bodies(msg)
        infos: List[AttachmentInfo] = []
        for idx, part in enumerate(_attachment_parts(msg)):
            payload = part.get_payload(decode=True) or b""
            infos.append(
                AttachmentInfo(
                    index=idx,
                    filename=str(part.get_filename() or ""),
                    content_type=part.get_content_type(),
                    size=len(payload),
                    disposition=str(part.get_content_disposition() or ""),
                    content_id=_header(part, "Content-ID"),
                )
            )
        return MessageDetail(
            summary=summary,
            reply_to=_header(msg, "Reply-To"),
            in_reply_to=_header(msg, "In-Reply-To"),
            references=_split_ids(_header(msg, "References")),
            text=text,
            html=html,
            attachments=tuple(infos),
            raw=raw if include_raw else b"",
        )

    def download_attachment(
        self,
        uid: Any,
        index: int,
        dest_dir: str,
        *,
        folder: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Write attachment `index` of message `uid` into `dest_dir` (never overwrites)."""

        target_dir = Path(os.path.expanduser(str(dest_dir or "")))
        if not str(dest_dir or "").strip() or not target_dir.is_dir():
            raise EmailInvalidSettings(
                f"The destination folder {str(dest_dir)!r} does not exist.",
                "Give an existing folder to save the attachment into.",
            )
        detail = self.get(uid, folder=folder, include_raw=True)
        msg = email.message_from_bytes(detail.raw, policy=email.policy.default)
        parts = _attachment_parts(msg)
        try:
            idx = int(index)
        except (TypeError, ValueError):
            idx = -1
        if idx < 0 or idx >= len(parts):
            raise EmailAttachmentNotFound(
                f"Message {detail.summary.uid} has no attachment number {index}.",
                f"Use an index from read_email's attachments list (0 to {len(parts) - 1})." if parts else "This message has no attachments.",
            )
        part = parts[idx]
        payload = part.get_payload(decode=True) or b""
        name = safe_filename(str(part.get_filename() or ""), fallback=f"attachment-{idx}")
        path = target_dir / name
        stem, suffix = os.path.splitext(name)
        n = 1
        while path.exists():
            path = target_dir / f"{stem} ({n}){suffix}"
            n += 1
        with open(path, "xb") as fh:
            fh.write(payload)
        return {
            "path": str(path),
            "filename": path.name,
            "original_filename": str(part.get_filename() or ""),
            "content_type": part.get_content_type(),
            "size": len(payload),
            "uid": detail.summary.uid,
            "index": idx,
        }

    def fetch_new(self, cursor: Optional[MailCursor] = None, *, folder: Optional[str] = None, limit: Optional[int] = None) -> FetchResult:
        """Messages that arrived after `cursor`, oldest first, and the cursor to store next.

        - No cursor: a baseline — no messages, the cursor at the newest message (a watcher
          only sees mail that arrives after it starts).
        - UIDVALIDITY changed (the server rebuilt the folder): `reset=True`, the messages whose
          INTERNALDATE day is on/after the day before the cursor's `last_internaldate`
          (callers dedupe by Message-ID), and a cursor in the new epoch.
        - Otherwise: UID > last_uid. The cursor advances to the last message RETURNED, so a
          caller that stops early re-reads the rest next time.
        """

        name = (cursor.folder if cursor else None) or folder
        with self._imap() as conn:
            state = self._examine(conn, name)
            uv = int(state["uidvalidity"])
            if cursor is None:
                uidnext = state.get("uidnext")
                if uidnext:
                    last = int(uidnext) - 1
                else:
                    all_uids = self._uid_search(conn, ["ALL"])
                    last = all_uids[-1] if all_uids else 0
                return FetchResult(messages=(), cursor=MailCursor(uv, max(0, last), state["folder"]), baseline=True)
            reset = uv != cursor.uidvalidity
            if reset:
                if cursor.last_internaldate:
                    try:
                        # IMAP SINCE compares dates in the server's own time zone: start one
                        # day earlier so no message of that day is missed (callers dedupe).
                        day = datetime.fromisoformat(cursor.last_internaldate).date() - timedelta(days=1)
                        uids = self._uid_search(conn, ["SINCE", imap_date(day)])
                    except ValueError:
                        uids = []
                else:
                    uids = []
            else:
                uids = [u for u in self._uid_search(conn, ["UID", f"{cursor.last_uid + 1}:*"]) if u > cursor.last_uid]
            uids.sort()
            if limit:
                uids = uids[: int(limit)]
            fetched = self._fetch(conn, uids, _SUMMARY_ITEMS)
        messages = [self._summary(u, fetched[u], state["folder"], uv) for u in uids if u in fetched]
        if messages:
            last = messages[-1]
            nxt = MailCursor(uv, last.uid, state["folder"], last.internaldate or cursor.last_internaldate)
        else:
            nxt = MailCursor(uv, cursor.last_uid if not reset else 0, state["folder"], cursor.last_internaldate)
        return FetchResult(messages=tuple(messages), cursor=nxt, reset=reset)

    # -- sending --------------------------------------------------------------------------

    def build_mime(self, message: OutgoingMessage) -> EmailMessage:
        acct = self.account
        mime = EmailMessage(policy=email.policy.SMTP)
        try:
            mime["From"] = formataddr((acct.display_name, acct.address)) if acct.display_name else acct.address
            if message.to:
                mime["To"] = ", ".join(message.to)
            if message.cc:
                mime["Cc"] = ", ".join(message.cc)
            mime["Subject"] = message.subject
            mime["Date"] = format_datetime(datetime.now().astimezone())
            domain = acct.address.split("@", 1)[1]
            mime["Message-ID"] = make_msgid(domain=domain)
            if message.in_reply_to:
                mime["In-Reply-To"] = message.in_reply_to
            if message.references:
                mime["References"] = " ".join(message.references)
        except (ValueError, TypeError) as exc:
            raise EmailInvalidMessage(
                "A message header is not valid (line breaks are not allowed in headers).",
                "Give the subject and addresses on one line each.",
            ) from exc
        text = message.text or ""
        html = message.html or ""
        if html and text:
            mime.set_content(text)
            mime.add_alternative(html, subtype="html")
        elif html:
            mime.set_content("This message contains HTML content.")
            mime.add_alternative(html, subtype="html")
        else:
            mime.set_content(text)
        for att in message.attachments:
            data = att.read()
            ctype = att.content_type or mimetypes.guess_type(att.filename)[0] or "application/octet-stream"
            if "/" not in ctype:
                ctype = "application/octet-stream"
            maintype, subtype = ctype.split("/", 1)
            mime.add_attachment(data, maintype=maintype, subtype=subtype, filename=safe_filename(att.filename))
        return mime

    def send(self, message: OutgoingMessage) -> SendResult:
        """Send as-is (no policy, no limits: use `sending.guarded_send` for that)."""

        if not message.recipients:
            raise EmailInvalidMessage("The message has no recipient.", "Give at least one recipient in To, Cc or Bcc.")
        if not (message.subject or "").strip():
            raise EmailInvalidMessage("The message has no subject.", "Give a subject.")
        s = self._smtp_settings()
        try:
            envelope = parse_recipients(message.recipients)
        except ValueError as exc:
            raise EmailInvalidMessage(f"A recipient is not valid: {exc}.", "Give recipients as name@example.test.") from None
        mime = self.build_mime(message)
        with self._smtp() as conn:
            try:
                refused = conn.send_message(mime, from_addr=self.account.address, to_addrs=envelope) or {}
            except smtplib.SMTPRecipientsRefused as exc:
                codes = {str(k): int(v[0]) for k, v in (exc.recipients or {}).items()}
                first_addr, first_code = next(iter(codes.items()), ("", 550))
                err = classify_smtp_code(first_code, host=s.host, port=s.port, stage="recipient", recipient=first_addr)
                err.details["refused"] = codes
                raise err from None
            except smtplib.SMTPSenderRefused as exc:
                raise classify_smtp_code(exc.smtp_code, host=s.host, port=s.port, stage="sender") from None
            except smtplib.SMTPDataError as exc:
                raise classify_smtp_code(exc.smtp_code, host=s.host, port=s.port, stage="data") from None
            except smtplib.SMTPResponseException as exc:
                raise classify_smtp_code(exc.smtp_code, host=s.host, port=s.port, stage="data") from None
        refused_codes = {str(k): int(v[0]) for k, v in refused.items()}
        accepted = tuple(a for a in envelope if a not in refused_codes)
        return SendResult(message_id=str(mime["Message-ID"]), accepted=accepted, refused=refused_codes)

    def build_reply(
        self,
        uid: Any,
        *,
        text: str = "",
        html: str = "",
        reply_all: bool = False,
        attachments: Sequence[Attachment] = (),
        folder: Optional[str] = None,
    ) -> Tuple[OutgoingMessage, MessageDetail]:
        """The reply to message `uid`: recipients from Reply-To (else From), threading headers set."""

        original = self.get(uid, folder=folder)
        own = normalize_address(self.account.address)
        primary_src = original.reply_to or original.summary.from_
        to: List[str] = []
        seen = set()

        def add(dst: List[str], header_value: str) -> None:
            for _n, addr in getaddresses([header_value or ""]):
                if not addr:
                    continue
                try:
                    norm = normalize_address(addr)
                except ValueError:
                    continue
                if norm == own or norm in seen:
                    continue
                seen.add(norm)
                dst.append(addr)

        add(to, primary_src)
        cc: List[str] = []
        if reply_all:
            add(cc, original.summary.to)
            add(cc, original.summary.cc)
        if not to and not cc:
            raise EmailInvalidMessage(
                "The original message has no address to reply to (other than this account).",
                "Use send_email with explicit recipients instead.",
            )
        subject = original.summary.subject or ""
        if not subject[:3].lower() == "re:":
            subject = f"Re: {subject}" if subject else "Re:"
        refs = list(original.references)
        if original.summary.message_id and original.summary.message_id not in refs:
            refs.append(original.summary.message_id)
        msg = OutgoingMessage(
            to=tuple(to),
            cc=tuple(cc),
            subject=subject,
            text=text,
            html=html,
            attachments=tuple(attachments),
            in_reply_to=original.summary.message_id,
            references=tuple(refs),
        )
        return msg, original
