"""Typed email errors, classified from protocol codes and exception classes.

Every error carries a stable `code`, a `cause` (what happened, in words a user can act on),
a `fix` (what to do) and `retryable` (whether trying again later can succeed without a
change). Classification never reads free text: IMAP errors are classified by the command
that failed and the RFC 5530 response code in brackets (`[AUTHENTICATIONFAILED]`,
`[OVERQUOTA]`, ...), SMTP errors by their numeric reply code, network errors by exception
class.

Secrets never enter an error: no password, no AUTH exchange. Endpoint facts (protocol,
host, port) travel in `details`, which a caller may drop before handing the error to a
model (`to_dict(include_details=False)`).
"""

from __future__ import annotations

import socket
import ssl
from typing import Any, Dict, Optional

SETTINGS_HINT = "abstractcore email connect"


class EmailError(Exception):
    """Base class. `code` is stable and machine-readable; `cause` and `fix` are sentences."""

    code = "email_error"
    retryable = False

    def __init__(
        self,
        cause: str,
        fix: str,
        *,
        code: Optional[str] = None,
        retryable: Optional[bool] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.cause = str(cause or "").strip()
        self.fix = str(fix or "").strip()
        if code:
            self.code = code
        if retryable is not None:
            self.retryable = bool(retryable)
        self.details: Dict[str, Any] = dict(details or {})
        super().__init__(self.message)

    @property
    def message(self) -> str:
        return f"{self.cause} Fix: {self.fix}" if self.fix else self.cause

    def to_dict(self, *, include_details: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "code": self.code,
            "cause": self.cause,
            "fix": self.fix,
            "retryable": bool(self.retryable),
        }
        if include_details and self.details:
            out["details"] = dict(self.details)
        return out


class EmailNotConfigured(EmailError):
    code = "email_not_configured"


class EmailDisabled(EmailError):
    code = "email_disabled"


class EmailAgentToolsOff(EmailDisabled):
    """The account works, but "Agent email tools" is off: agents may not use it (default)."""

    code = "email_agent_tools_off"


class EmailMessageTooLarge(EmailError):
    """A message (or attachment) is larger than the fetch limit (`max_message_bytes`).

    A resource limit on FETCHING, never a truncation: reading a message returns its headers,
    attachment list and a typed skip record instead of the bodies; downloading an attachment
    raises this error. `details`: `{uid, folder, size, limit}`.
    """

    code = "email_message_too_large"


class EmailInvalidSettings(EmailError):
    code = "email_invalid_settings"


class EmailInvalidMessage(EmailError):
    code = "email_invalid_message"


class EmailSecretUnavailable(EmailError):
    code = "email_secret_unavailable"


class EmailAuthFailed(EmailError):
    code = "email_auth_failed"


class EmailTlsFailed(EmailError):
    code = "email_tls_failed"


class EmailUnreachable(EmailError):
    code = "email_unreachable"
    retryable = True


class EmailMailboxMissing(EmailError):
    code = "email_mailbox_missing"


class EmailQuotaExceeded(EmailError):
    code = "email_quota_exceeded"
    retryable = True


class EmailRecipientRefused(EmailError):
    """The mail SERVER refused one or more recipients (SMTP 550/553 ...)."""

    code = "email_recipient_refused"


class EmailPolicyRefused(EmailError):
    """The user's recipient POLICY refused the message (nothing was sent)."""

    code = "email_policy_refused"


class EmailRateLimited(EmailError):
    code = "email_rate_limited"
    retryable = True


class EmailMessageNotFound(EmailError):
    code = "email_message_not_found"


class EmailAttachmentNotFound(EmailError):
    code = "email_attachment_not_found"


class EmailTransient(EmailError):
    code = "email_transient"
    retryable = True


class EmailServerError(EmailError):
    code = "email_server_error"


class EmailProtocolError(EmailError):
    code = "email_protocol_error"


class EmailOAuthReauthorize(EmailError):
    """The OAuth2 grant is gone (expired, revoked, password changed): sign in again."""

    code = "email_oauth_reauthorize"


class EmailOAuthFailed(EmailError):
    """The OAuth2 token endpoint refused the client or the request."""

    code = "email_oauth_failed"


class EmailOAuthPending(EmailError):
    """Device authorization not completed yet (internal to the polling loop)."""

    code = "email_oauth_pending"
    retryable = True


class EmailReadOnlyViolation(EmailError):
    """A command that could change the mailbox was about to be sent; refused locally."""

    code = "email_read_only"


# ---------------------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------------------


def _endpoint(protocol: str, host: str, port: int) -> Dict[str, Any]:
    return {"protocol": protocol, "host": host, "port": int(port)}


def _label(protocol: str) -> str:
    return "IMAP" if protocol == "imap" else "SMTP"


def classify_connection_error(exc: BaseException, *, protocol: str, host: str, port: int) -> EmailError:
    """Network / TLS failures while connecting or talking, by exception class."""

    if isinstance(exc, EmailError):
        return exc
    label = _label(protocol)
    details = _endpoint(protocol, host, port)
    if isinstance(exc, ssl.SSLCertVerificationError):
        details["verify_message"] = str(getattr(exc, "verify_message", "") or "")
        return EmailTlsFailed(
            f"The {label} server's TLS certificate could not be verified for this host name, so the "
            "connection was refused before any password was sent.",
            "Use the host name the provider's setup guide gives (the one on its certificate). For a "
            "self-hosted server signed by a private CA, set the account's CA file to that CA's PEM file.",
            details=details,
        )
    if isinstance(exc, ssl.SSLError):
        details["ssl_reason"] = str(getattr(exc, "reason", "") or "")
        return EmailTlsFailed(
            f"The TLS handshake with the {label} server failed.",
            "Check the security mode against the port: SSL (implicit TLS) is usually IMAP 993 / SMTP 465, "
            "STARTTLS is usually IMAP 143 / SMTP 587.",
            details=details,
        )
    if isinstance(exc, socket.gaierror):
        return EmailUnreachable(
            f"The {label} host name could not be resolved.",
            "Check the host name for typos and that this machine has network access.",
            details=details,
        )
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return EmailUnreachable(
            f"The {label} server did not answer in time.",
            "Check the host and port, and that a firewall does not block the connection; then retry.",
            details=details,
        )
    if isinstance(exc, ConnectionRefusedError):
        return EmailUnreachable(
            f"The {label} server refused the connection on this port.",
            "Check the port number and the security mode (SSL or STARTTLS) for this server.",
            details=details,
        )
    if isinstance(exc, (ConnectionError, OSError, EOFError)):
        details["error_class"] = type(exc).__name__
        return EmailUnreachable(
            f"The connection to the {label} server failed or was closed.",
            "Check the host, port and network access; then retry.",
            details=details,
        )
    details["error_class"] = type(exc).__name__
    return EmailServerError(
        f"An unexpected {label} error occurred ({type(exc).__name__}).",
        "Retry; if it persists, test the account and check the server settings.",
        details=details,
    )


def imap_response_code(data: Any) -> str:
    """The RFC 3501 resp-text-code of a server response (`[AUTHENTICATIONFAILED] ...` -> the atom).

    A grammar read of the leading bracketed atom, not a text search: returns "" when the
    response text does not start with a code.
    """

    raw = data
    if isinstance(raw, (list, tuple)):
        raw = raw[-1] if raw else b""
    if isinstance(raw, (bytes, bytearray)):
        text = bytes(raw).decode("utf-8", errors="replace")
    else:
        text = str(raw or "")
    text = text.strip()
    if text.startswith("b'") or text.startswith('b"'):
        text = text[2:]
    if not text.startswith("["):
        return ""
    end = text.find("]")
    if end <= 1:
        return ""
    atom = text[1:end].split(" ", 1)[0]
    return atom.upper()


def classify_imap_failure(command: str, data: Any, *, host: str, port: int, folder: str = "") -> EmailError:
    """A NO/BAD answer to an IMAP command, by command and response code."""

    code = imap_response_code(data)
    details = _endpoint("imap", host, port)
    details["command"] = command.upper()
    if code:
        details["response_code"] = code
    cmd = command.upper()
    if code in {"OVERQUOTA"}:
        return EmailQuotaExceeded(
            "The mailbox is over its storage quota.",
            "Free space in the mailbox (or raise the quota with the provider), then retry.",
            details=details,
        )
    if code in {"UNAVAILABLE", "INUSE", "LIMIT"}:
        return EmailTransient(
            "The IMAP server is temporarily unable to complete the request.",
            "Retry later.",
            details=details,
        )
    if cmd in {"LOGIN", "AUTHENTICATE"} or code in {"AUTHENTICATIONFAILED", "AUTHORIZATIONFAILED", "EXPIRED"}:
        return EmailAuthFailed(
            "The IMAP server rejected the user name or password.",
            "Check the user name and password (many providers need an app password when two-step "
            f"verification is on), then connect again with `{SETTINGS_HINT} ... --password <value>`.",
            details=details,
        )
    if cmd in {"EXAMINE", "SELECT", "STATUS"} or code in {"NONEXISTENT", "TRYCREATE"}:
        if folder:
            details["folder"] = folder
        return EmailMailboxMissing(
            f"The mailbox folder {folder!r} does not exist on the server (or cannot be opened)."
            if folder
            else "The mailbox folder does not exist on the server (or cannot be opened).",
            "List the folders (`abstractcore email folders`) and use one of the names it shows.",
            details=details,
        )
    return EmailProtocolError(
        f"The IMAP server refused the {cmd} command.",
        "Retry; if it persists, test the account and check that the server supports IMAP4rev1.",
        details=details,
    )


def classify_smtp_code(code: int, *, host: str, port: int, stage: str, recipient: str = "") -> EmailError:
    """An SMTP reply code (RFC 5321) at a given stage (auth, sender, recipient, data)."""

    details = _endpoint("smtp", host, port)
    details["smtp_code"] = int(code)
    details["stage"] = stage
    if recipient:
        details["recipient"] = recipient
    if stage == "auth" and code in {530, 534, 535, 538}:
        return EmailAuthFailed(
            "The SMTP server rejected the user name or password.",
            "Check the user name and password (many providers need an app password when two-step "
            f"verification is on), then connect again with `{SETTINGS_HINT} ... --password <value>`.",
            details=details,
        )
    if code in {452, 552}:
        return EmailQuotaExceeded(
            "The SMTP server refused the message for size or storage reasons.",
            "Send a smaller message (fewer or smaller attachments) or retry later.",
            details=details,
        )
    if stage == "recipient" and code in {550, 551, 553, 554, 521, 556}:
        return EmailRecipientRefused(
            "The SMTP server refused a recipient address.",
            "Check the recipient address; the server does not accept mail for it from this account.",
            details=details,
        )
    if stage == "sender" and code in {550, 553, 554, 555}:
        return EmailServerError(
            "The SMTP server refused the sender address for this account.",
            "Set the account's sender address to one this login may send as (usually the login address).",
            details=details,
        )
    if 400 <= code < 500:
        return EmailTransient(
            "The SMTP server refused temporarily.",
            "Retry later.",
            details=details,
        )
    return EmailServerError(
        f"The SMTP server refused the request at the {stage} stage.",
        "Check the account settings and the message; retry later if the server was busy.",
        details=details,
    )
