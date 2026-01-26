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
import os
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
    name = str(env_var or "").strip()
    if not name:
        return None, f"Missing {label} env var name"
    value = os.getenv(name)
    if value is None or not str(value).strip():
        return None, f"Missing env var {name} for {label}"
    return str(value), None


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
    description="Send an email via SMTP (supports text and HTML bodies).",
    tags=["comms", "email"],
    when_to_use="Use to send an email notification or report to one or more recipients.",
    examples=[
        {
            "description": "Send a simple text email (credentials via env)",
            "arguments": {
                "smtp_host": "smtp.example.com",
                "username": "me@example.com",
                "password_env_var": "EMAIL_PASSWORD",
                "to": "you@example.com",
                "subject": "Hello",
                "body_text": "Hi there!",
            },
        }
    ],
)
def send_email(
    smtp_host: str,
    username: str,
    to: Any,
    subject: str,
    *,
    body_text: Optional[str] = None,
    body_html: Optional[str] = None,
    from_email: Optional[str] = None,
    cc: Any = None,
    bcc: Any = None,
    reply_to: Optional[str] = None,
    smtp_port: int = 587,
    use_starttls: bool = True,
    password_env_var: str = "EMAIL_PASSWORD",
    timeout_s: float = 30.0,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Send an email via SMTP, resolving the SMTP password from an env var."""
    password, err = _resolve_required_env(password_env_var, label="SMTP password")
    if err is not None:
        return {"success": False, "error": err}

    smtp_host = str(smtp_host or "").strip()
    username = str(username or "").strip()
    subject = str(subject or "").strip()
    if not smtp_host:
        return {"success": False, "error": "smtp_host is required"}
    if not username:
        return {"success": False, "error": "username is required"}
    if not subject:
        return {"success": False, "error": "subject is required"}

    to_list = _coerce_str_list(to)
    cc_list = _coerce_str_list(cc)
    bcc_list = _coerce_str_list(bcc)
    if not to_list and not cc_list and not bcc_list:
        return {"success": False, "error": "At least one recipient is required (to/cc/bcc)"}

    body_text = (body_text or "").strip()
    body_html = (body_html or "").strip()
    if not body_text and not body_html:
        return {"success": False, "error": "Provide body_text and/or body_html"}

    sender = (from_email or "").strip() or username

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender
    if to_list:
        msg["To"] = ", ".join(to_list)
    if cc_list:
        msg["Cc"] = ", ".join(cc_list)
    if reply_to and str(reply_to).strip():
        msg["Reply-To"] = str(reply_to).strip()
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid()

    extra_headers = headers if isinstance(headers, dict) else None
    if extra_headers:
        for k, v in extra_headers.items():
            if not isinstance(k, str) or not k.strip():
                continue
            msg[str(k).strip()] = str(v)

    if body_text and body_html:
        msg.set_content(body_text)
        msg.add_alternative(body_html, subtype="html")
    elif body_html:
        # Some clients handle HTML-only emails poorly; include a minimal text fallback.
        msg.set_content(body_text or "(This email contains HTML content.)")
        msg.add_alternative(body_html, subtype="html")
    else:
        msg.set_content(body_text)

    smtp_port_i = int(smtp_port or 0)
    if smtp_port_i <= 0:
        return {"success": False, "error": "smtp_port must be a positive integer"}

    timeout = float(timeout_s) if isinstance(timeout_s, (int, float)) else 30.0
    if timeout <= 0:
        timeout = 30.0

    try:
        if use_starttls:
            client: Any = smtplib.SMTP(smtp_host, smtp_port_i, timeout=timeout)
            try:
                client.ehlo()
                client.starttls()
                client.ehlo()
                client.login(username, password)
                client.send_message(msg, from_addr=sender, to_addrs=to_list + cc_list + bcc_list)
            finally:
                try:
                    client.quit()
                except Exception:
                    pass
        else:
            client2: Any = smtplib.SMTP_SSL(smtp_host, smtp_port_i, timeout=timeout)
            try:
                client2.login(username, password)
                client2.send_message(msg, from_addr=sender, to_addrs=to_list + cc_list + bcc_list)
            finally:
                try:
                    client2.quit()
                except Exception:
                    pass

        return {
            "success": True,
            "message_id": str(msg.get("Message-ID") or ""),
            "from": sender,
            "to": list(to_list),
            "cc": list(cc_list),
            "bcc": list(bcc_list),
            "smtp": {"host": smtp_host, "port": smtp_port_i, "username": username, "starttls": bool(use_starttls)},
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "smtp": {"host": smtp_host, "port": smtp_port_i, "username": username, "starttls": bool(use_starttls)},
        }


@tool(
    description="List recent emails from an IMAP mailbox (supports since + read/unread filters).",
    tags=["comms", "email"],
    when_to_use="Use to fetch a digest of recent emails (subject/from/date/flags) for review or routing.",
    examples=[
        {
            "description": "List unread emails from the last 7 days",
            "arguments": {
                "imap_host": "imap.example.com",
                "username": "me@example.com",
                "password_env_var": "EMAIL_PASSWORD",
                "since": "7d",
                "status": "unread",
                "limit": 10,
            },
        }
    ],
)
def list_emails(
    imap_host: str,
    username: str,
    *,
    password_env_var: str = "EMAIL_PASSWORD",
    mailbox: str = "INBOX",
    since: Optional[str] = None,
    status: str = "all",
    limit: int = 20,
    imap_port: int = 993,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    """List email headers from an IMAP mailbox, resolving the password from an env var."""
    password, err = _resolve_required_env(password_env_var, label="IMAP password")
    if err is not None:
        return {"success": False, "error": err}

    imap_host = str(imap_host or "").strip()
    username = str(username or "").strip()
    mailbox = str(mailbox or "").strip() or "INBOX"

    if not imap_host:
        return {"success": False, "error": "imap_host is required"}
    if not username:
        return {"success": False, "error": "username is required"}

    try:
        limit_i = int(limit or 0)
    except Exception:
        limit_i = 0
    if limit_i <= 0:
        limit_i = 20

    try:
        imap_port_i = int(imap_port or 0)
    except Exception:
        imap_port_i = 0
    if imap_port_i <= 0:
        return {"success": False, "error": "imap_port must be a positive integer"}

    dt_since, dt_err = _parse_since(since)
    if dt_err is not None:
        return {"success": False, "error": dt_err}

    status_norm = str(status or "").strip().lower() or "all"
    if status_norm not in {"all", "unread", "read"}:
        return {"success": False, "error": "status must be one of: all, unread, read"}

    timeout = float(timeout_s) if isinstance(timeout_s, (int, float)) else 30.0
    if timeout <= 0:
        timeout = 30.0

    client: Optional[imaplib.IMAP4_SSL] = None
    try:
        client = imaplib.IMAP4_SSL(imap_host, imap_port_i)
        try:
            if getattr(client, "sock", None) is not None:
                client.sock.settimeout(timeout)  # type: ignore[attr-defined]
        except Exception:
            pass

        client.login(username, password)
        typ, _ = client.select(mailbox, readonly=True)
        if typ != "OK":
            return {"success": False, "error": f"Failed to select mailbox: {mailbox}"}

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
            return {"success": False, "error": "IMAP search failed"}

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
            "mailbox": mailbox,
            "filter": {"since": dt_since.isoformat() if dt_since else None, "status": status_norm, "limit": limit_i},
            "counts": {"returned": len(messages), "unread": unread, "read": read},
            "messages": messages,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "mailbox": mailbox}
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
            "description": "Read an email by UID",
            "arguments": {
                "imap_host": "imap.example.com",
                "username": "me@example.com",
                "password_env_var": "EMAIL_PASSWORD",
                "uid": "12345",
            },
        }
    ],
)
def read_email(
    imap_host: str,
    username: str,
    uid: str,
    *,
    password_env_var: str = "EMAIL_PASSWORD",
    mailbox: str = "INBOX",
    imap_port: int = 993,
    timeout_s: float = 30.0,
    max_body_chars: int = 20000,
) -> Dict[str, Any]:
    """Read a single email by UID from an IMAP mailbox (best-effort; does not mark as read)."""
    password, err = _resolve_required_env(password_env_var, label="IMAP password")
    if err is not None:
        return {"success": False, "error": err}

    imap_host = str(imap_host or "").strip()
    username = str(username or "").strip()
    mailbox = str(mailbox or "").strip() or "INBOX"
    uid = str(uid or "").strip()
    if not imap_host:
        return {"success": False, "error": "imap_host is required"}
    if not username:
        return {"success": False, "error": "username is required"}
    if not uid:
        return {"success": False, "error": "uid is required"}

    try:
        imap_port_i = int(imap_port or 0)
    except Exception:
        imap_port_i = 0
    if imap_port_i <= 0:
        return {"success": False, "error": "imap_port must be a positive integer"}

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
        client = imaplib.IMAP4_SSL(imap_host, imap_port_i)
        try:
            if getattr(client, "sock", None) is not None:
                client.sock.settimeout(timeout)  # type: ignore[attr-defined]
        except Exception:
            pass

        client.login(username, password)
        typ, _ = client.select(mailbox, readonly=True)
        if typ != "OK":
            return {"success": False, "error": f"Failed to select mailbox: {mailbox}"}

        typ2, fetched = client.uid("fetch", uid, "(FLAGS BODY.PEEK[])")
        if typ2 != "OK" or not fetched:
            return {"success": False, "error": f"Email not found for uid={uid}"}

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
            return {"success": False, "error": f"Failed to fetch email bytes for uid={uid}"}

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
            "mailbox": mailbox,
            "uid": uid,
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
        return {"success": False, "error": str(e), "mailbox": mailbox, "uid": uid}
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
