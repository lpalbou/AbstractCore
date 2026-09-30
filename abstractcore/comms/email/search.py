"""Typed mailbox search: fields only, compiled to IMAP SEARCH keys, checked exactly.

No free-text parsing, no patterns. The server narrows with SEARCH (FROM / TO / SUBJECT /
SINCE / BEFORE / UNSEEN / SEEN); every returned message is then checked exactly on its
decoded headers, because IMAP's FROM and SUBJECT are server-defined substring matches:

- `from_address`: the sender's address equals it (normalised);
- `from_domain`: the sender's domain equals it (normalised; a subdomain is a different domain);
- `to_address`: one of the To/Cc addresses equals it;
- `subject_contains`: the decoded subject contains this literal text (case-insensitive);
- `since` / `before`: INTERNALDATE day bounds (IMAP semantics: since inclusive, before exclusive);
- `unseen`: True = unread only, False = read only, None = both;
- `has_attachment`: True = only messages with at least one attachment, False = only messages
  without, None = both. IMAP has no search key for this: it is checked on each message's
  structure (BODYSTRUCTURE: a part with disposition `attachment` or a file name); a message
  whose structure the server could not describe matches neither True nor False.

Text that is not ASCII is not sent to the server (not every server supports SEARCH CHARSET
UTF-8); the exact client-side check still applies.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from email.utils import getaddresses
from typing import Any, List, Optional

from .errors import EmailInvalidSettings
from .imap_codec import quote
from .policy import normalize_address, normalize_domain, split_address

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def imap_date(d: date) -> str:
    return f"{d.day:02d}-{_MONTHS[d.month - 1]}-{d.year:04d}"


def parse_day(value: Any, *, label: str = "since") -> Optional[date]:
    """A calendar day from a date, a datetime, ISO text, or `<N>d` (N days ago, UTC)."""

    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    body = text[:-1] if text.lower().endswith("d") else text
    if body.isdigit():
        return (datetime.now(timezone.utc) - timedelta(days=int(body))).date()
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        raise EmailInvalidSettings(
            f"The {label} value {text!r} is not a date.",
            f"Give {label} as an ISO date (2026-09-01), an ISO date-time, or a number of days ago (7d).",
        )
    return dt.date()


def _ascii(text: str) -> bool:
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return False
    return True


@dataclass(frozen=True)
class SearchCriteria:
    from_address: str = ""
    from_domain: str = ""
    to_address: str = ""
    subject_contains: str = ""
    since: Optional[date] = None
    before: Optional[date] = None
    unseen: Optional[bool] = None
    has_attachment: Optional[bool] = None

    @classmethod
    def build(
        cls,
        *,
        from_address: Any = "",
        from_domain: Any = "",
        to_address: Any = "",
        subject_contains: Any = "",
        since: Any = None,
        before: Any = None,
        unseen: Optional[bool] = None,
        has_attachment: Optional[bool] = None,
    ) -> "SearchCriteria":
        fa = str(from_address or "").strip()
        fd = str(from_domain or "").strip()
        ta = str(to_address or "").strip()
        sc = str(subject_contains or "")
        if any(c in sc for c in "\r\n"):
            raise EmailInvalidSettings("The subject text contains a line break.", "Give the subject text on one line.")
        try:
            if fa:
                normalize_address(fa)
            if ta:
                normalize_address(ta)
        except ValueError:
            raise EmailInvalidSettings(
                "A search address is not a valid email address.",
                "Give from_address / to_address as name@example.test (use from_domain for a whole domain).",
            )
        if fd:
            try:
                normalize_domain(fd)
            except ValueError:
                raise EmailInvalidSettings(
                    f"The search domain {fd!r} is not a domain.",
                    "Give from_domain as example.test (no wildcards).",
                )
        return cls(
            from_address=fa,
            from_domain=fd,
            to_address=ta,
            subject_contains=sc.strip(),
            since=parse_day(since, label="since"),
            before=parse_day(before, label="before"),
            unseen=unseen if unseen in (True, False) else None,
            has_attachment=has_attachment if has_attachment in (True, False) else None,
        )

    def imap_keys(self) -> List[str]:
        keys: List[str] = []
        if self.from_address and _ascii(self.from_address):
            keys += ["FROM", quote(self.from_address)]
        elif self.from_domain:
            dom = normalize_domain(self.from_domain)
            keys += ["FROM", quote("@" + dom)]
        if self.to_address and _ascii(self.to_address):
            keys += ["OR", "TO", quote(self.to_address), "CC", quote(self.to_address)]
        if self.subject_contains and _ascii(self.subject_contains):
            keys += ["SUBJECT", quote(self.subject_contains)]
        if self.since is not None:
            keys += ["SINCE", imap_date(self.since)]
        if self.before is not None:
            keys += ["BEFORE", imap_date(self.before)]
        if self.unseen is True:
            keys.append("UNSEEN")
        elif self.unseen is False:
            keys.append("SEEN")
        return keys or ["ALL"]

    def matches(
        self,
        *,
        from_header: str,
        to_header: str,
        cc_header: str,
        subject: str,
        seen: bool,
        has_attachments: Optional[bool] = None,
    ) -> bool:
        if self.from_address or self.from_domain:
            senders = [a for _n, a in getaddresses([from_header or ""]) if a]
            try:
                parts = [split_address(a) for a in senders]
            except ValueError:
                return False
            if self.from_address:
                want = normalize_address(self.from_address)
                if not any(f"{l}@{d}" == want for l, d in parts):
                    return False
            if self.from_domain:
                dom = normalize_domain(self.from_domain)
                if not any(d == dom for _l, d in parts):
                    return False
        if self.to_address:
            want = normalize_address(self.to_address)
            found = False
            for _n, a in getaddresses([to_header or "", cc_header or ""]):
                try:
                    if a and normalize_address(a) == want:
                        found = True
                        break
                except ValueError:
                    continue
            if not found:
                return False
        if self.subject_contains and self.subject_contains.casefold() not in (subject or "").casefold():
            return False
        if self.unseen is True and seen:
            return False
        if self.unseen is False and not seen:
            return False
        if self.has_attachment is not None and has_attachments is not self.has_attachment:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "from_address": self.from_address or None,
            "from_domain": self.from_domain or None,
            "to_address": self.to_address or None,
            "subject_contains": self.subject_contains or None,
            "since": self.since.isoformat() if self.since else None,
            "before": self.before.isoformat() if self.before else None,
            "unseen": self.unseen,
            "has_attachment": self.has_attachment,
        }
