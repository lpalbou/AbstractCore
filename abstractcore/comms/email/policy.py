"""Recipient policy: who CAN receive mail from this account at all.

Same logic as tool policies, deterministic, evaluated before any send (tool call,
automation action, notification):

- `allowlist`: refuse every recipient except the listed addresses and domains;
- `denylist`: accept every recipient except the listed addresses and domains.

Entries are exact addresses (`me@example.test`) or domains (`example.test`, also written
`@example.test`). A domain entry matches that domain only; a subdomain matches only when it
is written as its own entry. There is no pattern language: `*`, `?`, regular expressions and
partial strings are refused when an entry is added.

The policy applies to To, Cc and Bcc. A message with ANY refused recipient is refused as a
whole, and the refusal names each refused address, the field it was in and the rule that
refused it.

Normalisation: the domain is compared case-insensitively after IDNA encoding (so
`bücher.example` and `xn--bcher-kva.example` are one domain) and without a trailing dot; the
local part is compared case-insensitively too (a denylist entry must not be bypassed by
changing letter case). Display names are ignored: `Name <a@b>` is the address `a@b`.

The approval gate still applies on top of the policy: the policy decides who CAN receive
mail; approval decides whether a given send runs unattended.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from email.utils import getaddresses
from typing import Iterable, List, Optional, Sequence, Tuple

from .errors import EmailInvalidSettings, EmailPolicyRefused

POLICY_MODES = ("allowlist", "denylist")

# Characters a domain label may hold after IDNA encoding (RFC 1035 LDH rule).
_LDH = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")
# Characters never valid in an addr-spec local part we accept (whitespace, controls, brackets
# and separators). Quoted local parts are not supported.
_LOCAL_FORBIDDEN = frozenset(" \t\r\n<>()[],;:\\\"@")
_PATTERN_CHARS = frozenset("*?^$|+{}[]()\\")


def normalize_domain(raw: str) -> str:
    """Lower-case, IDNA-encoded, no trailing dot. Raises ValueError when not a domain."""

    text = str(raw or "").strip()
    if text.startswith("@"):
        text = text[1:]
    if text.endswith("."):
        text = text[:-1]
    if not text:
        raise ValueError("empty domain")
    if any(c in _PATTERN_CHARS for c in text):
        raise ValueError("patterns are not supported; write the exact domain")
    try:
        ascii_domain = text.encode("idna").decode("ascii").lower()
    except UnicodeError as exc:
        raise ValueError("not a valid domain name") from exc
    labels = ascii_domain.split(".")
    if len(labels) < 2:
        raise ValueError("a domain needs at least two labels (example.test)")
    for label in labels:
        if not label or len(label) > 63:
            raise ValueError("not a valid domain name")
        if label.startswith("-") or label.endswith("-"):
            raise ValueError("not a valid domain name")
        if any(c not in _LDH for c in label):
            raise ValueError("not a valid domain name")
    return ascii_domain


def split_address(raw: str) -> Tuple[str, str]:
    """(local, domain) of one bare addr-spec, normalised. Raises ValueError when invalid."""

    text = str(raw or "").strip()
    if text.count("@") != 1:
        raise ValueError("not a valid email address")
    local, domain = text.split("@", 1)
    if not local or any(c in _LOCAL_FORBIDDEN for c in local) or any(ord(c) < 32 for c in local):
        raise ValueError("not a valid email address")
    if any(c in "*?" for c in local):
        raise ValueError("patterns are not supported; write the exact address")
    if local.startswith(".") or local.endswith(".") or ".." in local:
        raise ValueError("not a valid email address")
    return local.lower(), normalize_domain(domain)


def normalize_address(raw: str) -> str:
    """The comparable form of one bare address: lower-case local part @ IDNA domain."""

    local, domain = split_address(raw)
    return f"{local}@{domain}"


def parse_recipients(value: object) -> List[str]:
    """Bare addresses from a recipient field (a string with `,`-separated entries, or a list).

    Display names are dropped (`Name <a@b>` -> `a@b`); each address keeps its original
    spelling (the SMTP envelope gets exactly what the user typed, the policy compares the
    normalised form). Raises ValueError naming the first entry that is not an address.
    """

    items: List[str] = []
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple)):
        items = [str(v) for v in value if v is not None]
    else:
        items = [str(value)]
    out: List[str] = []
    for item in items:
        if not str(item).strip():
            continue
        # `;` is a common separator typed by people; RFC 5322 uses `,`.
        text = str(item).replace(";", ",")
        pairs = getaddresses([text])
        if not pairs:
            raise ValueError(f"not a valid recipient list: {item!r}")
        for _name, addr in pairs:
            addr = str(addr or "").strip()
            if not addr:
                raise ValueError(f"not a valid email address: {item!r}")
            split_address(addr)  # validates
            out.append(addr)
    return out


def normalize_entry(raw: str) -> str:
    """One policy entry, normalised: `local@domain` for an address, `domain` for a domain."""

    text = str(raw or "").strip()
    if not text:
        raise ValueError("empty entry")
    if any(c in _PATTERN_CHARS for c in text):
        raise ValueError(f"{text!r}: patterns are not supported; write an exact address or domain")
    if text.startswith("@"):
        return normalize_domain(text)
    if "@" in text:
        try:
            return normalize_address(text)
        except ValueError as exc:
            raise ValueError(f"{text!r}: {exc}") from exc
    try:
        return normalize_domain(text)
    except ValueError as exc:
        raise ValueError(f"{text!r}: {exc}") from exc


@dataclass(frozen=True)
class RecipientPolicy:
    mode: str = "allowlist"
    entries: Tuple[str, ...] = ()

    @classmethod
    def build(cls, mode: str, entries: Iterable[str]) -> "RecipientPolicy":
        m = str(mode or "").strip().lower()
        if m not in POLICY_MODES:
            raise EmailInvalidSettings(
                f"The recipient policy mode {mode!r} is not one of: allowlist, denylist.",
                "Use --mode allowlist (only the listed recipients) or --mode denylist (everyone except the listed recipients).",
            )
        out: List[str] = []
        for e in entries or ():
            try:
                n = normalize_entry(e)
            except ValueError as exc:
                raise EmailInvalidSettings(
                    f"The recipient policy entry is not valid: {exc}.",
                    "Write an exact address (name@example.test) or a domain (example.test).",
                ) from exc
            if n not in out:
                out.append(n)
        return cls(mode=m, entries=tuple(out))

    @classmethod
    def default_for(cls, registered_address: str) -> "RecipientPolicy":
        """A new account's policy: allowlist holding only the user's registered address."""

        entries: List[str] = []
        if registered_address:
            try:
                entries.append(normalize_address(registered_address))
            except ValueError:
                entries = []
        return cls(mode="allowlist", entries=tuple(entries))

    @classmethod
    def from_dict(cls, raw: object) -> Optional["RecipientPolicy"]:
        if not isinstance(raw, dict) or not raw:
            return None
        entries = raw.get("entries")
        if not isinstance(entries, (list, tuple)):
            entries = []
        return cls.build(str(raw.get("mode") or "allowlist"), [str(e) for e in entries])

    def to_dict(self) -> dict:
        return {"mode": self.mode, "entries": list(self.entries)}

    def with_changes(
        self,
        *,
        mode: Optional[str] = None,
        add: Sequence[str] = (),
        remove: Sequence[str] = (),
        clear: bool = False,
    ) -> "RecipientPolicy":
        entries = [] if clear else list(self.entries)
        removed = set()
        for r in remove or ():
            try:
                removed.add(normalize_entry(r))
            except ValueError as exc:
                raise EmailInvalidSettings(
                    f"The entry to remove is not valid: {exc}.",
                    "Write the entry exactly as `abstractcore email policy show` lists it.",
                ) from exc
        missing = sorted(removed - set(entries))
        if missing:
            raise EmailInvalidSettings(
                f"The recipient policy has no entry {missing[0]!r}.",
                "Write the entry exactly as `abstractcore email policy show` lists it.",
            )
        entries = [e for e in entries if e not in removed]
        return RecipientPolicy.build(mode or self.mode, entries + list(add or ()))


@dataclass(frozen=True)
class RecipientVerdict:
    address: str
    field: str  # "to" | "cc" | "bcc"
    allowed: bool
    rule: str  # the entry that decided, or the mode's default rule
    reason: str

    def to_dict(self) -> dict:
        return {
            "address": self.address,
            "field": self.field,
            "allowed": self.allowed,
            "rule": self.rule,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class PolicyDecision:
    mode: str
    verdicts: Tuple[RecipientVerdict, ...] = field(default_factory=tuple)

    @property
    def allowed(self) -> bool:
        return bool(self.verdicts) and all(v.allowed for v in self.verdicts)

    @property
    def refused(self) -> List[RecipientVerdict]:
        return [v for v in self.verdicts if not v.allowed]

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "allowed": self.allowed,
            "recipients": [v.to_dict() for v in self.verdicts],
            "refused": [v.to_dict() for v in self.refused],
        }

    def raise_if_refused(self) -> None:
        if not self.verdicts:
            raise EmailPolicyRefused(
                "The message has no recipient.",
                "Give at least one recipient in To, Cc or Bcc.",
            )
        refused = self.refused
        if not refused:
            return
        listed = "; ".join(f"{v.address} ({v.field}: {v.reason})" for v in refused)
        raise EmailPolicyRefused(
            f"The recipient policy ({self.mode}) refused the whole message; nothing was sent. Refused: {listed}.",
            "Remove the refused recipients, or change the recipient policy in the email settings "
            "(`abstractcore email policy set --add <address or domain>`).",
            details={"mode": self.mode, "refused": [v.to_dict() for v in refused]},
        )


def evaluate(
    policy: RecipientPolicy,
    *,
    to: Sequence[str] = (),
    cc: Sequence[str] = (),
    bcc: Sequence[str] = (),
) -> PolicyDecision:
    """Evaluate every recipient of one message against the policy."""

    addresses = {e for e in policy.entries if "@" in e}
    domains = {e for e in policy.entries if "@" not in e}
    verdicts: List[RecipientVerdict] = []
    for field_name, values in (("to", to), ("cc", cc), ("bcc", bcc)):
        for raw in values or ():
            shown = str(raw or "").strip()
            try:
                local, domain = split_address(shown)
            except ValueError:
                verdicts.append(RecipientVerdict(shown, field_name, False, "address", "not a valid email address"))
                continue
            norm = f"{local}@{domain}"
            matched = norm if norm in addresses else (domain if domain in domains else "")
            if policy.mode == "allowlist":
                if matched:
                    verdicts.append(RecipientVerdict(shown, field_name, True, matched, f"allowed by {matched}"))
                else:
                    verdicts.append(
                        RecipientVerdict(shown, field_name, False, "allowlist", "not in the allowlist")
                    )
            else:
                if matched:
                    verdicts.append(RecipientVerdict(shown, field_name, False, matched, f"denied by {matched}"))
                else:
                    verdicts.append(RecipientVerdict(shown, field_name, True, "denylist", "not in the denylist"))
    return PolicyDecision(mode=policy.mode, verdicts=tuple(verdicts))
