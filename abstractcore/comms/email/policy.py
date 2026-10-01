"""Recipient policy: who CAN receive mail from this account at all.

Deterministic, evaluated before any send (agent tool call, automation action, notification):
two lists and one mode.

- `always_allow` ("Always allowed"): exact addresses or domains always allowed;
- `always_deny` ("Always denied"): exact addresses or domains always refused;
- `mode`, for every recipient on neither list: `allowlist` ("Only the Allowed list") refuses
  it, `denylist` ("Anyone not on the Denied list") allows it.

Precedence, per recipient (one function: `evaluate`):

1. the account's own address (registered address, mailbox address) -> allowed;
2. on Always denied -> refused ("Not sent: <recipient> is on your Always denied list.");
3. on Always allowed -> allowed;
4. else the mode: allowlist -> refused, denylist -> allowed.

Denied always wins over allowed. Entries are exact addresses (`me@example.test`) or domains
(`example.test`, also written `@example.test`); a domain covers that domain AND its
subdomains (dot-suffix: `example.test` covers `a@mail.example.test`, never
`a@badexample.test`). There is no pattern language: `*`, `?`, regular expressions and partial
strings are refused when an entry is added (structural check only).

The policy applies to To, Cc and Bcc. A message with ANY refused recipient is refused as a
whole, and the refusal names each refused address, the field it was in and the rule that
refused it.

Stored shape: `{mode, entries, always_allow, always_deny}`; `entries` repeats the list the
mode uses (allowlist -> always_allow, denylist -> always_deny), so readers of the older
`{mode, entries}` shape keep their meaning. A stored policy WITHOUT the two lists (written
before they existed) migrates on load: an allowlist's entries become Always allowed, a
denylist's entries become Always denied; the mode is kept.

Normalisation: the domain is compared case-insensitively after IDNA encoding (so
`bücher.example` and `xn--bcher-kva.example` are one domain) and without a trailing dot; the
local part is compared case-insensitively too (a denied entry must not be bypassed by
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


def _normalize_entries(entries: Iterable[str], label: str) -> Tuple[str, ...]:
    out: List[str] = []
    for e in entries or ():
        try:
            n = normalize_entry(e)
        except ValueError as exc:
            raise EmailInvalidSettings(
                f"The {label} entry is not valid: {exc}.",
                "Write an exact address (name@example.test) or a domain (example.test).",
            ) from exc
        if n not in out:
            out.append(n)
    return tuple(out)


def _stored_list(raw: dict, key: str) -> List[str]:
    value = raw.get(key)
    if not isinstance(value, (list, tuple)):
        return []
    return [str(e) for e in value]


def entry_matches(entry: str, local: str, domain: str) -> bool:
    """Whether one normalised Always allowed / Always denied entry covers a recipient.

    An address entry matches that exact address; a domain entry matches that domain and every
    subdomain of it (dot-suffix: `example.test` covers `mail.example.test`, not
    `badexample.test`)."""

    if "@" in entry:
        return entry == f"{local}@{domain}"
    return domain == entry or domain.endswith("." + entry)


@dataclass(frozen=True)
class RecipientPolicy:
    mode: str = "allowlist"
    always_allow: Tuple[str, ...] = ()
    always_deny: Tuple[str, ...] = ()

    @property
    def entries(self) -> Tuple[str, ...]:
        """The list the mode uses (allowlist -> Always allowed, denylist -> Always denied): the
        older `{mode, entries}` view, kept for readers of that shape."""

        return self.always_allow if self.mode == "allowlist" else self.always_deny

    @classmethod
    def build(
        cls,
        mode: str,
        entries: Iterable[str] = (),
        always_allow: Iterable[str] = (),
        always_deny: Iterable[str] = (),
    ) -> "RecipientPolicy":
        """A validated policy. `entries` (the older shape) are added to the mode's list."""

        m = str(mode or "").strip().lower()
        if m not in POLICY_MODES:
            raise EmailInvalidSettings(
                f"The recipient policy mode {mode!r} is not one of: allowlist, denylist.",
                "Use --mode allowlist (only the Always allowed list) or --mode denylist (anyone not on the Always denied list).",
            )
        allow = list(always_allow or ())
        deny = list(always_deny or ())
        (allow if m == "allowlist" else deny).extend(entries or ())
        return cls(
            mode=m,
            always_allow=_normalize_entries(allow, "Always allowed"),
            always_deny=_normalize_entries(deny, "Always denied"),
        )

    @classmethod
    def default_for(cls, registered_address: str) -> "RecipientPolicy":
        """A new account's policy: Only the Allowed list, holding the user's registered address."""

        entries: List[str] = []
        if registered_address:
            try:
                entries.append(normalize_address(registered_address))
            except ValueError:
                entries = []
        return cls(mode="allowlist", always_allow=tuple(entries))

    @classmethod
    def from_dict(cls, raw: object) -> Optional["RecipientPolicy"]:
        """The stored policy. Without `always_allow`/`always_deny` (stored before they existed)
        the entries migrate into the mode's list (allowlist -> Always allowed, denylist ->
        Always denied); with them, `entries` is only the derived view and is ignored."""

        if not isinstance(raw, dict) or not raw:
            return None
        mode = str(raw.get("mode") or "allowlist")
        if "always_allow" in raw or "always_deny" in raw:
            return cls.build(mode, (), _stored_list(raw, "always_allow"), _stored_list(raw, "always_deny"))
        return cls.build(mode, _stored_list(raw, "entries"))

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "entries": list(self.entries),
            "always_allow": list(self.always_allow),
            "always_deny": list(self.always_deny),
        }

    def with_changes(
        self,
        *,
        mode: Optional[str] = None,
        add: Sequence[str] = (),
        remove: Sequence[str] = (),
        clear: bool = False,
        always_allow: Optional[Sequence[str]] = None,
        always_deny: Optional[Sequence[str]] = None,
    ) -> "RecipientPolicy":
        """A changed copy.

        `add` / `remove` / `clear` act on the list the (new) mode uses, as the older
        `{mode, entries}` interface did: `clear` empties it, `remove` takes exact entries out
        (an unknown entry is refused), `add` appends. Then `always_allow` / `always_deny`, when
        given, REPLACE that list (None keeps it): an explicit list wins over the edits."""

        base = RecipientPolicy.build(str(mode or self.mode), (), self.always_allow, self.always_deny)
        target = list(base.always_allow if base.mode == "allowlist" else base.always_deny)
        if clear:
            target = []
        removed = set()
        for r in remove or ():
            try:
                removed.add(normalize_entry(r))
            except ValueError as exc:
                raise EmailInvalidSettings(
                    f"The entry to remove is not valid: {exc}.",
                    "Write the entry exactly as `abstractcore email policy show` lists it.",
                ) from exc
        missing = sorted(removed - set(target))
        if missing:
            raise EmailInvalidSettings(
                f"The recipient policy has no entry {missing[0]!r}.",
                "Write the entry exactly as `abstractcore email policy show` lists it.",
            )
        target = [e for e in target if e not in removed] + list(add or ())
        allow = target if base.mode == "allowlist" else list(base.always_allow)
        deny = target if base.mode == "denylist" else list(base.always_deny)
        return RecipientPolicy.build(
            base.mode,
            (),
            allow if always_allow is None else list(always_allow),
            deny if always_deny is None else list(always_deny),
        )


@dataclass(frozen=True)
class RecipientVerdict:
    address: str
    field: str  # "to" | "cc" | "bcc"
    allowed: bool
    rule: str  # the entry that decided, or the mode's default rule
    reason: str
    # Which step of the precedence decided: "self", "always_deny", "always_allow" or "mode".
    source: str = "mode"

    def to_dict(self) -> dict:
        return {
            "address": self.address,
            "field": self.field,
            "allowed": self.allowed,
            "rule": self.rule,
            "reason": self.reason,
            "source": self.source,
        }


def always_denied_sentence(verdict: "RecipientVerdict") -> str:
    """"Not sent: <recipient> is on your Always denied list." (+ the entry when it is a domain)."""

    try:
        norm = normalize_address(verdict.address)
    except ValueError:
        norm = verdict.address
    suffix = "" if verdict.rule == norm else f" ({verdict.rule})"
    return f"Not sent: {verdict.address} is on your Always denied list{suffix}."


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
        denied = [v for v in refused if v.source == "always_deny"]
        others = [v for v in refused if v.source != "always_deny"]
        parts = [always_denied_sentence(v) for v in denied]
        if others:
            listed = "; ".join(f"{v.address} ({v.field}: {v.reason})" for v in others)
            parts.append(
                f"The recipient policy ({self.mode}) refused the whole message; nothing was sent. Refused: {listed}."
            )
        elif len(self.verdicts) > 1:
            parts.append("The whole message was refused; nothing was sent to anyone.")
        raise EmailPolicyRefused(
            " ".join(parts),
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
    self_addresses: Sequence[str] = (),
) -> PolicyDecision:
    """Evaluate every recipient (To, Cc and Bcc) of one message against the policy.

    Precedence per recipient: `self_addresses` (the account's own addresses: the context passes
    the registered address and the mailbox address) -> allowed; `always_deny` -> refused;
    `always_allow` -> allowed; else the mode (allowlist -> refused, denylist -> allowed).

    Self is its own first step, never merged into a list, so the own address is allowed in
    denylist mode too, even when Always denied names it (earlier versions merged it into the
    mode's matched set, so a denylist DENIED the own address when a caller passed it)."""

    selves = set()
    for raw_self in self_addresses or ():
        try:
            selves.add(normalize_address(str(raw_self or "")))
        except ValueError:
            continue
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
            if norm in selves:
                verdicts.append(RecipientVerdict(shown, field_name, True, norm, "your own address", "self"))
                continue
            deny = next((e for e in policy.always_deny if entry_matches(e, local, domain)), "")
            if deny:
                verdicts.append(
                    RecipientVerdict(
                        shown, field_name, False, deny, f"on your Always denied list ({deny})", "always_deny"
                    )
                )
                continue
            allow = next((e for e in policy.always_allow if entry_matches(e, local, domain)), "")
            if allow:
                verdicts.append(
                    RecipientVerdict(
                        shown, field_name, True, allow, f"on your Always allowed list ({allow})", "always_allow"
                    )
                )
                continue
            if policy.mode == "allowlist":
                verdicts.append(
                    RecipientVerdict(shown, field_name, False, "allowlist", "not in the allowlist")
                )
            else:
                verdicts.append(
                    RecipientVerdict(shown, field_name, True, "denylist", "not in the denylist")
                )
    return PolicyDecision(mode=policy.mode, verdicts=tuple(verdicts))
