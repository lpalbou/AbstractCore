"""Opt-in LIVE email test harness: a real test mailbox, credentials from the environment only.

The environment variables below are the TEST HARNESS's input, nothing else. Product code never
reads them: the tests build the account through AbstractCore's own settings types
(`EmailAccount`, `EmailSecret`, `EmailContext`, `EmailAccountStore`), exactly as a user or a host
would.

    AF_TEST_EMAIL_IMAP_HOST  AF_TEST_EMAIL_IMAP_PORT  AF_TEST_EMAIL_IMAP_SECURITY
    AF_TEST_EMAIL_SMTP_HOST  AF_TEST_EMAIL_SMTP_PORT  AF_TEST_EMAIL_SMTP_SECURITY
    AF_TEST_EMAIL_USERNAME   AF_TEST_EMAIL_ADDRESS    AF_TEST_EMAIL_PASSWORD

Safety rails, all enforced here:

- The values never reach test output: `LiveMailbox` has a redacting `repr`, the password only
  lives inside an `EmailSecret` (redacting `repr`), assertions compare booleans, and the live
  `conftest.py` scrubs every value from failure reports and captured output as a last line.
- Mail goes to the test account's OWN address only: `SmtpSelfOnlyGuard` wraps
  `smtplib.SMTP.sendmail` and refuses (before MAIL FROM) any envelope recipient that is not the
  account's address, and counts every SMTP transaction so a test can prove "no SMTP send".
- Messages are small and tagged `[af-live-test]` plus a random nonce (a read-only mailbox cannot
  be cleaned up).
"""

from __future__ import annotations

import json
import os
import secrets
import smtplib
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

LIVE_VARS: Tuple[str, ...] = (
    "AF_TEST_EMAIL_IMAP_HOST",
    "AF_TEST_EMAIL_IMAP_PORT",
    "AF_TEST_EMAIL_IMAP_SECURITY",
    "AF_TEST_EMAIL_SMTP_HOST",
    "AF_TEST_EMAIL_SMTP_PORT",
    "AF_TEST_EMAIL_SMTP_SECURITY",
    "AF_TEST_EMAIL_USERNAME",
    "AF_TEST_EMAIL_ADDRESS",
    "AF_TEST_EMAIL_PASSWORD",
)
# Values that are not secrets on their own (and too short to scrub without mangling output).
_NOT_SCRUBBED = ("AF_TEST_EMAIL_IMAP_PORT", "AF_TEST_EMAIL_SMTP_PORT", "AF_TEST_EMAIL_IMAP_SECURITY", "AF_TEST_EMAIL_SMTP_SECURITY")

TAG = "[af-live-test]"
REDACTED = "«redacted»"
HOW_TO_RUN = "see CONTRIBUTING.md, 'Live email tests'"
ARRIVAL_TIMEOUT_S = 120.0
POLL_EVERY_S = 10.0
FOREIGN = "af-live-test@example.invalid"  # never deliverable (RFC 2606); the policy must refuse it first


def missing_vars(environ: Optional[Dict[str, str]] = None) -> List[str]:
    env = os.environ if environ is None else environ
    return [name for name in LIVE_VARS if not str(env.get(name, "")).strip()]


def skip_reason(missing: Sequence[str]) -> str:
    return (
        "live email tests need a real test mailbox in the environment; missing: "
        + ", ".join(missing)
        + f" ({HOW_TO_RUN})"
    )


class LiveMailbox:
    """The test mailbox's settings. Holds the values privately; `repr`/`str` never show them."""

    __slots__ = ("_env",)

    def __init__(self, environ: Optional[Dict[str, str]] = None) -> None:
        env = os.environ if environ is None else environ
        self._env = {name: str(env.get(name, "")).strip() for name in LIVE_VARS}

    def __repr__(self) -> str:
        return f"LiveMailbox({REDACTED})"

    __str__ = __repr__

    def __reduce__(self):  # never pickled into a cache or a report
        raise TypeError("LiveMailbox is not picklable")

    # -- values (use them, never print them) -----------------------------------------------

    @property
    def address(self) -> str:
        return self._env["AF_TEST_EMAIL_ADDRESS"]

    @property
    def username(self) -> str:
        return self._env["AF_TEST_EMAIL_USERNAME"]

    @property
    def imap_host(self) -> str:
        return self._env["AF_TEST_EMAIL_IMAP_HOST"]

    @property
    def smtp_host(self) -> str:
        return self._env["AF_TEST_EMAIL_SMTP_HOST"]

    @property
    def imap_port(self) -> int:
        return int(self._env["AF_TEST_EMAIL_IMAP_PORT"])

    @property
    def smtp_port(self) -> int:
        return int(self._env["AF_TEST_EMAIL_SMTP_PORT"])

    @property
    def imap_security(self) -> str:
        return self._env["AF_TEST_EMAIL_IMAP_SECURITY"].lower()

    @property
    def smtp_security(self) -> str:
        return self._env["AF_TEST_EMAIL_SMTP_SECURITY"].lower()

    def secret(self):
        from abstractcore.comms.email import EmailSecret

        return EmailSecret(self._env["AF_TEST_EMAIL_PASSWORD"])

    def is_self(self, address: str) -> bool:
        from abstractcore.comms.email import normalize_address

        try:
            return normalize_address(address) == normalize_address(self.address)
        except ValueError:
            return False

    def contains_secret(self, data: Any) -> bool:
        """True when the password occurs in `data` (str or bytes). Report the boolean only."""

        pw = self._env["AF_TEST_EMAIL_PASSWORD"]
        if isinstance(data, (bytes, bytearray)):
            return pw.encode("utf-8") in bytes(data)
        return pw in str(data)

    def scrub_values(self) -> List[str]:
        values = set()
        for name in LIVE_VARS:
            if name in _NOT_SCRUBBED:
                continue
            v = self._env.get(name, "")
            if len(v) >= 3:
                values.update({v, v.lower()})
        addr = self._env.get("AF_TEST_EMAIL_ADDRESS", "")
        if "@" in addr:
            domain = addr.split("@", 1)[1]
            if len(domain) >= 4:
                values.update({domain, domain.lower()})
        return sorted(values, key=len, reverse=True)

    # -- AbstractCore settings built from them ----------------------------------------------

    def account(self, *, imap_security: Optional[str] = None, imap_port: Optional[int] = None,
                smtp_security: Optional[str] = None, smtp_port: Optional[int] = None):
        from abstractcore.comms.email import EmailAccount, ImapSettings, SmtpSettings

        return EmailAccount.build(
            address=self.address,
            username=self.username,
            imap=ImapSettings.build(self.imap_host, port=imap_port or self.imap_port, security=imap_security or self.imap_security),
            smtp=SmtpSettings.build(self.smtp_host, port=smtp_port or self.smtp_port, security=smtp_security or self.smtp_security),
        )

    def context(self, state_dir: Path, *, per_hour: int = 20, per_day: int = 100, policy_entries: Optional[List[str]] = None):
        """An in-memory `EmailContext` (what a host builds per user): allowlist = own address."""

        from abstractcore.comms.email import EmailContext, RecipientPolicy, SendLimits, SendRateLimiter

        limits = SendLimits.build(per_hour, per_day)
        state_dir = Path(state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        return EmailContext(
            account=self.account(),
            secret=self.secret(),
            policy=RecipientPolicy.build("allowlist", policy_entries if policy_entries is not None else [self.address]),
            limits=limits,
            limiter=SendRateLimiter(state_dir / f"sends-{secrets.token_hex(4)}.json", limits),
            registered_address=self.address,
        )


def load_live_mailbox() -> Tuple[Optional[LiveMailbox], List[str]]:
    missing = missing_vars()
    if missing:
        return None, missing
    return LiveMailbox(), []


def scrub(text: Any, mailbox: Optional[LiveMailbox]) -> str:
    out = str(text)
    if mailbox is None:
        return out
    for value in mailbox.scrub_values():
        out = out.replace(value, REDACTED)
    return out


def new_nonce() -> str:
    return secrets.token_hex(8)


def tagged_subject(kind: str, nonce: str) -> str:
    return f"{TAG} {kind} {nonce}"


def fact(mailbox: Optional[LiveMailbox], label: str, value: Any) -> None:
    """One `LIVE-FACT` line for the report (run pytest with -s to see them); scrubbed."""

    print(f"LIVE-FACT {label}: {scrub(json.dumps(value, default=str, sort_keys=True), mailbox)}", flush=True)


class SmtpSelfOnlyGuard:
    """Wraps `smtplib.SMTP.sendmail`: refuses any recipient but the account's own address.

    Every SMTP transaction (allowed or refused) is recorded as `{n_rcpts, all_self}`, so a test
    can assert that NO SMTP send happened. The refusal raises before MAIL FROM is sent.
    """

    def __init__(self, mailbox: LiveMailbox) -> None:
        self.mailbox = mailbox
        self.transactions: List[Dict[str, Any]] = []

    @property
    def count(self) -> int:
        return len(self.transactions)

    def install(self, monkeypatch) -> "SmtpSelfOnlyGuard":
        real = smtplib.SMTP.sendmail
        guard = self

        def sendmail(smtp_self, from_addr, to_addrs, msg, *args, **kwargs):
            rcpts = [to_addrs] if isinstance(to_addrs, str) else list(to_addrs or [])
            all_self = bool(rcpts) and all(guard.mailbox.is_self(r) for r in rcpts)
            guard.transactions.append({"n_rcpts": len(rcpts), "all_self": all_self})
            if not all_self:
                raise AssertionError(
                    "live test guard: refused an SMTP send to an address other than the test account's own"
                )
            return real(smtp_self, from_addr, to_addrs, msg, *args, **kwargs)

        monkeypatch.setattr(smtplib.SMTP, "sendmail", sendmail)
        return self


def wait_for(probe: Callable[[], Any], *, timeout_s: float = ARRIVAL_TIMEOUT_S, every_s: float = POLL_EVERY_S) -> Tuple[Any, float, int]:
    """Call `probe()` until it returns a truthy value: (value, elapsed seconds, attempts)."""

    start = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        value = probe()
        elapsed = time.monotonic() - start
        if value:
            return value, elapsed, attempts
        if elapsed + every_s > timeout_s:
            return None, elapsed, attempts
        time.sleep(every_s)
