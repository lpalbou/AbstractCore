"""`EmailContext`: everything one caller needs to use one account, and the guarded send.

A context bundles the account, its secret (or token provider), the recipient policy, the send
limits (with their durable counter) and the enabled switch. The core store builds one for
the local account (`EmailAccountStore.context()`); a host (the runtime / gateway) builds one
per user from its own per-user store and hands it to the tools through
`abstractcore.tools.comms_tools.use_email_context(...)` or `set_email_account_resolver(...)`.

`guarded_send` is the ONLY way the tools, the CLI and the consoles send mail:

    enabled?  ->  account can send?  ->  recipient policy (To, Cc, Bcc; whole message)
              ->  send limits (hour, day)  ->  SMTP

Approval (whether a send may run unattended) is the host's gate and runs BEFORE this; the
policy and the limits run here, always, for every caller.
"""

from __future__ import annotations

import ssl
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, Optional

from .client import DEFAULT_MAX_MESSAGE_BYTES, EmailClient
from .errors import EmailDisabled, EmailInvalidMessage, EmailNotConfigured
from .limits import SendRateLimiter
from .models import (
    AUTO_SUBMITTED_GENERATED,
    AUTO_SUBMITTED_REPLIED,
    EmailAccount,
    EmailSecret,
    OutgoingMessage,
    SendLimits,
    SendResult,
    automation_marker_value,
)
from .oauth import OAuthTokenProvider
from .policy import PolicyDecision, RecipientPolicy, evaluate, parse_recipients


@dataclass
class EmailContext:
    account: EmailAccount
    secret: Optional[EmailSecret]
    policy: RecipientPolicy
    limits: SendLimits
    limiter: SendRateLimiter
    enabled: bool = True
    token_provider: Optional[OAuthTokenProvider] = None
    ssl_context: Optional[ssl.SSLContext] = None
    timeout: float = 30.0
    source: str = "settings"
    registered_address: str = ""
    # The reading limit: text/html bodies of one message (or one attachment) larger than this
    # are not fetched; the read returns a typed skip record instead (never a cut body).
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES
    on_sent: Optional[Callable[[Dict[str, Any]], None]] = field(default=None, repr=False)
    # Set by a host for a context used by something that sends automatically (an automation
    # occurrence, a notification): every message sent through it carries RFC 3834
    # `Auto-Submitted` and the framework marker (`X-AbstractFramework-Automation: <value>`),
    # so this account's own mail watcher never admits it (no self-triggering loop).
    automation_marker: str = ""

    def __repr__(self) -> str:
        return f"EmailContext(address={self.account.address!r}, enabled={self.enabled}, source={self.source!r})"

    def require_enabled(self) -> None:
        if not self.enabled:
            raise EmailDisabled(
                "Email is turned off for this account.",
                "Turn it on in the email settings (`abstractcore email enable`); an administrator may have turned it off.",
            )

    def client(self) -> EmailClient:
        self.require_enabled()
        return EmailClient(
            self.account,
            self.secret,
            token_provider=self.token_provider,
            timeout=self.timeout,
            ssl_context=self.ssl_context,
            max_message_bytes=self.max_message_bytes,
        )

    def evaluate(self, message: OutgoingMessage) -> PolicyDecision:
        try:
            to = parse_recipients(list(message.to))
            cc = parse_recipients(list(message.cc))
            bcc = parse_recipients(list(message.bcc))
        except ValueError as exc:
            raise EmailInvalidMessage(f"A recipient is not valid: {exc}.", "Give recipients as name@example.test.") from None
        return evaluate(self.policy, to=to, cc=cc, bcc=bcc, self_addresses=self.self_addresses())

    def self_addresses(self) -> tuple:
        """The account's own addresses (registered address, mailbox address): always allowed."""

        return tuple(a for a in (self.registered_address, self.account.address if self.account else "") if a)

    def send(self, message: OutgoingMessage) -> SendResult:
        return guarded_send(self, message)


def mark_automatic(message: OutgoingMessage, marker: str) -> OutgoingMessage:
    """`message` with the framework marker and RFC 3834 `Auto-Submitted` (kept when set).

    `Auto-Submitted` is `auto-replied` for an answer to one message (In-Reply-To set), else
    `auto-generated`.
    """

    marker_value = automation_marker_value(marker)
    if not marker_value:
        raise EmailInvalidMessage(
            "The automation marker must be one line of printable ASCII (at most 200 characters).",
            "Give an identifier such as occurrence:<run id>.",
        )
    auto = message.auto_submitted or (AUTO_SUBMITTED_REPLIED if message.in_reply_to else AUTO_SUBMITTED_GENERATED)
    return replace(message, auto_submitted=auto, automation_marker=message.automation_marker or marker_value)


def guarded_send(ctx: EmailContext, message: OutgoingMessage) -> SendResult:
    ctx.require_enabled()
    if ctx.automation_marker:
        message = mark_automatic(message, ctx.automation_marker)
    if not ctx.account.can_send:
        raise EmailNotConfigured(
            "This account has no SMTP (send) settings.",
            "Connect the account again with --smtp-host to send mail.",
        )
    decision = ctx.evaluate(message)
    decision.raise_if_refused()
    client = ctx.client()
    token = ctx.limiter.reserve()
    try:
        result = client.send(message)
    except Exception:
        # SMTP did not accept the message: it does not count against the limits.
        ctx.limiter.refund(token)
        raise
    if ctx.on_sent is not None:
        try:
            ctx.on_sent(
                {
                    "message_id": result.message_id,
                    "accepted": list(result.accepted),
                    "refused": dict(result.refused),
                    "auto_submitted": message.auto_submitted,
                    "automation_marker": message.automation_marker,
                }
            )
        except Exception:
            pass
    return result
