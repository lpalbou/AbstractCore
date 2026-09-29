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
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from .client import EmailClient
from .errors import EmailDisabled, EmailInvalidMessage, EmailNotConfigured
from .limits import SendRateLimiter
from .models import EmailAccount, EmailSecret, OutgoingMessage, SendLimits, SendResult
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
    on_sent: Optional[Callable[[Dict[str, Any]], None]] = field(default=None, repr=False)

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
        )

    def evaluate(self, message: OutgoingMessage) -> PolicyDecision:
        try:
            to = parse_recipients(list(message.to))
            cc = parse_recipients(list(message.cc))
            bcc = parse_recipients(list(message.bcc))
        except ValueError as exc:
            raise EmailInvalidMessage(f"A recipient is not valid: {exc}.", "Give recipients as name@example.test.") from None
        return evaluate(self.policy, to=to, cc=cc, bcc=bcc)

    def send(self, message: OutgoingMessage) -> SendResult:
        return guarded_send(self, message)


def guarded_send(ctx: EmailContext, message: OutgoingMessage) -> SendResult:
    ctx.require_enabled()
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
            ctx.on_sent({"message_id": result.message_id, "accepted": list(result.accepted), "refused": dict(result.refused)})
        except Exception:
            pass
    return result
