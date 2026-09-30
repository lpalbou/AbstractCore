"""`abstractcore.comms.email` — the framework's one mail implementation.

    from abstractcore.comms.email import (
        EmailAccount, ImapSettings, SmtpSettings, EmailSecret, EmailClient,
        EmailAccountStore, RecipientPolicy, SendLimits, SearchCriteria, OutgoingMessage,
    )

    account = EmailAccount.build(
        address="me@example.test",
        imap=ImapSettings.build("imap.example.test", security="ssl"),
        smtp=SmtpSettings.build("smtp.example.test", security="starttls"),
    )
    store = EmailAccountStore()                     # the local AbstractCore settings
    store.connect(account, EmailSecret("app-password"))
    ctx = store.context()
    ctx.client().search(SearchCriteria.build(unseen=True, since="7d"), limit=20)
    ctx.send(OutgoingMessage(to=("me@example.test",), subject="Hello", text="Hi"))

Modules: `models` (typed settings), `client` (IMAP read-only + SMTP, verified TLS),
`bodystructure` (the MIME structure used to fetch only the parts needed),
`policy` (recipient allowlist / denylist), `limits` (send rate limits), `vault` (encrypted
credentials), `oauth` (OAuth2 / XOAUTH2), `store` (settings + secret + status), `context`
(the guarded send), `errors` (typed errors), `legacy` (one-time import of pre-2.20 settings),
`discovery` (a mailbox's IMAP/SMTP servers from its address: known providers, autoconfig, ISPDB,
DNS SRV, MX).
"""

from .client import DEFAULT_MAX_MESSAGE_BYTES, MAX_LIST_LIMIT, EmailClient, auto_submitted_value, tls_context
from .context import EmailContext, guarded_send, mark_automatic
from .errors import (
    EmailAgentToolsOff,
    EmailAttachmentNotFound,
    EmailAuthFailed,
    EmailDisabled,
    EmailError,
    EmailInvalidMessage,
    EmailInvalidSettings,
    EmailMailboxMissing,
    EmailMessageNotFound,
    EmailMessageTooLarge,
    EmailNotConfigured,
    EmailOAuthFailed,
    EmailOAuthPending,
    EmailOAuthReauthorize,
    EmailPolicyRefused,
    EmailProtocolError,
    EmailQuotaExceeded,
    EmailRateLimited,
    EmailReadOnlyViolation,
    EmailRecipientRefused,
    EmailSecretUnavailable,
    EmailServerError,
    EmailTlsFailed,
    EmailTransient,
    EmailUnreachable,
)
from .discovery import EmailDiscoveryFailed, discover_servers, require_servers
from .limits import SendRateLimiter
from .models import (
    AUTO_SUBMITTED_GENERATED,
    AUTO_SUBMITTED_HEADER,
    AUTO_SUBMITTED_REPLIED,
    AUTO_SUBMITTED_VALUES,
    AUTOMATION_MARKER_HEADER,
    Attachment,
    AttachmentInfo,
    EmailAccount,
    EmailSecret,
    FetchResult,
    ImapSettings,
    MailCursor,
    MessageDetail,
    MessageSummary,
    OAuthSettings,
    OutgoingMessage,
    SendLimits,
    SendResult,
    SmtpSettings,
    automation_marker_value,
)
from .oauth import (
    BUILTIN_CLIENTS,
    DeviceAuthorization,
    LoopbackAuthorization,
    OAuthTokenClient,
    OAuthTokenProvider,
    TokenSet,
    builtin_client,
    oauth_providers_public,
    provider_preset,
    resolve_oauth_client,
    xoauth2_string,
)
from .policy import PolicyDecision, RecipientPolicy, evaluate, normalize_address, normalize_domain, parse_recipients
from .search import SearchCriteria
from .store import EmailAccountStore, EmailSettings
from .vault import SecretVault

__all__ = [
    "AUTO_SUBMITTED_GENERATED",
    "AUTO_SUBMITTED_HEADER",
    "AUTO_SUBMITTED_REPLIED",
    "AUTO_SUBMITTED_VALUES",
    "AUTOMATION_MARKER_HEADER",
    "auto_submitted_value",
    "automation_marker_value",
    "mark_automatic",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "MAX_LIST_LIMIT",
    "BUILTIN_CLIENTS",
    "Attachment",
    "AttachmentInfo",
    "DeviceAuthorization",
    "EmailAccount",
    "EmailAccountStore",
    "EmailAgentToolsOff",
    "EmailAttachmentNotFound",
    "EmailAuthFailed",
    "EmailClient",
    "EmailContext",
    "EmailDisabled",
    "EmailDiscoveryFailed",
    "EmailError",
    "EmailInvalidMessage",
    "EmailInvalidSettings",
    "EmailMailboxMissing",
    "EmailMessageNotFound",
    "EmailMessageTooLarge",
    "EmailNotConfigured",
    "EmailOAuthFailed",
    "EmailOAuthPending",
    "EmailOAuthReauthorize",
    "EmailPolicyRefused",
    "EmailProtocolError",
    "EmailQuotaExceeded",
    "EmailRateLimited",
    "EmailReadOnlyViolation",
    "EmailRecipientRefused",
    "EmailSecret",
    "EmailSecretUnavailable",
    "EmailServerError",
    "EmailSettings",
    "EmailTlsFailed",
    "EmailTransient",
    "EmailUnreachable",
    "FetchResult",
    "ImapSettings",
    "LoopbackAuthorization",
    "MailCursor",
    "MessageDetail",
    "MessageSummary",
    "OAuthSettings",
    "OAuthTokenClient",
    "OAuthTokenProvider",
    "OutgoingMessage",
    "PolicyDecision",
    "RecipientPolicy",
    "SearchCriteria",
    "SecretVault",
    "SendLimits",
    "SendRateLimiter",
    "SendResult",
    "SmtpSettings",
    "TokenSet",
    "builtin_client",
    "discover_servers",
    "oauth_providers_public",
    "require_servers",
    "evaluate",
    "guarded_send",
    "normalize_address",
    "normalize_domain",
    "parse_recipients",
    "provider_preset",
    "resolve_oauth_client",
    "tls_context",
    "xoauth2_string",
]
