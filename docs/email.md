# Email

AbstractCore connects one email account per install: it reads the mailbox over IMAP (read-only)
and sends over SMTP. The same account is used by the `abstractcore email` CLI, the Email page of
the [web console](console.md), the Email screen of the [terminal console](console-tui.md), the
email tools ([Tool Calling](tool-calling.md)) and the Python library `abstractcore.comms.email`.

What you can rely on:

- **Settings, not environment variables.** The account lives in the `email` section of
  AbstractCore's config file ([Centralized Config](centralized-config.md)); credentials are given as
  direct parameters (`--password <value>`) and stored encrypted.
- **Encrypted credentials.** The password (or OAuth2 tokens) is sealed with AES-256-GCM in
  `<config dir>/email/secret.enc`. The key is kept in the OS keychain (macOS Keychain, Windows
  Credential Manager, Linux Secret Service). On a host without a keychain the key is written to a
  0600 file next to the sealed credentials, and `abstractcore email status` says so: a copy of that
  whole folder carries both, so protect it like a password. No command, page, tool result or error
  ever shows a password or token.
- **Verified TLS on every connection.** IMAP and SMTP use SSL (implicit TLS) or STARTTLS; the
  certificate chain and the host name are always checked, and a failure is refused before any
  password is sent. There is no plaintext mode. A self-hosted server signed by a private CA is
  trusted by giving that CA's PEM file (`--ca-file`).
- **A read-only mailbox.** Folders are opened with EXAMINE and messages fetched with BODY.PEEK:
  nothing is marked read, moved or deleted.
- **A recipient policy** decides who can receive mail at all, and **send limits** cap how much is
  sent, for every sender (the CLI, the consoles, tools, and hosts such as the gateway).
- **Whole messages.** Bodies are returned in full, never cut.

## Connect an account

```bash
abstractcore email connect \
  --address me@example.com \
  --imap-host imap.example.com --smtp-host smtp.example.com \
  --password <value>
```

`connect` signs in to IMAP and SMTP first and stores nothing if either fails; the error says what
went wrong and what to do. Useful options:

| Option | Meaning |
|---|---|
| `--username <name>` | Sign-in name when it differs from the address |
| `--display-name <name>` | Name shown to recipients |
| `--imap-port`, `--imap-security ssl\|starttls`, `--imap-folder` | IMAP settings (default SSL on 993, folder `INBOX`) |
| `--smtp-port`, `--smtp-security ssl\|starttls` | SMTP settings (default SSL on 465; port 587 selects STARTTLS) |
| `--ca-file <pem>` | Trust a private CA (also `--imap-ca-file`, `--smtp-ca-file`) |
| `--registered-address <address>` | Your own address, the default allowlist entry (default: `--address`) |
| `--no-test` | Store without the connection test |
| `--key-storage auto\|keyring\|file` | Where the encryption key goes (before the verb: `abstractcore email --key-storage file connect ...`) |

Many providers require an **app password** when two-step verification is on (Gmail, iCloud,
Fastmail, Yahoo).

Other verbs:

```bash
abstractcore email status            # account, credentials storage, policy, limits, last test
abstractcore email test              # sign in to IMAP and SMTP with the stored account
abstractcore email folders           # list the mailbox folders
abstractcore email disable           # turn email off (no reading, no sending; settings kept)
abstractcore email enable
abstractcore email disconnect --yes  # delete the stored credentials and account (policy and limits kept)
```

Every verb accepts `--json`. Exit codes: 0 success, 1 error, 2 refused (a failed connection test,
or `disconnect` without `--yes`).

### OAuth2 (Google, Microsoft)

Instead of a password you can sign in with OAuth2. You need an OAuth client registered with the
provider (a Google Cloud OAuth client with the Gmail scope, a Microsoft Entra app with the IMAP and
SMTP scopes):

```bash
# Microsoft: device code (prints a URL and a code to enter)
abstractcore email connect --address me@outlook.com --oauth microsoft --client-id <value>

# Google: a browser on this machine (a one-shot listener on 127.0.0.1 receives the sign-in)
abstractcore email connect --address me@gmail.com --oauth google --client-id <value> --client-secret <value>
```

Google does not allow Gmail scopes in its device flow, so Google accounts use the browser flow
(`--oauth-flow loopback`, the default for Google). Provider presets fill in the IMAP/SMTP hosts
and scopes; `--tenant` selects a Microsoft tenant; `--oauth custom` takes explicit
`--token-endpoint`, `--authorization-endpoint`, `--device-endpoint` and `--scope`. The refresh
token and client secret are sealed like a password; access tokens are refreshed before they expire.
When the provider revokes the grant, the error says to sign in again.

## Recipient policy

The policy decides who can receive mail from this account:

- `allowlist` — only the listed recipients;
- `denylist` — everyone except the listed recipients.

Entries are exact addresses (`boss@example.com`) or domains (`example.com`, also written
`@example.com`). A domain entry matches that domain only; a subdomain matches only when it is
written as its own entry. Patterns (`*`, `?`, regular expressions) are refused. Comparison ignores
letter case and display names, and internationalized domains are compared in their IDNA form.

The policy applies to To, Cc and Bcc. A message with any refused recipient is not sent at all; the
error lists each refused address, the field it was in and the rule that refused it.

A newly connected account starts with an **allowlist holding only your registered address**.

```bash
abstractcore email policy show
abstractcore email policy set --add example.org --add colleague@example.net
abstractcore email policy set --mode denylist --clear --add competitor.example
abstractcore email policy set --remove example.org
abstractcore email policy check someone@example.org   # exit 0 allowed, 2 refused
```

In hosts that approve tool calls (AbstractRuntime, AbstractGateway), approval still applies on top:
the policy decides who *can* receive mail; approval decides whether a given send runs unattended.

## Send limits

Each account has a limit per rolling hour and per rolling day (default 20 and 100). The count is
kept in `<config dir>/email/sends.json`, shared by every process using the account; a message the
server did not accept does not count. `0` means no sending in that window.

```bash
abstractcore email limits show
abstractcore email limits set --per-hour 20 --per-day 100
```

## Errors

Every failure has a stable code, a cause and a fix, for example:

| Code | Typical cause |
|---|---|
| `email_auth_failed` | wrong user name or password (IMAP `[AUTHENTICATIONFAILED]`, SMTP 535) |
| `email_tls_failed` | certificate not trusted for the host name, or SSL/STARTTLS used on the wrong port |
| `email_unreachable` | host not found, connection refused, timeout |
| `email_mailbox_missing` | the folder does not exist |
| `email_quota_exceeded` | mailbox over quota, or message too large (SMTP 452/552) |
| `email_recipient_refused` | the server refused a recipient (SMTP 550/553) |
| `email_policy_refused` | the recipient policy refused the message |
| `email_rate_limited` | a send limit is reached |
| `email_oauth_reauthorize` | the OAuth2 grant expired or was revoked: sign in again |
| `email_not_configured`, `email_disabled`, `email_secret_unavailable` | no account, email turned off, credentials missing |

Errors are classified from protocol reply codes and exception types, never from message text.

## Consoles

- **Web console** (`abstractcore serve`, then `/console`): the **Email** tab has the Account form
  (Save and test), Test, Turn off / Turn on, Disconnect (with an inline confirmation), the
  recipient policy editor with a Check field, the send limits and the status of the last test. It
  uses the `/acore/email` routes ([Server](server.md)).
- **Terminal console** (`abstractcore-console`): the **Email** screen (`@`) shows the same
  sections with the same words: `c` connect, `t` test, `o` turn off/on, `x` disconnect, `p`
  recipient policy, `l` send limits.

## Python

```python
from abstractcore.comms.email import (
    EmailAccountStore, EmailAccount, ImapSettings, SmtpSettings, EmailSecret,
    SearchCriteria, OutgoingMessage,
)

store = EmailAccountStore()          # the local AbstractCore settings
ctx = store.context()                # raises a typed error when not connected or turned off

client = ctx.client()
found = client.search(SearchCriteria.build(from_domain="example.org", since="7d"), limit=20)
message = client.get(found["messages"][0].uid)

ctx.send(OutgoingMessage(to=("me@example.com",), subject="Report", text="Done."))  # policy + limits
```

`SearchCriteria` takes typed fields only: `from_address`, `from_domain`, `to_address`,
`subject_contains` (a literal, case-insensitive substring), `since`, `before` and `unseen`.
`client.fetch_new(cursor)` returns the messages that arrived after a stored
`{uidvalidity, last_uid}` cursor, and reports `reset=True` when the server rebuilt the folder.
`client.download_attachment(uid, index, folder_path)` saves an attachment under a sanitized name
and never overwrites a file. `client.build_reply(uid, text=...)` prepares a reply with
`In-Reply-To` / `References` set and the recipient taken from `Reply-To` (else `From`).

A host that serves several users (the gateway) keeps one `EmailAccountStore(config_file=...)` per
user and gives the tools the account of the run they execute for, with
`abstractcore.tools.comms_tools.use_email_context(ctx)` or `set_email_account_resolver(fn)`. While a
resolver is installed, the tools never fall back to the install's own account.

Hermetic test servers (IMAP, SMTP with SSL/STARTTLS, an OAuth2 token endpoint, a throwaway CA) are
available in `abstractcore.testing.mailserver`; the SMTP server needs `aiosmtpd` (part of the
`test` extra).

## Upgrading from the environment-variable configuration

Releases before 2.20 read the account from `ABSTRACT_EMAIL_*` environment variables, an accounts
file named by `ABSTRACT_EMAIL_ACCOUNTS_CONFIG`, or the flat `email.smtp_*` / `email.imap_*` config
fields, with the password in the environment variable named by `*_password_env_var`. The first time
email is used without a configured account, that configuration is imported once into the settings
(the password is read from its variable and sealed) and `abstractcore email status` reports the
import. From then on those variables are ignored, and `status` names each one still set together
with the setting that replaced it. Only the default account of a multi-account file is imported.
