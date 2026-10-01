# Email

AbstractCore connects one email account per install: it reads the mailbox over IMAP (read-only)
and sends over SMTP. The same account is used by the `abstractcore email` CLI, the Email page of
the [web console](console.md), the Email screen of the [terminal console](console-tui.md), the
email tools ([Tool Calling](tool-calling.md)) and the Python library `abstractcore.comms.email`.

What you can rely on:

- **Settings, not environment variables.** The account lives in the `email` section of
  AbstractCore's config file ([Centralized Config](centralized-config.md)); credentials are given as
  direct parameters (`--password <value>`), or on stdin from scripts (`--password-stdin`), and
  stored encrypted.
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
- **Whole messages.** Bodies are returned in full, never cut. Reading a message fetches its
  structure and then only its text and HTML parts; attachments are listed (name, type, size) and
  downloaded one at a time on request. A message whose bodies exceed the reading limit (25 MiB by
  default) is returned with its headers, its attachment list and a typed skip record instead of
  the bodies.
- **Agents only with your consent.** Agents get the email tools only when you turn on **Agent email tools**
  (off by default).

This page describes the account of an AbstractCore install, kept in AbstractCore's own settings.
An AbstractGateway keeps a separate account for each of its users (see
[One install, one gateway: two accounts](#one-install-one-gateway-two-accounts)).

## Connect an account

```bash
abstractcore email connect \
  --address me@example.com \
  --imap-host imap.example.com --smtp-host smtp.example.com \
  --password <value>
```

`connect` signs in to IMAP and SMTP first and stores nothing if either fails; the error says what
went wrong and what to do.

**From a script or another program, pass the password on stdin.** A command line is visible to
every local user (`ps`) while the command runs; stdin is not. `--password-stdin` reads exactly one
line from stdin and removes only its trailing newline (spaces are kept):

```bash
printf '%s\n' "$MAIL_PASSWORD" | abstractcore email connect \
  --address me@example.com \
  --imap-host imap.example.com --smtp-host smtp.example.com \
  --password-stdin
```

`--password-stdin` cannot be combined with `--password` or `--oauth`; the terminal console passes
the password this way. Useful options:

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
abstractcore email agent-tools on    # let agents use the email tools (default off)
abstractcore email agent-tools off
abstractcore email disconnect --yes  # delete the stored credentials and account (policy and limits kept)
```

Every verb accepts `--json`. Exit codes: 0 success, 1 error, 2 refused (a failed connection test,
or `disconnect` without `--yes`).

### OAuth2 (Google, Microsoft)

Instead of a password you can sign in with OAuth2. The sign-in uses an OAuth client registered
with the provider:

- **Your own client** (bring your own): a Google Cloud OAuth client with the Gmail scope, or a
  Microsoft Entra app with the IMAP and SMTP scopes. Pass `--client-id <value>` (and
  `--client-secret <value>` when the provider issued one; from a script, `--client-secret-stdin`
  reads it from stdin as one line, like `--password-stdin`).
- **The built-in AbstractFramework client**: used when you omit `--client-id` and this version
  ships a registered client for the provider (`abstractcore.comms.email.BUILTIN_CLIENTS`). When
  none is registered, the command stops before contacting the provider and asks for your own
  client. `status` shows which client signed in (`client_source`: `own` or `builtin`).

```bash
# Microsoft: device code (prints a URL and a code to enter, from any browser on any machine)
abstractcore email connect --address me@outlook.com --oauth microsoft --client-id <value> --client-secret <value>

# Google: a browser on this machine (a one-shot listener on 127.0.0.1 receives the sign-in)
abstractcore email connect --address me@gmail.com --oauth google --client-id <value> --client-secret <value>
```

Google does not allow Gmail scopes in its device flow, so Google accounts use the browser flow
(`--oauth-flow loopback`, the default for Google). Provider presets fill in the IMAP/SMTP hosts
and scopes; `--tenant` selects a Microsoft tenant; `--oauth custom` takes explicit
`--token-endpoint`, `--authorization-endpoint`, `--device-endpoint` and `--scope`. The refresh
token and client secret are sealed like a password; access tokens are refreshed before they expire.
When the provider revokes the grant, the error says to sign in again.

With `--json`, stdout carries only the result document; while the command waits for your approval
it prints the sign-in prompt to stderr as one JSON line, which the terminal console displays:

```json
{"oauth_prompt": {"flow": "device", "user_code": "WDJB-MJHT", "verification_uri": "https://...", "verification_uri_complete": "", "expires_at": 1790000000.0}}
{"oauth_prompt": {"flow": "loopback", "authorization_url": "https://...", "expires_at": 1790000000.0}}
```

The web console (**Sign in with OAuth2** on the Email tab) and the terminal console (the Google
and Microsoft tabs of the Email screen's Mailbox card) offer the same sign-in with the same fields: provider, address, client id, client
secret, Microsoft tenant, and flow. The browser flow listens on 127.0.0.1 of the machine running
AbstractCore, so open its sign-in page in a browser on that machine; the device-code flow works
from any browser.

## Agent email tools

The email tools (`list_emails`, `read_email`, `send_email`, ...; see [Tool Calling](tool-calling.md))
use this account only when **Agent email tools** is on. It is off by default, and it takes effect
only while the account is connected and turned on:

```bash
abstractcore email agent-tools on
abstractcore email agent-tools off
```

The same switch is on the Email tab of the web console and the Email screen of the terminal
console (`a`), and at `PUT /acore/email/agent-tools`. `status` shows its state as
`agent_tools: {enabled, active, reason}`: `enabled` is your choice, `active` says whether agents
have the tools right now, and `reason` says why not. While it is off, every email tool answers
`email_agent_tools_off` with the command that turns it on. You still use the account yourself
from the CLI and the consoles; every send by an agent still passes the recipient policy, the send
limits and the host's approval gate.

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

Each account has a limit per rolling hour and per rolling day (default 100 and 1000). The count is
kept in `<config dir>/email/sends.json`, shared by every process using the account; a message the
server did not accept does not count. `0` means no sending in that window.

```bash
abstractcore email limits show
abstractcore email limits set --per-hour 100 --per-day 1000
abstractcore email limits reset     # forget the stored limits: follow the defaults
```

Limits you set are stored and kept across upgrades; an account where nobody set them follows the
defaults, including a later change of the defaults. Until 2.21 the defaults were 20 and 100, and
connecting an account stored them. Such a stored 20 / 100 cannot be told apart from a user who
chose 20 / 100, so an upgrade keeps it: `limits show` marks it "stored by an earlier version", and
`limits reset` (or setting new values) moves the account on. `limits show --json` reports the
origin as `source`: `default`, `user` or `legacy`.

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
| `email_agent_tools_off` | an agent used an email tool while **Agent email tools** is off |
| `email_message_too_large` | a message's bodies (or an attachment) exceed the reading limit: the read returns a skip record, a download is refused |
| `email_oauth_reauthorize` | the OAuth2 grant expired or was revoked: sign in again |
| `email_not_configured`, `email_disabled`, `email_secret_unavailable` | no account, email turned off, credentials missing |

Errors are classified from protocol reply codes and exception types, never from message text.

## Consoles

- **Web console** (`abstractcore serve`, then `/console`): the **Email** tab shows, in order:
  - **Email address**: your own address (the registered address: where notifications go and the
    first address your agents may write to), with its own **Save**.
  - **Mailbox**: tabs **Google**, **Microsoft** and **Other**. Google and Microsoft sign in with
    the provider (your own client under Advanced). Other asks for the email address and the
    password only: the servers are discovered from the address and shown on one line with
    **Edit**; **Server settings** open by themselves when discovery finds nothing. **Connect**
    tests reading and sending, then stores; an error names the step that failed. Once connected:
    the status line ("Connected as ... · Password · checked 2 min ago"), **Test**, and
    **Disconnect** with an inline confirmation.
  - **Agent email tools**: a switch (off by default; unavailable until a mailbox is connected).
  - **Advanced**: the recipient rules (mode, entries, Check a recipient), the send limits and the
    folder (both saved when you leave the field), and the **Use this mailbox** switch.

  Switches and Advanced fields apply at once; the Email address is the only field with a Save
  button. The tab uses the `/acore/email` routes ([Server](server.md)).
- **Terminal console** (`abstractcore-console`): the **Email** screen (`@`) has the same cards
  with the same words: the email address with its own Save, the Mailbox card (Google / Microsoft
  / Other, servers found from the address, one Connect; connected: Test and Disconnect), the
  **Agent email tools** switch, and Advanced (recipient rules, send limits, folder, **Use this
  mailbox**). Switches read `[x]` on, `[ ]` off, `[-]` unavailable with the reason; `Space`
  switches ([Terminal console](console-tui.md#email-keys)).

Both consoles name the account they configure: the account of this AbstractCore install, with the
path of its settings file.

## One install, one gateway: two accounts

AbstractCore and AbstractGateway keep separate email settings:

| Where | Whose account | Configured with |
|---|---|---|
| AbstractCore settings (`<config dir>/abstractcore.json`, `<config dir>/email/`) | the account of this AbstractCore install, used by `abstractcore email`, the AbstractCore consoles and the email tools in plain Python use | `abstractcore email ...`, the AbstractCore web and terminal consoles |
| AbstractGateway data folder (one store per user) | each gateway user's own account, used by that user's agents, automations and notifications | the gateway console (**My email**), `abstractgateway email ...` |

On a single-user machine that runs both, you may therefore see two accounts. The gateway never
reads the AbstractCore account: connect the mailbox you want your gateway agents to use in the
gateway console. Each store has its own **Agent email tools** switch. The only automatic transfer
is the one-time import of the pre-2.20 environment configuration described below; the gateway
does the same once for its administrator (see the gateway's email documentation).

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
`subject_contains` (a literal, case-insensitive substring), `since`, `before`, `unseen` and
`has_attachment` (checked on each message's MIME structure). `client.search(...)` returns
`has_more` and `next_before_uid` when it stopped at `limit`; pass `before_uid=` to continue.

Message summaries carry `uid`, `subject`, `from`, `to`, `cc`, `date`, `internaldate`, `flags`,
`seen`, `size`, `has_attachments`, `reply_to`, `in_reply_to`, `list_unsubscribe` (the header is
present) and the priority headers as typed values: `importance` (`low` | `normal` | `high`),
`x_priority` (1 highest to 5 lowest) and `priority` (`normal` | `urgent` | `non-urgent`); a value
outside those sets is `None`. Two more fields mark automatic mail: `auto_submitted` (RFC 3834
`Auto-Submitted`, the lower-cased keyword such as `auto-generated`, `auto-replied` or `no`;
`None` when the header is absent) and `framework_marker` (the `X-AbstractFramework-Automation`
header AbstractFramework puts on the mail it sends automatically; `""` when absent).

### Automatic mail

Mail that software sends on its own (notifications, automation results, auto-replies) should say
so, so that other automations and auto-responders never answer it in a loop. Set it per message
with `OutgoingMessage(auto_submitted="auto-generated", automation_marker="<id>")`
(`auto-replied` for an automatic answer to one message), or for every send through a context
with `EmailContext.automation_marker`: `guarded_send` then adds `Auto-Submitted`
(`auto-replied` when `in_reply_to` is set, else `auto-generated`) and
`X-AbstractFramework-Automation: <marker>`, and `on_sent` reports both next to the Message-ID.
A marker is one line of printable ASCII, at most 200 characters.

`client.get(uid)` fetches the message structure, then only its text/plain and text/html parts.
Attachments are listed with `filename`, `content_type`, `size` (the size on the wire, encoded, as
the server reports it), `encoding`, `disposition` and `content_id`. When the bodies exceed the
reading limit (`max_message_bytes`, default 25 MiB, set on `EmailClient(...)`, on
`EmailContext`, or per call), the detail has `skipped = {code: "email_message_too_large", cause,
fix, uid, folder, size, limit}` and no bodies (`body_text` / `body_html` are `None` in
`to_dict()`, with `body_skipped` holding the record).
`client.fetch_new(cursor)` returns the messages that arrived after a stored
`{uidvalidity, last_uid}` cursor, and reports `reset=True` when the server rebuilt the folder;
when nothing is left to resynchronise it returns a new baseline (`reset=True, baseline=True`, the
cursor at the newest message), so a rebuilt folder is never replayed.
`client.download_attachment(uid, index, folder_path)` fetches only that attachment, saves it under
a sanitized name and never overwrites a file (above the reading limit it raises
`EmailMessageTooLarge`). `client.build_reply(uid, text=...)` prepares a reply with
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
