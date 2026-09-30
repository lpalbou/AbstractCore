"""`abstractcore email ...` — connect, test, inspect and configure the email account.

    abstractcore email connect --address me@example.com --imap-host imap.example.com \\
        --smtp-host smtp.example.com --password <value>
    abstractcore email connect --address me@outlook.com --oauth microsoft --client-id <value> --client-secret <value>
    abstractcore email connect --address me@gmail.com --oauth google --client-id <value> --client-secret <value>
    abstractcore email test | status | folders
    abstractcore email disconnect --yes
    abstractcore email policy show | set --mode allowlist --add me@example.com --add example.org | check <address>...
    abstractcore email limits set --per-hour 20 --per-day 100
    abstractcore email enable | disable
    abstractcore email agent-tools on | off      "Agent email tools" (default off)
    abstractcore email registered-address me@example.com

Credentials are direct parameters (`--password <value>`, `--client-secret <value>`) and are
stored encrypted; they are never printed. Every verb takes `--json` (the `email_settings_v1`
document of `EmailAccountStore.public()`, or `{ok: false, error: {code, cause, fix}}`).
Exit codes: 0 ok, 1 error, 2 refused (a failed connection test, or `disconnect` without
`--yes`).

OAuth2 (`--oauth google|microsoft|custom`): without `--client-id`, the built-in
AbstractFramework client of the provider signs in when this version has one registered
(`abstractcore.comms.email.BUILTIN_CLIENTS`); otherwise bring your own client. While the
command waits for the approval, `--json` prints the sign-in prompt to stderr as one JSON line
`{"oauth_prompt": {"flow": "device", "user_code", "verification_uri", ...}}` or
`{"oauth_prompt": {"flow": "loopback", "authorization_url", ...}}`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import webbrowser
from typing import Any, Dict, List, Optional

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2


# A connection test that fails with one of these refuses the connect (exit 2): nothing stored.
_CONNECTION_CODES = frozenset(
    {
        "email_auth_failed",
        "email_tls_failed",
        "email_unreachable",
        "email_mailbox_missing",
        "email_server_error",
        "email_protocol_error",
        "email_transient",
        "email_quota_exceeded",
        "email_oauth_failed",
        "email_oauth_reauthorize",
    }
)


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _store(args: argparse.Namespace):
    from abstractcore.comms.email import EmailAccountStore

    return EmailAccountStore(key_backend=getattr(args, "key_storage", None) or "auto")


def _fail(err: Any, as_json: bool, *, code: int = EXIT_ERROR) -> int:
    if as_json:
        _print_json({"ok": False, "error": err.to_dict(include_details=True)})
    else:
        print(f"Error: {err.cause}", file=sys.stderr)
        if err.fix:
            print(f"Fix: {err.fix}", file=sys.stderr)
    return code


def _leg_line(name: str, leg: Dict[str, Any]) -> str:
    if not leg:
        return f"  {name}: -"
    if leg.get("ok") is None:
        return f"  {name}: not configured"
    if leg.get("ok"):
        return f"  {name}: ok"
    return f"  {name}: FAILED ({leg.get('code')}) {leg.get('cause')}\n        Fix: {leg.get('fix')}"


def _print_status(doc: Dict[str, Any], notices: List[str]) -> None:
    for n in notices:
        print(f"Notice: {n}")
    if not doc.get("configured"):
        print("Email: not connected")
        print("  Connect: abstractcore email connect --address <a> --imap-host <h> --smtp-host <h> --password <value>")
    else:
        print(f"Email: {doc['address']} ({'on' if doc.get('enabled') else 'OFF'})")
        if doc.get("display_name"):
            print(f"  Display name: {doc['display_name']}")
        print(f"  User name: {doc.get('username')}")
        print(f"  Sign-in: {doc.get('auth_kind')}" + (f" ({doc['oauth']['provider']})" if doc.get("oauth") else ""))
        imap = doc.get("imap")
        smtp = doc.get("smtp")
        print(f"  IMAP (read): {imap['host']}:{imap['port']} {imap['security']}, folder {imap['folder']}" if imap else "  IMAP (read): not configured")
        print(f"  SMTP (send): {smtp['host']}:{smtp['port']} {smtp['security']}" if smtp else "  SMTP (send): not configured")
        storage = {"os-keychain": "encrypted, key in the OS keychain", "key-file": "encrypted, key in a 0600 file"}.get(doc.get("secret_storage", ""), "not stored")
        print(f"  Credentials: {storage}")
        if doc.get("secret_warning"):
            print(f"  Warning: {doc['secret_warning']}")
        st = doc.get("status") or {}
        if st.get("last_test"):
            err = st.get("last_error")
            print(f"  Last test: {st['last_test']} " + ("ok" if not err else f"FAILED ({err.get('code')}) {err.get('cause')}"))
            if err:
                print(f"    Fix: {err.get('fix')}")
    pol = doc.get("policy") or {}
    if pol:
        entries = ", ".join(pol.get("entries") or []) or "(none)"
        print(f"  Recipient policy: {pol.get('mode')} {entries}" + (" (default)" if pol.get("default") else ""))
    lim = doc.get("limits") or {}
    if lim:
        print(
            f"  Send limits: {lim.get('per_hour')} per hour ({lim.get('used_last_hour')} used), "
            f"{lim.get('per_day')} per day ({lim.get('used_last_day')} used)"
        )
    if doc.get("registered_address"):
        print(f"  Registered address: {doc['registered_address']}")
    print(f"  Agent email tools: {_agent_tools_text(doc.get('agent_tools') or {})}")
    if doc.get("config_file"):
        print(f"  Settings: this AbstractCore install ({doc['config_file']})")


def _agent_tools_text(at: Dict[str, Any]) -> str:
    """The consoles' words for the "Agent email tools" toggle."""

    if at.get("active"):
        return "on: your agents and workflows have the email tools (policy, limits and approval still apply)"
    reason = str(at.get("reason") or "")
    return f"off — {reason}" if reason else "off"


# ---------------------------------------------------------------------------------------
# Verbs
# ---------------------------------------------------------------------------------------


def _oauth_prompt(prompt: Dict[str, Any], text: List[str], as_json: bool) -> None:
    """Show what the person must do to sign in.

    With `--json`, stdout is reserved for the one result document, so the prompt goes to
    stderr as ONE JSON line `{"oauth_prompt": {...}}` (the terminal console reads it while the
    command waits for the approval); otherwise as text on stdout.
    """

    if as_json:
        print(json.dumps({"oauth_prompt": prompt}, sort_keys=True), file=sys.stderr, flush=True)
    else:
        for line in text:
            print(line, flush=True)


def _obtain_oauth_tokens(args: argparse.Namespace, oauth: Any, client_secret: str, verify: Any, as_json: bool):
    from abstractcore.comms.email import LoopbackAuthorization, OAuthTokenClient

    client = OAuthTokenClient(oauth, client_secret=client_secret, verify=verify)
    flow = args.oauth_flow or ("device" if oauth.provider == "microsoft" else "loopback")
    if flow == "device":
        device = client.start_device_authorization()
        _oauth_prompt(
            {"flow": "device", **device.public()},
            [f"To sign in, open {device.verification_uri} and enter the code {device.user_code}"],
            as_json,
        )
        return client.poll_device_authorization(device)
    loop = LoopbackAuthorization(client, login_hint=args.address or "")
    url = loop.start()
    _oauth_prompt(
        {"flow": "loopback", "authorization_url": url, "expires_at": time.time() + float(args.oauth_timeout)},
        ["To sign in, open this address in a browser on this machine:", url],
        as_json,
    )
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    return loop.finish(timeout_s=float(args.oauth_timeout))


def cmd_connect(args: argparse.Namespace) -> int:
    from abstractcore.comms.email import (
        EmailAccount,
        EmailError,
        EmailInvalidSettings,
        EmailSecret,
        ImapSettings,
        OAuthSettings,
        SmtpSettings,
        provider_preset,
        resolve_oauth_client,
        tls_context,
    )

    as_json = bool(args.json)
    try:
        preset = provider_preset(args.oauth, tenant=args.tenant or "") if args.oauth and args.oauth != "custom" else {}
        address = (args.address or "").strip() or (args.username or "").strip()
        if not address:
            raise EmailInvalidSettings("The account address is missing.", "Give --address name@example.com.")
        imap_host = args.imap_host or (preset.get("imap") or {}).get("host")
        smtp_host = args.smtp_host or (preset.get("smtp") or {}).get("host")
        imap = None
        if imap_host:
            sec = args.imap_security or (preset.get("imap") or {}).get("security") or "ssl"
            port = args.imap_port or ((preset.get("imap") or {}).get("port") if not args.imap_security else None)
            imap = ImapSettings.build(imap_host, port=port, security=sec, folder=args.imap_folder or "INBOX", ca_file=args.imap_ca_file or args.ca_file or "")
        smtp = None
        if smtp_host:
            sec = args.smtp_security
            if not sec:
                if args.smtp_port in (587, 25):
                    sec = "starttls"
                else:
                    sec = (preset.get("smtp") or {}).get("security") or "ssl"
            port = args.smtp_port or ((preset.get("smtp") or {}).get("port") if not args.smtp_security else None)
            smtp = SmtpSettings.build(smtp_host, port=port, security=sec, ca_file=args.smtp_ca_file or args.ca_file or "")
        ctx = tls_context(args.ca_file) if args.ca_file else None
        if args.oauth:
            if args.password:
                raise EmailInvalidSettings("--password is not used with --oauth.", "Remove --password; OAuth2 signs in through the browser.")
            client = resolve_oauth_client(args.oauth, args.client_id or "", args.client_secret or "")
            oauth = OAuthSettings.build(
                args.oauth,
                client["client_id"],
                client_source=client["source"],
                token_endpoint=args.token_endpoint or "",
                authorization_endpoint=args.authorization_endpoint or "",
                device_authorization_endpoint=args.device_endpoint or "",
                scopes=args.scope or None,
                tenant=args.tenant or "",
            )
            account = EmailAccount.build(
                address=address, username=args.username or address, imap=imap, smtp=smtp,
                display_name=args.display_name or "", auth_kind="oauth2", oauth=oauth,
            )
            tokens = _obtain_oauth_tokens(args, oauth, client["client_secret"], ctx, as_json)
            secret = EmailSecret(
                refresh_token=tokens.refresh_token,
                access_token=tokens.access_token,
                expires_at=tokens.expires_at,
                client_secret=client["client_secret"],
            )
        else:
            if not args.password:
                raise EmailInvalidSettings(
                    "The password is missing.",
                    "Give --password <value> (an app password for providers with two-step verification), or --oauth <provider>.",
                )
            account = EmailAccount.build(
                address=address, username=args.username or address, imap=imap, smtp=smtp,
                display_name=args.display_name or "",
            )
            secret = EmailSecret(args.password)
        store = _store(args)
        doc = store.connect(account, secret, test=not args.no_test, registered_address=args.registered_address, ssl_context=ctx)
    except EmailError as err:
        connection_failed = err.code in _CONNECTION_CODES
        return _fail(err, as_json, code=EXIT_REFUSED if connection_failed else EXIT_ERROR)
    if as_json:
        _print_json({"ok": True, **doc})
    else:
        print(f"Connected {doc['address']}" + ("" if args.no_test else " (connection test passed)") + ".")
        _print_status(doc, [])
    return EXIT_OK


def cmd_test(args: argparse.Namespace) -> int:
    from abstractcore.comms.email import EmailError, tls_context

    try:
        ctx = tls_context(args.ca_file) if getattr(args, "ca_file", None) else None
        result = _store(args).test(ssl_context=ctx)
    except EmailError as err:
        return _fail(err, bool(args.json))
    if args.json:
        _print_json(result)
    else:
        print("Email connection test:")
        print(_leg_line("IMAP (read)", result.get("imap") or {}))
        print(_leg_line("SMTP (send)", result.get("smtp") or {}))
    return EXIT_OK if result.get("ok") else EXIT_REFUSED


def cmd_status(args: argparse.Namespace) -> int:
    from abstractcore.comms.email import EmailError

    try:
        store = _store(args)
        notices = store.ensure_legacy_imported()
        doc = store.public()
    except EmailError as err:
        return _fail(err, bool(args.json))
    if args.json:
        _print_json({**doc, "notices": notices})
    else:
        _print_status(doc, notices)
    return EXIT_OK


def cmd_folders(args: argparse.Namespace) -> int:
    from abstractcore.comms.email import EmailError

    try:
        folders = _store(args).context().client().list_folders()
    except EmailError as err:
        return _fail(err, bool(args.json))
    if args.json:
        _print_json({"folders": folders})
    else:
        for f in folders:
            print(f["name"])
    return EXIT_OK


def cmd_disconnect(args: argparse.Namespace) -> int:
    from abstractcore.comms.email import EmailError

    if not args.yes:
        msg = "Disconnecting deletes the stored password or tokens. Re-run with --yes to confirm."
        if args.json:
            _print_json({"ok": False, "status": "refused", "message": msg})
        else:
            print(msg, file=sys.stderr)
        return EXIT_REFUSED
    try:
        doc = _store(args).disconnect()
    except EmailError as err:
        return _fail(err, bool(args.json))
    if args.json:
        _print_json({"ok": True, **doc})
    else:
        print("Email account disconnected; the stored credentials were deleted (policy and limits kept).")
    return EXIT_OK


def cmd_policy(args: argparse.Namespace) -> int:
    from abstractcore.comms.email import EmailError, evaluate, parse_recipients

    as_json = bool(args.json)
    store = _store(args)
    try:
        if args.policy_cmd == "set":
            if not (args.mode or args.add or args.remove or args.clear):
                print("Nothing to change: give --mode, --add, --remove or --clear.", file=sys.stderr)
                return EXIT_ERROR
            doc = store.set_policy(mode=args.mode, add=args.add or (), remove=args.remove or (), clear=bool(args.clear))
            pol = doc["policy"]
        elif args.policy_cmd == "check":
            policy = store.settings().policy
            try:
                addrs = parse_recipients(list(args.addresses))
            except ValueError as exc:
                print(f"Error: {exc}", file=sys.stderr)
                return EXIT_ERROR
            decision = evaluate(policy, to=addrs)
            if as_json:
                _print_json(decision.to_dict())
            else:
                for v in decision.verdicts:
                    print(f"{'ALLOWED' if v.allowed else 'REFUSED'}  {v.address}  ({v.reason})")
            return EXIT_OK if decision.allowed else EXIT_REFUSED
        else:
            pol = store.public()["policy"]
    except EmailError as err:
        return _fail(err, as_json)
    if as_json:
        _print_json(pol)
    else:
        print(f"Recipient policy: {pol['mode']}" + (" (default)" if pol.get("default") else ""))
        for e in pol.get("entries") or []:
            print(f"  {e}")
        if not pol.get("entries"):
            print("  (no entries)" + ("  -- an empty allowlist refuses every recipient" if pol["mode"] == "allowlist" else ""))
    return EXIT_OK


def cmd_limits(args: argparse.Namespace) -> int:
    from abstractcore.comms.email import EmailError

    try:
        if args.limits_cmd == "set":
            if args.per_hour is None and args.per_day is None:
                print("Nothing to change: give --per-hour and/or --per-day.", file=sys.stderr)
                return EXIT_ERROR
            doc = _store(args).set_limits(per_hour=args.per_hour, per_day=args.per_day)
        else:
            doc = _store(args).public()
    except EmailError as err:
        return _fail(err, bool(args.json))
    lim = doc["limits"]
    if args.json:
        _print_json(lim)
    else:
        print(f"Send limits: {lim['per_hour']} per hour ({lim['used_last_hour']} used), {lim['per_day']} per day ({lim['used_last_day']} used)")
    return EXIT_OK


def cmd_enable(args: argparse.Namespace, enabled: bool) -> int:
    from abstractcore.comms.email import EmailError

    try:
        doc = _store(args).set_enabled(enabled)
    except EmailError as err:
        return _fail(err, bool(args.json))
    if args.json:
        _print_json({"ok": True, "enabled": doc["enabled"]})
    else:
        print("Email turned " + ("on." if enabled else "off (no reading, no sending; settings kept)."))
    return EXIT_OK


def cmd_agent_tools(args: argparse.Namespace) -> int:
    from abstractcore.comms.email import EmailError

    enabled = args.state == "on"
    try:
        doc = _store(args).set_agent_tools(enabled)
    except EmailError as err:
        return _fail(err, bool(args.json))
    at = doc.get("agent_tools") or {}
    if args.json:
        _print_json({"ok": True, "agent_tools": at})
    else:
        print(f"Agent email tools: {_agent_tools_text(at)}")
        if enabled and not at.get("active"):
            print("  They take effect once the account is connected and turned on.")
    return EXIT_OK


def cmd_registered(args: argparse.Namespace) -> int:
    from abstractcore.comms.email import EmailError

    try:
        doc = _store(args).set_registered_address(args.address)
    except EmailError as err:
        return _fail(err, bool(args.json))
    if args.json:
        _print_json({"ok": True, "registered_address": doc["registered_address"]})
    else:
        print(f"Registered address: {doc['registered_address'] or '(the account address)'}")
    return EXIT_OK


# ---------------------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="abstractcore email", description="Configure the email account (IMAP read-only + SMTP).")
    parser.add_argument(
        "--key-storage",
        choices=("auto", "keyring", "file"),
        default="auto",
        help="Where the encryption key of new credentials goes: the OS keychain (auto, when available) or a 0600 key file",
    )
    sub = parser.add_subparsers(dest="cmd")

    def common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--json", action="store_true", help="Print JSON")
        return p

    c = common(sub.add_parser("connect", help="Connect (test, then store) the email account"))
    c.add_argument("--address", help="The account's email address (the sender)")
    c.add_argument("--display-name", help="Name shown to recipients")
    c.add_argument("--username", help="Sign-in user name (default: the address)")
    c.add_argument("--password", help="Password or app password (stored encrypted)")
    c.add_argument("--imap-host")
    c.add_argument("--imap-port", type=int)
    c.add_argument("--imap-security", choices=("ssl", "starttls"))
    c.add_argument("--imap-folder", help="Folder to read (default INBOX)")
    c.add_argument("--imap-ca-file", help="PEM file of a private CA for the IMAP server")
    c.add_argument("--smtp-host")
    c.add_argument("--smtp-port", type=int)
    c.add_argument("--smtp-security", choices=("ssl", "starttls"))
    c.add_argument("--smtp-ca-file", help="PEM file of a private CA for the SMTP server")
    c.add_argument("--ca-file", help="PEM file of a private CA for both servers (and a custom OAuth endpoint)")
    c.add_argument("--registered-address", help="Your own address (the default allowlist entry); default: --address")
    c.add_argument("--no-test", action="store_true", help="Store without testing the connection")
    c.add_argument("--oauth", choices=("google", "microsoft", "custom"), help="Sign in with OAuth2 instead of a password")
    c.add_argument("--client-id", help="Your own OAuth client id (default: the built-in AbstractFramework client, when one is registered for the provider)")
    c.add_argument("--client-secret", help="Your own OAuth client secret (stored encrypted)")
    c.add_argument("--tenant", help="Microsoft tenant (default common)")
    c.add_argument("--oauth-flow", choices=("device", "loopback"), help="device code (Microsoft default) or browser on this machine (Google default)")
    c.add_argument("--token-endpoint")
    c.add_argument("--authorization-endpoint")
    c.add_argument("--device-endpoint")
    c.add_argument("--scope", action="append", help="OAuth scope (repeatable; custom provider)")
    c.add_argument("--no-browser", action="store_true", help="Print the sign-in address without opening a browser")
    c.add_argument("--oauth-timeout", type=float, default=300.0, help=argparse.SUPPRESS)

    t = common(sub.add_parser("test", help="Sign in to IMAP and SMTP with the stored account"))
    t.add_argument("--ca-file", help=argparse.SUPPRESS)
    common(sub.add_parser("status", help="Show the account, policy, limits and last test"))
    common(sub.add_parser("folders", help="List the mailbox folders"))
    d = common(sub.add_parser("disconnect", help="Delete the stored credentials and account settings"))
    d.add_argument("--yes", action="store_true")

    p = sub.add_parser("policy", help="Recipient policy (allowlist / denylist)")
    psub = p.add_subparsers(dest="policy_cmd")
    common(psub.add_parser("show"))
    ps = common(psub.add_parser("set"))
    ps.add_argument("--mode", choices=("allowlist", "denylist"))
    ps.add_argument("--add", action="append", help="Exact address or domain (repeatable)")
    ps.add_argument("--remove", action="append", help="Entry to remove (repeatable)")
    ps.add_argument("--clear", action="store_true", help="Remove every entry first")
    pc = common(psub.add_parser("check", help="Would these recipients be allowed?"))
    pc.add_argument("addresses", nargs="+")

    lm = sub.add_parser("limits", help="Send limits per hour / per day")
    lsub = lm.add_subparsers(dest="limits_cmd")
    common(lsub.add_parser("show"))
    ls = common(lsub.add_parser("set"))
    ls.add_argument("--per-hour", type=int)
    ls.add_argument("--per-day", type=int)

    common(sub.add_parser("enable", help="Turn email on"))
    common(sub.add_parser("disable", help="Turn email off (settings kept)"))
    at = common(sub.add_parser(
        "agent-tools",
        help='"Agent email tools": let agents use the email tools with this account (default off)',
    ))
    at.add_argument("state", choices=("on", "off"))
    r = common(sub.add_parser("registered-address", help="Set your own address (the default allowlist entry)"))
    r.add_argument("address")
    return parser


def handle_email(argv: List[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "connect":
        return cmd_connect(args)
    if args.cmd == "test":
        return cmd_test(args)
    if args.cmd == "status":
        return cmd_status(args)
    if args.cmd == "folders":
        return cmd_folders(args)
    if args.cmd == "disconnect":
        return cmd_disconnect(args)
    if args.cmd == "policy":
        if not args.policy_cmd:
            args.policy_cmd = "show"
            args.json = False
        return cmd_policy(args)
    if args.cmd == "limits":
        if not args.limits_cmd:
            args.limits_cmd = "show"
            args.json = False
        return cmd_limits(args)
    if args.cmd == "enable":
        return cmd_enable(args, True)
    if args.cmd == "disable":
        return cmd_enable(args, False)
    if args.cmd == "agent-tools":
        return cmd_agent_tools(args)
    if args.cmd == "registered-address":
        return cmd_registered(args)
    parser.print_help()
    return EXIT_ERROR
