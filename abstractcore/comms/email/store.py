"""`EmailAccountStore`: one account's settings, sealed credentials, status and send counter.

Where things live (for a config file `<dir>/abstractcore.json`):

    <dir>/abstractcore.json   section `email`: enabled, agent_tools, account, policy, limits,
                              registered_address
    <dir>/email/secret.enc    the password or OAuth tokens, AES-256-GCM (vault.py)
    <dir>/email/secret.key    only when no OS keychain is available (0600)
    <dir>/email/status.json   last test time, result, typed error
    <dir>/email/sends.json    send timestamps of the last 24 h (the limits' counter)

The core CLI and consoles use the default config file (`~/.abstractcore/config`, or
`ABSTRACTCORE_CONFIG_FILE` / `ABSTRACTCORE_CONFIG_DIR`). A host keeping one store per user
passes that user's config file: the gateway's per-user AbstractCore overlay
(`<plane>/config/abstractcore.json`) fits as is.

No method returns or logs a secret. `public()` is the one read shape for CLIs, consoles and
APIs.
"""

from __future__ import annotations

import json
import os
import secrets as _secrets
import ssl
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from . import legacy
from .client import EmailClient, tls_context
from .context import EmailContext
from .errors import (
    EmailAgentToolsOff,
    EmailError,
    EmailInvalidSettings,
    EmailNotConfigured,
    EmailSecretUnavailable,
)
from .limits import SendRateLimiter
from .models import (
    DEFAULT_PER_DAY,
    DEFAULT_PER_HOUR,
    EmailAccount,
    EmailSecret,
    ImapSettings,
    SendLimits,
    SmtpSettings,
)
from .oauth import OAuthTokenProvider
from .policy import RecipientPolicy, normalize_address
from .vault import KEY_FILE_WARNING, SecretVault

SCHEMA = "email_settings_v1"

# "Agent email tools" (default OFF): agents get the email tools with this account only when the
# account is connected, turned on, and this toggle is on. The same words as the gateway's
# per-user toggle (Settings -> My email -> Agent email tools).
AGENT_TOOLS_LABEL = "Agent email tools"
AGENT_TOOLS_OFF_CAUSE = "Agent email tools are off for this AbstractCore install."
AGENT_TOOLS_OFF_FIX = (
    "Turn on \"Agent email tools\": `abstractcore email agent-tools on`, or the Email page of the "
    "AbstractCore console (web or terminal). The account must also be connected and turned on."
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def limits_source(stored: Any) -> str:
    """Where an account's send limits come from, from the stored `limits` section.

    - nothing stored -> "default": the current defaults (100 per hour, 1000 per day; 20 / 100 until 2.21);
    - `set_by: user` -> "user": the user's choice, kept across upgrades;
    - exactly the old defaults (20 per hour AND 100 per day) without the marker -> "default": 2.21 and
      earlier stored the then-defaults at connect, so an unmarked 20 / 100 is the old default, not a
      choice (operator ruling 2026-10-01: those accounts move to the new defaults). Nothing is rewritten
      on disk; the next `set_limits` stores the user's choice with the marker;
    - any other values without the marker -> "legacy": a value someone set before the marker existed,
      kept as stored: `abstractcore email limits reset` (or setting new values) moves the account on.
    """

    if not isinstance(stored, dict) or not any(k in stored for k in ("per_hour", "per_day")):
        return "default"
    if stored.get("set_by") == "user":
        return "user"
    if _is_old_default_pair(stored):
        return "default"
    return "legacy"


# The send limits 2.21 and earlier wrote at connect (no `set_by` marker existed then).
OLD_DEFAULT_LIMITS = {"per_hour": 20, "per_day": 100}


def _is_old_default_pair(stored: Dict[str, Any]) -> bool:
    try:
        return (
            int(stored.get("per_hour")) == OLD_DEFAULT_LIMITS["per_hour"]
            and int(stored.get("per_day")) == OLD_DEFAULT_LIMITS["per_day"]
        )
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class EmailSettings:
    enabled: bool
    agent_tools: bool
    account: Optional[EmailAccount]
    policy: RecipientPolicy
    policy_is_default: bool
    limits: SendLimits
    registered_address: str
    legacy_import: Dict[str, Any]
    # Where `limits` comes from: "default" (nothing stored: the current defaults, which follow
    # a later change of the defaults), "user" (set with `set_limits`, never changed by an
    # upgrade) or "legacy" (stored by 2.21 or earlier without a marker: kept as stored, see
    # `limits_source`).
    limits_source: str = "default"

    @property
    def self_address(self) -> str:
        """The registered address, else the account's address."""

        if self.registered_address:
            return self.registered_address
        return self.account.address if self.account else ""

    @property
    def self_addresses(self) -> tuple:
        """Every own address (registered address and mailbox address): always allowed by the
        recipient policy."""

        return tuple(a for a in (self.registered_address, self.account.address if self.account else "") if a)


def default_display_name(address: str) -> str:
    """The sender name of a new connection nobody named: the address's local part
    (`jean.dupont@example.org` -> `jean.dupont`)."""

    return str(address or "").strip().split("@", 1)[0]


class EmailAccountStore:
    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        *,
        config_dir: Optional[Union[str, Path]] = None,
        key_backend: str = "auto",
        environ: Optional[Dict[str, str]] = None,
    ) -> None:
        from abstractcore.config.manager import resolve_config_file

        self.config_file = resolve_config_file(config_dir, config_file)
        self.email_dir = self.config_file.parent / "email"
        self.vault = SecretVault(self.email_dir, key_backend=key_backend)
        self._environ = environ

    def __repr__(self) -> str:
        return f"EmailAccountStore(config_file={str(self.config_file)!r})"

    # -- config section --------------------------------------------------------------------

    def _manager(self):
        from abstractcore.config.manager import ConfigurationManager

        return ConfigurationManager(config_file=self.config_file, apply_env=False)

    def _section(self) -> Dict[str, Any]:
        return asdict(self._manager().config.email)

    def _update(self, **values: Any) -> None:
        self._manager().update_email_settings(**values)

    def settings(self) -> EmailSettings:
        sec = self._section()
        try:
            account = EmailAccount.from_dict(sec.get("account") or {})
        except EmailInvalidSettings as err:
            raise EmailInvalidSettings(
                f"The stored email account settings are not valid: {err.cause}",
                f"{err.fix} Or connect the account again (`abstractcore email connect ...`).",
            ) from None
        registered = str(sec.get("registered_address") or "").strip()
        policy = RecipientPolicy.from_dict(sec.get("policy") or {})
        is_default = policy is None
        if policy is None:
            policy = RecipientPolicy.default_for(registered or (account.address if account else ""))
        return EmailSettings(
            enabled=bool(sec.get("enabled", True)),
            agent_tools=sec.get("agent_tools") is True,
            account=account,
            policy=policy,
            policy_is_default=is_default,
            limits=SendLimits.from_dict(
                {} if limits_source(sec.get("limits")) == "default" else (sec.get("limits") or {})
            ),
            limits_source=limits_source(sec.get("limits")),
            registered_address=registered,
            legacy_import=dict(sec.get("legacy_import") or {}),
        )

    # -- status ----------------------------------------------------------------------------

    @property
    def _status_path(self) -> Path:
        return self.email_dir / "status.json"

    def _read_status(self) -> Dict[str, Any]:
        try:
            doc = json.loads(self._status_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return doc if isinstance(doc, dict) else {}

    def _write_status(self, doc: Dict[str, Any]) -> None:
        self.email_dir.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.email_dir, 0o700)
        except OSError:
            pass
        tmp = self._status_path.with_name(f"status.json.{os.getpid()}-{_secrets.token_hex(4)}.tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2)
        os.replace(tmp, self._status_path)

    def _record_test(self, result: Dict[str, Any]) -> None:
        legs = {k: v for k, v in result.items() if k in ("imap", "smtp")}
        failures = [v for v in legs.values() if isinstance(v, dict) and v.get("ok") is False]
        doc = self._read_status()
        doc["last_test"] = _now_iso()
        doc["legs"] = {
            k: ({"ok": v.get("ok")} if v.get("ok") is not False else {"ok": False, "code": v.get("code"), "cause": v.get("cause"), "fix": v.get("fix")})
            for k, v in legs.items()
            if isinstance(v, dict)
        }
        if failures:
            f = failures[0]
            doc["last_error"] = {"code": f.get("code"), "cause": f.get("cause"), "fix": f.get("fix"), "at": doc["last_test"]}
        else:
            doc["last_ok"] = doc["last_test"]
            doc.pop("last_error", None)
        self._write_status(doc)

    def record_error(self, err: EmailError) -> None:
        """Hosts (watchers, dispatchers) record the latest typed failure for the status view."""

        doc = self._read_status()
        doc["last_error"] = {"code": err.code, "cause": err.cause, "fix": err.fix, "at": _now_iso()}
        self._write_status(doc)

    # -- secret ----------------------------------------------------------------------------

    def _load_secret(self) -> Optional[EmailSecret]:
        payload = self.vault.load()
        if payload is None:
            return None
        return EmailSecret.from_sealed_payload(payload)

    def _limiter(self, limits: SendLimits) -> SendRateLimiter:
        return SendRateLimiter(self.email_dir / "sends.json", limits)

    # -- operations ------------------------------------------------------------------------

    def connect(
        self,
        account: EmailAccount,
        secret: EmailSecret,
        *,
        test: bool = True,
        registered_address: Optional[str] = None,
        ssl_context: Optional[ssl.SSLContext] = None,
    ) -> Dict[str, Any]:
        """Test (unless `test=False`), then store the account and seal the secret.

        A failed test raises the first leg's typed error and stores nothing. A new account's
        recipient policy defaults to an allowlist holding the registered address.

        Nobody is asked for a display name (the From header's name, `client.build_mime`): an
        account given none keeps the stored account's name, else gets the address's local part
        (`default_display_name`). The email address ("self") is set to the mailbox address when
        none is stored and none is given (`registered_address`), so the address the console
        shows is always explicit.
        """

        if not isinstance(secret, EmailSecret) or not secret.is_set:
            raise EmailSecretUnavailable(
                "No password or OAuth token was given.",
                "Connect with --password <value> (or --oauth <provider> for OAuth2 sign-in).",
            )
        if registered_address:
            try:
                normalize_address(registered_address)
            except ValueError:
                raise EmailInvalidSettings(
                    f"The registered address {registered_address!r} is not a valid email address.",
                    "Give --registered-address as name@example.test.",
                ) from None
        result: Optional[Dict[str, Any]] = None
        if test:
            client = EmailClient(account, secret, ssl_context=ssl_context)
            result = client.test()
            for leg in ("imap", "smtp"):
                r = result.get(leg) or {}
                if r.get("ok") is False:
                    raise _error_from_leg(r)
        self.vault.store(secret.sealed_payload(), reuse_key=False)
        current = self.settings()
        if not account.display_name:
            kept = current.account.display_name if current.account is not None else ""
            account = replace(account, display_name=kept or default_display_name(account.address))
        values: Dict[str, Any] = {"account": account.to_dict()}
        if registered_address is not None:
            values["registered_address"] = registered_address.strip()
        elif not current.registered_address:
            registered_address = account.address
            values["registered_address"] = account.address
        if current.policy_is_default:
            base = (registered_address or current.registered_address or account.address).strip()
            values["policy"] = RecipientPolicy.default_for(base).to_dict()
        # The send limits are not written here: nothing stored means the current defaults
        # (`limits_source` "default"), so a later change of the defaults reaches this account.
        if not current.legacy_import:
            values["legacy_import"] = {"done": True, "at": _now_iso(), "source": "none"}
        self._update(**values)
        doc = {"connected_at": _now_iso()}
        self._write_status(doc)
        if result is not None:
            self._record_test(result)
        return self.public()

    def test(self, *, ssl_context: Optional[ssl.SSLContext] = None) -> Dict[str, Any]:
        """Sign in on each leg of the stored account; records and returns the per-leg result."""

        ctx = self.context(ssl_context=ssl_context, require_enabled=False)
        result = EmailClient(ctx.account, ctx.secret, token_provider=ctx.token_provider, ssl_context=ssl_context).test()
        self._record_test(result)
        result["ok"] = all((result.get(k) or {}).get("ok") is not False for k in ("imap", "smtp"))
        return result

    def disconnect(self) -> Dict[str, Any]:
        """Delete the sealed secret and its key, and the account settings. Policy and limits stay."""

        self.vault.delete()
        self._update(account={})
        try:
            self._status_path.unlink()
        except OSError:
            pass
        return self.public()

    def set_policy(
        self,
        *,
        mode: Optional[str] = None,
        add: Sequence[str] = (),
        remove: Sequence[str] = (),
        clear: bool = False,
        always_allow: Optional[Sequence[str]] = None,
        always_deny: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Change the recipient policy. `always_allow` / `always_deny`, when given, replace
        that list; None keeps it."""

        current = self.settings().policy
        new = current.with_changes(
            mode=mode, add=add, remove=remove, clear=clear, always_allow=always_allow, always_deny=always_deny
        )
        self._update(policy=new.to_dict())
        return self.public()

    def set_limits(self, *, per_hour: Optional[int] = None, per_day: Optional[int] = None) -> Dict[str, Any]:
        """Set the send limits the user chose. Only the windows set (now or before) are stored,
        marked `set_by: user`; an upgrade never changes them. A window never set keeps following
        the defaults."""

        stored = self._section().get("limits") or {}
        chosen = {k: stored[k] for k in ("per_hour", "per_day") if isinstance(stored, dict) and k in stored}
        if per_hour is not None:
            chosen["per_hour"] = per_hour
        if per_day is not None:
            chosen["per_day"] = per_day
        new = SendLimits.build(chosen.get("per_hour", DEFAULT_PER_HOUR), chosen.get("per_day", DEFAULT_PER_DAY))
        doc: Dict[str, Any] = {k: getattr(new, k) for k in ("per_hour", "per_day") if k in chosen}
        if doc:
            doc["set_by"] = "user"
        self._update(limits=doc)
        return self.public()

    def reset_limits(self) -> Dict[str, Any]:
        """Forget the stored send limits: the account follows the defaults again."""

        self._update(limits={})
        return self.public()

    def set_enabled(self, enabled: bool) -> Dict[str, Any]:
        self._update(enabled=bool(enabled))
        return self.public()

    def set_folder(self, folder: str) -> Dict[str, Any]:
        """The IMAP folder the mailbox is read from (agents' list/search, the mail watcher).

        Empty = INBOX. The connection (servers, password or tokens) is kept; nothing is tested.
        """

        st = self.settings()
        acct = st.account
        if acct is None:
            raise EmailNotConfigured(
                "No mailbox is connected, so there is no folder to set.",
                "Connect a mailbox first (`abstractcore email connect --address <a> --password-stdin`).",
            )
        if acct.imap is None:
            raise EmailInvalidSettings(
                "This mailbox has no IMAP (reading) server, so it has no folder.",
                "Connect it again with an IMAP server to read mail.",
            )
        name = str(folder or "").strip() or "INBOX"
        if any(ord(ch) < 32 or ord(ch) == 127 for ch in name) or len(name) > 255:
            raise EmailInvalidSettings(
                f"{name[:60]!r} is not a usable folder name (control characters or longer than 255).",
                "Give the folder's name as the mail server lists it (`abstractcore email folders`), e.g. INBOX.",
            )
        raw = acct.to_dict()
        raw["imap"] = {**(raw.get("imap") or {}), "folder": name}
        self._update(account=EmailAccount.from_dict(raw).to_dict())
        return self.public()

    def set_agent_tools(self, enabled: bool) -> Dict[str, Any]:
        """The "Agent email tools" toggle (default off). Turning it on without a connected,
        turned-on account is allowed and changes nothing until the account is usable."""

        self._update(agent_tools=bool(enabled))
        return self.public()

    def agent_tools_status(self, st: Optional[EmailSettings] = None) -> Dict[str, Any]:
        """`{enabled, active, reason}`: the toggle, whether agents have the tools, and why not."""

        if st is None:
            try:
                st = self.settings()
            except EmailInvalidSettings:
                return {"enabled": False, "active": False, "reason": "the stored email settings are not valid"}
        usable = bool(st.account is not None and st.enabled and self.vault.exists())
        reason = ""
        if not st.agent_tools:
            reason = "off (your choice; default)"
        elif not usable:
            reason = "no connected, turned-on email account"
        return {"enabled": bool(st.agent_tools), "active": bool(st.agent_tools and usable), "reason": reason}

    def agent_context(self, *, ssl_context: Optional[ssl.SSLContext] = None) -> EmailContext:
        """`context()` for an AGENT's tool call: also requires "Agent email tools" to be on.

        Raises `EmailNotConfigured` / `EmailSecretUnavailable` / `EmailDisabled` first (the
        account itself), then `EmailAgentToolsOff` when the toggle is off.
        """

        ctx = self.context(ssl_context=ssl_context)
        if not self.settings().agent_tools:
            raise EmailAgentToolsOff(AGENT_TOOLS_OFF_CAUSE, AGENT_TOOLS_OFF_FIX)
        return ctx

    def set_registered_address(self, address: str) -> Dict[str, Any]:
        addr = str(address or "").strip()
        if addr:
            try:
                normalize_address(addr)
            except ValueError:
                raise EmailInvalidSettings(
                    f"The registered address {addr!r} is not a valid email address.",
                    "Give the address as name@example.test.",
                ) from None
        # The default Always allowed list holds the registered address: when it changes, the
        # entry follows (the old one goes, the new one comes), so the policy the console shows
        # stays true.
        try:
            st = self.settings()
            old_addr = (st.registered_address or "").strip().lower()
            pol = st.policy
        except EmailInvalidSettings:
            old_addr, pol = "", None
        self._update(registered_address=addr)
        if pol is not None and pol.mode == "allowlist":
            entries = list(pol.always_allow)
            if old_addr and old_addr in entries:
                entries = [e for e in entries if e != old_addr]
            if addr:
                try:
                    norm = normalize_address(addr)
                except ValueError:
                    norm = ""
                if norm and norm not in entries:
                    entries.append(norm)
            if tuple(entries) != tuple(pol.always_allow):
                self._update(policy=replace(pol, always_allow=tuple(entries)).to_dict())
        return self.public()

    def context(self, *, ssl_context: Optional[ssl.SSLContext] = None, require_enabled: bool = True) -> EmailContext:
        """The account ready for use: raises a typed error when not connected / turned off."""

        st = self.settings()
        if st.account is None:
            raise EmailNotConfigured(
                "No email account is connected.",
                "Connect one: `abstractcore email connect --address <a> --imap-host <h> --smtp-host <h> "
                "--password <value>` (or the Email page of the AbstractCore console).",
            )
        secret = self._load_secret()
        if secret is None:
            raise EmailSecretUnavailable(
                "The email account has no stored password or token.",
                "Connect the account again with --password <value> (or --oauth <provider>).",
            )
        provider = None
        if st.account.auth_kind == "oauth2":
            verify = ssl_context
            if verify is None:
                ca = (st.account.imap.ca_file if st.account.imap else "") or (st.account.smtp.ca_file if st.account.smtp else "")
                verify = tls_context(ca) if ca else None
            provider = OAuthTokenProvider(
                st.account.oauth,
                secret,
                on_update=lambda new: self.vault.store(new.sealed_payload(), reuse_key=True),
                verify=verify,
            )
        ctx = EmailContext(
            account=st.account,
            secret=secret,
            policy=st.policy,
            limits=st.limits,
            limiter=self._limiter(st.limits),
            enabled=st.enabled,
            token_provider=provider,
            ssl_context=ssl_context,
            source="settings",
            registered_address=st.self_address,
        )
        if require_enabled:
            ctx.require_enabled()
        return ctx

    # -- legacy ----------------------------------------------------------------------------

    def ensure_legacy_imported(self) -> List[str]:
        """Import the pre-2.20 configuration once (see `legacy.py`). Returns notices to show."""

        env = dict(os.environ if self._environ is None else self._environ)
        sec = self._section()
        done = bool((sec.get("legacy_import") or {}).get("done"))
        has_account = bool(sec.get("account"))
        if done or has_account:
            if not done:
                self._update(legacy_import={"done": True, "at": _now_iso(), "source": "none"})
            return legacy.notices(env)
        found = legacy.detect(sec, env)
        if found is None:
            return []
        notes: List[str] = []
        try:
            imap = ImapSettings.build(
                found.imap["host"],
                port=found.imap.get("port"),
                security=found.imap.get("security", "ssl"),
                folder=found.imap.get("folder", "INBOX"),
                ca_file=found.imap.get("ca_file", ""),
            ) if found.imap.get("host") else None
            smtp = SmtpSettings.build(
                found.smtp["host"],
                port=found.smtp.get("port"),
                security=found.smtp.get("security", "starttls"),
                ca_file=found.smtp.get("ca_file", ""),
            ) if found.smtp.get("host") else None
            account = EmailAccount.build(address=found.address, username=found.username, imap=imap, smtp=smtp)
        except EmailInvalidSettings as err:
            self._update(legacy_import={"done": True, "at": _now_iso(), "source": found.source, "error": err.cause})
            return [f"The pre-2.20 email configuration ({found.source}) could not be imported: {err.cause} "
                    "Connect the account with `abstractcore email connect ...`."] + legacy.notices(env)
        password, problem = legacy.read_password(found, env)
        imported_secret = False
        if password:
            self.vault.store(EmailSecret(password).sealed_payload(), reuse_key=False)
            imported_secret = True
        else:
            notes.append(
                f"The password was not imported ({problem}); set it with "
                "`abstractcore email connect ... --password <value>`."
            )
        values: Dict[str, Any] = {
            "account": account.to_dict(),
            "legacy_import": {"done": True, "at": _now_iso(), "source": found.source, "secret_imported": imported_secret},
        }
        if not sec.get("policy"):
            values["policy"] = RecipientPolicy.default_for(account.address).to_dict()
        manager = self._manager()
        manager.clear_legacy_email_fields()
        manager.update_email_settings(**values)
        label = {"file": "the ABSTRACT_EMAIL_ACCOUNTS_CONFIG file", "environment": "ABSTRACT_EMAIL_* environment variables", "config": "the pre-2.20 email fields of abstractcore.json"}[found.source]
        notes.insert(
            0,
            f"Imported the email account {account.address} from {label} into AbstractCore settings (email.account); "
            "the old configuration is ignored from now on. Review it with `abstractcore email status`; the recipient "
            f"policy starts as an allowlist holding {account.address}.",
        )
        notes.extend(found.notes)
        return notes + legacy.notices(env)

    # -- read shape ------------------------------------------------------------------------

    def public(self) -> Dict[str, Any]:
        """Settings and status for CLIs, consoles and APIs. Never a secret."""

        try:
            st = self.settings()
            settings_error = None
        except EmailInvalidSettings as err:
            st = None
            settings_error = err.to_dict(include_details=False)
        location = self.vault.location()
        status = self._read_status()
        acct = st.account if st else None
        out: Dict[str, Any] = {
            "schema": SCHEMA,
            "configured": acct is not None,
            "enabled": st.enabled if st else False,
            "agent_tools": self.agent_tools_status(st) if st else {"enabled": False, "active": False, "reason": "the stored email settings are not valid"},
            "address": acct.address if acct else "",
            "display_name": acct.display_name if acct else "",
            "username": acct.username if acct else "",
            "auth_kind": acct.auth_kind if acct else "",
            "imap": acct.imap.to_dict() if acct and acct.imap else None,
            "smtp": acct.smtp.to_dict() if acct and acct.smtp else None,
            "oauth": (
                {
                    "provider": acct.oauth.provider,
                    "client_id": acct.oauth.client_id,
                    "client_source": acct.oauth.client_source,
                    "tenant": acct.oauth.tenant,
                    "scopes": list(acct.oauth.scopes),
                }
                if acct and acct.oauth
                else None
            ),
            "can_read": bool(acct and acct.can_read),
            "can_send": bool(acct and acct.can_send),
            "secret_set": bool(location),
            "secret_storage": {"keyring": "os-keychain", "file": "key-file"}.get(location, ""),
            "secret_warning": KEY_FILE_WARNING if location == "file" else "",
            "policy": (
                {**st.policy.to_dict(), "default": st.policy_is_default}
                if st
                else None
            ),
            "limits": None,
            "registered_address": st.self_address if st else "",
            # The email address as stored ("" = none set; `registered_address` then falls back
            # to the mailbox's own address).
            "registered_address_stored": st.registered_address if st else "",
            "status": {
                "last_test": status.get("last_test", ""),
                "last_ok": status.get("last_ok", ""),
                "last_error": status.get("last_error"),
                "legs": status.get("legs", {}),
            },
            "source": "settings",
            "legacy_import": st.legacy_import if st else {},
            "config_file": str(self.config_file),
        }
        if st:
            try:
                usage = self._limiter(st.limits).usage()
            except OSError:
                usage = {"used_last_hour": 0, "used_last_day": 0}
            out["limits"] = {
                "per_hour": st.limits.per_hour,
                "per_day": st.limits.per_day,
                "source": st.limits_source,
                "used_last_hour": usage.get("used_last_hour", 0),
                "used_last_day": usage.get("used_last_day", 0),
            }
        if settings_error:
            out["settings_error"] = settings_error
        return out


def _error_from_leg(leg: Dict[str, Any]) -> EmailError:
    from . import errors as E

    classes = {cls.code: cls for cls in vars(E).values() if isinstance(cls, type) and issubclass(cls, E.EmailError)}
    cls = classes.get(str(leg.get("code") or ""), E.EmailError)
    return cls(str(leg.get("cause") or ""), str(leg.get("fix") or ""), details=leg.get("details") or {})
