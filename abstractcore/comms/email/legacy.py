"""One-time import of the pre-2.20 email configuration, then the old sources are ignored.

Before 2.20 the email tools read one process-wide account from, in order:

1. an accounts file named by `ABSTRACT_EMAIL_ACCOUNTS_CONFIG` (YAML/JSON, `${VAR}` values);
2. `ABSTRACT_EMAIL_{SMTP,IMAP}_*` environment variables;
3. the flat `email.smtp_* / email.imap_*` fields of `abstractcore.json`;

with the password read from the environment variable named by `*_password_env_var`.

Email is now configured in AbstractCore's settings (`abstractcore email connect`), with the
credentials encrypted. On first use, when no account is configured and one of the sources
above describes one, it is imported ONCE into the settings (the password is read from its
variable at that moment and sealed) and `email.legacy_import` records it. From then on the
variables are ignored; `notices()` names each variable still set and the setting that
replaced it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

LEGACY_ENV_PREFIX = "ABSTRACT_EMAIL_"

# Environment variable -> the setting that replaces it (for the "ignored" notice).
_REPLACEMENTS = {
    "ABSTRACT_EMAIL_ACCOUNTS_CONFIG": "email.account (abstractcore email connect)",
    "ABSTRACT_EMAIL_DEFAULT_ACCOUNT": "email.account (one account per install)",
    "ABSTRACT_EMAIL_ACCOUNT_NAME": "email.account (one account per install)",
    "ABSTRACT_EMAIL_SMTP_HOST": "email.account.smtp.host (--smtp-host)",
    "ABSTRACT_EMAIL_SMTP_PORT": "email.account.smtp.port (--smtp-port)",
    "ABSTRACT_EMAIL_SMTP_USERNAME": "email.account.username (--username)",
    "ABSTRACT_EMAIL_SMTP_PASSWORD_ENV_VAR": "the encrypted password (--password <value>)",
    "ABSTRACT_EMAIL_SMTP_STARTTLS": "email.account.smtp.security (--smtp-security ssl|starttls)",
    "ABSTRACT_EMAIL_FROM": "email.account.address (--address)",
    "ABSTRACT_EMAIL_REPLY_TO": "email.account.address (replies go to the account address)",
    "ABSTRACT_EMAIL_IMAP_HOST": "email.account.imap.host (--imap-host)",
    "ABSTRACT_EMAIL_IMAP_PORT": "email.account.imap.port (--imap-port)",
    "ABSTRACT_EMAIL_IMAP_USERNAME": "email.account.username (--username)",
    "ABSTRACT_EMAIL_IMAP_PASSWORD_ENV_VAR": "the encrypted password (--password <value>)",
    "ABSTRACT_EMAIL_IMAP_FOLDER": "email.account.imap.folder (--imap-folder)",
}


@dataclass
class LegacyAccount:
    source: str  # "file" | "environment" | "config"
    address: str = ""
    username: str = ""
    display_name: str = ""
    imap: Dict[str, Any] = field(default_factory=dict)
    smtp: Dict[str, Any] = field(default_factory=dict)
    password_env_var: str = ""
    notes: List[str] = field(default_factory=list)


def _s(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _i(v: Any, default: int) -> int:
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _b(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    s = _s(v).lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return None


def legacy_env_names(environ: Optional[Dict[str, str]] = None) -> List[str]:
    env = os.environ if environ is None else environ
    return sorted(k for k in env if k.startswith(LEGACY_ENV_PREFIX) and _s(env.get(k)))


def notices(environ: Optional[Dict[str, str]] = None) -> List[str]:
    """One sentence per legacy variable still set, naming the setting that replaced it."""

    out = []
    for name in legacy_env_names(environ):
        repl = _REPLACEMENTS.get(name, "the email settings (abstractcore email status)")
        out.append(f"{name} is set but ignored: email is configured in AbstractCore settings ({repl}).")
    return out


def _interpolate(value: Any, env: Dict[str, str], missing: set) -> Any:
    if isinstance(value, dict):
        return {k: _interpolate(v, env, missing) for k, v in value.items()}
    if isinstance(value, list):
        return [_interpolate(v, env, missing) for v in value]
    if not isinstance(value, str) or "${" not in value:
        return value
    out = []
    i = 0
    while i < len(value):
        start = value.find("${", i)
        if start == -1:
            out.append(value[i:])
            break
        end = value.find("}", start)
        if end == -1:
            out.append(value[i:])
            break
        out.append(value[i:start])
        body = value[start + 2 : end]
        name, _sep, default = body.partition(":-")
        got = env.get(name.strip())
        if got is not None and got.strip():
            out.append(got)
        elif _sep:
            out.append(default)
        else:
            missing.add(name.strip())
        i = end + 1
    return "".join(out)


def _from_file(path: str, env: Dict[str, str]) -> Optional[LegacyAccount]:
    p = Path(os.path.expanduser(path))
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            data = json.loads(text)
        else:
            import yaml  # type: ignore

            data = yaml.safe_load(text)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    missing: set = set()
    data = _interpolate(data, env, missing)
    accounts = data.get("accounts")
    if not isinstance(accounts, dict) or not accounts:
        return None
    name = _s(env.get("ABSTRACT_EMAIL_DEFAULT_ACCOUNT")) or _s(data.get("default_account"))
    if not name:
        name = next(iter(accounts))
    raw = accounts.get(name)
    if not isinstance(raw, dict):
        return None
    acc = LegacyAccount(source="file")
    imap = raw.get("imap") if isinstance(raw.get("imap"), dict) else {}
    smtp = raw.get("smtp") if isinstance(raw.get("smtp"), dict) else {}
    if imap:
        acc.imap = {
            "host": _s(imap.get("host") or imap.get("imap_host")),
            "port": _i(imap.get("port") or imap.get("imap_port"), 993),
            "security": "ssl",
            "folder": _s(imap.get("mailbox") or imap.get("folder")) or "INBOX",
            "ca_file": _s(imap.get("ca_file")),
        }
    if smtp:
        port = _i(smtp.get("port") or smtp.get("smtp_port"), 587)
        starttls = _b(smtp.get("use_starttls"))
        if starttls is None:
            starttls = _b(smtp.get("starttls"))
        if starttls is None:
            starttls = port != 465
        acc.smtp = {
            "host": _s(smtp.get("host") or smtp.get("smtp_host")),
            "port": port,
            "security": "starttls" if starttls else "ssl",
            "ca_file": _s(smtp.get("ca_file")),
        }
    acc.username = _s(smtp.get("username") or smtp.get("user")) or _s(imap.get("username") or imap.get("user"))
    acc.address = _s(smtp.get("from_email") or smtp.get("from")) or acc.username
    acc.password_env_var = _s(smtp.get("password_env_var")) or _s(imap.get("password_env_var")) or "EMAIL_PASSWORD"
    others = [k for k in accounts if k != name]
    if others:
        acc.notes.append(f"Only the default account {name!r} was imported (one account per install); not imported: {', '.join(others)}.")
    if missing:
        acc.notes.append(f"The accounts file referenced unset variables: {', '.join(sorted(missing))}.")
    return acc


def _from_flat(values: Dict[str, Any], source: str) -> Optional[LegacyAccount]:
    smtp_host = _s(values.get("smtp_host"))
    imap_host = _s(values.get("imap_host"))
    if not smtp_host and not imap_host:
        return None
    acc = LegacyAccount(source=source)
    if imap_host:
        acc.imap = {
            "host": imap_host,
            "port": _i(values.get("imap_port"), 993),
            "security": "ssl",
            "folder": _s(values.get("imap_folder")) or "INBOX",
        }
    if smtp_host:
        port = _i(values.get("smtp_port"), 587)
        # The pre-2.20 rule: an explicit ABSTRACT_EMAIL_SMTP_STARTTLS wins; otherwise port 465
        # meant implicit TLS whatever the stored flag said.
        starttls = _b(values.get("smtp_starttls_env"))
        if starttls is None:
            starttls = False if port == 465 else (_b(values.get("smtp_use_starttls")) is not False)
        acc.smtp = {"host": smtp_host, "port": port, "security": "starttls" if starttls else "ssl"}
    acc.username = _s(values.get("smtp_username")) or _s(values.get("imap_username"))
    acc.address = _s(values.get("from_email")) or acc.username
    acc.password_env_var = (
        (_s(values.get("smtp_password_env_var")) if smtp_host else "")
        or (_s(values.get("imap_password_env_var")) if imap_host else "")
        or "EMAIL_PASSWORD"
    )
    if _s(values.get("smtp_username")) and _s(values.get("imap_username")) and _s(values.get("smtp_username")) != _s(values.get("imap_username")):
        acc.notes.append("IMAP and SMTP used different user names; the SMTP user name was imported for both.")
    return acc


def detect(config_section: Dict[str, Any], environ: Optional[Dict[str, str]] = None) -> Optional[LegacyAccount]:
    """The legacy account the pre-2.20 tools would have used, or None."""

    env = dict(os.environ if environ is None else environ)
    path = _s(env.get("ABSTRACT_EMAIL_ACCOUNTS_CONFIG"))
    if path:
        acc = _from_file(path, env)
        if acc is not None:
            return acc
    flat_env = {
        "smtp_host": env.get("ABSTRACT_EMAIL_SMTP_HOST"),
        "smtp_port": env.get("ABSTRACT_EMAIL_SMTP_PORT"),
        "smtp_username": env.get("ABSTRACT_EMAIL_SMTP_USERNAME"),
        "smtp_password_env_var": env.get("ABSTRACT_EMAIL_SMTP_PASSWORD_ENV_VAR"),
        "smtp_starttls_env": env.get("ABSTRACT_EMAIL_SMTP_STARTTLS"),
        "from_email": env.get("ABSTRACT_EMAIL_FROM"),
        "imap_host": env.get("ABSTRACT_EMAIL_IMAP_HOST"),
        "imap_port": env.get("ABSTRACT_EMAIL_IMAP_PORT"),
        "imap_username": env.get("ABSTRACT_EMAIL_IMAP_USERNAME"),
        "imap_password_env_var": env.get("ABSTRACT_EMAIL_IMAP_PASSWORD_ENV_VAR"),
        "imap_folder": env.get("ABSTRACT_EMAIL_IMAP_FOLDER"),
    }
    # Environment values override the flat config fields key by key (the old precedence).
    merged = {k: (v if _s(v) else config_section.get(k)) for k, v in flat_env.items()}
    merged["smtp_use_starttls"] = config_section.get("smtp_use_starttls")
    env_used = any(_s(v) for v in flat_env.values())
    acc = _from_flat(merged, "environment" if env_used else "config")
    return acc


def read_password(acc: LegacyAccount, environ: Optional[Dict[str, str]] = None) -> Tuple[str, str]:
    """(password, problem). The password comes from the variable the legacy config named."""

    env = os.environ if environ is None else environ
    name = acc.password_env_var
    if not name or not (name[0].isalpha() or name[0] == "_") or not all(c.isalnum() or c == "_" for c in name):
        return "", "the legacy password_env_var is not a variable name"
    value = env.get(name)
    if value is None or not str(value).strip():
        return "", f"the variable {name} is not set"
    return str(value), ""
