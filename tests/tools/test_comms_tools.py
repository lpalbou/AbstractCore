from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path
import textwrap
from unittest.mock import MagicMock, patch

import pytest


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here] + list(here.parents):
        if (p / "configs" / "emails.yaml").is_file():
            return p
    raise AssertionError("Could not find repo root containing configs/emails.yaml")


def test_send_email_fails_when_no_accounts_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import send_email

    monkeypatch.delenv("ABSTRACT_EMAIL_ACCOUNTS_CONFIG", raising=False)
    monkeypatch.delenv("ABSTRACT_EMAIL_SMTP_HOST", raising=False)
    monkeypatch.delenv("ABSTRACT_EMAIL_SMTP_USERNAME", raising=False)
    monkeypatch.delenv("ABSTRACT_EMAIL_IMAP_HOST", raising=False)
    monkeypatch.delenv("ABSTRACT_EMAIL_IMAP_USERNAME", raising=False)

    with patch("abstractcore.config.manager.get_config_manager", side_effect=Exception("no config")):
        out = send_email(to="you@example.com", subject="Hello", body_text="Hi")
    assert out["success"] is False
    assert "No email accounts configured" in str(out.get("error") or "")


def test_list_email_accounts_reports_env_fallback_account(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import list_email_accounts

    monkeypatch.setenv("ABSTRACT_EMAIL_ACCOUNT_NAME", "primary")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_PORT", "587")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_USERNAME", "me@example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_PASSWORD_ENV_VAR", "DEFAULT_EMAIL_PASSWORD")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_HOST", "imap.example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_PORT", "993")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_USERNAME", "me@example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_PASSWORD_ENV_VAR", "DEFAULT_EMAIL_PASSWORD")

    out = list_email_accounts()
    assert out["success"] is True
    assert out["source"] == "env"
    assert any(a.get("account") == "primary" and a.get("can_read") and a.get("can_send") for a in out.get("accounts") or [])


def test_accounts_config_default_file_interpolates_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import list_email_accounts

    cfg_path = _repo_root() / "configs" / "emails.yaml"

    monkeypatch.setenv("ABSTRACT_EMAIL_ACCOUNTS_CONFIG", str(cfg_path))

    monkeypatch.setenv("DEFAULT_EMAIL_PASSWORD", "pw")

    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_HOST", "imap.example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_PORT", "993")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_USERNAME", "me@example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_PASSWORD_ENV_VAR", "DEFAULT_EMAIL_PASSWORD")

    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_PORT", "587")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_USERNAME", "me@example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_PASSWORD_ENV_VAR", "DEFAULT_EMAIL_PASSWORD")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_STARTTLS", "1")

    out = list_email_accounts()
    assert out["success"] is True
    assert out["source"] == "yaml"
    assert out.get("default_account") == "default"
    accounts = out.get("accounts")
    assert isinstance(accounts, list) and len(accounts) == 1
    assert accounts[0].get("account") == "default"
    assert accounts[0].get("can_read") is True
    assert accounts[0].get("can_send") is True


def test_accounts_config_default_file_fails_fast_when_required_env_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import list_email_accounts

    cfg_path = _repo_root() / "configs" / "emails.yaml"
    monkeypatch.setenv("ABSTRACT_EMAIL_ACCOUNTS_CONFIG", str(cfg_path))

    monkeypatch.delenv("ABSTRACT_EMAIL_IMAP_HOST", raising=False)
    monkeypatch.delenv("ABSTRACT_EMAIL_IMAP_USERNAME", raising=False)
    monkeypatch.delenv("ABSTRACT_EMAIL_SMTP_HOST", raising=False)
    monkeypatch.delenv("ABSTRACT_EMAIL_SMTP_USERNAME", raising=False)

    out = list_email_accounts()
    assert out["success"] is False
    err = str(out.get("error") or "")
    assert "missing env vars" in err.lower()
    assert "ABSTRACT_EMAIL_IMAP_HOST" in err


def test_send_email_uses_starttls_when_port_587(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import send_email

    monkeypatch.setenv("DEFAULT_EMAIL_PASSWORD", "pw")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_PORT", "587")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_USERNAME", "me@example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_PASSWORD_ENV_VAR", "DEFAULT_EMAIL_PASSWORD")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_STARTTLS", "1")

    smtp = MagicMock()
    with patch("smtplib.SMTP", return_value=smtp) as smtp_ctor:
        out = send_email(to=["you@example.com"], subject="Hello", body_text="Hi")

    assert out["success"] is True
    smtp_ctor.assert_called_once()
    smtp.starttls.assert_called_once()
    smtp.login.assert_called_once_with("me@example.com", "pw")
    smtp.send_message.assert_called_once()


def test_send_email_uses_smtp_ssl_when_port_465_and_starttls_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import send_email

    monkeypatch.setenv("DEFAULT_EMAIL_PASSWORD", "pw")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_PORT", "465")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_USERNAME", "me@example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_PASSWORD_ENV_VAR", "DEFAULT_EMAIL_PASSWORD")
    monkeypatch.delenv("ABSTRACT_EMAIL_SMTP_STARTTLS", raising=False)

    smtp_ssl = MagicMock()
    with patch("smtplib.SMTP_SSL", return_value=smtp_ssl) as smtp_ssl_ctor:
        with patch("smtplib.SMTP") as smtp_ctor:
            out = send_email(to="you@example.com", subject="Hello", body_text="Hi")

    assert out["success"] is True
    smtp_ctor.assert_not_called()
    smtp_ssl_ctor.assert_called_once()
    smtp_ssl.login.assert_called_once_with("me@example.com", "pw")
    smtp_ssl.send_message.assert_called_once()


def test_send_email_accepts_literal_password_when_env_ref_is_not_identifier(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import send_email

    # A non-identifier "env var name" is treated as a literal secret when no env var exists with that name.
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_PORT", "465")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_USERNAME", "me@example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_PASSWORD_ENV_VAR", "literal-secret-with-dash")
    monkeypatch.delenv("literal-secret-with-dash", raising=False)

    smtp_ssl = MagicMock()
    with patch("smtplib.SMTP_SSL", return_value=smtp_ssl):
        out = send_email(to="you@example.com", subject="Hello", body_text="Hi")

    assert out["success"] is True
    smtp_ssl.login.assert_called_once_with("me@example.com", "literal-secret-with-dash")
    smtp_ssl.send_message.assert_called_once()


def test_list_emails_uses_imap_env_config_and_parses_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import list_emails

    monkeypatch.setenv("DEFAULT_EMAIL_PASSWORD", "pw")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_HOST", "imap.example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_PORT", "993")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_USERNAME", "me@example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_PASSWORD_ENV_VAR", "DEFAULT_EMAIL_PASSWORD")

    header = EmailMessage()
    header["Subject"] = "=?utf-8?b?VGVzdCDwn5iA?="  # "Test 😀"
    header["From"] = "Alice <alice@example.com>"
    header["To"] = "me@example.com"
    header["Date"] = "Sun, 11 Jan 2026 00:00:00 +0000"
    header["Message-ID"] = "<m1@example.com>"

    class FakeImap:
        def __init__(self) -> None:
            self.sock = MagicMock()
            self.login = MagicMock(return_value=("OK", [b""]))
            self.select = MagicMock(return_value=("OK", [b""]))
            self.logout = MagicMock(return_value=None)

        def uid(self, command: str, *_: object) -> tuple[str, list[object]]:
            if command.lower() == "search":
                return ("OK", [b"1"])
            if command.lower() == "fetch":
                meta = b'1 (FLAGS (\\\\Seen) RFC822.SIZE 123 BODY[HEADER.FIELDS (FROM TO CC BCC SUBJECT DATE MESSAGE-ID)] {0}'
                return ("OK", [(meta, header.as_bytes())])
            raise AssertionError(f"Unexpected command: {command}")

    fake = FakeImap()
    with patch("imaplib.IMAP4_SSL", return_value=fake) as ctor:
        out = list_emails(since="7d", status="all", limit=5)

    assert out["success"] is True
    ctor.assert_called_once_with("imap.example.com", 993)
    fake.login.assert_called_once_with("me@example.com", "pw")
    msgs = out.get("messages")
    assert isinstance(msgs, list) and len(msgs) == 1
    assert msgs[0]["uid"] == "1"
    assert msgs[0]["seen"] is True
    assert msgs[0]["subject"] == "Test 😀"


def test_read_email_uses_imap_env_config_and_extracts_body(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import read_email

    monkeypatch.setenv("DEFAULT_EMAIL_PASSWORD", "pw")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_HOST", "imap.example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_PORT", "993")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_USERNAME", "me@example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_IMAP_PASSWORD_ENV_VAR", "DEFAULT_EMAIL_PASSWORD")

    msg = EmailMessage()
    msg["Subject"] = "Hello"
    msg["From"] = "Alice <alice@example.com>"
    msg["To"] = "me@example.com"
    msg.set_content("Plain body")

    class FakeImap:
        def __init__(self) -> None:
            self.sock = MagicMock()
            self.login = MagicMock(return_value=("OK", [b""]))
            self.select = MagicMock(return_value=("OK", [b""]))
            self.logout = MagicMock(return_value=None)

        def uid(self, command: str, *_: object) -> tuple[str, list[object]]:
            if command.lower() == "fetch":
                meta = b"1 (FLAGS (\\\\Seen) BODY[] {0}"
                return ("OK", [(meta, msg.as_bytes())])
            raise AssertionError(f"Unexpected command: {command}")

    fake = FakeImap()
    with patch("imaplib.IMAP4_SSL", return_value=fake):
        out = read_email(uid="1")

    assert out["success"] is True
    fake.login.assert_called_once_with("me@example.com", "pw")
    assert out["subject"] == "Hello"
    assert "Plain body" in out.get("body_text", "")


def test_multi_account_yaml_requires_explicit_account(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from abstractcore.tools.comms_tools import list_email_accounts, list_emails

    cfg_path = tmp_path / "emails.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            accounts:
              a:
                imap:
                  host: imap.example.com
                  port: 993
                  username: a@example.com
                  password_env_var: A_PW
                smtp:
                  host: smtp.example.com
                  port: 587
                  username: a@example.com
                  password_env_var: A_PW
                  use_starttls: true
              b:
                imap:
                  host: imap.example.com
                  port: 993
                  username: b@example.com
                  password_env_var: B_PW
            """
        ).lstrip(),
        encoding="utf-8",
    )

    monkeypatch.setenv("ABSTRACT_EMAIL_ACCOUNTS_CONFIG", str(cfg_path))

    out = list_email_accounts()
    assert out["success"] is True
    assert out["source"] == "yaml"

    out2 = list_emails(since="1d", status="all", limit=5)
    assert out2["success"] is False
    assert "Multiple email accounts configured" in str(out2.get("error") or "")


def test_send_whatsapp_message_twilio_prefixes_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import send_whatsapp_message

    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok")

    class Resp:
        ok = True
        status_code = 201

        def json(self) -> dict:
            return {"sid": "SM123", "status": "sent", "to": "whatsapp:+1", "from": "whatsapp:+2"}

    with patch("requests.post", return_value=Resp()) as post:
        out = send_whatsapp_message(to="+1", from_number="+2", body="hi")

    assert out["success"] is True
    assert out["sid"] == "SM123"
    _, kwargs = post.call_args
    assert kwargs["auth"] == ("AC123", "tok")
    data = kwargs["data"]
    assert ("To", "whatsapp:+1") in data or data.get("To") == "whatsapp:+1"


def test_list_whatsapp_messages_filters_direction(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import list_whatsapp_messages

    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok")

    class Resp:
        ok = True
        status_code = 200

        def json(self) -> dict:
            return {
                "messages": [
                    {"sid": "SM1", "direction": "inbound", "body": "in"},
                    {"sid": "SM2", "direction": "outbound-api", "body": "out"},
                ]
            }

    with patch("requests.get", return_value=Resp()):
        out = list_whatsapp_messages(direction="inbound", limit=10)

    assert out["success"] is True
    msgs = out["messages"]
    assert [m["sid"] for m in msgs] == ["SM1"]


def test_read_whatsapp_message_returns_body(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import read_whatsapp_message

    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "AC123")
    monkeypatch.setenv("TWILIO_AUTH_TOKEN", "tok")

    class Resp:
        ok = True
        status_code = 200

        def json(self) -> dict:
            return {"sid": "SM1", "body": "Hello", "direction": "inbound"}

    with patch("requests.get", return_value=Resp()):
        out = read_whatsapp_message("SM1")

    assert out["success"] is True
    assert out["sid"] == "SM1"
    assert out["body"] == "Hello"
