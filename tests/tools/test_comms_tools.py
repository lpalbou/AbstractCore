from __future__ import annotations

from email.message import EmailMessage
from unittest.mock import MagicMock, patch

import pytest


def test_send_email_requires_password_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import send_email

    monkeypatch.delenv("EMAIL_PASSWORD", raising=False)
    out = send_email(
        smtp_host="smtp.example.com",
        username="me@example.com",
        password_env_var="EMAIL_PASSWORD",
        to="you@example.com",
        subject="Hello",
        body_text="Hi",
    )
    assert out["success"] is False
    assert "Missing env var" in str(out.get("error") or "")


def test_send_email_uses_starttls_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import send_email

    monkeypatch.setenv("EMAIL_PASSWORD", "pw")

    smtp = MagicMock()
    with patch("smtplib.SMTP", return_value=smtp) as smtp_ctor:
        out = send_email(
            smtp_host="smtp.example.com",
            smtp_port=587,
            username="me@example.com",
            password_env_var="EMAIL_PASSWORD",
            to=["you@example.com"],
            subject="Hello",
            body_text="Hi",
        )

    assert out["success"] is True
    smtp_ctor.assert_called_once()
    smtp.starttls.assert_called_once()
    smtp.login.assert_called_once_with("me@example.com", "pw")
    smtp.send_message.assert_called_once()


def test_list_emails_parses_headers_and_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import list_emails

    monkeypatch.setenv("EMAIL_PASSWORD", "pw")

    header = EmailMessage()
    header["Subject"] = "=?utf-8?b?VGVzdCDwn5iA?="  # "Test 😀"
    header["From"] = "Alice <alice@example.com>"
    header["To"] = "me@example.com"
    header["Date"] = "Sun, 11 Jan 2026 00:00:00 +0000"
    header["Message-ID"] = "<m1@example.com>"

    class FakeImap:
        def __init__(self) -> None:
            self.sock = MagicMock()

        def login(self, *_: object, **__: object) -> tuple[str, list[bytes]]:
            return ("OK", [b""])

        def select(self, *_: object, **__: object) -> tuple[str, list[bytes]]:
            return ("OK", [b""])

        def uid(self, command: str, *_: object) -> tuple[str, list[object]]:
            if command.lower() == "search":
                return ("OK", [b"1"])
            if command.lower() == "fetch":
                meta = b'1 (FLAGS (\\\\Seen) RFC822.SIZE 123 BODY[HEADER.FIELDS (FROM TO CC BCC SUBJECT DATE MESSAGE-ID)] {0}'
                return ("OK", [(meta, header.as_bytes())])
            raise AssertionError(f"Unexpected command: {command}")

        def logout(self) -> None:
            return None

    with patch("imaplib.IMAP4_SSL", return_value=FakeImap()):
        out = list_emails(imap_host="imap.example.com", username="me@example.com", since="7d", status="all", limit=5)

    assert out["success"] is True
    msgs = out.get("messages")
    assert isinstance(msgs, list) and len(msgs) == 1
    assert msgs[0]["uid"] == "1"
    assert msgs[0]["seen"] is True
    assert msgs[0]["subject"] == "Test 😀"


def test_read_email_extracts_text_body(monkeypatch: pytest.MonkeyPatch) -> None:
    from abstractcore.tools.comms_tools import read_email

    monkeypatch.setenv("EMAIL_PASSWORD", "pw")

    msg = EmailMessage()
    msg["Subject"] = "Hello"
    msg["From"] = "Alice <alice@example.com>"
    msg["To"] = "me@example.com"
    msg.set_content("Plain body")

    class FakeImap:
        def __init__(self) -> None:
            self.sock = MagicMock()

        def login(self, *_: object, **__: object) -> tuple[str, list[bytes]]:
            return ("OK", [b""])

        def select(self, *_: object, **__: object) -> tuple[str, list[bytes]]:
            return ("OK", [b""])

        def uid(self, command: str, *_: object) -> tuple[str, list[object]]:
            if command.lower() == "fetch":
                meta = b"1 (FLAGS (\\\\Seen) BODY[] {0}"
                return ("OK", [(meta, msg.as_bytes())])
            raise AssertionError(f"Unexpected command: {command}")

        def logout(self) -> None:
            return None

    with patch("imaplib.IMAP4_SSL", return_value=FakeImap()):
        out = read_email(imap_host="imap.example.com", username="me@example.com", uid="1")

    assert out["success"] is True
    assert out["subject"] == "Hello"
    assert "Plain body" in out.get("body_text", "")


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
