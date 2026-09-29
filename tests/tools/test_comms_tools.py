"""WhatsApp tools (Twilio). The email tools are covered by tests/email/ against hermetic
IMAP / SMTP servers (backlog 0992)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


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
