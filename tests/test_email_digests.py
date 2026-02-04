from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_render_email_digest_text_is_deterministic() -> None:
    from abstractcore.comms.email_digests import render_email_digest_text

    body = render_email_digest_text(
        title="Daily Digest",
        intro="Hi",
        sections=[{"title": "Decisions", "items": ["Approve A", "Defer B"]}],
        footer="Bye",
        max_items_per_section=50,
    )
    assert body == "Daily Digest\n\nHi\n\n## Decisions\n- Approve A\n- Defer B\n\nBye\n"


def test_send_email_digest_uses_send_email_defaults(monkeypatch) -> None:
    from abstractcore.comms.email_digests import send_email_digest

    monkeypatch.setenv("DEFAULT_EMAIL_PASSWORD", "pw")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_PORT", "587")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_STARTTLS", "1")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_USERNAME", "me@example.com")
    monkeypatch.setenv("ABSTRACT_EMAIL_SMTP_PASSWORD_ENV_VAR", "DEFAULT_EMAIL_PASSWORD")

    smtp = MagicMock()
    with patch("smtplib.SMTP", return_value=smtp) as smtp_ctor:
        out = send_email_digest(
            to="you@example.com",
            subject="Digest",
            title="Daily Digest",
            sections=[{"title": "Inbox", "items": ["Item 1"]}],
        )

    assert out["success"] is True
    smtp_ctor.assert_called_once()
    smtp.send_message.assert_called_once()
