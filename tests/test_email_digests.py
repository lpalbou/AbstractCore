from __future__ import annotations



def test_render_email_digest_text_is_deterministic() -> None:
    from abstractcore.tools.email_digests import render_email_digest_text

    body = render_email_digest_text(
        title="Daily Digest",
        intro="Hi",
        sections=[{"title": "Decisions", "items": ["Approve A", "Defer B"]}],
        footer="Bye",
        max_items_per_section=50,
    )
    assert body == "Daily Digest\n\nHi\n\n## Decisions\n- Approve A\n- Defer B\n\nBye\n"


# send_email_digest end to end (connected account, recipient policy, send limits):
# tests/email/test_email_store_cli_tools.py::test_send_email_digest_sends_from_the_account_through_the_policy
