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



def test_the_digest_module_renders_only_and_never_sends() -> None:
    # send_email_digest (a direct-send seam outside the tools and the host's approval gate)
    # was removed in 2.20 (backlog 0992): sending goes through send_email / guarded_send.
    import abstractcore.tools.email_digests as digests

    assert not hasattr(digests, "send_email_digest")
