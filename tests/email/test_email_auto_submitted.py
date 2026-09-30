"""Automatic mail is marked (RFC 3834 + framework marker) and summaries expose both (core 2.20.2).

The 0.7.0 end-to-end proof found an email automation that re-triggered itself: the "email me
the result" notice went from the user's account to the user's inbox with no marker, and the
trigger admitted it again. Core now:

- sends `Auto-Submitted` and `X-AbstractFramework-Automation` for a message that asks for them,
  and stamps both on every message sent through a context with `automation_marker` set
  (`auto-replied` for an answer to one message, else `auto-generated`);
- exposes both on message summaries as typed fields (`auto_submitted`, `framework_marker`),
  so a mail watcher can refuse them without parsing headers itself.
"""

from __future__ import annotations

import email
import email.policy
from dataclasses import replace

import pytest

from abstractcore.comms.email import (
    AUTOMATION_MARKER_HEADER,
    EmailAccountStore,
    EmailClient,
    EmailInvalidMessage,
    EmailSecret,
    OutgoingMessage,
    auto_submitted_value,
    automation_marker_value,
    guarded_send,
)

from email_fixtures import ME, PASSWORD, account_for

pytestmark = pytest.mark.basic


def build_message(*, sender: str, to: str, subject: str, text: str, headers=None) -> bytes:
    from email.message import EmailMessage

    m = EmailMessage()
    m["From"] = sender
    m["To"] = to
    m["Subject"] = subject
    m["Message-ID"] = f"<{subject}@example.test>"
    for k, v in (headers or {}).items():
        m[k] = v
    m.set_content(text)
    return m.as_bytes()


def _sent(smtp):
    return email.message_from_bytes(smtp.messages[-1]["data"], policy=email.policy.default)


def test_a_person_s_message_carries_no_automation_headers(smtp, ca) -> None:
    c = EmailClient(account_for(None, smtp, ca), EmailSecret(PASSWORD), timeout=10)
    c.send(OutgoingMessage(to=(ME,), subject="hello", text="x"))
    got = _sent(smtp)
    assert got["Auto-Submitted"] is None and got[AUTOMATION_MARKER_HEADER] is None


def test_a_message_that_asks_is_sent_with_both_headers(smtp, ca) -> None:
    c = EmailClient(account_for(None, smtp, ca), EmailSecret(PASSWORD), timeout=10)
    c.send(OutgoingMessage(to=(ME,), subject="s", text="x", auto_submitted="auto-generated", automation_marker="notification:k1"))
    got = _sent(smtp)
    assert got["Auto-Submitted"] == "auto-generated"
    assert got[AUTOMATION_MARKER_HEADER] == "notification:k1"


@pytest.mark.parametrize("bad", [{"auto_submitted": "no"}, {"auto_submitted": "yes please"}, {"automation_marker": "aé"}, {"automation_marker": "x" * 201}])
def test_invalid_automation_values_are_refused(smtp, ca, bad) -> None:
    c = EmailClient(account_for(None, smtp, ca), EmailSecret(PASSWORD), timeout=10)
    with pytest.raises(EmailInvalidMessage):
        c.send(OutgoingMessage(to=(ME,), subject="s", text="x", **bad))
    assert smtp.messages == []


def test_a_context_with_a_marker_stamps_every_send(imap, smtp, ca, config_file) -> None:
    store = EmailAccountStore(config_file)
    store.connect(account_for(imap, smtp, ca), EmailSecret(PASSWORD))
    seen = []
    ctx = replace(store.context(), automation_marker="occurrence:run-1", on_sent=seen.append)
    guarded_send(ctx, OutgoingMessage(to=(ME,), subject="result", text="x"))
    got = _sent(smtp)
    assert got["Auto-Submitted"] == "auto-generated" and got[AUTOMATION_MARKER_HEADER] == "occurrence:run-1"
    assert seen[-1]["automation_marker"] == "occurrence:run-1" and seen[-1]["auto_submitted"] == "auto-generated"
    assert seen[-1]["message_id"] == got["Message-ID"]
    # An answer to one message is auto-replied (RFC 3834 5.2).
    guarded_send(ctx, OutgoingMessage(to=(ME,), subject="Re: x", text="x", in_reply_to="<orig@example.test>"))
    assert _sent(smtp)["Auto-Submitted"] == "auto-replied"
    # Without the marker the context sends like before (a person's send).
    guarded_send(store.context(), OutgoingMessage(to=(ME,), subject="plain", text="x"))
    assert _sent(smtp)["Auto-Submitted"] is None and _sent(smtp)[AUTOMATION_MARKER_HEADER] is None


def test_summaries_and_details_expose_the_typed_fields(imap, smtp, ca) -> None:
    c = EmailClient(account_for(imap, smtp, ca), EmailSecret(PASSWORD), timeout=10)
    marked = c.build_mime(OutgoingMessage(to=(ME,), subject="marked", text="x", auto_submitted="auto-generated", automation_marker="occurrence:r9"))
    imap.add_message("INBOX", marked.as_bytes())
    imap.add_message("INBOX", build_message(sender="Bot <bot@example.test>", to=ME, subject="vacation", text="away",
                                            headers={"Auto-Submitted": 'auto-replied; owner-email="x@example.test"'}))
    imap.add_message("INBOX", build_message(sender="Ann <ann@example.test>", to=ME, subject="person", text="hi",
                                            headers={"Auto-Submitted": "no"}))
    imap.add_message("INBOX", build_message(sender="Bob <bob@example.test>", to=ME, subject="plain", text="hi"))
    got = {m.subject: m for m in c.search()["messages"]}
    assert (got["marked"].auto_submitted, got["marked"].framework_marker) == ("auto-generated", "occurrence:r9")
    assert (got["vacation"].auto_submitted, got["vacation"].framework_marker) == ("auto-replied", "")
    assert got["person"].auto_submitted == "no"
    assert (got["plain"].auto_submitted, got["plain"].framework_marker) == (None, "")
    detail = c.get(got["marked"].uid).to_dict()
    assert detail["auto_submitted"] == "auto-generated" and detail["framework_marker"] == "occurrence:r9"
    # fetch_new (what the mail watcher reads) carries them too.
    base = c.fetch_new(None)
    imap.add_message("INBOX", marked.as_bytes())
    (new,) = c.fetch_new(base.cursor).messages
    assert new.framework_marker == "occurrence:r9" and new.auto_submitted == "auto-generated"


def test_header_value_parsers() -> None:
    assert auto_submitted_value("") is None
    assert auto_submitted_value("  Auto-Generated ") == "auto-generated"
    assert auto_submitted_value("auto-replied; owner-email=\"a@b\"") == "auto-replied"
    assert auto_submitted_value("auto-notified") == "auto-notified"  # an extension keyword stays itself
    assert auto_submitted_value("what is this") == "unknown"
    assert automation_marker_value(" occurrence:1 ") == "occurrence:1"
    assert automation_marker_value("bad\nline") == ""
