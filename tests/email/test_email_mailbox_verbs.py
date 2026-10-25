"""Mailbox verbs of `EmailClient` (deterministic email handling in Flow): mark, label, move,
delete, the typed search query, thread, and reply / forward through the recipient policy.

Every change runs on a connection that allows exactly the commands the verbs send (a body FETCH
stays PEEK-only); a read-only account (IMAP without SMTP) refuses every change before connecting.
All against AbstractCore's fake IMAP/SMTP servers."""

from __future__ import annotations

from typing import Iterator

import pytest

from abstractcore.comms.email import EmailClient, EmailMailboxReadOnly, EmailSecret
from abstractcore.comms.email.errors import EmailInvalidMessage, EmailMailboxMissing
from abstractcore.testing.mailserver import FakeImapServer, build_message

from email_fixtures import ME, PASSWORD, account_for

pytestmark = pytest.mark.basic


def _server(ca, tokens, **kw) -> FakeImapServer:
    return FakeImapServer(ca, users={ME: PASSWORD}, security="ssl", tokens=tokens, **kw)


@pytest.fixture
def box(ca, tokens) -> Iterator[FakeImapServer]:
    server = _server(ca, tokens, folders=("INBOX", "Sent", "Archive", "Bin"), trash="Bin")
    yield server
    server.close()


def client(imap, smtp, ca) -> EmailClient:
    return EmailClient(account_for(imap, smtp, ca), EmailSecret(PASSWORD), timeout=10)


def _mail(subject: str, *, from_: str = "alice@example.test", text: str = "hello", message_id: str = "", references: str = "") -> bytes:
    return build_message(from_=from_, to=ME, subject=subject, text=text, message_id=message_id, references=references)


def test_mark_seen_and_flagged_in_batch(box, smtp, ca) -> None:
    c = client(box, smtp, ca)
    a, b = box.add_message("INBOX", _mail("a")), box.add_message("INBOX", _mail("b"))
    out = c.mark([a, b], seen=True, flagged=True)
    assert out == {"folder": "INBOX", "uids": [a, b], "seen": True, "flagged": True}
    assert {"\\Seen", "\\Flagged"} <= box.flags_of("INBOX", a) and {"\\Seen", "\\Flagged"} <= box.flags_of("INBOX", b)
    c.mark(b, seen=False)
    assert box.flags_of("INBOX", b) == {"\\Flagged"}
    with pytest.raises(EmailInvalidMessage):
        c.mark([a])


def test_keyword_labels_and_their_search(box, smtp, ca) -> None:
    c = client(box, smtp, ca)
    a, b = box.add_message("INBOX", _mail("invoice")), box.add_message("INBOX", _mail("other"))
    out = c.label([a], add=["Invoices", "to-review"])
    assert out["mode"] == "keywords" and {"Invoices", "to-review"} <= box.flags_of("INBOX", a)
    c.label(a, remove=["to-review"])
    assert "to-review" not in box.flags_of("INBOX", a)
    assert [m.uid for m in c.search(labels=["Invoices"])["messages"]] == [a]
    with pytest.raises(EmailInvalidMessage):
        c.label([b], add=["two words"])  # not an IMAP keyword
    with pytest.raises(EmailInvalidMessage):
        c.label([b], add=["\\Seen"])  # never a system flag


def test_gmail_labels_when_the_server_advertises_them(ca, tokens, smtp) -> None:
    gmail = _server(ca, tokens, gmail=True)
    try:
        c = client(gmail, smtp, ca)
        u = gmail.add_message("INBOX", _mail("x"))
        out = c.label([u], add=["Clients/ACME", "Needs reply"])
        assert out["mode"] == "gmail_labels" and gmail.labels_of("INBOX", u) == {"Clients/ACME", "Needs reply"}
        assert gmail.flags_of("INBOX", u) == set()  # labels, not keywords
        assert [m.uid for m in c.search(labels=["Needs reply"])["messages"]] == [u]
    finally:
        gmail.close()


def test_move_with_move_and_without(box, smtp, ca, tokens) -> None:
    c = client(box, smtp, ca)
    u = box.add_message("INBOX", _mail("keep"))
    assert c.move([u], "Archive")["method"] == "move"
    assert box.uids_in("INBOX") == [] and len(box.uids_in("Archive")) == 1
    old = _server(ca, tokens, folders=("INBOX", "Archive"), move=False)
    try:
        c2 = client(old, smtp, ca)
        keep, go = old.add_message("INBOX", _mail("stays")), old.add_message("INBOX", _mail("goes"))
        old.flags_of("INBOX", keep)
        assert c2.move([go], "Archive")["method"] == "copy_expunge"
        assert old.uids_in("INBOX") == [keep] and len(old.uids_in("Archive")) == 1
        assert "UID EXPUNGE" in old.commands  # only that message expunged (UIDPLUS)
    finally:
        old.close()


def test_delete_to_trash_then_permanent(box, smtp, ca, tokens) -> None:
    c = client(box, smtp, ca)
    u = box.add_message("INBOX", _mail("bye"))
    out = c.delete([u])
    assert (out["to"], out["permanent"]) == ("Bin", False) and box.uids_in("INBOX") == []  # special-use \Trash
    [in_bin] = box.uids_in("Bin")
    assert c.delete([in_bin], folder="Bin")["permanent"] is True and box.uids_in("Bin") == []  # already in the trash
    v = box.add_message("INBOX", _mail("now"))
    assert c.delete(v, permanent=True) == {"folder": "INBOX", "uids": [v], "to": None, "permanent": True}
    assert box.uids_in("INBOX") == []
    named = _server(ca, tokens, folders=("INBOX", "Deleted Items"))  # no special-use: a usual name
    bare = _server(ca, tokens, folders=("INBOX",))
    try:
        w = named.add_message("INBOX", _mail("x"))
        assert client(named, smtp, ca).delete([w])["to"] == "Deleted Items"
        z = bare.add_message("INBOX", _mail("y"))
        with pytest.raises(EmailMailboxMissing):
            client(bare, smtp, ca).delete([z])
        assert bare.uids_in("INBOX") == [z]
    finally:
        named.close()
        bare.close()


def test_a_read_only_account_refuses_every_change_before_connecting(box, ca) -> None:
    ro = EmailClient(account_for(box, None, ca), EmailSecret(PASSWORD), timeout=10)  # IMAP only
    u = box.add_message("INBOX", _mail("x"))
    logins = len(box.logins)
    assert ro.account.can_write is False
    for call in (lambda: ro.mark([u], seen=True), lambda: ro.label([u], add=["a"]), lambda: ro.move([u], "Archive"),
                 lambda: ro.delete([u]), lambda: ro.session(write=True).__enter__()):
        with pytest.raises(EmailMailboxReadOnly):
            call()
    assert len(box.logins) == logins and box.flags_of("INBOX", u) == set()
    assert [m.uid for m in ro.search()["messages"]] == [u]  # reading still works


def test_a_write_session_signs_in_once_for_many_changes(box, smtp, ca) -> None:
    c = client(box, smtp, ca)
    uids = [box.add_message("INBOX", _mail(f"m{i}")) for i in range(5)]
    logins = len(box.logins)
    with c.session(write=True):
        c.mark(uids, seen=True)
        c.label(uids[:2], add=["Done"])
        c.move(uids[2:], "Archive")
        assert len(c.search(folder="Archive")["messages"]) == 3
    assert len(box.logins) - logins == 1
    assert box.uids_in("INBOX") == uids[:2]


def test_typed_search_query_newest_first(box, smtp, ca) -> None:
    c = client(box, smtp, ca)
    a = box.add_message("INBOX", _mail("Report Q1", from_="bob@acme.test", text="numbers attached"))
    b = box.add_message("INBOX", _mail("Lunch", from_="carol@example.test", text="pizza"))
    d = box.add_message("INBOX", _mail("Report Q2", from_="bob@acme.test", text="more numbers"))
    c.mark([d], flagged=True)
    assert [m.uid for m in c.search(from_="acme.test")["messages"]] == [d, a]  # a domain; newest first
    assert [m.uid for m in c.search(from_="carol@example.test")["messages"]] == [b]
    assert [m.uid for m in c.search(subject="report", text="more")["messages"]] == [d]
    assert [m.uid for m in c.search(flagged=True)["messages"]] == [d]
    assert [m.uid for m in c.search(flagged=False, limit=1)["messages"]] == [b]


def test_thread_follows_references_both_ways(box, smtp, ca) -> None:
    c = client(box, smtp, ca)
    root = box.add_message("INBOX", _mail("Plan", message_id="<root@x.test>"))
    other = box.add_message("INBOX", _mail("Unrelated", message_id="<other@x.test>"))
    r1 = box.add_message("INBOX", _mail("Re: Plan", message_id="<r1@x.test>", references="<root@x.test>"))
    r2 = box.add_message("INBOX", _mail("Re: Plan", message_id="<r2@x.test>", references="<root@x.test> <r1@x.test>"))
    assert [m.uid for m in c.thread(r1)] == [root, r1, r2]
    assert [m.uid for m in c.thread(root)] == [root, r1, r2]
    assert [m.uid for m in c.thread(other)] == [other]


def test_reply_and_forward_go_through_the_recipient_policy(box, smtp, ca, tmp_path) -> None:
    from abstractcore.comms.email import EmailContext, EmailPolicyRefused, RecipientPolicy, SendLimits, SendRateLimiter

    acct = account_for(box, smtp, ca)
    ctx = EmailContext(account=acct, secret=EmailSecret(PASSWORD),
                       policy=RecipientPolicy.build("allowlist", always_allow=["example.test"]),
                       limits=SendLimits(), limiter=SendRateLimiter(tmp_path / "sends.json", SendLimits()))
    u = box.add_message("INBOX", build_message(from_="alice@example.test", to=ME, subject="Hi", text="original body",
                                               message_id="<o@example.test>", attachments=[("a.txt", "text/plain", b"ATTACHED")]))
    sent = ctx.reply(u, "thanks")
    assert sent.accepted == ("alice@example.test",)
    fwd = ctx.forward(u, "bob@example.test", note="FYI")
    assert fwd.accepted == ("bob@example.test",)
    data = smtp.messages[-1]["data"]
    assert b"Fwd: Hi" in data and b"FYI" in data and b"original body" in data and b"a.txt" in data
    with pytest.raises(EmailPolicyRefused):
        ctx.forward(u, "outsider@elsewhere.test")
    assert "\\Seen" not in box.flags_of("INBOX", u)  # reading to reply/forward never marks it read


def test_gmail_labels_are_read_with_the_message_and_keywords_stay_in_flags(ca, tokens, smtp, box) -> None:
    gmail = _server(ca, tokens, gmail=True)
    try:
        c = client(gmail, smtp, ca)
        u = gmail.add_message("INBOX", _mail("x"), flags={"\\Flagged"}, labels={"Work", "Needs reply"})
        detail = c.get(u)
        assert set(detail.summary.labels) == {"Work", "Needs reply"} and detail.to_dict()["labels"] in (["Needs reply", "Work"], ["Work", "Needs reply"])
        assert set(c.get_many([u])[u].summary.labels) == {"Work", "Needs reply"}
        assert set(c.search()["messages"][0].labels) == {"Work", "Needs reply"}
    finally:
        gmail.close()
    v = box.add_message("INBOX", _mail("y"), flags={"Invoices"})
    plain = client(box, smtp, ca).get(v)
    assert plain.summary.labels == () and "Invoices" in plain.summary.flags
    assert not any("X-GM-LABELS" in f for f in box.fetches)  # never asked of a non-Gmail server
