"""`EmailClient.get_many` and `session()`: reading many messages on ONE connection.

A mail watcher reading a burst of new mail used `get(uid)` per message: one connection (login,
EXAMINE, two FETCHes) each, ~10 s per message on a remote server. `get_many` reads them on one
connection with one header/structure FETCH per 100 UIDs and one body FETCH per distinct text
section; `session()` lets `fetch_new` and `get_many` share that connection (one login)."""

from __future__ import annotations

import pytest

from abstractcore.comms.email import EmailClient
from abstractcore.comms.email.errors import EmailMessageNotFound
from abstractcore.testing.mailserver import build_message

from email_fixtures import ME, PASSWORD, account_for


def client(imap, smtp, ca, password=PASSWORD, **overrides) -> EmailClient:
    from abstractcore.comms.email import EmailSecret

    return EmailClient(account_for(imap, smtp, ca, **overrides), EmailSecret(password), timeout=10)


def _mail(i: int, *, html: bool = False, attach: bool = False) -> bytes:
    return build_message(from_="a@example.test", to=ME, subject=f"m{i}", text=f"body {i}", html=f"<p>body {i}</p>" if html else "",
                         attachments=[("a.txt", "text/plain", b"x" * 10)] if attach else ())


def test_fifty_new_messages_one_login_and_at_most_four_fetches(imap, smtp, ca) -> None:
    c = client(imap, smtp, ca)
    cursor = c.fetch_new(None).cursor  # baseline
    uids = [imap.add_message("INBOX", _mail(i)) for i in range(50)]
    logins, fetches = len(imap.logins), len(imap.fetches)
    with c.session():
        new = c.fetch_new(cursor)
        details = c.get_many([m.uid for m in new.messages])
    assert len(imap.logins) - logins == 1  # one connection for the listing AND the reading
    assert len(imap.fetches) - fetches <= 4, imap.fetches[fetches:]
    assert sorted(details) == uids and [details[u].summary.subject for u in uids] == [f"m{i}" for i in range(50)]
    for u in uids:
        assert "\\\\Seen" not in imap.flags_of("INBOX", u)  # PEEK only


def test_each_detail_is_exactly_what_get_reads(imap, smtp, ca) -> None:
    c = client(imap, smtp, ca)
    uids = [imap.add_message("INBOX", _mail(0)), imap.add_message("INBOX", _mail(1, html=True)),
            imap.add_message("INBOX", _mail(2, html=True, attach=True))]
    many = c.get_many(uids)
    for u in uids:
        assert many[u] == c.get(u)


def test_a_missing_uid_is_a_typed_value_and_big_bodies_are_skipped(imap, smtp, ca) -> None:
    c = client(imap, smtp, ca, )
    small = imap.add_message("INBOX", _mail(0))
    big = imap.add_message("INBOX", build_message(from_="a@example.test", to=ME, subject="big", text="y" * 5000))
    out = EmailClient(c.account, c._secret, timeout=10, max_message_bytes=1000).get_many([small, big, 999])
    assert isinstance(out[999], EmailMessageNotFound)
    assert out[small].text.strip() == "body 0" and out[small].skipped is None
    assert out[big].text == "" and out[big].skipped["code"] == "email_message_too_large"
    assert c.get_many([]) == {}


def test_a_session_reuses_one_connection_for_every_read(imap, smtp, ca) -> None:
    c = client(imap, smtp, ca)
    u = imap.add_message("INBOX", _mail(0))
    logins = len(imap.logins)
    with c.session():
        c.get(u)
        c.get(u)
        c.get_many([u])
    assert len(imap.logins) - logins == 1
    c.get(u)  # outside: its own connection again
    assert len(imap.logins) - logins == 2


def test_an_error_of_the_callers_own_inside_a_session_is_not_a_mail_error(imap, smtp, ca) -> None:
    c = client(imap, smtp, ca)
    with pytest.raises(OSError, match="disk full"):
        with c.session():
            raise OSError("disk full")
