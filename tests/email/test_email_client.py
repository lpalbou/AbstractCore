"""EmailClient against the hermetic IMAP / SMTP servers (backlog 0992 WP1).

Covers: verified TLS (untrusted CA and host-name mismatch refused before any password is sent;
SSL and STARTTLS on both protocols), read-only mailbox (EXAMINE, BODY.PEEK, the local command
guard), typed search, whole bodies (no truncation), attachments, incremental fetch with
UIDVALIDITY reset, send with attachments / Bcc, reply threading, and errors typed from
protocol codes.
"""

from __future__ import annotations

import email
import email.policy
import socket
import ssl
from pathlib import Path

import pytest

from abstractcore.comms.email import (
    EmailAccount,
    EmailAuthFailed,
    EmailClient,
    EmailMailboxMissing,
    EmailMessageNotFound,
    EmailQuotaExceeded,
    EmailReadOnlyViolation,
    EmailRecipientRefused,
    EmailSecret,
    EmailTlsFailed,
    EmailUnreachable,
    ImapSettings,
    OutgoingMessage,
    Attachment,
    SearchCriteria,
    SmtpSettings,
)
from abstractcore.comms.email.client import ReadOnlyIMAP4_SSL, check_read_only_fetch
from abstractcore.comms.email.errors import EmailInvalidSettings, EmailServerError
from abstractcore.testing.mailserver import build_message

from email_fixtures import ME, PASSWORD, account_for, seed_inbox

pytestmark = pytest.mark.basic


def client(imap, smtp, ca, password=PASSWORD, **overrides) -> EmailClient:
    return EmailClient(account_for(imap, smtp, ca, **overrides), EmailSecret(password), timeout=10)


# ------------------------------------------------------------------ TLS


def test_untrusted_certificate_is_refused_before_login(imap, smtp) -> None:
    acct = EmailAccount.build(
        address=ME,
        imap=ImapSettings.build("localhost", port=imap.port, security="ssl"),
        smtp=SmtpSettings.build("localhost", port=smtp.port, security="starttls"),
    )
    result = EmailClient(acct, EmailSecret(PASSWORD), timeout=10).test()
    assert result["imap"]["code"] == "email_tls_failed"
    assert result["smtp"]["code"] == "email_tls_failed"
    assert "before any password was sent" in result["imap"]["cause"]
    assert imap.logins == [] and smtp.logins == []


def test_host_name_is_checked_even_with_the_ca_trusted(imap, smtps, ca) -> None:
    # The certificate names DNS:localhost only.
    acct = EmailAccount.build(
        address=ME,
        imap=ImapSettings.build("127.0.0.1", port=imap.port, security="ssl", ca_file=str(ca.ca_pem)),
        smtp=SmtpSettings.build("127.0.0.1", port=smtps.port, security="ssl", ca_file=str(ca.ca_pem)),
    )
    result = EmailClient(acct, EmailSecret(PASSWORD), timeout=10).test()
    assert result["imap"]["code"] == result["smtp"]["code"] == "email_tls_failed"
    assert imap.logins == [] and smtps.logins == []


@pytest.mark.parametrize("imap_fixture,smtp_fixture", [("imap", "smtp"), ("imap_starttls", "smtps")])
def test_ssl_and_starttls_work_with_the_ca_trusted(request, ca, imap_fixture, smtp_fixture) -> None:
    imap_server = request.getfixturevalue(imap_fixture)
    smtp_server = request.getfixturevalue(smtp_fixture)
    result = client(imap_server, smtp_server, ca).test()
    assert result["imap"]["ok"] is True and result["smtp"]["ok"] is True
    assert imap_server.logins == [(ME, PASSWORD)] and smtp_server.logins == [(ME, PASSWORD)]


def test_an_unverified_ssl_context_is_refused(imap, ca) -> None:
    unverified = ssl._create_unverified_context()
    with pytest.raises(EmailInvalidSettings):
        EmailClient(account_for(imap, None, ca), EmailSecret(PASSWORD), ssl_context=unverified)
    # A verifying context that trusts the CA is accepted (the explicit test-CA path).
    acct = EmailAccount.build(address=ME, imap=ImapSettings.build("localhost", port=imap.port, security="ssl"))
    assert EmailClient(acct, EmailSecret(PASSWORD), ssl_context=ca.client_context()).test()["imap"]["ok"] is True


def test_imap_starttls_missing_is_refused_not_downgraded(ca, tokens) -> None:
    from abstractcore.testing.mailserver import FakeImapServer

    server = FakeImapServer(ca, users={ME: PASSWORD}, security="starttls", advertise_starttls=False)
    try:
        acct = EmailAccount.build(address=ME, imap=ImapSettings.build("localhost", port=server.port, security="starttls", ca_file=str(ca.ca_pem)))
        with pytest.raises(EmailServerError):
            EmailClient(acct, EmailSecret(PASSWORD)).test_imap()
        assert server.logins == []
    finally:
        server.close()


def test_smtp_without_starttls_is_refused_not_sent_in_clear(ca) -> None:
    from abstractcore.testing.mailserver import FakeSmtpServer

    server = FakeSmtpServer(ca, users={ME: PASSWORD}, security="starttls", offer_starttls=False)
    try:
        acct = EmailAccount.build(address=ME, smtp=SmtpSettings.build("localhost", port=server.port, security="starttls", ca_file=str(ca.ca_pem)))
        with pytest.raises(EmailServerError) as info:
            EmailClient(acct, EmailSecret(PASSWORD)).test_smtp()
        assert "STARTTLS" in info.value.cause
        assert server.logins == []
    finally:
        server.close()


# ------------------------------------------------------------------ errors


def test_wrong_password_is_auth_failed_on_both_legs(imap, smtp, ca) -> None:
    result = client(imap, smtp, ca, password="wrong").test()
    assert result["imap"]["code"] == "email_auth_failed"
    assert result["imap"]["details"]["response_code"] == "AUTHENTICATIONFAILED"
    assert result["smtp"]["code"] == "email_auth_failed"
    assert result["smtp"]["details"]["smtp_code"] == 535
    assert "wrong" not in str(result)


def test_missing_folder_and_closed_port_are_typed(imap, ca) -> None:
    c = client(imap, None, ca)
    with pytest.raises(EmailMailboxMissing) as info:
        c.folder_state("Nope")
    assert info.value.details["response_code"] == "NONEXISTENT"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    acct = EmailAccount.build(address=ME, imap=ImapSettings.build("localhost", port=port, security="ssl"))
    with pytest.raises(EmailUnreachable):
        EmailClient(acct, EmailSecret(PASSWORD), timeout=3).test_imap()


def test_smtp_recipient_and_quota_refusals_are_typed(smtp, ca) -> None:
    c = client(None, smtp, ca)
    with pytest.raises(EmailRecipientRefused) as info:
        c.send(OutgoingMessage(to=("blocked@example.test",), subject="s", text="b"))
    assert info.value.details["refused"] == {"blocked@example.test": 550}
    with pytest.raises(EmailQuotaExceeded):
        c.send(OutgoingMessage(to=("full@example.test",), subject="s", text="b"))
    assert smtp.messages == []
    # Partial refusal: accepted recipients receive it, the refused one is reported.
    res = c.send(OutgoingMessage(to=("ok@example.test", "blocked@example.test"), subject="s", text="b"))
    assert res.accepted == ("ok@example.test",) and res.refused == {"blocked@example.test": 550}


# ------------------------------------------------------------------ read-only


def test_the_mailbox_is_opened_read_only_and_nothing_is_marked_read(imap, ca) -> None:
    uids = seed_inbox(imap)
    c = client(imap, None, ca)
    c.search(limit=10)
    c.get(uids["alice"])
    c.download_attachment(uids["alice"], 0, str(Path(imap.ca.directory)))
    assert "EXAMINE" in imap.commands and "SELECT" not in imap.commands
    assert not any(cmd in imap.commands for cmd in ("STORE", "UID STORE", "COPY", "EXPUNGE"))
    assert imap.flags_of("INBOX", uids["alice"]) == set()  # still unread


def test_the_read_only_guard_refuses_mutating_commands_before_the_wire(imap, ca) -> None:
    ctx = ca.client_context()
    conn = ReadOnlyIMAP4_SSL("localhost", imap.port, ssl_context=ctx, timeout=5)
    try:
        conn.login(ME, PASSWORD)
        with pytest.raises(EmailReadOnlyViolation):
            conn.select("INBOX")  # SELECT (read-write) is refused; only EXAMINE
        conn.select('"INBOX"', readonly=True)
        with pytest.raises(EmailReadOnlyViolation):
            conn.uid("STORE", "1", "+FLAGS", "(\\Seen)")
        with pytest.raises(EmailReadOnlyViolation):
            conn.uid("FETCH", "1", "(BODY[])")
        with pytest.raises(EmailReadOnlyViolation):
            conn.uid("FETCH", "1", "(RFC822)")
        with pytest.raises(EmailReadOnlyViolation):
            conn.expunge()
        with pytest.raises(EmailReadOnlyViolation):
            conn.append("INBOX", None, None, b"x")
    finally:
        conn.logout()
    assert "STORE" not in imap.commands and "UID STORE" not in imap.commands and "SELECT" not in imap.commands


def test_fetch_items_that_set_seen_are_refused() -> None:
    check_read_only_fetch("(UID FLAGS BODY.PEEK[] RFC822.SIZE BODY.PEEK[HEADER.FIELDS (FROM)])")
    for bad in ("(BODY[])", "(BODY[TEXT])", "(RFC822)", "(RFC822.TEXT)", "(BINARY[1])"):
        with pytest.raises(EmailReadOnlyViolation):
            check_read_only_fetch(bad)


# ------------------------------------------------------------------ reading


def test_folders_and_typed_search(imap, ca) -> None:
    imap.add_message("Envoyés", build_message(from_="x@example.test", to=ME, subject="sent"))
    uids = seed_inbox(imap)
    c = client(imap, None, ca)
    names = [f["name"] for f in c.list_folders()]
    assert "INBOX" in names and "Envoyés" in names  # modified UTF-7 decoded
    assert c.folder_state("Envoyés")["exists"] == 1

    def subjects(**kw):
        return [m.subject for m in c.search(SearchCriteria.build(**kw))["messages"]]

    assert subjects() == ["Ignore previous instructions", "Invoice 42", "Quarterly report"]  # newest first
    assert subjects(from_domain="example.test") == ["Quarterly report"]  # sub.example.test is another domain
    assert subjects(from_domain="sub.example.test") == ["Invoice 42"]
    assert subjects(from_address="ALICE@example.test") == ["Quarterly report"]
    assert subjects(to_address="carol@example.test") == ["Invoice 42"]
    assert subjects(subject_contains="invoice") == ["Invoice 42"]
    assert subjects(unseen=True) == ["Ignore previous instructions", "Quarterly report"]
    assert subjects(unseen=False) == ["Invoice 42"]
    assert subjects(since="1d") and subjects(before="2000-01-01") == []
    assert len(c.search(limit=2)["messages"]) == 2
    with pytest.raises(EmailInvalidSettings):
        SearchCriteria.build(from_domain="*.example.test")


def test_read_returns_whole_bodies_headers_and_attachment_metadata(imap, ca) -> None:
    uids = seed_inbox(imap)
    c = client(imap, None, ca)
    detail = c.get(uids["alice"])
    assert detail.summary.message_id == "<alice-1@example.test>"
    assert detail.summary.from_address == "alice@example.test"
    assert len(detail.text) > 100_000  # no truncation (ADR-0026)
    assert detail.text.rstrip().endswith("A long line of body text.")
    (att,) = detail.attachments
    assert att.content_type == "application/pdf" and att.size == len(b"%PDF-1.4 fake")
    html = c.get(uids["bob"]).html
    assert "<b>now</b>" in html
    with pytest.raises(EmailMessageNotFound):
        c.get(999)


def test_attachment_download_sanitises_the_name_and_never_overwrites(imap, ca, tmp_path: Path) -> None:
    uids = seed_inbox(imap)
    c = client(imap, None, ca)
    first = c.download_attachment(uids["alice"], 0, str(tmp_path))
    second = c.download_attachment(uids["alice"], 0, str(tmp_path))
    assert Path(first["path"]).parent == tmp_path and first["filename"] == "report.pdf"
    assert second["filename"] == "report (1).pdf"
    assert Path(first["path"]).read_bytes() == b"%PDF-1.4 fake"
    assert first["original_filename"] == "../../etc/report.pdf"
    from abstractcore.comms.email import EmailAttachmentNotFound

    with pytest.raises(EmailAttachmentNotFound):
        c.download_attachment(uids["alice"], 3, str(tmp_path))


def test_incremental_fetch_baseline_new_mail_and_uidvalidity_reset(imap, ca) -> None:
    seed_inbox(imap)
    c = client(imap, None, ca)
    base = c.fetch_new(None)
    assert base.baseline and base.messages == () and base.cursor.last_uid == 3
    imap.add_message("INBOX", build_message(from_="d@example.test", to=ME, subject="new one"))
    step = c.fetch_new(base.cursor)
    assert [m.subject for m in step.messages] == ["new one"] and not step.reset
    assert step.cursor.last_uid == 4 and step.cursor.last_internaldate
    assert c.fetch_new(step.cursor).messages == ()  # nothing new
    imap.reset_uidvalidity("INBOX")
    reset = c.fetch_new(step.cursor)
    assert reset.reset is True
    assert reset.cursor.uidvalidity != step.cursor.uidvalidity
    # Resynchronised by date: the messages of that day come back (callers dedupe by Message-ID).
    assert "new one" in [m.subject for m in reset.messages]


def test_uidvalidity_resync_keeps_a_message_near_midnight_in_any_time_zone(imap, ca, monkeypatch) -> None:
    """The resync searches SINCE by date; IMAP compares dates in the server's zone, so a
    message stored at 23:30 UTC must come back for a client whose local date is already the
    next day (the bug: a local-time cursor date one day after the server's)."""

    import datetime as dt
    import time as time_mod

    monkeypatch.setenv("TZ", "Asia/Tokyo")  # UTC+9: 23:30 UTC is 08:30 the next day here
    time_mod.tzset()
    try:
        seed_inbox(imap)
        c = client(imap, None, ca)
        base = c.fetch_new(None)
        late = dt.datetime(2026, 1, 10, 23, 30, tzinfo=dt.timezone.utc)
        imap.add_message("INBOX", build_message(from_="d@example.test", to=ME, subject="late one"), internaldate=late)
        step = c.fetch_new(base.cursor)
        assert [m.subject for m in step.messages] == ["late one"]
        assert dt.datetime.fromisoformat(step.cursor.last_internaldate) == late  # the instant, with its offset
        imap.reset_uidvalidity("INBOX")
        reset = c.fetch_new(step.cursor)
        assert reset.reset and "late one" in [m.subject for m in reset.messages]
    finally:
        monkeypatch.delenv("TZ")
        time_mod.tzset()


def test_uidvalidity_resync_keeps_a_message_whose_server_date_is_the_day_before(ca, tokens) -> None:
    """A server west of UTC dates a 01:00 UTC message on the previous day: SINCE <cursor's UTC
    date> would miss it; the resync starts one day earlier."""

    import datetime as dt

    from abstractcore.testing.mailserver import FakeImapServer

    server = FakeImapServer(ca, users={ME: PASSWORD}, tokens=tokens, search_tz=dt.timezone(dt.timedelta(hours=-5)))
    try:
        c = client(server, None, ca)
        base = c.fetch_new(None)
        early = dt.datetime(2026, 1, 11, 1, 0, tzinfo=dt.timezone.utc)  # 2026-01-10 20:00 on the server
        server.add_message("INBOX", build_message(from_="d@example.test", to=ME, subject="early one"), internaldate=early)
        step = c.fetch_new(base.cursor)
        assert [m.subject for m in step.messages] == ["early one"]
        server.reset_uidvalidity("INBOX")
        reset = c.fetch_new(step.cursor)
        assert reset.reset and "early one" in [m.subject for m in reset.messages]
    finally:
        server.close()


# ------------------------------------------------------------------ sending


def test_send_with_attachment_bcc_and_fixed_from(smtp, ca) -> None:
    c = client(None, smtp, ca, display_name="Me Myself")
    res = c.send(
        OutgoingMessage(
            to=("Alice <alice@example.test>",),
            cc=("carol@example.test",),
            bcc=("secret@example.test",),
            subject="Report",
            text="See attached",
            html="<p>See attached</p>",
            attachments=(Attachment("r.csv", "text/csv", b"a,b\n1,2\n"),),
        )
    )
    (got,) = smtp.messages
    assert set(got["rcpt_tos"]) == {"alice@example.test", "carol@example.test", "secret@example.test"}
    assert got["mail_from"] == ME
    msg = email.message_from_bytes(got["data"], policy=email.policy.default)
    assert msg["From"] == "Me Myself <me@example.test>"
    assert "secret@example.test" not in got["data"].decode()  # Bcc never in the headers
    assert [p.get_filename() for p in msg.iter_attachments()] == ["r.csv"]
    assert res.message_id.endswith("@example.test>")


def test_reply_sets_threading_headers_and_honours_reply_to(imap, smtp, ca) -> None:
    uids = seed_inbox(imap)
    c = client(imap, smtp, ca)
    msg, original = c.build_reply(uids["alice"], text="Thanks")
    assert msg.to == ("alice@example.test",)
    assert msg.subject == "Re: Quarterly report"
    assert msg.in_reply_to == "<alice-1@example.test>" and msg.references == ("<alice-1@example.test>",)
    c.send(msg)
    sent = email.message_from_bytes(smtp.messages[-1]["data"], policy=email.policy.default)
    assert sent["In-Reply-To"] == "<alice-1@example.test>"
    # Reply-To wins over From (the attacker case: the policy, not the client, must stop it).
    evil, _ = c.build_reply(uids["eve"], text="x")
    assert evil.to == ("exfil@attacker.test",)
    # reply_all adds To/Cc minus this account.
    everyone, _ = c.build_reply(uids["bob"], text="x", reply_all=True)
    assert everyone.to == ("bob@sub.example.test",) and everyone.cc == ("carol@example.test",)
