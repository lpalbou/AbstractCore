"""The mailbox access choice (`EmailAccount.access`): "read" (read only) or "write" (read and
write), an explicit choice never derived from SMTP once made. Accounts stored without it keep the
rule they followed: write when an outgoing server is set, else read.

All against AbstractCore's fake IMAP/SMTP servers and a scratch config file."""

from __future__ import annotations

from dataclasses import replace

import pytest

from abstractcore.comms.email import (
    EmailAccount,
    EmailClient,
    EmailMailboxReadOnly,
    EmailNotConfigured,
    EmailSecret,
    OutgoingMessage,
    SmtpSettings,
)
from abstractcore.comms.email.context import guarded_send
from abstractcore.comms.email.errors import EmailInvalidSettings
from abstractcore.comms.email.store import EmailAccountStore
from abstractcore.testing.mailserver import build_message

from email_fixtures import ME, PASSWORD, account_for

pytestmark = pytest.mark.basic


def test_the_default_access_keeps_what_stored_accounts_could_do(imap, smtp, ca) -> None:
    full, watched, send_only = account_for(imap, smtp, ca), account_for(imap, None, ca), account_for(None, smtp, ca)
    assert (full.access, full.can_write, full.can_send) == ("write", True, True)
    assert (watched.access, watched.can_write, watched.can_send) == ("read", False, False)
    assert (send_only.access, send_only.can_send) == ("write", True)
    # Stored before 2.29 (no "access" key): the same rule.
    legacy = full.to_dict()
    legacy.pop("access")
    assert EmailAccount.from_dict(legacy).access == "write"
    legacy_watched = watched.to_dict()
    legacy_watched.pop("access")
    assert EmailAccount.from_dict(legacy_watched).access == "read"


def test_the_choice_is_explicit_and_round_trips(imap, smtp, ca) -> None:
    ro = account_for(imap, smtp, ca, access="read")  # an outgoing server, yet read only
    assert (ro.can_write, ro.can_send, ro.access_text) == (False, False, "Read only")
    assert EmailAccount.from_dict(ro.to_dict()).access == "read" and ro.to_dict()["access"] == "read"
    rw_no_smtp = account_for(imap, None, ca, access="write")
    assert (rw_no_smtp.can_write, rw_no_smtp.can_send, rw_no_smtp.access_text) == (True, False, "Read and write")
    assert EmailAccount.from_dict(rw_no_smtp.to_dict()).access == "write"
    assert replace(ro, access="WRITE").access == "write"
    with pytest.raises(EmailInvalidSettings):
        account_for(imap, smtp, ca, access="admin")


def test_a_read_only_account_with_smtp_refuses_changes_and_sends_before_connecting(imap, smtp, ca, config_file) -> None:
    u = imap.add_message("INBOX", build_message(from_="alice@example.test", to=ME, subject="x", text="t"))
    ro = EmailClient(account_for(imap, smtp, ca, access="read"), EmailSecret(PASSWORD), timeout=10)
    logins = len(imap.logins)
    with pytest.raises(EmailMailboxReadOnly) as err:
        ro.mark([u], seen=True)
    assert err.value.code == "email_mailbox_read_only" and "read-only" in err.value.cause and "Read and write" in err.value.fix
    assert len(imap.logins) == logins and imap.flags_of("INBOX", u) == set()
    store = EmailAccountStore(config_file)
    store.connect(account_for(imap, smtp, ca, access="read"), EmailSecret(PASSWORD))
    with pytest.raises(EmailMailboxReadOnly):
        guarded_send(store.context(), OutgoingMessage(to=(ME,), subject="s", text="x"))
    assert smtp.messages == []
    # Read and write without an outgoing server: it can change the mailbox, not send.
    store.connect(account_for(imap, None, ca, access="write"), EmailSecret(PASSWORD))
    with pytest.raises(EmailNotConfigured):
        guarded_send(store.context(), OutgoingMessage(to=(ME,), subject="s", text="x"))


def test_set_access_flips_without_connecting_again(imap, smtp, ca, config_file) -> None:
    store = EmailAccountStore(config_file)
    store.connect(account_for(imap, None, ca), EmailSecret(PASSWORD))
    pub = store.public()
    assert (pub["access"], pub["access_text"], pub["can_write"], pub["can_send"]) == ("read", "Read only", False, False)
    leg = SmtpSettings.build("localhost", port=smtp.port, security=smtp.security, ca_file=str(ca.ca_pem))
    smtp_logins = len(smtp.logins)
    pub = store.set_access("write", smtp=leg)
    assert (pub["access"], pub["access_text"], pub["can_write"], pub["can_send"]) == ("write", "Read and write", True, True)
    assert pub["smtp"]["port"] == smtp.port and len(smtp.logins) == smtp_logins + 1  # the new leg was signed in to
    guarded_send(store.context(), OutgoingMessage(to=(ME,), subject="now", text="x"))
    assert len(smtp.messages) == 1
    # Read only again: the outgoing server is kept but unused.
    pub = store.set_access("read")
    assert (pub["access"], pub["can_send"], pub["smtp"]["port"]) == ("read", False, smtp.port)
    assert store.set_access("write")["can_send"] is True  # no new leg needed, nothing tested
    with pytest.raises(EmailInvalidSettings):
        store.set_access("everything")


def test_set_access_with_a_refused_outgoing_server_changes_nothing(imap, smtp, ca, config_file) -> None:
    store = EmailAccountStore(config_file)
    store.connect(account_for(imap, None, ca), EmailSecret(PASSWORD))
    bad = SmtpSettings.build("localhost", port=1, security="starttls", ca_file=str(ca.ca_pem))
    with pytest.raises(Exception) as err:
        store.set_access("write", smtp=bad)
    assert getattr(err.value, "code", "").startswith("email_")
    pub = EmailAccountStore(config_file).public()
    assert (pub["access"], pub["smtp"]) == ("read", None)
    # Not configured at all: a typed refusal.
    other = EmailAccountStore(config_file.parent / "empty" / "abstractcore.json")
    with pytest.raises(EmailNotConfigured):
        other.set_access("write")


def test_the_cli_status_says_the_access(imap, smtp, ca, config_file, capsys) -> None:
    from abstractcore.config.email_cli import handle_email

    EmailAccountStore(config_file).connect(account_for(imap, smtp, ca), EmailSecret(PASSWORD))
    assert handle_email(["status"]) == 0
    assert "  Access: Read and write" in capsys.readouterr().out
