"""LIVE email tests against a real test mailbox (opt-in; framework backlog 0992).

Skipped unless every AF_TEST_EMAIL_* variable is set (the harness input; product code takes the
account through AbstractCore settings only) AND pytest runs with --allow-network. Run:

    set -a; . <your mailbox env file>; set +a; \
    python -m pytest tests/email/live -m live_email --allow-network -s --durations=0

What is proved against the real server: verified TLS with the default trust store (and a
negative control that must fail), connect + test through `EmailAccountStore` (the secret never
in its files), capabilities / folders / UIDVALIDITY, STARTTLS variants when offered, a
uniquely-tagged message sent to the account's OWN address and delivered through `fetch_new`,
read (headers, body, attachment metadata), search by the subject nonce, reply to self with
threading headers, the mailbox read-only (\\Seen unchanged by our reads), the recipient policy
refusing a non-self address before any SMTP call, and the send limits refusing before SMTP.

Mail only ever goes to the test account's own address (a guard refuses anything else before
MAIL FROM). Messages are tagged "[af-live-test]" and stay small: a read-only mailbox cannot be
cleaned up.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import imaplib
import smtplib
import socket
import ssl
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from live_email_support import (  # noqa: E402
    ARRIVAL_TIMEOUT_S,
    FOREIGN,
    fact,
    new_nonce,
    tagged_subject,
    wait_for,
)

pytestmark = [
    pytest.mark.live_email,
    pytest.mark.network("live email tests talk to the real test mailbox (IMAP + SMTP)"),
]

ATTACHMENT_NAME = "af-live-test.txt"


def _attachment_bytes(nonce: str) -> bytes:
    return f"AbstractFramework live email test attachment {nonce}\n".encode("ascii")


# --- TLS -----------------------------------------------------------------------------------


def _handshake(host: str, port: int, ctx: ssl.SSLContext) -> dict:
    with socket.create_connection((host, port), timeout=20) as raw:
        with ctx.wrap_socket(raw, server_hostname=host) as tls:
            cert = tls.getpeercert() or {}
            issuer = {k: v for part in cert.get("issuer", ()) for (k, v) in part}
            not_after = cert.get("notAfter")
            days_left = None
            if not_after:
                days_left = int((ssl.cert_time_to_seconds(not_after) - time.time()) // 86400)
            return {
                "verified": True,
                "tls_version": tls.version(),
                "cipher": (tls.cipher() or ("",))[0],
                "issuer_org": issuer.get("organizationName", ""),
                "issuer_cn": issuer.get("commonName", ""),
                "cert_days_left": days_left,
                "san_count": len(cert.get("subjectAltName", ())),
            }


def test_live_tls_certificates_verify_with_the_default_trust_store(live_mailbox):
    """Implicit TLS on the IMAP and SMTP ports, verified by ssl.create_default_context()."""
    paths = ssl.get_default_verify_paths()
    ctx = ssl.create_default_context()
    fact(live_mailbox, "trust_store", {"openssl": ssl.OPENSSL_VERSION, "cafile": paths.cafile, "capath": paths.capath,
                                       "ca_count": ctx.cert_store_stats().get("x509_ca")})
    assert live_mailbox.imap_security == "ssl" and live_mailbox.smtp_security == "ssl", "expected implicit TLS on both legs"
    results = {}
    for leg, host, port in (("imap", live_mailbox.imap_host, live_mailbox.imap_port),
                            ("smtp", live_mailbox.smtp_host, live_mailbox.smtp_port)):
        try:
            results[leg] = _handshake(host, port, ssl.create_default_context())
        except ssl.SSLCertVerificationError as exc:
            results[leg] = {"verified": False, "verify_code": exc.verify_code, "verify_message": exc.verify_message,
                            "reason": exc.reason}
        fact(live_mailbox, f"tls_{leg}", {"port": port, **results[leg]})
    assert results["imap"]["verified"], "the IMAP server certificate does not verify with the default trust store"
    assert results["smtp"]["verified"], "the SMTP server certificate does not verify with the default trust store"


def test_live_client_refuses_a_certificate_it_cannot_verify(live_mailbox):
    """Negative control: with an EMPTY trust store the client must fail TLS (typed), never log in."""
    from abstractcore.comms.email import EmailClient

    empty = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)  # verify_mode CERT_REQUIRED + check_hostname, no CAs
    result = EmailClient(live_mailbox.account(), live_mailbox.secret(), ssl_context=empty, timeout=20).test()
    codes = {leg: (result.get(leg) or {}).get("code") for leg in ("imap", "smtp")}
    fact(live_mailbox, "tls_negative_control", codes)
    assert codes == {"imap": "email_tls_failed", "smtp": "email_tls_failed"}


# --- the store -----------------------------------------------------------------------------


def test_live_connect_and_test_through_the_account_store(live_mailbox, tmp_path):
    from abstractcore.comms.email import EmailAccountStore

    store = EmailAccountStore(config_dir=tmp_path / "config")
    t0 = time.monotonic()
    public = store.connect(live_mailbox.account(), live_mailbox.secret())  # tests both legs first
    connect_s = time.monotonic() - t0
    t0 = time.monotonic()
    result = store.test()
    test_s = time.monotonic() - t0
    fact(live_mailbox, "store_connect", {"connect_s": round(connect_s, 2), "test_s": round(test_s, 2),
                                         "imap_ok": result["imap"].get("ok"), "smtp_ok": result["smtp"].get("ok"),
                                         "imap_messages": result["imap"].get("messages"),
                                         "auth_kind": public.get("auth_kind") or (public.get("account") or {}).get("auth_kind")})
    assert result["ok"] is True and result["imap"]["ok"] is True and result["smtp"]["ok"] is True
    # The secret is sealed: no store file carries the password, and public() never does.
    leaked = [p.name for p in (tmp_path / "config").rglob("*") if p.is_file() and live_mailbox.contains_secret(p.read_bytes())]
    assert leaked == [], "the password appears in a store file"
    assert not live_mailbox.contains_secret(repr(store.public())), "public() carries the password"
    # The stored settings are usable: the context's policy defaults to the own address only.
    ctx = store.context()
    assert ctx.policy.mode == "allowlist" and len(ctx.policy.entries) == 1 and live_mailbox.is_self(ctx.policy.entries[0])
    assert ctx.client().test_imap()["ok"] is True


# --- capabilities / folders ------------------------------------------------------------------


def test_live_capabilities_folders_and_uidvalidity(live_mailbox, tmp_path):
    ctx = live_mailbox.context(tmp_path)
    client = ctx.client()
    pre_auth = client.capabilities()
    with client._imap() as conn:  # a signed-in connection: CAPABILITY often grows after LOGIN
        typ, data = conn.capability()
    post_auth = sorted({tok for chunk in (data or []) if isinstance(chunk, bytes) for tok in chunk.decode().split()})
    folders = client.list_folders()
    state = client.folder_state("INBOX")
    caps = set(post_auth) | set(pre_auth)
    fact(live_mailbox, "imap_capabilities", {"pre_auth": pre_auth, "post_auth": post_auth})
    fact(live_mailbox, "imap_features", {"IDLE": "IDLE" in caps, "MOVE": "MOVE" in caps, "UIDPLUS": "UIDPLUS" in caps,
                                         "CONDSTORE": "CONDSTORE" in caps, "IMAP4rev2": "IMAP4rev2" in caps,
                                         "SPECIAL-USE": "SPECIAL-USE" in caps, "X-GM-EXT-1": "X-GM-EXT-1" in caps})
    fact(live_mailbox, "folders", {"count": len(folders), "names": [f["name"] for f in folders][:40],
                                   "delimiter": folders[0]["delimiter"] if folders else None})
    fact(live_mailbox, "inbox_state", {"uidvalidity": state.get("uidvalidity"), "uidnext_reported": state.get("uidnext") is not None,
                                       "exists": state.get("exists")})
    assert typ == "OK" and "IMAP4REV1" in {c.upper() for c in caps}
    assert any(f["name"].upper() == "INBOX" for f in folders)
    assert isinstance(state["uidvalidity"], int) and state["uidvalidity"] > 0


def test_live_starttls_ports_when_offered(live_mailbox, tmp_path):
    """Record whether the server also offers STARTTLS (IMAP 143, SMTP 587); sign in through it if so."""
    from abstractcore.comms.email import EmailClient

    observed = {}
    try:
        conn = imaplib.IMAP4(live_mailbox.imap_host, 143, timeout=5)
        try:
            caps = {str(c).upper() for c in (conn.capabilities or ())}
            observed["imap_143"] = {"reachable": True, "STARTTLS": "STARTTLS" in caps, "LOGINDISABLED": "LOGINDISABLED" in caps}
        finally:
            conn.shutdown()
    except (OSError, imaplib.IMAP4.error) as exc:
        observed["imap_143"] = {"reachable": False, "error": type(exc).__name__}
    try:
        s = smtplib.SMTP(live_mailbox.smtp_host, 587, timeout=5)
        try:
            s.ehlo()
            observed["smtp_587"] = {"reachable": True, "STARTTLS": bool(s.has_extn("starttls")),
                                    "AUTH_before_tls": bool(s.has_extn("auth"))}
        finally:
            s.close()
    except (OSError, smtplib.SMTPException) as exc:
        observed["smtp_587"] = {"reachable": False, "error": type(exc).__name__}
    if observed["imap_143"].get("STARTTLS") or observed["smtp_587"].get("STARTTLS"):
        account = live_mailbox.account(
            imap_security="starttls" if observed["imap_143"].get("STARTTLS") else None,
            imap_port=143 if observed["imap_143"].get("STARTTLS") else None,
            smtp_security="starttls" if observed["smtp_587"].get("STARTTLS") else None,
            smtp_port=587 if observed["smtp_587"].get("STARTTLS") else None,
        )
        result = EmailClient(account, live_mailbox.secret(), timeout=20).test()
        observed["starttls_signin"] = {leg: {"ok": (result.get(leg) or {}).get("ok"), "code": (result.get(leg) or {}).get("code")}
                                       for leg in ("imap", "smtp")}
    fact(live_mailbox, "starttls", observed)
    for leg, port in (("imap", 143), ("smtp", 587)):
        key = f"{leg}_{port}"
        if observed[key].get("STARTTLS"):
            assert observed["starttls_signin"][leg]["ok"] is True, f"STARTTLS sign-in failed on {leg} {port}"
    if not (observed["imap_143"].get("STARTTLS") or observed["smtp_587"].get("STARTTLS")):
        pytest.skip("the server offers no STARTTLS on 143/587 (implicit TLS only); recorded as a fact")


# --- send to self, deliver, read, search, reply ------------------------------------------------


@pytest.fixture
def delivered(live_mailbox, live_state, smtp_self_only, tmp_path_factory):
    """Send ONE tagged message to the account's own address, then fetch_new until it arrives.

    Computed on first use and shared by the tests of this run (the send is not repeated)."""
    if "delivered" in live_state:
        return live_state["delivered"]
    from abstractcore.comms.email import Attachment, OutgoingMessage

    ctx = live_mailbox.context(tmp_path_factory.mktemp("live-send"))
    client = ctx.client()
    baseline = client.fetch_new(None)
    nonce = new_nonce()
    subject = tagged_subject("core", nonce)
    before = smtp_self_only.count
    t0 = time.monotonic()
    sent = ctx.send(OutgoingMessage(
        to=(live_mailbox.address,),
        subject=subject,
        text=f"AbstractFramework live email test (core). Nonce: {nonce}\nThis message is safe to ignore.\n",
        attachments=(Attachment(filename=ATTACHMENT_NAME, content_type="text/plain", data=_attachment_bytes(nonce)),),
    ))
    send_s = time.monotonic() - t0
    assert smtp_self_only.count == before + 1 and smtp_self_only.transactions[-1]["all_self"]
    state = {"cursor": baseline.cursor, "others": 0, "polls": []}

    def probe():
        res = client.fetch_new(state["cursor"])
        state["cursor"] = res.cursor
        state["polls"].append(len(res.messages))
        mine = [m for m in res.messages if nonce in (m.subject or "")]
        state["others"] += len(res.messages) - len(mine)
        return mine

    mine, elapsed, attempts = wait_for(probe)
    record = {
        "nonce": nonce,
        "subject": subject,
        "sent": sent,
        "send_s": send_s,
        "arrival_s": elapsed,
        "polls": attempts,
        "baseline": baseline,
        "found": list(mine or []),
        "cursor_after": state["cursor"],
        "other_messages_seen": state["others"],
    }
    live_state["delivered"] = record
    return record


def test_live_send_to_self_arrives_through_fetch_new(live_mailbox, delivered, tmp_path):
    fact(live_mailbox, "delivery", {"send_s": round(delivered["send_s"], 2), "arrival_s": round(delivered["arrival_s"], 1),
                                    "polls": delivered["polls"], "baseline_uidvalidity": delivered["baseline"].cursor.uidvalidity,
                                    "baseline_was_baseline": delivered["baseline"].baseline,
                                    "copies_found": len(delivered["found"]),
                                    "other_messages_seen": delivered["other_messages_seen"],
                                    "accepted_count": len(delivered["sent"].accepted), "refused": delivered["sent"].refused})
    assert delivered["baseline"].baseline and delivered["baseline"].messages == ()
    assert len(delivered["sent"].accepted) == 1 and live_mailbox.is_self(delivered["sent"].accepted[0])
    assert delivered["found"], f"the tagged message did not arrive within {ARRIVAL_TIMEOUT_S:.0f} s"
    assert len(delivered["found"]) == 1, "the tagged message arrived more than once in INBOX"
    [msg] = delivered["found"]
    assert msg.uidvalidity == delivered["baseline"].cursor.uidvalidity and msg.uid > delivered["baseline"].cursor.last_uid
    # The next fetch from the cursor never returns it again.
    again = live_mailbox.context(tmp_path).client().fetch_new(delivered["cursor_after"])
    assert all(m.uid != msg.uid for m in again.messages)


def test_live_read_headers_body_and_attachment_metadata(live_mailbox, delivered, tmp_path):
    [summary] = delivered["found"]
    client = live_mailbox.context(tmp_path).client()
    detail = client.get(summary.uid)
    att = detail.attachments[0] if detail.attachments else None
    fact(live_mailbox, "read", {"message_id_kept": detail.summary.message_id == delivered["sent"].message_id,
                                "has_date": bool(detail.summary.date), "internaldate": bool(detail.summary.internaldate),
                                "size": detail.summary.size, "text_len": len(detail.text), "html_len": len(detail.html),
                                "attachments": [{"filename": a.filename, "content_type": a.content_type, "size": a.size,
                                                 "disposition": a.disposition} for a in detail.attachments]})
    assert detail.summary.subject == delivered["subject"]
    assert live_mailbox.is_self(detail.summary.from_address)
    assert live_mailbox.is_self(detail.summary.to)
    assert delivered["nonce"] in detail.text and detail.html == ""
    assert detail.summary.message_id == delivered["sent"].message_id, "the server rewrote the Message-ID"
    assert att is not None and len(detail.attachments) == 1
    sent_bytes = _attachment_bytes(delivered["nonce"])
    assert (att.filename, att.content_type) == (ATTACHMENT_NAME, "text/plain")
    # `size` is the server-reported ENCODED size of the part (a transfer encoding never shrinks the
    # content); the decoded size is the download result's.
    assert att.size >= len(sent_bytes) > 0
    # Downloading the attachment (BODY.PEEK) returns the exact bytes.
    out = tmp_path / "att"
    out.mkdir()
    saved = client.download_attachment(summary.uid, 0, str(out))
    assert saved["size"] == len(sent_bytes)
    assert Path(saved["path"]).read_bytes() == sent_bytes


def test_live_search_by_subject_nonce(live_mailbox, delivered, tmp_path):
    from abstractcore.comms.email import SearchCriteria

    [summary] = delivered["found"]
    client = live_mailbox.context(tmp_path).client()
    t0 = time.monotonic()
    res = client.search(SearchCriteria.build(subject_contains=delivered["nonce"]), limit=10)
    search_s = time.monotonic() - t0
    uids = [m.uid for m in res["messages"]]
    by_from = client.search(SearchCriteria.build(from_address=live_mailbox.address, since=dt.date.today() - dt.timedelta(days=1)), limit=50)
    fact(live_mailbox, "search", {"search_s": round(search_s, 2), "candidates": res["candidates"], "matches": len(uids),
                                  "from_self_since_yesterday_found": summary.uid in [m.uid for m in by_from["messages"]]})
    assert summary.uid in uids
    assert all(delivered["nonce"] in m.subject for m in res["messages"])
    assert summary.uid in [m.uid for m in by_from["messages"]]


def test_live_reads_never_change_the_seen_flag(live_mailbox, delivered, tmp_path):
    """The mailbox is read-only: EXAMINE + BODY.PEEK; our reads leave \\Seen exactly as it was."""
    from abstractcore.comms.email import SearchCriteria

    [summary] = delivered["found"]
    client = live_mailbox.context(tmp_path).client()
    crit = SearchCriteria.build(subject_contains=delivered["nonce"])

    def flags_now():
        return next(m.flags for m in client.search(crit, limit=10)["messages"] if m.uid == summary.uid)

    before = flags_now()
    client.get(summary.uid)
    client.get(summary.uid, include_raw=True)
    out = tmp_path / "att"
    out.mkdir()
    client.download_attachment(summary.uid, 0, str(out))
    client.fetch_new(delivered["baseline"].cursor)
    after = flags_now()
    fact(live_mailbox, "flags", {"at_arrival": list(summary.flags), "before_reads": list(before), "after_reads": list(after)})
    assert ("\\Seen" in before) == ("\\Seen" in after), "a read changed the \\Seen flag"
    assert set(before) == set(after)


def test_live_reply_to_self_threads(live_mailbox, delivered, smtp_self_only, tmp_path):
    from abstractcore.comms.email import EmailInvalidMessage, OutgoingMessage
    from abstractcore.tools.comms_tools import reply_email, use_email_context

    [original] = delivered["found"]
    ctx = live_mailbox.context(tmp_path / "ctx")
    client = ctx.client()
    # build_reply never answers the account's own address (a reply goes to the other party):
    # on a message we sent to ourselves there is nobody else, so it is refused before SMTP.
    before = smtp_self_only.count
    with pytest.raises(EmailInvalidMessage):
        client.build_reply(original.uid, text="x")
    with use_email_context(ctx):
        tool_out = reply_email(uid=str(original.uid), body_text="x")
    assert tool_out.get("success") is False and tool_out.get("error_code") == "email_invalid_message"
    assert smtp_self_only.count == before
    # A threaded reply addressed to self explicitly (what a person replying to a note-to-self does).
    detail = client.get(original.uid)
    reply_subject = f"Re: {detail.summary.subject}"
    cursor = client.fetch_new(None).cursor
    sent = ctx.send(OutgoingMessage(
        to=(live_mailbox.address,),
        subject=reply_subject,
        text=f"Reply to the live test message {delivered['nonce']}.\n",
        in_reply_to=detail.summary.message_id,
        references=tuple(detail.references) + (detail.summary.message_id,),
    ))
    assert smtp_self_only.count == before + 1
    state = {"cursor": cursor}

    def probe():
        res = client.fetch_new(state["cursor"])
        state["cursor"] = res.cursor
        return [m for m in res.messages if m.subject == reply_subject]

    found, elapsed, attempts = wait_for(probe)
    assert found, f"the reply did not arrive within {ARRIVAL_TIMEOUT_S:.0f} s"
    got = client.get(found[0].uid)
    fact(live_mailbox, "reply", {"arrival_s": round(elapsed, 1), "polls": attempts, "tool_error": tool_out.get("error_code"),
                                 "in_reply_to_kept": got.in_reply_to == detail.summary.message_id,
                                 "references_kept": detail.summary.message_id in got.references,
                                 "message_id_kept": got.summary.message_id == sent.message_id})
    assert got.in_reply_to == detail.summary.message_id
    assert detail.summary.message_id in got.references
    assert live_mailbox.is_self(got.summary.from_address)


# --- policy and limits: refused before any SMTP call ---------------------------------------------


def test_live_recipient_policy_refuses_a_non_self_address_before_smtp(live_mailbox, smtp_self_only, tmp_path, monkeypatch):
    from abstractcore.comms.email import EmailClient, EmailPolicyRefused, OutgoingMessage
    from abstractcore.tools.comms_tools import send_email, use_email_context

    opened = []
    real_smtp = EmailClient._smtp
    monkeypatch.setattr(EmailClient, "_smtp", lambda self: (opened.append(1), real_smtp(self))[1])
    ctx = live_mailbox.context(tmp_path)
    nonce = new_nonce()
    for message in (
        OutgoingMessage(to=(FOREIGN,), subject=tagged_subject("policy", nonce), text="never sent"),
        OutgoingMessage(to=(live_mailbox.address,), cc=(FOREIGN,), subject=tagged_subject("policy", nonce), text="never sent"),
        OutgoingMessage(to=(live_mailbox.address,), bcc=(FOREIGN,), subject=tagged_subject("policy", nonce), text="never sent"),
    ):
        with pytest.raises(EmailPolicyRefused) as exc:
            ctx.send(message)
        refused = exc.value.details.get("refused") or []
        assert [r["address"] for r in refused] == [FOREIGN] and refused[0]["rule"] == "allowlist"
    with use_email_context(ctx):
        out = send_email(to=FOREIGN, subject=tagged_subject("policy", nonce), body_text="never sent")
    assert out.get("success") is False and out.get("error_code") == "email_policy_refused"
    fact(live_mailbox, "policy", {"smtp_connections_opened": len(opened), "smtp_transactions": smtp_self_only.count})
    assert opened == [] and smtp_self_only.count == 0  # no SMTP connection at all, let alone a send


def test_live_send_limits_refuse_before_smtp(live_mailbox, smtp_self_only, tmp_path, monkeypatch):
    from abstractcore.comms.email import EmailClient, EmailRateLimited, OutgoingMessage

    opened = []
    real_smtp = EmailClient._smtp
    monkeypatch.setattr(EmailClient, "_smtp", lambda self: (opened.append(1), real_smtp(self))[1])
    ctx = live_mailbox.context(tmp_path, per_hour=1, per_day=1)
    ctx.limiter.reserve()  # the hour's one message is already used (no mail sent for it)
    with pytest.raises(EmailRateLimited) as exc:
        ctx.send(OutgoingMessage(to=(live_mailbox.address,), subject=tagged_subject("limits", new_nonce()), text="never sent"))
    fact(live_mailbox, "limits", {"code": exc.value.code, "window": exc.value.details.get("window"),
                                  "retry_after_s": exc.value.details.get("retry_after_s")})
    assert exc.value.details.get("window") == "hour" and exc.value.details.get("limit") == 1
    assert opened == [] and smtp_self_only.count == 0


def test_live_wrong_password_is_a_typed_auth_failure(live_mailbox):
    """One failed sign-in per leg: the server's protocol code maps to email_auth_failed."""
    from abstractcore.comms.email import EmailClient, EmailSecret

    wrong = EmailSecret("af-live-test-wrong-password-" + new_nonce())
    result = EmailClient(live_mailbox.account(), wrong, timeout=20).test()
    legs = {leg: {"code": (result.get(leg) or {}).get("code"), "retryable": (result.get(leg) or {}).get("retryable")}
            for leg in ("imap", "smtp")}
    fact(live_mailbox, "wrong_password", legs)
    assert legs["imap"]["code"] == "email_auth_failed"
    assert legs["smtp"]["code"] == "email_auth_failed"


def test_live_tagged_messages_are_small(live_mailbox, delivered):
    """Housekeeping check: what we leave in a read-only mailbox stays tiny."""
    [summary] = delivered["found"]
    assert summary.size is not None and summary.size < 16 * 1024
