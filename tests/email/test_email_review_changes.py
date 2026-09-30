"""Adversarial-review changes to the email layer (backlog 0992, core 2.20).

- "Agent email tools" (default OFF): the local account reaches agents only when the user turned
  the toggle on (CLI, HTTP, store); a host's resolver / injected context is not affected.
- Reading by structure: BODYSTRUCTURE first, then only the text/html parts by section; the
  attachments are listed, never downloaded by a read; an attachment download fetches only its
  part; `max_message_bytes` is a typed skip (never a cut body).
- Summaries: has_attachments, reply_to, in_reply_to, typed Importance / X-Priority / Priority,
  List-Unsubscribe presence; search has_attachment; list_email_folders; limit capped with a
  typed error and pagination by next_cursor (nothing dropped silently).
- fetch_new after a UIDVALIDITY rebuild with nothing to resynchronise: a baseline cursor at
  the newest message (never UID 0).
"""

from __future__ import annotations

import json
from email.message import EmailMessage
from pathlib import Path

import pytest

from abstractcore.comms.email import (
    MAX_LIST_LIMIT,
    EmailAccountStore,
    EmailClient,
    EmailMessageTooLarge,
    EmailSecret,
    MailCursor,
    SearchCriteria,
)
from abstractcore.comms.email.bodystructure import attachment_parts, body_parts, parse_bodystructure
from abstractcore.comms.email.imap_codec import parse_segments
from abstractcore.testing.mailserver import FakeImapServer, build_message

from email_fixtures import ME, PASSWORD, account_for, seed_inbox

pytestmark = pytest.mark.basic


def client(imap, ca, **kw) -> EmailClient:
    return EmailClient(account_for(imap, None, ca), EmailSecret(PASSWORD), timeout=10, **kw)


def _tools():
    from abstractcore.tools import comms_tools

    return comms_tools


def _message(*, subject: str, headers=None, text="body", html="", attachments=(), inline_msg=None) -> bytes:
    m = EmailMessage()
    m["From"] = "Sender <sender@example.test>"
    m["To"] = ME
    m["Subject"] = subject
    m["Message-ID"] = f"<{subject.replace(' ', '-')}@example.test>"
    for k, v in (headers or {}).items():
        m[k] = v
    if html:
        m.set_content(text)
        m.add_alternative(html, subtype="html")
    else:
        m.set_content(text)
    for filename, ctype, data in attachments:
        maintype, subtype = ctype.split("/", 1)
        m.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
    if inline_msg is not None:
        m.add_attachment(inline_msg)
    return m.as_bytes()


# ------------------------------------------------------------------ Agent email tools


def _connect(config_file, imap, smtp, ca) -> EmailAccountStore:
    store = EmailAccountStore(config_file)
    store.connect(account_for(imap, smtp, ca), EmailSecret(PASSWORD))
    return store


def test_agent_email_tools_are_off_by_default_and_every_tool_says_how_to_turn_them_on(imap, smtp, ca, config_file) -> None:
    seed_inbox(imap)
    store = _connect(config_file, imap, smtp, ca)
    doc = store.public()
    assert doc["agent_tools"] == {"enabled": False, "active": False, "reason": "off (your choice; default)"}
    assert json.loads(config_file.read_text())["email"]["agent_tools"] is False
    logins_before = len(imap.logins)
    t = _tools()
    calls = {
        "list_email_accounts": lambda: t.list_email_accounts(),
        "list_email_folders": lambda: t.list_email_folders(),
        "list_emails": lambda: t.list_emails(),
        "search_emails": lambda: t.search_emails(from_domain="example.test"),
        "read_email": lambda: t.read_email(uid="1"),
        "get_email_attachment": lambda: t.get_email_attachment(uid="1", index=0, output_dir=str(config_file.parent)),
        "send_email": lambda: t.send_email(to=ME, subject="s", body_text="b"),
        "reply_email": lambda: t.reply_email(uid="1", body_text="b"),
    }
    for name, call in calls.items():
        out = call()
        assert out["success"] is False and out["error_code"] == "email_agent_tools_off", (name, out)
        assert "Agent email tools" in out["fix"] and "abstractcore email agent-tools on" in out["fix"], name
    assert len(imap.logins) == logins_before and smtp.messages == []  # nothing reached the servers

    store.set_agent_tools(True)
    assert store.public()["agent_tools"] == {"enabled": True, "active": True, "reason": ""}
    assert t.list_emails()["success"] and t.send_email(to=ME, subject="s", body_text="b")["success"]
    store.set_agent_tools(False)
    assert t.list_emails()["error_code"] == "email_agent_tools_off"


def test_agent_tools_toggle_without_a_usable_account_is_not_active(imap, smtp, ca, config_file) -> None:
    store = EmailAccountStore(config_file)
    doc = store.set_agent_tools(True)
    assert doc["agent_tools"] == {"enabled": True, "active": False, "reason": "no connected, turned-on email account"}
    assert _tools().list_emails()["error_code"] == "email_not_configured"  # the account comes first
    _connect(config_file, imap, smtp, ca)
    store.set_enabled(False)
    assert store.public()["agent_tools"]["active"] is False
    assert _tools().list_emails()["error_code"] == "email_disabled"
    store.set_enabled(True)
    assert store.public()["agent_tools"]["active"] is True


def test_a_host_context_is_not_governed_by_the_local_toggle(imap, smtp, ca, config_file) -> None:
    store = _connect(config_file, imap, smtp, ca)  # agent tools OFF locally
    t = _tools()
    with t.use_email_context(store.context()):
        assert t.send_email(to=ME, subject="host decides", body_text="b")["success"]
    t.set_email_account_resolver(lambda: store.context())
    assert t.list_emails()["success"]


def test_cli_agent_tools_on_off_and_status(imap, smtp, ca, config_file, capsys) -> None:
    from abstractcore.config.email_cli import handle_email

    _connect(config_file, imap, smtp, ca)
    assert handle_email(["status"]) == 0
    out = capsys.readouterr().out
    assert "Agent email tools: off — off (your choice; default)" in out
    assert f"Settings: this AbstractCore install ({config_file})" in out
    assert handle_email(["agent-tools", "on", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "agent_tools": {"enabled": True, "active": True, "reason": ""}}
    assert handle_email(["agent-tools", "off"]) == 0
    assert "Agent email tools: off" in capsys.readouterr().out
    assert EmailAccountStore(config_file).public()["agent_tools"]["enabled"] is False
    with pytest.raises(SystemExit):
        handle_email(["agent-tools", "maybe"])


def test_http_agent_tools_route(imap, smtp, ca, config_file, monkeypatch) -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from abstractcore.server import app as server_app
    from abstractcore.server.email_routes import router

    monkeypatch.setattr(server_app, "_server_auth_enabled", lambda: False)
    monkeypatch.setattr(server_app, "_server_allows_unauthenticated", lambda: True)
    app = FastAPI()
    app.include_router(router)
    http = TestClient(app)
    assert http.get("/acore/email").json()["agent_tools"]["enabled"] is False
    r = http.put("/acore/email/agent-tools", json={"enabled": True})
    assert r.status_code == 200 and r.json()["agent_tools"]["reason"] == "no connected, turned-on email account"
    _connect(config_file, imap, smtp, ca)
    assert http.get("/acore/email").json()["agent_tools"] == {"enabled": True, "active": True, "reason": ""}
    assert http.put("/acore/email/agent-tools", json={"enabled": False}).json()["agent_tools"]["enabled"] is False
    assert http.put("/acore/email/agent-tools", json={}).status_code == 422


# ------------------------------------------------------------------ BODYSTRUCTURE


def _bs(text: str):
    (tok,) = parse_segments([(text.encode(), None)])
    return parse_bodystructure(tok)


def test_bodystructure_sections_follow_rfc3501() -> None:
    # multipart/mixed ( multipart/alternative (text, html), pdf attachment, message/rfc822 inline )
    root = _bs(
        '((("TEXT" "PLAIN" ("CHARSET" "utf-8") NIL NIL "QUOTED-PRINTABLE" 10 1 NIL NIL NIL NIL)'
        '("TEXT" "HTML" ("CHARSET" "utf-8") NIL NIL "7BIT" 20 1 NIL NIL NIL NIL) "ALTERNATIVE" ("BOUNDARY" "b2") NIL NIL NIL)'
        '("APPLICATION" "PDF" NIL NIL NIL "BASE64" 30 NIL ("ATTACHMENT" ("FILENAME*" "utf-8\'\'na%C3%AFve%20r%C3%A9sum%C3%A9.pdf")) NIL NIL)'
        '("MESSAGE" "RFC822" NIL NIL NIL "7BIT" 40 (NIL NIL NIL NIL NIL NIL NIL NIL NIL NIL)'
        ' ("TEXT" "PLAIN" ("CHARSET" "us-ascii") NIL NIL "7BIT" 5 1 NIL NIL NIL NIL) 3 NIL NIL NIL NIL)'
        '("IMAGE" "PNG" ("NAME" "=?utf-8?B?w6kucG5n?=") "<cid1>" NIL "BASE64" 50 NIL ("INLINE" NIL) NIL NIL)'
        ' "MIXED" ("BOUNDARY" "b1") NIL NIL NIL)'
    )
    bodies = body_parts(root)
    assert [(p.section, p.content_type) for p in bodies] == [("1.1", "text/plain"), ("1.2", "text/html"), ("3.1", "text/plain")]
    atts = attachment_parts(root)
    assert [(p.section, p.filename, p.size) for p in atts] == [("2", "naïve résumé.pdf", 30), ("4", "é.png", 50)]
    assert atts[1].content_id == "<cid1>" and atts[1].disposition == "inline"


def test_a_single_part_message_has_its_body_at_section_1() -> None:
    root = _bs('("TEXT" "PLAIN" ("CHARSET" "utf-8") NIL NIL "7BIT" 12 1)')  # no extension data
    assert [(p.section, p.size) for p in body_parts(root)] == [("1", 12)] and attachment_parts(root) == []


def test_read_fetches_structure_then_only_the_text_parts(imap, ca) -> None:
    pdf = b"%PDF-1.4 " + b"x" * 50_000
    uid = imap.add_message("INBOX", _message(subject="with parts", text="plain body é", html="<p>html body</p>",
                                             attachments=[("report.pdf", "application/pdf", pdf)]))
    c = client(imap, ca)
    imap.fetches.clear()
    d = c.get(uid)
    assert d.text == "plain body é" and d.html == "<p>html body</p>" and d.skipped is None
    (att,) = d.attachments
    assert att.filename == "report.pdf" and att.encoding == "base64" and att.size > len(pdf)
    fetched = " ".join(imap.fetches)
    assert "BODYSTRUCTURE" in fetched and "BODY.PEEK[HEADER]" in fetched
    assert "BODY.PEEK[1.1]" in fetched and "BODY.PEEK[1.2]" in fetched
    assert "BODY.PEEK[]" not in fetched and "BODY.PEEK[2]" not in fetched  # never the whole message or the PDF
    assert imap.flags_of("INBOX", uid) == set()
    # The attachment download fetches that part only.
    imap.fetches.clear()
    saved = c.download_attachment(uid, 0, str(Path(ca.directory)))
    assert Path(saved["path"]).read_bytes() == pdf and saved["size"] == len(pdf)
    assert any("BODY.PEEK[2]" in f for f in imap.fetches) and not any("BODY.PEEK[]" in f for f in imap.fetches)


def test_attached_message_and_non_ascii_names(imap, ca, tmp_path) -> None:
    inner = EmailMessage()
    inner["From"] = "fwd@example.test"
    inner["Subject"] = "forwarded"
    inner.set_content("inner text")
    uid = imap.add_message("INBOX", _message(subject="fwd", text="outer text",
                                             attachments=[("naïve résumé.txt", "text/plain", "é".encode())], inline_msg=inner))
    d = client(imap, ca).get(uid)
    assert d.text == "outer text"
    names = [a.filename for a in d.attachments]
    assert names[0] == "naïve résumé.txt"
    assert [a.content_type for a in d.attachments][1] == "message/rfc822"  # an attached .eml is one attachment
    saved = client(imap, ca).download_attachment(uid, 1, str(tmp_path))
    assert b"inner text" in Path(saved["path"]).read_bytes()


def test_bodies_over_the_reading_limit_are_skipped_with_a_typed_record(imap, smtp, ca, config_file) -> None:
    uid = imap.add_message("INBOX", _message(subject="big", text="x" * 5000, attachments=[("a.bin", "application/octet-stream", b"1" * 10)]))
    small = imap.add_message("INBOX", _message(subject="small", text="tiny"))
    c = client(imap, ca, max_message_bytes=1000)
    imap.fetches.clear()
    d = c.get(uid)
    assert d.skipped["code"] == "email_message_too_large" and d.skipped["limit"] == 1000 and d.skipped["size"] > 5000
    assert d.text == "" and [a.filename for a in d.attachments] == ["a.bin"]
    doc = d.to_dict()
    assert doc["body_text"] is None and doc["body_html"] is None and doc["body_skipped"]["uid"] == uid
    assert not any("BODY.PEEK[1" in f for f in imap.fetches)  # the bodies were not fetched at all
    assert c.get(small).text == "tiny"
    assert c.get(uid, max_message_bytes=100_000).text == "x" * 5000  # a per-call limit
    with pytest.raises(EmailMessageTooLarge):
        c.download_attachment(uid, 0, str(config_file.parent.parent), max_message_bytes=5)
    with pytest.raises(EmailMessageTooLarge):
        c.get(uid, include_raw=True)
    # Through the tool: the typed record and a notice, never a cut body.
    store = EmailAccountStore(config_file)
    store.connect(account_for(imap, smtp, ca), EmailSecret(PASSWORD))
    from dataclasses import replace

    t = _tools()
    with t.use_email_context(replace(store.context(), max_message_bytes=1000)):
        out = t.read_email(uid=str(uid))
    assert out["success"] and out["body_text"] is None and out["body_skipped"]["code"] == "email_message_too_large"
    assert "over the 1000-byte reading limit" in out["notices"][0]


def test_an_unusable_bodystructure_falls_back_to_the_whole_message(ca, tokens, tmp_path) -> None:
    server = FakeImapServer(ca, users={ME: PASSWORD}, tokens=tokens, broken_bodystructure=True)
    try:
        uid = server.add_message("INBOX", _message(subject="legacy", text="still readable",
                                                   attachments=[("f.txt", "text/plain", b"file")]))
        c = client(server, ca)
        d = c.get(uid)
        assert d.text == "still readable" and [a.filename for a in d.attachments] == ["f.txt"]
        assert d.summary.has_attachments is True
        assert c.search()["messages"][0].has_attachments is None  # unknown from the summary fetch
        assert Path(c.download_attachment(uid, 0, str(tmp_path))["path"]).read_bytes() == b"file"
        assert client(server, ca, max_message_bytes=10).get(uid).skipped["code"] == "email_message_too_large"
    finally:
        server.close()


# ------------------------------------------------------------------ summaries


def test_summaries_carry_attachments_threading_priority_and_unsubscribe(imap, ca) -> None:
    imap.add_message("INBOX", _message(subject="plain"))
    imap.add_message("INBOX", _message(
        subject="flagged",
        headers={"Importance": "High", "X-Priority": "1 (Highest)", "Priority": "urgent", "Reply-To": "desk@example.test",
                 "In-Reply-To": "<orig@example.test>", "List-Unsubscribe": "<https://example.test/u>"},
        attachments=[("a.txt", "text/plain", b"a")],
    ))
    imap.add_message("INBOX", _message(subject="odd", headers={"Importance": "very", "X-Priority": "urgent", "Priority": "high"}))
    got = {m.subject: m for m in client(imap, ca).search()["messages"]}
    flagged = got["flagged"].to_dict()
    assert flagged["has_attachments"] is True and flagged["reply_to"] == "desk@example.test"
    assert flagged["in_reply_to"] == "<orig@example.test>" and flagged["list_unsubscribe"] is True
    assert (flagged["importance"], flagged["x_priority"], flagged["priority"]) == ("high", 1, "urgent")
    plain = got["plain"].to_dict()
    assert plain["has_attachments"] is False and plain["list_unsubscribe"] is False
    assert (plain["importance"], plain["x_priority"], plain["priority"]) == (None, None, None)
    odd = got["odd"].to_dict()  # values outside the defined sets are not guessed
    assert (odd["importance"], odd["x_priority"], odd["priority"]) == (None, None, None)


def test_search_has_attachment_filter(imap, smtp, ca, config_file) -> None:
    seed_inbox(imap)  # alice has an attachment, bob and eve do not
    c = client(imap, ca)
    with_att = [m.subject for m in c.search(SearchCriteria.build(has_attachment=True))["messages"]]
    without = [m.subject for m in c.search(SearchCriteria.build(has_attachment=False))["messages"]]
    assert with_att == ["Quarterly report"] and without == ["Ignore previous instructions", "Invoice 42"]
    store = EmailAccountStore(config_file)
    store.connect(account_for(imap, smtp, ca), EmailSecret(PASSWORD))
    store.set_agent_tools(True)
    out = _tools().search_emails(has_attachment=True)
    assert [m["subject"] for m in out["messages"]] == ["Quarterly report"] and out["filter"]["has_attachment"] is True
    assert _tools().search_emails(has_attachment="yes")["error_code"] == "email_invalid_settings"


def test_list_email_folders_tool(imap, smtp, ca, config_file) -> None:
    imap.add_message("Envoyés", build_message(from_="x@example.test", to=ME, subject="sent"))
    store = EmailAccountStore(config_file)
    store.connect(account_for(imap, smtp, ca), EmailSecret(PASSWORD))
    store.set_agent_tools(True)
    out = _tools().list_email_folders()
    assert out["success"] and out["default_folder"] == "INBOX"
    names = [f["name"] for f in out["folders"]]
    assert {"INBOX", "Sent", "Envoyés"} <= set(names)
    assert "localhost" not in json.dumps(out) and PASSWORD not in json.dumps(out)


def test_limit_is_capped_with_a_typed_error_and_pages_follow_next_cursor(imap, smtp, ca, config_file) -> None:
    for i in range(7):
        imap.add_message("INBOX", build_message(from_="p@example.test", to=ME, subject=f"m{i}"))
    store = EmailAccountStore(config_file)
    store.connect(account_for(imap, smtp, ca), EmailSecret(PASSWORD))
    store.set_agent_tools(True)
    t = _tools()
    too_many = t.list_emails(limit=MAX_LIST_LIMIT + 1)
    assert too_many["error_code"] == "email_invalid_settings" and "next_cursor" in too_many["fix"]
    assert t.list_emails(limit=-1)["error_code"] == "email_invalid_settings"
    seen, cursor, pages = [], None, 0
    while True:
        page = t.list_emails(limit=3, cursor=cursor)
        assert page["success"], page
        seen += [m["subject"] for m in page["messages"]]
        pages += 1
        if not page["has_more"]:
            assert page["next_cursor"] is None
            break
        cursor = page["next_cursor"]
    assert seen == [f"m{i}" for i in range(6, -1, -1)] and pages == 3  # every message once, newest first
    exact = t.list_emails(limit=7)
    assert exact["has_more"] is False and len(exact["messages"]) == 7
    assert t.list_emails(cursor="not-a-cursor")["error_code"] == "email_invalid_settings"
    first = t.list_emails(limit=3)
    imap.reset_uidvalidity("INBOX")
    stale = t.list_emails(limit=3, cursor=first["next_cursor"])
    assert stale["error_code"] == "email_invalid_settings" and "without cursor" in stale["fix"]


# ------------------------------------------------------------------ fetch_new after a rebuild


def test_fetch_new_after_a_rebuild_with_nothing_to_resync_is_a_baseline(imap, ca) -> None:
    seed_inbox(imap)
    c = client(imap, ca)
    base = c.fetch_new(None)
    imap.reset_uidvalidity("INBOX")
    # A cursor without a date (nothing to resynchronise by): a new baseline, never UID 0.
    undated = MailCursor(base.cursor.uidvalidity, base.cursor.last_uid, "INBOX", "")
    reset = c.fetch_new(undated)
    assert reset.reset is True and reset.baseline is True and reset.messages == ()
    state = c.folder_state("INBOX")
    assert reset.cursor.uidvalidity == state["uidvalidity"] and reset.cursor.last_uid == state["uidnext"] - 1 == 3
    # The next poll delivers only mail that arrived after the rebuild: the folder is not replayed.
    imap.add_message("INBOX", build_message(from_="n@example.test", to=ME, subject="after the rebuild"))
    nxt = c.fetch_new(reset.cursor)
    assert [m.subject for m in nxt.messages] == ["after the rebuild"] and not nxt.reset
    # A dated cursor whose day has no message in the rebuilt folder: also a baseline.
    imap.reset_uidvalidity("INBOX")
    future = MailCursor(nxt.cursor.uidvalidity, nxt.cursor.last_uid, "INBOX", "2099-01-01T00:00:00+00:00")
    again = c.fetch_new(future)
    assert again.reset and again.baseline and again.cursor.last_uid == 4
    assert again.cursor.last_internaldate == "2099-01-01T00:00:00+00:00"


# ------------------------------------------------------------------ consoles


def test_web_console_email_tab_names_its_account_and_offers_agent_email_tools() -> None:
    from abstractcore.console.web import _EMAIL_HTML, render_console_html

    page = render_console_html()
    assert 'data-acc="email-scope"' in _EMAIL_HTML and "the email settings of this AbstractCore install" in _EMAIL_HTML
    assert "gateway console (My account)" in _EMAIL_HTML  # where a gateway user's own settings live
    # "Agent email tools" is a switch labelled by the feature (DESIGN §2), applied at once.
    assert '<span class="af-switch__label">Agent email tools</span>' in _EMAIL_HTML and 'data-acc="email-agent-tools"' in _EMAIL_HTML
    assert 'role="switch"' in _EMAIL_HTML and "abstractcore email agent-tools on|off" in _EMAIL_HTML
    assert "`${base}/agent-tools`" in page and 'ctx.request("PUT", path, { enabled: next })' in page
    assert "d.config_file" in page  # the scope line names the core config file
