"""Verified TLS on every email connection (framework backlog 0992 WP0).

Hermetic: local IMAP/SMTP servers on random 127.0.0.1 ports and a throwaway test CA
generated per session with the openssl CLI (its server certificate names `DNS:localhost`
only, so connecting to `127.0.0.1` exercises the host-name check). No real mailbox, no
network beyond loopback, no key committed.

Each "refused" case is the mutation check: pass no `ssl_context=` / `context=` (the
stdlib default verifies nothing on CPython 3.12) and the connection succeeds, the
server receives the password, and the test goes red.
"""

from __future__ import annotations

import base64
import json
import shutil
import socket
import ssl
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

PASSWORD_ENV = "WP0_TEST_MAIL_PASSWORD"
PASSWORD = "wp0-test-password"


def make_test_certs(out: Path, openssl: str = "openssl") -> Dict[str, Path]:
    """Generate a throwaway test CA and a server certificate for DNS:localhost only.

    Built at test time with the openssl CLI (no key is committed; `*.pem` is gitignored).
    The certificate names no IP address, so connecting to 127.0.0.1 exercises the host-name check.
    """
    exe = shutil.which(openssl)
    if exe is None:
        pytest.fail("the TLS tests need the openssl command line tool on PATH")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ca.cnf").write_text(
        "[req]\ndistinguished_name=dn\nprompt=no\n[dn]\nCN=AbstractFramework throwaway TEST CA\n"
        "[v3_ca]\nbasicConstraints=critical,CA:TRUE,pathlen:0\nkeyUsage=critical,keyCertSign,cRLSign\n"
        "subjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid:always\n",
        encoding="utf-8",
    )
    (out / "leaf.cnf").write_text(
        "[req]\ndistinguished_name=dn\nprompt=no\n[dn]\nCN=localhost\n"
        "[v3_leaf]\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\nsubjectAltName=DNS:localhost\n"
        "subjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid:always\n",
        encoding="utf-8",
    )

    def run(*args: str) -> None:
        subprocess.run([exe, *args], cwd=out, check=True, capture_output=True)

    run("genrsa", "-out", "ca.key", "2048")
    run("req", "-x509", "-new", "-key", "ca.key", "-sha256", "-days", "2", "-config", "ca.cnf",
        "-extensions", "v3_ca", "-out", "ca.pem")
    run("genrsa", "-out", "server.key", "2048")
    run("req", "-new", "-key", "server.key", "-config", "leaf.cnf", "-out", "server.csr")
    run("x509", "-req", "-in", "server.csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-CAcreateserial",
        "-sha256", "-days", "2", "-extfile", "leaf.cnf", "-extensions", "v3_leaf", "-out", "server.pem")
    return {"ca": out / "ca.pem", "cert": out / "server.pem", "key": out / "server.key"}


@pytest.fixture(scope="module")
def tls(tmp_path_factory) -> Dict[str, Path]:
    return make_test_certs(tmp_path_factory.mktemp("wp0-tls"))


def _server_context(tls: Dict[str, Path]) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(tls["cert"], tls["key"])
    return ctx


class _FakeMailServer:
    """One-connection-at-a-time IMAP (implicit TLS) or SMTP (implicit TLS / STARTTLS) server."""

    def __init__(self, kind: str, tls: Dict[str, Path]) -> None:
        assert kind in {"imap", "smtps", "smtp-starttls"}
        self.kind = kind
        self.logins: List[str] = []  # the password each successful LOGIN/AUTH carried
        self.messages: List[bytes] = []
        self.handshake_errors: List[str] = []
        self._ctx = _server_context(tls)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(5)
        self._sock.settimeout(0.2)
        self.port = int(self._sock.getsockname()[1])
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._sock.close()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            conn.settimeout(5)
            try:
                if self.kind == "imap":
                    self._imap(conn)
                else:
                    self._smtp(conn, implicit=self.kind == "smtps")
            except ssl.SSLError as e:
                self.handshake_errors.append(str(e))
            except (OSError, ValueError):
                pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _wrap(self, conn: socket.socket) -> ssl.SSLSocket:
        return self._ctx.wrap_socket(conn, server_side=True)

    # --- IMAP (implicit TLS) -----------------------------------------------------------
    def _imap(self, raw: socket.socket) -> None:
        conn = self._wrap(raw)
        f = conn.makefile("rwb")

        def send(line: str) -> None:
            f.write(line.encode() + b"\r\n")
            f.flush()

        send("* OK IMAP4rev1 test server ready")
        while True:
            line = f.readline()
            if not line:
                return
            parts = line.decode().rstrip("\r\n").split(" ")
            tag, cmd, args = parts[0], parts[1].upper() if len(parts) > 1 else "", parts[2:]
            if cmd == "CAPABILITY":
                send("* CAPABILITY IMAP4rev1 AUTH=PLAIN")
                send(f"{tag} OK CAPABILITY completed")
            elif cmd == "LOGIN":
                self.logins.append(args[1].strip('"') if len(args) > 1 else "")
                send(f"{tag} OK LOGIN completed")
            elif cmd in {"SELECT", "EXAMINE"}:
                send("* 0 EXISTS")
                send("* 0 RECENT")
                send("* OK [UIDVALIDITY 1] UIDs valid")
                send(f"{tag} OK [READ-ONLY] {cmd} completed")
            elif cmd == "UID" and args and args[0].upper() == "SEARCH":
                send("* SEARCH")
                send(f"{tag} OK SEARCH completed")
            elif cmd == "LOGOUT":
                send("* BYE logging out")
                send(f"{tag} OK LOGOUT completed")
                return
            else:
                send(f"{tag} BAD unsupported")

    # --- SMTP (implicit TLS or STARTTLS) ------------------------------------------------
    def _smtp(self, raw: socket.socket, *, implicit: bool) -> None:
        conn: socket.socket = self._wrap(raw) if implicit else raw
        tls = implicit
        f = conn.makefile("rwb")

        def send(line: str) -> None:
            f.write(line.encode() + b"\r\n")
            f.flush()

        send("220 localhost ESMTP test server")
        while True:
            line = f.readline()
            if not line:
                return
            text = line.decode().rstrip("\r\n")
            verb = text.split(" ", 1)[0].upper()
            if verb in {"EHLO", "HELO"}:
                if tls:
                    send("250-localhost")
                    send("250 AUTH PLAIN")
                else:
                    send("250-localhost")
                    send("250 STARTTLS")
            elif verb == "STARTTLS" and not tls:
                send("220 ready to start TLS")
                conn = self._wrap(conn)
                f = conn.makefile("rwb")
                tls = True
            elif verb == "AUTH" and tls:
                blob = text.split(" ")[2] if len(text.split(" ")) > 2 else ""
                self.logins.append(base64.b64decode(blob).split(b"\0")[-1].decode())
                send("235 authentication succeeded")
            elif verb in {"MAIL", "RCPT", "RSET", "NOOP"}:
                send("250 ok")
            elif verb == "DATA":
                send("354 end with <CRLF>.<CRLF>")
                data = b""
                while True:
                    chunk = f.readline()
                    if not chunk or chunk == b".\r\n":
                        break
                    data += chunk
                self.messages.append(data)
                send("250 queued")
            elif verb == "QUIT":
                send("221 bye")
                return
            else:
                send("502 not implemented")


@pytest.fixture
def mail_server(tls):
    servers: List[_FakeMailServer] = []

    def start(kind: str) -> _FakeMailServer:
        s = _FakeMailServer(kind, tls)
        servers.append(s)
        return s

    yield start
    for s in servers:
        s.close()


def _write_accounts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, imap: Optional[Dict[str, Any]] = None,
                    smtp: Optional[Dict[str, Any]] = None) -> None:
    account: Dict[str, Any] = {}
    if imap is not None:
        account["imap"] = {"username": "me@example.invalid", "password_env_var": PASSWORD_ENV, **imap}
    if smtp is not None:
        account["smtp"] = {"username": "me@example.invalid", "password_env_var": PASSWORD_ENV, **smtp}
    path = tmp_path / "emails.json"
    path.write_text(json.dumps({"accounts": {"test": account}}), encoding="utf-8")
    monkeypatch.setenv("ABSTRACT_EMAIL_ACCOUNTS_CONFIG", str(path))
    monkeypatch.setenv(PASSWORD_ENV, PASSWORD)


# --- IMAP ---------------------------------------------------------------------------------


def test_imap_refuses_a_certificate_the_system_does_not_trust(mail_server, tls, tmp_path, monkeypatch) -> None:
    from abstractcore.tools.comms_tools import list_emails

    server = mail_server("imap")
    _write_accounts(tmp_path, monkeypatch, imap={"host": "localhost", "port": server.port})

    out = list_emails(limit=5, timeout_s=5)

    assert out["success"] is False, out
    assert "IMAP TLS certificate verification failed" in out["error"]
    assert "Fix:" in out["error"]
    assert server.logins == []  # the password never left the client


def test_imap_connects_when_the_test_ca_is_trusted_via_ca_file(mail_server, tls, tmp_path, monkeypatch) -> None:
    from abstractcore.tools.comms_tools import list_emails, read_email

    server = mail_server("imap")
    _write_accounts(tmp_path, monkeypatch, imap={"host": "localhost", "port": server.port, "ca_file": str(tls["ca"])})

    out = list_emails(limit=5, timeout_s=5)

    assert out["success"] is True, out
    assert out["counts"]["returned"] == 0
    assert server.logins == [PASSWORD]

    # read_email takes the same verified path (refused without the CA above; here it
    # reaches the server and asks for a UID the empty mailbox does not have).
    out2 = read_email(uid="1", timeout_s=5)
    assert "certificate" not in str(out2.get("error") or "")
    assert server.logins == [PASSWORD, PASSWORD]


def test_imap_checks_the_host_name_even_with_the_ca_trusted(mail_server, tls, tmp_path, monkeypatch) -> None:
    from abstractcore.tools.comms_tools import list_emails

    server = mail_server("imap")
    # The certificate names DNS:localhost only; 127.0.0.1 is not on it.
    _write_accounts(tmp_path, monkeypatch, imap={"host": "127.0.0.1", "port": server.port, "ca_file": str(tls["ca"])})

    out = list_emails(limit=5, timeout_s=5)

    assert out["success"] is False, out
    assert "IMAP TLS certificate verification failed for 127.0.0.1" in out["error"]
    assert server.logins == []


def test_read_email_refuses_an_untrusted_certificate(mail_server, tls, tmp_path, monkeypatch) -> None:
    from abstractcore.tools.comms_tools import read_email

    server = mail_server("imap")
    _write_accounts(tmp_path, monkeypatch, imap={"host": "localhost", "port": server.port})

    out = read_email(uid="1", timeout_s=5)

    assert out["success"] is False, out
    assert "IMAP TLS certificate verification failed" in out["error"]
    assert server.logins == []


# --- SMTP ---------------------------------------------------------------------------------


@pytest.mark.parametrize("kind,starttls", [("smtps", False), ("smtp-starttls", True)])
def test_smtp_refuses_a_certificate_the_system_does_not_trust(mail_server, tls, tmp_path, monkeypatch, kind, starttls) -> None:
    from abstractcore.tools.comms_tools import send_email

    server = mail_server(kind)
    _write_accounts(tmp_path, monkeypatch, smtp={"host": "localhost", "port": server.port, "use_starttls": starttls})

    out = send_email(to="someone@example.invalid", subject="s", body_text="b", timeout_s=5)

    assert out["success"] is False, out
    assert "SMTP TLS certificate verification failed" in out["error"]
    assert server.logins == []
    assert server.messages == []


@pytest.mark.parametrize("kind,starttls", [("smtps", False), ("smtp-starttls", True)])
def test_smtp_sends_when_the_test_ca_is_trusted_via_ca_file(mail_server, tls, tmp_path, monkeypatch, kind, starttls) -> None:
    from abstractcore.tools.comms_tools import send_email

    server = mail_server(kind)
    _write_accounts(
        tmp_path,
        monkeypatch,
        smtp={"host": "localhost", "port": server.port, "use_starttls": starttls, "ca_file": str(tls["ca"])},
    )

    out = send_email(to="someone@example.invalid", subject="hello", body_text="body", timeout_s=5)

    assert out["success"] is True, out
    assert server.logins == [PASSWORD]
    assert len(server.messages) == 1 and b"Subject: hello" in server.messages[0]


@pytest.mark.parametrize("kind,starttls", [("smtps", False), ("smtp-starttls", True)])
def test_smtp_checks_the_host_name_even_with_the_ca_trusted(mail_server, tls, tmp_path, monkeypatch, kind, starttls) -> None:
    from abstractcore.tools.comms_tools import send_email

    server = mail_server(kind)
    _write_accounts(
        tmp_path,
        monkeypatch,
        smtp={"host": "127.0.0.1", "port": server.port, "use_starttls": starttls, "ca_file": str(tls["ca"])},
    )

    out = send_email(to="someone@example.invalid", subject="s", body_text="b", timeout_s=5)

    assert out["success"] is False, out
    assert "SMTP TLS certificate verification failed for 127.0.0.1" in out["error"]
    assert server.logins == []


def test_ca_file_that_does_not_exist_is_a_config_error(tmp_path, monkeypatch) -> None:
    from abstractcore.tools.comms_tools import list_email_accounts

    _write_accounts(tmp_path, monkeypatch, imap={"host": "localhost", "port": 993, "ca_file": str(tmp_path / "nope.pem")})
    out = list_email_accounts()
    assert out["success"] is False
    assert "imap.ca_file not found" in out["error"]
