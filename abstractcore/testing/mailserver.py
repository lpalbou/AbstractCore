"""Hermetic mail servers for tests: a throwaway CA, IMAP, SMTP and an OAuth2 token endpoint.

Everything binds 127.0.0.1 on a free port, uses a CA generated per test session (no key is
committed), and holds its state in memory. Nothing reaches a real mail service.

    from abstractcore.testing.mailserver import TestCA, FakeImapServer, FakeSmtpServer, FakeOAuthServer

    ca = TestCA.create(tmp_path)                      # CA + server cert for DNS:localhost
    imap = FakeImapServer(ca, users={"me@example.test": "pw"}, security="ssl")
    imap.add_message("INBOX", raw_bytes, flags=())
    smtp = FakeSmtpServer(ca, users={"me@example.test": "pw"}, security="starttls")   # needs aiosmtpd
    ...
    imap.close(); smtp.close()

- IMAP: a minimal IMAP4rev1 server (CAPABILITY, STARTTLS, LOGIN, AUTHENTICATE PLAIN/XOAUTH2,
  LIST, EXAMINE, SELECT, STATUS, UID SEARCH / FETCH, FETCH, STORE, NOOP, LOGOUT). It records
  every command name (`commands`) and flags a message \\Seen on a non-PEEK body fetch, so a
  test can prove a client stayed read-only. `reset_uidvalidity(folder)` simulates a server
  rebuilding a folder.
- SMTP: aiosmtpd (a test dependency) with implicit TLS or STARTTLS, AUTH PLAIN/LOGIN/XOAUTH2,
  per-recipient refusal codes, and the received messages in `messages`.
- OAuth: an https token endpoint (refresh_token, authorization_code, device_code grants and
  device authorization) issuing access tokens the IMAP and SMTP fakes accept.

The server certificate names only `DNS:localhost`: connecting to `127.0.0.1` exercises the
client's host-name check.
"""

from __future__ import annotations

import base64
import datetime as _dt
import email
import email.policy
import http.server
import json
import secrets
import socket
import socketserver
import ssl
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------------------
# Throwaway CA
# ---------------------------------------------------------------------------------------


@dataclass
class TestCA:
    __test__ = False  # not a pytest class

    ca_pem: Path
    cert_pem: Path
    key_pem: Path
    directory: Path

    @classmethod
    def create(cls, directory: Path, *, dns_names: Iterable[str] = ("localhost",)) -> "TestCA":
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        now = _dt.datetime.now(_dt.timezone.utc)
        ca_key = ec.generate_private_key(ec.SECP256R1())
        ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "AbstractFramework throwaway TEST CA")])
        ca_cert = (
            x509.CertificateBuilder()
            .subject_name(ca_name)
            .issuer_name(ca_name)
            .public_key(ca_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _dt.timedelta(minutes=5))
            .not_valid_after(now + _dt.timedelta(days=2))
            .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
            .add_extension(
                x509.KeyUsage(
                    digital_signature=False, content_commitment=False, key_encipherment=False, data_encipherment=False,
                    key_agreement=False, key_cert_sign=True, crl_sign=True, encipher_only=False, decipher_only=False,
                ),
                critical=True,
            )
            .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        leaf_key = ec.generate_private_key(ec.SECP256R1())
        leaf = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "localhost")]))
            .issuer_name(ca_name)
            .public_key(leaf_key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _dt.timedelta(minutes=5))
            .not_valid_after(now + _dt.timedelta(days=2))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.SubjectAlternativeName([x509.DNSName(n) for n in dns_names]), critical=False)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
            .sign(ca_key, hashes.SHA256())
        )
        ca_pem = directory / "test-ca.pem"
        cert_pem = directory / "server.pem"
        key_pem = directory / "server.key"
        ca_pem.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
        cert_pem.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
        key_pem.write_bytes(
            leaf_key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
        )
        return cls(ca_pem=ca_pem, cert_pem=cert_pem, key_pem=key_pem, directory=directory)

    def server_context(self) -> ssl.SSLContext:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(str(self.cert_pem), str(self.key_pem))
        return ctx

    def client_context(self) -> ssl.SSLContext:
        """A VERIFYING client context that trusts this CA (what a private-CA user configures)."""

        ctx = ssl.create_default_context()
        ctx.load_verify_locations(cafile=str(self.ca_pem))
        return ctx


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# ---------------------------------------------------------------------------------------
# Shared token registry (OAuth fakes <-> mail fakes)
# ---------------------------------------------------------------------------------------


class TokenRegistry:
    """Access tokens the fake OAuth server issued, and who they belong to."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokens: Dict[str, Tuple[str, float]] = {}

    def issue(self, user: str, ttl_s: float = 3600.0) -> str:
        token = "at-" + secrets.token_urlsafe(18)
        with self._lock:
            self._tokens[token] = (user, time.time() + ttl_s)
        return token

    def check(self, user: str, token: str) -> bool:
        with self._lock:
            got = self._tokens.get(token)
        return bool(got) and got[0] == user and got[1] > time.time()

    def revoke_all(self) -> None:
        with self._lock:
            self._tokens.clear()


def parse_xoauth2(raw: bytes) -> Tuple[str, str]:
    """(user, token) from a decoded XOAUTH2 initial response."""

    text = raw.decode("utf-8", errors="replace")
    user = token = ""
    for part in text.split("\x01"):
        if part.startswith("user="):
            user = part[5:]
        elif part.startswith("auth="):
            value = part[5:]
            if value.lower().startswith("bearer "):
                token = value[7:]
    return user, token


# ---------------------------------------------------------------------------------------
# IMAP
# ---------------------------------------------------------------------------------------


@dataclass
class FakeMessage:
    uid: int
    raw: bytes
    flags: Set[str] = field(default_factory=set)
    internaldate: _dt.datetime = field(default_factory=lambda: _dt.datetime.now(_dt.timezone.utc))


@dataclass
class FakeFolder:
    name: str
    uidvalidity: int
    uidnext: int = 1
    messages: List[FakeMessage] = field(default_factory=list)


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _imap_internaldate(d: _dt.datetime) -> str:
    d = d.astimezone(_dt.timezone.utc)
    return f"{d.day:02d}-{_MONTHS[d.month - 1]}-{d.year:04d} {d.hour:02d}:{d.minute:02d}:{d.second:02d} +0000"


def _parse_search_date(text: str) -> _dt.date:
    day, mon, year = text.split("-")
    return _dt.date(int(year), _MONTHS.index(mon) + 1, int(day))


def _tokenize(line: str) -> List[str]:
    """IMAP command arguments: atoms, quoted strings (unquoted here), parenthesised groups kept raw."""

    out: List[str] = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == " ":
            i += 1
            continue
        if c == '"':
            j = i + 1
            buf = []
            while j < len(line) and line[j] != '"':
                if line[j] == "\\" and j + 1 < len(line):
                    buf.append(line[j + 1])
                    j += 2
                    continue
                buf.append(line[j])
                j += 1
            out.append("\x00Q" + "".join(buf))
            i = j + 1
            continue
        if c == "(":
            depth = 0
            j = i
            while j < len(line):
                if line[j] == "(":
                    depth += 1
                elif line[j] == ")":
                    depth -= 1
                    if depth == 0:
                        break
                elif line[j] == "[":
                    k = line.find("]", j)
                    j = k if k != -1 else j
                j += 1
            out.append(line[i : j + 1])
            i = j + 1
            continue
        j = i
        depth = 0
        while j < len(line):
            if line[j] == "[":
                depth += 1
            elif line[j] == "]":
                depth -= 1
            elif line[j] == " " and depth <= 0:
                break
            j += 1
        out.append(line[i:j])
        i = j
    return out


def _unq(tok: str) -> str:
    return tok[2:] if tok.startswith("\x00Q") else tok


class FakeImapServer:
    def __init__(
        self,
        ca: TestCA,
        *,
        users: Optional[Dict[str, str]] = None,
        security: str = "ssl",
        tokens: Optional[TokenRegistry] = None,
        folders: Iterable[str] = ("INBOX", "Sent"),
        advertise_starttls: bool = True,
        overquota: bool = False,
        search_tz: _dt.tzinfo = _dt.timezone.utc,
    ) -> None:
        assert security in {"ssl", "starttls"}
        # SEARCH SINCE/BEFORE compare dates in the server's own time zone (RFC 3501 leaves the
        # zone to the server); a test can put the server west or east of UTC.
        self.search_tz = search_tz
        self.ca = ca
        self.security = security
        self.users = dict(users or {})
        self.tokens = tokens
        self.advertise_starttls = advertise_starttls
        self.overquota = overquota
        self.commands: List[str] = []
        self.logins: List[Tuple[str, str]] = []  # (user, secret) of each SUCCESSFUL sign-in
        self.handshake_errors: List[str] = []
        self._lock = threading.Lock()
        self.folders: Dict[str, FakeFolder] = {}
        for i, name in enumerate(folders):
            self.folders[name] = FakeFolder(name=name, uidvalidity=1000 + i)
        self._ctx = ca.server_context()
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._sock.settimeout(0.2)
        self.port = int(self._sock.getsockname()[1])
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    # -- mailbox state --------------------------------------------------------------------

    def add_message(
        self,
        folder: str,
        raw: bytes,
        *,
        flags: Iterable[str] = (),
        internaldate: Optional[_dt.datetime] = None,
    ) -> int:
        with self._lock:
            f = self.folders.setdefault(folder, FakeFolder(name=folder, uidvalidity=2000 + len(self.folders)))
            uid = f.uidnext
            f.uidnext += 1
            f.messages.append(
                FakeMessage(uid=uid, raw=raw, flags=set(flags), internaldate=internaldate or _dt.datetime.now(_dt.timezone.utc))
            )
            return uid

    def reset_uidvalidity(self, folder: str) -> None:
        """The server rebuilt the folder: new UIDVALIDITY, UIDs renumbered from 1."""

        with self._lock:
            f = self.folders[folder]
            f.uidvalidity += 7
            for i, m in enumerate(f.messages, start=1):
                m.uid = i
            f.uidnext = len(f.messages) + 1

    def flags_of(self, folder: str, uid: int) -> Set[str]:
        with self._lock:
            for m in self.folders[folder].messages:
                if m.uid == uid:
                    return set(m.flags)
        return set()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._sock.close()

    # -- server loop ----------------------------------------------------------------------

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            conn.settimeout(10)
            threading.Thread(target=self._session, args=(conn,), daemon=True).start()

    def _session(self, raw: socket.socket) -> None:
        conn: Any = raw
        try:
            if self.security == "ssl":
                conn = self._ctx.wrap_socket(raw, server_side=True)
            _ImapSession(self, conn).run()
        except ssl.SSLError as e:
            self.handshake_errors.append(str(e))
        except (OSError, ValueError):
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass


class _ImapSession:
    def __init__(self, server: FakeImapServer, conn: Any) -> None:
        self.s = server
        self.conn = conn
        self.f = conn.makefile("rwb")
        self.user: Optional[str] = None
        self.folder: Optional[FakeFolder] = None
        self.readonly = True
        self.tls = server.security == "ssl"

    def send(self, line: str) -> None:
        self.f.write(line.encode("utf-8") + b"\r\n")
        self.f.flush()

    def send_bytes(self, data: bytes) -> None:
        self.f.write(data)
        self.f.flush()

    def caps(self) -> str:
        caps = ["IMAP4rev1", "UIDPLUS"]
        if not self.tls and self.s.advertise_starttls:
            caps.append("STARTTLS")
        if self.tls:
            caps += ["AUTH=PLAIN", "AUTH=XOAUTH2"]
        else:
            caps.append("LOGINDISABLED")
        return " ".join(caps)

    def run(self) -> None:
        self.send(f"* OK [CAPABILITY {self.caps()}] fake IMAP ready")
        while True:
            line = self.f.readline()
            if not line:
                return
            text = line.decode("utf-8", errors="replace").rstrip("\r\n")
            if not text:
                continue
            tag, _, rest = text.partition(" ")
            cmd, _, args = rest.partition(" ")
            cmd = cmd.upper()
            if cmd == "UID":
                sub, _, args2 = args.partition(" ")
                self.s.commands.append(f"UID {sub.upper()}")
            else:
                self.s.commands.append(cmd)
            handler = getattr(self, "cmd_" + cmd.replace(" ", "_"), None)
            if handler is None:
                self.send(f"{tag} BAD unknown command")
                continue
            if handler(tag, args) == "logout":
                return

    # -- commands -------------------------------------------------------------------------

    def cmd_CAPABILITY(self, tag: str, args: str) -> None:
        self.send(f"* CAPABILITY {self.caps()}")
        self.send(f"{tag} OK CAPABILITY completed")

    def cmd_NOOP(self, tag: str, args: str) -> None:
        self.send(f"{tag} OK NOOP completed")

    def cmd_LOGOUT(self, tag: str, args: str) -> str:
        self.send("* BYE logging out")
        self.send(f"{tag} OK LOGOUT completed")
        return "logout"

    def cmd_STARTTLS(self, tag: str, args: str) -> None:
        if self.tls or not self.s.advertise_starttls:
            self.send(f"{tag} BAD STARTTLS not available")
            return
        self.send(f"{tag} OK begin TLS")
        self.conn = self.s._ctx.wrap_socket(self.conn, server_side=True)
        self.f = self.conn.makefile("rwb")
        self.tls = True

    def _auth_ok(self, user: str, secret: str) -> bool:
        return user in self.s.users and self.s.users[user] == secret

    def cmd_LOGIN(self, tag: str, args: str) -> None:
        if not self.tls:
            self.send(f"{tag} NO [PRIVACYREQUIRED] LOGIN needs TLS")
            return
        toks = [_unq(t) for t in _tokenize(args)]
        if len(toks) != 2:
            self.send(f"{tag} BAD LOGIN needs user and password")
            return
        if self._auth_ok(toks[0], toks[1]):
            self.user = toks[0]
            self.s.logins.append((toks[0], toks[1]))
            self.send(f"{tag} OK [CAPABILITY {self.caps()}] LOGIN completed")
        else:
            self.send(f"{tag} NO [AUTHENTICATIONFAILED] Invalid credentials")

    def cmd_AUTHENTICATE(self, tag: str, args: str) -> None:
        if not self.tls:
            self.send(f"{tag} NO [PRIVACYREQUIRED] needs TLS")
            return
        mech = args.strip().split(" ")[0].upper()
        self.send("+ ")
        resp = self.f.readline().strip()
        try:
            decoded = base64.b64decode(resp)
        except Exception:
            self.send(f"{tag} BAD invalid base64")
            return
        if mech == "PLAIN":
            parts = decoded.split(b"\0")
            if len(parts) == 3 and self._auth_ok(parts[1].decode("utf-8"), parts[2].decode("utf-8")):
                self.user = parts[1].decode("utf-8")
                self.s.logins.append((self.user, parts[2].decode("utf-8")))
                self.send(f"{tag} OK AUTHENTICATE completed")
                return
            self.send(f"{tag} NO [AUTHENTICATIONFAILED] Invalid credentials")
            return
        if mech == "XOAUTH2":
            user, token = parse_xoauth2(decoded)
            if self.s.tokens is not None and self.s.tokens.check(user, token):
                self.user = user
                self.s.logins.append((user, token))
                self.send(f"{tag} OK AUTHENTICATE completed")
                return
            err = base64.b64encode(json.dumps({"status": "401", "schemes": "Bearer"}).encode()).decode()
            self.send(f"+ {err}")
            self.f.readline()  # the client's empty answer
            self.send(f"{tag} NO [AUTHENTICATIONFAILED] Invalid credentials")
            return
        self.send(f"{tag} NO unsupported mechanism")

    def _need_auth(self, tag: str) -> bool:
        if self.user is None:
            self.send(f"{tag} BAD not authenticated")
            return True
        return False

    def cmd_LIST(self, tag: str, args: str) -> None:
        if self._need_auth(tag):
            return
        from abstractcore.comms.email.imap_codec import encode_mailbox

        for name in self.s.folders:
            flag = "\\HasNoChildren"
            if name == "Sent":
                flag += " \\Sent"
            self.send(f'* LIST ({flag}) "/" "{encode_mailbox(name)}"')
        self.send(f"{tag} OK LIST completed")

    def _open(self, tag: str, args: str, readonly: bool, cmd: str) -> None:
        if self._need_auth(tag):
            return
        from abstractcore.comms.email.imap_codec import decode_mailbox

        toks = _tokenize(args)
        name = decode_mailbox(_unq(toks[0])) if toks else ""
        folder = self.s.folders.get(name)
        if folder is None:
            self.send(f"{tag} NO [NONEXISTENT] Unknown mailbox")
            return
        self.folder = folder
        self.readonly = readonly
        self.send(f"* {len(folder.messages)} EXISTS")
        self.send("* 0 RECENT")
        self.send("* FLAGS (\\Seen \\Answered \\Flagged \\Deleted \\Draft)")
        self.send(f"* OK [UIDVALIDITY {folder.uidvalidity}] UIDs valid")
        self.send(f"* OK [UIDNEXT {folder.uidnext}] next UID")
        mode = "READ-ONLY" if readonly else "READ-WRITE"
        self.send(f"{tag} OK [{mode}] {cmd} completed")

    def cmd_EXAMINE(self, tag: str, args: str) -> None:
        self._open(tag, args, True, "EXAMINE")

    def cmd_SELECT(self, tag: str, args: str) -> None:
        self._open(tag, args, False, "SELECT")

    def cmd_STATUS(self, tag: str, args: str) -> None:
        if self._need_auth(tag):
            return
        toks = _tokenize(args)
        folder = self.s.folders.get(_unq(toks[0]) if toks else "")
        if folder is None:
            self.send(f"{tag} NO [NONEXISTENT] Unknown mailbox")
            return
        self.send(f'* STATUS "{folder.name}" (MESSAGES {len(folder.messages)} UIDNEXT {folder.uidnext} UIDVALIDITY {folder.uidvalidity})')
        self.send(f"{tag} OK STATUS completed")

    def cmd_STORE(self, tag: str, args: str) -> None:
        if self.readonly:
            self.send(f"{tag} NO [READ-ONLY] mailbox opened with EXAMINE")
            return
        self.send(f"{tag} OK STORE completed")

    def cmd_UID(self, tag: str, args: str) -> None:
        if self._need_auth(tag):
            return
        if self.folder is None:
            self.send(f"{tag} BAD no mailbox selected")
            return
        sub, _, rest = args.partition(" ")
        sub = sub.upper()
        if sub == "SEARCH":
            self._search(tag, rest, uid=True)
        elif sub == "FETCH":
            self._fetch(tag, rest, uid=True)
        elif sub == "STORE":
            self.cmd_STORE(tag, rest)
        else:
            self.send(f"{tag} BAD unsupported UID command")

    def cmd_SEARCH(self, tag: str, args: str) -> None:
        self._search(tag, args, uid=False)

    def cmd_FETCH(self, tag: str, args: str) -> None:
        self._fetch(tag, args, uid=False)

    # -- SEARCH ---------------------------------------------------------------------------

    def _matches(self, msg: FakeMessage, toks: List[str], i: int) -> Tuple[bool, int]:
        key = _unq(toks[i]).upper()
        parsed = email.message_from_bytes(msg.raw, policy=email.policy.default)

        def hdr(name: str) -> str:
            v = parsed.get(name)
            return str(v) if v is not None else ""

        if key == "ALL":
            return True, i + 1
        if key == "SEEN":
            return "\\Seen" in msg.flags, i + 1
        if key == "UNSEEN":
            return "\\Seen" not in msg.flags, i + 1
        if key in {"FROM", "TO", "CC", "SUBJECT"}:
            needle = _unq(toks[i + 1]).casefold()
            return needle in hdr(key.capitalize() if key != "CC" else "Cc").casefold(), i + 2
        if key in {"SINCE", "BEFORE"}:
            day = _parse_search_date(_unq(toks[i + 1]))
            mday = msg.internaldate.astimezone(self.s.search_tz).date()
            return (mday >= day) if key == "SINCE" else (mday < day), i + 2
        if key == "UID":
            spec = _unq(toks[i + 1])
            return self._in_set(msg.uid, spec), i + 2
        if key == "OR":
            a, j = self._matches(msg, toks, i + 1)
            b, k = self._matches(msg, toks, j)
            return a or b, k
        if key == "NOT":
            a, j = self._matches(msg, toks, i + 1)
            return not a, j
        raise ValueError(f"unsupported search key {key}")

    def _in_set(self, value: int, spec: str) -> bool:
        top = max((m.uid for m in self.folder.messages), default=0)
        for part in spec.split(","):
            if ":" in part:
                a, b = part.split(":", 1)
                lo = top if a == "*" else int(a)
                hi = top if b == "*" else int(b)
                lo, hi = min(lo, hi), max(lo, hi)
                if lo <= value <= hi:
                    return True
            else:
                if (top if part == "*" else int(part)) == value:
                    return True
        return False

    def _search(self, tag: str, args: str, *, uid: bool) -> None:
        toks = _tokenize(args)
        if toks and toks[0].upper() == "CHARSET":
            toks = toks[2:]
        hits = []
        try:
            for seq, msg in enumerate(self.folder.messages, start=1):
                ok = True
                i = 0
                while i < len(toks):
                    m, i = self._matches(msg, toks, i)
                    ok = ok and m
                if ok:
                    hits.append(msg.uid if uid else seq)
        except (ValueError, IndexError):
            self.send(f"{tag} BAD invalid search")
            return
        self.send("* SEARCH" + ("" if not hits else " " + " ".join(str(h) for h in hits)))
        self.send(f"{tag} OK SEARCH completed")

    # -- FETCH ----------------------------------------------------------------------------

    def _fetch(self, tag: str, args: str, *, uid: bool) -> None:
        toks = _tokenize(args)
        if len(toks) < 2:
            self.send(f"{tag} BAD FETCH needs a set and items")
            return
        spec = toks[0]
        items = " ".join(toks[1:]).strip()
        if items.startswith("(") and items.endswith(")"):
            items = items[1:-1]
        upper = items.upper()
        for seq, msg in enumerate(list(self.folder.messages), start=1):
            ident = msg.uid if uid else seq
            wanted = self._in_set(ident, spec) if uid else self._in_seq(seq, spec)
            if not wanted:
                continue
            parts: List[bytes] = [f"* {seq} FETCH (UID {msg.uid}".encode()]
            if "FLAGS" in upper:
                parts.append(f" FLAGS ({' '.join(sorted(msg.flags))})".encode())
            if "RFC822.SIZE" in upper:
                parts.append(f" RFC822.SIZE {len(msg.raw)}".encode())
            if "INTERNALDATE" in upper:
                parts.append(f' INTERNALDATE "{_imap_internaldate(msg.internaldate)}"'.encode())
            body: Optional[bytes] = None
            label = ""
            if "HEADER.FIELDS" in upper:
                start = upper.index("HEADER.FIELDS")
                fields_part = items[items.index("(", start) + 1 : items.index(")", start)]
                names = [n.strip().lower() for n in fields_part.split() if n.strip()]
                parsed = email.message_from_bytes(msg.raw)
                lines = []
                for name in names:
                    for v in parsed.get_all(name) or []:
                        lines.append(f"{name.title()}: {v}")
                body = ("\r\n".join(lines) + "\r\n\r\n").encode("utf-8")
                label = f"BODY[HEADER.FIELDS ({fields_part.upper()})]"
                peek = "BODY.PEEK[" in upper
            elif "BODY.PEEK[]" in upper or "BODY[]" in upper or "RFC822" in upper.replace("RFC822.SIZE", ""):
                body = msg.raw
                label = "BODY[]"
                peek = "BODY.PEEK[]" in upper
            else:
                peek = True
            if body is not None:
                parts.append(f" {label} {{{len(body)}}}\r\n".encode())
                parts.append(body)
                # RFC 3501: a non-PEEK body fetch sets \Seen (unless the mailbox was EXAMINEd).
                if not peek and not self.readonly:
                    msg.flags.add("\\Seen")
            parts.append(b")\r\n")
            self.send_bytes(b"".join(parts))
        self.send(f"{tag} OK FETCH completed")

    def _in_seq(self, seq: int, spec: str) -> bool:
        top = len(self.folder.messages)
        for part in spec.split(","):
            if ":" in part:
                a, b = part.split(":", 1)
                lo = top if a == "*" else int(a)
                hi = top if b == "*" else int(b)
                if min(lo, hi) <= seq <= max(lo, hi):
                    return True
            elif (top if part == "*" else int(part)) == seq:
                return True
        return False


# ---------------------------------------------------------------------------------------
# SMTP (aiosmtpd)
# ---------------------------------------------------------------------------------------


class FakeSmtpServer:
    """aiosmtpd-based SMTP server: `security` "ssl" (implicit TLS) or "starttls"."""

    def __init__(
        self,
        ca: TestCA,
        *,
        users: Optional[Dict[str, str]] = None,
        security: str = "starttls",
        tokens: Optional[TokenRegistry] = None,
        refuse: Optional[Dict[str, int]] = None,
        offer_starttls: bool = True,
    ) -> None:
        try:
            from aiosmtpd.controller import Controller
            from aiosmtpd.smtp import AuthResult
        except ImportError as exc:  # pragma: no cover - test dependency
            raise RuntimeError("FakeSmtpServer needs aiosmtpd, a test dependency of AbstractCore (its `test` extra)") from exc
        assert security in {"ssl", "starttls"}
        self.ca = ca
        self.security = security
        self.users = dict(users or {})
        self.tokens = tokens
        self.refuse = {k.lower(): int(v) for k, v in (refuse or {}).items()}
        self.messages: List[Dict[str, Any]] = []
        self.logins: List[Tuple[str, str]] = []
        server = self

        def authenticator(smtp, session, envelope, mechanism, auth_data):
            user = auth_data.login.decode("utf-8") if isinstance(auth_data.login, bytes) else str(auth_data.login)
            pw = auth_data.password.decode("utf-8") if isinstance(auth_data.password, bytes) else str(auth_data.password)
            if server.users.get(user) == pw:
                server.logins.append((user, pw))
                return AuthResult(success=True)
            return AuthResult(success=False, handled=False)

        class Handler:
            async def auth_XOAUTH2(self, smtp, args):
                if len(args) > 1:
                    blob = base64.b64decode(args[1])
                else:
                    got = await smtp.challenge_auth("")
                    if not isinstance(got, bytes):
                        return AuthResult(success=False, handled=True)
                    blob = got
                user, token = parse_xoauth2(blob)
                if server.tokens is not None and server.tokens.check(user, token):
                    server.logins.append((user, token))
                    return AuthResult(success=True)
                await smtp.challenge_auth(json.dumps({"status": "401", "schemes": "bearer"}))
                return AuthResult(success=False, handled=False)

            async def handle_RCPT(self, smtp, session, envelope, address, rcpt_options):
                code = server.refuse.get(str(address).lower())
                if code:
                    return f"{code} 5.1.1 recipient refused by test server"
                envelope.rcpt_tos.append(address)
                return "250 OK"

            async def handle_DATA(self, smtp, session, envelope):
                server.messages.append(
                    {
                        "mail_from": envelope.mail_from,
                        "rcpt_tos": list(envelope.rcpt_tos),
                        "data": bytes(envelope.original_content or envelope.content or b""),
                    }
                )
                return "250 Message accepted"

        self.port = free_port()
        kwargs: Dict[str, Any] = {
            "hostname": "127.0.0.1",
            "port": self.port,
            "authenticator": authenticator,
            "auth_require_tls": True,
            "server_hostname": "localhost",
        }
        if security == "ssl":
            kwargs["ssl_context"] = ca.server_context()
            # aiosmtpd only tracks TLS started by STARTTLS; the implicit-TLS channel is already
            # encrypted end to end, so AUTH is offered on it.
            kwargs["auth_require_tls"] = False
        elif offer_starttls:
            kwargs["tls_context"] = ca.server_context()
            kwargs["require_starttls"] = True
        else:
            kwargs["auth_require_tls"] = False
        self._controller = Controller(Handler(), **kwargs)
        self._controller.start()

    def close(self) -> None:
        self._controller.stop()


# ---------------------------------------------------------------------------------------
# OAuth2 token endpoint (https)
# ---------------------------------------------------------------------------------------


class FakeOAuthServer:
    """An https OAuth2 server: /token, /device, /authorize (the redirect target is not followed)."""

    def __init__(self, ca: TestCA, tokens: TokenRegistry, *, client_id: str = "test-client", client_secret: str = "test-secret", user: str = "") -> None:
        self.ca = ca
        self.tokens = tokens
        self.client_id = client_id
        self.client_secret = client_secret
        self.user = user
        self.refresh_tokens: Dict[str, str] = {}  # refresh token -> user
        self.codes: Dict[str, Tuple[str, str, str]] = {}  # code -> (user, redirect_uri, challenge)
        self.devices: Dict[str, Dict[str, Any]] = {}
        self.requests: List[Dict[str, str]] = []
        self.rotate_refresh = False
        self.access_ttl_s = 3600.0
        srv = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a: Any) -> None:
                return

            def _json(self, status: int, doc: Dict[str, Any]) -> None:
                body = json.dumps(doc).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                form = dict(urllib.parse.parse_qsl(self.rfile.read(length).decode()))
                srv.requests.append({"path": self.path, **{k: v for k, v in form.items() if k in ("grant_type", "client_id", "scope")}})
                if form.get("client_id") != srv.client_id or (srv.client_secret and form.get("client_secret") != srv.client_secret):
                    return self._json(401, {"error": "invalid_client"})
                if self.path == "/device":
                    code = "dc-" + secrets.token_urlsafe(8)
                    srv.devices[code] = {"approved": False, "denied": False, "polls": 0}
                    return self._json(200, {
                        "device_code": code, "user_code": "WDJB-MJHT",
                        "verification_uri": "https://localhost/device", "expires_in": 600, "interval": 1,
                    })
                if self.path != "/token":
                    return self._json(404, {"error": "not_found"})
                grant = form.get("grant_type")
                if grant == "refresh_token":
                    user = srv.refresh_tokens.get(form.get("refresh_token", ""))
                    if not user:
                        return self._json(400, {"error": "invalid_grant"})
                    doc = {"access_token": srv.tokens.issue(user, srv.access_ttl_s), "expires_in": srv.access_ttl_s, "token_type": "Bearer"}
                    if srv.rotate_refresh:
                        new = "rt-" + secrets.token_urlsafe(12)
                        del srv.refresh_tokens[form["refresh_token"]]
                        srv.refresh_tokens[new] = user
                        doc["refresh_token"] = new
                    return self._json(200, doc)
                if grant == "authorization_code":
                    got = srv.codes.pop(form.get("code", ""), None)
                    if not got or got[1] != form.get("redirect_uri"):
                        return self._json(400, {"error": "invalid_grant"})
                    import hashlib

                    verifier = form.get("code_verifier", "")
                    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
                    if challenge != got[2]:
                        return self._json(400, {"error": "invalid_grant"})
                    return self._json(200, srv._grant(got[0]))
                if grant == "urn:ietf:params:oauth:grant-type:device_code":
                    dev = srv.devices.get(form.get("device_code", ""))
                    if dev is None:
                        return self._json(400, {"error": "invalid_grant"})
                    dev["polls"] += 1
                    if dev["denied"]:
                        return self._json(400, {"error": "access_denied"})
                    if not dev["approved"]:
                        return self._json(400, {"error": "authorization_pending"})
                    return self._json(200, srv._grant(srv.user))
                return self._json(400, {"error": "unsupported_grant_type"})

        class TLSServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        self._httpd = TLSServer(("127.0.0.1", 0), Handler)
        self._httpd.socket = ca.server_context().wrap_socket(self._httpd.socket, server_side=True)
        self.port = int(self._httpd.server_address[1])
        self.base_url = f"https://localhost:{self.port}"
        self._thread = threading.Thread(target=self._httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
        self._thread.start()

    def _grant(self, user: str) -> Dict[str, Any]:
        refresh = "rt-" + secrets.token_urlsafe(12)
        self.refresh_tokens[refresh] = user
        return {
            "access_token": self.tokens.issue(user, self.access_ttl_s),
            "refresh_token": refresh,
            "expires_in": self.access_ttl_s,
            "token_type": "Bearer",
        }

    def issue_refresh_token(self, user: str) -> str:
        refresh = "rt-" + secrets.token_urlsafe(12)
        self.refresh_tokens[refresh] = user
        return refresh

    def authorize(self, authorization_url: str, user: str) -> str:
        """Play the provider's consent page: returns the redirect URL carrying a code."""

        q = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(authorization_url).query))
        code = "code-" + secrets.token_urlsafe(8)
        self.codes[code] = (user, q["redirect_uri"], q["code_challenge"])
        return q["redirect_uri"] + "?" + urllib.parse.urlencode({"code": code, "state": q["state"]})

    def approve_all_devices(self) -> None:
        for dev in self.devices.values():
            dev["approved"] = True

    def revoke_refresh_tokens(self) -> None:
        self.refresh_tokens.clear()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


def build_message(
    *,
    from_: str,
    to: str,
    subject: str,
    text: str = "",
    html: str = "",
    cc: str = "",
    reply_to: str = "",
    message_id: str = "",
    references: str = "",
    attachments: Iterable[Tuple[str, str, bytes]] = (),
) -> bytes:
    """An RFC 5322 message for the IMAP fake (`attachments`: (filename, content_type, data))."""

    from email.message import EmailMessage
    from email.utils import formatdate, make_msgid

    m = EmailMessage()
    m["From"] = from_
    m["To"] = to
    if cc:
        m["Cc"] = cc
    if reply_to:
        m["Reply-To"] = reply_to
    m["Subject"] = subject
    m["Date"] = formatdate(localtime=False)
    m["Message-ID"] = message_id or make_msgid(domain="example.test")
    if references:
        m["References"] = references
    if html and text:
        m.set_content(text)
        m.add_alternative(html, subtype="html")
    elif html:
        m.set_content(html, subtype="html")
    else:
        m.set_content(text or "")
    for filename, ctype, data in attachments:
        maintype, subtype = ctype.split("/", 1)
        m.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)
    return m.as_bytes()
