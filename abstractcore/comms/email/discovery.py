"""Mail server auto-discovery: the IMAP and SMTP servers of an address, from deterministic lookups.

    from abstractcore.comms.email import discover_servers

    found = discover_servers("me@fastmail.com")
    # {"address": "me@fastmail.com", "domain": "fastmail.com", "found": True, "source": "known",
    #  "provider": None, "imap": {"host": "imap.fastmail.com", "port": 993, "security": "ssl"},
    #  "smtp": {"host": "smtp.fastmail.com", "port": 465, "security": "ssl"},
    #  "username": "me@fastmail.com", "tried": [{"step": "known", "result": "found"}]}

The steps, in order (the first one that yields BOTH an IMAP and an SMTP server wins):

1. ``known``      a built-in table of common providers (values from the Thunderbird ISPDB,
                  https://autoconfig.thunderbird.net/v1.1/<domain>, checked 2026-09-30);
                  Google and Microsoft carry ``provider`` so a client can offer their sign-in.
2. ``autoconfig`` the domain's own Mozilla autoconfig file:
                  ``https://autoconfig.<domain>/mail/config-v1.1.xml?emailaddress=<address>``,
                  then ``https://<domain>/.well-known/autoconfig/mail/config-v1.1.xml``.
3. ``ispdb``      the Thunderbird ISPDB: ``https://autoconfig.thunderbird.net/v1.1/<domain>``.
4. ``srv``        DNS SRV records (RFC 6186 / RFC 8314): ``_imaps._tcp`` (IMAP over TLS),
                  ``_submissions._tcp`` (SMTP over TLS), ``_submission._tcp`` (SMTP STARTTLS).
5. ``mx``         the domain's MX hosts, matched by DNS suffix to a known provider
                  (``*.google.com`` / ``*.googlemail.com`` -> Google Workspace,
                  ``*.mail.protection.outlook.com`` -> Microsoft 365,
                  ``*.messagingengine.com`` -> Fastmail).

Every step is a lookup with a structured answer (a table key, an XML document, a DNS record, a
DNS suffix); nothing reads free text. Only encrypted servers are returned (``ssl`` or
``starttls``); a configuration that offers only plain connections is skipped.

Safety: HTTPS only (redirects too), a response size cap, per-step timeouts and an overall time
budget; the default fetcher refuses hosts that resolve to loopback, private, link-local or other
non-public addresses (a gateway must not become a probe of its own network); XML is parsed with
``defusedxml`` when installed, else with expat and every DTD / entity declaration refused. SRV and
MX records are resolved with dnspython when importable, else with a minimal stdlib DNS query to
the system's nameserver. Every network step is injectable (``http_get``, ``resolve_srv``,
``resolve_mx``) so tests run without a network.
"""

from __future__ import annotations

import ipaddress
import random
import socket
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

from .errors import EmailError
from .policy import split_address

MAX_RESPONSE_BYTES = 256 * 1024
BUDGET_FACTOR = 3.0  # the whole discovery gets `timeout * BUDGET_FACTOR` seconds
ISPDB_URL = "https://autoconfig.thunderbird.net/v1.1/{domain}"
SOURCES = ("known", "autoconfig", "ispdb", "srv", "mx")

HttpGet = Callable[[str, float], Optional[bytes]]
ResolveSrv = Callable[[str, float], List[Tuple[int, int, int, str]]]
ResolveMx = Callable[[str, float], List[Tuple[int, str]]]


class EmailDiscoveryFailed(EmailError):
    """No step found both mail servers of the address's domain. `details`: `{domain, tried}`."""

    code = "email_discovery_failed"

    @property
    def message(self) -> str:  # one sentence pair, the way a form shows it
        return f"{self.cause} {self.fix}".strip()


# ---------------------------------------------------------------------------------------
# 1. Known providers (Thunderbird ISPDB values)
# ---------------------------------------------------------------------------------------

def _leg(host: str, port: int, security: str) -> Dict[str, Any]:
    return {"host": host, "port": port, "security": security}


# username: "address" (%EMAILADDRESS%) or "local" (%EMAILLOCALPART%). When ISPDB gives different
# forms for IMAP and SMTP (iCloud: IMAP local part, SMTP address) the full address is kept: one
# account has one user name, and every such provider accepts the address on both.
_PROVIDERS: Dict[str, Dict[str, Any]] = {
    "google": {
        "provider": "google",
        "imap": _leg("imap.gmail.com", 993, "ssl"),
        "smtp": _leg("smtp.gmail.com", 465, "ssl"),
        "username": "address",
        "domains": ("gmail.com", "googlemail.com"),
    },
    "microsoft_consumer": {
        "provider": "microsoft",
        "imap": _leg("outlook.office365.com", 993, "ssl"),
        "smtp": _leg("smtp-mail.outlook.com", 587, "starttls"),
        "username": "address",
        "domains": (
            "outlook.com", "hotmail.com", "live.com", "msn.com",
            "outlook.fr", "outlook.de", "outlook.es", "outlook.it", "outlook.be",
            "hotmail.fr", "hotmail.de", "hotmail.es", "hotmail.it", "hotmail.be", "hotmail.co.uk",
            "live.fr", "live.de", "live.it", "live.be", "live.nl", "live.co.uk", "live.ca",
        ),
    },
    "microsoft_365": {
        "provider": "microsoft",
        "imap": _leg("outlook.office365.com", 993, "ssl"),
        "smtp": _leg("smtp.office365.com", 587, "starttls"),
        "username": "address",
        "domains": ("office365.com",),
    },
    "icloud": {
        "provider": None,
        "imap": _leg("imap.mail.me.com", 993, "ssl"),
        "smtp": _leg("smtp.mail.me.com", 587, "starttls"),
        "username": "address",
        "domains": ("icloud.com", "me.com", "mac.com"),
    },
    "yahoo": {
        "provider": None,
        "imap": _leg("imap.mail.yahoo.com", 993, "ssl"),
        "smtp": _leg("smtp.mail.yahoo.com", 465, "ssl"),
        "username": "address",
        "domains": ("yahoo.com", "yahoo.fr", "yahoo.de", "yahoo.co.uk", "yahoo.es", "yahoo.it", "ymail.com", "rocketmail.com"),
    },
    "aol": {
        "provider": None,
        "imap": _leg("imap.aol.com", 993, "ssl"),
        "smtp": _leg("smtp.aol.com", 465, "ssl"),
        "username": "address",
        "domains": ("aol.com",),
    },
    # Not in the ISPDB: Fastmail publishes the same values as RFC 6186 SRV records.
    "fastmail": {
        "provider": None,
        "imap": _leg("imap.fastmail.com", 993, "ssl"),
        "smtp": _leg("smtp.fastmail.com", 465, "ssl"),
        "username": "address",
        "domains": ("fastmail.com", "fastmail.fm"),
    },
    "gmx_com": {
        "provider": None,
        "imap": _leg("imap.gmx.com", 993, "ssl"),
        "smtp": _leg("mail.gmx.com", 465, "ssl"),
        "username": "address",
        "domains": ("gmx.com",),
    },
    "gmx_net": {
        "provider": None,
        "imap": _leg("imap.gmx.net", 993, "ssl"),
        "smtp": _leg("mail.gmx.net", 465, "ssl"),
        "username": "address",
        "domains": ("gmx.net", "gmx.de", "gmx.at", "gmx.ch"),
    },
    "webde": {
        "provider": None,
        "imap": _leg("imap.web.de", 993, "ssl"),
        "smtp": _leg("smtp.web.de", 465, "ssl"),
        "username": "local",
        "domains": ("web.de",),
    },
    "mailcom": {
        "provider": None,
        "imap": _leg("imap.mail.com", 993, "ssl"),
        "smtp": _leg("smtp.mail.com", 465, "ssl"),
        "username": "address",
        "domains": ("mail.com",),
    },
    "zoho": {
        "provider": None,
        "imap": _leg("imap.zoho.com", 993, "ssl"),
        "smtp": _leg("smtp.zoho.com", 465, "ssl"),
        "username": "address",
        "domains": ("zoho.com", "zohomail.com"),
    },
    "yandex": {
        "provider": None,
        "imap": _leg("imap.yandex.com", 993, "ssl"),
        "smtp": _leg("smtp.yandex.com", 465, "ssl"),
        "username": "address",
        "domains": ("yandex.com", "yandex.ru"),
    },
    "mailru": {
        "provider": None,
        "imap": _leg("imap.mail.ru", 993, "ssl"),
        "smtp": _leg("smtp.mail.ru", 465, "ssl"),
        "username": "address",
        "domains": ("mail.ru",),
    },
    "orange": {
        "provider": None,
        "imap": _leg("imap.orange.fr", 993, "ssl"),
        "smtp": _leg("smtp.orange.fr", 465, "ssl"),
        "username": "address",
        "domains": ("orange.fr", "wanadoo.fr"),
    },
    "free": {
        "provider": None,
        "imap": _leg("imap.free.fr", 993, "ssl"),
        "smtp": _leg("smtp.free.fr", 465, "ssl"),
        "username": "local",
        "domains": ("free.fr",),
    },
    "laposte": {
        "provider": None,
        "imap": _leg("imap.laposte.net", 993, "ssl"),
        "smtp": _leg("smtp.laposte.net", 465, "ssl"),
        "username": "local",
        "domains": ("laposte.net",),
    },
    "sfr": {
        "provider": None,
        "imap": _leg("imap.sfr.fr", 993, "ssl"),
        "smtp": _leg("smtp.sfr.fr", 465, "ssl"),
        "username": "address",
        "domains": ("sfr.fr",),
    },
    "tonline": {
        "provider": None,
        "imap": _leg("secureimap.t-online.de", 993, "ssl"),
        "smtp": _leg("securesmtp.t-online.de", 465, "ssl"),
        "username": "address",
        "domains": ("t-online.de",),
    },
}

KNOWN_DOMAINS: Dict[str, str] = {d: key for key, spec in _PROVIDERS.items() for d in spec["domains"]}

# MX host DNS suffix -> provider table key (hosted domains: Google Workspace, Microsoft 365, Fastmail).
MX_SUFFIXES: Tuple[Tuple[str, str], ...] = (
    ("google.com", "google"),
    ("googlemail.com", "google"),
    ("mail.protection.outlook.com", "microsoft_365"),
    ("messagingengine.com", "fastmail"),
)

# An autoconfig / ISPDB / SRV answer whose IMAP host is one of these is that provider's mailbox
# (a Google Workspace or Microsoft 365 domain found through its own configuration).
PROVIDER_IMAP_HOSTS: Dict[str, str] = {
    "imap.gmail.com": "google",
    "outlook.office365.com": "microsoft",
    "imap-mail.outlook.com": "microsoft",
}


def _username(form: str, address: str) -> str:
    return address.split("@", 1)[0] if form == "local" else address


def _from_table(key: str, address: str) -> Dict[str, Any]:
    spec = _PROVIDERS[key]
    return {
        "provider": spec["provider"],
        "imap": dict(spec["imap"]),
        "smtp": dict(spec["smtp"]),
        "username": _username(spec["username"], address),
    }


# ---------------------------------------------------------------------------------------
# Safe XML (autoconfig / ISPDB)
# ---------------------------------------------------------------------------------------

class _UnsafeXml(ValueError):
    pass


def parse_xml(data: bytes) -> Any:
    """An ElementTree element from untrusted bytes; DTDs and entity declarations are refused."""

    try:
        import defusedxml.ElementTree as dET  # type: ignore

        try:
            return dET.fromstring(data, forbid_dtd=True, forbid_entities=True, forbid_external=True)
        except dET.ParseError as exc:  # type: ignore[attr-defined]
            raise ValueError(f"not XML: {exc}") from None
        except Exception as exc:  # defusedxml.DefusedXmlException and its subclasses
            raise _UnsafeXml(type(exc).__name__) from None
    except ImportError:
        pass

    import xml.etree.ElementTree as ET
    from xml.parsers import expat

    builder = ET.TreeBuilder()
    parser = expat.ParserCreate()

    def refuse(*_args: Any) -> None:
        raise _UnsafeXml("DTD or entity declaration")

    parser.StartDoctypeDeclHandler = refuse
    parser.EntityDeclHandler = refuse
    parser.UnparsedEntityDeclHandler = refuse
    parser.ExternalEntityRefHandler = refuse  # type: ignore[assignment]
    parser.StartElementHandler = lambda tag, attrs: builder.start(tag, attrs)
    parser.EndElementHandler = lambda tag: builder.end(tag)
    parser.CharacterDataHandler = builder.data
    try:
        parser.Parse(data, True)
        return builder.close()
    except expat.ExpatError as exc:
        raise ValueError(f"not XML: {exc}") from None


def _local(tag: Any) -> str:
    t = str(tag or "")
    return t.rsplit("}", 1)[-1]


def _child_text(el: Any, name: str) -> str:
    for c in list(el):
        if _local(c.tag) == name:
            return str(c.text or "").strip()
    return ""


_SOCKET_TYPES = {"SSL": "ssl", "STARTTLS": "starttls"}


def _placeholders(value: str, address: str, domain: str) -> str:
    local = address.split("@", 1)[0]
    return value.replace("%EMAILADDRESS%", address).replace("%EMAILLOCALPART%", local).replace("%EMAILDOMAIN%", domain)


def parse_autoconfig(data: bytes, address: str, domain: str) -> Dict[str, Any]:
    """`{imap, smtp, username}` from a Mozilla autoconfig document (config-v1.1).

    The first encrypted server of each kind, in the document's order (its preference). Raises
    ValueError when the bytes are not a clientConfig document.
    """

    root = parse_xml(data)
    if _local(root.tag) != "clientConfig":
        raise ValueError("not an autoconfig document")
    out: Dict[str, Any] = {"imap": None, "smtp": None, "username": ""}
    users: Dict[str, str] = {}
    for provider in root.iter():
        if _local(provider.tag) != "emailProvider":
            continue
        for server in list(provider):
            kind = _local(server.tag)
            want = {"incomingServer": ("imap", "imap"), "outgoingServer": ("smtp", "smtp")}.get(kind)
            if want is None or str(server.get("type") or "").lower() != want[0]:
                continue
            leg = want[1]
            if out[leg] is not None:
                continue
            security = _SOCKET_TYPES.get(_child_text(server, "socketType").upper())
            host = _placeholders(_child_text(server, "hostname"), address, domain).strip().rstrip(".").lower()
            try:
                port = int(_child_text(server, "port"))
            except ValueError:
                continue
            if not security or not host or not 1 <= port <= 65535 or any(c.isspace() for c in host) or "/" in host or "@" in host:
                continue
            out[leg] = _leg(host, port, security)
            user = _child_text(server, "username")
            if user:
                users[leg] = _placeholders(user, address, domain)
        break  # one emailProvider per document
    if users:
        forms = set(users.values())
        out["username"] = forms.pop() if len(forms) == 1 else address
    return out


# ---------------------------------------------------------------------------------------
# Default network steps (tests inject their own)
# ---------------------------------------------------------------------------------------

class _RefusedHost(OSError):
    pass


def _public_host(host: str, port: int = 443) -> None:
    """Refuse a host that resolves to a non-public address (loopback, private, link-local ...)."""

    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise _RefusedHost(f"does not resolve ({exc.__class__.__name__})") from None
    if not infos:
        raise _RefusedHost("does not resolve")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%", 1)[0])
        if not ip.is_global:
            raise _RefusedHost("resolves to a non-public address")


class _HttpsOnlyRedirects(urllib.request.HTTPRedirectHandler):
    max_redirections = 3

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        parts = urllib.parse.urlsplit(newurl)
        if parts.scheme != "https" or not parts.hostname:
            raise urllib.error.URLError("redirect to a non-HTTPS address refused")
        _public_host(parts.hostname, parts.port or 443)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def https_get(url: str, timeout: float) -> Optional[bytes]:
    """GET an HTTPS URL: the body (at most MAX_RESPONSE_BYTES), None on 404/410. Raises OSError /
    ValueError on anything else (refused host, TLS failure, timeout, oversize body)."""

    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise ValueError("only HTTPS addresses are fetched")
    _public_host(parts.hostname, parts.port or 443)
    opener = urllib.request.build_opener(_HttpsOnlyRedirects())
    req = urllib.request.Request(url, headers={"Accept": "application/xml, text/xml", "User-Agent": "AbstractCore-mail-discovery"})
    try:
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 410):
            return None
        raise OSError(f"HTTP {exc.code}") from None
    if len(body) > MAX_RESPONSE_BYTES:
        raise ValueError("response larger than the size cap")
    return body


def _dnspython_available() -> bool:
    try:
        import dns.resolver  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_srv_default(name: str, timeout: float) -> List[Tuple[int, int, int, str]]:
    """SRV records `(priority, weight, port, target)`; [] when there are none."""

    if _dnspython_available():
        import dns.exception
        import dns.resolver

        try:
            answer = dns.resolver.resolve(name, "SRV", lifetime=timeout)
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
            return []
        except dns.exception.DNSException as exc:
            raise OSError(f"DNS: {type(exc).__name__}") from None
        return [(int(r.priority), int(r.weight), int(r.port), str(r.target).rstrip(".").lower()) for r in answer]
    return [rec for rec in _stdlib_query(name, 33, timeout)]  # type: ignore[misc]


def resolve_mx_default(domain: str, timeout: float) -> List[Tuple[int, str]]:
    """MX records `(preference, host)`; [] when there are none."""

    if _dnspython_available():
        import dns.exception
        import dns.resolver

        try:
            answer = dns.resolver.resolve(domain, "MX", lifetime=timeout)
        except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
            return []
        except dns.exception.DNSException as exc:
            raise OSError(f"DNS: {type(exc).__name__}") from None
        return [(int(r.preference), str(r.exchange).rstrip(".").lower()) for r in answer]
    return [rec for rec in _stdlib_query(domain, 15, timeout)]  # type: ignore[misc]


# -- a minimal stdlib DNS client (UDP, one question, the system's first nameserver) -------

def _system_nameserver() -> str:
    try:
        with open("/etc/resolv.conf", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) >= 2 and parts[0] == "nameserver":
                    return parts[1]
    except OSError:
        pass
    raise OSError("no DNS nameserver configured (install dnspython for this platform)")


def _encode_name(name: str) -> bytes:
    out = b""
    for label in name.rstrip(".").split("."):
        raw = label.encode("idna") if label else b""
        if not raw or len(raw) > 63:
            raise ValueError("invalid DNS name")
        out += bytes([len(raw)]) + raw
    return out + b"\x00"


def _read_name(msg: bytes, offset: int) -> Tuple[str, int]:
    labels: List[str] = []
    jumped = False
    end = offset
    hops = 0
    while True:
        if offset >= len(msg):
            raise ValueError("truncated DNS name")
        length = msg[offset]
        if length & 0xC0 == 0xC0:
            if offset + 1 >= len(msg):
                raise ValueError("truncated DNS pointer")
            pointer = ((length & 0x3F) << 8) | msg[offset + 1]
            if not jumped:
                end = offset + 2
            jumped = True
            hops += 1
            if hops > 32:
                raise ValueError("DNS name pointer loop")
            offset = pointer
            continue
        if length == 0:
            if not jumped:
                end = offset + 1
            break
        labels.append(msg[offset + 1 : offset + 1 + length].decode("ascii", errors="replace"))
        offset += 1 + length
    return ".".join(labels).lower(), end


def parse_dns_response(msg: bytes, qid: int, qtype: int) -> List[Tuple[Any, ...]]:
    """SRV `(priority, weight, port, target)` or MX `(preference, host)` records of a response."""

    if len(msg) < 12:
        raise ValueError("short DNS response")
    rid, flags, qd, an, _ns, _ar = struct.unpack("!HHHHHH", msg[:12])
    if rid != qid:
        raise ValueError("DNS response id mismatch")
    rcode = flags & 0x000F
    if flags & 0x0200:
        raise OSError("DNS response truncated")
    if rcode == 3:
        return []
    if rcode != 0:
        raise OSError(f"DNS error rcode {rcode}")
    offset = 12
    for _ in range(qd):
        _name, offset = _read_name(msg, offset)
        offset += 4
    out: List[Tuple[Any, ...]] = []
    for _ in range(an):
        _name, offset = _read_name(msg, offset)
        if offset + 10 > len(msg):
            raise ValueError("truncated DNS record")
        rtype, _rclass, _ttl, rdlen = struct.unpack("!HHIH", msg[offset : offset + 10])
        offset += 10
        rdata_at = offset
        offset += rdlen
        if rtype != qtype:
            continue
        if qtype == 33:
            prio, weight, port = struct.unpack("!HHH", msg[rdata_at : rdata_at + 6])
            target, _ = _read_name(msg, rdata_at + 6)
            out.append((prio, weight, port, target))
        elif qtype == 15:
            (pref,) = struct.unpack("!H", msg[rdata_at : rdata_at + 2])
            host, _ = _read_name(msg, rdata_at + 2)
            out.append((pref, host))
    return out


def _stdlib_query(name: str, qtype: int, timeout: float) -> List[Tuple[Any, ...]]:
    server = _system_nameserver()
    qid = random.SystemRandom().randrange(0, 65536)
    packet = struct.pack("!HHHHHH", qid, 0x0100, 1, 0, 0, 0) + _encode_name(name) + struct.pack("!HH", qtype, 1)
    family = socket.AF_INET6 if ":" in server else socket.AF_INET
    with socket.socket(family, socket.SOCK_DGRAM) as sock:
        sock.settimeout(timeout)
        sock.sendto(packet, (server, 53))
        data, _ = sock.recvfrom(4096)
    return parse_dns_response(data, qid, qtype)


# ---------------------------------------------------------------------------------------
# discover_servers
# ---------------------------------------------------------------------------------------

def _step_error(exc: BaseException) -> str:
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return "error: timed out"
    if isinstance(exc, _UnsafeXml):
        return "error: unsafe XML refused"
    if isinstance(exc, _RefusedHost):
        return f"error: host {exc}"
    if isinstance(exc, urllib.error.URLError) and isinstance(getattr(exc, "reason", None), (socket.timeout, TimeoutError)):
        return "error: timed out"
    return f"error: {type(exc).__name__}"


def _provider_for(imap: Optional[Dict[str, Any]]) -> Optional[str]:
    return PROVIDER_IMAP_HOSTS.get(str((imap or {}).get("host") or "").lower()) if imap else None


def discover_servers(
    address: str,
    *,
    timeout: float = 4.0,
    http_get: Optional[HttpGet] = None,
    resolve_srv: Optional[ResolveSrv] = None,
    resolve_mx: Optional[ResolveMx] = None,
) -> Dict[str, Any]:
    """The IMAP and SMTP servers of `address` (see the module docstring for the steps).

    Returns `{address, domain, found, source, provider, imap, smtp, username, tried}`; `tried`
    lists every step taken with its result. When no step finds both servers, `found` is False,
    `source` is None and `imap` / `smtp` hold the first partial answer (or None) so a form can
    prefill what was found. Raises ValueError when `address` is not an email address.
    """

    text = str(address or "").strip()
    local, domain = split_address(text)  # ValueError for a non-address
    del local
    http_get = http_get or _net.http_get
    resolve_srv = resolve_srv or _net.resolve_srv
    resolve_mx = resolve_mx or _net.resolve_mx
    timeout = max(0.1, float(timeout))
    deadline = time.monotonic() + timeout * BUDGET_FACTOR
    tried: List[Dict[str, str]] = []
    partial: Dict[str, Any] = {}

    def result(found: bool, source: Optional[str], data: Dict[str, Any]) -> Dict[str, Any]:
        imap = data.get("imap") if found else partial.get("imap")
        smtp = data.get("smtp") if found else partial.get("smtp")
        return {
            "address": text,
            "domain": domain,
            "found": found,
            "source": source if found else None,
            "provider": (data.get("provider") or _provider_for(imap)) if found else None,
            "imap": imap,
            "smtp": smtp,
            "username": (data.get("username") or text) if found else (partial.get("username") or text),
            "tried": tried,
        }

    def remaining() -> float:
        return deadline - time.monotonic()

    def complete(data: Dict[str, Any]) -> bool:
        if data.get("imap") and data.get("smtp"):
            return True
        if (data.get("imap") or data.get("smtp")) and not partial:
            partial.update({k: data.get(k) for k in ("imap", "smtp", "username")})
        return False

    # 1. known providers
    key = KNOWN_DOMAINS.get(domain)
    if key:
        tried.append({"step": "known", "result": "found"})
        return result(True, "known", _from_table(key, text))
    tried.append({"step": "known", "result": "no match"})

    # IP literals and single-label names have no mail configuration to look up.
    try:
        ipaddress.ip_address(domain.strip("[]"))
        is_ip = True
    except ValueError:
        is_ip = False
    if is_ip or "." not in domain:
        tried.append({"step": "autoconfig", "result": "skipped: not a DNS domain"})
        return result(False, None, {})

    # 2-3. autoconfig documents (the domain's own, then the ISPDB)
    quoted = urllib.parse.quote(text, safe="@")
    documents = (
        ("autoconfig", f"https://autoconfig.{domain}/mail/config-v1.1.xml?emailaddress={quoted}"),
        ("autoconfig", f"https://{domain}/.well-known/autoconfig/mail/config-v1.1.xml"),
        ("ispdb", ISPDB_URL.format(domain=domain)),
    )
    for step, url in documents:
        if remaining() <= 0:
            tried.append({"step": step, "url": url, "result": "skipped: time budget spent"})
            continue
        try:
            body = http_get(url, min(timeout, remaining()))
        except Exception as exc:  # noqa: BLE001 - every failure is one tried row
            tried.append({"step": step, "url": url, "result": _step_error(exc)})
            continue
        if body is None:
            tried.append({"step": step, "url": url, "result": "not found"})
            continue
        try:
            data = parse_autoconfig(body, text, domain)
        except _UnsafeXml:
            tried.append({"step": step, "url": url, "result": "error: unsafe XML refused"})
            continue
        except ValueError:
            tried.append({"step": step, "url": url, "result": "error: not an autoconfig document"})
            continue
        if complete(data):
            tried.append({"step": step, "url": url, "result": "found"})
            return result(True, step, data)
        tried.append({"step": step, "url": url, "result": "partial" if (data.get("imap") or data.get("smtp")) else "no encrypted IMAP/SMTP server"})

    # 4. SRV (RFC 6186 / 8314)
    srv_data: Dict[str, Any] = {"imap": None, "smtp": None, "username": text}
    for name, leg, security in (
        (f"_imaps._tcp.{domain}", "imap", "ssl"),
        (f"_submissions._tcp.{domain}", "smtp", "ssl"),
        (f"_submission._tcp.{domain}", "smtp", "starttls"),
    ):
        if srv_data[leg] is not None:
            continue
        if remaining() <= 0:
            tried.append({"step": "srv", "name": name, "result": "skipped: time budget spent"})
            continue
        try:
            records = resolve_srv(name, min(timeout, remaining()))
        except Exception as exc:  # noqa: BLE001
            tried.append({"step": "srv", "name": name, "result": _step_error(exc)})
            continue
        usable = sorted(
            (r for r in records if str(r[3] or "").strip(".") and 1 <= int(r[2]) <= 65535),
            key=lambda r: (int(r[0]), -int(r[1])),
        )
        if not usable:
            tried.append({"step": "srv", "name": name, "result": "not found"})
            continue
        best = usable[0]
        srv_data[leg] = _leg(str(best[3]).rstrip(".").lower(), int(best[2]), security)
        tried.append({"step": "srv", "name": name, "result": "found"})
    if complete(srv_data):
        return result(True, "srv", srv_data)

    # 5. MX -> a known hosting provider
    if remaining() <= 0:
        tried.append({"step": "mx", "result": "skipped: time budget spent"})
        return result(False, None, {})
    try:
        mx = resolve_mx(domain, min(timeout, remaining()))
    except Exception as exc:  # noqa: BLE001
        tried.append({"step": "mx", "result": _step_error(exc)})
        return result(False, None, {})
    for _pref, host in sorted(mx, key=lambda r: int(r[0])):
        h = str(host or "").rstrip(".").lower()
        for suffix, table_key in MX_SUFFIXES:
            if h == suffix or h.endswith("." + suffix):
                tried.append({"step": "mx", "result": f"found ({h})"})
                return result(True, "mx", _from_table(table_key, text))
    tried.append({"step": "mx", "result": "no known provider" if mx else "not found"})
    return result(False, None, {})


def require_servers(address: str, **kwargs: Any) -> Dict[str, Any]:
    """`discover_servers`, raising `EmailDiscoveryFailed` (details `{domain, tried}`) when not found."""

    found = discover_servers(address, **kwargs)
    if not found["found"]:
        raise EmailDiscoveryFailed(
            f"Couldn't find the mail servers for {found['domain']}.",
            "Open Server settings and enter them.",
            details={"domain": found["domain"], "tried": found["tried"]},
        )
    return found


STANDARD_IMAP_PORT = 993
STANDARD_SMTP_PORT = 465


def _clean_leg(leg: Any) -> Optional[Dict[str, Any]]:
    """A discovered `{host, port, security}` leg, or None when it has no host."""

    if not isinstance(leg, dict) or not str(leg.get("host") or "").strip():
        return None
    return {"host": str(leg["host"]).strip(), "port": int(leg["port"]), "security": str(leg.get("security") or "ssl")}


def server_defaults(address: str, discovered: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The server fields a mailbox form pre-fills for `address`.

    `discovered` is a `discover_servers(address)` result; when it is None, discovery runs here
    (network). With `discovered` given this is a pure function.

    Returns `{imap: {host, port, security}, smtp: {host, port, security}, login, source, provider,
    message}`:

    - discovery found both servers -> its values (`source: "discovered"`, `login` = its user name:
      the address, or the local part for a provider that signs in with it);
    - otherwise -> the standard `imap.<domain>` 993 SSL and `smtp.<domain>` 465 SSL, `login` =
      the address (`source: "standard"`); a server discovery found on its own is kept.

    `message` is one sentence a form shows under the fields. Raises ValueError when `address` is
    not an email address.
    """

    text = str(address or "").strip()
    _local, domain = split_address(text)  # ValueError for a non-address
    if discovered is None:
        discovered = discover_servers(text)
    found = bool(discovered.get("found"))
    imap = _clean_leg(discovered.get("imap"))
    smtp = _clean_leg(discovered.get("smtp"))
    if found and imap and smtp:
        return {
            "imap": imap,
            "smtp": smtp,
            "login": str(discovered.get("username") or text),
            "source": "discovered",
            "provider": discovered.get("provider") or None,
            "message": f"Settings found for {domain}.",
        }
    return {
        "imap": imap or _leg(f"imap.{domain}", STANDARD_IMAP_PORT, "ssl"),
        "smtp": smtp or _leg(f"smtp.{domain}", STANDARD_SMTP_PORT, "ssl"),
        "login": text,
        "source": "standard",
        "provider": None,
        "message": f"Standard settings for {domain} — change them if your provider uses others.",
    }


class _Net:
    """The default network steps, looked up at call time (tests replace them)."""

    http_get: HttpGet = staticmethod(https_get)  # type: ignore[assignment]
    resolve_srv: ResolveSrv = staticmethod(resolve_srv_default)  # type: ignore[assignment]
    resolve_mx: ResolveMx = staticmethod(resolve_mx_default)  # type: ignore[assignment]


_net = _Net()

__all__ = [
    "EmailDiscoveryFailed",
    "KNOWN_DOMAINS",
    "MX_SUFFIXES",
    "discover_servers",
    "parse_autoconfig",
    "parse_xml",
    "require_servers",
    "server_defaults",
]
