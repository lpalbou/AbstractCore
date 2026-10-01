"""Mail server auto-discovery (`abstractcore.comms.email.discovery`) and its use by
`/acore/email` and `abstractcore email connect`.

Hermetic: every network step is injected (the conftest refuses the defaults and fails a test
that reaches them). The order is the contract: known table, the domain's autoconfig (two URLs),
the ISPDB, SRV, then MX; the first step that yields BOTH servers wins.
"""

from __future__ import annotations

import json
import socket
import struct
from typing import Any, Dict, List, Optional, Tuple

import pytest

from abstractcore.comms.email import discovery
from abstractcore.comms.email.discovery import EmailDiscoveryFailed, discover_servers, parse_autoconfig, require_servers

pytestmark = pytest.mark.basic


def autoconfig_xml(
    imap: Tuple[str, int, str] = ("imap.%EMAILDOMAIN%", 993, "SSL"),
    smtp: Tuple[str, int, str] = ("smtp.%EMAILDOMAIN%", 465, "SSL"),
    username: str = "%EMAILADDRESS%",
    extra_in: str = "",
) -> bytes:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<clientConfig version="1.1">
  <emailProvider id="corp.test">
    <domain>corp.test</domain>
    {extra_in}
    <incomingServer type="imap">
      <hostname>{imap[0]}</hostname><port>{imap[1]}</port><socketType>{imap[2]}</socketType>
      <username>{username}</username><authentication>password-cleartext</authentication>
    </incomingServer>
    <outgoingServer type="smtp">
      <hostname>{smtp[0]}</hostname><port>{smtp[1]}</port><socketType>{smtp[2]}</socketType>
      <username>{username}</username><authentication>password-cleartext</authentication>
    </outgoingServer>
  </emailProvider>
</clientConfig>""".encode()


class FakeNet:
    """Scripted lookups; records every call in order."""

    def __init__(self, http: Optional[Dict[str, Any]] = None, srv: Optional[Dict[str, Any]] = None, mx: Any = None) -> None:
        self.http = http or {}
        self.srv = srv or {}
        self.mx = mx if mx is not None else []
        self.calls: List[Tuple[str, str]] = []

    def http_get(self, url: str, timeout: float) -> Optional[bytes]:
        self.calls.append(("http", url))
        for prefix, value in self.http.items():
            if url.startswith(prefix):
                if isinstance(value, BaseException):
                    raise value
                return value
        return None

    def resolve_srv(self, name: str, timeout: float):
        self.calls.append(("srv", name))
        value = self.srv.get(name, [])
        if isinstance(value, BaseException):
            raise value
        return value

    def resolve_mx(self, domain: str, timeout: float):
        self.calls.append(("mx", domain))
        if isinstance(self.mx, BaseException):
            raise self.mx
        return self.mx

    def kw(self) -> Dict[str, Any]:
        return {"http_get": self.http_get, "resolve_srv": self.resolve_srv, "resolve_mx": self.resolve_mx}


AUTOCONFIG = "https://autoconfig.corp.test/mail/config-v1.1.xml?emailaddress=me@corp.test"
WELL_KNOWN = "https://corp.test/.well-known/autoconfig/mail/config-v1.1.xml"
ISPDB = "https://autoconfig.thunderbird.net/v1.1/corp.test"


def steps(found: Dict[str, Any]) -> List[str]:
    return [f"{row['step']}:{row['result']}" for row in found["tried"]]


# ------------------------------------------------------------------ the order


def test_known_provider_needs_no_lookup_and_carries_the_provider() -> None:
    net = FakeNet()
    found = discover_servers("Me@GMail.com", **net.kw())
    assert net.calls == []
    assert found["found"] is True and found["source"] == "known" and found["provider"] == "google"
    assert found["imap"] == {"host": "imap.gmail.com", "port": 993, "security": "ssl"}
    assert found["smtp"] == {"host": "smtp.gmail.com", "port": 465, "security": "ssl"}
    assert found["username"] == "Me@GMail.com" and found["domain"] == "gmail.com"
    ms = discover_servers("someone@hotmail.com", **net.kw())
    assert ms["provider"] == "microsoft" and ms["smtp"] == {"host": "smtp-mail.outlook.com", "port": 587, "security": "starttls"}
    icloud = discover_servers("someone@icloud.com", **net.kw())
    assert icloud["provider"] is None and icloud["imap"]["host"] == "imap.mail.me.com"
    # The local-part user name form (ISPDB %EMAILLOCALPART%).
    assert discover_servers("jean.dupont@free.fr", **net.kw())["username"] == "jean.dupont"


def test_the_domains_own_autoconfig_wins_before_the_ispdb() -> None:
    net = FakeNet(http={AUTOCONFIG: autoconfig_xml(), ISPDB: autoconfig_xml(imap=("wrong.test", 993, "SSL"))})
    found = discover_servers("me@corp.test", **net.kw())
    assert found["source"] == "autoconfig" and found["imap"]["host"] == "imap.corp.test"
    assert net.calls == [("http", AUTOCONFIG)]
    assert steps(found) == ["known:no match", "autoconfig:found"]


def test_well_known_autoconfig_then_ispdb_then_srv_then_mx_in_that_order() -> None:
    net = FakeNet(mx=[(10, "mx.nowhere.test")])
    found = discover_servers("me@corp.test", **net.kw())
    assert found["found"] is False and found["source"] is None and found["imap"] is None and found["smtp"] is None
    assert net.calls == [
        ("http", AUTOCONFIG),
        ("http", WELL_KNOWN),
        ("http", ISPDB),
        ("srv", "_imaps._tcp.corp.test"),
        ("srv", "_submissions._tcp.corp.test"),
        ("srv", "_submission._tcp.corp.test"),
        ("mx", "corp.test"),
    ]
    assert steps(found) == [
        "known:no match",
        "autoconfig:not found",
        "autoconfig:not found",
        "ispdb:not found",
        "srv:not found",
        "srv:not found",
        "srv:not found",
        "mx:no known provider",
    ]


def test_ispdb_answers_when_the_domain_has_no_autoconfig() -> None:
    net = FakeNet(http={AUTOCONFIG: OSError("refused"), ISPDB: autoconfig_xml(username="%EMAILLOCALPART%")})
    found = discover_servers("me@corp.test", **net.kw())
    assert found["source"] == "ispdb" and found["username"] == "me"
    assert steps(found)[:4] == ["known:no match", "autoconfig:error: OSError", "autoconfig:not found", "ispdb:found"]


def test_srv_records_prefer_implicit_tls_and_priority_and_ignore_dot_targets() -> None:
    net = FakeNet(
        srv={
            "_imaps._tcp.corp.test": [(10, 0, 993, "imap2.corp.test."), (0, 5, 993, "imap1.corp.test.")],
            "_submissions._tcp.corp.test": [(0, 0, 0, ".")],
            "_submission._tcp.corp.test": [(0, 0, 587, "smtp.corp.test.")],
        }
    )
    found = discover_servers("me@corp.test", **net.kw())
    assert found["source"] == "srv"
    assert found["imap"] == {"host": "imap1.corp.test", "port": 993, "security": "ssl"}
    assert found["smtp"] == {"host": "smtp.corp.test", "port": 587, "security": "starttls"}
    assert ("mx", "corp.test") not in net.calls

    both = FakeNet(
        srv={
            "_imaps._tcp.corp.test": [(0, 0, 993, "imap.corp.test")],
            "_submissions._tcp.corp.test": [(0, 0, 465, "smtp.corp.test")],
            "_submission._tcp.corp.test": [(0, 0, 587, "other.corp.test")],
        }
    )
    found = discover_servers("me@corp.test", **both.kw())
    assert found["smtp"] == {"host": "smtp.corp.test", "port": 465, "security": "ssl"}
    assert ("srv", "_submission._tcp.corp.test") not in both.calls  # implicit TLS found first


def test_mx_maps_hosted_domains_to_google_and_microsoft() -> None:
    google = discover_servers("me@corp.test", **FakeNet(mx=[(5, "alt1.aspmx.l.google.com."), (1, "aspmx.l.google.com.")]).kw())
    assert google["source"] == "mx" and google["provider"] == "google" and google["imap"]["host"] == "imap.gmail.com"
    ms = discover_servers("me@corp.test", **FakeNet(mx=[(0, "corp-test.mail.protection.outlook.com")]).kw())
    assert ms["source"] == "mx" and ms["provider"] == "microsoft"
    assert ms["smtp"] == {"host": "smtp.office365.com", "port": 587, "security": "starttls"}
    # A suffix match on DNS labels, never a substring.
    evil = discover_servers("me@corp.test", **FakeNet(mx=[(0, "mail.notgoogle.com")]).kw())
    assert evil["found"] is False


def test_autoconfig_naming_the_google_imap_host_carries_the_provider() -> None:
    net = FakeNet(http={AUTOCONFIG: autoconfig_xml(imap=("imap.gmail.com", 993, "SSL"), smtp=("smtp.gmail.com", 465, "SSL"))})
    assert discover_servers("me@corp.test", **net.kw())["provider"] == "google"


def test_require_servers_raises_the_typed_error_with_what_was_tried() -> None:
    with pytest.raises(EmailDiscoveryFailed) as info:
        require_servers("me@corp.test", **FakeNet().kw())
    err = info.value
    assert err.code == "email_discovery_failed"
    assert err.message == "Couldn't find the mail servers for corp.test. Standard settings are filled in: check them and change any your provider does differently."
    assert err.details["domain"] == "corp.test" and err.details["tried"][0] == {"step": "known", "result": "no match"}


def test_not_an_address_is_a_value_error_and_ip_literals_are_not_looked_up() -> None:
    with pytest.raises(ValueError):
        discover_servers("not-an-address", **FakeNet().kw())
    net = FakeNet()
    with pytest.raises(ValueError):
        discover_servers("me@localhost", **net.kw())  # a single label is not a mail domain
    found = discover_servers("me@10.0.0.1", **net.kw())
    assert found["found"] is False and net.calls == []


# ------------------------------------------------------------------ parsing and safety


def test_plain_servers_are_skipped_and_placeholders_substituted() -> None:
    doc = autoconfig_xml(
        extra_in='<incomingServer type="imap"><hostname>plain.corp.test</hostname><port>143</port><socketType>plain</socketType></incomingServer>',
    )
    got = parse_autoconfig(doc, "me@corp.test", "corp.test")
    assert got["imap"] == {"host": "imap.corp.test", "port": 993, "security": "ssl"}
    assert got["username"] == "me@corp.test"


def test_differing_user_name_forms_keep_the_full_address() -> None:
    doc = autoconfig_xml().replace(b"<username>%EMAILADDRESS%</username>", b"<username>%EMAILLOCALPART%</username>", 1)
    assert parse_autoconfig(doc, "me@corp.test", "corp.test")["username"] == "me@corp.test"


def test_dtd_and_entities_are_refused_and_the_next_step_runs() -> None:
    bomb = (
        b'<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;&lol;">]>'
        b"<clientConfig><emailProvider><incomingServer type='imap'><hostname>&lol2;</hostname></incomingServer></emailProvider></clientConfig>"
    )
    net = FakeNet(http={AUTOCONFIG: bomb, ISPDB: autoconfig_xml()})
    found = discover_servers("me@corp.test", **net.kw())
    assert "autoconfig:error: unsafe XML refused" in steps(found)
    assert found["source"] == "ispdb"
    with pytest.raises(ValueError):
        parse_autoconfig(b"<html>not a config</html>", "me@corp.test", "corp.test")


def test_expat_fallback_refuses_dtds_when_defusedxml_is_missing(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def no_defusedxml(name, *args, **kwargs):
        if name.startswith("defusedxml"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_defusedxml)
    with pytest.raises(discovery._UnsafeXml):
        discovery.parse_xml(b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "b">]><x>&a;</x>')
    assert discovery.parse_xml(b"<clientConfig><a>1</a></clientConfig>").tag == "clientConfig"


def test_a_partial_answer_prefills_but_is_not_found() -> None:
    net = FakeNet(srv={"_imaps._tcp.corp.test": [(0, 0, 993, "imap.corp.test")]})
    found = discover_servers("me@corp.test", **net.kw())
    assert found["found"] is False and found["source"] is None
    assert found["imap"] == {"host": "imap.corp.test", "port": 993, "security": "ssl"} and found["smtp"] is None


def test_the_time_budget_skips_the_remaining_steps(monkeypatch) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr(discovery.time, "monotonic", lambda: clock["t"])

    def slow_get(url, timeout):
        clock["t"] += 100.0
        return None

    net = FakeNet()
    found = discover_servers("me@corp.test", timeout=1.0, http_get=slow_get, resolve_srv=net.resolve_srv, resolve_mx=net.resolve_mx)
    assert found["found"] is False
    assert all("time budget" in r for r in steps(found)[2:])
    assert net.calls == []


def test_https_get_refuses_plain_http_and_non_public_hosts(monkeypatch) -> None:
    with pytest.raises(ValueError):
        discovery.https_get("http://corp.test/x", 1.0)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443))])
    with pytest.raises(OSError, match="non-public"):
        discovery.https_get("https://autoconfig.corp.test/mail/config-v1.1.xml", 1.0)
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 443))])
    with pytest.raises(OSError, match="non-public"):
        discovery.https_get("https://corp.test/.well-known/autoconfig/mail/config-v1.1.xml", 1.0)


def _name(n: str) -> bytes:
    return b"".join(bytes([len(p)]) + p.encode() for p in n.split(".")) + b"\x00"


def test_stdlib_dns_parser_reads_srv_and_mx_with_compression() -> None:
    qname = _name("_imaps._tcp.corp.test")
    header = struct.pack("!HHHHHH", 77, 0x8180, 1, 2, 0, 0)
    question = qname + struct.pack("!HH", 33, 1)
    target = _name("imap.corp.test")
    rr1 = b"\xc0\x0c" + struct.pack("!HHIH", 33, 1, 300, 6 + len(target)) + struct.pack("!HHH", 0, 5, 993) + target
    # second target compressed: "imap2" + pointer to "corp.test" inside the first target
    first_target_at = 12 + len(question) + 2 + 10 + 6
    corp_at = first_target_at + 1 + len("imap")
    t2 = b"\x05imap2" + struct.pack("!H", 0xC000 | corp_at)
    rr2 = b"\xc0\x0c" + struct.pack("!HHIH", 33, 1, 300, 6 + len(t2)) + struct.pack("!HHH", 10, 0, 993) + t2
    msg = header + question + rr1 + rr2
    assert discovery.parse_dns_response(msg, 77, 33) == [(0, 5, 993, "imap.corp.test"), (10, 0, 993, "imap2.corp.test")]

    mxq = _name("corp.test") + struct.pack("!HH", 15, 1)
    host = _name("aspmx.l.google.com")
    mx = struct.pack("!HHHHHH", 9, 0x8180, 1, 1, 0, 0) + mxq + b"\xc0\x0c" + struct.pack("!HHIH", 15, 1, 60, 2 + len(host)) + struct.pack("!H", 1) + host
    assert discovery.parse_dns_response(mx, 9, 15) == [(1, "aspmx.l.google.com")]
    nx = struct.pack("!HHHHHH", 9, 0x8183, 1, 0, 0, 0) + mxq
    assert discovery.parse_dns_response(nx, 9, 15) == []
    with pytest.raises(ValueError):
        discovery.parse_dns_response(mx, 10, 15)  # id mismatch
    loop = struct.pack("!HHHHHH", 9, 0x8180, 1, 1, 0, 0) + mxq + b"\xc0\x0c" + struct.pack("!HHIH", 15, 1, 60, 4) + struct.pack("!H", 1) + struct.pack("!H", 0xC000 | (12 + len(mxq) + 12 + 2))
    with pytest.raises(ValueError):
        discovery.parse_dns_response(loop, 9, 15)


def test_known_table_values_are_encrypted_and_well_formed() -> None:
    for domain, key in discovery.KNOWN_DOMAINS.items():
        spec = discovery._PROVIDERS[key]
        for leg in ("imap", "smtp"):
            assert spec[leg]["security"] in ("ssl", "starttls"), domain
            assert 1 <= spec[leg]["port"] <= 65535 and "." in spec[leg]["host"], domain
        assert spec["provider"] in (None, "google", "microsoft")


# ------------------------------------------------------------------ /acore/email and the CLI


@pytest.fixture
def http(config_file, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from abstractcore.server import app as server_app
    from abstractcore.server.email_routes import router

    monkeypatch.setattr(server_app, "_server_auth_enabled", lambda: False)
    monkeypatch.setattr(server_app, "_server_allows_unauthenticated", lambda: True)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _use_net(monkeypatch, net: FakeNet) -> None:
    monkeypatch.setattr(discovery._net, "http_get", net.http_get)
    monkeypatch.setattr(discovery._net, "resolve_srv", net.resolve_srv)
    monkeypatch.setattr(discovery._net, "resolve_mx", net.resolve_mx)


def test_http_discover_route(http, monkeypatch) -> None:
    _use_net(monkeypatch, FakeNet(http={AUTOCONFIG: autoconfig_xml()}))
    r = http.post("/acore/email/discover", json={"address": "me@corp.test"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["found"] is True and body["source"] == "autoconfig" and body["imap"]["host"] == "imap.corp.test"
    assert set(body) == {"address", "domain", "found", "source", "provider", "imap", "smtp", "username", "tried", "defaults"}
    assert body["defaults"] == {
        "imap": {"host": "imap.corp.test", "port": 993, "security": "ssl"},
        "smtp": {"host": "smtp.corp.test", "port": 465, "security": "ssl"},
        "login": "me@corp.test", "source": "discovered", "provider": None, "message": "Settings found for corp.test.",
    }
    _use_net(monkeypatch, FakeNet())
    std = http.post("/acore/email/discover", json={"address": "me@corp.test"}).json()
    assert std["found"] is False and std["defaults"]["source"] == "standard"
    assert std["defaults"]["smtp"] == {"host": "smtp.corp.test", "port": 465, "security": "ssl"}
    bad = http.post("/acore/email/discover", json={"address": "nope"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "email_invalid_settings"


def test_http_connect_without_servers_discovers_them(http, monkeypatch) -> None:
    net = FakeNet(http={AUTOCONFIG: autoconfig_xml(username="%EMAILLOCALPART%")})
    _use_net(monkeypatch, net)
    r = http.put("/acore/email", json={"address": "me@corp.test", "password": "pw-1", "test": False})
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["imap"]["host"] == "imap.corp.test" and doc["imap"]["port"] == 993 and doc["imap"]["security"] == "ssl"
    assert doc["smtp"]["host"] == "smtp.corp.test" and doc["username"] == "me"
    assert doc["discovery"]["source"] == "autoconfig"
    # An explicit user name wins over the discovered form; given servers are never looked up.
    r = http.put("/acore/email", json={"address": "me@corp.test", "password": "pw-1", "username": "custom", "test": False})
    assert r.json()["username"] == "custom"
    calls = len(net.calls)
    r = http.put("/acore/email", json={"address": "me@corp.test", "password": "pw-1", "test": False, "imap": {"host": "given.corp.test"}})
    assert r.status_code == 200 and r.json()["imap"]["host"] == "given.corp.test" and "discovery" not in r.json()
    assert len(net.calls) == calls


def test_http_connect_discovered_servers_are_the_ones_tested(http, monkeypatch) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        closed = s.getsockname()[1]
    _use_net(monkeypatch, FakeNet(http={AUTOCONFIG: autoconfig_xml(imap=("127.0.0.1", closed, "SSL"), smtp=("127.0.0.1", closed, "SSL"))}))
    r = http.put("/acore/email", json={"address": "me@corp.test", "password": "pw-1"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "email_unreachable"
    assert r.json()["error"]["details"]["port"] == closed


def test_http_connect_discovery_failure_is_a_typed_400(http, monkeypatch) -> None:
    _use_net(monkeypatch, FakeNet())
    r = http.put("/acore/email", json={"address": "me@corp.test", "password": "pw-1"})
    assert r.status_code == 400, r.text
    err = r.json()["error"]
    assert err["code"] == "email_discovery_failed"
    assert err["message"] == "Couldn't find the mail servers for corp.test. Standard settings are filled in: check them and change any your provider does differently."
    assert [row["step"] for row in err["tried"]][:2] == ["known", "autoconfig"]
    assert http.get("/acore/email").json()["configured"] is False


def test_http_status_lists_oauth_providers_and_the_stored_email_address(http) -> None:
    doc = http.get("/acore/email").json()
    assert [p["id"] for p in doc["oauth_providers"]] == ["google", "microsoft"]
    for p in doc["oauth_providers"]:
        assert p["available"] is False and "Advanced" in p["reason"]
    r = http.put("/acore/email/registered-address", json={"address": "me@example.test"})
    assert r.status_code == 200 and r.json()["registered_address"] == "me@example.test"
    assert r.json()["registered_address_stored"] == "me@example.test"
    r = http.put("/acore/email/registered-address", json={"address": ""})
    assert r.json()["registered_address_stored"] == ""


def test_oauth_providers_available_with_a_builtin_or_configured_client(monkeypatch) -> None:
    from abstractcore.comms.email import oauth

    monkeypatch.setitem(oauth.BUILTIN_CLIENTS, "google", {"client_id": "builtin-id"})
    rows = {p["id"]: p for p in oauth.oauth_providers_public({"microsoft": True})}
    assert rows["google"] == {"id": "google", "available": True, "reason": None}
    assert rows["microsoft"]["available"] is True


def test_cli_connect_without_hosts_discovers_and_discover_verb(config_file, monkeypatch, capsys) -> None:
    from abstractcore.config.email_cli import handle_email

    _use_net(monkeypatch, FakeNet(http={AUTOCONFIG: autoconfig_xml()}))
    assert handle_email(["discover", "me@corp.test", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["source"] == "autoconfig"
    # The same `defaults` as POST /acore/email/discover (the core TUI reads it from here).
    assert out["defaults"] == discovery.server_defaults("me@corp.test", {k: v for k, v in out.items() if k != "defaults"})
    assert out["defaults"]["source"] == "discovered" and out["defaults"]["message"] == "Settings found for corp.test."
    assert handle_email(["connect", "--address", "me@corp.test", "--password", "pw-1", "--no-test", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["imap"]["host"] == "imap.corp.test" and doc["smtp"]["port"] == 465

    _use_net(monkeypatch, FakeNet())
    assert handle_email(["connect", "--address", "me@corp.test", "--password", "pw-1", "--no-test", "--json"]) == 1
    err = json.loads(capsys.readouterr().out)["error"]
    assert err["code"] == "email_discovery_failed" and "--imap-host" in err["fix"]
    assert handle_email(["discover", "me@corp.test"]) == 1
    assert "Couldn't find the mail servers for corp.test" in capsys.readouterr().out


# ------------------------------------------------------------------ server_defaults (what a form pre-fills)


def test_server_defaults_uses_what_discovery_found_including_starttls_and_the_login_form() -> None:
    net = FakeNet()
    ms = discovery.server_defaults("someone@hotmail.com", discover_servers("someone@hotmail.com", **net.kw()))
    assert ms["source"] == "discovered" and ms["provider"] == "microsoft"
    assert ms["smtp"] == {"host": "smtp-mail.outlook.com", "port": 587, "security": "starttls"}
    assert ms["login"] == "someone@hotmail.com" and ms["message"] == "Settings found for hotmail.com."
    # The local-part login form resolves to the actual string.
    free = discovery.server_defaults("jean.dupont@free.fr", discover_servers("jean.dupont@free.fr", **net.kw()))
    assert free["login"] == "jean.dupont" and free["source"] == "discovered"
    assert net.calls == []


def test_server_defaults_falls_back_to_standard_servers_and_keeps_a_partial_leg() -> None:
    nothing = discover_servers("me@corp.test", **FakeNet().kw())
    std = discovery.server_defaults("me@corp.test", nothing)
    assert std == {
        "imap": {"host": "imap.corp.test", "port": 993, "security": "ssl"},
        "smtp": {"host": "smtp.corp.test", "port": 465, "security": "ssl"},
        "login": "me@corp.test", "source": "standard", "provider": None,
        "message": "Standard settings for corp.test — change them if your provider uses others.",
    }
    partial = discover_servers("me@corp.test", **FakeNet(srv={"_imaps._tcp.corp.test": [(0, 0, 1993, "mail.corp.test")]}).kw())
    got = discovery.server_defaults("me@corp.test", partial)
    assert got["source"] == "standard" and got["imap"] == {"host": "mail.corp.test", "port": 1993, "security": "ssl"}
    assert got["smtp"] == {"host": "smtp.corp.test", "port": 465, "security": "ssl"}


def test_server_defaults_is_pure_when_given_a_result_and_refuses_a_non_address(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise AssertionError("no network when a discovery result is given")

    monkeypatch.setattr(discovery, "discover_servers", boom)
    got = discovery.server_defaults("a@b.example", {"found": False, "imap": None, "smtp": None})
    assert got["imap"]["host"] == "imap.b.example"
    with pytest.raises(ValueError):
        discovery.server_defaults("not-an-address", {"found": False})
