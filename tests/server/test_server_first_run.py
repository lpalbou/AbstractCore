"""First run of `abstractcore serve` on loopback: generated token, one-time
console claim, and `POST /acore/session/claim` (loopback peers only)."""

from __future__ import annotations

import json
import os
import re
import socket
import stat
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from abstractcore.server import app as server_app
from abstractcore.server import first_run

REPO_ROOT = Path(__file__).resolve().parents[2]
CLAIM = "/acore/session/claim"
LOOPBACK_HOST = {"Host": "127.0.0.1:8000"}


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_CONFIG_DIR", str(tmp_path / "config"))
    # set-then-delete so monkeypatch restores (removes) what serve writes.
    for key in ("ABSTRACTCORE_CONFIG_FILE", "ABSTRACTCORE_AUTH_TOKEN",
                "ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", first_run.TOKEN_SOURCE_ENV):
        monkeypatch.setenv(key, "x")
        monkeypatch.delenv(key)
    monkeypatch.setenv("ABSTRACTCORE_JOBS_PERSIST", "0")
    return tmp_path / "config"


# --------------------------------------------------------------- unit level


@pytest.mark.parametrize(
    "host,expected",
    [("127.0.0.1", True), ("localhost", True), ("::1", True), ("[::1]", True), ("127.0.0.5", True),
     ("0.0.0.0", False), ("::", False), ("192.168.1.10", False), ("example.com", False), ("", False)],
)
def test_loopback_detection(host, expected):
    assert first_run.is_loopback_host(host) is expected


def test_token_is_created_once_with_private_mode(cfg):
    token, created = first_run.load_or_create_token(cfg)
    assert created and len(token) >= 32
    path = cfg / "server-token"
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    again, created_again = first_run.load_or_create_token(cfg)
    assert (again, created_again) == (token, False)
    # A hand-written fixed token is honoured; loose permissions are tightened.
    path.write_text("my-fixed-token\n")
    if os.name == "posix":
        os.chmod(path, 0o644)
    assert first_run.load_or_create_token(cfg) == ("my-fixed-token", False)
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    path.write_text("   \n")
    with pytest.raises(RuntimeError):
        first_run.load_or_create_token(cfg)


def test_claim_is_single_use_hashed_and_expires(cfg):
    code = first_run.mint_claim(cfg)
    files = list((cfg / "claims").glob("*.json"))
    assert len(files) == 1 and code not in files[0].name and code not in files[0].read_text()
    assert first_run.redeem_claim(code, cfg).ok
    assert first_run.redeem_claim(code, cfg).reason == "used_or_unknown"
    assert not list((cfg / "claims").iterdir())

    old = first_run.mint_claim(cfg, now=time.time() - 3600)
    assert first_run.redeem_claim(old, cfg).reason == "expired"
    assert first_run.redeem_claim("short", cfg).reason == "invalid"
    assert first_run.redeem_claim("x" * 40, cfg).reason == "used_or_unknown"
    # Expired records are pruned when the next code is minted.
    first_run.mint_claim(cfg, now=time.time() - 3600)
    first_run.mint_claim(cfg)
    assert len(list((cfg / "claims").glob("*.json"))) == 1


@pytest.mark.parametrize("bind", ["127.0.0.1", "localhost", "::1"])
def test_prepare_on_loopback_loads_the_token(cfg, monkeypatch, bind):
    state = first_run.prepare_server_auth(bind)
    assert state.active and state.token_created and state.claim_code
    assert os.environ["ABSTRACTCORE_AUTH_TOKEN"] == (cfg / "server-token").read_text().strip()
    assert first_run.claims_enabled()
    assert server_app._server_auth_enabled()


@pytest.mark.parametrize("bind", ["0.0.0.0", "192.168.1.10", "::"])
def test_prepare_on_non_loopback_changes_nothing(cfg, bind):
    state = first_run.prepare_server_auth(bind)
    assert not state.active
    assert "ABSTRACTCORE_AUTH_TOKEN" not in os.environ
    assert not (cfg / "server-token").exists()


def test_explicit_auth_choices_win(cfg, monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", "explicit")
    assert not first_run.prepare_server_auth("127.0.0.1").active
    assert not first_run.claims_enabled()
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN")
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    assert not first_run.prepare_server_auth("127.0.0.1").active
    assert not (cfg / "server-token").exists()


def test_banner_names_dir_url_claim_and_cli_twins(cfg):
    import io

    state = first_run.prepare_server_auth("127.0.0.1")
    out = io.StringIO()
    first_run.print_banner(state, "127.0.0.1", 8123, out=out)
    text = out.getvalue()
    assert str(cfg) in text and "http://127.0.0.1:8123" in text
    assert f"http://127.0.0.1:8123/console#claim={state.claim_code}" in text
    assert "abstractcore serve --print-token" in text and "--claim-url" in text
    assert os.environ["ABSTRACTCORE_AUTH_TOKEN"] not in text  # the secret itself is never printed


# ------------------------------------------------------------- HTTP (in process)


@pytest.fixture
def first_run_app(cfg):
    first_run.prepare_server_auth("127.0.0.1")
    return cfg


def _client(peer: str) -> TestClient:
    return TestClient(server_app.app, client=(peer, 50000))


def test_claim_from_loopback_returns_the_token_once(first_run_app):
    code = first_run.mint_claim(first_run_app)
    c = _client("127.0.0.1")
    res = c.post(CLAIM, json={"code": code}, headers=LOOPBACK_HOST)
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["ok"] and body["token"] == os.environ["ABSTRACTCORE_AUTH_TOKEN"]
    assert res.headers["cache-control"] == "no-store"
    # The token works; nothing works without it.
    assert c.get("/acore/jobs").status_code == 401
    assert c.get("/acore/jobs", headers={"Authorization": f"Bearer {body['token']}"}).status_code == 200
    again = c.post(CLAIM, json={"code": code}, headers=LOOPBACK_HOST)
    assert again.status_code == 403 and again.json()["reason"] == "used_or_unknown"


@pytest.mark.parametrize(
    "header",
    [{"X-Forwarded-For": "127.0.0.1"}, {"X-Forwarded-For": "203.0.113.9"}, {"Forwarded": "for=127.0.0.1"},
     {"X-Real-IP": "127.0.0.1"}],
)
def test_forwarded_requests_are_refused_even_from_loopback(first_run_app, header):
    code = first_run.mint_claim(first_run_app)
    res = _client("127.0.0.1").post(CLAIM, json={"code": code}, headers={**LOOPBACK_HOST, **header})
    assert res.status_code == 403 and res.json()["reason"] == "not_loopback"
    # The refusal did not burn the code.
    assert _client("127.0.0.1").post(CLAIM, json={"code": code}, headers=LOOPBACK_HOST).status_code == 200


def test_remote_peer_and_rebinding_host_are_refused(first_run_app):
    code = first_run.mint_claim(first_run_app)
    assert _client("203.0.113.5").post(CLAIM, json={"code": code}, headers=LOOPBACK_HOST).status_code == 403
    rebound = _client("127.0.0.1").post(CLAIM, json={"code": code}, headers={"Host": "evil.example:8000"})
    assert rebound.status_code == 403 and rebound.json()["reason"] == "not_loopback"
    ok = _client("::1").post(CLAIM, json={"code": code}, headers={"Host": "[::1]:8000"})
    assert ok.status_code == 200


def test_claim_is_disabled_with_an_explicit_token(cfg, monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", "explicit")
    code = first_run.mint_claim(cfg)
    res = _client("127.0.0.1").post(CLAIM, json={"code": code}, headers=LOOPBACK_HOST)
    assert res.status_code == 404 and res.json()["reason"] == "not_enabled"
    assert "explicit" not in res.text


def test_bad_codes_are_refused(first_run_app):
    c = _client("127.0.0.1")
    assert c.post(CLAIM, json={"code": "nope"}, headers=LOOPBACK_HOST).json()["reason"] == "invalid"
    assert c.post(CLAIM, json={}, headers=LOOPBACK_HOST).status_code == 403


def test_console_page_carries_the_claim_redeemer(first_run_app):
    page = _client("127.0.0.1").get("/console").text
    assert "#claim=" in page and "/session/claim" in page and "replaceState" in page


# ----------------------------------------------------- serve --print-token / --claim-url


def _serve(args, env):
    return subprocess.run(
        [sys.executable, "-m", "abstractcore.config.main", "serve", *args],
        env=env, capture_output=True, text=True, timeout=120,
    )


def _env(cfg: Path) -> dict:
    env = dict(os.environ)
    for key in ("ABSTRACTCORE_AUTH_TOKEN", "ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", first_run.TOKEN_SOURCE_ENV,
                "ABSTRACTCORE_CONFIG_FILE", "HOST", "PORT"):
        env.pop(key, None)
    env.update(
        ABSTRACTCORE_CONFIG_DIR=str(cfg),
        ABSTRACTCORE_SERVER_DISABLE_CENTRALIZED_CONFIG="1",
        ABSTRACTCORE_JOBS_PERSIST="0",
        PYTHONPATH=os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]),
    )
    return env


@pytest.mark.slow
def test_cli_twins(cfg):
    env = _env(cfg)
    token = _serve(["--print-token"], env)
    assert token.returncode == 0, token.stderr
    assert token.stdout.strip() == (cfg / "server-token").read_text().strip()
    url = _serve(["--claim-url", "--port", "8123"], env)
    assert url.returncode == 0 and re.fullmatch(r"http://127\.0\.0\.1:8123/console#claim=[A-Za-z0-9_-]+\n", url.stdout)
    remote = _serve(["--claim-url", "--host", "0.0.0.0"], env)
    assert remote.returncode == 2 and "loopback" in remote.stderr
    explicit = _serve(["--claim-url"], dict(env, ABSTRACTCORE_AUTH_TOKEN="explicit"))
    assert explicit.returncode == 2


# ------------------------------------------------------ real server subprocess


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(method: str, url: str, *, token: str | None = None, body: dict | None = None) -> tuple[int, dict | str]:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            raw, status = res.read().decode(), res.status
    except urllib.error.HTTPError as e:
        raw, status = e.read().decode(), e.code
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw


@pytest.mark.slow
def test_bare_loopback_serve_starts_and_its_claim_url_works(cfg):
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "abstractcore.config.main", "serve", "--port", str(port)],
        env=_env(cfg), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    base = f"http://127.0.0.1:{port}"
    lines: list[str] = []
    try:
        claim_url = None
        deadline = time.monotonic() + 90
        while claim_url is None or "use a fixed token" not in lines[-1]:
            line = proc.stdout.readline()
            if not line:
                pytest.fail("server exited before printing its claim URL:\n" + "".join(lines))
            lines.append(line)
            m = re.search(rf"(http://127\.0\.0\.1:{port}/console#claim=[A-Za-z0-9_-]+)", line)
            if m:
                claim_url = m.group(1)
            if time.monotonic() > deadline:
                pytest.fail("no claim URL:\n" + "".join(lines))
        banner = "".join(lines)
        assert str(cfg) in banner and "--print-token" in banner
        while True:
            try:
                if _http("GET", f"{base}/health")[0] == 200:
                    break
            except OSError:
                pass
            assert proc.poll() is None and time.monotonic() < deadline, "server did not come up"
            time.sleep(0.3)

        status, page = _http("GET", f"{base}/console")
        assert status == 200 and "acc-tab-button-catalog" in page
        assert _http("GET", f"{base}/acore/jobs")[0] == 401  # not server_auth_not_configured
        code = claim_url.split("#claim=", 1)[1]
        status, body = _http("POST", f"{base}{CLAIM}", body={"code": code})
        assert status == 200 and body["token"] == (cfg / "server-token").read_text().strip()
        assert _http("GET", f"{base}/acore/jobs", token=body["token"])[0] == 200
        status, again = _http("POST", f"{base}{CLAIM}", body={"code": code})
        assert status == 403 and again["reason"] == "used_or_unknown"
        # A link minted by another process for this server works too.
        fresh = _serve(["--claim-url", "--port", str(port)], _env(cfg)).stdout.strip()
        assert _http("POST", f"{base}{CLAIM}", body={"code": fresh.split('#claim=', 1)[1]})[0] == 200
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
