"""Core listener lifecycle and auth are independent of framework run execution."""
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import urllib.error
import urllib.request

import pytest


def _status(url, token=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=5) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def test_import_does_not_load_http_or_framework_packages(tmp_path):
    script = """
import sys
from abstractcore.server.managed import start_managed_server
assert 'abstractcore.server.app' not in sys.modules
assert 'uvicorn' not in sys.modules
assert not any(n == 'abstractgateway' or n.startswith('abstractgateway.') for n in sys.modules)
assert not any(n == 'abstractruntime' or n.startswith('abstractruntime.') for n in sys.modules)
"""
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_listener_uses_core_auth_and_stops_without_model_cleanup(monkeypatch):
    from abstractcore.server.managed import start_managed_server

    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", "core-lifecycle-test-token")
    monkeypatch.setenv("ABSTRACTGATEWAY_AUTH_TOKEN", "gateway-lifecycle-test-token")
    server = start_managed_server(host="127.0.0.1", port=0)
    try:
        assert server.is_running
        assert server.status()["running"] is True
        assert _status(server.url + "/health") == 200
        protected = server.url + "/managed-lifecycle-missing-route"
        assert _status(protected) == 401
        assert _status(protected, "gateway-lifecycle-test-token") == 401
        assert _status(protected, "core-lifecycle-test-token") == 404
    finally:
        server.stop()
    assert not server.is_running
    server.stop()
    # A fresh listener can start while its host remains alive.
    restarted = start_managed_server(host="127.0.0.1", port=0)
    try:
        assert _status(restarted.url + "/health") == 200
    finally:
        restarted.stop()


def test_loopback_first_run_reuses_existing_core_token_file(monkeypatch):
    from abstractcore.server.first_run import load_or_create_token
    from abstractcore.server.managed import start_managed_server

    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", raising=False)
    token, _ = load_or_create_token()
    server = start_managed_server(host="127.0.0.1", port=0)
    try:
        assert _status(server.url + "/managed-lifecycle-missing-route", token) == 404
        assert _status(server.url + "/managed-lifecycle-missing-route") == 401
    finally:
        server.stop()


def test_failed_bind_does_not_import_http_application(monkeypatch):
    import socket
    from abstractcore.server.managed import start_managed_server

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        with pytest.raises(OSError):
            start_managed_server(host="127.0.0.1", port=busy.getsockname()[1])


@pytest.mark.parametrize("port", [True, -1, 65536, "8000"])
def test_invalid_port_is_refused_before_serving(port):
    from abstractcore.server.managed import start_managed_server

    with pytest.raises(ValueError, match="port"):
        start_managed_server(host="127.0.0.1", port=port)


@pytest.mark.parametrize("host", ["", "[]", "   "])
def test_empty_host_is_refused_before_binding(host):
    from abstractcore.server.managed import start_managed_server

    with pytest.raises(ValueError, match="host"):
        start_managed_server(host=host, port=0)


def test_each_listener_uses_its_actual_bind_without_changing_legacy_policy(monkeypatch):
    import asyncio
    from abstractcore.server.app import app
    from abstractcore.server.host_routes import server_allows_engine_install
    from abstractcore.server.managed import start_managed_server

    token = "core-listener-policy-test-token"
    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", token)
    monkeypatch.setenv("ABSTRACTCORE_SERVER_BIND_HOST", "127.0.0.1")
    monkeypatch.delenv("ABSTRACTCORE_ALLOW_ENGINE_INSTALL", raising=False)
    routes = list(app.router.routes)

    async def policy():
        # Overlap requests on both listeners to expose process-global policy.
        await asyncio.sleep(0.01)
        return {"allowed": server_allows_engine_install()}

    app.add_api_route("/managed-listener-policy", policy, methods=["GET"])
    local = remote = None
    try:
        local = start_managed_server(host="127.0.0.1", port=0)
        remote = start_managed_server(host="0.0.0.0", port=0)
        local_url = local.url + "/managed-listener-policy"
        remote_url = remote.url.replace("0.0.0.0", "127.0.0.1") + "/managed-listener-policy"

        def allowed(url):
            req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
            with urllib.request.urlopen(req, timeout=5) as response:
                return json.load(response)["allowed"]

        with ThreadPoolExecutor(max_workers=8) as pool:
            probes = [(local_url, True), (remote_url, False)] * 8
            results = list(pool.map(allowed, [url for url, _ in probes]))
        assert results == [expected for _, expected in probes]
        assert os.environ["ABSTRACTCORE_SERVER_BIND_HOST"] == "127.0.0.1"
        assert server_allows_engine_install() is True  # legacy env fallback untouched
        for setting, expected in [("1", True), ("0", False)]:
            monkeypatch.setenv("ABSTRACTCORE_ALLOW_ENGINE_INSTALL", setting)
            assert allowed(local_url) is expected and allowed(remote_url) is expected
    finally:
        if remote is not None:
            remote.stop()
        if local is not None:
            local.stop()
        app.router.routes[:] = routes


@pytest.mark.parametrize("timeout", ["bad", None, True, -1, float("nan"), float("inf")])
def test_invalid_timeout_is_refused_before_binding(monkeypatch, timeout):
    import abstractcore.server.managed as managed

    def unexpected_socket(*_args, **_kwargs):
        pytest.fail("invalid startup deadline allocated a socket")

    monkeypatch.setattr(managed.socket, "socket", unexpected_socket)
    with pytest.raises(ValueError, match="timeout"):
        managed.start_managed_server(host="127.0.0.1", port=0, timeout_s=timeout)


def test_failure_after_thread_start_stops_the_listener(monkeypatch):
    from types import SimpleNamespace
    import abstractcore.server.managed as managed
    import uvicorn

    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", "core-startup-cleanup-test-token")
    servers = []

    class Server:
        def __init__(self, _config):
            self.started = False
            self.should_exit = False
            self.force_exit = False
            self.finished = threading.Event()
            servers.append(self)

        def run(self, **_kwargs):
            self.started = True
            while not (self.should_exit or self.force_exit):
                self.finished.wait(0.01)
            self.finished.set()

    monkeypatch.setattr(uvicorn, "Server", Server)

    def deadline_failure():
        raise RuntimeError("deadline clock unavailable")

    monkeypatch.setattr(managed, "time", SimpleNamespace(monotonic=deadline_failure))
    try:
        with pytest.raises(RuntimeError, match="deadline clock unavailable"):
            managed.start_managed_server(host="127.0.0.1", port=0)
        assert servers[0].should_exit and servers[0].finished.wait(1)
    finally:
        for server in servers:
            server.should_exit = True
