"""Core listener lifecycle and auth are independent of framework run execution."""
import os
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
