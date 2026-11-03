"""Explicit lifecycle for a Core HTTP listener sharing its host's local resources.

Importing this module does not import the HTTP application or start a listener.
Standalone ``abstractcore serve`` retains its own entrypoint and authentication.
"""
from __future__ import annotations

import socket
import threading
import time
import math
from typing import Any, Optional


def _timeout_seconds(value: Any) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Core HTTP timeout must be a finite non-negative number") from exc
    if isinstance(value, bool) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Core HTTP timeout must be a finite non-negative number")
    return seconds


class CoreHTTPServer:
    """An explicitly started HTTP listener; stopping it leaves model resources resident."""

    def __init__(self, server: Any, sock: socket.socket, host: str) -> None:
        self._server = server
        self._socket = sock
        self.last_error: Optional[str] = None
        port = int(sock.getsockname()[1])
        address = f"[{host}]" if ":" in host else host
        self.url = f"http://{address}:{port}"
        self._thread = threading.Thread(target=self._run, name="abstractcore-http", daemon=True)

    def _run(self) -> None:
        try:
            self._server.run(sockets=[self._socket])
        except BaseException as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            self._socket.close()

    @property
    def is_running(self) -> bool:
        return self._thread.is_alive() and bool(self._server.started) and not self._server.should_exit

    def status(self) -> dict[str, Any]:
        return {"running": self.is_running, "url": self.url, "error": self.last_error}

    def stop(self, *, timeout_s: float = 6.0) -> None:
        """Stop HTTP serving only. Repeated calls are safe."""
        timeout_s = _timeout_seconds(timeout_s)
        self._server.should_exit = True
        self._thread.join(timeout=timeout_s)
        if self._thread.is_alive():
            self._server.force_exit = True
            self._thread.join(timeout=1.0)
        if self._thread.is_alive():
            raise RuntimeError("Core HTTP listener has not stopped; model resources remain in use")


def start_managed_server(
    *, host: Optional[str] = None, port: Optional[int] = None, timeout_s: float = 15.0,
) -> CoreHTTPServer:
    """Start Core's own authenticated app, using its persisted server bind by default.

    The default bind is 127.0.0.1:8000. Explicit port 0 selects an ephemeral
    port. Core's existing first-run token handling and server policy apply.
    This function creates no provider or model of its own.
    """
    timeout_s = _timeout_seconds(timeout_s)
    from ..config.manager import get_config_manager

    configured = get_config_manager().config.server
    bind_host = host if host is not None else (configured.host or "127.0.0.1")
    bind_port = port if port is not None else (configured.port if configured.port is not None else 8000)
    if not isinstance(bind_host, str) or not bind_host.strip():
        raise ValueError("Core server host must be a non-empty string")
    if isinstance(bind_port, bool) or not isinstance(bind_port, int) or not 0 <= bind_port <= 65535:
        raise ValueError("Core server port must be an integer between 0 and 65535")
    bind_host = bind_host.strip().strip("[]")
    if not bind_host:
        raise ValueError("Core server host must be a non-empty string")
    sock = socket.socket(socket.AF_INET6 if ":" in bind_host else socket.AF_INET, socket.SOCK_STREAM)
    handle = None
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_host, bind_port))
        sock.listen(128)
        sock.set_inheritable(False)

        from .first_run import prepare_server_auth

        prepare_server_auth(bind_host)
        from .app import app
        from .host_routes import use_server_bind_host
        import uvicorn

        async def listener_app(scope, receive, send):
            with use_server_bind_host(bind_host):
                await app(scope, receive, send)

        server = uvicorn.Server(uvicorn.Config(
            listener_app, lifespan="off", log_level="warning", access_log=False,
            timeout_graceful_shutdown=5,
        ))
        handle = CoreHTTPServer(server, sock, bind_host)
        handle._thread.start()
        deadline = time.monotonic() + timeout_s
        while not handle.is_running:
            if not handle._thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError(handle.last_error or "Core HTTP listener did not become ready")
            time.sleep(0.01)
        return handle
    except BaseException:
        try:
            if handle is not None and handle._thread.ident is not None:
                handle.stop()
        finally:
            sock.close()
        raise
