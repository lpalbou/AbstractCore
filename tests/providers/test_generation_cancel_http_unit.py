"""Pins: host cancel on the HTTP lanes SEVERS the in-flight request.

Before 2026-09-23 the HTTP providers (Ollama, LM Studio, any OpenAI-compatible
server incl. llama.cpp server and vLLM) only saw a cancel "between stream
chunks": a NON-streaming request sat in a blocking socket read for the whole
generation, and a streaming request in PREFILL (no chunk yet) sat there too —
unreachable by the per-chunk check AND by the gateway kill switch (an async
exception lands only when the thread runs Python again). Measured live on LM
Studio 0.4.20: without the sever the client stayed blocked 45 s after the
cancel and the server kept decoding (worker CPU 48 %); with it the call
returns in ~1 ms and the server stops (CPU 0 %).

A local stub server stands in for the upstream: it HOLDS the response (no
bytes for non-streaming / prefill, one chunk every 50 ms for decode) and
records when it observes the client disconnect (EOF on its socket).

Contract under test (`providers/generation_cancel.HttpCancelGuard`):
- the provider's blocked call returns within ~1 s of `event.set()` and raises
  `GenerationCancelledError` (never a transport error, never retried, never
  an "Error:" chunk, never an LM Studio fallback request);
- the SERVER observes the disconnect (that is what makes it stop decoding).
"""

from __future__ import annotations

import json
import select
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List

import pytest

from abstractcore.exceptions import GenerationCancelledError

HOLD_S = 6.0  # the stub's hold; a test that waits this long has FAILED to sever


class _Stub:
    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.disconnects: List[float] = []
        self.mode = "prefill"  # prefill | decode
        self.lock = threading.Lock()


def _make_handler(stub: _Stub):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # silence
            pass

        def _wait_disconnect(self, seconds: float) -> bool:
            deadline = time.monotonic() + seconds
            sock = self.connection
            while time.monotonic() < deadline:
                r, _, _ = select.select([sock], [], [], 0.02)
                if r:
                    try:
                        data = sock.recv(1, socket.MSG_PEEK)
                    except OSError:
                        data = b""
                    if not data:
                        with stub.lock:
                            stub.disconnects.append(time.monotonic())
                        return True
            return False

        def do_GET(self):  # model listing for provider construction
            body = json.dumps({"data": [{"id": "stub-model"}], "models": [{"name": "stub-model"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
            with stub.lock:
                stub.requests.append({"path": self.path, "payload": payload, "at": time.monotonic()})
            streaming = bool(payload.get("stream"))
            if not streaming:
                # Non-streaming: nothing is sent until the "generation" ends.
                if self._wait_disconnect(HOLD_S):
                    return
                body = json.dumps(_final_body(self.path)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self.wfile.flush()
            if stub.mode == "prefill":
                # Headers only, then silence: a long prompt evaluation.
                self._wait_disconnect(HOLD_S)
                return
            deadline = time.monotonic() + HOLD_S
            while time.monotonic() < deadline:
                line = _stream_line(self.path)
                try:
                    self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
                    self.wfile.flush()
                except OSError:
                    with stub.lock:
                        stub.disconnects.append(time.monotonic())
                    return
                if self._wait_disconnect(0.05):
                    return

    return H


def _final_body(path: str) -> Dict[str, Any]:
    if path.startswith("/api/v1/chat"):
        return {"output": [{"type": "message", "content": "done"}], "stats": {}}
    if path.startswith("/api/"):
        return {"message": {"role": "assistant", "content": "done"}, "done": True, "done_reason": "stop"}
    return {"choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}]}


def _stream_line(path: str) -> bytes:
    if path.startswith("/api/v1/chat"):
        return b'event: message.delta\ndata: {"type": "message.delta", "content": "tok "}\n\n'
    if path.startswith("/api/"):
        return b'{"message": {"role": "assistant", "content": "tok "}, "done": false}\n'
    return b'data: {"choices": [{"delta": {"content": "tok "}, "finish_reason": null}]}\n\n'


@pytest.fixture()
def stub_server():
    stub = _Stub()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(stub))
    server.daemon_threads = True
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    stub.url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield stub
    finally:
        server.shutdown()
        server.server_close()


def _provider(kind: str, base: str):
    if kind == "openai-compatible":
        from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

        return OpenAICompatibleProvider(model="stub-model", base_url=f"{base}/v1")
    if kind == "lmstudio":
        from abstractcore.providers.lmstudio_provider import LMStudioProvider

        return LMStudioProvider(model="stub-model", base_url=f"{base}/v1")
    if kind == "ollama":
        from abstractcore.providers.ollama_provider import OllamaProvider

        return OllamaProvider(model="stub-model", base_url=base)
    raise AssertionError(kind)


def _run_cancel(llm, *, stream: bool, cancel_after: float = 0.4, stub: "_Stub" = None, **gen_kwargs) -> Dict[str, Any]:
    ev = threading.Event()
    out: Dict[str, Any] = {"chunks": 0}

    def call():
        try:
            res = llm.generate("hello", stream=stream, cancel_event=ev, max_output_tokens=64, **gen_kwargs)
            if stream:
                for _ in res:
                    out["chunks"] += 1
            out["result"] = "completed"
        except BaseException as e:  # noqa: BLE001
            out["error"] = e
        out["returned"] = time.monotonic()

    th = threading.Thread(target=call, daemon=True)
    th.start()
    if stub is not None:
        # Cancel only once the upstream HOLDS the request (the case under test:
        # a thread blocked on a server that is prefilling / decoding).
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not [r for r in stub.requests if r["path"] != "/v1/models"]:
            time.sleep(0.01)
    time.sleep(cancel_after)
    out["cancel_at"] = time.monotonic()
    ev.set()
    th.join(HOLD_S + 2)
    return out


@pytest.mark.parametrize("kind", ["openai-compatible", "lmstudio", "ollama"])
@pytest.mark.parametrize("stream,mode", [(False, "prefill"), (True, "prefill"), (True, "decode")])
def test_http_lane_cancel_severs_the_request_and_the_server_sees_it(stub_server, kind, stream, mode):
    stub_server.mode = mode
    llm = _provider(kind, stub_server.url)
    assert llm.supports_generation_cancel() is True
    out = _run_cancel(llm, stream=stream, stub=stub_server, thinking="off")
    assert "returned" in out, "the provider call never returned after the cancel"
    returned_after = out["returned"] - out["cancel_at"]
    assert returned_after < 1.0, f"call returned {returned_after:.2f}s after the cancel (not severed)"
    assert isinstance(out.get("error"), GenerationCancelledError), repr(out.get("error") or out.get("result"))
    # The server saw the client go away (what makes a real server stop decoding).
    # (Its handler polls in 20 ms slices: give it up to 1 s to notice.)
    deadline = time.monotonic() + 1.0
    while not stub_server.disconnects and time.monotonic() < deadline:
        time.sleep(0.01)
    assert stub_server.disconnects, "server never observed the disconnect"
    assert stub_server.disconnects[-1] - out["cancel_at"] < 1.0
    # Exactly one upstream request: no retry, no LM Studio fallback request.
    posts = [r for r in stub_server.requests if r["path"] != "/v1/models"]
    assert len(posts) == 1, [r["path"] for r in posts]
    # The event never leaks into a request payload.
    assert "_cancel_event" not in json.dumps(posts[0]["payload"], default=str)


def test_lmstudio_native_rest_lane_cancel_never_falls_back(stub_server):
    """The native /api/v1/chat lane (engaged by `reasoning=`) severs too, and a
    cancel is never turned into a second request on the OpenAI-compatible path."""
    import warnings

    stub_server.mode = "prefill"
    for stream in (False, True):
        llm = _provider("lmstudio", stub_server.url)  # fresh: the fallback warning is once per instance
        stub_server.requests.clear()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = _run_cancel(llm, stream=stream, stub=stub_server, reasoning="off")
        assert isinstance(out.get("error"), GenerationCancelledError), repr(out.get("error"))
        assert out["returned"] - out["cancel_at"] < 1.0
        paths = [r["path"] for r in stub_server.requests]
        assert paths == ["/api/v1/chat"], paths
        fallbacks = [w for w in caught if "using the OpenAI-compatible endpoint instead" in str(w.message)]
        assert not fallbacks, "a host cancel was treated as a native-route failure and fell back"


def test_uncancelled_http_call_is_unchanged(stub_server):
    """No event -> the pooled client, no guard, the normal answer."""
    stub_server.mode = "decode"
    import abstractcore.providers.generation_cancel as gc_mod

    llm = _provider("openai-compatible", stub_server.url)
    t0 = time.monotonic()
    # Non-streaming with no event: the stub answers after its hold.
    orig = gc_mod.HttpCancelGuard
    used = []

    class _Spy(orig):  # type: ignore[misc,valid-type]
        def __init__(self, *a, **k):
            used.append(1)
            super().__init__(*a, **k)

    gc_mod.HttpCancelGuard = _Spy
    try:
        stub_server.requests.clear()
        # Use a streaming call and stop consuming after two chunks.
        res = llm.generate("hello", stream=True, max_output_tokens=8, thinking="off")
        got = [next(iter(res)).content for _ in range(2)]
        res.close()
    finally:
        gc_mod.HttpCancelGuard = orig
    assert got and not used
    assert time.monotonic() - t0 < HOLD_S


def test_guard_severs_a_raw_blocked_read():
    """The primitive itself: a thread blocked in recv() on a socket the guard
    captured returns as soon as the event is set."""
    import httpx

    from abstractcore.providers.generation_cancel import HttpCancelGuard

    stub = _Stub()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(stub))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
    ev = threading.Event()
    out: Dict[str, Any] = {}

    def call():
        with HttpCancelGuard(ev, provider="t", model="m", url=url) as g:
            with g.client(timeout=httpx.Timeout(30.0)) as c:
                try:
                    c.post(url, json={"stream": False}, extensions=g.extensions)
                    out["result"] = "completed"
                except Exception as e:  # noqa: BLE001
                    out["error"] = e
                    out["cancelled"] = g.cancelled_error_from(e)
        out["returned"] = time.monotonic()
        out["severed"] = g.severed_sockets

    th = threading.Thread(target=call, daemon=True)
    th.start()
    time.sleep(0.3)
    t_cancel = time.monotonic()
    ev.set()
    th.join(HOLD_S + 2)
    server.shutdown()
    server.server_close()
    assert out.get("returned", 1e9) - t_cancel < 1.0, out
    assert out.get("severed") == 1
    assert isinstance(out.get("cancelled"), GenerationCancelledError)


def test_base_converts_a_transport_failure_after_cancel_and_never_retries():
    """Any failure of a call whose event is set is the typed stop (one call)."""
    from tests.provider_stubs import StaticProvider

    class _P(StaticProvider):
        calls = 0

        def supports_generation_cancel(self) -> bool:
            return True

        def _generate_internal(self, prompt, **kwargs):  # type: ignore[override]
            type(self).calls += 1
            kwargs["_cancel_event"].set()
            raise ConnectionError("peer closed connection")

    p = _P("static-model")
    with pytest.raises(GenerationCancelledError):
        p.generate("x", cancel_event=threading.Event())
    assert _P.calls == 1


def test_core_server_wires_client_disconnect_to_cancel_for_http_providers(stub_server):
    """The AbstractCore server runs HTTP-backed providers off the loop with a
    client-disconnect watcher (the remote runtime's Stop = a severed request);
    in-process providers stay serialized on the loop."""
    from abstractcore.server.app import _client_disconnect_cancels
    from tests.provider_stubs import StaticProvider

    for kind in ("openai-compatible", "lmstudio", "ollama"):
        assert _client_disconnect_cancels(_provider(kind, stub_server.url)) is True
    assert _client_disconnect_cancels(StaticProvider("static-model")) is False
