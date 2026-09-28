"""No route lets an unauthenticated caller spend a key the server holds.

Every capability plugin here is a stand-in that spends the way the real ones
do: discovery calls probe their remote providers, voice execution spends when
the engine it runs is remote (AbstractVoice's `engine_runtime` stand-in),
music and vision execution spend (the only backends registered are remote),
residency bookkeeping spends nothing. A spend authenticates with the
owner-config key the real plugin reads first (`voice_openai_api_key`,
`vision_api_key`, `music_acemusic_api_key`, ...), else the env var it falls
back to (OPENAI_API_KEY, ACEMUSIC_API_KEY), and is recorded. A server key on
the wire (httpx, requests, urllib) is recorded too.

The test ENUMERATES the app's routes (every GET, every POST of the media and
capability families), calls each as three callers, and derives the set of
routes that touch provider keys from the authenticated pass: whatever spends
the server's key there is a key-touching route. None of those may spend it for
an anonymous caller (ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED=1) or for a
caller that brings its own key (that key must be the one spent). A new route
that reaches a plugin without the shared guard (`abstractcore.server.credentials`)
fails here without anyone listing it.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import os
from typing import Any, Dict, List, Tuple

import pytest
from fastapi import UploadFile
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from abstractcore.server import audio_endpoints as ae
from abstractcore.server.app import app
from tests.server.test_server_voice_credentials import _stand_in_status

SERVER = "sk-SENTINEL-SERVERHELD"
CALLER = "sk-SENTINEL-CALLER"
TOKEN = "srv-token-test"
SPENT: List[Tuple[str, str, str]] = []

# capability -> (owner-config keys the real plugin reads first, env fallback)
_KEYS = {
    "voice": (("voice_openai_api_key", "voice_remote_api_key"), "OPENAI_API_KEY"),
    "audio": (("voice_openai_api_key", "voice_remote_api_key"), "OPENAI_API_KEY"),
    "vision": (("vision_api_key",), "OPENAI_API_KEY"),
    "music": (("music_acemusic_api_key",), "ACEMUSIC_API_KEY"),
}


_RESIDENCY = {"list_resident_models", "list_loaded_models", "load_resident_model", "unload_resident_model"}
_VOICE_EXECUTION = {"tts": "voice_tts_engine", "tts_stream": "voice_tts_engine", "stt": "voice_stt_engine",
                    "transcribe": "voice_stt_engine", "clone": "voice_cloning_engine"}
_VOICE_DEFAULT_ENGINE = {"voice_tts_engine": "openai", "voice_stt_engine": "openai", "voice_cloning_engine": "omnivoice"}


def _spends_for(capability: str, method: str, cfg: Dict[str, Any], kwargs: Dict[str, Any]) -> bool:
    if method in _RESIDENCY:
        return False
    if capability in ("voice", "audio") and method in _VOICE_EXECUTION:
        config_key = _VOICE_EXECUTION[method]
        engine = (kwargs.get("cloning_engine") if method == "clone" else None) or kwargs.get("provider")
        engine = engine or cfg.get(config_key) or _VOICE_DEFAULT_ENGINE[config_key]
        try:
            return _stand_in_status(engine).remote
        except ValueError:
            return True
    return True


class _Spender:
    """A plugin stand-in: a call that spends authenticates (config key, else env)."""

    def __init__(self, owner: Any, capability: str, backend_id: str):
        self._owner = owner
        self._capability = capability
        self.backend_id = backend_id

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        cap = self._capability
        config_keys, env_var = _KEYS[cap]

        def call(*args: Any, **kwargs: Any) -> Any:
            cfg = getattr(self._owner, "config", {}) or {}
            if _spends_for(cap, name, cfg, kwargs):
                key = next((cfg.get(k) for k in config_keys if cfg.get(k)), None) or os.environ.get(env_var)
                SPENT.append((cap, name, "server" if key == SERVER else "caller" if key == CALLER else "none"))
            if name in {"tts", "t2m", "t2i", "i2i", "t2v", "i2v"}:
                return b"bytes"
            if name in {"stt", "transcribe"}:
                return "text"
            return {}

        return call


class _EP:
    name = "spender"
    value = "tests.spender:register"

    @staticmethod
    def load():
        def register(registry):
            registry.register_voice_backend(backend_id="spender-voice", factory=lambda o: _Spender(o, "voice", "spender-voice"), priority=99)
            registry.register_audio_backend(backend_id="spender-audio", factory=lambda o: _Spender(o, "audio", "spender-audio"), priority=99)
            registry.register_vision_backend(backend_id="spender-vision", factory=lambda o: _Spender(o, "vision", "spender-vision"), priority=99)
            registry.register_music_backend(
                backend_id="abstractmusic:acemusic", factory=lambda o: _Spender(o, "music", "abstractmusic:acemusic"), priority=99
            )

        return register


class _EntryPoints:
    def select(self, *, group):
        return [_EP()] if group == "abstractcore.capabilities_plugins" else []


_PATH_VALUES = {
    "capability": ["voice", "audio", "vision", "music", "scene3d", "camera"],
    "provider": ["openai", "compatible", "acemusic", "diffusers"],
}
_POST_FAMILIES = ("/audio", "/voice", "/images", "/videos", "/vision", "/music", "/capabilities", "/scene3d", "/camera")


def _expand(path: str) -> List[str]:
    out = [path]
    for param, values in _PATH_VALUES.items():
        token = "{" + param + "}"
        out = [p.replace(token, v) for p in out for v in values] if any(token in p for p in out) else out
    return [p for p in out if "{" not in p]


def _walk(routes: Any, prefix: str = ""):
    """(path, methods, endpoint) for every HTTP route, included routers too
    (FastAPI >= 0.140 keeps an included router as one lazy entry)."""
    for route in routes:
        if isinstance(route, APIRoute):
            yield prefix + route.path, set(route.methods or ()), route.endpoint
        ctx = getattr(route, "include_context", None)
        if ctx is not None:
            yield from _walk(ctx.included_router.routes, prefix + (ctx.prefix or ""))


def _uploads(endpoint: Any) -> List[str]:
    names = []
    for name, param in inspect.signature(endpoint).parameters.items():
        annotation = param.annotation
        if annotation is UploadFile or "UploadFile" in str(annotation):
            names.append(name)
    return names


def _requests() -> List[Tuple[str, str, Any]]:
    out = []
    for route_path, methods, endpoint in _walk(app.routes):
        for method in sorted(methods):
            if method == "GET":
                pass
            elif method == "POST" and any(f in route_path for f in _POST_FAMILIES):
                pass
            else:
                continue
            uploads = _uploads(endpoint) if method == "POST" else []
            for path in _expand(route_path):
                out.append((f"{method} {route_path}", method, (path, uploads)))
    return out


def _call(client: TestClient, method: str, spec: Any, headers: Dict[str, str]) -> None:
    path, uploads = spec
    if method == "GET":
        client.get(path, headers=headers)
    elif uploads:
        files = {name: (f"{name}.bin", b"\x00" * 64, "application/octet-stream") for name in uploads}
        client.post(path, files=files, data={"prompt": "x", "input": "x", "name": "x"}, headers=headers)
    else:
        for body in ({"prompt": "x"}, {"input": "x"}, {"text": "x"}):
            if client.post(path, json=body, headers=headers).status_code != 422:
                break


def _spends(client: TestClient, headers: Dict[str, str]) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for label, method, spec in _requests():
        before = len(SPENT)
        try:
            _call(client, method, spec, headers)
        except Exception:
            pass  # a crash after the spend is still a spend
        for _cap, _name, who in SPENT[before:]:
            out.setdefault(label, set()).add(who)
    return out


def _record_wire(blob: Any) -> None:
    text = str(blob)
    if SERVER in text or CALLER in text:
        SPENT.append(("wire", "http", "server" if SERVER in text else "caller"))


@pytest.fixture()
def server_holds_keys(monkeypatch):
    """Server keys held; plugins are spenders; outbound HTTP is recorded
    (a server key on the wire is a spend too) and answered 503 in-process."""
    import httpx
    import requests

    real_send = httpx.Client.send
    real_asend = httpx.AsyncClient.send

    def send(self, req, **kw):
        if req.url.host == "testserver":
            return real_send(self, req, **kw)
        _record_wire(req.headers)
        return httpx.Response(503, json={"error": "offline"}, request=req)

    async def asend(self, req, **kw):
        if req.url.host == "testserver":
            return await real_asend(self, req, **kw)
        _record_wire(req.headers)
        return httpx.Response(503, json={"error": "offline"}, request=req)

    def request(self, method, url, **kw):
        _record_wire({**dict(self.headers), **dict(kw.get("headers") or {})})
        raise requests.ConnectionError("offline (test)")

    import urllib.request

    def urlopen(req, *a, **k):
        _record_wire(dict(req.header_items()) if hasattr(req, "header_items") else req)
        raise OSError("offline (test)")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(httpx.Client, "send", send)
    monkeypatch.setattr(httpx.AsyncClient, "send", asend)
    monkeypatch.setattr(requests.Session, "request", request)
    import sys
    import types

    runtime = types.ModuleType("abstractvoice.engine_runtime")
    runtime.engine_runtime_status = _stand_in_status
    monkeypatch.setitem(sys.modules, "abstractvoice.engine_runtime", runtime)
    # Voice through OpenAI (the fresh config's voice routes are local engines).
    real_config = ae._capability_config
    monkeypatch.setattr(
        ae, "_capability_config", lambda: {**real_config(), "voice_tts_engine": "openai", "voice_stt_engine": "openai"}
    )
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda: _EntryPoints())
    monkeypatch.setattr(ae, "_CORE", None)
    for name in ("OPENAI_API_KEY", "ACEMUSIC_API_KEY", "ELEVENLABS_API_KEY"):
        monkeypatch.setenv(name, SERVER)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    SPENT.clear()
    yield
    SPENT.clear()
    # The enumeration reached real vision lanes: drop the backends and jobs
    # they cached, or later tests see them as resident models.
    from abstractcore.server import vision_endpoints

    with vision_endpoints._BACKEND_CACHE_LOCK:
        vision_endpoints._BACKEND_CACHE.clear()
    with vision_endpoints._JOBS_LOCK:
        vision_endpoints._JOBS.clear()


def test_the_walk_sees_the_included_routers():
    paths = {path for path, _m, _e in _walk(app.routes)}
    assert {"/v1/audio/speech", "/v1/audio/music", "/v1/vision/models", "/v1/capabilities/{capability}/models"} <= paths


def test_no_route_spends_a_server_key_without_server_auth(server_holds_keys, monkeypatch):
    # Authenticated: which routes touch provider keys at all.
    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", TOKEN)
    monkeypatch.delenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", raising=False)
    authed = _spends(TestClient(app), {"Authorization": f"Bearer {TOKEN}"})
    key_routes = {label for label, who in authed.items() if "server" in who}
    must = {
        "GET /v1/capabilities/{capability}/models",
        "GET /v1/capabilities/{capability}/providers",
        "GET /v1/audio/music/providers",
        "GET /v1/audio/music/models",
        "GET /v1/audio/music/provider-details",
        "GET /v1/audio/speech/providers",
        "POST /v1/audio/speech",
        "POST /v1/audio/music",
    }
    assert must <= key_routes, f"the stand-in plugins no longer reach {sorted(must - key_routes)}"

    # Anonymous caller admitted by ALLOW_UNAUTHENTICATED.
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN")
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    anon = _spends(TestClient(app), {})
    offenders = sorted(label for label, who in anon.items() if "server" in who)
    assert offenders == [], "anonymous callers spend the server's key on: " + ", ".join(offenders)

    # A caller with its own key: that key, never the server's.
    monkeypatch.delenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED")
    own = _spends(TestClient(app), {"X-AbstractCore-Provider-API-Key": CALLER})
    offenders = sorted(label for label, who in own.items() if "server" in who)
    assert offenders == [], "callers with their own key spend the server's key on: " + ", ".join(offenders)
    assert any("caller" in who for who in own.values()), "a caller key must still be usable"
