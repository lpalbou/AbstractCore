import asyncio
import importlib

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from abstractcore.server.auth_policy import (
    ServerAuthPolicy, current_server_auth_policy, server_auth_token,
    server_allows_unauthenticated, use_server_auth_policy,
)


def request(headers=()):
    return Request({"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": headers})


def test_policy_restores_nested_context_after_exception(monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", "standalone")
    with use_server_auth_policy(ServerAuthPolicy("outer")):
        with pytest.raises(RuntimeError):
            with use_server_auth_policy(ServerAuthPolicy("inner", True)):
                assert server_auth_token() == "inner"
                assert server_allows_unauthenticated()
                raise RuntimeError("failed request")
        assert server_auth_token() == "outer"
    assert current_server_auth_policy() is None
    assert server_auth_token() == "standalone"


def test_concurrent_requests_and_worker_threads_have_isolated_policies(monkeypatch):
    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", "standalone")
    async def run():
        async def one(token):
            with use_server_auth_policy(ServerAuthPolicy(token)):
                await asyncio.sleep(0)
                assert await asyncio.to_thread(server_auth_token) == token
                return server_auth_token()
        assert await asyncio.gather(one("a"), one("b")) == ["a", "b"]
        assert server_auth_token() == "standalone"
    asyncio.run(run())


def test_open_managed_policy_reserves_authorization_and_preserves_provider_guards(monkeypatch):
    app = importlib.import_module("abstractcore.server.app")
    audio = importlib.import_module("abstractcore.server.audio_endpoints")
    vision = importlib.import_module("abstractcore.server.vision_endpoints")
    monkeypatch.setenv("ABSTRACTCORE_AUTH_TOKEN", "standalone")
    monkeypatch.setenv("OPENAI_API_KEY", "server-cloud-key")
    async def accepted(req):
        return bool(getattr(req.state, "abstractcore_server_authenticated", False))
    async def run():
        with use_server_auth_policy(ServerAuthPolicy("managed", True)):
            anonymous = request()
            assert await app._enforce_server_auth(anonymous, accepted) is False
            with pytest.raises(HTTPException) as error:
                app._guard_unauthenticated_server_provider_key_use("openai", explicit_provider_key=False, http_request=anonymous)
            assert error.value.status_code == 401
            explicit = request([(b"x-abstractcore-provider-api-key", b"client-cloud-key")])
            assert await app._enforce_server_auth(explicit, accepted) is False
            app._guard_unauthenticated_server_provider_key_use("openai", explicit_provider_key=True, http_request=explicit)
            for module in [app, audio, vision]:
                assert module._server_auth_enabled()
                assert module._provider_api_key_from_request(explicit) == "client-cloud-key"
                assert module._provider_api_key_from_request(request([(b"authorization", b"Bearer managed")])) is None
            for token in [b"wrong", b"standalone", b"client-cloud-key"]:
                rejected = await app._enforce_server_auth(request([(b"authorization", b"Bearer " + token)]), accepted)
                assert rejected.status_code == 401
            assert await app._enforce_server_auth(request([(b"authorization", b"Bearer managed")]), accepted) is True
    asyncio.run(run())


def test_required_managed_policy_rejects_missing_token_and_standalone_semantics_survive(monkeypatch):
    app = importlib.import_module("abstractcore.server.app")
    monkeypatch.delenv("ABSTRACTCORE_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED", "1")
    async def accepted(req): return "ok"
    async def run():
        with use_server_auth_policy(ServerAuthPolicy("managed")):
            assert not server_allows_unauthenticated()
            assert (await app._enforce_server_auth(request(), accepted)).status_code == 401
        assert await app._enforce_server_auth(request(), accepted) == "ok"
        assert app._provider_api_key_from_request(request([(b"authorization", b"Bearer client-key")])) == "client-key"
    asyncio.run(run())


def test_policy_requires_real_token_and_boolean_open_flag():
    with pytest.raises(ValueError): ServerAuthPolicy("")
    with pytest.raises(TypeError): ServerAuthPolicy("key", "false")
    assert "secret" not in repr(ServerAuthPolicy("secret"))
