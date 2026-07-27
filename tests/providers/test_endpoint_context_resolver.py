"""Host-injected endpoint-profile resolver context (cross-package fix, 2026-07-26).

Live incident pinned here: a run on ``endpoint:airelay`` (a GATEWAY-registered
profile) called ``analyze_media``, whose session route performs a NESTED
``create_llm("endpoint:airelay", ...)`` — and the registry raised
``Unknown provider: endpoint:airelay`` because gateway profiles live in the
host's store, not in ``~/.abstractcore``, and a tool-side construction has no
provider instance to inherit the host's resolver from.

The converged design (hub, three seats): a contextvar channel the HOST enters
around tool execution (``use_provider_endpoint_profile_resolver``), consulted
by the registry ONLY after local config misses an ``endpoint:*`` spec.
Invariants pinned below:
- local config FIRST (tripwire: a locally-resolvable spec never consults);
- non-endpoint specs never consult and keep today's error byte-identical;
- bare core (no resolver installed) keeps today's error byte-identical;
- both-miss errors name BOTH sources;
- malformed resolver payloads are labeled misses, never crashes;
- secrets (api_key) never appear in errors or log messages;
- contextvar-hit constructions carry ``resolve_provider_endpoint_profile``
  (the BaseProvider setattr-propagation pattern) so nested/fallback
  constructions inherit the host's resolution reach.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from abstractcore.providers.endpoint_context import (
    current_provider_endpoint_profile_resolver,
    use_provider_endpoint_profile_resolver,
)

SECRET = "sk-gw-secret-0deadbeef1234567890"


def _reset_global_config(monkeypatch, config_file) -> None:
    """Point the global config manager at a per-test file and reset singletons
    (the tests/config/test_provider_profiles_config.py pattern) so operator
    profiles on the dev machine can never leak into these assertions."""
    monkeypatch.setenv("ABSTRACTCORE_CONFIG_FILE", str(config_file))
    import abstractcore.config.manager as manager_module
    import abstractcore.providers.registry as registry_module

    manager_module._config_manager = None
    registry_module._registry = None


class _LogRecorder:
    """Deterministic log capture: the registry logs through StructuredLogger,
    which may route via structlog — caplog capture is environment-dependent,
    a direct recorder is not."""

    def __init__(self):
        self.messages: list[tuple[str, str]] = []

    def _record(self, level):
        def _log(message, **kwargs):
            self.messages.append((level, str(message)))

        return _log

    def __getattr__(self, name):
        if name in {"debug", "info", "warning", "error", "critical"}:
            return self._record(name)
        raise AttributeError(name)

    def text(self) -> str:
        return "\n".join(m for _, m in self.messages)

    def warnings(self) -> list[str]:
        return [m for lvl, m in self.messages if lvl == "warning"]


@pytest.fixture()
def record_registry_logs(monkeypatch) -> _LogRecorder:
    import abstractcore.providers.registry as registry_module

    recorder = _LogRecorder()
    monkeypatch.setattr(registry_module, "logger", recorder)
    return recorder


def _gateway_payload(profile_id: str = "gw-only", **overrides) -> dict:
    """The gateway ``private_resolution()`` shape (the dict core must accept),
    including fields core does NOT consume (scope/capabilities/fingerprints)."""
    payload = {
        "id": profile_id,
        "virtual_provider": f"endpoint:{profile_id}",
        "display_name": "GW Only",
        "description": "host-registered profile",
        "provider_family": "openai-compatible",
        "provider": "openai-compatible",
        "base_url": "http://127.0.0.1:65500/v1",
        "base_url_configured": True,
        "api_key": SECRET,
        "api_key_set": True,
        "api_key_fingerprint": "abcd1234",
        "scope": "user",
        "capabilities": ["text"],
        "allowed_models": ["ctx-model-1"],
        "enabled": True,
        "created_at": "2026-07-26T00:00:00Z",
        "updated_at": "2026-07-26T00:00:00Z",
    }
    payload.update(overrides)
    return payload


def _patch_offline_construction(monkeypatch):
    """Real OpenAICompatibleProvider construction without network: model
    validation is the only construction-time HTTP (the established
    tests/config/test_provider_profiles_config.py trick)."""
    from abstractcore.providers.openai_compatible_provider import OpenAICompatibleProvider

    monkeypatch.setattr(OpenAICompatibleProvider, "_validate_model", lambda self: None)
    return OpenAICompatibleProvider


# ---------------------------------------------------------------------------
# Contextvar semantics
# ---------------------------------------------------------------------------


def test_contextvar_default_set_nest_and_restore() -> None:
    outer = lambda spec: None  # noqa: E731
    inner = lambda spec: {"provider_family": "openai-compatible"}  # noqa: E731

    assert current_provider_endpoint_profile_resolver() is None
    with use_provider_endpoint_profile_resolver(outer):
        assert current_provider_endpoint_profile_resolver() is outer
        with use_provider_endpoint_profile_resolver(inner):
            assert current_provider_endpoint_profile_resolver() is inner
        assert current_provider_endpoint_profile_resolver() is outer
    assert current_provider_endpoint_profile_resolver() is None


def test_contextvar_restores_on_exception_and_none_masks() -> None:
    outer = lambda spec: None  # noqa: E731

    with use_provider_endpoint_profile_resolver(outer):
        # Explicit None masks the outer resolver (host fencing sub-work).
        with use_provider_endpoint_profile_resolver(None):
            assert current_provider_endpoint_profile_resolver() is None
        assert current_provider_endpoint_profile_resolver() is outer

        with pytest.raises(RuntimeError):
            with use_provider_endpoint_profile_resolver(lambda s: None):
                raise RuntimeError("boom")
        assert current_provider_endpoint_profile_resolver() is outer
    assert current_provider_endpoint_profile_resolver() is None


def test_public_import_paths_are_stable() -> None:
    """The runtime imports these; both the cheap deep path and the lazy
    package attribute must serve the same objects."""
    import abstractcore.providers as providers_pkg
    from abstractcore.providers.endpoint_context import (
        use_provider_endpoint_profile_resolver as deep_cm,
    )

    assert providers_pkg.use_provider_endpoint_profile_resolver is deep_cm
    assert callable(providers_pkg.current_provider_endpoint_profile_resolver)


# ---------------------------------------------------------------------------
# Resolution through the injected resolver
# ---------------------------------------------------------------------------


def test_endpoint_spec_resolves_via_injected_resolver_when_local_misses(
    monkeypatch, tmp_path
) -> None:
    _reset_global_config(monkeypatch, tmp_path / "abstractcore.json")
    provider_cls = _patch_offline_construction(monkeypatch)
    from abstractcore import create_llm

    calls: list[str] = []

    def resolver(spec: str):
        calls.append(spec)
        return _gateway_payload("gw-only") if spec == "endpoint:gw-only" else None

    with use_provider_endpoint_profile_resolver(resolver):
        llm = create_llm("endpoint:gw-only", model="ctx-model-1")

    assert isinstance(llm, provider_cls)
    assert llm.base_url == "http://127.0.0.1:65500/v1"
    assert llm.api_key == SECRET
    assert llm._abstractcore_virtual_provider == "endpoint:gw-only"
    assert llm._abstractcore_provider_family == "openai-compatible"
    # The resolver is consulted with the full spec, possibly more than once
    # (registry info + create paths) — every call must carry the same spec.
    assert calls and all(spec == "endpoint:gw-only" for spec in calls)


def test_context_hit_attaches_resolver_for_nested_constructions(
    monkeypatch, tmp_path
) -> None:
    _reset_global_config(monkeypatch, tmp_path / "abstractcore.json")
    _patch_offline_construction(monkeypatch)
    from abstractcore import create_llm

    def resolver(spec: str):
        return _gateway_payload("gw-only") if spec == "endpoint:gw-only" else None

    with use_provider_endpoint_profile_resolver(resolver):
        llm = create_llm("endpoint:gw-only", model="ctx-model-1")

    # Outside the context the ambient channel is gone (containment) ...
    with pytest.raises(ValueError, match="Unknown provider: endpoint:gw-only"):
        create_llm("endpoint:gw-only", model="ctx-model-1")
    # ... but the constructed instance carries the resolver (the BaseProvider
    # setattr-propagation pattern), so instance-mediated nested/fallback
    # constructions keep the host's resolution reach.
    attached = getattr(llm, "resolve_provider_endpoint_profile", None)
    assert callable(attached)
    assert attached("endpoint:gw-only")["base_url"] == "http://127.0.0.1:65500/v1"


def test_local_config_first_never_consults_context_for_local_profiles(
    monkeypatch, tmp_path
) -> None:
    _reset_global_config(monkeypatch, tmp_path / "abstractcore.json")
    _patch_offline_construction(monkeypatch)
    from abstractcore import create_llm
    from abstractcore.config import get_config_manager

    get_config_manager().set_provider_profile(
        "local-prof",
        provider_family="openai-compatible",
        base_url="http://127.0.0.1:65501/v1",
        api_key="local-key",
        allowed_models=["local-model"],
    )

    def tripwire(spec: str):  # pragma: no cover - firing IS the failure
        raise AssertionError(
            "a locally-resolvable endpoint spec must never consult the injected resolver"
        )

    with use_provider_endpoint_profile_resolver(tripwire):
        llm = create_llm("endpoint:local-prof", model="local-model")

    assert llm.base_url == "http://127.0.0.1:65501/v1"
    assert llm.api_key == "local-key"
    # Zero behavior change for operator-profile routes: locally-resolved
    # constructions do NOT grow a resolver attribute.
    assert getattr(llm, "resolve_provider_endpoint_profile", None) is None


def test_non_endpoint_unknown_provider_never_consults_and_error_is_byte_identical(
    monkeypatch, tmp_path
) -> None:
    _reset_global_config(monkeypatch, tmp_path / "abstractcore.json")
    from abstractcore import create_llm
    from abstractcore.providers.registry import get_provider_registry

    def tripwire(spec: str):  # pragma: no cover - firing IS the failure
        raise AssertionError("non-endpoint specs must never consult the injected resolver")

    expected = (
        "Unknown provider: definitely-not-a-provider. Available providers: "
        + ", ".join(get_provider_registry().list_provider_names())
    )
    with use_provider_endpoint_profile_resolver(tripwire):
        with pytest.raises(ValueError) as exc_info:
            create_llm("definitely-not-a-provider")
    assert str(exc_info.value) == expected


# ---------------------------------------------------------------------------
# Error honesty
# ---------------------------------------------------------------------------


def test_bare_core_error_stays_byte_identical_on_both_raise_sites(
    monkeypatch, tmp_path
) -> None:
    _reset_global_config(monkeypatch, tmp_path / "abstractcore.json")
    from abstractcore import create_llm
    from abstractcore.providers.registry import get_provider_registry

    registry = get_provider_registry()
    assert current_provider_endpoint_profile_resolver() is None

    expected_create = (
        "Unknown provider: endpoint:nope. Available providers: "
        + ", ".join(registry.list_provider_names())
    )
    with pytest.raises(ValueError) as exc_info:
        create_llm("endpoint:nope", model="x")
    assert str(exc_info.value) == expected_create

    with pytest.raises(ValueError) as exc_info:
        registry.get_provider_class("endpoint:nope")
    assert str(exc_info.value) == "Unknown provider: endpoint:nope"


def test_both_miss_error_names_both_sources(monkeypatch, tmp_path) -> None:
    _reset_global_config(monkeypatch, tmp_path / "abstractcore.json")
    from abstractcore import create_llm
    from abstractcore.providers.registry import get_provider_registry

    with use_provider_endpoint_profile_resolver(lambda spec: None):
        with pytest.raises(ValueError) as exc_info:
            create_llm("endpoint:airelay", model="gpt-5.6-sol")
        message = str(exc_info.value)
        assert message.startswith("Unknown provider: endpoint:airelay")
        assert "local AbstractCore config" in message
        assert "host-injected endpoint-profile resolver" in message
        assert "Available providers:" in message

        with pytest.raises(ValueError) as exc_info:
            get_provider_registry().get_provider_class("endpoint:airelay")
        message = str(exc_info.value)
        assert "local AbstractCore config" in message
        assert "host-injected endpoint-profile resolver" in message


# ---------------------------------------------------------------------------
# Malformed payloads: labeled miss, never a crash; secrets never leak
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload, expectation",
    [
        # Unsupported family (gateway allows arbitrary tokens; core does not).
        ({"provider_family": "no-such-family", "api_key": SECRET}, "not usable"),
        # Wrong type entirely.
        ("not-a-dict", "expected a dict"),
        # Explicitly disabled profiles must not be constructed.
        (_gateway_payload("gw-only", enabled=False), "disabled"),
        # A base_url core refuses (scheme validation) with a secret riding along.
        ({"provider_family": "openai-compatible", "base_url": "ftp://x", "api_key": SECRET}, "not usable"),
    ],
)
def test_malformed_or_disabled_payload_is_labeled_miss(
    monkeypatch, tmp_path, record_registry_logs, payload, expectation
) -> None:
    _reset_global_config(monkeypatch, tmp_path / "abstractcore.json")
    from abstractcore import create_llm

    with use_provider_endpoint_profile_resolver(lambda spec: payload):
        with pytest.raises(ValueError, match="Unknown provider: endpoint:bad-prof"):
            create_llm("endpoint:bad-prof", model="m")

    warnings = record_registry_logs.warnings()
    assert any("#FALLBACK" in w and expectation in w for w in warnings), warnings
    assert SECRET not in record_registry_logs.text()


def test_resolver_exception_is_labeled_miss_never_a_crash(
    monkeypatch, tmp_path, record_registry_logs
) -> None:
    _reset_global_config(monkeypatch, tmp_path / "abstractcore.json")
    from abstractcore import create_llm

    def broken(spec: str):
        raise RuntimeError("host store unavailable")

    with use_provider_endpoint_profile_resolver(broken):
        with pytest.raises(ValueError, match="Unknown provider: endpoint:bad-host"):
            create_llm("endpoint:bad-host", model="m")

    warnings = record_registry_logs.warnings()
    assert any("#FALLBACK" in w and "RuntimeError" in w for w in warnings), warnings


def test_secrets_never_in_errors_or_logs_on_success_or_failure(
    monkeypatch, tmp_path, record_registry_logs
) -> None:
    _reset_global_config(monkeypatch, tmp_path / "abstractcore.json")
    _patch_offline_construction(monkeypatch)
    from abstractcore import create_llm

    # Success path: construction logs must not carry the key.
    with use_provider_endpoint_profile_resolver(lambda spec: _gateway_payload("gw-only")):
        llm = create_llm("endpoint:gw-only", model="ctx-model-1")
    assert llm.api_key == SECRET
    assert SECRET not in record_registry_logs.text()

    # Failure path: a payload that trips normalization must warn WITHOUT the key.
    bad = _gateway_payload("gw-bad", provider_family="not-a-family", provider="not-a-family")
    with use_provider_endpoint_profile_resolver(lambda spec: bad):
        with pytest.raises(ValueError) as exc_info:
            create_llm("endpoint:gw-bad", model="m")
    assert SECRET not in str(exc_info.value)
    assert SECRET not in record_registry_logs.text()

    # The instance's public profile stamp must not carry the key either
    # (public_dict redacts to fingerprint + boolean).
    import json

    assert SECRET not in json.dumps(llm._abstractcore_provider_profile)


def test_empty_payload_is_a_labeled_miss_never_a_guess(
    monkeypatch, tmp_path, record_registry_logs
) -> None:
    """A `{}` payload names NO provider family — constructing against a
    silently-defaulted family would build a provider the host never
    described. Missing fields are a MISS (labeled), never a guess."""
    _reset_global_config(monkeypatch, tmp_path / "abstractcore.json")
    from abstractcore import create_llm

    with use_provider_endpoint_profile_resolver(lambda spec: {}):
        with pytest.raises(ValueError, match="Unknown provider: endpoint:empty-prof"):
            create_llm("endpoint:empty-prof", model="m")

    warnings = record_registry_logs.warnings()
    assert any("#FALLBACK" in w and "no provider_family" in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# The incident, end to end: analyze_media session route over a host profile
# ---------------------------------------------------------------------------


def _write_real_png(path: Path) -> Path:
    from PIL import Image

    Image.new("RGB", (4, 4), (200, 30, 30)).save(path, format="PNG")
    return path


def test_analyze_media_session_route_resolves_host_profile_end_to_end(
    monkeypatch, tmp_path
) -> None:
    """The exact incident path, unbroken: analyze_media (stamped with an
    endpoint:* session route) → create_description_via_route →
    create_llm("endpoint:gw-vision", ...) → registry resolves via the
    injected resolver → REAL OpenAICompatibleProvider construction with the
    profile's transport — only the two network surfaces are patched
    (_validate_model, generate)."""
    _reset_global_config(monkeypatch, tmp_path / "abstractcore.json")
    provider_cls = _patch_offline_construction(monkeypatch)

    import abstractcore.media.capabilities as media_caps
    from abstractcore.core.types import GenerateResponse
    from abstractcore.tools.common_tools import analyze_media

    # Registry says the session model sees (fake name avoids inference).
    monkeypatch.setattr(
        media_caps, "get_model_capabilities", lambda name: {"vision_support": True}
    )

    seen: dict = {}

    def _fake_generate(self, prompt, media=None, **kwargs):
        seen.update(model=self.model, base_url=self.base_url, media=list(media or []))
        return GenerateResponse(content="A crimson swatch on white.", model=self.model, finish_reason="stop")

    monkeypatch.setattr(provider_cls, "generate", _fake_generate)

    image = _write_real_png(tmp_path / "shot.png")

    def resolver(spec: str):
        return _gateway_payload("gw-vision") if spec == "endpoint:gw-vision" else None

    with use_provider_endpoint_profile_resolver(resolver):
        out = analyze_media(
            str(image),
            question="what color?",
            _session_route={"provider": "endpoint:gw-vision", "model": "ctx-sees-1"},
        )

    assert "A crimson swatch" in out
    assert "(observed by endpoint:gw-vision/ctx-sees-1)" in out
    assert "Error" not in out
    # The nested LLM really was the host profile's transport.
    assert seen["model"] == "ctx-sees-1"
    assert seen["base_url"] == "http://127.0.0.1:65500/v1"
    assert seen["media"] == [str(image)]
