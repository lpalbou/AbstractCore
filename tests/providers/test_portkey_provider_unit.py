"""Unit tests for the Portkey provider.

These tests validate routing mode precedence (config > virtual-key > provider-direct),
header construction, env-var resolution, and edge cases — without making real API calls.
"""
import pytest

from abstractcore.providers.portkey_provider import PortkeyProvider
from abstractcore.providers.registry import ProviderRegistry


def _disable_model_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(PortkeyProvider, "_validate_model", lambda self: None)


def _clean_portkey_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove all Portkey env vars so tests are hermetic."""
    for var in (
        "PORTKEY_API_KEY", "PORTKEY_BASE_URL", "PORTKEY_CONFIG",
        "PORTKEY_PROVIDER", "PORTKEY_VIRTUAL_KEY", "PORTKEY_PROVIDER_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


# ── Registration ───────────────────────────────────────────────────────

def test_portkey_provider_is_registered():
    registry = ProviderRegistry()
    info = registry.get_provider_info("portkey")
    assert info is not None
    assert info.name == "portkey"
    assert info.display_name == "Portkey"


# ── API-key fallback from AbstractCore config ─────────────────────────

def test_portkey_uses_config_api_key_when_env_missing(monkeypatch: pytest.MonkeyPatch):
    _clean_portkey_env(monkeypatch)
    monkeypatch.setattr(PortkeyProvider, "_get_api_key_from_config", lambda self: "pk-config-fallback")
    _disable_model_validation(monkeypatch)

    provider = PortkeyProvider(model="gpt-4o-mini")
    assert provider.api_key == "pk-config-fallback"


# ── Config mode (highest precedence) ──────────────────────────────────

def test_config_mode_sends_only_config_header(monkeypatch: pytest.MonkeyPatch):
    """When config_id is set, only x-portkey-config should be sent —
    virtual-key and provider headers must be suppressed."""
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)

    provider = PortkeyProvider(
        model="gpt-5-mini",
        api_key="pk-gateway-key",
        config_id="pcfg_456",
    )
    headers = provider._get_headers()

    # Gateway auth: both x-portkey-api-key and Authorization: Bearer
    assert headers["x-portkey-api-key"] == "pk-gateway-key"
    assert headers["Authorization"] == "Bearer pk-gateway-key"
    # Config routing header
    assert headers["x-portkey-config"] == "pcfg_456"
    # Mutually exclusive headers must NOT be present
    assert "x-portkey-virtual-key" not in headers
    assert "x-portkey-provider" not in headers


def test_config_mode_suppresses_virtual_key_from_env(monkeypatch: pytest.MonkeyPatch):
    """Even if PORTKEY_VIRTUAL_KEY lingers in the environment, config mode
    must NOT send x-portkey-virtual-key — that causes 'keys not valid' errors."""
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)
    # Simulate a stale env var (the exact scenario from the bug)
    monkeypatch.setenv("PORTKEY_VIRTUAL_KEY", "default-aws-vk-d1d8ac")

    provider = PortkeyProvider(
        model="gpt-5-mini",
        api_key="pk-test",
        config_id="pcfg_prod",
    )
    headers = provider._get_headers()

    assert headers["x-portkey-config"] == "pcfg_prod"
    # The stale virtual key must NOT appear in headers
    assert "x-portkey-virtual-key" not in headers
    # But the value should still be stored for introspection
    assert provider.virtual_key == "default-aws-vk-d1d8ac"


def test_config_mode_with_extra_portkey_headers(monkeypatch: pytest.MonkeyPatch):
    """User-supplied portkey_headers (escape hatch) are always included."""
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)

    provider = PortkeyProvider(
        model="gpt-5-mini",
        api_key="pk-test",
        config_id="pcfg_abc",
        portkey_headers={"x-portkey-trace-id": "trace-123", "x-custom": "val"},
    )
    headers = provider._get_headers()

    assert headers["x-portkey-config"] == "pcfg_abc"
    assert headers["x-portkey-trace-id"] == "trace-123"
    assert headers["x-custom"] == "val"


# ── Virtual-key mode ──────────────────────────────────────────────────

def test_virtual_key_mode_sends_only_virtual_key_header(monkeypatch: pytest.MonkeyPatch):
    """When virtual_key is set but no config_id, x-portkey-virtual-key is sent."""
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)

    provider = PortkeyProvider(
        model="claude-haiku-4-5",
        api_key="pk-key",
        virtual_key="pvk_anthropic_123",
    )
    headers = provider._get_headers()

    assert headers["x-portkey-api-key"] == "pk-key"
    assert headers["Authorization"] == "Bearer pk-key"
    assert headers["x-portkey-virtual-key"] == "pvk_anthropic_123"
    # Config and provider headers must not be present
    assert "x-portkey-config" not in headers
    assert "x-portkey-provider" not in headers


# ── Provider-direct mode ──────────────────────────────────────────────

def test_provider_direct_mode_sends_provider_and_auth(monkeypatch: pytest.MonkeyPatch):
    """When portkey_provider + provider_api_key are set (no config/vkey),
    provider-direct routing is used with upstream auth override."""
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)

    provider = PortkeyProvider(
        model="gpt-4o",
        api_key="pk-key",
        portkey_provider="openai",
        provider_api_key="sk-openai-upstream",
    )
    headers = provider._get_headers()

    assert headers["x-portkey-api-key"] == "pk-key"
    assert headers["x-portkey-provider"] == "openai"
    # Authorization should be the upstream provider key, not the Portkey key
    assert headers["Authorization"] == "Bearer sk-openai-upstream"
    assert "x-portkey-config" not in headers
    assert "x-portkey-virtual-key" not in headers


# ── Env-var resolution ─────────────────────────────────────────────────

def test_portkey_reads_all_env_values(monkeypatch: pytest.MonkeyPatch):
    """All Portkey-specific values can be read from environment variables."""
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)
    monkeypatch.setenv("PORTKEY_BASE_URL", "https://gateway.example.com/v1/")
    monkeypatch.setenv("PORTKEY_API_KEY", "pk-env-key")
    monkeypatch.setenv("PORTKEY_CONFIG", "pcfg_env")

    provider = PortkeyProvider(model="claude-haiku-4-5")

    assert provider.base_url == "https://gateway.example.com/v1"
    assert provider.api_key == "pk-env-key"
    assert provider.config_id == "pcfg_env"

    headers = provider._get_headers()
    assert headers["x-portkey-api-key"] == "pk-env-key"
    assert headers["Authorization"] == "Bearer pk-env-key"
    assert headers["x-portkey-config"] == "pcfg_env"


def test_explicit_params_override_env_vars(monkeypatch: pytest.MonkeyPatch):
    """Explicit constructor parameters take precedence over env vars."""
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)
    monkeypatch.setenv("PORTKEY_CONFIG", "pcfg_env")
    monkeypatch.setenv("PORTKEY_VIRTUAL_KEY", "pvk_env")

    provider = PortkeyProvider(
        model="gpt-4o",
        api_key="pk-explicit",
        config_id="pcfg_explicit",
    )

    assert provider.config_id == "pcfg_explicit"
    headers = provider._get_headers()
    assert headers["x-portkey-config"] == "pcfg_explicit"


# ── Edge cases ─────────────────────────────────────────────────────────

def test_portkey_skips_empty_header_values(monkeypatch: pytest.MonkeyPatch):
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)

    provider = PortkeyProvider(
        model="default",
        api_key="",
        config_id="",
        portkey_provider="",
        virtual_key="",
        provider_api_key="",
        portkey_headers={"x-portkey-empty": ""},
    )
    headers = provider._get_headers()

    assert headers == {"Content-Type": "application/json"}


def test_routing_mode_detection(monkeypatch: pytest.MonkeyPatch):
    """Verify the routing mode detection logic."""
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)

    # Config mode
    p = PortkeyProvider(model="default", config_id="pcfg_x")
    assert p._routing_mode() == "config"

    # Virtual-key mode
    p = PortkeyProvider(model="default", virtual_key="pvk_y")
    assert p._routing_mode() == "virtual_key"

    # Provider-direct mode (fallback)
    p = PortkeyProvider(model="default", portkey_provider="openai")
    assert p._routing_mode() == "provider_direct"

    # Config takes precedence over virtual_key
    p = PortkeyProvider(model="default", config_id="pcfg_x", virtual_key="pvk_y")
    assert p._routing_mode() == "config"


# ── Model validation ──────────────────────────────────────────────────

def test_portkey_skips_model_validation(monkeypatch: pytest.MonkeyPatch):
    """Portkey is a routing gateway — model validation via /models must be
    skipped because the gateway model catalogue doesn't reflect what's
    available through config-based routing."""
    _clean_portkey_env(monkeypatch)
    # Do NOT call _disable_model_validation — we want to verify the
    # provider's own _validate_model override is a no-op.

    # This must NOT raise ModelNotFoundError even for an "unknown" model,
    # because model availability depends on Portkey config routing.
    provider = PortkeyProvider(
        model="any-model-name-that-would-fail-validation",
        api_key="pk-test",
        config_id="pcfg_test",
    )
    assert provider.model == "any-model-name-that-would-fail-validation"


# ── Payload adaptation ─────────────────────────────────────────────

def test_mutate_payload_strips_unsolicited_defaults(monkeypatch: pytest.MonkeyPatch):
    """When the user doesn't explicitly set temperature, top_p, or max_tokens,
    the payload must NOT include them — Portkey forwards verbatim to backends
    that may reject unsolicited values (e.g. GPT-5 rejects temperature ≠ 1)."""
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)

    # No explicit generation params in constructor
    provider = PortkeyProvider(
        model="gpt-5-mini",
        api_key="pk-test",
        config_id="pcfg_test",
    )

    payload = {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "stream": False,
        "temperature": 0.7,   # default injected by parent
        "max_tokens": 4096,   # default injected by parent
        "top_p": 0.9,         # default injected by parent
    }
    # No kwargs in generate() call either
    result = provider._mutate_payload(payload)

    # All unsolicited defaults must be stripped
    assert "temperature" not in result
    assert "top_p" not in result
    assert "max_tokens" not in result
    assert "max_completion_tokens" not in result
    # Required fields preserved
    assert result["model"] == "gpt-5-mini"
    assert result["stream"] is False


def test_mutate_payload_keeps_explicit_generate_kwargs(monkeypatch: pytest.MonkeyPatch):
    """When the user explicitly passes temperature or max_tokens in the
    generate() call, those values must be preserved."""
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)

    provider = PortkeyProvider(
        model="gpt-4o",
        api_key="pk-test",
        config_id="pcfg_test",
    )

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.5,
        "max_tokens": 2048,
        "top_p": 0.95,
    }
    # Simulate user passing these in generate() call
    result = provider._mutate_payload(
        payload, temperature=0.5, max_tokens=2048, top_p=0.95
    )

    assert result["temperature"] == 0.5
    assert result["top_p"] == 0.95
    # Non-reasoning OpenAI models keep legacy max_tokens
    assert result["max_tokens"] == 2048
    assert "max_completion_tokens" not in result


def test_mutate_payload_keeps_explicit_constructor_kwargs(monkeypatch: pytest.MonkeyPatch):
    """When the user sets temperature in the constructor, it should be kept
    even without passing it in generate()."""
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)

    provider = PortkeyProvider(
        model="gpt-4o",
        api_key="pk-test",
        config_id="pcfg_test",
        temperature=0.3,  # explicit in constructor
    )

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.3,
        "max_tokens": 4096,
        "top_p": 0.9,
    }
    # No kwargs in generate() call
    result = provider._mutate_payload(payload)

    # temperature was explicit → keep it
    assert result["temperature"] == 0.3
    # top_p and max_tokens were defaults → strip them
    assert "top_p" not in result
    assert "max_tokens" not in result
    assert "max_completion_tokens" not in result


def test_mutate_payload_reasoning_model_strips_params(monkeypatch: pytest.MonkeyPatch):
    """Reasoning models (gpt-5/o1) must not receive temperature/top_p/etc."""
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)

    provider = PortkeyProvider(
        model="gpt-5-mini",
        api_key="pk-test",
        config_id="pcfg_test",
    )

    payload = {
        "model": "gpt-5-mini",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.5,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
        "repetition_penalty": 1.1,
        "max_tokens": 1234,
    }

    result = provider._mutate_payload(
        payload,
        temperature=0.5,
        top_p=0.9,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        repetition_penalty=1.1,
        max_tokens=1234,
    )

    assert "temperature" not in result
    assert "top_p" not in result
    assert "frequency_penalty" not in result
    assert "presence_penalty" not in result
    assert "repetition_penalty" not in result
    # Reasoning models use max_completion_tokens
    assert "max_tokens" not in result
    assert result["max_completion_tokens"] == 1234


def test_mutate_payload_ignores_explicit_none(monkeypatch: pytest.MonkeyPatch):
    """Passing None should not force inclusion of defaults."""
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)

    provider = PortkeyProvider(
        model="gpt-4o",
        api_key="pk-test",
        config_id="pcfg_test",
    )

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 2048,
    }

    result = provider._mutate_payload(payload, temperature=None, top_p=None)

    assert "temperature" not in result
    assert "top_p" not in result
    assert "max_tokens" not in result


def test_portkey_invalid_base_url_raises(monkeypatch: pytest.MonkeyPatch):
    """Base URL must include scheme + host to avoid opaque DNS errors."""
    _clean_portkey_env(monkeypatch)
    _disable_model_validation(monkeypatch)

    with pytest.raises(ValueError):
        PortkeyProvider(
            model="gpt-4o",
            api_key="pk-test",
            config_id="pcfg_test",
            base_url="api.portkey.ai/v1",
        )
