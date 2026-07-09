from __future__ import annotations

from types import SimpleNamespace

from abstractcore.providers.anthropic_provider import AnthropicProvider


def _fake_anthropic_response() -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="ok")],
        model="claude-haiku-4-5-20251001",
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1),
    )


def _provider(monkeypatch, captured):
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")

    def fake_create(**call_params):
        captured["call_params"] = call_params
        return _fake_anthropic_response()

    monkeypatch.setattr(provider.client.messages, "create", fake_create)
    return provider


def test_prompt_cache_key_places_breakpoint_on_last_system_block(monkeypatch):
    """A prompt_cache_key must yield an EXPLICIT per-block cache_control breakpoint on the
    last system text block (caching the tools+system static head). The old top-level
    `cache_control` request param was live-verified as a silent no-op (0 writes, 0 reads)."""
    captured = {}
    provider = _provider(monkeypatch, captured)

    provider.generate(
        prompt="hello",
        system_prompt="You are a terse assistant.",
        max_output_tokens=16,
        temperature=0.0,
        prompt_cache_key="session-123",
    )

    call_params = captured.get("call_params") or {}
    assert "cache_control" not in call_params  # top-level param is the no-op path — gone
    system = call_params.get("system")
    assert isinstance(system, list) and system, system
    last_text = [b for b in system if isinstance(b, dict) and b.get("type") == "text"][-1]
    assert last_text.get("cache_control") == {"type": "ephemeral"}


def test_prompt_cache_ttl_rides_the_breakpoint(monkeypatch):
    captured = {}
    provider = _provider(monkeypatch, captured)

    provider.generate(
        prompt="hello",
        system_prompt="You are a terse assistant.",
        max_output_tokens=16,
        temperature=0.0,
        prompt_cache_key="session-123",
        prompt_cache_ttl="1h",
    )

    system = (captured.get("call_params") or {}).get("system")
    last_text = [b for b in system if isinstance(b, dict) and b.get("type") == "text"][-1]
    assert last_text.get("cache_control") == {"type": "ephemeral", "ttl": "1h"}


def test_no_cache_key_means_no_breakpoints(monkeypatch):
    captured = {}
    provider = _provider(monkeypatch, captured)

    provider.generate(
        prompt="hello",
        system_prompt="You are a terse assistant.",
        max_output_tokens=16,
        temperature=0.0,
    )

    call_params = captured.get("call_params") or {}
    assert "cache_control" not in call_params
    system = call_params.get("system")
    if isinstance(system, list):
        assert all("cache_control" not in b for b in system if isinstance(b, dict))


def test_no_system_prompt_skips_breakpoint(monkeypatch):
    # No stable head to cache: never mark volatile message content.
    captured = {}
    provider = _provider(monkeypatch, captured)

    provider.generate(prompt="hello", max_output_tokens=16, temperature=0.0, prompt_cache_key="s1")

    call_params = captured.get("call_params") or {}
    assert "cache_control" not in call_params
    for m in call_params.get("messages") or []:
        content = m.get("content")
        if isinstance(content, list):
            assert all("cache_control" not in b for b in content if isinstance(b, dict))


def test_usage_dict_normalizes_cache_fields():
    usage = SimpleNamespace(
        input_tokens=100,
        output_tokens=20,
        cache_read_input_tokens=2000,
        cache_creation_input_tokens=500,
    )
    out = AnthropicProvider._build_usage_dict(usage)
    # Anthropic's input_tokens EXCLUDES cache traffic; normalized input is inclusive.
    assert out["input_tokens"] == 2600
    assert out["cached_input_tokens"] == 2000
    assert out["cache_write_tokens"] == 500
    assert out["total_tokens"] == 2620


def test_usage_dict_without_cache_fields_omits_keys():
    usage = SimpleNamespace(input_tokens=100, output_tokens=20)
    out = AnthropicProvider._build_usage_dict(usage)
    assert out["input_tokens"] == 100
    assert "cached_input_tokens" not in out  # absent != 0 is contractual
    assert "cache_write_tokens" not in out


def test_caller_placed_breakpoints_are_respected(monkeypatch):
    """If the caller already marked blocks (owning the 4-breakpoint budget), the provider
    must NOT add its own marker — a 5th breakpoint is an API 400."""
    captured = {}
    provider = _provider(monkeypatch, captured)

    provider.generate(
        prompt="hello",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "big stable doc", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "question"},
                ],
            }
        ],
        system_prompt="You are a terse assistant.",
        max_output_tokens=16,
        temperature=0.0,
        prompt_cache_key="session-123",
    )

    system = (captured.get("call_params") or {}).get("system")
    if isinstance(system, list):
        assert all("cache_control" not in b for b in system if isinstance(b, dict))
    else:
        assert isinstance(system, str)  # untouched
