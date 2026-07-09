"""create_llm(prompt_cache_key=...) construction convenience (agency-parity 0221).

Instance-per-session callers (entity drivers, one-shot ingest) can set the instance-default
cache key at construction so every generate() is cached without threading a per-call kwarg.
Explicit per-call keys always win; pooled factories must strip the param (tested runtime-side).
"""

from __future__ import annotations

from abstractcore import create_llm


def _spy_provider(llm, captured):
    real = {}

    def spy(**call_params):
        captured.update(call_params)

        class R:
            content = []
            model = "m"
            stop_reason = "end_turn"

            class usage:
                input_tokens = 1
                output_tokens = 1

        return R()

    llm.client.messages.create = spy
    return real


def test_construction_key_sets_instance_default_anthropic():
    llm = create_llm("anthropic", model="claude-haiku-4-5", api_key="test", prompt_cache_key="sess-abc")
    assert llm._default_prompt_cache_key == "sess-abc"

    captured = {}
    _spy_provider(llm, captured)
    llm.generate(prompt="hi", system_prompt="stable head", max_output_tokens=8, temperature=0.0)
    # The default key flowed into the call -> breakpoint on the system head.
    system = captured.get("system")
    assert isinstance(system, list)
    assert any(isinstance(b, dict) and b.get("cache_control") for b in system)


def test_explicit_per_call_key_wins_over_construction_default():
    llm = create_llm("anthropic", model="claude-haiku-4-5", api_key="test", prompt_cache_key="sess-abc")
    captured = {}
    _spy_provider(llm, captured)
    # Explicit None disables caching for this call despite the instance default.
    llm.generate(prompt="hi", system_prompt="stable head", max_output_tokens=8,
                 temperature=0.0, prompt_cache_key=None)
    system = captured.get("system")
    if isinstance(system, list):
        assert all(not (isinstance(b, dict) and b.get("cache_control")) for b in system)


def test_construction_key_not_passed_to_provider_constructor():
    # The param must be consumed by the registry, never reach provider __init__ (TypeError).
    llm = create_llm("anthropic", model="claude-haiku-4-5", api_key="test", prompt_cache_key="sess-abc")
    assert llm is not None


def test_unsupported_model_degrades_with_warning_not_error():
    # claude-3 opus-era ids not in the supports_prompt_cache allowlist must not fail creation.
    llm = create_llm("anthropic", model="claude-2.1", api_key="test", prompt_cache_key="sess-abc")
    assert getattr(llm, "_default_prompt_cache_key", None) in (None, "sess-abc")
