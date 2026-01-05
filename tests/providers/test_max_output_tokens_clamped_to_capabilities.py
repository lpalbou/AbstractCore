from __future__ import annotations

from abstractcore.providers.anthropic_provider import AnthropicProvider


def test_prepare_generation_kwargs_clamps_max_output_tokens_to_model_cap() -> None:
    provider = AnthropicProvider(model="claude-haiku-4-5", api_key="test")
    # Even if upstream requests above the model cap, providers must clamp to avoid 400s.
    kwargs = provider._prepare_generation_kwargs(max_output_tokens=999999, temperature=0)
    assert kwargs["max_output_tokens"] <= int(provider.max_output_tokens)



