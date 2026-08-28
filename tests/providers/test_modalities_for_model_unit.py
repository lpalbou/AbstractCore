"""Unit tests for `modalities_for_model` (registry-honest modality routes).

The contract under test: a registry HIT returns the model's capability route
keys; a registry MISS returns None — never the architecture-default text-only
fallback (`get_model_capabilities` fabricates one; discovery surfaces must
present absent knowledge as absent, ADR 0008).
"""

from __future__ import annotations

from abstractcore.providers.model_capabilities import (
    get_model_capability_routes,
    modalities_for_model,
)


def test_modalities_for_model_registry_hit_matches_capability_routes() -> None:
    routes = modalities_for_model("gpt-4o")

    assert routes == get_model_capability_routes("gpt-4o")
    assert "input.text" in routes
    assert "input.image" in routes
    assert "output.text" in routes


def test_modalities_for_model_registry_miss_returns_none() -> None:
    # `get_model_capability_routes` would fabricate a text-only default here;
    # the modality helper must return None instead.
    assert get_model_capability_routes("totally-unknown-model-zzz-9000") == ["input.text", "output.text"]
    assert modalities_for_model("totally-unknown-model-zzz-9000") is None


def test_modalities_for_model_tolerates_junk_input() -> None:
    assert modalities_for_model("") is None
    assert modalities_for_model(None) is None  # type: ignore[arg-type]
