from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


@pytest.mark.e2e
def test_two_stage_vision_fallback_end_to_end(sample_media_files, monkeypatch) -> None:
    """
    Level C end-to-end test (opt-in):
    - Calls a real vision model to extract visual context
    - Feeds that context into a real text-only model

    Enable with:
      ABSTRACT_E2E_VISION_FALLBACK=1

    Required env vars (example for OpenRouter):
      OPENROUTER_API_KEY=...
      ABSTRACT_E2E_TEXT_PROVIDER=openrouter
      ABSTRACT_E2E_TEXT_MODEL=qwen/qwen3-next-80b
      ABSTRACT_E2E_VISION_PROVIDER=openrouter
      ABSTRACT_E2E_VISION_MODEL=qwen/qwen3-vl-4b
    """
    if os.environ.get("ABSTRACT_E2E_VISION_FALLBACK") != "1":
        pytest.skip("Set ABSTRACT_E2E_VISION_FALLBACK=1 to run this test.")

    text_provider = os.environ.get("ABSTRACT_E2E_TEXT_PROVIDER", "openrouter").strip()
    text_model = os.environ.get("ABSTRACT_E2E_TEXT_MODEL", "qwen/qwen3-next-80b").strip()
    vision_provider = os.environ.get("ABSTRACT_E2E_VISION_PROVIDER", "openrouter").strip()
    vision_model = os.environ.get("ABSTRACT_E2E_VISION_MODEL", "qwen/qwen3-vl-4b").strip()

    # Ensure VisionFallbackHandler can resolve a configured vision model without relying on local files.
    dummy_cfg = SimpleNamespace(
        config=SimpleNamespace(
            vision=SimpleNamespace(
                strategy="two_stage",
                caption_provider=vision_provider,
                caption_model=vision_model,
                fallback_chain=[],
                local_models_path="",
            )
        ),
        get_status=lambda: {
            "vision": {
                "strategy": "two_stage",
                "caption_provider": vision_provider,
                "caption_model": vision_model,
                "fallback_chain": [],
                "local_models_path": "",
            }
        },
    )

    import abstractcore.config as config_module

    monkeypatch.setattr(config_module, "get_config_manager", lambda: dummy_cfg)

    from abstractcore import create_llm

    llm = create_llm(text_provider, model=text_model)
    response = llm.generate(
        "What is the dominant color in this image?",
        media=[str(sample_media_files["png"])],
        max_output_tokens=128,
    )

    content = (response.content or "").lower()
    assert "red" in content
    assert "description" not in content

