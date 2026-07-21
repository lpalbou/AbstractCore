"""analyze_media — delegated sight (backlog 0825, ruled GO 2026-07-21).

Pins the c3977 constraints: rides the EXISTING vision-fallback config with a
loud actionable refusal when unconfigured; bounded text output; nested LLM
call runs ONE attempt (no retry stacking); classified read-only with
model_cost=True in the builtin inventory (schema v2).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from abstractcore.media.vision_fallback import VisionFallbackHandler, VisionNotConfiguredError
from abstractcore.tools.common_tools import _ANALYZE_MEDIA_MAX_CHARS, analyze_media


def _write_real_png(path: Path) -> Path:
    """A genuinely decodable 4x4 PNG (the tool's honesty gate PIL-verifies)."""
    from PIL import Image

    Image.new("RGB", (4, 4), (200, 30, 30)).save(path, format="PNG")
    return path


@pytest.fixture()
def image_file(tmp_path: Path) -> Path:
    return _write_real_png(tmp_path / "shot.png")


def test_missing_file_refuses(tmp_path: Path) -> None:
    out = analyze_media(str(tmp_path / "nope.jpg"))
    assert out.startswith("Error:") and "does not exist" in out


def test_non_image_refuses_honestly(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    out = analyze_media(str(video))
    assert out.startswith("Error:") and "images only" in out and "frame" in out


def test_unconfigured_vision_refuses_with_setup_hint(image_file: Path, monkeypatch) -> None:
    def _raise(self, path, user_prompt=None):
        raise VisionNotConfiguredError("Vision fallback is disabled")

    monkeypatch.setattr(VisionFallbackHandler, "create_description_with_trace", _raise)
    out = analyze_media(str(image_file))
    assert out.startswith("Error:")
    assert "abstractcore --config" in out, "refusal must carry the actionable setup hint"


def test_corrupt_image_refuses_never_fabricates(tmp_path: Path, monkeypatch) -> None:
    """P0 (adversary 2026-07-21): a truncated/corrupt capture — the realistic
    mid-loop input — must REFUSE before dispatch, never reach the vision model
    (which would describe a placeholder and get a provenance stamp)."""
    corrupt = tmp_path / "capture.jpg"
    corrupt.write_bytes(b"\xff\xd8\xff\xe0not-a-real-jpeg")

    def _must_not_run(self, path, user_prompt=None):  # pragma: no cover
        raise AssertionError("vision model must NOT be called for a non-decodable image")

    monkeypatch.setattr(VisionFallbackHandler, "create_description_with_trace", _must_not_run)
    out = analyze_media(str(corrupt))
    assert out.startswith("Error:") and "did not decode as a valid image" in out


def test_renamed_nonimage_with_image_suffix_refuses(tmp_path: Path, monkeypatch) -> None:
    fake = tmp_path / "notes.png"
    fake.write_text("this is plain text renamed to .png")

    monkeypatch.setattr(
        VisionFallbackHandler,
        "create_description_with_trace",
        lambda self, path, user_prompt=None: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )
    out = analyze_media(str(fake))
    assert out.startswith("Error:") and "did not decode" in out


def test_configured_route_returns_observation_with_provenance(image_file: Path, monkeypatch) -> None:
    def _describe(self, path, user_prompt=None):
        assert Path(path) == image_file
        assert user_prompt == "what color?"
        return "A red square on a white table.", {
            "backend": {"kind": "llm", "provider": "ollama", "model": "qwen2.5vl", "source": "primary"}
        }

    monkeypatch.setattr(VisionFallbackHandler, "create_description_with_trace", _describe)
    out = analyze_media(str(image_file), question="what color?")
    assert "A red square" in out
    assert "(observed by ollama/qwen2.5vl)" in out, "the calling agent must see WHO saw"


def test_local_backend_gets_provenance(image_file: Path, monkeypatch) -> None:
    """P2: local-model traces carry a model but no provider — provenance must
    still render as local/<model>, not silently drop."""
    monkeypatch.setattr(
        VisionFallbackHandler,
        "create_description_with_trace",
        lambda self, path, user_prompt=None: ("A cat.", {"backend": {"kind": "local", "model": "smolvlm"}}),
    )
    out = analyze_media(str(image_file))
    assert "(observed by local/smolvlm)" in out


def test_configured_route_failure_is_not_misdiagnosed(image_file: Path, monkeypatch) -> None:
    """P1: a configured-but-failing route must NOT tell the user to configure
    it — it must surface the runtime cause."""
    from abstractcore.media.vision_fallback import VisionGenerationError

    def _raise(self, path, user_prompt=None):
        raise VisionGenerationError("All vision fallback providers failed: connection refused")

    monkeypatch.setattr(VisionFallbackHandler, "create_description_with_trace", _raise)
    out = analyze_media(str(image_file))
    assert out.startswith("Error:")
    assert "connection refused" in out
    assert "abstractcore --config" not in out, "must not send a live-failure to the config fix"
    assert "route is configured" in out


def test_output_is_bounded_with_truncation_label(image_file: Path, monkeypatch) -> None:
    def _describe(self, path, user_prompt=None):
        return "x" * (_ANALYZE_MEDIA_MAX_CHARS * 3), {}

    monkeypatch.setattr(VisionFallbackHandler, "create_description_with_trace", _describe)
    out = analyze_media(str(image_file))
    assert "#TRUNCATION" in out
    assert len(out) < _ANALYZE_MEDIA_MAX_CHARS + 200


def test_single_attempt_and_timeout_reach_create_llm_on_all_paths(image_file: Path, monkeypatch) -> None:
    """P1 (adversary: the old pin was vacuous — it monkeypatched the handler
    away). This drives the REAL path: a configured primary that FAILS plus a
    fallback-chain entry, and asserts create_llm received max_attempts==1 AND
    a bounded timeout on BOTH constructions. Reverting the `**self._llm_kwargs`
    splat makes this fail."""
    import abstractcore

    calls: list = []

    class _FakeVisionLLM:
        def __init__(self, provider, model, **kwargs):
            calls.append({"provider": provider, "model": model, **kwargs})

        def generate(self, prompt, media=None):
            raise RuntimeError("simulated vision endpoint failure")

    monkeypatch.setattr(abstractcore, "create_llm", _FakeVisionLLM)

    class _Cfg:
        strategy = "auto"
        caption_provider = "ollama"
        caption_model = "qwen2.5vl"
        fallback_chain = [{"provider": "openai", "model": "gpt-4o-mini"}]
        local_models_path = None

    class _Mgr:
        class config:
            vision = _Cfg()

    monkeypatch.setattr(
        "abstractcore.config.get_config_manager", lambda: _Mgr(), raising=False
    )

    out = analyze_media(str(image_file))
    # Both providers were tried (primary + one fallback), both failed → the
    # configured-route-failure message (not the config-gap message).
    assert out.startswith("Error:") and "route is configured" in out
    assert len(calls) == 2, f"expected primary + one fallback create_llm calls, got {len(calls)}"
    for c in calls:
        rc = c.get("retry_config")
        assert rc is not None and rc.max_attempts == 1, "single-attempt must reach every nested create_llm"
        assert c.get("timeout") == pytest.approx(120.0), "bounded timeout must reach every nested create_llm"


def test_empty_observation_is_an_error(image_file: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        VisionFallbackHandler,
        "create_description_with_trace",
        lambda self, path, user_prompt=None: ("   ", {}),
    )
    out = analyze_media(str(image_file))
    assert out.startswith("Error:") and "empty observation" in out


def test_inventory_classifies_analyze_media_model_cost() -> None:
    from abstractcore.tools.inventory import (
        INVENTORY_SCHEMA_VERSION,
        list_builtin_tool_inventory,
    )

    assert INVENTORY_SCHEMA_VERSION >= 2
    rows = {d.name: d for d in list_builtin_tool_inventory()}
    row = rows.get("analyze_media")
    assert row is not None, "analyze_media must be in the builtin inventory"
    assert row.mutating is False
    assert row.remote_write_capable is False
    assert row.model_cost is True, "nested-LLM cost must be a visible classification fact"
    assert row.to_dict()["model_cost"] is True
    # Every other current builtin carries model_cost=False.
    others = [d for d in rows.values() if d.name != "analyze_media"]
    assert all(d.model_cost is False for d in others)


def test_vision_fallback_default_llm_kwargs_unchanged() -> None:
    """The generate() fallback lane must be byte-identical: no llm_kwargs by
    default (the tool's single-attempt discipline is opt-in, never imposed
    on the caption-inside-generate path)."""
    handler = VisionFallbackHandler(config_manager=object())
    assert handler._llm_kwargs == {}
