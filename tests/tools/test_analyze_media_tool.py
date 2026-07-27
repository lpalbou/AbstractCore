"""analyze_media — delegated sight (backlog 0825, ruled GO 2026-07-21).

Pins the c3977 constraints: rides the EXISTING vision-fallback config with a
loud actionable refusal when unconfigured; bounded text output; nested LLM
call runs ONE attempt (no retry stacking); classified read-only with
model_cost=True in the builtin inventory (schema v2).

Session-route resolution (operator ruling 2026-07-26, backlog 0837 item B,
amending c3977's fallback-only route): when the host stamps the hidden
``_session_route`` param AND that model declares vision, sight runs through
the session route natively — the configured fallback is SOLELY for models
that lack vision. Unstamped behavior stays byte-identical (every pre-ruling
test below runs unstamped and must stay green).
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


def test_ignored_path_refused_before_any_vision_dispatch(tmp_path: Path, monkeypatch) -> None:
    """Item 0834: analyze_media is the one file tool that exfiltrates bytes to a possibly
    remote vision route, so an .abstractignore'd path must be refused BEFORE any dispatch.
    The monkeypatched handler asserts create_description_with_trace is never reached."""
    (tmp_path / ".abstractignore").write_text("secrets/\n", encoding="utf-8")
    secret_dir = tmp_path / "secrets"
    secret_dir.mkdir()
    img = _write_real_png(secret_dir / "private.png")

    called = {"dispatched": False}

    def _tripwire(self, path, user_prompt=None):
        called["dispatched"] = True
        raise AssertionError("vision route was dispatched for an ignored path")

    monkeypatch.setattr(VisionFallbackHandler, "create_description_with_trace", _tripwire)
    out = analyze_media(str(img))
    assert out.startswith("Error:") and "ignored by .abstractignore" in out
    assert called["dispatched"] is False, "no bytes may leave the host for an ignored path"


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


# ---------------------------------------------------------------------------
# Session-route resolution (operator ruling 2026-07-26, backlog 0837 item B)
# ---------------------------------------------------------------------------


def _force_registry_vision(monkeypatch, value: bool) -> None:
    """Control the REGISTRY answer while keeping the real capability-gate code
    (MediaCapabilities fold + the tool's local-read helper) in the loop. Fake
    model names below avoid the name-pattern inference ('vl'/'vision'/'4o'/
    'multimodal') so the registry answer is the deciding signal."""
    import abstractcore.media.capabilities as media_caps

    monkeypatch.setattr(
        media_caps, "get_model_capabilities", lambda name: {"vision_support": value}
    )


def _tripwire(message: str):
    def _fail(self, *args, **kwargs):  # pragma: no cover - firing IS the failure
        raise AssertionError(message)

    return _fail


def test_stamped_vision_capable_session_route_used_natively(image_file: Path, monkeypatch) -> None:
    """Ruling case 1: the run's own route is used when its model declares
    vision — natively, with NO fallback-config consultation at all."""
    _force_registry_vision(monkeypatch, True)
    seen: dict = {}

    def _gen(self, provider, model, image_path, user_prompt=None):
        seen.update(provider=provider, model=model, path=image_path, question=user_prompt)
        return "A red square on a white table."

    monkeypatch.setattr(VisionFallbackHandler, "_generate_description", _gen)
    monkeypatch.setattr(
        VisionFallbackHandler,
        "create_description_with_trace",
        _tripwire("the configured fallback must not be consulted when the session model sees"),
    )

    out = analyze_media(
        str(image_file),
        question="what color?",
        _session_route={"provider": "endpoint:test", "model": "sees-1"},
    )
    assert "A red square" in out
    assert "(observed by endpoint:test/sees-1)" in out, "provenance must name the session route"
    assert "#FALLBACK" not in out
    assert seen == {
        "provider": "endpoint:test",
        "model": "sees-1",
        "path": str(image_file),
        "question": "what color?",
    }


def test_unstamped_call_never_touches_session_lane(image_file: Path, monkeypatch) -> None:
    """Graceful degradation: with no stamp, the session lane must not run —
    the configured fallback resolves exactly as before the ruling."""
    monkeypatch.setattr(
        VisionFallbackHandler,
        "create_description_via_route",
        _tripwire("the session lane must not run unstamped"),
    )
    monkeypatch.setattr(
        VisionFallbackHandler,
        "create_description_with_trace",
        lambda self, path, user_prompt=None: (
            "A cat.",
            {"backend": {"kind": "llm", "provider": "ollama", "model": "qwen2.5vl", "source": "primary"}},
        ),
    )
    out = analyze_media(str(image_file))
    assert "A cat." in out
    assert "(observed by ollama/qwen2.5vl)" in out


def test_stamped_text_only_session_model_uses_configured_fallback(image_file: Path, monkeypatch) -> None:
    """Ruling case 2: the fallback's sole purpose — the session model lacks
    vision, so sight resolves through the configured vision fallback."""
    _force_registry_vision(monkeypatch, False)
    monkeypatch.setattr(
        VisionFallbackHandler,
        "create_description_via_route",
        _tripwire("a text-only session model must never receive the image"),
    )
    monkeypatch.setattr(
        VisionFallbackHandler,
        "create_description_with_trace",
        lambda self, path, user_prompt=None: (
            "Seen by the fallback.",
            {"backend": {"kind": "llm", "provider": "ollama", "model": "qwen2.5vl", "source": "primary"}},
        ),
    )
    out = analyze_media(
        str(image_file), _session_route={"provider": "lmstudio", "model": "text-1"}
    )
    assert "Seen by the fallback." in out
    assert "(observed by ollama/qwen2.5vl)" in out
    assert "#FALLBACK" not in out, "the fallback serving a text-only model is its PURPOSE, not a degradation"


def test_stamped_text_only_refusal_names_model_and_where_to_configure(image_file: Path, monkeypatch) -> None:
    """Ruling case 3: no session vision + no fallback -> the refusal names
    WHICH model lacked vision and WHERE to configure, and never suggests
    pointing the model at itself."""
    _force_registry_vision(monkeypatch, False)

    def _raise(self, path, user_prompt=None):
        raise VisionNotConfiguredError("Vision fallback is disabled")

    monkeypatch.setattr(VisionFallbackHandler, "create_description_with_trace", _raise)
    out = analyze_media(
        str(image_file), _session_route={"provider": "lmstudio", "model": "text-1"}
    )
    assert out.startswith("Error:")
    assert "'lmstudio/text-1'" in out, "the refusal must name WHICH model lacked vision"
    assert "does not declare vision support" in out
    assert "abstractcore --config" in out, "the refusal must name WHERE to configure"
    assert "vision-CAPABLE" in out
    assert "your current chat endpoint/model" not in out, "never ask an operator to point a model at itself"


def test_session_route_failure_falls_back_with_fallback_label(image_file: Path, monkeypatch) -> None:
    """A vision-capable session route that FAILS at runtime degrades to an
    already-configured fallback — loudly labeled (#FALLBACK house rule), with
    provenance naming who actually saw."""
    _force_registry_vision(monkeypatch, True)

    def _gen(self, provider, model, image_path, user_prompt=None):
        raise RuntimeError("images rejected by endpoint")

    monkeypatch.setattr(VisionFallbackHandler, "_generate_description", _gen)
    monkeypatch.setattr(
        VisionFallbackHandler,
        "create_description_with_trace",
        lambda self, path, user_prompt=None: (
            "Seen by the fallback.",
            {"backend": {"kind": "llm", "provider": "ollama", "model": "qwen2.5vl", "source": "primary"}},
        ),
    )
    out = analyze_media(
        str(image_file), _session_route={"provider": "endpoint:test", "model": "sees-1"}
    )
    assert "Seen by the fallback." in out
    assert "#FALLBACK" in out
    assert "endpoint:test/sees-1" in out and "images rejected by endpoint" in out
    assert "(observed by ollama/qwen2.5vl)" in out


def test_session_route_failure_without_fallback_reports_runtime_cause(image_file: Path, monkeypatch) -> None:
    """Session model declares vision, its attempt fails, nothing configured:
    the error must surface the LIVE cause and state that no fallback is
    required for a vision-capable model — never a bare 'configure it'."""
    _force_registry_vision(monkeypatch, True)

    def _gen(self, provider, model, image_path, user_prompt=None):
        raise RuntimeError("boom")

    def _raise(self, path, user_prompt=None):
        raise VisionNotConfiguredError("Vision fallback is disabled")

    monkeypatch.setattr(VisionFallbackHandler, "_generate_description", _gen)
    monkeypatch.setattr(VisionFallbackHandler, "create_description_with_trace", _raise)
    out = analyze_media(
        str(image_file), _session_route={"provider": "endpoint:test", "model": "sees-1"}
    )
    assert out.startswith("Error:")
    assert "'endpoint:test/sees-1'" in out and "declares vision" in out
    assert "boom" in out, "the LIVE cause must be surfaced"
    assert "fallbacks are solely for models without vision" in out
    assert "no vision model is configured for delegated sight" not in out, (
        "a live session-route failure must not be misdiagnosed as the config gap"
    )


def test_both_routes_failing_reports_both_causes(image_file: Path, monkeypatch) -> None:
    _force_registry_vision(monkeypatch, True)

    def _gen(self, provider, model, image_path, user_prompt=None):
        raise RuntimeError("session down")

    def _raise(self, path, user_prompt=None):
        from abstractcore.media.vision_fallback import VisionGenerationError

        raise VisionGenerationError("fallback down")

    monkeypatch.setattr(VisionFallbackHandler, "_generate_description", _gen)
    monkeypatch.setattr(VisionFallbackHandler, "create_description_with_trace", _raise)
    out = analyze_media(
        str(image_file), _session_route={"provider": "endpoint:test", "model": "sees-1"}
    )
    assert out.startswith("Error:") and "both routes" in out
    assert "session down" in out and "fallback down" in out
    assert "abstractcore --config" not in out, "two live failures must not be sent to the config fix"


def test_empty_session_observation_degrades_to_fallback(image_file: Path, monkeypatch) -> None:
    """A blank session caption is a soft route failure, not an answer — the
    configured fallback may still serve, labeled."""
    _force_registry_vision(monkeypatch, True)
    monkeypatch.setattr(
        VisionFallbackHandler,
        "_generate_description",
        lambda self, provider, model, image_path, user_prompt=None: "   ",
    )
    monkeypatch.setattr(
        VisionFallbackHandler,
        "create_description_with_trace",
        lambda self, path, user_prompt=None: ("Seen by the fallback.", {}),
    )
    out = analyze_media(
        str(image_file), _session_route={"provider": "endpoint:test", "model": "sees-1"}
    )
    assert "Seen by the fallback." in out
    assert "#FALLBACK" in out and "empty observation" in out


def test_malformed_stamp_degrades_to_configured_fallback(image_file: Path, monkeypatch) -> None:
    """A present-but-malformed stamp is a HOST bug: warn and degrade to the
    configured fallback — never fail the analysis, never dispatch on garbage."""
    monkeypatch.setattr(
        VisionFallbackHandler,
        "create_description_via_route",
        _tripwire("a malformed stamp must not reach the session lane"),
    )
    monkeypatch.setattr(
        VisionFallbackHandler,
        "create_description_with_trace",
        lambda self, path, user_prompt=None: ("A cat.", {}),
    )
    for bad in ("not-json", {"provider": "lmstudio"}, {"model": "x"}, 42, {}):
        out = analyze_media(str(image_file), _session_route=bad)
        assert "A cat." in out


def test_session_route_param_hidden_from_llm_schema() -> None:
    """The stamp is host-injected only: hidden from the model-facing schema
    (hide_args, the _registry_namespace precedent) while the callable still
    accepts it."""
    import inspect

    params = analyze_media.tool_definition.parameters
    assert "_session_route" not in params
    assert "_session_route" not in analyze_media.tool_definition.to_dict()["parameters"]
    assert "_session_route" in inspect.signature(analyze_media).parameters


def test_session_discipline_and_no_raw_transport_reach_create_llm(image_file: Path, monkeypatch) -> None:
    """Drives the REAL nested path for the session leg + fallthrough: the
    single-attempt/bounded-timeout discipline reaches every create_llm
    (session AND configured), and a base_url smuggled into the stamp NEVER
    reaches create_llm (the egress bound: provider+model only)."""
    import abstractcore

    _force_registry_vision(monkeypatch, True)
    calls: list = []

    class _FakeVisionLLM:
        def __init__(self, provider, model=None, **kwargs):
            calls.append({"provider": provider, "model": model, **kwargs})

        def generate(self, prompt, media=None):
            raise RuntimeError("simulated vision endpoint failure")

    monkeypatch.setattr(abstractcore, "create_llm", _FakeVisionLLM)

    class _Cfg:
        strategy = "auto"
        caption_provider = "ollama"
        caption_model = "qwen2.5vl"
        fallback_chain = []
        local_models_path = None

    class _Mgr:
        class config:
            vision = _Cfg()

    monkeypatch.setattr(
        "abstractcore.config.get_config_manager", lambda: _Mgr(), raising=False
    )

    out = analyze_media(
        str(image_file),
        _session_route={
            "provider": "endpoint:test",
            "model": "sees-1",
            "base_url": "https://evil.example",
        },
    )
    assert out.startswith("Error:") and "both routes" in out
    assert len(calls) == 2, f"expected session + configured-primary create_llm calls, got {len(calls)}"
    assert calls[0]["provider"] == "endpoint:test" and calls[0]["model"] == "sees-1"
    assert "base_url" not in calls[0], "raw transport must never ride the stamp into create_llm"
    for c in calls:
        rc = c.get("retry_config")
        assert rc is not None and rc.max_attempts == 1, "single-attempt must reach every nested create_llm"
        assert c.get("timeout") == pytest.approx(120.0), "bounded timeout must reach every nested create_llm"
