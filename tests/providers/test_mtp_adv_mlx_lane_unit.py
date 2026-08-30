"""Adversarial pins for the MLX native-MTP lane.

The MLX lane swaps the whole runtime -- target model, processor, tokenizer,
`generate_fn` and `stream_generate_fn` all change when speculation engages -- so
the interesting failures are not "does it go fast" but "what else did it change
on the way". These tests exercise the decision logic with fake mlx-vlm modules
injected into `sys.modules`; nothing here loads a model, imports real MLX, or
touches the network.

Provider instances are built with `__new__` (the pattern the existing MLX
prompt-cache unit tests use) so `__init__` never runs and no weights are needed.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional

import pytest

from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.providers.speculation import (
    SpeculationRequest,
    SpeculationUnavailableError,
)


class _FakeLogger:
    def __init__(self) -> None:
        self.warnings: List[str] = []
        self.infos: List[str] = []

    def warning(self, msg: str, *a: Any, **k: Any) -> None:
        self.warnings.append(str(msg))

    def info(self, msg: str, *a: Any, **k: Any) -> None:
        self.infos.append(str(msg))

    def debug(self, msg: str, *a: Any, **k: Any) -> None:
        pass

    def error(self, msg: str, *a: Any, **k: Any) -> None:
        pass


def _provider(**attrs: Any) -> MLXProvider:
    """An MLXProvider with only the fields these paths read."""
    p = MLXProvider.__new__(MLXProvider)
    p.logger = _FakeLogger()
    p.model = "mlx-community/Qwen3.8-27B-4bit"
    p.model_capabilities = {}
    p._speculation_request = None
    p._mtp_drafter = None
    p._mtp_kind = None
    p._mtp_block_size = None
    p._mtp_processor = None
    p._mtp_reason = None
    p._mtp_drafter_id = None
    p._mtp_outcome_at_load = None
    p._mtp_prompt_cache_warned = False
    for key, value in attrs.items():
        setattr(p, key, value)
    return p


# ---------------------------------------------------------------------------
# Load order: the drafter is 256 MB, the target is 15 GB.
# ---------------------------------------------------------------------------


class _LoadRecorder:
    """Fake `mlx_vlm` that records what got loaded, in order."""

    def __init__(self, drafter_raises: bool = False) -> None:
        self.calls: List[str] = []
        self.drafter_raises = drafter_raises

    def install(self, monkeypatch) -> None:
        recorder = self

        def vlm_load(path, *a, **k):
            recorder.calls.append("target")
            return object(), types.SimpleNamespace(tokenizer=object())

        def load_drafter(repo, *a, **k):
            recorder.calls.append("drafter")
            if recorder.drafter_raises:
                raise RuntimeError("drafter repo not found")
            return object(), "mtp"

        mlx_vlm = types.ModuleType("mlx_vlm")
        mlx_vlm.load = vlm_load
        mlx_vlm.generate = lambda *a, **k: types.SimpleNamespace(text="")
        mlx_vlm.stream_generate = lambda *a, **k: iter(())

        speculative = types.ModuleType("mlx_vlm.speculative")
        drafters = types.ModuleType("mlx_vlm.speculative.drafters")
        drafters.load_drafter = load_drafter
        speculative.drafters = drafters
        mlx_vlm.speculative = speculative

        monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
        monkeypatch.setitem(sys.modules, "mlx_vlm.speculative", speculative)
        monkeypatch.setitem(sys.modules, "mlx_vlm.speculative.drafters", drafters)


def test_the_cheap_drafter_is_loaded_before_the_expensive_target(monkeypatch):
    """A drafter failure must not cost a 15 GB load first.

    `_enter_mtp_lane` returns False on ANY failure so the caller falls back to
    the ordinary mlx-lm load. That fallback re-loads the target. So if the
    target is loaded FIRST and the drafter then fails, the process pays 15 GB,
    throws it away, and pays 15 GB again -- on a 128 GB machine that is a
    survivable ~30 GB spike, on a 32 GB one it is an OOM, and either way it is
    pure waste for a failure the 256 MB drafter load would have surfaced first.

    The drafter is the cheap, fallible half (a repo name that may not exist, a
    `kind` that may not resolve). Load it first.
    """
    recorder = _LoadRecorder()
    recorder.install(monkeypatch)

    provider = _provider(_speculation_request=SpeculationRequest(mode="native_mtp"))
    ok = provider._enter_mtp_lane(
        "mlx-community/Qwen3.8-27B-4bit",
        {"drafter": "mlx-community/Qwen3.8-27B-MTP-4bit", "block_size": 3},
    )

    assert ok is True
    assert recorder.calls == ["drafter", "target"], (
        "the 15 GB target was loaded before the 256 MB drafter "
        f"(order was {recorder.calls}). Load the drafter first so a bad drafter "
        "costs 256 MB instead of 15 GB plus a second 15 GB on fallback."
    )


def test_failed_drafter_does_not_leave_a_loaded_target_behind(monkeypatch):
    """When the drafter fails, nothing big should have been loaded yet."""
    recorder = _LoadRecorder(drafter_raises=True)
    recorder.install(monkeypatch)

    provider = _provider(_speculation_request=SpeculationRequest(mode="native_mtp"))
    ok = provider._enter_mtp_lane(
        "mlx-community/Qwen3.8-27B-4bit",
        {"drafter": "does-not-exist/nope", "block_size": 3},
    )

    assert ok is False, "a failed drafter must fall back, not activate the lane"
    assert "target" not in recorder.calls, (
        "the target was loaded even though the drafter failed; the caller now "
        f"loads it a SECOND time on fallback (calls were {recorder.calls})"
    )
    assert provider._mtp_drafter is None
    warnings = provider.logger.warnings
    assert any("mtp_drafter_load_failed" in w for w in warnings), (
        f"a failed drafter load must warn with its reason slug; got {warnings}"
    )


def test_failed_drafter_raises_under_require_acceleration(monkeypatch):
    recorder = _LoadRecorder(drafter_raises=True)
    recorder.install(monkeypatch)

    provider = _provider(
        _speculation_request=SpeculationRequest(
            mode="native_mtp", require_acceleration=True
        )
    )
    with pytest.raises(SpeculationUnavailableError) as excinfo:
        provider._enter_mtp_lane(
            "mlx-community/Qwen3.8-27B-4bit",
            {"drafter": "does-not-exist/nope", "block_size": 3},
        )
    assert excinfo.value.reason == "mtp_drafter_load_failed"


# ---------------------------------------------------------------------------
# `used` must track the drafter, never the request.
# ---------------------------------------------------------------------------


def test_used_is_false_whenever_no_drafter_is_loaded():
    provider = _provider(_speculation_request=SpeculationRequest(mode="native_mtp"))
    assert provider._mtp_active is False
    assert provider._mtp_kwargs(None) == {}

    outcome = provider._mtp_outcome(bool(provider._mtp_kwargs(None)))
    assert outcome.used is False
    assert outcome.requested is True
    assert outcome.to_metadata()["used"] is False
    assert outcome.to_metadata().get("reason"), (
        "an unaccelerated call that was REQUESTED must carry a reason"
    )


def test_no_speculation_metadata_claim_when_nothing_was_requested():
    provider = _provider()
    outcome = provider._mtp_outcome(False)
    assert outcome.requested is False
    assert outcome.used is False


def test_uninitialised_provider_reads_as_lane_off():
    """The prompt-cache harnesses build providers via __new__ with no MTP attrs."""
    bare = MLXProvider.__new__(MLXProvider)
    assert bare._mtp_active is False
    assert bare._mtp_kwargs(None) == {}


def test_used_requires_a_drafter_object_not_merely_a_capability_entry():
    """A registry entry naming a drafter is not a drafter."""
    provider = _provider(
        _speculation_request=SpeculationRequest(mode="native_mtp"),
        model_capabilities={
            "speculation": {
                "native_mtp": True,
                "runtimes": {
                    "mlx": {
                        "mode": "drafter_repo",
                        "drafter": "mlx-community/Qwen3.8-27B-MTP-4bit",
                    }
                },
            }
        },
    )
    assert provider._mtp_active is False
    assert provider._mtp_outcome(bool(provider._mtp_kwargs(None))).used is False


# ---------------------------------------------------------------------------
# The lane must not quietly change the vision or prompt-cache contracts.
# ---------------------------------------------------------------------------


def test_vision_calls_do_not_claim_acceleration():
    provider = _provider(
        _speculation_request=SpeculationRequest(mode="native_mtp"),
        _mtp_drafter=object(),
        _mtp_kind="mtp",
        _mtp_drafter_id="mlx-community/Qwen3.8-27B-MTP-4bit",
    )
    assert provider._mtp_kwargs(None), "text calls should use the drafter"
    assert provider._mtp_kwargs(object()) == {}, (
        "v1 is text-only: an input_embeddings (vision) call must not use MTP"
    )
    assert provider._mtp_outcome(bool(provider._mtp_kwargs(object()))).used is False


def test_a_media_call_on_the_mtp_lane_fails_closed(monkeypatch):
    """The vision add-on's `input_embeddings` is an mlx-lm entry point.

    `mlx_vlm.generate` takes `image=`/`audio=`/`video=` and a bare `**kwargs`,
    so an `input_embeddings=` passed through does NOT raise -- it is swallowed
    and ignored. That is the silent-image-drop shape: the caller attaches a
    picture and the lane answers as if there were none.

    Since the MTP lane replaces `generate_fn` wholesale, EVERY call goes through
    mlx-vlm once speculation is on, vision calls included. Three ways out were
    on the table: translate the media into mlx-vlm's vocabulary (impossible here
    -- the add-on has already reduced the image to embeddings, so there is no
    image left to hand over), route the call back to mlx-lm (needs a SECOND
    resident copy of a 15 GB target), or refuse. Refusing is the right one, and
    it matches the precedent three call sites down: "Either would answer from
    text while the lane had already claimed sight. Fail closed instead."

    What this test pins is the part that actually matters: the refusal happens
    BEFORE mlx-vlm is invoked, so the embeddings are never handed to something
    that would drop them.
    """
    from abstractcore.exceptions import ProviderAPIError

    recorder = _LoadRecorder()
    recorder.install(monkeypatch)
    invoked: List[str] = []
    sys.modules["mlx_vlm"].generate = lambda *a, **k: invoked.append("generate")
    sys.modules["mlx_vlm"].stream_generate = lambda *a, **k: invoked.append("stream")

    provider = _provider(
        _speculation_request=SpeculationRequest(mode="native_mtp"),
        _mtp_drafter=object(),
        _mtp_kind="mtp",
        _mtp_processor=object(),
    )
    sentinel = object()

    with pytest.raises(ProviderAPIError) as excinfo:
        provider._mtp_generate_fn(
            object(), object(), prompt="hi", input_embeddings=sentinel, max_tokens=8
        )
    message = str(excinfo.value)
    assert "speculation" in message.lower(), (
        "the refusal must name speculation as the cause and how to turn it off, "
        f"got: {message}"
    )

    with pytest.raises(ProviderAPIError):
        provider._mtp_stream_generate_fn(
            object(), object(), prompt="hi", input_embeddings=sentinel, max_tokens=8
        )

    assert invoked == [], (
        "mlx-vlm was invoked on a media request before the lane refused it "
        f"({invoked}); the embeddings reached a callee that ignores them"
    )


def test_text_calls_still_reach_mlx_vlm_after_the_media_guard(monkeypatch):
    """The fail-closed guard must not have broken the lane it protects."""
    recorder = _LoadRecorder()
    recorder.install(monkeypatch)
    seen: Dict[str, Any] = {}

    def fake_generate(model, processor, text, **kwargs):
        seen.update(kwargs)
        return types.SimpleNamespace(text="ok")

    sys.modules["mlx_vlm"].generate = fake_generate

    drafter = object()
    provider = _provider(
        _speculation_request=SpeculationRequest(mode="native_mtp"),
        _mtp_drafter=drafter,
        _mtp_kind="mtp",
        _mtp_block_size=3,
        _mtp_processor=object(),
    )
    out = provider._mtp_generate_fn(object(), object(), prompt="hi", max_tokens=8)

    assert out == "ok", "the adapter must unwrap mlx-vlm's result object to text"
    assert seen.get("draft_model") is drafter
    assert seen.get("draft_kind") == "mtp"
    assert seen.get("draft_block_size") == 3
    assert "input_embeddings" not in seen


def test_prompt_cache_is_declined_and_warned_exactly_once():
    """Mixing the mlx-lm keyed snapshot cache into mlx-vlm would desync it."""
    provider = _provider(
        _speculation_request=SpeculationRequest(mode="native_mtp"),
        _mtp_drafter=object(),
        _mtp_kind="mtp",
    )
    first = provider._mtp_call_kwargs({"prompt_cache": ["warm"], "max_tokens": 8})
    second = provider._mtp_call_kwargs({"prompt_cache": ["warm"], "max_tokens": 8})

    assert "prompt_cache" not in first and "prompt_cache" not in second, (
        "an mlx-lm prompt cache must never reach mlx-vlm's speculative loop"
    )
    assert len(provider.logger.warnings) == 1, (
        "declining the warm cache must warn once per provider, not once per call "
        f"(got {len(provider.logger.warnings)} warnings)"
    )
    assert "prompt cache" in provider.logger.warnings[0].lower()


# ---------------------------------------------------------------------------
# Planning: never guess a drafter.
# ---------------------------------------------------------------------------


def test_no_drafter_is_invented_when_the_registry_has_none():
    provider = _provider(
        _speculation_request=SpeculationRequest(mode="native_mtp"),
        model_capabilities={
            "speculation": {
                "native_mtp": True,
                "runtimes": {"llama_cpp": {"mode": "embedded"}},
            }
        },
    )
    assert provider._plan_mtp_lane("some/target") is None
    assert any(
        "no_mtp_drafter_for_model" in w for w in provider.logger.warnings
    ), provider.logger.warnings


def test_planning_is_inert_when_speculation_was_never_requested():
    provider = _provider()
    assert provider._plan_mtp_lane("some/target") is None
    assert provider.logger.warnings == [], (
        "a provider nobody asked to accelerate must not warn about speculation"
    )


# ---------------------------------------------------------------------------
# Per-call `speculation=`: consumed, and honored-or-reported. Never silent.
#
# These assert BEHAVIOUR. An earlier version of this test grepped the source for
# `kwargs.pop("speculation")`, which was a bad proxy twice over: it missed the
# real fix (`kwargs.pop("speculation", None)` -- the default argument breaks the
# substring) and it would have passed for a line that pops the kwarg and throws
# it away, which is precisely the defect being hunted. A consumer that discards
# is indistinguishable from no consumer at the source level and obviously
# distinguishable at the behaviour level, so test the behaviour.
# ---------------------------------------------------------------------------


def _lane_off_provider() -> MLXProvider:
    """A provider whose MTP lane is NOT loaded (the common case)."""
    return _provider()


def _accelerated_provider() -> MLXProvider:
    """A provider whose MTP lane IS loaded."""
    return _provider(
        _speculation_request=SpeculationRequest(mode="native_mtp"),
        _mtp_drafter=object(),
        _mtp_kind="mtp",
        _mtp_block_size=3,
        _mtp_drafter_id="mlx-community/Qwen3.8-27B-MTP-4bit",
        _mtp_processor=object(),
    )


def test_per_call_strict_request_raises_when_the_lane_is_off():
    """A: `require_acceleration=True` per call must RAISE, not warn."""
    provider = _lane_off_provider()
    with pytest.raises(SpeculationUnavailableError) as excinfo:
        provider._apply_per_call_speculation(
            {"mode": "native_mtp", "require_acceleration": True}
        )
    assert excinfo.value.reason == "speculation_is_load_time", (
        "the strict per-call refusal must carry a machine-readable reason "
        f"callers can branch on; got {excinfo.value.reason!r}"
    )


def test_per_call_non_strict_request_warns_and_is_reported_in_metadata():
    """B: non-strict per call warns AND shows up in the response metadata.

    A warning alone is not enough -- it is one log line the caller may never
    see. The outcome has to ride the response.
    """
    provider = _lane_off_provider()
    provider._apply_per_call_speculation({"mode": "native_mtp"})

    assert any(
        "speculation_is_load_time" in w for w in provider.logger.warnings
    ), provider.logger.warnings

    metadata = provider._mtp_outcome(bool(provider._mtp_kwargs(None))).to_metadata()
    assert metadata == {
        "requested": True,
        "mode": "native_mtp",
        "used": False,
        "reason": "speculation_is_load_time",
    }, metadata


def test_per_call_off_is_honored_on_an_accelerated_provider():
    """C: `{'mode': 'off'}` per call is the one per-call change that IS honorable.

    Skipping the drafter needs no reload, so this must actually take effect --
    and it must be scoped to the call, not latch on for the provider's life.
    """
    provider = _accelerated_provider()

    # Baseline: with no per-call request the drafter is used.
    provider._apply_per_call_speculation(None)
    assert provider._mtp_kwargs(None), "the loaded lane should be in use"
    assert provider._mtp_outcome(bool(provider._mtp_kwargs(None))).used is True

    # Per-call off: honored.
    provider._apply_per_call_speculation({"mode": "off"})
    assert provider._mtp_kwargs(None) == {}, (
        "a per-call speculation={'mode': 'off'} was ignored: the drafter kwargs "
        "were still handed to mlx-vlm"
    )
    outcome = provider._mtp_outcome(bool(provider._mtp_kwargs(None)))
    assert outcome.used is False
    assert provider._mtp_drafter_id == "mlx-community/Qwen3.8-27B-MTP-4bit", (
        "turning speculation off for one call must not erase which drafter the "
        "provider has loaded"
    )

    # And it does not latch: the next call without the kwarg accelerates again.
    provider._apply_per_call_speculation(None)
    assert provider._mtp_kwargs(None), (
        "per-call `off` leaked into the following call -- it must be scoped to "
        "the call that asked for it"
    )


def test_per_call_kwarg_is_consumed_before_it_can_reach_mlx_vlm(monkeypatch):
    """D: the kwarg is eaten at the seam, so no callee ever sees it.

    `mlx_vlm.generate` absorbs unknown kwargs into `**kwargs`, so a leaked
    `speculation={...}` would neither raise nor do anything -- the silent shape.
    """
    provider = _lane_off_provider()
    seen: Dict[str, Any] = {}

    def fake_core(prompt, **kwargs):
        seen.update(kwargs)
        return "response"

    monkeypatch.setattr(provider, "_generate_core", fake_core)
    monkeypatch.setattr(
        "abstractcore.media.delivery.attach_media_report", lambda out, report: out
    )

    provider._generate_internal("hi", speculation={"mode": "native_mtp"})

    assert seen, (
        "non-vacuity guard: _generate_core was never reached, so 'speculation "
        "not in seen' would pass for the wrong reason"
    )
    assert "speculation" not in seen, (
        "the per-call `speculation` kwarg was forwarded into _generate_core and "
        "on toward mlx-vlm, which would absorb and ignore it"
    )
    assert any(
        "speculation_is_load_time" in w for w in provider.logger.warnings
    ), "consumed but not reported -- the exact defect this test replaced"


def test_streaming_does_not_downgrade_a_strict_per_call_raise(monkeypatch):
    """(b) The streaming lane must not turn the strict refusal into a chunk.

    `_stream_generate`'s `except Exception as e:` yields
    `GenerateResponse(content=f"Error: {e}", finish_reason="error")`, which would
    convert a `require_acceleration=True` refusal into something that merely
    LOOKS like an answer. It does not, because `_generate_internal` is a plain
    function, not a generator: `_apply_per_call_speculation` runs eagerly at call
    time, before `_generate_core` ever builds the stream. This pins that -- if
    anyone makes `_generate_internal` a generator, or moves the speculation
    consumer inside `_stream_generate`, the raise silently becomes a chunk.
    """
    provider = _lane_off_provider()
    built = []

    def fake_core(prompt, **kwargs):
        built.append("core")

        def _gen():
            yield "chunk"

        return _gen()

    monkeypatch.setattr(provider, "_generate_core", fake_core)
    monkeypatch.setattr(
        "abstractcore.media.delivery.attach_media_report", lambda out, report: out
    )

    with pytest.raises(SpeculationUnavailableError) as excinfo:
        provider._generate_internal(
            "hi",
            stream=True,
            speculation={"mode": "native_mtp", "require_acceleration": True},
        )
    assert excinfo.value.reason == "speculation_is_load_time"
    assert built == [], (
        "the stream was constructed before the strict refusal fired; the "
        "refusal can now be swallowed by _stream_generate's except Exception"
    )


def test_a_per_call_refusal_does_not_poison_later_calls():
    """Per-call state must not be written into the LOAD-time field.

    `_apply_per_call_speculation` records a per-call drafter mismatch by
    assigning `self._mtp_outcome_at_load`, which is provider-lifetime state that
    `_mtp_outcome` falls back to on every later unaccelerated call. So one call
    asking for a drafter the provider did not load leaves a stale reason -- and
    a stale `drafter` -- attached to calls made afterwards.

    Concretely: the provider loaded 'loaded/drafter'. One call asks for
    'other/drafter' and is correctly refused. A LATER call then reports
    `drafter: 'other/drafter'` -- naming a checkpoint this provider has never
    loaded, in the very metadata block whose job is to say what actually ran.
    """
    provider = _provider(
        _speculation_request=SpeculationRequest(mode="native_mtp"),
        _mtp_drafter=object(),
        _mtp_kind="mtp",
        _mtp_block_size=3,
        _mtp_drafter_id="loaded/drafter",
    )

    # One call asks for a different drafter and is refused (correctly).
    provider._apply_per_call_speculation(
        {"mode": "native_mtp", "drafter": "other/drafter"}
    )

    # A LATER, unrelated call that simply does not use the drafter.
    provider._apply_per_call_speculation({"mode": "off"})
    metadata = provider._mtp_outcome(bool(provider._mtp_kwargs(None))).to_metadata()

    assert metadata.get("drafter") != "other/drafter", (
        "a later call reports drafter='other/drafter', which this provider "
        f"never loaded (it holds 'loaded/drafter'). Full metadata: {metadata}. "
        "Record per-call refusals in a per-call field, not in "
        "`_mtp_outcome_at_load`."
    )


# ---------------------------------------------------------------------------
# The refusal must survive the error plumbing with its type and reason intact.
# ---------------------------------------------------------------------------


def test_error_handler_passes_the_refusal_through_unwrapped():
    """`_handle_api_error` must not re-wrap it into ProviderAPIError.

    Wrapping destroys both the type strict callers catch and the `.reason` that
    is the whole point of strict mode.
    """
    provider = _provider()
    error = SpeculationUnavailableError("nope", reason="speculation_is_load_time")
    handled = MLXProvider._handle_api_error(provider, error)

    assert handled is error, (
        f"the refusal was re-wrapped as {type(handled).__name__}, losing .reason"
    )
    assert getattr(handled, "reason", None) == "speculation_is_load_time"


def test_the_passthrough_is_not_broadened_to_sibling_runtime_errors():
    """Guard the OTHER direction: the passthrough must stay narrow.

    `SpeculationUnavailableError` subclasses `RuntimeError`. If the check were
    ever loosened to `isinstance(error, RuntimeError)`, every unrelated runtime
    fault would skip the central timeout/status normalization AND become
    non-retryable, silently degrading the retry behaviour of the whole provider.
    """
    provider = _provider()
    plain = RuntimeError("some unrelated runtime fault")
    handled = MLXProvider._handle_api_error(provider, plain)

    assert not isinstance(handled, SpeculationUnavailableError)
    assert handled is not plain, (
        "a plain RuntimeError is passing through the speculation escape hatch; "
        "the isinstance check has been broadened past the one concrete type"
    )


def test_the_refusal_is_classified_non_retryable_by_name():
    """`core/retry.py` matches on `type(error).__name__`, so a typo fails silently.

    Retrying is actively harmful here: the lane cannot start speculating on
    attempt 2, so three attempts just multiply the latency the caller was trying
    to avoid.
    """
    from abstractcore.core.retry import RetryableErrorType, RetryConfig, RetryManager

    manager = RetryManager(RetryConfig())
    error = SpeculationUnavailableError("nope", reason="speculation_is_load_time")

    assert type(error).__name__ in manager.non_retryable_errors, (
        f"'{type(error).__name__}' is not in non_retryable_errors "
        f"({sorted(manager.non_retryable_errors)}) -- check for a typo, the "
        "match is on the class NAME and a mismatch fails silently"
    )
    assert manager.classify_error(error) is RetryableErrorType.UNKNOWN


def test_the_refusal_is_actually_attempted_once(monkeypatch):
    """End-to-end: the classification above must translate into one attempt."""
    from abstractcore.core.retry import RetryConfig, RetryManager

    manager = RetryManager(RetryConfig())
    attempts = {"n": 0}

    def boom():
        attempts["n"] += 1
        raise SpeculationUnavailableError("nope", reason="speculation_is_load_time")

    with pytest.raises(SpeculationUnavailableError) as excinfo:
        manager.execute_with_retry(boom, provider_key="mlx:test")

    assert attempts["n"] == 1, (
        f"the refusal was retried {attempts['n']} times for a condition that "
        "cannot change between attempts"
    )
    assert excinfo.value.reason == "speculation_is_load_time", (
        "the retry manager destroyed the .reason attribute on the way out"
    )


def test_explicit_drafter_overrides_the_registry():
    provider = _provider(
        _speculation_request=SpeculationRequest(
            mode="native_mtp", drafter="me/my-own-drafter", num_draft_tokens=5
        ),
        model_capabilities={
            "speculation": {
                "native_mtp": True,
                "runtimes": {
                    "mlx": {"mode": "drafter_repo", "drafter": "registry/drafter"}
                },
            }
        },
    )
    plan = provider._plan_mtp_lane("some/target")
    assert plan == {"drafter": "me/my-own-drafter", "block_size": 5}
