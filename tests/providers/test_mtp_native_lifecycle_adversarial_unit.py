"""CPU-only tests for native MLX request ownership and cache consistency."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.exceptions import ProviderAPIError
from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.providers.base import PromptCacheError
from abstractcore.providers.speculation import SpeculationRequest, SpeculationUnavailableError


def _provider():
    p = MLXProvider.__new__(MLXProvider)
    p.model = "publisher/Qwen3.8-Flash-Next-Q4-mtp"
    p.logger = Mock()
    p._mtp_processor = object()
    p._mtp_drafter = SimpleNamespace()
    p._mtp_drafter_id = p.model
    p._mtp_kind = "mtp"
    p._mtp_block_size = 3
    p._mtp_outcome_at_load = None
    p._mtp_prompt_cache_warned = False
    p._mtp_generation_lock = threading.Lock()
    p._speculation_request = SpeculationRequest(mode="native_mtp", num_draft_tokens=3)
    return p


def test_native_route_uses_bound_model_without_mutating_caller_kwargs(monkeypatch):
    from abstractcore.providers.base import BaseProvider

    p = _provider()
    p._native_qwen4 = object()
    p.provider = "mlx"
    caller_kwargs = {"top_p": 0.8}
    captured = []

    def route(self, *, request, output, thinking, kwargs):
        captured.append(kwargs)
        return "route-result"

    monkeypatch.setattr(BaseProvider, "_resolve_generate_route", route)
    result = p._resolve_generate_route(request=object(), output=None, thinking=False, kwargs=caller_kwargs)
    assert result == "route-result"
    assert captured == [{"top_p": 0.8, "_provider": "mlx", "_model": p.model}]
    assert caller_kwargs == {"top_p": 0.8}


def test_native_route_preserves_explicit_request_route(monkeypatch):
    from abstractcore.providers.base import BaseProvider

    p = _provider()
    p._native_qwen4 = object()
    p.provider = "mlx"
    explicit = {"_provider": "mlx", "_model": "explicit-request-model"}
    monkeypatch.setattr(BaseProvider, "_resolve_generate_route", lambda self, **kw: kw["kwargs"])
    assert p._resolve_generate_route(request=object(), output=None, thinking=None, kwargs=explicit) == explicit


def test_native_text_output_route_does_not_inherit_unrelated_global_model(monkeypatch):
    from abstractcore.core import generate_contract

    p = _provider()
    p._native_qwen4 = object()
    p.provider = "mlx"
    monkeypatch.setattr(
        generate_contract, "resolve_capability_default_route",
        lambda *args, **kwargs: {"provider": "mlx", "model": "unrelated/global-default", "source": "test_defaults"},
    )
    output = {"modality": "text", "task": "text_generation"}
    route = p._resolve_generate_route(
        request=generate_contract.GenerateRequest(text="Hello"),
        output=output, thinking=False, kwargs={},
    )
    assert route.text_route.model == p.model
    assert route.output_routes[0].model == p.model
    assert route.output_specs[0]["model"] == p.model
    assert output == {"modality": "text", "task": "text_generation"}


def _install_stream(p):
    events = []

    def core(prompt, **kwargs):
        def stream():
            try:
                for _ in range(2):
                    width = p._mtp_kwargs(None).get("draft_block_size")
                    events.append((prompt, width))
                    yield GenerateResponse(content=str(width), model=p.model)
            finally:
                events.append((prompt, "closed"))

        return stream()

    p._generate_core = core
    return events


def test_two_unstarted_streams_do_not_change_each_others_depth():
    p = _provider()
    events = _install_stream(p)
    first = p._generate_internal("first", stream=True, speculation={"num_draft_tokens": 5})
    second = p._generate_internal("second", stream=True, speculation=False)
    assert events == [], "constructing an iterator prematurely entered the model"
    assert next(first).content == "6"
    first.close()
    assert next(second).content == "None"
    second.close()
    default = p._generate_internal("default", stream=True)
    assert next(default).content == "4"
    default.close()
    assert not p._mtp_generation_lock.locked()


def test_busy_second_call_is_rejected_without_mutating_live_stream():
    p = _provider()
    events = _install_stream(p)
    first = p._generate_internal("first", stream=True, speculation={"num_draft_tokens": 5})
    assert next(first).content == "6"
    try:
        with pytest.raises(ProviderAPIError, match="already active"):
            p._generate_internal("intruder", speculation=False)
        second = p._generate_internal("second", stream=True, speculation=False)
        with pytest.raises(ProviderAPIError, match="already active"):
            next(second)
        assert next(first).content == "6"
        assert all(prompt not in {"intruder", "second"} for prompt, _ in events)
    finally:
        first.close()
    assert ("first", "closed") in events
    assert not p._mtp_generation_lock.locked()


def test_closing_unstarted_stream_never_claims_generation_lock():
    p = _provider()
    events = _install_stream(p)
    stream = p._generate_internal("never consumed", stream=True)
    stream.close()
    assert events == []
    assert not p._mtp_generation_lock.locked()


def test_failed_stream_releases_exclusive_ownership():
    p = _provider()

    def broken(prompt, **kwargs):
        yield GenerateResponse(content="prefix", model=p.model)
        raise RuntimeError("native verifier failed")

    p._generate_core = broken
    stream = p._generate_internal("one", stream=True)
    assert next(stream).content == "prefix"
    with pytest.raises(RuntimeError, match="verifier failed"):
        next(stream)
    assert not p._mtp_generation_lock.locked()
    events = _install_stream(p)
    following = p._generate_internal("following", stream=True, speculation={"num_draft_tokens": 1})
    assert next(following).content == "2"
    following.close()
    assert ("following", "closed") in events


def test_stream_cleanup_error_still_releases_generation_lock():
    p = _provider()

    def close_fails(prompt, **kwargs):
        try:
            yield GenerateResponse(content="prefix", model=p.model)
        finally:
            raise RuntimeError("decoder cleanup failed")

    p._generate_core = close_fails
    stream = p._generate_internal("one", stream=True)
    next(stream)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        stream.close()
    assert not p._mtp_generation_lock.locked(), "cleanup exception permanently locked native session"


def test_two_providers_sharing_one_session_cannot_reset_same_drafter_concurrently():
    first_provider = _provider()
    second_provider = _provider()
    second_provider._mtp_generation_lock = first_provider._mtp_generation_lock
    second_provider._mtp_drafter = first_provider._mtp_drafter
    _install_stream(first_provider)
    _install_stream(second_provider)
    stream = first_provider._generate_internal("first", stream=True)
    next(stream)
    try:
        with pytest.raises(ProviderAPIError, match="already active"):
            second_provider._generate_internal("second", speculation=False)
    finally:
        stream.close()
    assert not second_provider._mtp_generation_lock.locked()


def test_sidecar_native_cache_never_uses_legacy_delta_feed_or_discards_history():
    p = _provider()
    manager = object()
    p._native_session = SimpleNamespace(prompt_cache=Mock(return_value=manager))
    p.llm = object()
    p.tokenizer = object()
    p.temperature = 0.0
    p.max_output_tokens = 8
    p.tool_handler = SimpleNamespace(supports_prompted=False)
    p._build_prompt = Mock(return_value="FULL HISTORY AND CURRENT QUESTION")
    p._prepare_generation_kwargs = lambda **kwargs: kwargs
    p._prompt_cache_store = Mock()
    warm_cache = [object()]
    p._prompt_cache_store.get.return_value = warm_cache
    p._prepare_cache_delta_feed = Mock(return_value=(warm_cache, [99], [1, 2, 99]))
    p.prompt_cache_key_meta = Mock(return_value={})
    p._record_fed_token_ids = Mock()
    feeds = []

    def single(prompt, *args, **kwargs):
        # Native APC must receive complete history, never the legacy cache's
        # delta. The ordinary cache object remains deliberately incompatible.
        call_kwargs = p._mtp_call_kwargs({"prompt_cache": args[-1]})
        feeds.append((prompt, call_kwargs))
        return GenerateResponse(content="ok", model=p.model, finish_reason="stop")

    p._single_generate = single
    response = p._generate_internal("question", messages=[], prompt_cache_key="session")
    assert response.finish_reason != "error", response.content
    assert feeds[0][0] == "FULL HISTORY AND CURRENT QUESTION"
    assert feeds[0][1].get("prompt_cache") is None
    assert feeds[0][1]["apc_manager"] is manager
    assert json.loads(feeds[0][1]["apc_tenant"])[-1] == "session"
    p._prepare_cache_delta_feed.assert_not_called()
    p._record_fed_token_ids.assert_not_called()


def test_native_apc_receives_full_history_and_cache_tenant_does_not_leak():
    p = _provider()
    manager = object()
    p._native_qwen4 = SimpleNamespace(prompt_cache=Mock(return_value=manager))
    p.llm = object()
    p.tokenizer = object()
    p.temperature = 0.0
    p.max_output_tokens = 8
    p.tool_handler = SimpleNamespace(supports_prompted=False)
    p._build_prompt = Mock(return_value="FULL HISTORY AND CURRENT QUESTION")
    p._prepare_generation_kwargs = lambda **kwargs: kwargs
    p._prepare_cache_delta_feed = Mock(side_effect=AssertionError("base cache must not trim native APC inputs"))
    p._record_fed_token_ids = Mock(side_effect=AssertionError("base cache must not record native APC inputs"))
    feeds = []

    def single(prompt, *args, **kwargs):
        forwarded = p._mtp_call_kwargs({"prompt_cache": args[-1]})
        feeds.append((prompt, forwarded))
        p._mtp_last_result = SimpleNamespace(cached_tokens=128)
        return GenerateResponse(content="ok", model=p.model, finish_reason="stop")

    p._single_generate = single
    one = p._generate_internal("question", messages=[], prompt_cache_key="tenant-one")
    two = p._generate_internal("question", messages=[], prompt_cache_key="tenant-two")
    three = p._generate_internal("question", messages=[])
    assert all(response.finish_reason != "error" for response in (one, two, three))
    assert [prompt for prompt, _ in feeds] == ["FULL HISTORY AND CURRENT QUESTION"] * 3
    assert feeds[0][1]["apc_manager"] is manager
    assert json.loads(feeds[0][1]["apc_tenant"]) == ["local", "tenant-one"]
    assert json.loads(feeds[1][1]["apc_tenant"]) == ["local", "tenant-two"]
    assert "apc_tenant" not in feeds[2][1]
    assert "apc_manager" not in feeds[2][1]
    assert one.metadata["prompt_cache"]["cached_tokens"] == 128
    p._prepare_cache_delta_feed.assert_not_called()
    p._record_fed_token_ids.assert_not_called()


@pytest.mark.parametrize("parameter,value", [
    ("presence_penalty", 1.5),
    ("repetition_penalty", 1.1),
    ("frequency_penalty", 0.4),
    ("min_p", 0.08),
])
def test_native_generation_does_not_silently_discard_supported_sampling_controls(parameter, value):
    p = _provider()
    p._native_qwen4 = SimpleNamespace()
    p.llm = object()
    p.tokenizer = SimpleNamespace(encode=lambda text: list(text))
    p.temperature = 0.0
    p.max_output_tokens = 8
    p.tool_handler = SimpleNamespace(supports_prompted=False)
    p._build_prompt = Mock(return_value="FULL PROMPT")
    p._prepare_generation_kwargs = lambda **kwargs: kwargs
    p._postprocess_generated_text = lambda text, **_: (text, None)
    p._calculate_usage = lambda prompt, text: {"input_tokens": len(prompt), "output_tokens": len(text)}
    p.generate_fn = Mock(return_value="ok")
    response = p._generate_internal("question", **{parameter: value})
    assert response.finish_reason != "error", response.content
    assert p.generate_fn.call_count == 1
    assert p.generate_fn.call_args.kwargs.get(parameter) == value
    if parameter != "min_p":
        assert p._mtp_call_disabled is True, "unsupported logits processors silently used MTP"
        assert response.metadata["speculation"]["used"] is False
        assert response.metadata["speculation"].get("reason")
        assert p.logger.warning.called


@pytest.mark.parametrize("parameter,value", [
    ("presence_penalty", 1.5), ("repetition_penalty", 1.1),
    ("frequency_penalty", 0.4), ("logit_bias", {42: 2.0}),
])
def test_strict_mtp_refuses_logit_processors_upstream_cannot_apply_after_first_token(parameter, value):
    p = _provider()
    p._native_qwen4 = SimpleNamespace()
    p._generate_core = Mock(return_value=GenerateResponse(content="must not run", model=p.model))
    with pytest.raises(SpeculationUnavailableError):
        p._generate_internal("question", speculation={"require_acceleration": True}, **{parameter: value})
    p._generate_core.assert_not_called()
    assert not p._mtp_generation_lock.locked()


@pytest.mark.parametrize("parameter,value", [
    ("presence_penalty", 0.0), ("repetition_penalty", 1.0),
    ("frequency_penalty", 0.0), ("logit_bias", {}), ("min_p", 0.08),
])
def test_neutral_processor_controls_and_min_p_do_not_disable_native_mtp(parameter, value):
    p = _provider()
    p._native_qwen4 = SimpleNamespace()
    p._generate_core = Mock(return_value=GenerateResponse(content="ok", model=p.model))
    response = p._generate_internal("question", speculation={"require_acceleration": True}, **{parameter: value})
    assert response.content == "ok"
    assert p._mtp_call_disabled is False


def test_native_keyed_fragment_mode_fails_before_forgetting_context():
    p = _provider()
    p._native_qwen4 = SimpleNamespace()
    p._generate_core = Mock(return_value=GenerateResponse(content="must not run", model=p.model))
    with pytest.raises((ValueError, ProviderAPIError), match="messages|history"):
        p._generate_internal("only a new fragment", prompt_cache_key="existing-conversation")
    p._generate_core.assert_not_called()
    assert not p._mtp_generation_lock.locked()


def test_native_cache_capabilities_do_not_claim_legacy_control_plane_support():
    p = _provider()
    p._native_qwen4 = SimpleNamespace(apc=None)
    caps = p.get_prompt_cache_capabilities()
    assert caps.supported is True and caps.mode == "keyed"
    assert caps.supports_clear is True and caps.supports_stats is True
    for operation in ("set", "update", "fork", "prepare_modules", "save", "load"):
        assert caps.supports_operation(operation) is False
    assert p.prompt_cache_supports_kv_source_of_truth() is False
    assert p.get_prompt_cache_stats()["full_history_required"] is True
    assert p.get_prompt_cache_stats()["stats"] == {}
    with pytest.raises(PromptCacheError):
        p.prompt_cache_set("tenant")
    with pytest.raises(ProviderAPIError, match="manual prompt_cache_save"):
        p.prompt_cache_save("tenant", "never-written.safetensors")
    with pytest.raises(ProviderAPIError, match="manual prompt_cache_load"):
        p.prompt_cache_load("never-read.safetensors")


def test_native_cache_clear_reports_shared_scope_and_cannot_race_generation():
    p = _provider()
    apc = Mock()
    apc.stats_snapshot.return_value = {"hits": 4}
    p._native_qwen4 = SimpleNamespace(lock=p._mtp_generation_lock, apc=apc)
    assert p.get_prompt_cache_stats()["stats"] == {"hits": 4}
    p._mtp_generation_lock.acquire()
    try:
        with pytest.raises(ProviderAPIError, match="during generation"):
            p.prompt_cache_clear("tenant-one")
        apc.clear.assert_not_called()
    finally:
        p._mtp_generation_lock.release()
    assert p.prompt_cache_clear("tenant-one") is True
    apc.clear.assert_called_once()
    assert "all shared" in p.logger.warning.call_args.args[0]
    assert not p._mtp_generation_lock.locked()


def test_native_unload_is_refused_while_stream_owns_model_without_dropping_refs():
    p = _provider()
    session = SimpleNamespace(lock=p._mtp_generation_lock)
    p._native_qwen4 = session
    p.llm = object()
    model = p.llm
    draft = p._mtp_drafter
    p._unload_model_unlocked = Mock()
    p._mtp_generation_lock.acquire()
    try:
        with pytest.raises(ProviderAPIError, match="during generation"):
            p.unload_model(p.model)
        assert p._native_qwen4 is session
        assert p.llm is model
        assert p._mtp_drafter is draft
        p._unload_model_unlocked.assert_not_called()
    finally:
        p._mtp_generation_lock.release()


def test_native_unload_drops_only_callers_session_reference_and_releases_lock():
    p = _provider()
    apc = Mock()
    session = SimpleNamespace(lock=p._mtp_generation_lock, apc=apc, model=object())
    p._native_qwen4 = session
    other = _provider()
    other._native_qwen4 = session
    p._unload_model_unlocked = Mock()
    p.unload_model(p.model)
    assert p._native_qwen4 is None
    assert other._native_qwen4 is session
    assert other._native_qwen4.model is session.model
    apc.clear.assert_not_called()
    p._unload_model_unlocked.assert_called_once_with(p.model)
    assert not p._mtp_generation_lock.locked()
