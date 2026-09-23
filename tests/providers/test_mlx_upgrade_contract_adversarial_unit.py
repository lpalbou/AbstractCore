"""Independent CPU-only checks for the native-provider request boundary.

These tests deliberately avoid model/runtime initialization. They exercise the
same facade and outcome code used by concurrent public requests without letting
shared weights stand in for shared mutable request state.
"""

import asyncio
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.providers.speculation import SpeculationRequest
from abstractcore.core.types import GenerateResponse
from abstractcore.core.retry import RetryConfig, RetryManager
from abstractcore.providers.mlx_runtime import NativeRuntimeError
from abstractcore.providers.base import BaseProvider
from abstractcore.tools.handler import UniversalToolHandler


try:  # Python 3.9 has no builtin anext()
    anext
except NameError:  # pragma: no cover - exercised on Python 3.9 only
    async def anext(iterator):  # noqa: A001
        return await iterator.__anext__()


def _provider():
    provider = MLXProvider.__new__(MLXProvider)
    provider.model = "mlx-community/Qwen3.8-27B-4bit"
    provider.provider = "mlx"
    provider.logger = Mock()
    provider.llm = object()
    provider.tokenizer = object()
    provider._mtp_processor = object()
    provider._mtp_drafter = object()
    provider._mtp_drafter_id = "mlx-community/Qwen3.8-27B-MTP-4bit"
    provider._mtp_kind = "mtp"
    provider._mtp_block_size = 2
    provider._mtp_outcome_at_load = None
    provider._mtp_output_preserving = None
    provider._speculation_request = SpeculationRequest(
        mode="native_mtp", num_draft_tokens=2,
    )
    provider._native_runtime = object()
    provider._native_owner_id = "existing-owner"
    provider._mlx_cache_scope = "private-example-scope"
    provider._vision_side = {"parent": "retained"}
    provider._last_output_budget_clamp = {"parent": "retained"}
    provider._native_runtime_metadata = {"parent": "retained"}
    provider.generate_fn = provider._mtp_generate_fn
    provider.stream_generate_fn = provider._mtp_stream_generate_fn
    provider._apply_per_call_speculation(None)
    return provider


def test_request_facade_rebinds_both_bound_adapters_and_keeps_weight_identity():
    provider = _provider()
    view = provider._native_request_view()
    assert view is not provider
    assert view.generate_fn.__self__ is view
    assert view.stream_generate_fn.__self__ is view
    assert provider.generate_fn.__self__ is provider
    assert provider.stream_generate_fn.__self__ is provider
    assert view.llm is provider.llm
    assert view._mtp_drafter is provider._mtp_drafter
    assert view._native_runtime is provider._native_runtime
    assert view._native_owner_id == provider._native_owner_id


def test_request_facades_do_not_share_cancellation_or_mutable_telemetry():
    provider = _provider()
    first = provider._native_request_view()
    second = provider._native_request_view()
    first._native_cancel_event.set()
    first._native_runtime_metadata["first"] = True
    first._vision_side["first"] = True
    assert not second._native_cancel_event.is_set()
    assert second._native_runtime_metadata == {}
    assert second._vision_side == {}
    assert provider._native_runtime_metadata == {"parent": "retained"}
    assert provider._vision_side == {"parent": "retained"}
    assert first._last_output_budget_clamp is None
    assert provider._last_output_budget_clamp == {"parent": "retained"}


def test_lazy_request_facade_keeps_original_model_lease_alive():
    provider = _provider()
    owner = weakref.ref(provider)
    view = provider._native_request_view()
    del provider
    gc.collect()
    assert owner() is not None, "temporary provider finalized before its lazy stream could start"
    del view
    gc.collect()
    assert owner() is None


def test_interleaved_facade_depth_and_disable_never_modify_sibling_or_default():
    provider = _provider()
    first = provider._native_request_view()
    second = provider._native_request_view()
    first._apply_per_call_speculation({"num_draft_tokens": 5})
    second._apply_per_call_speculation(False)
    assert first._mtp_call_block_size == 5
    assert first._mtp_call_disabled is False
    assert second._mtp_call_disabled is True
    assert provider._mtp_call_block_size == 2
    assert provider._mtp_call_disabled is False
    assert provider._mtp_block_size == 2
    assert provider._speculation_request.num_draft_tokens == 2


@pytest.mark.parametrize("native_qwen4", [None, object()])
@pytest.mark.parametrize("depth", [1, 2, 5])
def test_public_depth_is_proposals_for_both_native_architectures(native_qwen4, depth):
    provider = _provider()
    provider._native_qwen4 = native_qwen4
    provider._apply_per_call_speculation({"num_draft_tokens": depth})
    assert provider._mtp_kwargs(None)["draft_block_size"] == depth + 1
    assert provider.speculation_status()["effective_draft_tokens"] == depth


def test_observed_backend_execution_cannot_overwrite_another_request_outcome():
    provider = _provider()
    first = provider._native_request_view()
    second = provider._native_request_view()
    first._apply_per_call_speculation({"num_draft_tokens": 5})
    second._apply_per_call_speculation(False)
    result = SimpleNamespace(
        metadata={"speculation": {"used": True, "stats": {"rounds": 7}}},
        media_records=[],
    )
    first._observe_native_runtime_result(result)
    assert first._mtp_last_used is True
    assert first._mtp_call_stats == {"rounds": 7}
    assert second._mtp_last_used is False
    assert second._mtp_call_stats == {}
    assert provider._mtp_last_used is False
    assert provider._mtp_call_stats == {}
    result.metadata["speculation"]["stats"]["rounds"] = 999
    assert first._mtp_call_stats == {"rounds": 7}


def test_reported_concurrency_requires_an_actual_runtime_owner():
    provider = _provider()
    provider._mlx_batching = True
    provider._native_runtime = None
    assert not provider.supports_concurrent_generation()
    provider._native_runtime = object()
    assert provider.supports_concurrent_generation()


@pytest.mark.parametrize("stream", [False, True])
def test_async_native_cancel_closes_waiting_handle_without_blocking_loop(monkeypatch, stream):
    provider = _provider()
    entered = threading.Event()
    closed = threading.Event()

    class Handle:
        def __iter__(self):
            return self

        def __next__(self):
            entered.set()
            assert closed.wait(2), "async cancellation did not reach native stream handle"
            raise StopIteration

        def close(self):
            closed.set()

    def open_handle(request, *, cancel_event=None):
        assert isinstance(cancel_event, threading.Event)
        return Handle()

    provider._native_runtime = SimpleNamespace(stream=open_handle)
    monkeypatch.setattr(MLXProvider, "_native_runtime_request", lambda self, text, kwargs: object())

    def generate(view, prompt, *, stream=False, **kwargs):
        if stream:
            return view._mtp_stream_generate_fn(view.llm, view.tokenizer, prompt)
        return view._mtp_generate_fn(view.llm, view.tokenizer, prompt)

    monkeypatch.setattr(MLXProvider, "generate", generate)

    async def exercise():
        if stream:
            source = await provider._agenerate_internal("p", None, None, None, None, True)
            task = asyncio.create_task(anext(source))
        else:
            source = None
            task = asyncio.create_task(provider._agenerate_internal("p", None, None, None, None, False))
        assert await asyncio.to_thread(entered.wait, 1)
        # This coroutine must run while the synchronous iterator remains blocked.
        await asyncio.wait_for(asyncio.sleep(0), timeout=.5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        assert closed.is_set()
        if source is not None:
            await source.aclose()

    asyncio.run(exercise())


def test_async_public_annotation_wrapper_closes_nested_stream():
    provider = _provider()
    provider._annotate_output_truncation = lambda chunk: None
    closed = []

    async def source():
        try:
            yield SimpleNamespace(content="one")
            yield SimpleNamespace(content="two")
        finally:
            closed.append(True)

    async def exercise():
        wrapper = provider._annotate_async_stream(source())
        assert (await anext(wrapper)).content == "one"
        await wrapper.aclose()

    asyncio.run(exercise())
    assert closed == [True]


def test_native_streaming_tools_emit_complete_results_then_one_accounted_terminal():
    provider = _provider()
    provider._native_execute_tools = True
    markup = 'Before <tool_call>{"name":"add_numbers","arguments":{"a":17,"b":23}}</tool_call>'
    cleaned = "Before "
    usage = {"input_tokens": 25, "output_tokens": 18, "total_tokens": 43}
    metadata = {"execution": {"mode": "cohort", "peak_batch_size": 2}}
    closed = []

    def source(*args, **kwargs):
        try:
            yield GenerateResponse(content=markup, model=provider.model, finish_reason="stop",
                                   usage=usage, metadata=metadata)
        finally:
            closed.append(True)

    provider._stream_generate = source
    provider.tool_handler = SimpleNamespace(
        supports_prompted=True,
        parse_response=lambda *args, **kwargs: SimpleNamespace(content=cleaned),
    )
    provider._handle_prompted_tool_execution = Mock(return_value=GenerateResponse(
        content=cleaned + "\nTool result: 40", model=provider.model, finish_reason="stop",
    ))
    chunks = list(provider._stream_generate_with_tools("prompt", 30, 0, 1, tools=[{"name": "add_numbers"}]))
    assert [chunk.content for chunk in chunks] == [markup, "\nTool result: 40", ""]
    assert [chunk.finish_reason for chunk in chunks] == [None, None, "stop"]
    assert chunks[-1].usage == usage
    assert chunks[-1].metadata == metadata
    assert provider._handle_prompted_tool_execution.call_args.kwargs == {"execute_tools_param": True}
    assert closed == [True]


def test_native_streaming_explicit_tool_disable_overrides_constructor_default():
    provider = _provider()
    provider.execute_tools = True
    provider._native_execute_tools = False
    provider.tool_handler = SimpleNamespace(supports_prompted=True)
    response = GenerateResponse(content="tool call markup", model=provider.model, finish_reason="stop")
    provider._stream_generate = lambda *args, **kwargs: iter([response])
    provider._handle_prompted_tool_execution = Mock(side_effect=lambda response, tools, **kwargs: response)
    chunks = list(provider._stream_generate_with_tools("prompt", 30, 0, 1, tools=[{"name": "add_numbers"}]))
    assert provider._handle_prompted_tool_execution.call_args.kwargs == {"execute_tools_param": False}
    assert sum(chunk.finish_reason is not None for chunk in chunks) == 1
    assert "".join(chunk.content or "" for chunk in chunks) == "tool call markup"


def test_native_per_call_tool_execution_choice_does_not_leak_to_following_call():
    provider = _provider()
    observed = []

    def core(prompt, **kwargs):
        observed.append(provider._native_execute_tools)
        return GenerateResponse(content="answer", model=provider.model, finish_reason="stop")

    provider._generate_core = core
    provider._generate_internal_unlocked("prompt", execute_tools=False)
    provider._generate_internal_unlocked("prompt", execute_tools=True)
    provider._generate_internal_unlocked("prompt")
    assert observed == [False, True, None]


@pytest.mark.parametrize("code,message", [
    ("queue_full", "Native MLX waiting queue is full; retry after active requests finish"),
    ("queue_timeout", "Native MLX request timed out waiting for admission"),
    ("cancelled", "Native MLX request was cancelled before admission"),
    ("stream_backpressure", "Native MLX stream consumer is too slow; bounded output queue overflowed"),
])
def test_native_request_local_errors_keep_details_and_never_poison_shared_retry_breaker(code, message):
    provider = _provider()
    manager = RetryManager(RetryConfig(initial_delay=0, max_delay=0, use_jitter=False))
    calls = []
    error = NativeRuntimeError(message, code=code)
    mapped = provider._handle_api_error(error)
    assert mapped is error
    assert str(mapped) == message, "queue timeout was rewritten as an unrelated provider timeout"

    def fail():
        calls.append(True)
        raise mapped

    # Enough failures to open the ordinary transient breaker if misclassified.
    for _ in range(10):
        with pytest.raises(NativeRuntimeError) as raised:
            manager.execute_with_retry(fail, provider_key="shared-native-model")
        assert raised.value is error
    assert len(calls) == 10, "request-local admission failure was retried behind the caller's back"
    breaker = manager.get_circuit_breaker("shared-native-model")
    assert breaker.failure_count == 0
    assert manager.execute_with_retry(lambda: "healthy", provider_key="shared-native-model") == "healthy"


def test_native_backend_faults_still_receive_existing_transient_retry_and_breaker_accounting():
    provider = _provider()
    manager = RetryManager(RetryConfig(initial_delay=0, max_delay=0, use_jitter=False))
    calls = []
    error = NativeRuntimeError("Native batch engine stopped without terminal responses")
    mapped = provider._handle_api_error(error)

    def fail():
        calls.append(True)
        raise mapped

    with pytest.raises(Exception):
        manager.execute_with_retry(fail, provider_key="shared-native-model")
    assert len(calls) == 2, "backend faults must not be disguised as request-local refusals"
    assert manager.get_circuit_breaker("shared-native-model").failure_count == 2


def test_public_concurrent_calls_keep_trace_tools_thinking_and_facade_state_request_local(monkeypatch):
    provider = _provider()
    BaseProvider.__init__(provider, provider.model, enable_tracing=True, timeout=None, tool_timeout=None)
    provider.provider = "mlx"
    provider.structured_output_method = "prompted"
    provider.tool_handler = UniversalToolHandler(provider.model)
    handler_before = dict(vars(provider.tool_handler))
    gate = threading.Barrier(2, timeout=2)
    observations = []
    events = []
    monkeypatch.setattr("abstractcore.events.emit_global", lambda kind, data, **kwargs: events.append((kind, data)))

    def core(view, prompt, **kwargs):
        gate.wait()
        observations.append({
            "view": view, "prompt": prompt, "tools": kwargs["tools"],
            "disabled": view._mtp_call_disabled, "depth": view._mtp_call_block_size,
            "execute": view._native_execute_tools,
            "thinking": kwargs.get("_acore_mlx_enable_thinking"),
        })
        return GenerateResponse(content=prompt + " result", model=view.model, finish_reason="stop",
                                usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10})

    monkeypatch.setattr(MLXProvider, "_generate_core", core)

    def call(index):
        return provider.generate(
            "request-" + str(index), thinking=bool(index), execute_tools=False,
            speculation={"num_draft_tokens": 4} if index else False,
            tools=[{"name": "tool_" + str(index), "description": "unused", "parameters": {"type": "object"}}],
            trace_metadata={"request_marker": index}, max_output_tokens=32,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(call, [0, 1]))
    observed = {row["prompt"]: row for row in observations}
    assert observed["request-0"]["disabled"] is True
    assert observed["request-1"]["disabled"] is False
    assert observed["request-1"]["depth"] == 4
    assert observed["request-0"]["thinking"] is False
    assert observed["request-1"]["thinking"] is True
    assert all(row["view"] is not provider and row["execute"] is False for row in observations)
    for index, response in enumerate(responses):
        assert response.content == f"request-{index} result"
        trace = provider.get_traces(trace_id=response.metadata["trace_id"])
        assert trace["prompt"] == f"request-{index}"
        assert trace["metadata"] == {"request_marker": index}
        assert trace["tools"][0]["name"] == f"tool_{index}"
    assert len({response.metadata["trace_id"] for response in responses}) == 2
    assert provider._mtp_call_block_size == 2 and provider._mtp_call_disabled is False
    assert dict(vars(provider.tool_handler)) == handler_before
    completed = [data["prompt"] for kind, data in events if kind.value == "generation_completed"]
    assert sorted(completed) == ["request-0", "request-1"]
