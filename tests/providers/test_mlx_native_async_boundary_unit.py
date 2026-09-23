"""The scheduled async API must normalize exactly once, inside sync generate."""
import asyncio
import threading
from types import SimpleNamespace

import pytest

from abstractcore.providers.mlx_provider import MLXProvider


try:  # Python 3.9 has no builtin anext()
    anext
except NameError:  # pragma: no cover - exercised on Python 3.9 only
    async def anext(iterator):  # noqa: A001
        return await iterator.__anext__()


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking", [False, True, "high"])
def test_native_async_preserves_raw_public_controls(monkeypatch, stream, thinking):
    provider = MLXProvider.__new__(MLXProvider)
    provider._native_runtime = object()
    view = SimpleNamespace(_native_cancel_event=threading.Event(), _native_runtime_stream=None)
    seen = []
    caller_thread = threading.get_ident()

    def generate(prompt, **kwargs):
        seen.append((prompt, kwargs, threading.get_ident()))
        result = SimpleNamespace(content="391", finish_reason="stop")
        return iter([result]) if kwargs["stream"] else result

    view.generate = generate
    monkeypatch.setattr(provider, "_native_request_view", lambda: view)
    # Any attempt to enter BaseProvider.agenerate before handing the original
    # request to sync generate is the double-normalization defect.
    def normalized_twice(*args, **kwargs):
        raise AssertionError("Async path normalized controls before sync generate")
    monkeypatch.setattr(provider, "_normalize_system_prompt_alias", normalized_twice)

    async def exercise():
        response = await provider.agenerate(
            "question", stream=stream, thinking=thinking, max_tokens=64,
            seed=0, speculation=False, messages=[], execute_tools=False)
        if stream:
            try:
                assert (await anext(response)).content == "391"
            finally:
                await response.aclose()
        else:
            assert response.content == "391"

    asyncio.run(exercise())
    assert len(seen) == 1
    prompt, controls, thread = seen[0]
    assert prompt == "question"
    assert controls["thinking"] == thinking
    assert controls["max_tokens"] == 64
    assert controls["seed"] == 0
    assert controls["speculation"] is False
    assert controls["execute_tools"] is False
    assert controls["messages"] == []
    assert thread != caller_thread


def test_ordinary_mlx_async_retains_base_path(monkeypatch):
    from abstractcore.providers.base import BaseProvider
    provider = MLXProvider.__new__(MLXProvider)
    provider._native_runtime = None
    seen = []

    async def base(self, *args, **kwargs):
        seen.append((args, kwargs))
        return "ordinary"

    monkeypatch.setattr(BaseProvider, "agenerate", base)
    assert asyncio.run(provider.agenerate("p", thinking=False)) == "ordinary"
    assert seen[0][1]["thinking"] is False


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("thinking", [False, True])
def test_native_async_real_base_normalizes_once_and_preserves_public_metadata(monkeypatch, stream, thinking):
    """Exercise real public normalization/telemetry, replacing only model work."""
    from abstractcore.core.types import GenerateResponse
    from abstractcore.providers.base import BaseProvider
    from abstractcore.providers.speculation import SpeculationRequest
    from abstractcore.tools.handler import UniversalToolHandler

    provider = MLXProvider.__new__(MLXProvider)
    model = "mlx-community/Qwen3.8-27B-4bit"
    BaseProvider.__init__(provider, model, enable_tracing=True, timeout=None, tool_timeout=None)
    provider.provider = "mlx"
    provider.structured_output_method = "prompted"
    provider.tool_handler = UniversalToolHandler(model)
    provider.llm = provider.tokenizer = object()
    provider._mtp_processor = provider._mtp_drafter = object()
    provider._mtp_drafter_id = "mlx-community/Qwen3.8-27B-MTP-4bit"
    provider._mtp_kind = "mtp"
    provider._mtp_block_size = 2
    provider._mtp_outcome_at_load = provider._mtp_output_preserving = None
    provider._speculation_request = SpeculationRequest(mode="native_mtp", num_draft_tokens=2)
    provider._native_runtime = object()
    provider._native_owner_id = "fake-native-owner"
    provider._mlx_cache_scope = "async-boundary-test"
    provider._vision_side = {}
    provider._apply_per_call_speculation(None)
    calls, events = [], []
    caller_thread = threading.get_ident()
    original_thinking = BaseProvider._apply_thinking_request
    original_route = MLXProvider._resolve_generate_route

    def apply_thinking(view, **kwargs):
        calls.append(("thinking", kwargs["thinking"], threading.get_ident()))
        return original_thinking(view, **kwargs)

    def resolve_route(view, **kwargs):
        calls.append(("route", kwargs["thinking"], threading.get_ident()))
        return original_route(view, **kwargs)

    def core(view, prompt, **kwargs):
        calls.append(("core", kwargs.get("_acore_mlx_enable_thinking"), threading.get_ident()))
        assert view is not provider and view._native_parent is provider
        assert view._mtp_call_disabled is True and view._native_execute_tools is False
        assert kwargs["stream"] is stream
        response = GenerateResponse(
            content=prompt + " result", model=model, finish_reason="length",
            usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        )
        return iter([response]) if stream else response

    monkeypatch.setattr(BaseProvider, "_apply_thinking_request", apply_thinking)
    monkeypatch.setattr(MLXProvider, "_resolve_generate_route", resolve_route)
    monkeypatch.setattr(MLXProvider, "_generate_core", core)
    monkeypatch.setattr("abstractcore.events.emit_global",
                        lambda kind, data, **kwargs: events.append((kind.value, data)))

    async def exercise():
        response = await provider.agenerate(
            "async question", stream=stream, thinking=thinking, max_tokens=3,
            speculation=False, execute_tools=False, seed=0,
            trace_metadata={"request_marker": "native-async"},
        )
        if not stream:
            trace = provider.get_traces(trace_id=response.metadata["trace_id"])
            assert trace["prompt"] == "async question"
            assert trace["metadata"] == {"request_marker": "native-async"}
            return [response]
        try:
            return [chunk async for chunk in response]
        finally:
            await response.aclose()

    chunks = asyncio.run(asyncio.wait_for(exercise(), timeout=3))
    assert "".join(chunk.content or "" for chunk in chunks) == "async question result"
    terminal = [chunk for chunk in chunks if chunk.finish_reason == "length"]
    assert len(terminal) == 1
    assert terminal[0].metadata["output_truncated"] is True
    assert terminal[0].metadata["truncation_kind"] == "output_cap"
    assert terminal[0].metadata["thinking_effective"] == ("on" if thinking else "off")
    route = terminal[0].metadata["_resolved_generate_route"]
    assert route["reasoning"] == ("on" if thinking else "off")
    assert route["text_route"]["provider"] == "mlx"
    assert route["text_route"]["model"] == model
    assert [kind for kind, _, _ in calls] == ["route", "thinking", "core"]
    assert all(value is thinking for _, value, _ in calls)
    assert len({thread for _, _, thread in calls}) == 1
    assert calls[0][2] != caller_thread
    assert sum(kind == "generation_completed" for kind, _ in events) == 1
