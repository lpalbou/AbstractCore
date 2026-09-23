"""Independent HTTP contracts for provider-owned concurrency; no GPU or external services."""
from __future__ import annotations

import asyncio
import base64
import importlib
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.endpoint.app import create_app
from abstractcore.providers.mlx_runtime import NativeRuntimeError
from abstractcore.utils.async_stream import async_stream


try:  # Python 3.9 has no builtin anext()
    anext
except NameError:  # pragma: no cover - exercised on Python 3.9 only
    async def anext(iterator):  # noqa: A001
        return await iterator.__anext__()


class ScheduledProvider:
    model = "native-test-model"
    provider = "mlx"

    def __init__(self, *, barrier=False):
        self.calls = []
        self.lock = threading.Lock()
        self.active = self.peak = 0
        self.barrier = threading.Barrier(2, timeout=2) if barrier else None

    def supports_concurrent_generation(self):
        return True

    def generate(self, *, stream=False, **kwargs):
        with self.lock:
            self.calls.append(kwargs)
        if stream:
            def chunks():
                for text in ("A", " ", "B"):
                    yield GenerateResponse(content=text, model=self.model)
                yield self.result("")
            return chunks()
        with self.lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.barrier is not None:
                self.barrier.wait()
            return self.result("A B")
        finally:
            with self.lock:
                self.active -= 1

    def result(self, text):
        return GenerateResponse(
            content=text, model=self.model, finish_reason="length",
            usage={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            metadata={"execution": {"mode": "cohort", "peak_batch_size": 2},
                      "speculation": {"requested": True, "used": True, "num_draft_tokens": 2}},
        )


@pytest.fixture
def application_factory(monkeypatch):
    runtimes = []

    def make(kind, provider):
        if kind == "endpoint":
            return create_app(provider_instance=provider), provider.model
        server = importlib.import_module("abstractcore.server.app")
        runtime = server._GatewayLoadedRuntime(
            provider="mlx", model=provider.model, base_url=None,
            explicit_provider_key_hash=None, llm=provider,
        )
        runtimes.append(runtime)
        monkeypatch.setattr(server, "_get_loaded_gateway_runtime", lambda **kwargs: runtime)
        monkeypatch.setattr(server, "create_llm", lambda *args, **kwargs: provider)
        return server.app, "mlx/" + provider.model

    yield make
    for runtime in runtimes:
        runtime.provider_executor.shutdown(wait=True, cancel_futures=True)


def payload(model, **kwargs):
    return {"model": model, "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 3, "temperature": 0, "seed": 0, "top_p": .8,
            "speculation": False, **kwargs}


@pytest.mark.parametrize("kind", ["endpoint", "gateway"])
def test_http_admits_concurrent_native_requests_without_outer_serialization(application_factory, kind):
    provider = ScheduledProvider(barrier=True)
    app, model = application_factory(kind, provider)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            return await asyncio.gather(*[
                client.post("/v1/chat/completions", json=payload(model)) for _ in range(2)
            ])

    responses = asyncio.run(exercise())
    assert [response.status_code for response in responses] == [200, 200]
    assert provider.peak == 2, "HTTP executor serialized requests before the native scheduler"
    assert all(call["speculation"] is False for call in provider.calls)
    assert all(call["seed"] == 0 and call["top_p"] == .8 for call in provider.calls)


@pytest.mark.parametrize("kind", ["endpoint", "gateway"])
def test_http_keeps_native_finish_reason_usage_and_execution_evidence(application_factory, kind):
    provider = ScheduledProvider()
    app, model = application_factory(kind, provider)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.post("/v1/chat/completions", json=payload(model))

    response = asyncio.run(exercise())
    assert response.status_code == 200
    body = response.json()
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 3
    assert body["abstractcore"]["execution"]["peak_batch_size"] == 2
    assert body["abstractcore"]["speculation"]["used"] is True


@pytest.mark.parametrize("kind", ["endpoint", "gateway"])
def test_http_native_stream_preserves_whitespace_and_terminal_accounting(application_factory, kind):
    provider = ScheduledProvider()
    app, model = application_factory(kind, provider)

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.post("/v1/chat/completions", json=payload(model, stream=True))

    response = asyncio.run(exercise())
    assert response.status_code == 200
    events = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert "".join(event["choices"][0].get("delta", {}).get("content", "")
                   for event in events if event.get("choices")) == "A B"
    terminals = [event for event in events if event.get("choices")
                 and event["choices"][0].get("finish_reason") is not None]
    assert len(terminals) == 1
    assert terminals[0]["choices"][0]["finish_reason"] == "length"
    assert any(event.get("usage", {}).get("completion_tokens") == 3 for event in events)
    assert any(event.get("abstractcore", {}).get("execution", {}).get("peak_batch_size") == 2 for event in events)


@pytest.mark.parametrize("kind", ["endpoint", "gateway"])
@pytest.mark.parametrize("stream", [False, True])
def test_http_preserves_speculation_without_scheduler_metadata(application_factory, kind, stream):
    """MTP availability/actuals do not require the optional batch scheduler."""
    outcome = {"requested": True, "used": False, "reason": "unsupported_request"}

    class UnscheduledProvider(ScheduledProvider):
        def supports_concurrent_generation(self):
            return False

        def result(self, text):
            response = super().result(text)
            response.metadata = {"speculation": outcome, "private_provider_state": "not-on-wire"}
            return response

    app, model = application_factory(kind, UnscheduledProvider())

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.post("/v1/chat/completions", json=payload(model, stream=stream))

    response = asyncio.run(exercise())
    assert response.status_code == 200
    bodies = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"] if stream else [response.json()]
    extensions = [body["abstractcore"] for body in bodies if "abstractcore" in body]
    assert any(extension.get("speculation") == outcome for extension in extensions)
    assert all("execution" not in extension and "private_provider_state" not in extension
               for extension in extensions)


@pytest.mark.parametrize("kind", ["endpoint", "gateway"])
def test_http_stream_keeps_metadata_only_outcome_before_usage_terminal(application_factory, kind):
    outcome = {"requested": True, "used": True, "num_draft_tokens": 4}

    class MetadataOnlyProvider(ScheduledProvider):
        def supports_concurrent_generation(self):
            return False

        def generate(self, **kwargs):
            yield GenerateResponse(content="hello", model=self.model)
            yield GenerateResponse(content="", model=self.model, metadata={"speculation": outcome})
            yield GenerateResponse(content="", model=self.model, finish_reason="stop",
                                   usage={"prompt_tokens": 2, "completion_tokens": 1})

    app, model = application_factory(kind, MetadataOnlyProvider())

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.post("/v1/chat/completions", json=payload(model, stream=True))

    response = asyncio.run(exercise())
    assert response.status_code == 200
    events = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines()
              if line.startswith("data: ") and line != "data: [DONE]"]
    assert any(event.get("abstractcore", {}).get("speculation") == outcome for event in events)
    assert any(event.get("usage", {}).get("completion_tokens") == 1 for event in events)


@pytest.mark.parametrize("kind", ["endpoint", "gateway"])
@pytest.mark.parametrize("stream", [False, True])
def test_http_native_failure_preserves_actionable_error_not_a_fake_completion(application_factory, kind, stream):
    message = "Native MLX waiting queue is full; retry after active requests finish"

    class FailingProvider(ScheduledProvider):
        def generate(self, *, stream=False, **kwargs):
            error = NativeRuntimeError(message)
            error.code = "native_queue_full"
            error.http_status = 503
            if not stream:
                raise error

            def chunks():
                yield GenerateResponse(content="partial", model=self.model)
                raise error

            return chunks()

    app, model = application_factory(kind, FailingProvider())

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
                                     base_url="http://testserver") as client:
            return await client.post("/v1/chat/completions", json=payload(model, stream=stream))

    response = asyncio.run(exercise())
    if stream:
        events = [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines()
                  if line.startswith("data: ") and line != "data: [DONE]"]
        assert any(event.get("error", {}).get("message") == message for event in events), response.text
        assert not any(event.get("choices") and event["choices"][0].get("finish_reason") in ("stop", "length")
                       for event in events), "failed generation was reported as successfully complete"
    else:
        assert response.status_code >= 400
        body = response.json()
        detail = body.get("detail", body)
        assert detail["error"]["message"] == message
        if kind == "endpoint":
            assert response.status_code == 503


@pytest.mark.parametrize("kind", ["endpoint", "gateway"])
@pytest.mark.parametrize("stream", [False, True])
def test_real_http_disconnect_reaches_native_owner_without_asgi_task_cancellation(application_factory, kind, stream):
    uvicorn = pytest.importorskip("uvicorn")
    entered = threading.Event()
    observed_cancel = threading.Event()
    cleanup = threading.Event()

    class WaitingProvider(ScheduledProvider):
        def generate(self, *, stream=False, **kwargs):
            event = kwargs["_cancel_event"]

            def wait_for_cancel():
                entered.set()
                deadline = time.monotonic() + 4
                while not cleanup.is_set() and time.monotonic() < deadline:
                    if event.wait(.025):
                        observed_cancel.set()
                        return

            if stream:
                def chunks():
                    yield GenerateResponse(content="first-native-token", model=self.model)
                    wait_for_cancel()
                return chunks()
            wait_for_cancel()
            return self.result("finished")

    app, model = application_factory(kind, WaitingProvider())
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    port = listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", loop="asyncio", lifespan="off"))
    worker = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    worker.start()

    async def exercise():
        deadline = time.monotonic() + 3
        while not server.started and time.monotonic() < deadline:
            await asyncio.sleep(.01)
        assert server.started
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=3) as client:
            if stream:
                async with client.stream("POST", "/v1/chat/completions", json=payload(model, stream=True)) as response:
                    async for line in response.aiter_lines():
                        if "first-native-token" in line:
                            break
                # Closing the real client connection must cancel the backend
                # even while the server's producer is blocked inside next().
            else:
                pending = asyncio.create_task(client.post("/v1/chat/completions", json=payload(model)))
                assert await asyncio.to_thread(entered.wait, 2)
                pending.cancel()  # Cancels CLIENT I/O, not the server/ASGI task.
                with pytest.raises(asyncio.CancelledError):
                    await pending
            assert await asyncio.to_thread(observed_cancel.wait, 1.5), "real client disconnect left native generation running"

    try:
        asyncio.run(exercise())
    finally:
        cleanup.set()
        server.should_exit = True
        worker.join(timeout=5)
        listener.close()
        assert not worker.is_alive(), "test-owned localhost server did not stop"


@pytest.mark.parametrize("stream", [False, True])
def test_endpoint_native_inline_image_reaches_provider_as_exact_bytes_and_clean_text(application_factory, stream):
    from abstractcore.media.types import MediaType, ContentFormat
    provider = ScheduledProvider()
    app, model = application_factory("endpoint", provider)
    encoded = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aO2sAAAAASUVORK5CYII="
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded, "detail": "high"}}
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Be concise."}]},
        {"role": "user", "content": [
            {"type": "text", "text": "Identify this."}, image,
            {"type": "text", "text": "Use one word."},
        ]},
    ]

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.post("/v1/chat/completions", json=payload(model, messages=messages, stream=stream))

    response = asyncio.run(exercise())
    assert response.status_code == 200
    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["system_prompt"] == "Be concise."
    assert call["messages"] == [{"role": "user", "content": "Identify this.\nUse one word."}]
    assert len(call["media"]) == 1
    part = call["media"][0]
    assert part.media_type is MediaType.IMAGE and part.content_format is ContentFormat.BINARY
    assert part.content == base64.b64decode(encoded)
    assert part.metadata == {"detail": "high"}
    assert "data:" not in str(call["messages"])


@pytest.mark.parametrize("invalid", ["remote", "history", "unsupported_part"])
def test_endpoint_native_media_rejection_is_400_before_generation(application_factory, invalid):
    provider = ScheduledProvider()
    app, model = application_factory("endpoint", provider)
    encoded = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aO2sAAAAASUVORK5CYII="
    url = "https://127.0.0.1/private?secret=must-not-echo" if invalid == "remote" else "data:image/png;base64," + encoded
    messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}]}]
    if invalid == "history":
        messages.append({"role": "user", "content": "Follow-up"})
    if invalid == "unsupported_part":
        messages[0]["content"] = [{"type": "input_audio", "data": "must-not-echo"}]

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.post("/v1/chat/completions", json=payload(model, messages=messages))

    response = asyncio.run(exercise())
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "native_media_invalid"
    assert "must-not-echo" not in response.text
    assert not provider.calls


def test_managed_control_operations_remain_serial_and_worker_affine():
    server = importlib.import_module("abstractcore.server.app")
    runtime = server._GatewayLoadedRuntime(provider="mlx", model="test", base_url=None,
                                           explicit_provider_key_hash=None, llm=ScheduledProvider())
    entered, release, second_entered = threading.Event(), threading.Event(), threading.Event()
    threads = []

    def first():
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(2)

    def second():
        threads.append(threading.get_ident())
        second_entered.set()

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            a = pool.submit(server._run_loaded_gateway_runtime, runtime, first)
            assert entered.wait(1)
            b = pool.submit(server._run_loaded_gateway_runtime, runtime, second)
            assert not second_entered.wait(.05)
            release.set()
            a.result(timeout=1)
            b.result(timeout=1)
        assert len(set(threads)) == 1
        assert threads[0] != threading.get_ident()
    finally:
        release.set()
        runtime.provider_executor.shutdown(wait=True, cancel_futures=True)


def test_disconnect_cancels_request_while_source_next_is_blocked():
    entered, cancel_request, closed = threading.Event(), threading.Event(), threading.Event()
    threads = []

    class WaitingSource:
        def __iter__(self):
            return self

        def __next__(self):
            threads.append(threading.get_ident())
            entered.set()
            assert cancel_request.wait(2), "disconnect never reached queued native request"
            raise StopIteration

        def close(self):
            threads.append(threading.get_ident())
            closed.set()

    async def exercise():
        bridge = async_stream(WaitingSource(), on_cancel=cancel_request.set)
        waiting = asyncio.create_task(anext(bridge))
        assert await asyncio.to_thread(entered.wait, 1)
        waiting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiting
        assert await asyncio.to_thread(closed.wait, 1)
        await bridge.aclose()

    asyncio.run(exercise())
    assert len(threads) == 2 and len(set(threads)) == 1
    assert threads[0] != threading.get_ident(), "ASGI loop closed a running producer iterator"


def test_bridge_propagates_producer_error_and_still_closes_source():
    closed = threading.Event()
    failure = RuntimeError("injected stream transport failure")

    def source():
        try:
            yield "prefix"
            raise failure
        finally:
            closed.set()

    async def exercise():
        bridge = async_stream(source())
        assert await anext(bridge) == "prefix"
        with pytest.raises(RuntimeError, match="injected stream transport failure") as caught:
            await anext(bridge)
        assert caught.value is failure

    asyncio.run(exercise())
    assert closed.is_set()


def test_stalled_bridge_consumer_is_bounded_and_close_releases_producer():
    closed, third_item = threading.Event(), threading.Event()
    produced = []

    def source():
        try:
            for index in range(100):
                produced.append(index)
                if index >= 2:
                    third_item.set()
                yield str(index)
        finally:
            closed.set()

    async def exercise():
        bridge = async_stream(source(), max_buffer=1)
        assert await anext(bridge) == "0"
        assert await asyncio.to_thread(third_item.wait, 1)
        assert len(produced) <= 3
        await bridge.aclose()
        assert await asyncio.to_thread(closed.wait, 1)

    asyncio.run(exercise())


@pytest.mark.parametrize("kind", ["endpoint", "gateway"])
def test_nonstream_http_cancellation_signals_native_request(application_factory, kind):
    entered, finished = threading.Event(), threading.Event()

    class WaitingProvider(ScheduledProvider):
        def generate(self, **kwargs):
            cancel = kwargs["_cancel_event"]
            assert isinstance(cancel, threading.Event)
            entered.set()
            assert cancel.wait(2), "HTTP cancellation did not reach native request"
            finished.set()
            return self.result("")

    app, model = application_factory(kind, WaitingProvider())

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver") as client:
            task = asyncio.create_task(client.post("/v1/chat/completions", json=payload(model)))
            assert await asyncio.to_thread(entered.wait, 1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            assert await asyncio.to_thread(finished.wait, 1)

    asyncio.run(exercise())
