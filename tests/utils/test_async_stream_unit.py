"""CPU-only synchronous-to-async stream lifecycle and failure contracts."""

import asyncio
import contextvars
import threading

import pytest

from abstractcore.utils.async_stream import async_stream


try:  # Python 3.9 has no builtin anext()
    anext
except NameError:  # pragma: no cover - exercised on Python 3.9 only
    async def anext(iterator):  # noqa: A001
        return await iterator.__anext__()


@pytest.mark.parametrize("cancel", [False, True])
def test_producer_preserves_and_isolates_request_context_through_close(cancel):
    scope = contextvars.ContextVar("native_stream_scope", default="missing")
    seen = []
    gate = threading.Barrier(2, timeout=2)

    async def consume(label):
        token = scope.set(label)
        closed = threading.Event()

        class Source:
            def __iter__(self):
                seen.append((label, "iter", scope.get()))
                gate.wait()
                self.count = 0
                return self

            def __next__(self):
                self.count += 1
                if self.count > (100 if cancel else 2):
                    raise StopIteration
                seen.append((label, "next", scope.get()))
                return scope.get()

            def close(self):
                seen.append((label, "close", scope.get()))
                closed.set()

        bridge = async_stream(Source(), max_buffer=1)
        try:
            assert await anext(bridge) == label
            if not cancel:
                assert [value async for value in bridge] == [label]
        finally:
            await bridge.aclose()
            assert await asyncio.to_thread(closed.wait, 1)
            scope.reset(token)

    async def exercise():
        await asyncio.wait_for(asyncio.gather(consume("first"), consume("second")), 4)

    asyncio.run(exercise())
    assert all(label == value for label, _, value in seen)
    assert sum(kind == "close" for _, kind, _ in seen) == 2
    assert scope.get() == "missing"


def test_iterator_initialization_failure_is_delivered_without_hanging():
    failure = RuntimeError("source initialization failed")
    closed = threading.Event()
    threads = []

    class BrokenSource:
        def __iter__(self):
            threads.append(threading.get_ident())
            raise failure

        def close(self):
            threads.append(threading.get_ident())
            closed.set()

    async def exercise():
        bridge = async_stream(BrokenSource())
        with pytest.raises(RuntimeError, match="source initialization failed") as caught:
            await asyncio.wait_for(anext(bridge), timeout=.5)
        assert caught.value is failure
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(bridge), timeout=.5)

    asyncio.run(exercise())
    assert closed.is_set()
    assert len(threads) == 2 and len(set(threads)) == 1
    assert threads[0] != threading.get_ident()


def test_close_failure_after_normal_consumption_is_not_silent_success():
    failure = RuntimeError("source close failed")
    events = []

    class Source:
        remaining = ["first", "second"]

        def __iter__(self):
            events.append(("iter", threading.get_ident()))
            return self

        def __next__(self):
            events.append(("next", threading.get_ident()))
            if self.remaining:
                return self.remaining.pop(0)
            raise StopIteration

        def close(self):
            events.append(("close", threading.get_ident()))
            raise failure

    async def exercise():
        bridge = async_stream(Source(), max_buffer=1)
        assert await asyncio.wait_for(anext(bridge), timeout=.5) == "first"
        assert await asyncio.wait_for(anext(bridge), timeout=.5) == "second"
        with pytest.raises(RuntimeError, match="source close failed") as caught:
            await asyncio.wait_for(anext(bridge), timeout=.5)
        assert caught.value is failure
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(bridge), timeout=.5)

    asyncio.run(exercise())
    assert [name for name, _ in events].count("close") == 1
    assert len({thread for _, thread in events}) == 1
    assert events[0][1] != threading.get_ident()


def test_iteration_failure_wins_over_secondary_close_failure():
    primary = ValueError("primary decoding failure")
    secondary = RuntimeError("secondary cleanup failure")
    closed = threading.Event()

    class Source:
        def __iter__(self):
            return self

        def __next__(self):
            raise primary

        def close(self):
            closed.set()
            raise secondary

    async def exercise():
        bridge = async_stream(Source())
        with pytest.raises(ValueError, match="primary decoding failure") as caught:
            await asyncio.wait_for(anext(bridge), timeout=.5)
        assert caught.value is primary
        assert closed.is_set(), "A terminal response was delivered before owner cleanup"

    asyncio.run(exercise())


def test_close_lookup_failure_is_delivered_as_stream_error():
    failure = ValueError("close lookup failed")

    class Source:
        def __iter__(self):
            return self

        def __next__(self):
            raise StopIteration

        @property
        def close(self):
            raise failure

    async def exercise():
        with pytest.raises(ValueError) as caught:
            await asyncio.wait_for(anext(async_stream(Source())), timeout=.5)
        assert caught.value is failure

    asyncio.run(exercise())


@pytest.mark.parametrize("where", ["iter", "next", "close"])
def test_base_exception_from_producer_does_not_leave_consumer_waiting(where):
    class ProducerFailure(BaseException):
        pass

    failure = ProducerFailure("unexpected producer failure")
    closed = threading.Event()

    class Source:
        def __iter__(self):
            if where == "iter":
                raise failure
            return self

        def __next__(self):
            if where == "next":
                raise failure
            raise StopIteration

        def close(self):
            closed.set()
            if where == "close":
                raise failure

    async def exercise():
        with pytest.raises(ProducerFailure) as caught:
            await asyncio.wait_for(anext(async_stream(Source())), timeout=.5)
        assert caught.value is failure

    asyncio.run(exercise())
    assert closed.is_set()


def test_distinct_iterator_is_advanced_and_closed_on_its_initializing_thread():
    events, cancellations = [], []

    class Iterator:
        remaining = ["a", None, "", "b"]

        def __iter__(self):
            raise AssertionError("The bridge must not initialize the returned iterator twice")

        def __next__(self):
            events.append(("next", threading.get_ident()))
            if self.remaining:
                return self.remaining.pop(0)
            raise StopIteration

        def close(self):
            events.append(("close", threading.get_ident()))

    class Source:
        def __iter__(self):
            events.append(("iter", threading.get_ident()))
            return Iterator()

        def close(self):
            raise AssertionError("Only the created iterator owns successful iteration cleanup")

    async def exercise():
        result = []
        async for value in async_stream(Source(), max_buffer=1,
                                         on_cancel=lambda: cancellations.append(threading.get_ident())):
            result.append(value)
        return result

    assert asyncio.run(asyncio.wait_for(exercise(), timeout=1)) == ["a", None, "", "b"]
    assert [name for name, _ in events] == ["iter"] + ["next"] * 5 + ["close"]
    assert len({thread for _, thread in events}) == 1
    assert events[0][1] != threading.get_ident()
    assert cancellations == [threading.get_ident()]


def test_cancellation_during_iterator_initialization_closes_without_advancing():
    entered, cancelled, closed = threading.Event(), threading.Event(), threading.Event()
    events = []

    class Source:
        def __iter__(self):
            events.append(("iter", threading.get_ident()))
            entered.set()
            assert cancelled.wait(1), "Cancellation did not reach blocked initialization"
            return self

        def __next__(self):
            raise AssertionError("A cancelled producer must not begin advancing its source")

        def close(self):
            events.append(("close", threading.get_ident()))
            closed.set()

    async def exercise():
        bridge = async_stream(Source(), on_cancel=cancelled.set)
        pending = asyncio.create_task(anext(bridge))
        try:
            assert await asyncio.to_thread(entered.wait, .5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, timeout=.5)
            assert await asyncio.to_thread(closed.wait, .5)
            await bridge.aclose()
        finally:
            cancelled.set()

    asyncio.run(exercise())
    assert [name for name, _ in events] == ["iter", "close"]
    assert len({thread for _, thread in events}) == 1
    assert events[0][1] != threading.get_ident()


@pytest.mark.parametrize("value", [0, -1, True, False, 1.5, None])
def test_invalid_buffer_does_not_start_source(value):
    class Source:
        def __iter__(self):
            pytest.fail("Invalid configuration must not start a producer")

    async def exercise():
        with pytest.raises(ValueError, match="positive integer"):
            await asyncio.wait_for(anext(async_stream(Source(), max_buffer=value)), timeout=.5)

    asyncio.run(exercise())


def test_non_iterable_source_fails_promptly():
    async def exercise():
        with pytest.raises(TypeError, match="not iterable"):
            await asyncio.wait_for(anext(async_stream(object())), timeout=.5)

    asyncio.run(exercise())


def test_empty_source_has_one_normal_terminal_and_one_cancellation_callback():
    cancellations = []

    async def exercise():
        bridge = async_stream([], on_cancel=lambda: cancellations.append(True))
        for _ in range(2):
            with pytest.raises(StopAsyncIteration):
                await asyncio.wait_for(anext(bridge), timeout=.5)
        await bridge.aclose()

    asyncio.run(exercise())
    assert cancellations == [True]
