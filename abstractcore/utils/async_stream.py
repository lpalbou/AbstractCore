"""Bounded, cancellation-aware ASGI bridge for internally scheduled providers.

The producer alone advances and closes the synchronous iterator. In particular,
ASGI disconnects never call ``generator.close()`` while another thread is in next().
"""
from __future__ import annotations

import asyncio
import contextvars
import queue
import threading
from typing import Any, AsyncIterator, Iterable


def supports_concurrent_generation(provider: Any) -> bool:
    capability = getattr(provider, "supports_concurrent_generation", None)
    return callable(capability) and capability() is True


async def run_sync_with_disconnect(operation, *, request, cancel_event):
    """Signal a scheduled request when a nonstream HTTP client disappears.

    A socket disconnect does not automatically cancel a FastAPI route task.
    The body has already been parsed when this helper is entered.
    """
    async def watch():
        while True:
            # Do not use Request.is_disconnected()'s immediately cancelled
            # receive: BaseHTTPMiddleware checkpoints before forwarding it,
            # so that poll can miss a real disconnect forever. After the body
            # is parsed this task can await the actual terminal ASGI message.
            message = await request.receive()
            if message["type"] == "http.disconnect":
                cancel_event.set()
                return

    watcher = asyncio.create_task(watch())
    try:
        return await asyncio.to_thread(operation)
    finally:
        cancel_event.set()
        watcher.cancel()
        try:
            await watcher
        except asyncio.CancelledError:
            pass


async def async_stream(source: Iterable[Any], *, max_buffer: int = 32, on_cancel=None) -> AsyncIterator[Any]:
    if isinstance(max_buffer, bool) or not isinstance(max_buffer, int) or max_buffer < 1:
        raise ValueError("max_buffer must be a positive integer")
    out: queue.Queue = queue.Queue(maxsize=max_buffer)
    cancelled = threading.Event()

    def put(item):
        while not cancelled.is_set():
            try:
                out.put(item, timeout=0.05)
                return True
            except queue.Full:
                pass
        return False

    def produce():
        iterator = None
        error = None
        try:
            # Initialization belongs to the producer's failure boundary too:
            # __iter__ can raise or block before a first next() ever happens.
            iterator = iter(source)
            while not cancelled.is_set():
                try:
                    item = next(iterator)
                except StopIteration:
                    break
                if not put(("item", item)):
                    break
        except BaseException as exc:
            error = exc
        finally:
            try:
                # When initialization failed, the original source may still
                # own resources. Never close it from the ASGI consumer thread.
                close = getattr(iterator if iterator is not None else source, "close", None)
                if callable(close):
                    close()
            except BaseException as exc:
                # Preserve the original producer failure if cleanup also fails,
                # but do not turn a normal-consumption close failure into success.
                if error is None:
                    error = exc
            # Exactly one terminal envelope, after owner-thread cleanup. An
            # already cancelled consumer needs no terminal queue delivery.
            put(("error", error) if error is not None else ("done", None))

    # Match asyncio.to_thread's request-local context propagation. Endpoint
    # resolvers and tool/routing context must not disappear at this boundary.
    context = contextvars.copy_context()
    worker = threading.Thread(target=context.run, args=(produce,),
                              name="abstractcore-http-stream", daemon=True)
    worker.start()
    try:
        while True:
            # A timed get avoids leaving an executor thread blocked forever when
            # the awaiting ASGI task is cancelled before the model yields again.
            try:
                kind, value = await asyncio.to_thread(out.get, True, 0.1)
            except queue.Empty:
                continue
            if kind == "done":
                break
            if kind == "error":
                raise value
            yield value
    finally:
        cancelled.set()
        if on_cancel is not None:
            # Thread-safe request cancellation is separate from iterator.close:
            # it can cancel queued/prefill work while producer next() is blocked.
            on_cancel()
