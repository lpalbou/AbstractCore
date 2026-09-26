"""Host cancellation of an in-flight TEXT generation.

A host (AbstractRuntime, for a Stop in the UI) passes ``cancel_event=`` — a
``threading.Event`` — to ``generate()``. Originally the provider
boundary consumed it only to make retry BACKOFF waits cancellable, so a model
that was decoding when the run was cancelled kept decoding until it finished
(an hour, in the incident that motivated this module: a runaway repetitive
generation with no output limit, stopped only by killing the gateway).

CONTRACT
--------
- ``BaseProvider.generate_with_telemetry`` pops ``cancel_event`` (it never
  reaches a provider SDK's request payload) and, when the provider declares
  ``supports_generation_cancel()``, forwards it under the single canonical
  kwarg ``CANCEL_KWARG`` (``_cancel_event``) — the spelling the MLX native
  lane already accepted. A provider that does not declare support never sees
  it.
- A provider that honours the event stops at its next observable point (one
  sampled token on the MLX lanes) and raises ``GenerationCancelledError``.
  The base STREAM loop additionally checks the event between chunks for EVERY
  provider and closes the upstream stream (for HTTP providers: the response),
  so any streaming call stops within one chunk.
- A non-streaming HTTP request that the provider cannot interrupt runs to its
  end; the host's kill switch (AbstractGateway) is the backstop for that case.
- An event already set before the call starts raises immediately: nothing is
  sent to the model.

``GenerationCancelledError`` is ``request_local``: never retried, never
counted against endpoint health.
"""

from __future__ import annotations

import threading
from typing import Any, Optional

#: The one kwarg name the provider seam sees (see module docstring).
CANCEL_KWARG = "_cancel_event"


def as_cancel_event(value: Any) -> Optional[threading.Event]:
    """The value when it is a usable ``threading.Event``, else None."""

    return value if isinstance(value, threading.Event) else None


def is_cancelled(event: Optional[threading.Event]) -> bool:
    return bool(event is not None and event.is_set())


def cancelled_error(*, provider: Any = None, model: Any = None, where: str = "",
                    generated_tokens: Optional[int] = None, partial_text: Optional[str] = None):
    """Build the typed cancellation error (import kept local: no cycle)."""

    from ..exceptions import GenerationCancelledError

    label = f"{provider}/{model}" if provider or model else "provider"
    detail = f" {where}" if where else ""
    counted = f" after {generated_tokens} generated token(s)" if isinstance(generated_tokens, int) else ""
    return GenerationCancelledError(
        f"generation cancelled by the host{detail} ({label}){counted}",
        provider=provider,
        model=model,
        generated_tokens=generated_tokens,
        partial_text=partial_text,
    )


def raise_if_cancelled(event: Optional[threading.Event], **context: Any) -> None:
    if is_cancelled(event):
        raise cancelled_error(**context)


# ---------------------------------------------------------------------------
# HTTP lanes (Ollama, LM Studio, llama.cpp server, vLLM, any OpenAI-compatible
# server): SEVER the in-flight request when the host cancels.
#
# Checking the event "between chunks" is not enough on an HTTP lane: the
# calling thread sits in a blocking socket read (a non-streaming request for
# the WHOLE generation; a streaming request for the whole PREFILL, which is
# tens of seconds on a long prompt), so neither the per-chunk check nor the
# gateway's kill switch (an async exception, delivered only when the thread
# runs Python again) can reach it. The guard below owns a watcher thread that
# waits on the event and, the moment it is set, shuts the request's TCP socket
# down (``shutdown(SHUT_RDWR)`` wakes a reader blocked in ``recv`` on macOS
# and Linux; ``close()`` does not). The blocked read then raises at once and
# the provider reports ``GenerationCancelledError``; the server sees the
# client disconnect and stops decoding (measured per server — see
# docs/generation-cancel.md).
#
# The socket is captured through httpcore's ``trace`` extension
# (``connection.connect_tcp.complete``), which only fires for a NEW
# connection, so a guarded request runs on its own un-pooled client
# (``guard.client(...)``): one extra TCP connect per cancellable request.
# ---------------------------------------------------------------------------

import logging as _logging
import socket as _socket
import time as _time

_log = _logging.getLogger(__name__)

#: How long an idle watcher lingers after its request already ended before it
#: notices and exits. NOT a budget and not a cancel latency: the cancel is
#: observed through ``Event.wait`` the instant the event is set.
WATCH_EXIT_SLICE_S = 0.05


class HttpCancelGuard:
    """Sever one in-flight HTTP request when ``event`` is set.

    Usage (sync)::

        with HttpCancelGuard(event, provider="ollama", model=m, url=u) as guard:
            with guard.client(timeout=t) as client:
                resp = client.post(u, json=payload, extensions=guard.extensions)

    On cancel the blocked read raises a transport error; callers translate it
    with ``guard.cancelled_error_from(exc)`` (or let the BaseProvider do it:
    it converts ANY failure of a call whose event is set into
    ``GenerationCancelledError``).
    """

    def __init__(self, event: Optional[threading.Event], *, provider: Any = None, model: Any = None,
                 url: Any = None) -> None:
        self.event = event
        self.provider = provider
        self.model = model
        self.url = url
        self._lock = threading.Lock()
        self._streams: list = []
        self._done = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.severed_at: Optional[float] = None
        self.severed_sockets = 0

    # -- httpcore trace hook -------------------------------------------------
    def trace(self, name: str, info: Any) -> None:
        if name != "connection.connect_tcp.complete":
            return
        stream = info.get("return_value") if isinstance(info, dict) else None
        if stream is None:
            return
        with self._lock:
            self._streams.append(stream)
            already = self.severed_at is not None
        if already or is_cancelled(self.event):
            # Cancelled while the connection was being set up: cut it now.
            self._sever(force=True)

    @property
    def extensions(self) -> dict:
        return {"trace": self.trace}

    def client(self, *, timeout: Any = None):
        """A fresh, un-pooled ``httpx.Client`` for this one request."""

        import httpx

        return httpx.Client(timeout=timeout)

    # -- watcher -------------------------------------------------------------
    def __enter__(self) -> "HttpCancelGuard":
        if self.event is not None:
            self._thread = threading.Thread(target=self._watch, name="abstractcore-cancel-watch", daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._done.set()

    def _watch(self) -> None:
        event = self.event
        while event is not None and not self._done.is_set():
            if event.wait(WATCH_EXIT_SLICE_S):
                if not self._done.is_set():
                    self._sever()
                return

    def _sever(self, *, force: bool = False) -> None:
        with self._lock:
            if self.severed_at is not None and not force:
                return
            if self.severed_at is None:
                self.severed_at = _time.monotonic()
            streams = list(self._streams)
        cut = 0
        for stream in streams:
            try:
                sock = stream.get_extra_info("socket")
            except Exception:  # noqa: BLE001
                sock = None
            if sock is None:
                continue
            try:
                sock.shutdown(_socket.SHUT_RDWR)
                cut += 1
            except OSError:
                pass  # already closed by the transport: nothing left to cut
        self.severed_sockets += cut
        _log.info(
            "host cancel: severed %d HTTP connection(s) of the in-flight %s/%s request to %s",
            cut, self.provider, self.model, self.url,
        )

    # -- outcome -------------------------------------------------------------
    @property
    def cancelled(self) -> bool:
        return is_cancelled(self.event)

    def cancelled_error_from(self, error: Optional[BaseException] = None, *, where: str = "while waiting for the server"):
        """The typed cancel error when the host cancelled, else None."""

        if not self.cancelled:
            return None
        err = cancelled_error(provider=self.provider, model=self.model, where=where)
        if error is not None:
            err.__cause__ = error
        return err


# ---------------------------------------------------------------------------
# In-flight generations of ONE provider instance, so an EJECT stops them first.
#
# Unloading a model under a running decode is at best a wasted unload (the
# decoding frame still references the weights, so nothing is freed until it
# ends) and at worst a crash (llama.cpp: `Llama.close()` frees the context a
# running `eval` is using). Every unload path therefore calls
# `InflightGenerations.cancel_all()` first: it sets the cancel event of every
# ACTIVE call and waits (bounded, explicit, logged) for each to finish. A call
# still running at the deadline is reported, and the provider REFUSES the
# unload rather than freeing memory under it.
# ---------------------------------------------------------------------------

import weakref as _weakref

#: Default bound on how long an eject waits for cancelled calls to unwind.
#: Explicit and overridable per call (`drain_timeout_s=`); never silent: an
#: expiry is logged at ERROR and refuses the unload.
DEFAULT_EJECT_DRAIN_TIMEOUT_S = 30.0


class _InflightCall:
    __slots__ = ("event", "started", "done", "__weakref__")

    def __init__(self, event: threading.Event) -> None:
        self.event = event
        self.started = _time.monotonic()
        self.done = threading.Event()


class InflightGenerations:
    """Registry of the generations currently running on one provider instance.

    Only calls that carry a cancel event are registered (the host's, or a
    private one the provider creates, see BaseProvider). Entries are weakly
    held: an abandoned, garbage-collected stream disappears by itself.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: "_weakref.WeakSet[_InflightCall]" = _weakref.WeakSet()

    def begin(self, event: Optional[threading.Event]) -> Optional[_InflightCall]:
        if event is None:
            return None
        call = _InflightCall(event)
        with self._lock:
            self._calls.add(call)
        return call

    def end(self, call: Optional[_InflightCall]) -> None:
        if call is None:
            return
        call.done.set()
        with self._lock:
            self._calls.discard(call)

    def active(self) -> int:
        with self._lock:
            return len(self._calls)

    def cancel_all(self, *, reason: str, drain_timeout_s: Optional[float] = None,
                   provider: Any = None, model: Any = None) -> dict:
        """Cancel every active call and wait for each to unwind.

        Returns ``{"cancelled": n, "drained": bool, "still_running": k,
        "waited_s": s, "drain_timeout_s": t}``.
        """

        timeout = DEFAULT_EJECT_DRAIN_TIMEOUT_S if drain_timeout_s is None else float(drain_timeout_s)
        with self._lock:
            calls = list(self._calls)
        if not calls:
            return {"cancelled": 0, "drained": True, "still_running": 0, "waited_s": 0.0,
                    "drain_timeout_s": timeout}
        t0 = _time.monotonic()
        for call in calls:
            call.event.set()
        _log.warning(
            "%s/%s: %s — cancelled %d in-flight generation(s); waiting up to %.1fs for them to stop",
            provider, model, reason, len(calls), timeout,
        )
        deadline = t0 + max(0.0, timeout)
        for call in calls:
            call.done.wait(max(0.0, deadline - _time.monotonic()))
        still = [c for c in calls if not c.done.is_set()]
        waited = round(_time.monotonic() - t0, 3)
        if still:
            _log.error(
                "%s/%s: %s — %d generation(s) still running %.3fs after their cancel "
                "(drain_timeout_s=%.1f); the unload is refused so no memory is freed under them",
                provider, model, reason, len(still), waited, timeout,
            )
        else:
            _log.info("%s/%s: %s — %d cancelled generation(s) stopped in %.3fs",
                      provider, model, reason, len(calls), waited)
        return {"cancelled": len(calls), "drained": not still, "still_running": len(still),
                "waited_s": waited, "drain_timeout_s": timeout}


__all__ = [
    "CANCEL_KWARG",
    "DEFAULT_EJECT_DRAIN_TIMEOUT_S",
    "HttpCancelGuard",
    "InflightGenerations",
    "WATCH_EXIT_SLICE_S",
    "as_cancel_event",
    "cancelled_error",
    "is_cancelled",
    "raise_if_cancelled",
]
