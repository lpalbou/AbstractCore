"""Concurrent generations on ONE llama.cpp weights instance: a per-model slot pool.

A SLOT is one llama.cpp context (its own KV, n_ctx = the model's reserved context) built on the
model's shared weights (`gguf_shared`), with ONE worker thread: llama.cpp contexts are not
thread-safe, distinct contexts on one model run in parallel (the decode releases the GIL). A
generation BORROWS a slot for its duration and returns it; provider instances (a runtime pool,
a gateway-side client, a server request) own no context, they attach to the model's scheduler.

- Bound: at most `max_concurrent` slots per model (llama.cpp on a GPU / Metal: 4; CPU-only: 1;
  `model_concurrency` in the AbstractCore config overrides it per model).
- Lazy: slot 0 is the load's own context; another slot is built when a request arrives while
  every slot is busy, and only when its KV fits the free memory now (`slot_fits`), never by
  evicting another model and never with a smaller context; otherwise the request waits for a
  slot and the refusal reason is kept (`state()["slot_refusal_reason"]`).
- Queue: FIFO by admission, bounded (`max_queue`, default 32: `ModelQueueFullError`), each
  request waits at most `queue_timeout_s` (default 120: `ModelQueueTimeoutError`); a request
  cancelled while queued leaves the queue. The queue position is reported (`on_queued`).
- Idle extra slots are freed after `idle_ttl_s` (default 300 s); slot 0 stays with the weights,
  and so do the slots a person pinned (`model_concurrency`: built at load, used exactly).
- FIFO, with one bounded inversion: while the queue head builds a new slot (under the load lock),
  a slot freed meanwhile goes to the next request in line.
- `slot_fits` measures free GPU memory on one CUDA card, else available RAM (unified memory on
  Apple Silicon: Metal allocates from it).
- Holders are counted by attachment, not by reference: a holder that is simply dropped (a
  rebuilt host's old client) detaches through a finalizer, and the last detach frees every slot
  and, with the last context, the weights. Closing refuses while a slot is generating or a
  request waits (`ModelBusyError`): a context freed under a running decode would crash the
  process.

The interface is `_InstanceGenerationWorker`'s (`run`, `stream`), so the provider's lanes are
unchanged: while a call runs on a slot, the provider's `llm` is that slot's context
(`current_llm`).
"""

from __future__ import annotations

import contextvars
import threading
import time
import weakref
from collections import deque
from typing import Any, Callable, Deque, Dict, Iterator, List, Optional, Tuple

from ..exceptions import GenerationCancelledError, ProviderAPIError

DEFAULT_MAX_QUEUE = 32
DEFAULT_QUEUE_TIMEOUT_S = 120.0
DEFAULT_IDLE_TTL_S = 300.0
GPU_DEFAULT_CONCURRENCY = 4
CPU_DEFAULT_CONCURRENCY = 1

#: (scheduler, slot) of the call running on this thread, or None.
_CURRENT: "contextvars.ContextVar[Optional[Tuple[Any, Any]]]" = contextvars.ContextVar(
    "abstractcore_gguf_slot", default=None)


class ModelQueueFullError(ProviderAPIError):
    """Every slot of the model is busy and its queue is full: refused now, never silently dropped.
    Request-local (Core never resamples it); 503 with a Retry-After for the host's own retry."""

    request_local = True


class ModelQueueTimeoutError(ProviderAPIError):
    """The request waited longer than its queue timeout for a slot of the model (it never ran).
    Request-local; 504 with a Retry-After."""

    request_local = True


class ModelBusyError(ProviderAPIError):
    """The model's slot pool cannot be closed: a slot is generating or a request waits."""

    request_local = True


class ModelQueueCancelled(GenerationCancelledError):
    """The host cancelled the request while it waited for a slot (it never ran): a cancel, never a
    provider fault, so nothing retries it."""


def current_llm(scheduler: Any) -> Optional[Any]:
    """The context of the slot this thread's call borrowed from `scheduler`, else None."""
    cur = _CURRENT.get()
    if cur is not None and cur[0] is scheduler:
        return cur[1].llm
    return None


class _Slot:
    def __init__(self, index: int, llm: Any) -> None:
        self.index = index
        self.llm = llm
        self.busy = False
        self.last_used = time.monotonic()
        self._executor: Any = None
        self._lock = threading.Lock()

    def executor(self) -> Any:
        with self._lock:
            if self._executor is None:
                from concurrent.futures import ThreadPoolExecutor

                self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"abstractcore-gguf-slot{self.index}")
            return self._executor

    def shutdown(self) -> None:
        with self._lock:
            ex, self._executor = self._executor, None
        if ex is not None:
            ex.shutdown(wait=False)


class SlotScheduler:
    def __init__(self, *, key: str, model: str, first_llm: Any, n_ctx: int,
                 make_slot: Callable[[], Any], slot_fits: Callable[[], Optional[str]],
                 max_concurrent: int, max_queue: int = DEFAULT_MAX_QUEUE,
                 queue_timeout_s: float = DEFAULT_QUEUE_TIMEOUT_S, idle_ttl_s: float = DEFAULT_IDLE_TTL_S,
                 close_llm: Optional[Callable[[Any], None]] = None) -> None:
        self.key = key
        self.model = model
        self.n_ctx = int(n_ctx)
        self.max_concurrent = max(1, int(max_concurrent))
        self.max_queue = max(0, int(max_queue))
        self.queue_timeout_s = float(queue_timeout_s)
        self.idle_ttl_s = float(idle_ttl_s)
        self._make_slot = make_slot
        self._slot_fits = slot_fits
        self._close_llm = close_llm or _default_close
        self._cv = threading.Condition()
        self._slots: List[_Slot] = [_Slot(0, first_llm)]
        self._queue: Deque[object] = deque()
        self._growing = False
        self.slot_refusal_reason: Optional[str] = None
        self.holders: "weakref.WeakSet[Any]" = weakref.WeakSet()
        #: Attachment tokens (one per attached holder); the pool lives while one remains.
        self._attached: set = set()
        #: Slots a person pinned (`model_concurrency`): never reaped.
        self.pinned_slots = 1
        #: The last holder left while work was still running or waiting: close when it drains.
        self.close_when_idle = False
        self.closed = False
        # What a holder attaching to the resident model takes over (filled by the loader).
        self.attach_state: Dict[str, Any] = {}
        self.queued_total = 0
        self.max_queued_s = 0.0

    # -- introspection ---------------------------------------------------------------------
    @property
    def first_llm(self) -> Any:
        return self._slots[0].llm if self._slots else None

    def slot_llms(self) -> List[Any]:
        with self._cv:
            return [s.llm for s in self._slots]

    def state(self) -> Dict[str, Any]:
        with self._cv:
            return {"slots": len(self._slots), "slots_max": self.max_concurrent,
                    "active": sum(1 for s in self._slots if s.busy), "queued": len(self._queue),
                    "slot_refusal_reason": self.slot_refusal_reason, "slot_n_ctx": self.n_ctx,
                    "holders": len(self._attached)}

    def busy(self) -> bool:
        with self._cv:
            return bool(self._queue) or any(s.busy for s in self._slots)

    # -- admission -------------------------------------------------------------------------
    def _acquire(self, cancel_event: Optional[threading.Event], on_queued: Optional[Callable[[int, Dict[str, Any]], None]],
                 timeout_s: Optional[float]) -> Tuple[_Slot, float]:
        ticket = object()
        t0 = time.monotonic()
        deadline = t0 + (self.queue_timeout_s if timeout_s is None else float(timeout_s))
        last_position = None
        with self._cv:
            if self.closed:
                raise ProviderAPIError(f"{self.model} was unloaded; load it again to use it.")
            idle = next((s for s in self._slots if not s.busy), None)
            if idle is not None and not self._queue:
                idle.busy = True
                return idle, 0.0
            if len(self._queue) >= self.max_queue:
                raise ModelQueueFullError(
                    f"{self.model} is busy: all {len(self._slots)} of its slots are generating and "
                    f"{len(self._queue)} requests are already waiting (the queue holds {self.max_queue}). "
                    "Try again shortly.", status_code=503, retry_after_s=5.0)
            self._queue.append(ticket)
            self.queued_total += 1
            try:
                while True:
                    if self.closed:
                        raise ProviderAPIError(f"{self.model} was unloaded while this request waited for it.")
                    if self._queue and self._queue[0] is ticket:
                        idle = next((s for s in self._slots if not s.busy), None)
                        if idle is not None:
                            self._queue.popleft()
                            idle.busy = True
                            waited = time.monotonic() - t0
                            self.max_queued_s = max(self.max_queued_s, waited)
                            self._cv.notify_all()
                            return idle, waited
                        if len(self._slots) < self.max_concurrent and not self._growing:
                            reason = self._slot_fits()
                            if reason is None:
                                self._growing = True
                                self._cv.release()
                                try:
                                    llm, err = None, None
                                    try:
                                        llm = self._make_slot()
                                    except Exception as exc:  # noqa: BLE001 - kept as the reason, the request waits
                                        err = exc
                                finally:
                                    self._cv.acquire()
                                    self._growing = False
                                if llm is not None:
                                    slot = _Slot(len(self._slots), llm)
                                    slot.busy = True
                                    self._slots.append(slot)
                                    self._queue.popleft()
                                    self.slot_refusal_reason = None
                                    waited = time.monotonic() - t0
                                    self.max_queued_s = max(self.max_queued_s, waited)
                                    self._cv.notify_all()
                                    return slot, waited
                                self.slot_refusal_reason = (
                                    f"slot {len(self._slots) + 1} of {self.model} could not be built: {err}")
                            else:
                                self.slot_refusal_reason = reason
                    position = list(self._queue).index(ticket) + 1 if ticket in self._queue else 0
                    if on_queued is not None:
                        last_position = position
                        try:
                            on_queued(position, self._state_unlocked())
                        except Exception:  # noqa: BLE001 - observability never breaks admission
                            pass
                    if cancel_event is not None and cancel_event.is_set():
                        raise ModelQueueCancelled(f"The request for {self.model} was cancelled while it waited for a slot.",
                                                  provider="huggingface", model=self.model, generated_tokens=0)
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ModelQueueTimeoutError(
                            f"{self.model} stayed busy for {self.queue_timeout_s:.0f} s: all {len(self._slots)} of its "
                            f"slots were generating" + (f" ({self.slot_refusal_reason})" if self.slot_refusal_reason else "")
                            + ". The request was not run; try again.", status_code=504, retry_after_s=10.0)
                    self._cv.wait(timeout=min(0.25, remaining))
            except BaseException:
                try:
                    self._queue.remove(ticket)
                except ValueError:
                    pass
                self._cv.notify_all()
                raise

    def _state_unlocked(self) -> Dict[str, Any]:
        return {"slots": len(self._slots), "slots_max": self.max_concurrent,
                "active": sum(1 for s in self._slots if s.busy), "queued": len(self._queue),
                "slot_refusal_reason": self.slot_refusal_reason}

    def _release(self, slot: _Slot) -> None:
        with self._cv:
            slot.busy = False
            slot.last_used = time.monotonic()
            self._cv.notify_all()
            drained = self.close_when_idle and not self._queue and not any(s.busy for s in self._slots)
        if drained:
            # The last holder left while this work ran: free every slot (and the weights) now.
            try:
                self.close(wait_s=0.0)
            except ModelBusyError:
                pass  # work admitted meanwhile: the next release closes
            return
        self.reap_idle()

    def reap_idle(self, *, now: Optional[float] = None) -> int:
        """Free extra slots idle for longer than `idle_ttl_s` (slot 0 stays). Returns how many."""
        now = time.monotonic() if now is None else now
        freed: List[_Slot] = []
        with self._cv:
            pinned = max(1, int(self.pinned_slots))
            keep = list(self._slots[:pinned])
            for s in self._slots[pinned:]:
                if not s.busy and now - s.last_used >= self.idle_ttl_s:
                    freed.append(s)
                else:
                    keep.append(s)
            if freed:
                self._slots = keep
                for i, s in enumerate(self._slots):
                    s.index = i
        for s in freed:
            self._close_slot(s)
        return len(freed)

    def _close_slot(self, slot: _Slot) -> None:
        llm, slot.llm = slot.llm, None
        slot.shutdown()
        if llm is not None:
            self._close_llm(llm)

    def add_slot(self, llm: Any) -> None:
        """A slot built at load (a pinned `model_concurrency`)."""
        with self._cv:
            self._slots.append(_Slot(len(self._slots), llm))
            self._cv.notify_all()

    def close(self, *, wait_s: float = 10.0) -> None:
        """Free every slot (the last holder detached, or an eject). The caller stopped its own
        in-flight generations first (`unload_model`); slots still finishing get `wait_s` to return.
        Refuses (`ModelBusyError`, nothing freed) while a slot generates or a request waits: a
        llama.cpp context freed under a running decode crashes the process."""
        deadline = time.monotonic() + max(0.0, float(wait_s))
        with self._cv:
            while self._queue or any(s.busy for s in self._slots):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    active = sum(1 for s in self._slots if s.busy)
                    raise ModelBusyError(
                        f"{self.model} cannot be unloaded now: {active} generation(s) are running on it and "
                        f"{len(self._queue)} request(s) wait for it. Stop them first.", status_code=409)
                self._cv.wait(timeout=min(0.25, remaining))
            self.closed = True
            slots, self._slots = self._slots, []
            self._cv.notify_all()
        for s in reversed(slots):
            self._close_slot(s)

    # -- execution (the `_InstanceGenerationWorker` interface) -------------------------------
    def _on_worker(self) -> bool:
        cur = _CURRENT.get()
        return cur is not None and cur[0] is self

    def rebind_current(self, llm: Any) -> None:
        """Replace the context of the slot this thread borrowed (a warm rebuild after a failed prefill)."""
        cur = _CURRENT.get()
        if cur is None or cur[0] is not self:
            raise RuntimeError("rebind_current() outside a slot call")
        cur[1].llm = llm

    def run(self, call: Callable[[], Any], *, cancel_event: Optional[threading.Event] = None,
            on_queued: Optional[Callable[[int, Dict[str, Any]], None]] = None, timeout_s: Optional[float] = None) -> Any:
        if self._on_worker():
            return call()
        slot, _waited = self._acquire(cancel_event, on_queued, timeout_s)
        try:
            ctx = contextvars.copy_context()

            def task() -> Any:
                token = _CURRENT.set((self, slot))
                try:
                    return call()
                finally:
                    _CURRENT.reset(token)

            future = slot.executor().submit(ctx.run, task)
            try:
                return future.result()
            except (KeyboardInterrupt, SystemExit):
                if cancel_event is not None and not future.done():
                    cancel_event.set()
                raise
        finally:
            self._release(slot)

    def stream(self, call: Callable[[], Any], *, cancel_event: Optional[threading.Event] = None,
               on_done: Optional[Callable[[], None]] = None,
               on_queued: Optional[Callable[[int, Dict[str, Any]], None]] = None,
               timeout_s: Optional[float] = None) -> Iterator[Any]:
        """`call()` runs on a borrowed slot's worker; its chunks are produced there into a queue.
        The slot returns when the production ends (`on_done` runs then too)."""
        if self._on_worker():
            if on_done is not None:
                on_done()
            return call()
        import queue as _queue
        from concurrent.futures import Future

        try:
            slot, _waited = self._acquire(cancel_event, on_queued, timeout_s)
        except BaseException:
            if on_done is not None:
                on_done()
            raise
        started: "Future[None]" = Future()
        out: "_queue.Queue[Tuple[str, Any]]" = _queue.Queue()
        stop = threading.Event()

        def task() -> None:
            token = _CURRENT.set((self, slot))
            try:
                try:
                    chunks = call()
                except BaseException as exc:  # noqa: BLE001 - delivered to the caller at the call
                    started.set_exception(exc)
                    return
                started.set_result(None)
                try:
                    from ..core.types import GenerateResponse

                    if isinstance(chunks, GenerateResponse):
                        out.put(("chunk", chunks))
                        return
                    try:
                        for chunk in chunks:
                            out.put(("chunk", chunk))
                            if stop.is_set():
                                break
                    finally:
                        close = getattr(chunks, "close", None)
                        if stop.is_set() and callable(close):
                            close()
                except BaseException as exc:  # noqa: BLE001 - delivered to the consumer in order
                    out.put(("error", exc))
                finally:
                    out.put(("end", None))
            finally:
                _CURRENT.reset(token)
                try:
                    if on_done is not None:
                        on_done()
                finally:
                    self._release(slot)

        ctx = contextvars.copy_context()
        slot.executor().submit(ctx.run, task)
        try:
            started.result()
        except (KeyboardInterrupt, SystemExit):
            stop.set()
            if cancel_event is not None:
                cancel_event.set()
            raise

        def consume() -> Iterator[Any]:
            try:
                while True:
                    kind, value = out.get()
                    if kind == "end":
                        return
                    if kind == "error":
                        raise value
                    yield value
            finally:
                stop.set()

        return consume()


def _default_close(llm: Any) -> None:
    close = getattr(llm, "close", None)
    if callable(close):
        try:
            close()
        except Exception:  # noqa: BLE001
            pass


# -- registry: one scheduler per (weights key, n_ctx) ---------------------------------------
_SCHEDULERS: Dict[str, SlotScheduler] = {}
_REG_LOCK = threading.RLock()


def scheduler_key(weights_key: str, n_ctx: int) -> str:
    return f"{weights_key}|n_ctx={int(n_ctx)}"


def find(weights_key: str, n_ctx: Optional[int] = None) -> Optional[SlotScheduler]:
    """The resident scheduler of these weights (at `n_ctx` when given, else the first one)."""
    with _REG_LOCK:
        if n_ctx is not None:
            s = _SCHEDULERS.get(scheduler_key(weights_key, n_ctx))
            return s if s is not None and not s.closed else None
        for k, s in _SCHEDULERS.items():
            if k.startswith(weights_key + "|n_ctx=") and not s.closed:
                return s
        return None


def find_for_file(model_path: Any, n_ctx: Optional[int] = None) -> Optional[SlotScheduler]:
    """The resident pool of this GGUF FILE, whatever GPU-layer setting loaded it (a holder adopts
    the resident model as it is rather than loading a second copy with other settings)."""
    import os

    try:
        path = os.path.realpath(str(model_path))
    except Exception:  # noqa: BLE001
        path = str(model_path)
    with _REG_LOCK:
        for k, s in _SCHEDULERS.items():
            if s.closed or not k.startswith(path + "|"):
                continue
            if n_ctx is None or s.n_ctx == int(n_ctx):
                return s
        return None


def _close_all_at_exit() -> None:
    """Free every slot before the interpreter (and ggml's Metal/CUDA device) tears down: a
    context alive at that point trips ggml-metal's residency-set assertion."""
    with _REG_LOCK:
        pools = list(_SCHEDULERS.values())
        _SCHEDULERS.clear()
    for s in pools:
        with s._cv:
            s.closed = True
            slots, s._slots = s._slots, []
        for slot in reversed(slots):
            try:
                s._close_slot(slot)
            except Exception:  # noqa: BLE001
                pass


_EXIT_HOOK = False


def register(weights_key: str, scheduler: SlotScheduler) -> SlotScheduler:
    global _EXIT_HOOK
    with _REG_LOCK:
        if not _EXIT_HOOK:
            # Registered with the first pool, never at import: importing the provider has no side effect.
            import atexit

            atexit.register(_close_all_at_exit)
            _EXIT_HOOK = True
        _SCHEDULERS[scheduler_key(weights_key, scheduler.n_ctx)] = scheduler
        return scheduler


def attach(scheduler: SlotScheduler, holder: Any) -> None:
    """`holder` uses the model. A holder dropped without an unload detaches when it is collected
    (the finalizer holds the pool and a token, never the holder)."""
    token = object()
    with _REG_LOCK:
        scheduler._attached.add(id(token))
        scheduler.holders.add(holder)
    fin = weakref.finalize(holder, _detach_token, scheduler, id(token))
    fin.atexit = False
    holder.__dict__["_gguf_attach_token"] = (token, fin)


def _detach_token(scheduler: SlotScheduler, token_id: int) -> None:
    try:
        _detach_id(scheduler, token_id)
    except Exception:  # noqa: BLE001 - a finalizer never raises
        pass


def _detach_id(scheduler: SlotScheduler, token_id: int) -> bool:
    """Remove one attachment. The last one unregisters the pool (under the registry lock) and
    closes it OUTSIDE that lock: never a wait that stalls every other load. A pool still running
    or holding queued work closes itself when it drains (`close_when_idle`), never leaks."""
    with _REG_LOCK:
        if token_id not in scheduler._attached:
            return len(scheduler._attached) > 0
        scheduler._attached.discard(token_id)
        if scheduler._attached:
            return True
        for k, s in list(_SCHEDULERS.items()):
            if s is scheduler:
                _SCHEDULERS.pop(k, None)
        scheduler.close_when_idle = True
    try:
        scheduler.close(wait_s=0.0)
    except ModelBusyError:
        logger_msg = (f"{scheduler.model}: its last holder left while a generation ran or a request waited; "
                      "its slots are freed when that work ends")
        import logging

        logging.getLogger("abstractcore.residency").info(logger_msg)
    return False


def detach(scheduler: SlotScheduler, holder: Any) -> bool:
    """`holder` stops using the model. True when OTHER holders still use it (nothing freed);
    False when it was the last: every slot is closed and the weights go with the last context
    (at once, or when the work still running on the pool ends)."""
    token, fin = holder.__dict__.get("_gguf_attach_token") or (None, None)
    if token is None:
        return len(scheduler._attached) > 0
    still = _detach_id(scheduler, id(token))
    holder.__dict__.pop("_gguf_attach_token", None)
    if fin is not None:
        fin.detach()
    scheduler.holders.discard(holder)
    return still


def all_schedulers() -> List[SlotScheduler]:
    with _REG_LOCK:
        return [s for s in _SCHEDULERS.values() if not s.closed]


def default_concurrency(n_gpu_layers: Any) -> int:
    """llama.cpp on a GPU / Metal: 4 concurrent slots; CPU-only: 1 (threads oversubscribe)."""
    try:
        return CPU_DEFAULT_CONCURRENCY if int(n_gpu_layers) == 0 else GPU_DEFAULT_CONCURRENCY
    except Exception:  # noqa: BLE001 - None = all layers on the accelerator when one exists
        return GPU_DEFAULT_CONCURRENCY
