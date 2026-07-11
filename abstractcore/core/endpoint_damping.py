"""Per-endpoint retry damping (entity-topology plan item 12 / C3) — OPT-IN.

The herd mechanism this closes: circuit breakers are per provider INSTANCE
(keyed class:model on each object), so N instances pointing at ONE endpoint
never share failure state — a down endpoint is discovered N times, in
parallel, each discovery with its own retries. With damping enabled, provider
instances that target the same endpoint share ONE domain:

- a SHARED circuit breaker — the first instance to trip it stops the other
  N-1 from burning their own retry ladders;
- a RETRY-WAITER BUDGET (bounded semaphore) — at most K instances may sit in
  backoff for the endpoint at once; a request that cannot get a slot FAILS
  FAST with the last provider error labeled
  `[retry budget exhausted for endpoint ...]` instead of joining the herd.
  Failing fast is deliberate: queueing here would hide pressure the item-11
  admission layer is supposed to see.

Scope honesty (the C3/C4 boundary pin): this reaches only provider instances
inside ONE process. Cross-process fairness (N entity runtimes + a gateway all
dialing one endpoint) is the admission controller's lane (plan item 11) —
admission decides WHO may call; damping decides how a call that failed BEHAVES.

Keying: `(base_url, model)` — the ENDPOINT's identity, deliberately derived
from the model server's address (which is what herds hammer), never from any
application-level handle/address (the C1 relocation-stable-keys pin governs
application keys; a model server's URL is the damping subject itself).

Opt-in: `create_llm(..., endpoint_damping=True)`. Default off — single-instance
library users keep exactly today's behavior.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

from .retry import CircuitBreaker, RetryConfig
from ..utils.structured_logging import get_logger

logger = get_logger(__name__)

# At most K provider instances may wait in retry-backoff per endpoint at once.
# Deliberately small: the point of the budget is that a broken endpoint gets a
# few probes, not a crowd. Declared tunable (constructor param), not a cap
# derived from fear — single callers are unaffected (they need exactly 1 slot).
DEFAULT_MAX_RETRY_WAITERS = 2


class EndpointDampingDomain:
    """Shared failure state + retry budget for ONE endpoint (base_url, model)."""

    def __init__(self, key: Tuple[str, str], config: RetryConfig,
                 max_retry_waiters: int = DEFAULT_MAX_RETRY_WAITERS):
        self.key = key
        # One breaker for every instance on this endpoint (vs. per-instance today).
        self.breaker = CircuitBreaker(config)
        self._max_retry_waiters = max(1, int(max_retry_waiters))
        self._retry_slots = threading.BoundedSemaphore(self._max_retry_waiters)

    @property
    def label(self) -> str:
        return f"{self.key[0]}::{self.key[1]}"

    def try_acquire_retry_slot(self) -> bool:
        """Non-blocking: True if this caller may enter a retry-backoff wait."""
        return self._retry_slots.acquire(blocking=False)

    def release_retry_slot(self) -> None:
        try:
            self._retry_slots.release()
        except ValueError:
            # Over-release guard (BoundedSemaphore raises) — never propagate
            # bookkeeping errors into a caller's error path.
            logger.warning("endpoint damping: retry slot over-release for %s", self.label)


class EndpointDampingRegistry:
    """Process-wide registry of damping domains, keyed (base_url, model)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._domains: Dict[Tuple[str, str], EndpointDampingDomain] = {}

    def domain_for(self, base_url: str, model: str, config: RetryConfig,
                   max_retry_waiters: Optional[int] = None) -> EndpointDampingDomain:
        key = (str(base_url or "").strip().rstrip("/"), str(model or "").strip())
        with self._lock:
            domain = self._domains.get(key)
            if domain is None:
                domain = EndpointDampingDomain(
                    key, config,
                    max_retry_waiters=max_retry_waiters or DEFAULT_MAX_RETRY_WAITERS,
                )
                self._domains[key] = domain
            return domain

    def reset(self) -> None:
        """Test hook: drop all domains (never used on production paths)."""
        with self._lock:
            self._domains.clear()


_global_registry = EndpointDampingRegistry()


def get_endpoint_damping_registry() -> EndpointDampingRegistry:
    return _global_registry
