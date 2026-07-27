"""Host-injected endpoint-profile resolution context.

Why this exists (live incident, 2026-07-26): ``endpoint:<id>`` provider
profiles can be registered on a HOST (e.g. an AbstractGateway per-principal
store) rather than in Core's local ``~/.abstractcore`` config. The run's own
LLM calls resolve those profiles through a resolver the host attaches to
provider INSTANCES (``resolve_provider_endpoint_profile`` — see
``BaseProvider``'s video-route fallback), but a NESTED ``create_llm(...)``
during tool execution (e.g. ``analyze_media``'s session route) has no
instance to inherit that resolver from, so the registry raised
``Unknown provider: endpoint:<id>`` for a profile the host could resolve.

This module is the process-safe channel for that case: the host enters
``use_provider_endpoint_profile_resolver(resolver)`` around tool execution,
and ``create_llm``/the provider registry consult the ambient resolver ONLY
when local config resolution misses an ``endpoint:*``-shaped spec. Local
config always wins — operator-profile routes behave exactly as before.

Security properties (load-bearing):
- The resolver travels through a ``contextvars.ContextVar`` set by HOST code
  only. It never rides tool arguments, prompts, or any model-writable
  surface, so a model cannot inject or redirect endpoint resolution.
- Resolver payloads may carry secrets (``api_key``). Consumers must never
  log or embed secret values in errors; the registry scrubs payload-derived
  messages defensively (see ``registry._endpoint_profile_from_context_resolver``).

Contract for resolvers:
- Signature ``Callable[[str], Optional[dict]]`` — called with the full spec
  (``"endpoint:<id>"``); returns a profile dict (gateway
  ``private_resolution()`` shape: ``provider_family``/``provider``,
  ``base_url``, ``api_key``, ``allowed_models``, ``enabled``, ...) or ``None``
  for a clean miss. Malformed payloads are treated as a labeled miss, never
  a crash.
- Resolvers should be cheap and idempotent: one provider construction may
  consult the resolver more than once (registry info + create paths).
- ``ContextVar`` scope is per-thread / per-async-task: the host must enter
  the context in the SAME thread (or copied context) that executes the tool.
  Entering with ``None`` deliberately MASKS any outer resolver for the block
  (a host can fence sub-work it does not trust with ambient resolution).
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Dict, Iterator, Optional

__all__ = [
    "ProviderEndpointProfileResolver",
    "current_provider_endpoint_profile_resolver",
    "use_provider_endpoint_profile_resolver",
]

ProviderEndpointProfileResolver = Callable[[str], Optional[Dict[str, Any]]]

_RESOLVER_VAR: ContextVar[Optional[ProviderEndpointProfileResolver]] = ContextVar(
    "abstractcore_provider_endpoint_profile_resolver", default=None
)


def current_provider_endpoint_profile_resolver() -> Optional[ProviderEndpointProfileResolver]:
    """Return the resolver installed for the current context, or None."""
    return _RESOLVER_VAR.get()


@contextmanager
def use_provider_endpoint_profile_resolver(
    resolver: Optional[ProviderEndpointProfileResolver],
) -> Iterator[None]:
    """Install ``resolver`` as the ambient endpoint-profile resolver.

    Nesting: the innermost context wins; exiting restores the previous value
    (token-based reset, exception-safe). Passing ``None`` masks any outer
    resolver for the duration of the block.
    """
    token = _RESOLVER_VAR.set(resolver)
    try:
        yield
    finally:
        _RESOLVER_VAR.reset(token)
