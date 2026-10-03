"""Request-scoped authentication for hosts composing the Core ASGI server.

Standalone environment configuration stays authoritative outside an explicit
host policy. Hosts must keep the policy scope alive until ASGI streaming ends.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import os
from typing import Iterator, Optional


@dataclass(frozen=True)
class ServerAuthPolicy:
    token: str = field(repr=False)
    allow_unauthenticated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.token, str) or not self.token.strip():
            raise ValueError("Managed Core serving requires a non-empty server token")
        if not isinstance(self.allow_unauthenticated, bool):
            raise TypeError("allow_unauthenticated must be a boolean")
        object.__setattr__(self, "token", self.token.strip())


_policy: ContextVar[Optional[ServerAuthPolicy]] = ContextVar("abstractcore_server_auth_policy", default=None)


def current_server_auth_policy() -> Optional[ServerAuthPolicy]:
    return _policy.get()


@contextmanager
def use_server_auth_policy(policy: ServerAuthPolicy) -> Iterator[None]:
    if not isinstance(policy, ServerAuthPolicy):
        raise TypeError("policy must be a ServerAuthPolicy")
    previous = _policy.set(policy)
    try:
        yield
    finally:
        _policy.reset(previous)


def server_auth_token() -> str:
    policy = current_server_auth_policy()
    if policy is not None:
        return policy.token
    return str(os.getenv("ABSTRACTCORE_AUTH_TOKEN") or "").strip()


def server_allows_unauthenticated() -> bool:
    policy = current_server_auth_policy()
    if policy is not None:
        return policy.allow_unauthenticated
    return str(os.getenv("ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED") or "").strip().lower() in {"1", "true", "yes", "on"}
