"""Server-held provider credentials: one rule for every discovery route.

A request that is not server-authenticated never spends a key the SERVER
holds: the OpenAI key saved in AbstractCore's config (Providers /
`abstractcore --set-api-key openai`) or `OPENAI_API_KEY`. This holds also
with `ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED=1`, which admits anonymous
callers to the server but never lends them its keys. A caller that brings
its own key (`X-AbstractCore-Provider-API-Key`) spends only that key.

The audio and vision catalog routes share `guard_catalog_credentials`; the
generation routes apply the same rule per provider
(`audio_endpoints._guard_unauthenticated_server_provider_key_use`,
`app._guard_unauthenticated_server_provider_key_use`).
"""

from __future__ import annotations

import os
from typing import Optional

from fastapi import HTTPException, Request

__all__ = ["guard_catalog_credentials", "saved_openai_api_key", "server_holds_openai_key"]


def saved_openai_api_key() -> Optional[str]:
    """`api_keys.openai` from the centralized config, or None.

    An unreadable config degrades like the other global defaults (no key).
    """
    try:
        from ..config.manager import get_config_manager

        value = get_config_manager().config.api_keys.openai
    except Exception:
        return None
    return str(value or "").strip() or None


def server_holds_openai_key() -> bool:
    return bool(str(os.getenv("OPENAI_API_KEY") or "").strip()) or bool(saved_openai_api_key())


def guard_catalog_credentials(*, request: Request, explicit_provider_key: bool, surface: str) -> None:
    """Refuse (401) an unauthenticated discovery request while the server holds a key.

    `surface` names the routes in the message ("audio", "vision").
    """
    if bool(getattr(request.state, "abstractcore_server_authenticated", False)) or explicit_provider_key:
        return
    if not server_holds_openai_key():
        return
    raise HTTPException(
        status_code=401,
        detail=(
            f"Server-held {surface}/OpenAI credentials are configured, but inbound server auth was not used. "
            "Set ABSTRACTCORE_AUTH_TOKEN and send "
            "Authorization: Bearer <server-token>, or pass an explicit "
            "provider key with X-AbstractCore-Provider-API-Key for this request."
        ),
        headers={"WWW-Authenticate": "Bearer"},
    )
