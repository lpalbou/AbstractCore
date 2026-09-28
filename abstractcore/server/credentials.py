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
from typing import Any, List, Mapping, Optional

from fastapi import HTTPException, Request

__all__ = [
    "MUSIC_SERVER_KEYS",
    "guard_catalog_credentials",
    "guard_music_credentials",
    "saved_openai_api_key",
    "server_held_music_keys",
    "server_holds_openai_key",
]

# Every provider key AbstractMusic reads (abstractmusic 0.1.15:
# backends/acemusic.py, backends/elevenlabs_music.py and
# integrations/abstractcore_plugin.py): env var -> the owner-config key that
# wins over it. INTERIM, fail closed: whenever the server holds one of these,
# an unauthenticated music request is refused whatever backend it names,
# because core cannot yet ask AbstractMusic which backends are remote.
# Follow-up (root backlog): AbstractMusic's public remote-backend API, the
# music twin of `abstractvoice.engine_runtime.engine_runtime_status(...).remote`,
# then guard only the remote backends. Not listed: HF_TOKEN /
# HUGGINGFACE_HUB_TOKEN / HUGGING_FACE_HUB_TOKEN (backends/stable_audio_3.py),
# the Hugging Face download credential every local engine in the framework
# uses, never a paid provider key.
MUSIC_SERVER_KEYS = {
    "ACEMUSIC_API_KEY": "music_acemusic_api_key",
    "ELEVENLABS_API_KEY": "music_elevenlabs_api_key",
}


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


def server_held_music_keys(config: Mapping[str, Any]) -> List[str]:
    """The env vars in `MUSIC_SERVER_KEYS` the server holds, in the environment
    or through the capability config's owner key (`config`: `_capability_config()`)."""
    return [
        env_var
        for env_var, config_key in MUSIC_SERVER_KEYS.items()
        if str(os.getenv(env_var) or "").strip() or str(config.get(config_key) or "").strip()
    ]


def _request_is_authenticated(request: Request) -> bool:
    return bool(getattr(request.state, "abstractcore_server_authenticated", False))


def _refuse(what: str) -> HTTPException:
    return HTTPException(
        status_code=401,
        detail=(
            f"Server-held {what} credentials are configured, but inbound server auth was not used. "
            "Set ABSTRACTCORE_AUTH_TOKEN and send "
            "Authorization: Bearer <server-token>, or pass an explicit "
            "provider key with X-AbstractCore-Provider-API-Key for this request."
        ),
        headers={"WWW-Authenticate": "Bearer"},
    )


def guard_catalog_credentials(*, request: Request, explicit_provider_key: bool, surface: str) -> None:
    """Refuse (401) an unauthenticated discovery request while the server holds a key.

    `surface` names the routes in the message ("audio", "vision").
    """
    if _request_is_authenticated(request) or explicit_provider_key:
        return
    if not server_holds_openai_key():
        return
    raise _refuse(f"{surface}/OpenAI")


def guard_music_credentials(*, request: Request, explicit_provider_key: bool, config: Mapping[str, Any]) -> None:
    """Refuse (401) an unauthenticated music request while the server holds any
    key AbstractMusic reads (`MUSIC_SERVER_KEYS`). A caller key passes: the
    route hands it to every remote music backend in place of the server's."""
    if _request_is_authenticated(request) or explicit_provider_key:
        return
    held = server_held_music_keys(config)
    if held:
        raise _refuse(f"music ({', '.join(held)})")
