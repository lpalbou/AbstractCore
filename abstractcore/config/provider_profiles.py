"""Persisted provider endpoint profiles for AbstractCore.

Profiles make a reusable virtual provider id such as ``endpoint:ovh-provider``
resolve to a concrete provider family, endpoint URL, and optional API key.
They are intentionally single-principal/local in Core; hosted/user scoping stays
in AbstractGateway.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field, fields as dataclass_fields
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

VIRTUAL_PROVIDER_PREFIX = "endpoint:"

_PROFILE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SUPPORTED_PROVIDER_FAMILIES = {
    "anthropic",
    "lmstudio",
    "ollama",
    "openai",
    "openai-compatible",
    "openrouter",
    "portkey",
    "vllm",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_profile_id(value: str) -> str:
    raw = str(value or "").strip()
    if raw.lower().startswith(VIRTUAL_PROVIDER_PREFIX):
        raw = raw[len(VIRTUAL_PROVIDER_PREFIX) :]
    if not raw:
        raise ValueError("Provider profile id cannot be empty")
    if not _PROFILE_ID_RE.match(raw):
        raise ValueError(
            "Provider profile id must start with a letter or number and contain only letters, "
            "numbers, '.', '_', or '-'"
        )
    return raw


def virtual_provider_id(profile_id: str) -> str:
    return f"{VIRTUAL_PROVIDER_PREFIX}{normalize_profile_id(profile_id)}"


def profile_id_from_virtual_provider(provider: str) -> Optional[str]:
    raw = str(provider or "").strip()
    if not raw.lower().startswith(VIRTUAL_PROVIDER_PREFIX):
        return None
    return normalize_profile_id(raw[len(VIRTUAL_PROVIDER_PREFIX) :])


def normalize_provider_family(value: Optional[str]) -> str:
    raw = str(value or "openai-compatible").strip().lower().replace("_", "-")
    if raw not in SUPPORTED_PROVIDER_FAMILIES:
        supported = ", ".join(sorted(SUPPORTED_PROVIDER_FAMILIES))
        raise ValueError(f"Unsupported provider family {raw!r}. Supported: {supported}")
    return raw


def normalize_base_url(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if not (raw.startswith("http://") or raw.startswith("https://")):
        raise ValueError("Provider profile base URL must start with http:// or https://")
    return raw.rstrip("/")


def normalize_api_key(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if raw.upper() == "EMPTY":
        return "EMPTY"
    return raw


def normalize_api_key_env_var(value: Optional[str]) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if not _ENV_VAR_RE.match(raw):
        raise ValueError("API key env var must be a valid environment variable name")
    return raw


def split_api_key_value(value: Optional[str]) -> tuple[str, str]:
    """Split one CLI/API key value into raw-key or env-reference storage."""
    raw = normalize_api_key(value)
    if not raw:
        return "", ""
    if raw.startswith("${") and raw.endswith("}"):
        return "", normalize_api_key_env_var(raw[2:-1])
    if raw.startswith("$"):
        return "", normalize_api_key_env_var(raw[1:])
    return raw, ""


def normalize_scope(value: Optional[str]) -> str:
    """`gateway` (shared) or `user` (private to one runtime). Default: gateway.

    A hosting column, kept on the Core row so a hosted profile does not need a
    second store to say who may see it. Core reads its own profiles regardless
    of scope; only a Gateway acts on the distinction.
    """
    raw = str(value or "gateway").strip().lower()
    if raw not in {"gateway", "user"}:
        raise ValueError("Provider profile scope must be 'gateway' or 'user'")
    return raw


def normalize_string_list(values: Optional[Iterable[Any]]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        item = str(value or "").strip()
        if not item or item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def api_key_fingerprint(api_key: Optional[str]) -> Optional[str]:
    key = normalize_api_key(api_key)
    if not key:
        return None
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]


@dataclass
class ProviderProfile:
    """A Core provider profile exposed as ``endpoint:<id>``.

    ONE STORE FOR PROVIDER CONFIG (operator ruling 2026-08-01). A profile is
    provider configuration, so Core holds it whichever console created it --
    Core's own, or an AbstractGateway console writing through its Core-config
    seam. `scope` and `capabilities` carry the two columns a hosted Gateway
    needs on the same row (whether a profile is shared with its users, and
    which capabilities it serves) so that hosting a profile never requires a
    second copy of it somewhere else. Core itself reads neither.
    """

    id: str
    display_name: str = ""
    description: str = ""
    provider_family: str = "openai-compatible"
    base_url: str = ""
    api_key: str = ""
    api_key_env_var: str = ""
    allowed_models: list[str] = field(default_factory=list)
    enabled: bool = True
    scope: str = "gateway"
    capabilities: list[str] = field(default_factory=lambda: ["text"])
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        self.id = normalize_profile_id(self.id)
        self.display_name = str(self.display_name or self.id).strip() or self.id
        self.description = str(self.description or "").strip()
        self.provider_family = normalize_provider_family(self.provider_family)
        self.base_url = normalize_base_url(self.base_url)
        self.api_key = normalize_api_key(self.api_key)
        self.api_key_env_var = normalize_api_key_env_var(self.api_key_env_var)
        self.allowed_models = normalize_string_list(self.allowed_models)
        self.enabled = bool(self.enabled)
        self.scope = normalize_scope(self.scope)
        self.capabilities = normalize_string_list(self.capabilities) or ["text"]
        now = utc_now_iso()
        self.created_at = str(self.created_at or now)
        self.updated_at = str(self.updated_at or now)

    @property
    def virtual_provider_id(self) -> str:
        return virtual_provider_id(self.id)

    def resolved_api_key(self) -> str:
        if self.api_key_env_var:
            env_value = os.environ.get(self.api_key_env_var)
            if isinstance(env_value, str) and env_value.strip():
                return normalize_api_key(env_value)
        return self.api_key

    def public_dict(self) -> Dict[str, Any]:
        resolved_key = self.resolved_api_key()
        out: Dict[str, Any] = {
            "id": self.id,
            "virtual_provider": self.virtual_provider_id,
            "display_name": self.display_name,
            "description": self.description,
            "provider_family": self.provider_family,
            "base_url": self.base_url,
            "allowed_models": list(self.allowed_models),
            "enabled": bool(self.enabled),
            "scope": self.scope,
            "capabilities": list(self.capabilities),
            "api_key_set": bool(resolved_key),
            "api_key_env_var": self.api_key_env_var,
            "api_key_fingerprint": api_key_fingerprint(resolved_key),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        return out

    def private_resolution(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "id": self.id,
            "virtual_provider": self.virtual_provider_id,
            "provider": self.provider_family,
            "provider_family": self.provider_family,
            "base_url": self.base_url,
            "api_key": self.resolved_api_key(),
            "allowed_models": list(self.allowed_models),
            "enabled": bool(self.enabled),
        }
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "display_name": self.display_name,
            "description": self.description,
            "provider_family": self.provider_family,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "api_key_env_var": self.api_key_env_var,
            "allowed_models": list(self.allowed_models),
            "enabled": bool(self.enabled),
            "scope": self.scope,
            "capabilities": list(self.capabilities),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class ProviderProfilesConfig:
    profiles: Dict[str, ProviderProfile] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"profiles": {pid: profile.to_dict() for pid, profile in sorted(self.profiles.items())}}


def provider_profile_from_dict(profile_id: str, data: Any) -> ProviderProfile:
    if not isinstance(data, dict):
        raise ValueError("Provider profile row must be an object")
    payload = dict(data)
    payload.setdefault("id", profile_id)
    # Unknown columns are DROPPED, not fatal: a row written by a newer writer
    # must not make the whole config store unloadable (which reads, downstream,
    # as a corrupt store and a fallback to defaults).
    allowed = {f.name for f in dataclass_fields(ProviderProfile)}
    return ProviderProfile(**{key: value for key, value in payload.items() if key in allowed})


def provider_profiles_from_dict(data: Any) -> ProviderProfilesConfig:
    if not isinstance(data, dict):
        return ProviderProfilesConfig()

    rows = data.get("profiles", data)
    if not isinstance(rows, dict):
        return ProviderProfilesConfig()

    profiles: Dict[str, ProviderProfile] = {}
    for raw_id, raw_profile in rows.items():
        profile = provider_profile_from_dict(str(raw_id), raw_profile)
        profiles[profile.id.lower()] = profile
    return ProviderProfilesConfig(profiles=profiles)
