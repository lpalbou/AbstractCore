"""THE ONE ABSTRACTION for "are these weights here, and how do I fetch them".

TWO ENTRY POINTS, ONE ANSWER. A capability default names a provider and a model
(`lmstudio/qwen/qwen3.5-9b`), and both entry points -- the AbstractCore CLI /
console-TUI and the AbstractGateway console -- have to answer the same two
questions about it:

    probe(provider, model)              -> installed | absent | unknown | not_applicable
    download(provider, artifact, cb)    -> the provider's own tool, run once

Everything else in this module is one provider's implementation of those two
verbs. Surfaces render the result; they never re-derive it, and they never
shell out to a provider tool themselves.

FOUR RULES, and every one of them exists because breaking it produced a lie:

1. NEVER AUTO-DOWNLOAD. `probe` reads local state only: an HTTP GET against a
   server that is already running, a directory listing, a CLI that lists what
   is on disk. It never contacts a model hub, never warms a cache, and never
   costs the operator a byte. A download happens when a human asks for it, at
   `download()`, and nowhere else.

2. `unknown` IS A LEGAL ANSWER. When LM Studio is not running and its CLI is
   not installed, the honest answer is "I cannot tell", not "absent" (which
   would invite a pointless re-download of weights already on disk) and not
   "installed" (which would let a run fail at the first token). Every probe
   path that loses its evidence returns `unknown` with the reason attached.

3. SERVED IDS ARE NOT DOWNLOAD REFS. LM Studio serves `qwen/qwen3.5-9b` when a
   single quantization is installed, but the thing you FETCH is
   `qwen/qwen3.5-9b@4bit`. The capability route stores the served id; the
   artifact (`RECOMMENDED_MODEL_DOWNLOADS`, or whatever the operator types)
   names the exact weights. `split_artifact` is the one place that knows the
   `@quant` convention, and the presence matcher is deliberately TOLERANT in
   that one direction: a bare installed id satisfies a quantized artifact
   reference, because that is exactly what LM Studio reports for it.

4. THE PROVIDER'S ERROR IS THE ERROR. A failed download reports the tool's own
   stderr verbatim, plus exactly one line the operator can act on. Paraphrasing
   `ollama pull` into "download failed" throws away the only useful half.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import os
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, fields as _dc_fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple

from .capability_defaults import (
    capability_route_key,
    recommended_model_downloads,
)

# Reuse, do not reinvent: `abstractcore.download` already defines the progress
# vocabulary the async download API speaks. The materializer is the SYNC lane
# (a CLI streaming lines, a Gateway background job posting updates), and it
# speaks the same words so a surface that renders one renders the other.
from ..download import DownloadProgress, DownloadStatus

_LOG = logging.getLogger("abstractcore.model_materializer")

__all__ = [
    "PRESENCE_INSTALLED",
    "PRESENCE_ABSENT",
    "PRESENCE_UNKNOWN",
    "PRESENCE_NOT_APPLICABLE",
    "PRESENCE_STATES",
    "ModelPresence",
    "DownloadOutcome",
    "ProgressCallback",
    "split_artifact",
    "probe",
    "download",
    "presence_sweep",
    "supported_providers",
    "recommended_downloads",
    "recommended_plan",
    "annotate_route_availability",
    "hf_artifact_parts",
    "list_installed",
    "delete_artifact",
    "delete_blockers",
    "MODELS_INSTALLED_SCHEMA",
    "INSTALLED_PROVIDERS",
]


PRESENCE_INSTALLED = "installed"
PRESENCE_ABSENT = "absent"
PRESENCE_UNKNOWN = "unknown"
PRESENCE_NOT_APPLICABLE = "not_applicable"
PRESENCE_STATES: Tuple[str, ...] = (
    PRESENCE_INSTALLED,
    PRESENCE_ABSENT,
    PRESENCE_UNKNOWN,
    PRESENCE_NOT_APPLICABLE,
)

ProgressCallback = Callable[[DownloadProgress], None]

# Providers that serve models they do not store locally. There is nothing to
# download and nothing to probe -- saying `absent` about `openai/gpt-4o` would
# be a bug report, not a status. `endpoint:<profile>` is the same case: the
# weights live on whatever host the profile points at.
_RELAY_PROVIDERS = frozenset(
    {
        "openai",
        "anthropic",
        "openrouter",
        "portkey",
        "deepseek",
        "mistral",
        "groq",
        "together",
        "openai-compatible",
        "vllm",
    }
)

_HTTP_TIMEOUT = 3.0
_CLI_PROBE_TIMEOUT = 20.0

# ONE PROVIDER LISTING PER SWEEP.
#
# `lms ls --json` is a subprocess that walks a model library. On a large one --
# or a wedged CLI -- it costs the full `_CLI_PROBE_TIMEOUT`. One availability
# payload probes the whole grid AND the recommended set, so it ran `lms ls`
# three times: a hung CLI turned a single console refresh into SIXTY SECONDS,
# once per probe that could have shared one answer.
#
# The fix is a SWEEP, not a cache with a lifetime. Inside `presence_sweep()`
# each provider listing is read once and reused; outside one, `probe()` reads
# fresh every time, exactly as it always did. That distinction is the whole
# design:
#
#   - NO STALENESS. There is no TTL to outlive the truth. A sweep lasts as long
#     as one payload is being built, and nothing downloads during it.
#   - ONE PAYLOAD IS ONE SNAPSHOT. The "2 of 3 present" banner and the per-row
#     Weights column are computed from the SAME listing, so a download landing
#     mid-payload can no longer make them contradict each other.
#   - A LONE `probe()` IS UNCHANGED. No hidden state to leak between callers,
#     between requests, or between tests.
_sweep: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = contextvars.ContextVar(
    "abstractcore_model_presence_sweep", default=None
)


@contextlib.contextmanager
def presence_sweep() -> "Iterator[None]":
    """Read each provider's model listing at most once inside this block.

    Nestable: an outer sweep spans the inner ones, so a caller that builds a
    grid AND a recommended plan gets one consistent snapshot from one listing.
    """

    if _sweep.get() is not None:
        yield  # an outer sweep already owns the batching
        return
    token = _sweep.set({})
    try:
        yield
    finally:
        _sweep.reset(token)


def _cached_listing(key: str, produce: Callable[[], Any]) -> Any:
    scope = _sweep.get()
    if scope is None:
        return produce()
    if key not in scope:
        scope[key] = produce()
    return scope[key]


@dataclass
class ModelPresence:
    """Whether one provider/model's weights are on this machine."""

    provider: str
    artifact: str
    status: str
    #: How the answer was reached (`lms ls`, `GET /api/tags`, HF cache scan...).
    evidence: str = ""
    #: Free-form, human. Never a secret, never a stack trace.
    detail: str = ""
    #: Where the weights are, when `installed` and the path is known.
    location: Optional[str] = None
    #: Exactly one line the operator can act on, when there is one.
    instruction: Optional[str] = None
    #: True when this provider has a working download verb here.
    downloadable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "provider": self.provider,
            "artifact": self.artifact,
            "status": self.status,
            "downloadable": bool(self.downloadable),
        }
        for name in ("evidence", "detail", "location", "instruction"):
            value = getattr(self, name)
            if value:
                out[name] = value
        return out


@dataclass
class DownloadOutcome:
    """The result of running one provider's download tool, once."""

    provider: str
    artifact: str
    ok: bool
    status: str  # completed | already_installed | failed | not_applicable | planned
    message: str = ""
    #: The provider tool's own output, verbatim -- errors included.
    output: str = ""
    #: The command that ran (or would run, for a dry run). Argv, never a shell string.
    command: List[str] = field(default_factory=list)
    instruction: Optional[str] = None
    location: Optional[str] = None
    #: The model's COMPANION artifacts fetched in the same job (an MLX model's
    #: MTP drafter): `[{artifact, role, status, ok, location?, message?}]`.
    companions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "provider": self.provider,
            "artifact": self.artifact,
            "ok": bool(self.ok),
            "status": self.status,
        }
        for name in ("message", "output", "instruction", "location"):
            value = getattr(self, name)
            if value:
                out[name] = value
        if self.command:
            out["command"] = list(self.command)
        if self.companions:
            out["companions"] = [dict(c) for c in self.companions]
        return out


# ---------------------------------------------------------------------------
# The `@quant` convention -- the one place that knows it
# ---------------------------------------------------------------------------


def split_artifact(artifact: Any) -> Tuple[str, Optional[str]]:
    """`"qwen/qwen3.5-9b@4bit"` -> `("qwen/qwen3.5-9b", "4bit")`.

    The suffix after the LAST `@` is the quantization when it looks like one.
    A model id that merely contains `@` in a path segment keeps it: only a
    trailing token of `[0-9]*(bit|bits)`, `q4_k_m`-shaped, `f16`/`bf16`, or
    `int4`/`fp8` reads as a quant.
    """

    raw = str(artifact or "").strip()
    if "@" not in raw:
        return raw, None
    base, _, tail = raw.rpartition("@")
    quant = tail.strip()
    if not base.strip() or not quant:
        return raw, None
    if _QUANT_RE.match(quant):
        return base.strip(), quant
    return raw, None


_QUANT_RE = re.compile(
    r"^(?:\d+(?:\.\d+)?bits?|q\d[\w_]*|iq\d[\w_]*|f\d+|bf\d+|int\d+|fp\d+|mlx|gguf)$",
    re.IGNORECASE,
)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _matches_installed_id(installed: Any, artifact: str) -> bool:
    """Does an installed/served model id satisfy this artifact reference?

    TOLERANT IN EXACTLY ONE DIRECTION (rule 3). `qwen/qwen3.5-9b` satisfies
    `qwen/qwen3.5-9b@4bit`, because a single-quant install is what LM Studio
    reports under the bare id. The reverse is NOT true in general, but an
    installed `...@4bit` obviously satisfies a bare request for the same base,
    so both `@`-carrying forms are compared on their bases.
    """

    got = _norm(installed)
    want = _norm(artifact)
    if not got or not want:
        return False
    if got == want:
        return True
    want_base, _ = split_artifact(want)
    got_base, _ = split_artifact(got)
    return bool(want_base) and got_base == want_base


# ---------------------------------------------------------------------------
# Provider classification
# ---------------------------------------------------------------------------


def _provider_id(provider: Any) -> str:
    return _norm(provider).replace("_", "-")


def _is_relay(provider: str) -> bool:
    """Normalizes first, so `openai_compatible` and `OpenAI-Compatible` are one name."""

    pid = _provider_id(provider)
    return pid.startswith("endpoint:") or pid in _RELAY_PROVIDERS


def supported_providers() -> Dict[str, Dict[str, Any]]:
    """The provider matrix, as data: who can be probed, who can be fetched."""

    return {
        "lmstudio": {"probe": True, "download": True, "tool": "lms"},
        "ollama": {"probe": True, "download": True, "tool": "ollama"},
        "supertonic": {"probe": True, "download": True, "tool": "abstractvoice"},
        "mlx-gen": {"probe": True, "download": True, "tool": "huggingface_hub"},
        "mlx": {"probe": True, "download": True, "tool": "huggingface_hub"},
        "huggingface": {"probe": True, "download": True, "tool": "huggingface_hub"},
        "mlx-vlm": {"probe": True, "download": True, "tool": "huggingface_hub"},
        "diffusers": {"probe": True, "download": True, "tool": "huggingface_hub"},
    }


_HF_BACKED = frozenset({"mlx-gen", "mlx", "huggingface", "mlx-vlm", "diffusers", "mflux", "transformers"})


# ---------------------------------------------------------------------------
# THE PROVIDER INVENTORY: every provider, and what actually matters about it
# ---------------------------------------------------------------------------
#
# WHY THIS EXISTS. Both console-TUIs listed "providers" by enumerating the
# `api_keys` config section, so a provider that takes no key -- ollama,
# lmstudio, mlx, huggingface, every media engine -- did not exist on the
# Providers screen at all ("how come we don't have ollama, lmstudio,
# huggingface and mlx?", 2026-08-01). The api_keys section is a KEY STORE, not
# a provider list; the provider list is the registry.
#
# WHAT MATTERS DIFFERS PER PROVIDER, so the row says the ONE thing that decides
# whether that provider can run:
#   cloud API      -> is a key present, and where did it come from
#   local server   -> which base URL, and is anything answering there
#   local engine   -> nothing to configure; the weights are the whole question
# A single "status" column that pretended these were the same question is what
# made the old screen useless even for the providers it did list.

# Providers reached over HTTP at an operator-configurable address. The env var
# is the one the provider itself reads (see each provider module's
# BASE_URL_ENV_VAR); the default is what it falls back to.
_LOCAL_SERVER_ENDPOINTS: Dict[str, Tuple[str, str]] = {
    "lmstudio": ("LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
    "ollama": ("OLLAMA_BASE_URL", "http://localhost:11434"),
    "vllm": ("VLLM_BASE_URL", ""),
    "openai-compatible": ("OPENAI_BASE_URL", ""),
}

# `api_keys` config field -> the env var AbstractCore injects it into
# (manager.py `_apply_api_keys_to_env`). Provider id -> field is the mapping a
# row needs to answer "is there a key for THIS provider".
_PROVIDER_KEY_FIELD: Dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "openrouter": "openrouter",
    "portkey": "portkey",
    "openai-compatible": "openai_compatible",
    "vllm": "vllm",
}
_KEY_FIELD_ENV_VAR: Dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "portkey": "PORTKEY_API_KEY",
    "openai_compatible": "OPENAI_API_KEY",
    "vllm": "VLLM_API_KEY",
    "google": "GOOGLE_API_KEY",
}

# Providers whose key is OPTIONAL: they work unauthenticated and the key only
# unlocks something extra (gated HF repos, a secured self-hosted endpoint).
_OPTIONAL_KEY_PROVIDERS = frozenset({"vllm", "openai-compatible", "portkey", "huggingface"})

_PROVIDER_NOTES: Dict[str, str] = {
    "openai-compatible": "shares OPENAI_API_KEY with openai (openai wins when both are set)",
    "huggingface": "HF_TOKEN only for gated/private repos; public weights need no key",
    "mlx": "Apple Silicon text/vision inference",
    "mlx-gen": "image generation",
    "mlx-vlm": "vision",
    "mflux": "image generation",
    "diffusers": "image generation",
    "supertonic": "speech, ships with abstractvoice",
    "vllm": "base URL is required; no default is assumed",
    "openai": "",
}

# Engines with no server and no key: the only question is whether the weights
# are on disk, which the Weights column on the routes screen already answers.
_LOCAL_ENGINE_HINT = "local engine — nothing to configure"


def _resolved_api_key_state(field_name: str, stored: Optional[str]) -> Tuple[bool, str, str, str]:
    """`(is_set, source, env_var, fingerprint)` for one `api_keys` field.

    The RESOLVED answer, in AbstractCore's own precedence: a key stored in the
    config supersedes the environment (manager `_apply_api_keys_to_env`), and a
    bare env var still counts as configured because the provider will read it.
    The fingerprint is the same non-reversible 8 chars every other surface
    prints -- never key material.
    """
    from .provider_profiles import api_key_fingerprint

    env_var = _KEY_FIELD_ENV_VAR.get(field_name, "")
    if isinstance(stored, str) and stored.strip():
        return True, "config", env_var, str(api_key_fingerprint(stored) or "")
    env_value = str(os.environ.get(env_var) or "").strip() if env_var else ""
    if env_value:
        return True, f"env:{env_var}", env_var, str(api_key_fingerprint(env_value) or "")
    return False, "", env_var, ""


def _probe_local_server(provider: str, base_url: str) -> Tuple[Optional[bool], str]:
    """`(reachable, detail)` for a local server, using ONE cheap GET.

    Rule 1 of this module applies: this reads what is already running and never
    starts, warms or downloads anything. `None` means "no address to try", which
    is a different answer from "unreachable" and must not be shown as one.
    """
    if not base_url:
        return None, "no base URL configured"
    if provider == "ollama":
        payload, error = _http_json(f"{base_url.rstrip('/')}/api/tags")
        if payload is None:
            return False, error
        models = payload.get("models") if isinstance(payload, dict) else None
        return True, f"reachable ({len(models or [])} models)"
    url = base_url.rstrip("/")
    if not url.endswith("/models"):
        url = f"{url}/models"
    payload, error = _http_json(url)
    if payload is None:
        return False, error
    data = payload.get("data") if isinstance(payload, dict) else None
    return True, f"reachable ({len(data or [])} models)"


def provider_inventory(manager: Any = None, *, probe: bool = False) -> List[Dict[str, Any]]:
    """EVERY provider AbstractCore knows, one row each, with its real state.

    The LLM providers come from the provider registry -- the same list the
    "Unknown provider: x. Available providers: ..." error prints, so a surface
    can never offer a provider Core would refuse -- plus the endpoint profiles
    the registry itself appends. The media/engine backends come from
    `supported_providers()`, the table that already answers "can this be probed
    and fetched"; there is no third list.

    `probe=True` adds ONE cheap GET per local server. Off by default so a plain
    listing never blocks on a server that is not running.
    """

    try:
        from ..providers.registry import get_provider_registry

        registry = get_provider_registry()
        infos = {str(name): registry.get_provider_info(name) for name in registry.list_provider_names()}
    except Exception:
        registry = None
        infos = {}

    api_keys: Dict[str, Any] = {}
    profile_ids: Dict[str, Dict[str, Any]] = {}
    if manager is not None:
        try:
            api_keys = {f.name: getattr(manager.config.api_keys, f.name, None) for f in _dc_fields(manager.config.api_keys)}
        except Exception:
            api_keys = {}
        try:
            profile_ids = {
                str(p.get("virtual_provider") or ""): dict(p)
                for p in manager.list_provider_profiles(include_disabled=True)
            }
        except Exception:
            profile_ids = {}

    # Endpoint profiles come from THE MANAGER WE WERE GIVEN, not from the
    # registry's own lookup: the registry resolves profiles through the global
    # config manager, so a listing pointed at another store (`--config-file`,
    # a Gateway's per-principal overlay) would silently show the wrong
    # machine's endpoints -- or none at all.
    media_only = [name for name in supported_providers() if name not in infos]
    profile_only = [name for name in profile_ids if name and name not in infos]
    rows: List[Dict[str, Any]] = []
    for name in list(infos) + sorted(profile_only) + sorted(media_only):
        info = infos.get(name)
        profile = profile_ids.get(name)
        key_field = _PROVIDER_KEY_FIELD.get(name, "")
        key_set, key_source, key_env, key_fp = (
            _resolved_api_key_state(key_field, api_keys.get(key_field))
            if key_field
            else (False, "", "", "")
        )
        if name == "huggingface":
            # HF is keyless for public weights and key-taking for gated repos;
            # HF_TOKEN is the name the hub client itself reads.
            key_env = "HF_TOKEN"
            hf_token = str(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
            key_set = bool(hf_token)
            key_source = f"env:{key_env}" if key_set else ""
            if key_set:
                from .provider_profiles import api_key_fingerprint

                key_fp = str(api_key_fingerprint(hf_token) or "")
        if profile is not None:
            # An endpoint profile carries its OWN key (or a `$VAR` reference to
            # one); the shared `api_keys` section says nothing about it.
            key_env = str(profile.get("api_key_env_var") or "")
            key_set = bool(profile.get("api_key_set"))
            key_source = (f"env:{key_env}" if key_env else "profile") if key_set else ""
            key_fp = str(profile.get("api_key_fingerprint") or "")

        endpoint = _LOCAL_SERVER_ENDPOINTS.get(name)
        base_url = ""
        base_url_source = ""
        if profile is not None:
            base_url = str(profile.get("base_url") or "")
            base_url_source = "endpoint profile"
        elif endpoint is not None:
            env_var, default_url = endpoint
            env_value = str(os.environ.get(env_var) or "").strip()
            base_url = env_value or default_url
            base_url_source = f"env:{env_var}" if env_value else ("default" if default_url else "")

        if profile is not None:
            kind = "endpoint_profile"
        elif endpoint is not None:
            kind = "local_server"
        elif info is not None and not getattr(info, "local_provider", False):
            kind = "cloud_api"
        else:
            kind = "local_engine"

        if profile is not None:
            auth = "optional"
        elif key_field or name == "huggingface":
            auth = "optional" if name in _OPTIONAL_KEY_PROVIDERS else "required"
        elif info is not None and getattr(info, "authentication_required", False):
            auth = "required"
        else:
            auth = "none"

        reachable: Optional[bool] = None
        reachability = ""
        if probe and kind in {"local_server", "endpoint_profile"}:
            reachable, reachability = _probe_local_server(name, base_url)

        note = _PROVIDER_NOTES.get(name, "")
        if not note and kind == "local_engine":
            note = _LOCAL_ENGINE_HINT
        if profile is not None and not note:
            note = f"endpoint profile ({profile.get('provider_family') or 'openai-compatible'})"

        rows.append(
            {
                "provider": name,
                "display_name": str(getattr(info, "display_name", "") or (profile or {}).get("display_name") or name),
                "kind": kind,
                "auth": auth,
                "api_key_field": key_field,
                "api_key_env_var": key_env,
                "api_key_set": bool(key_set),
                "api_key_source": key_source,
                # Non-reversible 8 chars, the same fingerprint every other
                # surface prints. NEVER key material.
                "api_key_fingerprint": key_fp,
                "base_url": base_url,
                "base_url_source": base_url_source,
                "reachable": reachable,
                "reachability": reachability,
                "note": note,
                "description": str(getattr(info, "description", "") or ""),
            }
        )
    return rows


def _http_json(url: str, *, timeout: float = _HTTP_TIMEOUT) -> Tuple[Optional[Any], str]:
    """One short localhost GET. `(payload, "")` or `(None, why_not)`.

    Never raises: an unreachable local server is a normal state of the world
    for a probe, not an exception the grid should blow up on.
    """

    try:
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - localhost daemon
            raw = response.read()
    except urllib.error.HTTPError as exc:
        return None, f"GET {url} returned HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return None, f"GET {url} unreachable ({exc.reason})"
    except Exception as exc:
        return None, f"GET {url} failed ({exc})"
    try:
        return json.loads(raw.decode("utf-8")), ""
    except Exception as exc:
        return None, f"GET {url} returned unparseable JSON ({exc})"


# ---------------------------------------------------------------------------
# probe()
# ---------------------------------------------------------------------------


def probe(provider: Any, model: Any, *, base_url: Optional[str] = None) -> ModelPresence:
    """Are `model`'s weights present locally for `provider`? Never downloads.

    Cheap enough to call for every row of a grid on every render: at worst one
    localhost HTTP GET with a 3s timeout, or one directory listing.
    """

    pid = _provider_id(provider)
    artifact = str(model or "").strip()
    if not pid:
        return ModelPresence("", artifact, PRESENCE_UNKNOWN, evidence="no provider", detail="no provider named")
    if not artifact:
        return ModelPresence(pid, "", PRESENCE_UNKNOWN, evidence="no model", detail="no model named")

    if _is_relay(pid):
        return ModelPresence(
            pid,
            artifact,
            PRESENCE_NOT_APPLICABLE,
            evidence="relay provider",
            detail=f"{pid} serves models remotely; there is nothing to download locally",
        )

    try:
        if pid == "lmstudio":
            return _probe_lmstudio(artifact, base_url)
        if pid == "ollama":
            return _probe_ollama(artifact, base_url)
        if pid == "supertonic":
            return _probe_supertonic(artifact)
        if pid in _HF_BACKED:
            return _with_companion_presence(pid, artifact, _probe_huggingface(pid, artifact))
    except Exception as exc:  # pragma: no cover - a probe must never raise
        return ModelPresence(
            pid,
            artifact,
            PRESENCE_UNKNOWN,
            evidence="probe error",
            detail=str(exc),
        )

    return ModelPresence(
        pid,
        artifact,
        PRESENCE_UNKNOWN,
        evidence="no materializer",
        detail=f"AbstractCore has no local-weights probe for provider {pid!r}",
        instruction=(
            "Supported providers: " + ", ".join(sorted(supported_providers())) + ". "
            "Install this model with the provider's own tool."
        ),
    )


# --- lmstudio ---------------------------------------------------------------


def _lmstudio_base_url(base_url: Optional[str]) -> str:
    raw = (base_url or os.environ.get("LMSTUDIO_BASE_URL") or "http://localhost:1234/v1").strip()
    return raw.rstrip("/")


def _probe_lmstudio(artifact: str, base_url: Optional[str]) -> ModelPresence:
    """Downloaded set first (`lms ls --json`), served set second (`/v1/models`).

    The CLI is authoritative for the question actually being asked -- "are the
    weights on disk" -- because the HTTP endpoint only lists what the server
    currently serves. When only the HTTP answer is available, a HIT is still a
    hit (a served model is by definition downloaded) but a MISS is `unknown`,
    not `absent`: LM Studio serves a subset of what it stores.
    """

    ids, error = _lms_downloaded_ids()
    if ids is not None:
        hit = next((i for i in ids if _matches_installed_id(i, artifact)), None)
        if hit:
            return ModelPresence(
                "lmstudio",
                artifact,
                PRESENCE_INSTALLED,
                evidence="lms ls --json",
                detail=f"installed as {hit}",
                downloadable=True,
            )
        return ModelPresence(
            "lmstudio",
            artifact,
            PRESENCE_ABSENT,
            evidence="lms ls --json",
            detail=f"not among {len(ids)} downloaded LM Studio model(s)",
            instruction=f"lms get {artifact}",
            downloadable=True,
        )

    served, http_error = _lmstudio_served_ids(base_url)
    if served is not None:
        hit = next((i for i in served if _matches_installed_id(i, artifact)), None)
        if hit:
            return ModelPresence(
                "lmstudio",
                artifact,
                PRESENCE_INSTALLED,
                evidence="GET /v1/models",
                detail=f"served as {hit}",
                downloadable=_lms_cli() is not None,
            )
        return ModelPresence(
            "lmstudio",
            artifact,
            PRESENCE_UNKNOWN,
            evidence="GET /v1/models",
            detail=(
                "the LM Studio server does not serve this id, but it lists only loaded/served "
                "models -- the weights may still be on disk"
            ),
            instruction=_LMS_INSTALL_HINT,
            downloadable=False,
        )

    return ModelPresence(
        "lmstudio",
        artifact,
        PRESENCE_UNKNOWN,
        evidence="no lms CLI, server unreachable",
        detail="; ".join(x for x in (error, http_error) if x),
        instruction=_LMS_INSTALL_HINT,
        downloadable=False,
    )


_LMS_INSTALL_HINT = (
    "Install LM Studio (https://lmstudio.ai) and enable its CLI with `npx lmstudio install-cli`, "
    "or start the local server, then re-check."
)


def _lms_cli() -> Optional[str]:
    explicit = os.environ.get("ABSTRACTCORE_LMS_CLI", "").strip()
    if explicit:
        return explicit if Path(explicit).exists() or shutil.which(explicit) else None
    found = shutil.which("lms")
    if found:
        return found
    # LM Studio installs its CLI here but does not always add it to PATH.
    candidate = Path.home() / ".lmstudio" / "bin" / "lms"
    return str(candidate) if candidate.exists() else None


def _lms_downloaded_ids() -> Tuple[Optional[List[str]], str]:
    return _cached_listing("lms:ls", _read_lms_downloaded_ids)


def _read_lms_downloaded_ids() -> Tuple[Optional[List[str]], str]:
    """The downloaded-model ids `lms ls --json` reports, or why we have none.

    A SHAPE WE DO NOT RECOGNISE IS NOT AN EMPTY LIBRARY. `lms` is a third-party
    CLI whose JSON we do not own. If it ever answers with an object, or with
    rows carrying none of the id keys, the old code read that as "0 models
    downloaded" and every row of the grid turned `absent` -- an offer to
    re-fetch a library already on disk, from zero evidence. Losing the shape
    loses the evidence, and rule 2 says that is `unknown`. A genuinely EMPTY
    list is different: that is a real, empty library, and `absent` is right.
    """

    cli = _lms_cli()
    if not cli:
        return None, "the `lms` CLI is not installed"
    try:
        proc = subprocess.run(
            [cli, "ls", "--json"],
            capture_output=True,
            text=True,
            timeout=_CLI_PROBE_TIMEOUT,
        )
    except Exception as exc:
        return None, f"`lms ls` failed: {exc}"
    if proc.returncode != 0:
        return None, f"`lms ls` exited {proc.returncode}: {(proc.stderr or '').strip()[:200]}"
    try:
        payload = json.loads(proc.stdout or "[]")
    except Exception as exc:
        return None, f"`lms ls --json` returned unparseable output: {exc}"
    if not isinstance(payload, list):
        return None, f"`lms ls --json` returned a {type(payload).__name__}, not a list of models"
    ids: List[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        for key in ("modelKey", "indexedModelIdentifier", "path"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                ids.append(value.strip())
    if payload and not ids:
        return None, f"`lms ls --json` listed {len(payload)} entries carrying no recognisable model id"
    return ids, ""


def _lmstudio_served_ids(base_url: Optional[str]) -> Tuple[Optional[List[str]], str]:
    url = _lmstudio_base_url(base_url)
    if not url.endswith("/models"):
        url = f"{url}/models"
    payload, error = _cached_listing(f"lmstudio:served:{url}", lambda: _http_json(url))
    if payload is None:
        return None, error
    data = payload.get("data") if isinstance(payload, dict) else None
    ids = [str(row.get("id")).strip() for row in (data or []) if isinstance(row, dict) and row.get("id")]
    return ids, ""


# --- ollama -----------------------------------------------------------------


_OLLAMA_DEFAULT_URL = "http://localhost:11434"


def _ollama_base_url(base_url: Optional[str]) -> str:
    raw = (base_url or os.environ.get("OLLAMA_BASE_URL") or _OLLAMA_DEFAULT_URL).strip()
    return raw.rstrip("/")


def _ollama_endpoint_identity(url: str) -> Optional[Tuple[str, int]]:
    """`(host, port)` with loopback spellings folded together, or None."""

    try:
        parts = urllib.parse.urlsplit(url if "//" in url else f"//{url}")
        host = (parts.hostname or "").lower()
        port = parts.port or 11434
    except Exception:
        return None
    if not host:
        return None
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"}:
        host = "localhost"
    return host, port


def _ollama_is_this_machine(url: str) -> bool:
    """Is this URL the exact daemon the local `ollama` CLI would talk to?

    The CLI answers about ONE daemon -- the one `OLLAMA_HOST` names, localhost
    :11434 unless told otherwise. Using it as a fallback for any OTHER endpoint
    answered a different question than the one asked: it reported the local
    library and printed `ollama pull ...`, an instruction that fetches weights
    onto the wrong daemon. HOST AND PORT BOTH COUNT -- a second Ollama on
    :59999 is as much "not this one" as a host across the network. Only the
    loopback SPELLINGS (localhost / 127.0.0.1 / ::1) are folded together,
    because those really are one daemon.
    """

    want = _ollama_endpoint_identity(url)
    default = _ollama_endpoint_identity((os.environ.get("OLLAMA_BASE_URL") or _OLLAMA_DEFAULT_URL).strip())
    return want is not None and want == default


def _probe_ollama(artifact: str, base_url: Optional[str]) -> ModelPresence:
    """`GET /api/tags` is the whole answer: Ollama lists what it has stored.

    `:latest` is implicit in Ollama's own vocabulary, so `gemma3:1b` and
    `gemma3:1b:latest` are the same tag and both match.
    """

    host = _ollama_base_url(base_url)
    url = f"{host}/api/tags"
    payload, error = _cached_listing(f"ollama:tags:{url}", lambda: _http_json(url))
    if payload is None:
        local = _ollama_is_this_machine(host)
        cli_ids, cli_error = _ollama_cli_ids() if local else (None, "the local `ollama` CLI answers about this machine, not " + host)
        if cli_ids is None:
            return ModelPresence(
                "ollama",
                artifact,
                PRESENCE_UNKNOWN,
                evidence="GET /api/tags unreachable",
                detail="; ".join(x for x in (error, cli_error) if x),
                instruction=(
                    "Start Ollama (`ollama serve`) or install it from https://ollama.com, then re-check."
                    if local
                    else f"Make {host} reachable, then re-check."
                ),
                downloadable=False,
            )
        names = cli_ids
        evidence = "ollama list"
    else:
        models = payload.get("models") if isinstance(payload, dict) else None
        names = [str(row.get("name") or row.get("model") or "").strip() for row in (models or []) if isinstance(row, dict)]
        evidence = "GET /api/tags"

    names = [n for n in names if n]
    if any(_ollama_tag_match(name, artifact) for name in names):
        return ModelPresence(
            "ollama",
            artifact,
            PRESENCE_INSTALLED,
            evidence=evidence,
            detail=f"pulled ({len(names)} model(s) present)",
            downloadable=True,
        )
    return ModelPresence(
        "ollama",
        artifact,
        PRESENCE_ABSENT,
        evidence=evidence,
        detail=f"not among {len(names)} pulled model(s)",
        instruction=f"ollama pull {artifact}",
        downloadable=True,
    )


def _ollama_tag_match(name: str, artifact: str) -> bool:
    got = _norm(name)
    want = _norm(artifact)
    if not got or not want:
        return False
    if got == want:
        return True
    strip = lambda v: v[: -len(":latest")] if v.endswith(":latest") else v  # noqa: E731
    return strip(got) == strip(want)


def _ollama_cli_ids() -> Tuple[Optional[List[str]], str]:
    return _cached_listing("ollama:list", _read_ollama_cli_ids)


def _read_ollama_cli_ids() -> Tuple[Optional[List[str]], str]:
    cli = shutil.which("ollama")
    if not cli:
        return None, "the `ollama` CLI is not installed"
    try:
        proc = subprocess.run([cli, "list"], capture_output=True, text=True, timeout=_CLI_PROBE_TIMEOUT)
    except Exception as exc:
        return None, f"`ollama list` failed: {exc}"
    if proc.returncode != 0:
        return None, f"`ollama list` exited {proc.returncode}: {(proc.stderr or '').strip()[:200]}"
    names: List[str] = []
    for line in (proc.stdout or "").splitlines()[1:]:
        head = line.split()[0] if line.split() else ""
        if head:
            names.append(head)
    return names, ""


# --- supertonic -------------------------------------------------------------


def _probe_supertonic(artifact: str) -> ModelPresence:
    """AbstractVoice owns Supertonic's cache; ask IT, never re-derive the path."""

    try:
        from abstractvoice.supertonic.runtime import (  # type: ignore
            get_supertonic_cache_dir,
            is_supertonic_cached,
        )
    except Exception as exc:
        return ModelPresence(
            "supertonic",
            artifact,
            PRESENCE_UNKNOWN,
            evidence="abstractvoice not importable",
            detail=str(exc),
            instruction='pip install "abstractvoice[supertonic]"',
            downloadable=False,
        )
    try:
        cache_dir = get_supertonic_cache_dir(None)
        cached = bool(is_supertonic_cached(cache_dir))
    except Exception as exc:
        return ModelPresence(
            "supertonic",
            artifact,
            PRESENCE_UNKNOWN,
            evidence="abstractvoice cache check failed",
            detail=str(exc),
            downloadable=True,
        )
    if cached:
        return ModelPresence(
            "supertonic",
            artifact,
            PRESENCE_INSTALLED,
            evidence="abstractvoice supertonic cache",
            location=str(cache_dir),
            downloadable=True,
        )
    return ModelPresence(
        "supertonic",
        artifact,
        PRESENCE_ABSENT,
        evidence="abstractvoice supertonic cache",
        detail=f"no Supertonic 3 ONNX assets under {cache_dir}",
        instruction="abstractcore models download supertonic supertonic-3",
        downloadable=True,
    )


# --- huggingface-backed (mlx-gen, mlx, huggingface, ...) ---------------------


def _hf_cache_dirs() -> List[Path]:
    """The HF hub caches this install actually reads.

    Reuses AbstractCore's existing resolution (`capabilities.vision_catalog`)
    rather than minting a second opinion about where the cache lives -- the
    image catalog and this probe must agree, or the console will offer to
    download weights the generator can already see.
    """

    try:
        from ..capabilities.vision_catalog import _default_hf_hub_cache_dirs  # type: ignore

        return list(_default_hf_hub_cache_dirs())
    except Exception:
        from ..utils.model_cache import hf_hub_cache_dirs

        return hf_hub_cache_dirs()


def _hf_cached_snapshot(repo_id: str) -> Optional[Path]:
    try:
        from ..capabilities.vision_catalog import _cached_hf_snapshot  # type: ignore

        return _cached_hf_snapshot(repo_id, _hf_cache_dirs())
    except Exception:
        return None


def _hf_interrupted_downloads(repo_id: str) -> Tuple[int, int, Optional[Path]]:
    """`(file_count, bytes_on_disk, blobs_dir)` for this repo's `.incomplete` files.

    THE PARTIAL SNAPSHOT IS THE WORST LIE THIS MODULE CAN TELL, and finding it
    needs one fact about how `huggingface_hub` writes: an in-flight file is
    `<repo>/blobs/<sha>.incomplete`, and the symlink into
    `<repo>/snapshots/<rev>/<name>` is only created once that blob is COMPLETE.
    So an interrupted multi-shard download leaves a snapshot directory that
    looks perfectly healthy -- every file in it is whole -- while most of the
    weights are missing. A `.incomplete` scan of the snapshot directory finds
    nothing, because the evidence is one level up, in `blobs/`.

    Scanning `blobs/` is therefore the only offline way to tell "this repo is
    mid-download" from "this repo is here". It is a repo-level fact, not a
    per-revision one (an incomplete blob carries no revision), which is why the
    caller reports the interruption rather than silently guessing which
    revision it belongs to.
    """

    folder = "models--" + repo_id.replace("/", "--")
    count = 0
    size = 0
    where: Optional[Path] = None
    for base in _hf_cache_dirs():
        blobs = base / folder / "blobs"
        # A download marker whose planned files are not all whole is an
        # interruption even with no `.incomplete` left (a cancel that landed
        # between two files). See `_download_huggingface`.
        unfinished = _hf_marker_unfinished(base / folder) or 0
        partials = 0
        try:
            if blobs.is_dir():
                for path in blobs.iterdir():
                    if not path.name.endswith(".incomplete"):
                        continue
                    partials += 1
                    try:
                        size += path.stat().st_size
                    except Exception:
                        pass
        except Exception:
            pass
        if unfinished or partials:
            count += max(unfinished, partials)
            where = where or blobs
    return count, size, where


def _interrupted_presence(provider: str, artifact: str, repo_id: str, count: int, size: int, where: Optional[Path]) -> ModelPresence:
    gb = size / 1_000_000_000
    return ModelPresence(
        provider,
        artifact,
        PRESENCE_ABSENT,
        evidence="hf cache scan (interrupted download)",
        detail=(
            f"{repo_id} is partially downloaded: {count} interrupted file(s), "
            f"{gb:.1f} GB on disk. The files already fetched are whole, so the snapshot "
            "looks complete -- the rest of the weights are not here."
        ),
        location=str(where) if where else None,
        instruction=(
            f"Re-run the download to resume it: abstractcore models download {provider} {repo_id}"
            + (f" (or delete the stale .incomplete files under {where} if they are from an abandoned revision)" if where else "")
        ),
        downloadable=True,
    )


def _hf_broken_links(snapshot: Path) -> List[str]:
    """Snapshot entries whose symlink points at a blob that is not there.

    A cache copied without its `blobs/` folder (or with a blob deleted) keeps
    `snapshots/<rev>/<file>` links to nothing: the snapshot LOOKS complete
    (the names are all there) and no load can read it (mission KK).
    """

    broken: List[str] = []
    try:
        for path in snapshot.rglob("*"):
            if path.is_symlink() and not path.exists():
                broken.append(str(path.relative_to(snapshot)))
    except Exception:
        return broken
    return sorted(broken)


def _hf_unlinked_blobs(repo_id: str) -> Tuple[int, Optional[Path]]:
    """`(blob_bytes, repo_dir)` for a repo whose blobs are here but whose snapshot has no files.

    What `rsync -r` (without `-l`/`-a`) leaves: it skips symbolic links, so
    every `snapshots/<rev>/<file>` link is missing while `blobs/` holds all
    the data. A download repairs it without fetching the bytes again
    (huggingface_hub links a blob that is already whole).
    """

    folder = "models--" + repo_id.replace("/", "--")
    for base in _hf_cache_dirs():
        repo_dir = base / folder
        blobs = repo_dir / "blobs"
        snaps = repo_dir / "snapshots"
        try:
            if not blobs.is_dir() or not snaps.is_dir():
                continue
            if any(p.is_file() or p.is_symlink() for p in snaps.rglob("*")):
                continue
            size = sum(p.stat().st_size for p in blobs.iterdir() if p.is_file() and not p.name.endswith(".incomplete"))
        except Exception:
            continue
        if size:
            return size, repo_dir
    return 0, None


def _probe_huggingface(provider: str, artifact: str) -> ModelPresence:
    repo_id, quant, patterns = hf_artifact_parts(artifact)
    if "/" not in repo_id:
        return ModelPresence(
            provider,
            artifact,
            PRESENCE_UNKNOWN,
            evidence="hf cache scan",
            detail=f"{artifact!r} is not an <org>/<repo> Hugging Face reference",
            downloadable=False,
        )
    interrupted, interrupted_bytes, blobs_dir = _hf_interrupted_downloads(repo_id)
    snapshot = _hf_cached_snapshot(repo_id)
    if snapshot is not None:
        # A snapshot directory full of whole files is NOT proof the model is
        # here; see `_hf_interrupted_downloads`. An interrupted repo reports
        # `absent`, not `unknown`, because the repair is known and cheap: the
        # same download resumes exactly where it stopped.
        if interrupted:
            return _interrupted_presence(provider, artifact, repo_id, interrupted, interrupted_bytes, blobs_dir)
        broken = _hf_broken_links(snapshot)
        if broken:
            return ModelPresence(
                provider,
                artifact,
                PRESENCE_ABSENT,
                evidence="hf cache scan (missing data)",
                detail=(
                    f"{repo_id} is only partly on this computer: {len(broken)} of its files "
                    f"({', '.join(broken[:3])}{', ...' if len(broken) > 3 else ''}) point to data that is not "
                    "there (the cache was probably copied without its blobs folder). Download it again to repair it."
                ),
                location=str(snapshot),
                instruction=f"abstractcore models download {provider} {artifact}",
                downloadable=True,
            )
        if patterns and not _snapshot_has_matching_file(snapshot, patterns):
            # A multi-quant GGUF repo is cached, but not THIS quant: the
            # artifact names one file set, and that set is not here.
            return ModelPresence(
                provider,
                artifact,
                PRESENCE_ABSENT,
                evidence="hf cache scan",
                detail=f"{repo_id} is cached but has no {quant} GGUF file",
                location=str(snapshot),
                instruction=f"abstractcore models download {provider} {artifact}",
                downloadable=True,
            )
        return ModelPresence(
            provider,
            artifact,
            PRESENCE_INSTALLED,
            evidence="hf cache scan",
            location=str(snapshot),
            downloadable=True,
        )
    if interrupted:
        return _interrupted_presence(provider, artifact, repo_id, interrupted, interrupted_bytes, blobs_dir)
    unlinked, repo_dir = _hf_unlinked_blobs(repo_id)
    if unlinked:
        return ModelPresence(
            provider,
            artifact,
            PRESENCE_ABSENT,
            evidence="hf cache scan (links missing)",
            detail=(
                f"{repo_id}'s data is on this computer ({format_bytes(unlinked)}), but the file links that name it "
                "are missing (the cache was probably copied without symbolic links, e.g. rsync without -a). "
                "Download it again: the data already here is reused, not fetched again."
            ),
            location=str(repo_dir) if repo_dir else None,
            instruction=f"abstractcore models download {provider} {artifact}",
            downloadable=True,
        )
    dirs = _hf_cache_dirs()
    if not dirs:
        return ModelPresence(
            provider,
            artifact,
            PRESENCE_UNKNOWN,
            evidence="hf cache scan",
            detail="no Hugging Face cache directory exists on this machine yet",
            instruction=f"abstractcore models download {provider} {repo_id}",
            downloadable=True,
        )
    return ModelPresence(
        provider,
        artifact,
        PRESENCE_ABSENT,
        evidence="hf cache scan",
        detail=f"no complete snapshot of {repo_id} in {len(dirs)} cache dir(s)",
        instruction=f"abstractcore models download {provider} {repo_id}",
        downloadable=True,
    )


# ---------------------------------------------------------------------------
# download()
# ---------------------------------------------------------------------------


def download(
    provider: Any,
    artifact: Any,
    *,
    progress_cb: Optional[ProgressCallback] = None,
    base_url: Optional[str] = None,
    dry_run: bool = False,
    expected_bytes: Optional[int] = None,
) -> DownloadOutcome:
    """Fetch one artifact with the provider's own tool. Only on explicit request.

    `expected_bytes` (from the catalog, when known) arms a DISK PRE-CHECK: a
    download that cannot fit on the target filesystem with 5 GiB to spare is
    refused before the first byte, instead of filling the disk and failing
    half-way. Hugging Face downloads compute the exact figure themselves.

    `dry_run` resolves everything -- provider support, current presence, the
    exact command -- and stops before spending a byte, which is how the
    recommended journey can be demonstrated on a machine that must not fill up.
    """

    pid = _provider_id(provider)
    ref = str(artifact or "").strip()
    emit = progress_cb or (lambda _p: None)

    # A DOWNLOAD INVALIDATES EVERY LISTING. If a caller wrapped this in a
    # `presence_sweep()`, the library read before the bytes landed must not be
    # reused to answer "did it land?" -- so a download always runs outside the
    # ambient sweep, and the probes it makes are fresh.
    _outer_sweep = _sweep.set(None)
    try:
        return _download(pid, ref, emit, base_url=base_url, dry_run=dry_run, expected_bytes=expected_bytes)
    finally:
        _sweep.reset(_outer_sweep)


def _download(
    pid: str,
    ref: str,
    emit: ProgressCallback,
    *,
    base_url: Optional[str] = None,
    dry_run: bool = False,
    expected_bytes: Optional[int] = None,
) -> DownloadOutcome:
    if not pid or not ref:
        return DownloadOutcome(pid, ref, False, "failed", message="a provider and an artifact are required")

    if _is_relay(pid):
        return DownloadOutcome(
            pid,
            ref,
            False,
            "not_applicable",
            message=f"{pid} serves models remotely; there is nothing to download",
        )

    handler = _DOWNLOADERS.get(pid) or (_download_huggingface if pid in _HF_BACKED else None)
    if handler is None:
        return DownloadOutcome(
            pid,
            ref,
            False,
            "failed",
            message=f"AbstractCore has no download tool for provider {pid!r}",
            instruction="Supported providers: " + ", ".join(sorted(supported_providers())),
        )

    companions = companion_artifacts(pid, ref)
    if companions:
        return _download_with_companions(
            pid, ref, companions, handler, emit, base_url=base_url, dry_run=dry_run, expected_bytes=expected_bytes
        )

    presence = probe(pid, ref, base_url=base_url)
    if presence.status == PRESENCE_INSTALLED:
        return DownloadOutcome(
            pid,
            ref,
            True,
            "already_installed",
            message=presence.detail or "already installed",
            location=presence.location,
        )

    disk_problem = _disk_shortfall(pid, expected_bytes)

    if dry_run:
        return DownloadOutcome(
            pid,
            ref,
            True,
            "planned",
            message=f"would download {ref} with {pid}" + (f" -- WARNING: {disk_problem}" if disk_problem else ""),
            command=_planned_command(pid, ref),
        )

    if disk_problem:
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=disk_problem))
        return DownloadOutcome(
            pid,
            ref,
            False,
            "failed",
            message=disk_problem,
            command=_planned_command(pid, ref),
            instruction="Free disk space (abstractcore models list shows what is installed), then retry.",
        )

    emit(DownloadProgress(status=DownloadStatus.STARTING, message=f"{pid}: fetching {ref}"))
    token = _expected_bytes_var.set(expected_bytes if isinstance(expected_bytes, int) and expected_bytes > 0 else None)
    try:
        return handler(ref, emit, base_url)
    except Exception as exc:  # pragma: no cover - handlers convert their own failures
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=f"{type(exc).__name__}: {exc}"))
        return DownloadOutcome(pid, ref, False, "failed", message=f"{type(exc).__name__}: {exc}")
    finally:
        _expected_bytes_var.reset(token)


# ---------------------------------------------------------------------------
# Companion artifacts: an MLX model's separate MTP drafter
# ---------------------------------------------------------------------------
#
# Several MLX models accelerate with a SEPARATE MTP head repo (e.g.
# `mlx-works/Qwen3.5-9B-oQ4e-mtp` -> `mlx-community/Qwen3.5-9B-MTP-4bit`), named
# by the registry (`model_capabilities.json` `speculation.runtimes.mlx.drafter`,
# read through `providers.speculation.mlx_companion_repos` -- the SAME entry the
# provider loads it from). Loading never downloads (offline_first), so a model
# fetched without its companion ran without MTP forever, silently. Here the
# companion is part of the model: fetched in the same job, counted in its bytes,
# cancelled with it, and the model is "installed" only when both are whole.
# Built-in heads (`mode: embedded`, Qwen3.8 Flash-Next) have no companion.

COMPANION_ROLE = "mtp_companion"


def companion_artifacts(provider: Any, artifact: Any) -> List[str]:
    """The registry companions an artifact needs for `provider` ([] for none)."""

    if _provider_id(provider) != "mlx":
        return []
    repo_id, _quant, _patterns = hf_artifact_parts(artifact)
    from ..providers.speculation import mlx_companion_repos

    return list(mlx_companion_repos(repo_id))


def _with_companion_presence(pid: str, artifact: str, base: ModelPresence) -> ModelPresence:
    """`installed` only when the model AND every companion are whole."""

    if base.status != PRESENCE_INSTALLED:
        return base
    companions = companion_artifacts(pid, artifact)
    missing = [c for c in companions if _probe_huggingface(pid, c).status != PRESENCE_INSTALLED]
    if not missing:
        return base
    return ModelPresence(
        pid,
        artifact,
        PRESENCE_ABSENT,
        evidence="hf cache scan (model + MTP companion)",
        detail=(
            f"the model's weights are cached at {base.location}, but its MTP companion "
            f"{', '.join(missing)} is not downloaded, so MTP acceleration stays off"
        ),
        location=base.location,
        instruction=f"abstractcore models download {pid} {artifact}",
        downloadable=True,
    )


class _CompanionProgress:
    """One job's progress over a model and its companions: bytes add up, files are one list.

    Each part's own progress (the Hugging Face blob watcher's) is re-based into
    the whole: `downloaded_bytes`/`total_bytes` are sums, file rows carry the
    part they belong to (`artifact`, `role`; a companion's rows are named
    `<companion repo>/<file>`), and a part's COMPLETE is not the job's.
    """

    def __init__(self, emit: ProgressCallback, parts: List[Dict[str, Any]]):
        self.emit = emit
        self.parts = parts
        self.state: Dict[str, Dict[str, Any]] = {}
        for part in parts:
            files = [
                {
                    "name": self._name(part, f["name"]),
                    "bytes_done": 0,
                    "bytes_total": f.get("size"),
                    "state": "pending",
                    "artifact": part["artifact"],
                    "role": part["role"],
                }
                for f in (part.get("plan") or [])
            ]
            self.state[part["artifact"]] = {
                "done": 0,
                "total": part.get("total"),
                "files": files,
                "fetch": not part["installed"],
            }

    @staticmethod
    def _name(part: Dict[str, Any], name: str) -> str:
        return name if part["role"] == "model" else f"{part['artifact']}/{name}"

    def sink(self, part: Dict[str, Any]) -> ProgressCallback:
        return lambda progress: self._on(part, progress)

    def totals(self) -> Tuple[int, Optional[int]]:
        fetching = [st for st in self.state.values() if st["fetch"]]
        done = sum(int(st["done"] or 0) for st in fetching)
        totals = [st["total"] for st in fetching]
        total = sum(int(t) for t in totals) if all(isinstance(t, int) for t in totals) else None
        return done, total

    def _on(self, part: Dict[str, Any], progress: Any) -> None:
        st = self.state[part["artifact"]]
        status = getattr(progress, "status", None)
        if isinstance(getattr(progress, "downloaded_bytes", None), int):
            st["done"] = int(progress.downloaded_bytes)
        if isinstance(getattr(progress, "total_bytes", None), int) and progress.total_bytes > 0:
            st["total"] = int(progress.total_bytes)
        files = getattr(progress, "files", None)
        if isinstance(files, list):
            st["files"] = [
                dict(f, name=self._name(part, str(f.get("name"))), artifact=part["artifact"], role=part["role"])
                for f in files
                if isinstance(f, dict)
            ]
        if status == DownloadStatus.COMPLETE:
            if isinstance(st["total"], int):
                st["done"] = st["total"]
            for row in st["files"]:
                row["state"] = "done"
                if row.get("bytes_total"):
                    row["bytes_done"] = row["bytes_total"]
            # A part's end is not the job's: hold the job in `verifying`
            # (that part was just checked whole) until the next part's bytes flow.
            status = DownloadStatus.VERIFYING
        done, total = self.totals()
        message = str(getattr(progress, "message", "") or "")
        if part["role"] != "model" and message and part["artifact"] not in message:
            message = f"MTP companion {part['artifact']}: {message}"
        current = getattr(progress, "current_file", None)
        self.emit(
            DownloadProgress(
                status=status or DownloadStatus.DOWNLOADING,
                message=message,
                percent=(min(100.0, done / total * 100.0) if total else None),
                downloaded_bytes=done,
                total_bytes=total,
                phase=getattr(progress, "phase", None),
                files=[row for p in self.parts for row in self.state[p["artifact"]]["files"]] or None,
                current_file=self._name(part, str(current)) if current else None,
                # Explicit False once the total is known: the job keeps the
                # last explicit value, and the first event (before any file
                # list) must not leave "size unknown" stuck on it.
                size_unknown=None if (total is None and not done) else (total is None),
                size_note=getattr(progress, "size_note", None),
            )
        )


def _companion_entry(part: Dict[str, Any], status: str, outcome: Optional[DownloadOutcome] = None) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "artifact": part["artifact"],
        "role": part["role"],
        "status": status,
        "ok": status in ("completed", "already_installed", "planned"),
    }
    if part.get("total") is not None:
        entry["size_bytes"] = part["total"]
    location = (outcome.location if outcome is not None else None) or part["presence"].location
    if location:
        entry["location"] = location
    if outcome is not None and outcome.message:
        entry["message"] = outcome.message
    return entry


def _download_with_companions(
    pid: str,
    ref: str,
    companions: List[str],
    handler: Callable[[str, ProgressCallback, Optional[str]], DownloadOutcome],
    emit: ProgressCallback,
    *,
    base_url: Optional[str],
    dry_run: bool,
    expected_bytes: Optional[int],
) -> DownloadOutcome:
    """Fetch a model and its companion(s) as ONE job; `completed` only when all are whole."""

    parts: List[Dict[str, Any]] = [{"artifact": ref, "role": "model"}] + [
        {"artifact": c, "role": COMPANION_ROLE} for c in companions
    ]
    for part in parts:
        part["presence"] = _probe_huggingface(pid, part["artifact"])
        part["installed"] = part["presence"].status == PRESENCE_INSTALLED
    model, extras = parts[0], parts[1:]
    names = ", ".join(c["artifact"] for c in extras)

    if all(part["installed"] for part in parts):
        return DownloadOutcome(
            pid,
            ref,
            True,
            "already_installed",
            message=f"already installed, with its MTP companion {names}",
            location=model["presence"].location,
            companions=[_companion_entry(c, "already_installed") for c in extras],
        )

    # Sizes BEFORE the first byte: one metadata request per companion (the
    # model's own plan is read by its download, as for any artifact).
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    for part in extras:
        if not part["installed"]:
            plan, _revision, _error = _hf_file_plan(part["artifact"], None, token)
            part["plan"] = plan
            if plan is not None and all(isinstance(f.get("size"), int) for f in plan):
                part["total"] = sum(int(f["size"]) for f in plan)

    todo = [part for part in parts if not part["installed"]]
    disk_problem = _disk_shortfall(pid, expected_bytes) if not model["installed"] else None
    if dry_run:
        what = " and ".join(
            ("the model" if p["role"] == "model" else f"its MTP companion {p['artifact']}")
            + (f" ({format_bytes(p['total'])})" if p.get("total") else "")
            for p in todo
        )
        return DownloadOutcome(
            pid,
            ref,
            True,
            "planned",
            message=f"would download {what} for {ref} with {pid}" + (f" -- WARNING: {disk_problem}" if disk_problem else ""),
            command=_planned_command(pid, todo[0]["artifact"]),
            companions=[
                dict(_companion_entry(c, "already_installed" if c["installed"] else "planned"),
                     **({} if c["installed"] else {"command": _planned_command(pid, c["artifact"])}))
                for c in extras
            ],
        )
    if disk_problem:
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=disk_problem))
        return DownloadOutcome(
            pid,
            ref,
            False,
            "failed",
            message=disk_problem,
            command=_planned_command(pid, ref),
            instruction="Free disk space (abstractcore models list shows what is installed), then retry.",
        )

    emit(
        DownloadProgress(
            status=DownloadStatus.STARTING,
            message=f"{pid}: fetching {ref}" + (f" with its MTP companion {names}" if extras else ""),
        )
    )
    progress = _CompanionProgress(emit, parts)
    control = _job_control()
    results: Dict[str, DownloadOutcome] = {}
    stopped: Optional[str] = None
    for part in todo:
        if control is not None and control.is_cancelled():
            stopped = "cancelled"
            break
        size_token = _expected_bytes_var.set(
            expected_bytes if part["role"] == "model" and isinstance(expected_bytes, int) and expected_bytes > 0 else None
        )
        try:
            outcome = handler(part["artifact"], progress.sink(part), base_url)
        except Exception as exc:  # pragma: no cover - handlers convert their own failures
            outcome = DownloadOutcome(pid, part["artifact"], False, "failed", message=f"{type(exc).__name__}: {exc}")
        finally:
            _expected_bytes_var.reset(size_token)
        results[part["artifact"]] = outcome
        if not outcome.ok:
            stopped = "cancelled" if outcome.status == "cancelled" else "failed"
            break

    def status_of(part: Dict[str, Any]) -> str:
        if part["installed"]:
            return "already_installed"
        got = results.get(part["artifact"])
        if got is None:
            return "cancelled" if stopped == "cancelled" else "not_started"
        return "completed" if got.ok else got.status

    entries = [_companion_entry(c, status_of(c), results.get(c["artifact"])) for c in extras]
    model_outcome = results.get(ref)
    location = (model_outcome.location if model_outcome is not None else None) or model["presence"].location
    command = (model_outcome.command if model_outcome is not None else None) or _planned_command(pid, ref)

    if stopped == "cancelled":
        emit(DownloadProgress(status=DownloadStatus.CANCELLED, message="cancelled"))
        return DownloadOutcome(
            pid,
            ref,
            False,
            "cancelled",
            # huggingface_hub >= 1.0 never resumes a file in a later run (its
            # temp name is per process): whole files are kept, the one in
            # flight starts over. Say exactly that (mission KK).
            message=(
                "cancelled; files that finished downloading stay in the cache (model and MTP companion) and are "
                "not fetched again; the file that was in progress starts over on the next download"
            ),
            command=command,
            location=location,
            companions=entries,
        )
    if stopped == "failed":
        bad = next(p for p in todo if p["artifact"] in results and not results[p["artifact"]].ok)
        why = results[bad["artifact"]].message or "download failed"
        if bad["role"] == "model":
            message = f"{ref}: {why}"
        else:
            message = (
                f"{ref} is downloaded, but its MTP companion {bad['artifact']} failed: {why}. "
                "The model is not reported installed until its companion is here; retry the same "
                "download (files that finished are kept; the file that was in progress starts over)."
            )
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=message))
        return DownloadOutcome(
            pid,
            ref,
            False,
            "failed",
            message=message,
            output=results[bad["artifact"]].output,
            command=command,
            instruction=results[bad["artifact"]].instruction,
            location=location,
            companions=entries,
        )
    done_bits = [f"snapshot cached at {location}" if location else "model cached"]
    done_bits += [f"MTP companion {e['artifact']} cached at {e.get('location') or '?'}" for e in entries]
    emit(DownloadProgress(status=DownloadStatus.COMPLETE, message="; ".join(done_bits), percent=100.0))
    return DownloadOutcome(
        pid,
        ref,
        True,
        "completed",
        message="; ".join(done_bits),
        location=location,
        command=command,
        companions=entries,
    )


# The catalog's size for the artifact being fetched (the `expected_bytes` a
# caller passed), for sources that cannot report a total themselves.
_expected_bytes_var: "contextvars.ContextVar[Optional[int]]" = contextvars.ContextVar(
    "abstractcore_download_expected_bytes", default=None
)


def _planned_command(provider: str, artifact: str) -> List[str]:
    if provider == "lmstudio":
        return [_lms_cli() or "lms", "get", artifact, "--yes"]
    if provider == "ollama":
        return ["ollama", "pull", artifact]
    if provider == "supertonic":
        return ["python", "-m", "abstractvoice", "download", "--supertonic"]
    repo_id, _quant, patterns = hf_artifact_parts(artifact)
    cmd = ["huggingface_hub.snapshot_download", repo_id]
    for pattern in patterns or []:
        cmd += ["--include", pattern]
    return cmd


_DISK_HEADROOM_BYTES = 5 * 1024**3


def _store_dir_for(provider: str) -> Optional[Path]:
    try:
        from ..utils.host_profile import default_model_store_paths

        stores = default_model_store_paths()
    except Exception:
        return None
    if provider == "ollama":
        return stores.get("ollama")
    if provider == "lmstudio":
        return stores.get("lmstudio")
    if provider in _HF_BACKED:
        return stores.get("hf_cache")
    return None


def _disk_shortfall(provider: str, needed_bytes: Optional[int]) -> Optional[str]:
    """A human sentence when `needed_bytes` cannot land with 5 GiB to spare, else None.

    Only a KNOWN size can fail this check; an unknown size is not evidence.
    Remote Ollama daemons store on their own host, so they are never checked.
    """

    if not isinstance(needed_bytes, int) or needed_bytes <= 0:
        return None
    if provider == "ollama" and not _ollama_is_this_machine(_ollama_base_url(None)):
        return None
    target = _store_dir_for(provider)
    if target is None:
        return None
    from ..utils.host_profile import disk_free_bytes

    free = disk_free_bytes(target)
    if free is None or needed_bytes <= free - _DISK_HEADROOM_BYTES:
        return None
    return (
        f"not enough disk space for {provider}: the download needs {needed_bytes / 1e9:.1f} GB "
        f"plus 5 GiB headroom, and {target} has {free / 1e9:.1f} GB free"
    )


# --- lmstudio download ------------------------------------------------------


def _download_lmstudio(artifact: str, emit: ProgressCallback, base_url: Optional[str]) -> DownloadOutcome:
    cli = _lms_cli()
    if not cli:
        return DownloadOutcome(
            "lmstudio",
            artifact,
            False,
            "failed",
            message="the `lms` CLI is not installed, so AbstractCore cannot drive an LM Studio download",
            instruction=_LMS_INSTALL_HINT + f" Then run: lms get {artifact}",
            command=["lms", "get", artifact, "--yes"],
        )
    cmd = [cli, "get", artifact, "--yes"]
    emit(DownloadProgress(status=DownloadStatus.STARTING, message=f"lms get {artifact}: resolving in LM Studio's catalog", phase="resolving"))
    parser = _LmsGetProgress()
    watch = _LmStudioDiskWatch(parser, emit, _expected_bytes_var.get())
    watcher = threading.Thread(target=watch.run, name="lmstudio-disk-watch", daemon=True)
    watcher.start()
    try:
        outcome = _run_streaming(cmd, "lmstudio", artifact, emit, parser=parser, graceful_cancel=_lms_graceful_cancel)
    finally:
        watch.stop.set()
    if not outcome.ok:
        return outcome

    # VERIFY WHAT WE GOT IS WHAT WE ASKED FOR.
    #
    # `lms get` SEARCHES; it does not fetch an id. Its own `--yes` docs say so:
    # "if there are multiple models matching the search term, the first one
    # will be used". An exact id wins when it exists, and a `@quant` that does
    # not exist FAILS rather than silently substituting another -- both good --
    # but a stale, renamed or mistyped reference can still resolve to some
    # other repo entirely, and `--yes` approves it without asking.
    #
    # So we ask afterwards. Only a POSITIVE `absent` contradicts success: an
    # `unknown` (no CLI to list with) is not evidence of anything, and must not
    # turn a good download into a reported failure.
    check = probe("lmstudio", artifact)
    if check.status == PRESENCE_ABSENT:
        return DownloadOutcome(
            "lmstudio",
            artifact,
            False,
            "failed",
            message=(
                f"`lms get` reported success but {artifact} is not among the downloaded models. "
                "`lms get` searches rather than fetching an exact id, and `--yes` accepts the first "
                "match, so a stale or mistyped reference can fetch something else."
            ),
            output=outcome.output,
            command=list(cmd),
            instruction=f"Check what landed with `lms ls`, then retry with the exact id: lms get {artifact}",
        )
    return outcome


# --- ollama download --------------------------------------------------------


def _download_ollama(artifact: str, emit: ProgressCallback, base_url: Optional[str]) -> DownloadOutcome:
    """`POST /api/pull`, streamed.

    The HTTP lane over the CLI on purpose: it reports real byte counts, works
    when only the server is reachable (a remote Ollama host), and needs no
    extra dependency. When the socket refuses, we say so and name the fix
    rather than silently trying a CLI the operator may not have.
    """

    url = f"{_ollama_base_url(base_url)}/api/pull"
    body = json.dumps({"name": artifact, "stream": True}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
        method="POST",
    )
    lines: List[str] = []
    last_status = ""
    control = _job_control()
    progress = _OllamaPullProgress()
    try:
        with urllib.request.urlopen(request, timeout=None) as response:  # noqa: S310 - localhost daemon
            if control is not None:
                # A pull that went quiet blocks in a read and would never see
                # the cancel flag: closing the socket is what stops it now.
                control.on_cancel(lambda: _close_quietly(response))
            for raw in response:
                if control is not None and control.is_cancelled():
                    emit(DownloadProgress(status=DownloadStatus.CANCELLED, message="cancelled"))
                    return DownloadOutcome(
                        "ollama",
                        artifact,
                        False,
                        "cancelled",
                        message="cancelled; Ollama keeps the layers already fetched and resumes on the next pull",
                        output="\n".join(lines),
                        command=["POST", url, artifact],
                    )
                text = raw.decode("utf-8", errors="replace").strip()
                if not text:
                    continue
                try:
                    event = json.loads(text)
                except Exception:
                    continue
                if event.get("error"):
                    message = str(event["error"])
                    lines.append(message)
                    emit(DownloadProgress(status=DownloadStatus.ERROR, message=message))
                    return DownloadOutcome(
                        "ollama",
                        artifact,
                        False,
                        "failed",
                        message=message,
                        output="\n".join(lines),
                        command=["POST", url, artifact],
                        instruction=f"Check the tag exists: `ollama pull {artifact}` lists the same error.",
                    )
                status = str(event.get("status") or "").strip()
                update = progress.feed(event)
                if update is not None:
                    emit(update)
                if status and status != last_status:
                    lines.append(status)
                    last_status = status
    except urllib.error.URLError as exc:
        if control is not None and control.is_cancelled():
            emit(DownloadProgress(status=DownloadStatus.CANCELLED, message="cancelled"))
            return DownloadOutcome(
                "ollama",
                artifact,
                False,
                "cancelled",
                message="cancelled; Ollama keeps the layers already fetched and resumes on the next pull",
                output="\n".join(lines),
                command=["POST", url, artifact],
            )
        message = f"cannot reach the Ollama server at {_ollama_base_url(base_url)}: {exc.reason}"
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=message))
        return DownloadOutcome(
            "ollama",
            artifact,
            False,
            "failed",
            message=message,
            output="\n".join(lines),
            instruction="Start Ollama with `ollama serve`, or set OLLAMA_BASE_URL to the host that runs it.",
        )
    except Exception as exc:
        if control is not None and control.is_cancelled():
            emit(DownloadProgress(status=DownloadStatus.CANCELLED, message="cancelled"))
            return DownloadOutcome(
                "ollama",
                artifact,
                False,
                "cancelled",
                message="cancelled; Ollama keeps the layers already fetched and resumes on the next pull",
                output="\n".join(lines),
                command=["POST", url, artifact],
            )
        message = f"the Ollama pull stream broke: {type(exc).__name__}: {exc}"
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=message))
        return DownloadOutcome("ollama", artifact, False, "failed", message=message, output="\n".join(lines))

    if control is not None and control.is_cancelled():
        # The socket was closed under the loop: the stream ended, not the pull.
        emit(DownloadProgress(status=DownloadStatus.CANCELLED, message="cancelled"))
        return DownloadOutcome(
            "ollama",
            artifact,
            False,
            "cancelled",
            message="cancelled; Ollama keeps the layers already fetched and resumes on the next pull",
            output="\n".join(lines),
            command=["POST", url, artifact],
        )
    if not progress.succeeded:
        # Ollama ends every good pull with `{"status":"success"}`. A stream
        # that stops without it (daemon restarted, proxy cut) did NOT pull.
        message = "the Ollama pull stream ended without `success`" + (f" (last status: {last_status})" if last_status else "")
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=message))
        return DownloadOutcome(
            "ollama",
            artifact,
            False,
            "failed",
            message=message,
            output="\n".join(lines),
            command=["POST", url, artifact],
            instruction=f"Retry; `ollama pull {artifact}` resumes from the layers already fetched.",
        )

    emit(DownloadProgress(status=DownloadStatus.COMPLETE, message=f"pulled {artifact}", percent=100.0))
    return DownloadOutcome(
        "ollama",
        artifact,
        True,
        "completed",
        message=f"pulled {artifact}",
        output="\n".join(lines),
        command=["POST", url, artifact],
    )


def _close_quietly(obj: Any) -> None:
    """Abort an HTTP response from ANOTHER thread, unblocking a pending read.

    `close()` alone does not wake a thread blocked in `recv()` on a quiet
    socket (measured: a cancelled Ollama pull waited out the daemon's 3 s
    silence). Shutting the socket down does, on every POSIX platform.
    """

    import socket as _socket

    sock = None
    try:
        raw = getattr(getattr(obj, "fp", None), "raw", None)
        sock = getattr(raw, "_sock", None)
    except Exception:
        sock = None
    if sock is not None:
        try:
            sock.shutdown(_socket.SHUT_RDWR)
        except Exception:
            pass
    try:
        obj.close()
    except Exception:
        pass


class _OllamaPullProgress:
    """Ollama's `/api/pull` NDJSON -> ONE artifact-wide progress.

    Ollama reports each LAYER separately (`digest`, `total`, `completed`), and
    pulls layers concurrently. Forwarding one layer's numbers as the job's (as
    this lane once did) made the bar jump back to 0 % at every layer. Here the
    job's bytes are the SUM over every layer seen so far, and each layer is a
    `files` row. Status words map to phases:

        pulling manifest            -> resolving
        pulling <digest> (+ bytes)  -> downloading
        verifying sha256 digest     -> verifying
        writing manifest / removing -> installing
        success                     -> (the job's end decides `done`)
    """

    def __init__(self) -> None:
        self.layers: Dict[str, Dict[str, Any]] = {}
        self.order: List[str] = []
        self.succeeded = False
        self.phase = "resolving"

    def feed(self, event: Dict[str, Any]) -> Optional[DownloadProgress]:
        status = str(event.get("status") or "").strip()
        low = status.lower()
        digest = str(event.get("digest") or "").strip()
        total = event.get("total")
        completed = event.get("completed")
        current: Optional[str] = None
        if not digest and isinstance(total, int) and low.startswith("pulling ") and "manifest" not in low:
            digest = status.split(" ", 1)[1].strip()  # older daemons: the layer is only in the status
        if low == "success":
            self.succeeded = True
            for layer in self.layers.values():
                layer["state"] = "done"
                if layer.get("bytes_total"):
                    layer["bytes_done"] = layer["bytes_total"]
            return None
        if digest:
            layer = self.layers.get(digest)
            if layer is None:
                layer = {"name": _ollama_layer_name(digest), "bytes_done": 0, "bytes_total": None, "state": "pending"}
                self.layers[digest] = layer
                self.order.append(digest)
            if isinstance(total, int) and total > 0:
                layer["bytes_total"] = total
            if isinstance(completed, int) and completed >= 0:
                layer["bytes_done"] = completed
            if layer.get("bytes_total") and layer["bytes_done"] >= layer["bytes_total"]:
                layer["state"] = "done"
            elif isinstance(completed, int):
                layer["state"] = "downloading"
            current = layer["name"]  # the layer this event is about
            self.phase = "downloading"
        elif "manifest" in low and "pulling" in low:
            self.phase = "resolving"
        elif "verif" in low:
            self.phase = "verifying"
        elif "writing" in low or "removing" in low:
            self.phase = "installing"
        elif not status:
            return None
        done = sum(int(l.get("bytes_done") or 0) for l in self.layers.values())
        totals = [l.get("bytes_total") for l in self.layers.values()]
        total_all = sum(int(t) for t in totals if isinstance(t, int)) if totals and all(isinstance(t, int) for t in totals) else None
        files = [dict(self.layers[d]) for d in self.order]
        kind = {
            "resolving": DownloadStatus.STARTING,
            "verifying": DownloadStatus.VERIFYING,
        }.get(self.phase, DownloadStatus.DOWNLOADING)
        return DownloadProgress(
            status=kind,
            message=status or self.phase,
            percent=(min(100.0, done / total_all * 100.0) if total_all else None),
            downloaded_bytes=done if self.layers else None,
            total_bytes=total_all,
            phase=self.phase,
            files=files or None,
            current_file=current,
        )


def _ollama_layer_name(digest: str) -> str:
    short = digest.split(":", 1)[-1][:12]
    return f"layer {short}"


# --- supertonic download ----------------------------------------------------


_STREAM_CHUNK = 256 * 1024
_STREAM_READ_TIMEOUT_S = 300.0
_EMIT_EVERY_S = 0.1


def _download_supertonic(artifact: str, emit: ProgressCallback, base_url: Optional[str]) -> DownloadOutcome:
    """Supertonic 3's ONNX assets, streamed file by file with real byte progress.

    AbstractVoice OWNS what Supertonic needs -- the file list, the pinned
    revision, the cache directory and the "is it cached" rule -- and this
    reads all four from it (`abstractvoice.supertonic.runtime`), so the two
    can never disagree about what "installed" means. What it does NOT use is
    `prefetch_supertonic()` itself: that call has no progress hook and no
    cancel, so a 400 MB fetch showed a bare "downloading" for minutes and a
    cancel could not stop it. The write discipline is the same as
    AbstractVoice's (a temp file in the destination directory, renamed into
    place only when whole), so a cancelled or failed file never reads as
    cached, and the temp file is removed.
    """

    try:
        from abstractvoice.supertonic import runtime as st  # type: ignore
    except Exception as exc:
        return DownloadOutcome(
            "supertonic",
            artifact,
            False,
            "failed",
            message=f"abstractvoice is not importable: {exc}",
            instruction='pip install "abstractvoice[supertonic]"',
        )
    # SEAM with abstractvoice: these names are read, never re-derived; a
    # renamed one fails loudly here rather than downloading the wrong set.
    root = Path(st.get_supertonic_cache_dir(None))
    required = [Path(p) for p in st._REQUIRED_FILES]
    model_id = str(st.MODEL_ID)
    revision = str(st.DEFAULT_REVISION)
    endpoint = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
    base = f"{endpoint}/{model_id}/resolve/{revision}"
    command = ["GET", f"{base}/<{len(required)} Supertonic files>"]

    emit(
        DownloadProgress(
            status=DownloadStatus.STARTING,
            message=f"Supertonic 3: reading the sizes of {len(required)} files",
            phase="resolving",
        )
    )
    sizes = _http_sizes([f"{base}/{rel.as_posix()}" for rel in required])
    stream = _FileSetProgress(emit)
    for rel in required:
        dest = root / rel
        have = dest.stat().st_size if dest.exists() else 0
        size = sizes.get(f"{base}/{rel.as_posix()}")
        stream.add(rel.as_posix(), size if size is not None else (have or None), have if have > 0 else 0, "done" if have > 0 else "pending")
    if any(f["bytes_total"] is None for f in stream.files):
        stream.size_note = "the hub did not report every file's size"
    stream.push(DownloadStatus.DOWNLOADING, "Supertonic 3 assets", phase="downloading")

    control = _job_control()
    for rel in required:
        dest = root / rel
        name = rel.as_posix()
        if dest.exists() and dest.stat().st_size > 0:
            continue
        result = _stream_to_file(f"{base}/{name}", dest, name, stream, control)
        if result is not None:
            status, message = result
            if status == "cancelled":
                emit(DownloadProgress(status=DownloadStatus.CANCELLED, message="cancelled"))
                return DownloadOutcome(
                    "supertonic",
                    artifact,
                    False,
                    "cancelled",
                    message="cancelled; the files already fetched are kept, the partial one was removed",
                    location=str(root),
                    command=command,
                )
            emit(DownloadProgress(status=DownloadStatus.ERROR, message=message))
            return DownloadOutcome(
                "supertonic",
                artifact,
                False,
                "failed",
                message=message,
                location=str(root),
                command=command,
                instruction="Retry (files already fetched are kept), or fetch manually with `python -m abstractvoice download --supertonic`.",
            )

    stream.push(DownloadStatus.VERIFYING, "checking every Supertonic file is in place", phase="verifying")
    try:
        cached = bool(st.is_supertonic_cached(root))
    except Exception as exc:
        cached = False
        why = f"abstractvoice could not check the cache: {exc}"
    else:
        why = "abstractvoice does not see every required file"
    if not cached:
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=why))
        return DownloadOutcome("supertonic", artifact, False, "failed", message=why, location=str(root), command=command)
    emit(DownloadProgress(status=DownloadStatus.COMPLETE, message=f"Supertonic ready at {root}", percent=100.0))
    return DownloadOutcome(
        "supertonic",
        artifact,
        True,
        "completed",
        message="Supertonic 3 assets cached",
        location=str(root),
        command=command,
    )


class _FileSetProgress:
    """A set of files fetched one after another -> `DownloadProgress` events."""

    def __init__(self, emit: ProgressCallback):
        self.emit = emit
        self.files: List[Dict[str, Any]] = []
        self.current: Optional[str] = None
        self.size_note: Optional[str] = None
        self._last = 0.0

    def add(self, name: str, total: Optional[int], done: int, state: str) -> None:
        self.files.append({"name": name, "bytes_done": int(done), "bytes_total": total, "state": state})

    def row(self, name: str) -> Dict[str, Any]:
        for entry in self.files:
            if entry["name"] == name:
                return entry
        raise KeyError(name)

    def push(self, status: DownloadStatus, message: str, *, phase: Optional[str] = None, force: bool = True) -> None:
        import time as _time

        now = _time.monotonic()
        if not force and now - self._last < _EMIT_EVERY_S:
            return
        self._last = now
        totals = [f["bytes_total"] for f in self.files]
        known = all(isinstance(t, int) for t in totals)
        total = sum(int(t) for t in totals if isinstance(t, int)) if known and totals else None
        done = sum(int(f["bytes_done"] or 0) for f in self.files)
        self.emit(
            DownloadProgress(
                status=status,
                message=message,
                percent=(min(100.0, done / total * 100.0) if total else None),
                downloaded_bytes=done,
                total_bytes=total,
                phase=phase,
                files=[dict(f) for f in self.files],
                current_file=self.current,
                size_unknown=(not known) or None,
                size_note=self.size_note,
            )
        )


def _http_sizes(urls: List[str]) -> Dict[str, Optional[int]]:
    """`Content-Length` of each URL (HEAD, redirects followed), concurrently."""

    from concurrent.futures import ThreadPoolExecutor

    def _one(url: str) -> Tuple[str, Optional[int]]:
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"Accept-Encoding": "identity"})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - fixed hub URL
                raw = resp.headers.get("Content-Length")
                return url, int(raw) if raw and raw.isdigit() else None
        except Exception:
            return url, None

    if not urls:
        return {}
    with ThreadPoolExecutor(max_workers=min(8, len(urls))) as pool:
        return dict(pool.map(_one, urls))


def _stream_to_file(
    url: str,
    dest: Path,
    name: str,
    stream: _FileSetProgress,
    control: Any,
) -> Optional[Tuple[str, str]]:
    """Fetch `url` into `dest` via a temp file. None on success, else (status, why)."""

    row = stream.row(name)
    row["state"] = "downloading"
    row["bytes_done"] = 0
    stream.current = name
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.parent / f".{dest.name}.abstractcore-partial"
    response = None
    try:
        req = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
        response = urllib.request.urlopen(req, timeout=_STREAM_READ_TIMEOUT_S)  # noqa: S310 - fixed hub URL
        if control is not None:
            control.on_cancel(lambda r=response: _close_quietly(r))
        length = response.headers.get("Content-Length")
        if length and length.isdigit() and not row.get("bytes_total"):
            row["bytes_total"] = int(length)
        stream.push(DownloadStatus.DOWNLOADING, f"fetching {name}", phase="downloading")
        with open(partial, "wb") as fh:
            while True:
                if control is not None and control.is_cancelled():
                    raise _Cancelled()
                chunk = response.read(_STREAM_CHUNK)
                if not chunk:
                    break
                fh.write(chunk)
                row["bytes_done"] += len(chunk)
                stream.push(DownloadStatus.DOWNLOADING, f"fetching {name}", phase="downloading", force=False)
        if control is not None and control.is_cancelled():
            raise _Cancelled()
        expected = row.get("bytes_total")
        if isinstance(expected, int) and row["bytes_done"] != expected:
            raise OSError(f"{name}: received {row['bytes_done']} of {expected} bytes (connection closed early)")
        os.replace(partial, dest)
        row["state"] = "done"
        stream.push(DownloadStatus.DOWNLOADING, f"fetched {name}", phase="downloading")
        return None
    except _Cancelled:
        row["state"] = "cancelled"
        return "cancelled", "cancelled"
    except Exception as exc:
        if control is not None and control.is_cancelled():
            row["state"] = "cancelled"
            return "cancelled", "cancelled"
        row["state"] = "failed"
        return "failed", f"downloading {name} from {url} failed: {type(exc).__name__}: {exc}"
    finally:
        if response is not None:
            _close_quietly(response)
        if partial.exists():
            try:
                partial.unlink()
            except Exception:
                pass


class _Cancelled(Exception):
    pass


# --- huggingface download ---------------------------------------------------


def _download_huggingface(artifact: str, emit: ProgressCallback, base_url: Optional[str]) -> DownloadOutcome:
    """`snapshot_download`, narrowed to ONE quant when the artifact names one.

    `org/Model-GGUF:Q4_K_M` fetches only the Q4_K_M file(s) (`allow_patterns`);
    without the narrowing a multi-quant GGUF repo downloads EVERY quant --
    often 100+ GB for a model whose chosen file is 5 GB.

    PROGRESS FROM THE FIRST SECOND. The hub's file listing (one metadata
    request, `files_metadata=True`) gives every file's size AND its blob name
    (the LFS sha256, or the git blob id) before the first byte. That buys the
    total, the disk pre-check, and exact PER-FILE progress: huggingface_hub
    writes an in-flight file as `blobs/<etag>.incomplete` and renames it to
    `blobs/<etag>` when whole, so stat-ing those paths every 0.25 s is the
    ground truth, independent of the hub library's version or transport
    (plain HTTP or Xet). The listing also pins the revision, so the files
    watched are exactly the files fetched. Files already whole (an earlier,
    interrupted run) count as done: the download RESUMES.

    CANCEL THAT STOPS. Inside a host job the transfer runs in a child Python
    process (its own process group), so a cancel terminates it at once --
    `snapshot_download` has no cancel API and a thread cannot be killed.
    Outside a job (the CLI in the foreground) it runs in-process as before.

    NEVER "INSTALLED" WHEN PARTIAL. A marker (`.abstractcore-download.json`)
    naming the planned files sits in the repo folder from the first byte until
    every file is verified whole; `probe` reads a repo with a marker whose
    files are not all there as an interrupted download (`absent`), even when a
    cancel landed exactly between two files and no `.incomplete` is left.
    """

    repo_id, quant, patterns = hf_artifact_parts(artifact)
    try:
        import huggingface_hub  # type: ignore

        snapshot_download = huggingface_hub.snapshot_download
    except Exception as exc:
        return DownloadOutcome(
            "huggingface",
            artifact,
            False,
            "failed",
            message=f"huggingface_hub is not installed: {exc}",
            instruction='pip install "abstractcore[huggingface]"',
        )

    # AN EXPLICIT DOWNLOAD REACHES THE HUB. `offline_first` means "never
    # download on-demand while LOADING", never "this process is offline"; an
    # offline flag written in-process after start (a provider's import, a
    # library's load-time override) is not passed to the download child (see
    # `explicit_download_hf_env`). Only a flag the OPERATOR set before start
    # stops it -- and it says so here, not as a bare OfflineModeIsEnabled.
    from .manager import explicit_download_hf_env, operator_forces_hf_offline, operator_hf_offline_env

    forced = operator_forces_hf_offline()
    if forced:
        message = (
            f"{forced}={operator_hf_offline_env().get(forced)} was set in the environment before this process started, "
            f"so the Hugging Face Hub is offline and {repo_id} cannot be downloaded"
        )
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=message))
        return DownloadOutcome(
            "huggingface",
            artifact,
            False,
            "failed",
            message=message,
            command=_planned_command("huggingface", artifact),
            instruction=f"Unset {forced} and restart this process (for the gateway: restart it), then retry.",
        )

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    command = _planned_command("huggingface", artifact)
    emit(
        DownloadProgress(
            status=DownloadStatus.STARTING,
            message=f"reading the file list of {repo_id}" + (f" ({quant} only)" if patterns else ""),
            phase="resolving",
        )
    )
    plan, revision, plan_error = _hf_file_plan(repo_id, patterns, token)
    total: Optional[int] = None
    if plan is not None:
        sizes = [f.get("size") for f in plan]
        total = sum(int(s) for s in sizes if isinstance(s, int)) if all(isinstance(s, int) for s in sizes) else None
    if patterns and plan is not None and not plan:
        message = f"{repo_id} has no file matching {quant} ({', '.join(patterns)})"
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=message))
        return DownloadOutcome(
            "huggingface",
            artifact,
            False,
            "failed",
            message=message,
            command=command,
            instruction=f"Check the quant names on https://huggingface.co/{repo_id}/tree/main",
        )
    shortfall = _disk_shortfall("huggingface", total)
    if shortfall:
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=shortfall))
        return DownloadOutcome(
            "huggingface",
            artifact,
            False,
            "failed",
            message=shortfall,
            command=command,
            instruction="Free disk space (abstractcore models list shows what is installed), then retry.",
        )

    cache_dir = _hf_download_cache_dir()
    repo_dir = cache_dir / ("models--" + repo_id.replace("/", "--"))
    watcher = _HfBlobWatcher(repo_dir, plan, emit, repo_id)
    if plan is None:
        watcher.size_note = f"the hub did not list file sizes ({plan_error})" if plan_error else "the hub did not list file sizes"
    marker = _hf_write_marker(repo_dir, repo_id, revision, plan)

    kwargs: Dict[str, Any] = {"repo_id": repo_id, "token": token, "cache_dir": str(cache_dir)}
    if revision:
        kwargs["revision"] = revision
    if patterns:
        kwargs["allow_patterns"] = list(patterns)

    control = _job_control()
    if control is not None:
        child_env, dropped = explicit_download_hf_env()
        if dropped:
            offline_note = (
                "explicit download: Hub offline flags set in this process after start are not passed to the download ("
                + ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
                + "); offline_first applies to model loading only"
            )
        else:
            offline_note = "explicit download: no Hub offline flag in the download environment"
        _LOG.info("%s: %s", repo_id, offline_note)
        emit(DownloadProgress(status=DownloadStatus.STARTING, message=offline_note, phase="resolving"))
    watcher.push(force=True)
    if control is not None:
        ok, resolved, output = _hf_transfer_subprocess(kwargs, control, watcher, env=child_env)
        # The child is gone (finished or killed): its temp files are orphans.
        _hf_drop_orphan_partials(repo_dir / "blobs", plan)
    else:
        ok, resolved, output = _hf_transfer_inprocess(snapshot_download, kwargs, watcher)

    if control is not None and control.is_cancelled():
        watcher.mark_unfinished("cancelled")
        emit(DownloadProgress(status=DownloadStatus.CANCELLED, message="cancelled"))
        return DownloadOutcome(
            "huggingface",
            artifact,
            False,
            "cancelled",
            message=(
                "cancelled; files that finished downloading stay in the cache and are not fetched again; "
                "the file that was in progress starts over on the next download"
            ),
            command=command,
            output=output,
        )
    if not ok:
        watcher.mark_unfinished("failed")
        message = resolved or "snapshot_download failed"
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=message))
        # The instruction follows from WHAT failed (a dropped connection is
        # not a licence problem); the verbatim error stays in `message`.
        from .host_jobs import failure_reason

        instruction = failure_reason(
            message,
            {"provider": "huggingface", "downloaded_bytes": watcher.bytes_done(), "total_bytes": total},
            output=output,
        )
        return DownloadOutcome(
            "huggingface",
            artifact,
            False,
            "failed",
            message=message,
            output=output or message,
            command=command,
            instruction=instruction,
        )

    watcher.scan()
    missing = watcher.missing()
    emit(
        DownloadProgress(
            status=DownloadStatus.VERIFYING,
            message=f"checking {len(plan or [])} file(s) are whole",
            phase="verifying",
            downloaded_bytes=watcher.bytes_done(),
            total_bytes=total,
            files=watcher.rows(),
        )
    )
    if missing:
        message = f"snapshot_download returned but {len(missing)} planned file(s) are not whole: " + ", ".join(missing[:5])
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=message))
        return DownloadOutcome("huggingface", artifact, False, "failed", message=message, command=command, output=output)
    if marker is not None:
        try:
            marker.unlink()
        except Exception:
            pass

    # The download was PINNED to the listed commit (`revision=<sha>`), and
    # huggingface_hub writes no `refs/main` for a revision that already is a
    # commit hash. Record it now that every planned file is whole, so any
    # loader that resolves the repo by NAME offline (transformers, mlx_lm,
    # vLLM, the operator's own scripts) finds this snapshot.
    ref_state, ref_note = ensure_hf_main_ref(repo_dir, revision)
    if ref_note:
        _LOG.info("%s: %s", repo_id, ref_note)
    completed_message = f"snapshot cached at {resolved}"
    if ref_state in ("written", "kept_other"):
        completed_message += f"; {ref_note}"

    emit(DownloadProgress(status=DownloadStatus.COMPLETE, message=f"cached at {resolved}", percent=100.0))
    return DownloadOutcome(
        "huggingface",
        artifact,
        True,
        "completed",
        message=completed_message,
        location=str(resolved),
        command=command,
    )


_HF_MARKER = ".abstractcore-download.json"
_HF_POLL_S = 0.25


def _hf_download_cache_dir() -> Path:
    """Where `snapshot_download` writes: huggingface_hub's own resolution."""

    try:
        from huggingface_hub import constants  # type: ignore

        return Path(str(constants.HF_HUB_CACHE)).expanduser()
    except Exception:
        explicit = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        if explicit:
            return Path(explicit).expanduser()
        home = os.environ.get("HF_HOME")
        return (Path(home).expanduser() if home else Path.home() / ".cache" / "huggingface") / "hub"


def _hf_file_plan(
    repo_id: str, patterns: Optional[List[str]], token: Optional[str]
) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str], str]:
    """`([{name, size, etag}], revision_sha, "")`, or `(None, None, why)`.

    One metadata request. Failure is not fatal: the download proceeds with
    bytes counted from the blobs folder and `size_unknown` saying why.
    """

    try:
        from huggingface_hub import HfApi  # type: ignore

        info = HfApi().model_info(repo_id, files_metadata=True, token=token)
    except Exception as exc:
        return None, None, str(exc)
    files: List[Dict[str, Any]] = []
    for sibling in getattr(info, "siblings", None) or []:
        name = str(getattr(sibling, "rfilename", "") or "")
        if not name or (patterns and not _matches_any(name, patterns)):
            continue
        lfs = getattr(sibling, "lfs", None)
        sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        if sha256 is None and isinstance(lfs, dict):
            sha256 = lfs.get("sha256")
        etag = sha256 or getattr(sibling, "blob_id", None)
        size = getattr(sibling, "size", None)
        files.append({"name": name, "size": size if isinstance(size, int) else None, "etag": str(etag) if etag else None})
    revision = getattr(info, "sha", None)
    return files, (str(revision) if revision else None), ""


def _hf_remote_total(repo_id: str, patterns: Optional[List[str]], token: Optional[str]) -> Tuple[Optional[int], str]:
    """Exact bytes to fetch (sibling sizes, filtered by `patterns`), or (None, why)."""

    plan, _revision, error = _hf_file_plan(repo_id, patterns, token)
    if plan is None:
        return None, error
    return sum(int(f["size"]) for f in plan if isinstance(f.get("size"), int)), ""


def _hf_write_marker(repo_dir: Path, repo_id: str, revision: Optional[str], plan: Optional[List[Dict[str, Any]]]) -> Optional[Path]:
    try:
        repo_dir.mkdir(parents=True, exist_ok=True)
        marker = repo_dir / _HF_MARKER
        marker.write_text(
            json.dumps(
                {
                    "repo_id": repo_id,
                    "revision": revision,
                    "pid": os.getpid(),
                    "files": [{"name": f["name"], "size": f.get("size"), "etag": f.get("etag")} for f in (plan or [])],
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        return marker
    except Exception:
        return None


def _hf_marker_unfinished(repo_dir: Path) -> Optional[int]:
    """Planned files a download marker says are still missing, or None (no marker).

    A marker whose files are ALL whole is stale (someone else finished the
    job -- `hf download`, a generator's own fetch) and does not count.
    """

    marker = repo_dir / _HF_MARKER
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return 1
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, list) or not files:
        return 1
    missing = 0
    for entry in files:
        etag = entry.get("etag") if isinstance(entry, dict) else None
        size = entry.get("size") if isinstance(entry, dict) else None
        blob = repo_dir / "blobs" / str(etag or "")
        try:
            whole = bool(etag) and blob.is_file() and (not isinstance(size, int) or blob.stat().st_size == size)
        except Exception:
            whole = False
        if not whole:
            missing += 1
    return missing


# --- refs/main for pinned downloads -------------------------------------------
#
# huggingface_hub resolves a repo id to `snapshots/<sha>` through
# `<repo>/refs/<revision>` (a file holding the commit sha, no newline). A
# download pinned to a commit (`snapshot_download(revision=<sha>)`, which is
# what `_download_huggingface` does so the files watched are the files
# fetched) writes NO `refs/main`: the revision already is a hash. The snapshot
# is then complete on disk yet invisible by name to every offline loader that
# asks for "main" -- transformers (`local_files_only=True`), `mlx_lm.load(id)`,
# vLLM, the operator's scripts: "couldn't find them in the cached files".

_HF_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_HF_MODEL_FILE_SUFFIXES = (".safetensors", ".bin", ".gguf", ".npz", ".pt", ".pth", ".onnx", ".msgpack", ".h5")
_HF_MODEL_MARKER_FILES = ("config.json", "adapter_config.json", "model_index.json")


def ensure_hf_main_ref(repo_dir: Path, sha: Optional[str]) -> Tuple[str, str]:
    """Write `<repo_dir>/refs/main` = `sha` when the repo has no `refs/main` yet.

    Returns `(state, note)`; `state` is one of
      "written"    -- the ref was missing and now names `sha`;
      "present"    -- it already names `sha`;
      "kept_other" -- it names ANOTHER commit and is left exactly as found
                      (loading by name keeps resolving that one; `note` says so);
      "skipped"    -- `sha` is not a commit hash (an unpinned download: the hub
                      wrote its own ref) or `snapshots/<sha>` does not exist;
      "error"      -- the write failed (`note` has the reason).
    Never overwrites: the file is created exclusively (`open(..., "x")`), so a
    ref another writer created a moment earlier survives.
    """

    sha = str(sha or "").strip()
    if not _HF_SHA_RE.match(sha):
        return "skipped", ""
    repo_dir = Path(repo_dir)
    if not (repo_dir / "snapshots" / sha).is_dir():
        return "skipped", f"no snapshots/{sha} under {repo_dir}; refs/main not written"
    ref = repo_dir / "refs" / "main"
    for _attempt in range(2):
        try:
            existing = ref.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            existing = None
        except Exception as exc:
            return "error", f"cannot read {ref}: {exc}"
        if existing is not None:
            if existing == sha:
                return "present", ""
            return (
                "kept_other",
                f"refs/main already names {existing[:12]} (not the downloaded {sha[:12]}) and was left unchanged: "
                f"loading {repo_dir.name[len('models--'):].replace('--', '/', 1)} by name resolves {existing[:12]}",
            )
        try:
            ref.parent.mkdir(parents=True, exist_ok=True)
            with open(ref, "x", encoding="utf-8") as fh:
                fh.write(sha)
            return "written", f"wrote refs/main -> {sha[:12]} so the repo loads by name offline"
        except FileExistsError:
            continue  # someone wrote it between our read and our create: re-read, never overwrite
        except Exception as exc:
            return "error", f"cannot write {ref}: {exc}"
    return "error", f"{ref} changed while it was being written"


def _hf_snapshot_is_complete(repo_dir: Path, snapshot: Path) -> Tuple[bool, str]:
    """`(complete, why_not)` for one cached snapshot -- local files only.

    Complete means: the repo has no download in flight or interrupted (no
    `blobs/*.incomplete`, no unfinished `.abstractcore-download.json` marker),
    every file in the snapshot resolves to a blob (no dangling symlink), every
    shard a `*.index.json` weight map names is present, and the snapshot holds
    something a loader can load (a config, a model index, or a weight file) --
    a README-only snapshot is not a load target.
    """

    unfinished = _hf_marker_unfinished(repo_dir)
    if unfinished:
        return False, f"interrupted download ({unfinished} planned file(s) not whole)"
    try:
        if any(p.name.endswith(".incomplete") for p in (repo_dir / "blobs").iterdir()):
            return False, "a download is in flight or interrupted (.incomplete blob)"
    except FileNotFoundError:
        pass
    except Exception as exc:
        return False, f"cannot read blobs/: {exc}"
    files: List[Path] = []
    try:
        for path in snapshot.rglob("*"):
            if path.is_symlink() and not path.exists():
                return False, f"dangling file {path.relative_to(snapshot)}"
            if path.is_file():
                files.append(path)
    except Exception as exc:
        return False, f"cannot read the snapshot: {exc}"
    if not files:
        return False, "empty snapshot"
    names = {p.relative_to(snapshot).as_posix() for p in files}
    for index in (p for p in files if p.name.endswith(".index.json")):
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map") or {}
        except Exception:
            continue
        base = index.parent.relative_to(snapshot).as_posix()
        for shard in sorted(set(str(v) for v in weight_map.values())):
            rel = shard if base in ("", ".") else f"{base}/{shard}"
            if rel not in names:
                return False, f"{index.name} names {shard}, which is missing"
    loadable = any(
        p.name in _HF_MODEL_MARKER_FILES or p.name.lower().endswith(_HF_MODEL_FILE_SUFFIXES) for p in files
    )
    if not loadable:
        return False, "no config or weight file (not a load target)"
    return True, ""


def repair_hf_refs(*, apply: bool = False, cache_dirs: Optional[Iterable[Path]] = None) -> Dict[str, Any]:
    """Find cached model repos with no `refs/main` and, with `apply`, write it.

    A repo is REPAIRABLE when `refs/main` is missing and exactly one of its
    snapshots is complete (`_hf_snapshot_is_complete`); the ref then names that
    snapshot. Two or more complete snapshots are AMBIGUOUS (which one "main"
    was cannot be known offline) and are reported, never guessed. An existing
    `refs/main` is never rewritten -- one that names a missing snapshot is
    reported as `dangling_ref`. Quarantined caches (`model-quarantine`) are not
    scanned. Payload: `{apply, cache_dirs, rows, counts}`; each row has
    `repo_id, repo_path, status, snapshot, reason`, `status` one of
    repairable | repaired | ambiguous | no_complete_snapshot | dangling_ref | error.
    Repos whose `refs/main` is valid are counted (`ok`) but not listed.
    """

    dirs = [Path(d) for d in (cache_dirs if cache_dirs is not None else _hf_cache_dirs())]
    dirs = [d for d in dirs if "model-quarantine" not in d.parts]
    rows: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    seen: set = set()
    for cache_dir in dirs:
        try:
            repos = sorted(p for p in cache_dir.glob("models--*") if p.is_dir())
        except Exception as exc:
            rows.append({"repo_id": None, "repo_path": str(cache_dir), "status": "error", "snapshot": None, "reason": str(exc)})
            continue
        for repo_dir in repos:
            try:
                key = str(repo_dir.resolve())
            except Exception:
                key = str(repo_dir)
            if key in seen:
                continue
            seen.add(key)
            repo_id = repo_dir.name[len("models--"):].replace("--", "/", 1)
            snaps_dir = repo_dir / "snapshots"
            try:
                snapshots = sorted(p for p in snaps_dir.iterdir() if p.is_dir())
            except Exception:
                snapshots = []
            row: Dict[str, Any] = {"repo_id": repo_id, "repo_path": str(repo_dir), "snapshot": None, "reason": ""}
            try:
                existing = (repo_dir / "refs" / "main").read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                existing = None
            except Exception as exc:
                existing = ""
                row.update(status="error", reason=f"cannot read refs/main: {exc}")
            if existing:
                if (snaps_dir / existing).is_dir():
                    counts["ok"] = counts.get("ok", 0) + 1
                    continue
                row.update(status="dangling_ref", reason=f"refs/main names {existing[:12]}, which has no snapshot (left unchanged)")
            elif existing is None:
                complete: List[Path] = []
                why: List[str] = []
                for snap in snapshots:
                    ok, reason = _hf_snapshot_is_complete(repo_dir, snap)
                    if ok:
                        complete.append(snap)
                    else:
                        why.append(f"{snap.name[:12]}: {reason}")
                if len(complete) == 1:
                    sha = complete[0].name
                    row["snapshot"] = sha
                    if not _HF_SHA_RE.match(sha):
                        row.update(status="no_complete_snapshot", reason=f"snapshot folder {sha!r} is not a commit hash")
                    elif apply:
                        state, note = ensure_hf_main_ref(repo_dir, sha)
                        if state == "written":
                            row.update(status="repaired", reason=note)
                        elif state == "present":
                            counts["ok"] = counts.get("ok", 0) + 1
                            continue
                        else:
                            row.update(status="error", reason=note or state)
                    else:
                        row.update(status="repairable", reason=f"one complete snapshot; refs/main would name {sha[:12]}")
                elif len(complete) > 1:
                    row.update(
                        status="ambiguous",
                        reason=f"{len(complete)} complete snapshots ({', '.join(s.name[:12] for s in complete)}); "
                        "nothing written -- delete the stale one(s) or write refs/main by hand",
                    )
                else:
                    row.update(
                        status="no_complete_snapshot",
                        reason="; ".join(why) if why else "no snapshot on disk",
                    )
            counts[row["status"]] = counts.get(row["status"], 0) + 1
            rows.append(row)
    return {"apply": bool(apply), "cache_dirs": [str(d) for d in dirs], "rows": rows, "counts": counts}


class _HfBlobWatcher:
    """Per-file progress from `blobs/<etag>(.incomplete)` sizes."""

    def __init__(self, repo_dir: Path, plan: Optional[List[Dict[str, Any]]], emit: ProgressCallback, repo_id: str):
        self.repo_dir = repo_dir
        self.blobs = repo_dir / "blobs"
        self.emit = emit
        self.repo_id = repo_id
        self.plan = plan
        self.size_note: Optional[str] = None
        self.current: Optional[str] = None
        self._rows: List[Dict[str, Any]] = [
            {"name": f["name"], "bytes_done": 0, "bytes_total": f.get("size"), "state": "pending", "_etag": f.get("etag")}
            for f in (plan or [])
        ]
        self._baseline = self._blob_bytes() if plan is None else 0
        self._growth = 0
        self.scan()

    def _blob_bytes(self) -> int:
        found = 0
        try:
            for path in self.blobs.iterdir():
                try:
                    found += path.stat().st_size
                except Exception:
                    continue
        except Exception:
            return 0
        return found

    def scan(self) -> None:
        if self.plan is None:
            self._growth = max(0, self._blob_bytes() - self._baseline)
            return
        best: Tuple[int, Optional[str]] = (-1, None)
        for row in self._rows:
            etag = row.get("_etag")
            if not etag:
                continue
            whole = self.blobs / str(etag)
            try:
                if whole.is_file():
                    row["bytes_done"] = whole.stat().st_size
                    row["state"] = "done"
                    continue
                partial = _hf_partial_size(self.blobs, str(etag))
                if partial is not None:
                    row["bytes_done"] = partial
                    row["state"] = "downloading"
                    size = row.get("bytes_total") or 0
                    if size > best[0]:
                        best = (size, row["name"])
                elif row["state"] == "downloading":
                    row["state"] = "pending"
            except Exception:
                continue
        if best[1] is not None:
            self.current = best[1]

    def bytes_done(self) -> int:
        if self.plan is None:
            return self._growth
        return sum(int(r.get("bytes_done") or 0) for r in self._rows)

    def total(self) -> Optional[int]:
        if self.plan is None:
            return None
        sizes = [r.get("bytes_total") for r in self._rows]
        return sum(int(s) for s in sizes if isinstance(s, int)) if all(isinstance(s, int) for s in sizes) else None

    def rows(self) -> List[Dict[str, Any]]:
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in self._rows]

    def missing(self) -> List[str]:
        return [r["name"] for r in self._rows if r.get("_etag") and r["state"] != "done"]

    def mark_unfinished(self, state: str) -> None:
        self.scan()
        for row in self._rows:
            if row["state"] in ("downloading", "pending"):
                row["state"] = state

    def push(self, force: bool = False) -> None:
        self.scan()
        total = self.total()
        done = self.bytes_done()
        self.emit(
            DownloadProgress(
                status=DownloadStatus.DOWNLOADING,
                message=f"{self.repo_id}: {format_bytes(done)}" + (f" of {format_bytes(total)}" if total else " fetched"),
                percent=(min(100.0, done / total * 100.0) if total else None),
                downloaded_bytes=done,
                total_bytes=total,
                phase="downloading" if (done > 0 or not force) else None,
                files=self.rows() or None,
                current_file=self.current,
                size_unknown=(total is None) or None,
                size_note=self.size_note,
            )
        )


def _hf_partials(blobs: Path, etag: str) -> List[Path]:
    """In-flight files for one blob, across hub versions.

    huggingface_hub < 1.0 writes `<etag>.incomplete` (and resumes it);
    1.x writes a process-unique `<etag>.<8 hex>.incomplete` and deletes it on
    exit -- unless the process is killed, when it is left behind.
    """

    found: List[Path] = []
    try:
        for path in blobs.glob(f"{etag}*.incomplete"):
            rest = path.name[len(etag):]
            if rest == ".incomplete" or re.fullmatch(r"\.[0-9a-f]{8}\.incomplete", rest):
                found.append(path)
    except Exception:
        pass
    return found


def _hf_partial_size(blobs: Path, etag: str) -> Optional[int]:
    sizes = []
    for path in _hf_partials(blobs, etag):
        try:
            sizes.append(path.stat().st_size)
        except Exception:
            continue
    return max(sizes) if sizes else None


def _hf_drop_orphan_partials(blobs: Path, plan: Optional[List[Dict[str, Any]]]) -> int:
    """Remove the process-unique temp files a KILLED 1.x download left behind.

    They are never resumed (the next run picks a new name), so leaving them
    only wastes disk and makes `probe` report an interrupted download forever.
    The resumable `<etag>.incomplete` of older hubs is kept.
    """

    removed = 0
    for entry in plan or []:
        etag = entry.get("etag")
        if not etag:
            continue
        for path in _hf_partials(blobs, str(etag)):
            if path.name == f"{etag}.incomplete":
                continue
            try:
                path.unlink()
                removed += 1
            except Exception:
                pass
    return removed


def format_bytes(value: Any) -> str:
    from .host_jobs import format_bytes as _fmt

    return _fmt(value)


_HF_CHILD = r"""
import json, os, sys, threading, time
# The child runs in its own session (so a cancel can stop its whole group);
# it must not OUTLIVE its owner: when the gateway/CLI that started it exits
# (restart, crash), the job is reported failed, so the bytes must stop too --
# an orphan kept downloading and held the blob lock a retry then waited on.
# The owner passes its pid (argv[2]): reading getppid() here could already
# see the re-parented value if the owner died during this interpreter's start.
_owner = int(sys.argv[2]) if len(sys.argv) > 2 else os.getppid()
def _watch_owner():
    while True:
        time.sleep(1.0)
        if os.getppid() != _owner:
            os._exit(75)  # no goodbye line: the owner's end of the pipe is gone
threading.Thread(target=_watch_owner, daemon=True).start()
from huggingface_hub import snapshot_download
kwargs = json.loads(sys.argv[1])
path = snapshot_download(**kwargs)
print("ABSTRACTCORE_RESOLVED=" + str(path), flush=True)
"""


def _hf_transfer_subprocess(
    kwargs: Dict[str, Any],
    control: Any,
    watcher: _HfBlobWatcher,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str, str]:
    """Run `snapshot_download` in a child process; watch the blobs while it runs.

    Returns `(ok, resolved_path_or_error, output)`. The token travels in the
    environment (inherited), never on the command line. `env` defaults to
    `explicit_download_hf_env()`: this process's environment with the HF
    offline flags reset to what the operator set before start.
    """

    import sys
    import time as _time

    from .manager import explicit_download_hf_env

    child_kwargs = {k: v for k, v in kwargs.items() if k != "token"}
    env = dict(env) if env is not None else explicit_download_hf_env()[0]
    env.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    env["PYTHONUNBUFFERED"] = "1"
    # PLAIN HTTP, SO THE BYTES ARE VISIBLE. With `hf_xet` a file is rebuilt
    # from chunks and lands in `blobs/<etag>.incomplete` only at the end: a
    # 143 MB safetensors sat at "4.8 MB of 148 MB" and then jumped to 100 %
    # (measured, huggingface_hub 1.26 + hf_xet), which the stall detector
    # would rightly call a stall and a cancel could not target. Over HTTP the
    # `.incomplete` grows byte by byte and resumes with a Range request.
    # `ABSTRACTCORE_HF_XET=1` opts back into Xet (progress then moves only as
    # whole files land).
    if str(os.environ.get("ABSTRACTCORE_HF_XET") or "").strip().lower() not in ("1", "true", "yes", "on"):
        env["HF_HUB_DISABLE_XET"] = "1"
    try:
        proc = subprocess.Popen(  # noqa: S603 - fixed module code, JSON argv
            [sys.executable, "-c", _HF_CHILD, json.dumps(child_kwargs), str(os.getpid())],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            env=env,
            start_new_session=os.name != "nt",
        )
    except Exception as exc:
        return False, f"could not start the Hugging Face download process: {exc}", ""
    try:
        proc._abstractcore_own_group = os.name != "nt"  # type: ignore[attr-defined]
    except Exception:
        pass
    control.register_process(proc)
    lines: List[str] = []

    def _drain() -> None:
        assert proc.stdout is not None
        for raw in proc.stdout:
            text = raw.rstrip("\n")
            if text.strip():
                lines.append(text)

    reader = threading.Thread(target=_drain, daemon=True)
    reader.start()
    while proc.poll() is None:
        watcher.push()
        _time.sleep(_HF_POLL_S)
    reader.join(timeout=5)
    watcher.push()
    output = "\n".join(lines)
    resolved = next((l.split("=", 1)[1] for l in reversed(lines) if l.startswith("ABSTRACTCORE_RESOLVED=")), "")
    if proc.returncode == 0 and resolved:
        return True, resolved, output
    # The child's own last words ARE the reason (rule 4): the exception line.
    reason = next((l.strip() for l in reversed(lines) if l.strip() and not l.startswith(" ")), "")
    return False, reason or f"the Hugging Face download process exited {proc.returncode}", output


def _hf_transfer_inprocess(snapshot_download: Callable[..., Any], kwargs: Dict[str, Any], watcher: _HfBlobWatcher) -> Tuple[bool, str, str]:
    """Foreground (no job): `snapshot_download` on this thread, a watcher beside it."""

    stop = threading.Event()

    def _watch() -> None:
        while not stop.wait(_HF_POLL_S):
            watcher.push()

    thread = threading.Thread(target=_watch, daemon=True)
    thread.start()
    try:
        resolved = snapshot_download(**kwargs)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", str(exc)
    finally:
        stop.set()
        thread.join(timeout=2)
    watcher.push()
    return True, str(resolved), ""


_DOWNLOADERS: Dict[str, Callable[[str, ProgressCallback, Optional[str]], DownloadOutcome]] = {
    "lmstudio": _download_lmstudio,
    "ollama": _download_ollama,
    "supertonic": _download_supertonic,
}


def _run_streaming(
    cmd: List[str],
    provider: str,
    artifact: str,
    emit: ProgressCallback,
    *,
    env: Optional[Dict[str, str]] = None,
    parser: Any = None,
    graceful_cancel: Optional[Callable[[subprocess.Popen], None]] = None,
) -> DownloadOutcome:
    """Run a provider CLI and forward its lines verbatim (rule 4).

    Inside a host job the process is registered with the job's control, so a
    cancel terminates it; the outcome is then `cancelled`, not `failed`.

    Output is read as raw bytes and split on BOTH `\\n` and `\\r`: a progress
    bar redraws itself with `\\r` and never ends a line, so a line reader sees
    nothing until the tool exits -- exactly how `lms get` looked frozen.
    A `parser` (`feed(segment) -> DownloadProgress | None`) turns a tool's
    redraws into byte progress; without one each line is a message. With
    `graceful_cancel`, a cancel asks the tool to stop its own way first (the
    callback owns escalation) instead of an immediate SIGTERM.
    """

    lines: List[str] = []
    control = _job_control()
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv, never a shell string
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # Inside a host job nobody is at a terminal: a tool that stops to
            # ask (a sudo password, a y/N prompt) must fail fast, not hang --
            # unless the caller answers the tool's cancel prompt itself.
            stdin=(subprocess.PIPE if graceful_cancel is not None else subprocess.DEVNULL) if control is not None else None,
            env=env,
            # In a job the tool gets its own process group, so a cancel stops
            # the whole tree (brew/pip/installer children included).
            start_new_session=bool(control is not None and os.name != "nt"),
        )
    except Exception as exc:
        return DownloadOutcome(
            provider,
            artifact,
            False,
            "failed",
            message=f"could not run {cmd[0]}: {exc}",
            command=list(cmd),
        )
    if control is not None:
        try:
            proc._abstractcore_own_group = os.name != "nt"  # type: ignore[attr-defined]
        except Exception:
            pass
        if graceful_cancel is not None:
            control.on_cancel(lambda: threading.Thread(target=graceful_cancel, args=(proc,), daemon=True).start())
        else:
            control.register_process(proc)
    assert proc.stdout is not None

    def _segment(raw: str) -> None:
        text = _ANSI_RE.sub("", raw).strip()
        if not text:
            return
        matched = False
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            update = parser.feed(line) if parser is not None else None
            if update is not None:
                matched = True
                emit(update)
                continue
            if not lines or lines[-1] != line:
                lines.append(line)
            follow = parser.line(line) if parser is not None and hasattr(parser, "line") else None
            emit(follow or DownloadProgress(status=DownloadStatus.DOWNLOADING, message=line))
        if parser is not None and not matched and "\n" in text and "%" in text:
            # A redraw whose text wrapped over lines: try it as one.
            update = parser.feed(" ".join(text.split()))
            if update is not None:
                emit(update)

    buffer = ""
    fd = proc.stdout.fileno()
    while True:
        try:
            chunk = os.read(fd, 65536)
        except OSError:
            chunk = b""
        if not chunk:
            break
        buffer += chunk.decode("utf-8", errors="replace")
        cut = max(buffer.rfind("\r"), buffer.rfind("\n"))
        if cut < 0:
            continue
        ready, buffer = buffer[: cut + 1], buffer[cut + 1 :]
        for piece in re.split(r"\r\n|\r", ready):
            _segment(piece)
    if buffer:
        _segment(buffer)
    code = proc.wait()
    output = "\n".join(lines)
    if control is not None and control.is_cancelled():
        emit(DownloadProgress(status=DownloadStatus.CANCELLED, message="cancelled"))
        return DownloadOutcome(
            provider,
            artifact,
            False,
            "cancelled",
            message=f"cancelled ({cmd[0]} stopped)",
            output=output,
            command=list(cmd),
        )
    if code != 0:
        # Rule 4: the tool's own last words are the reason, not just a code.
        last = next((l for l in reversed(lines) if l.strip()), "")
        message = f"{cmd[0]} exited {code}" + (f": {last}" if last else "")
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=message))
        return DownloadOutcome(
            provider,
            artifact,
            False,
            "failed",
            message=message,
            output=output,
            command=list(cmd),
            instruction=f"Run `{' '.join(cmd)}` yourself to see the tool's full prompt.",
        )
    verb = "installed" if provider == "engine" else "downloaded"
    emit(DownloadProgress(status=DownloadStatus.COMPLETE, message=f"{provider}: {artifact} {verb}", percent=100.0))
    return DownloadOutcome(
        provider,
        artifact,
        True,
        "completed",
        message=f"{artifact} {verb}",
        output=output,
        command=list(cmd),
    )


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b[@-_]")

_SIZE_UNITS = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4}
# `lms get`'s bar, as its own `ProgressBar.refresh()` + `createDownloadPbUpdater`
# draw it (ANSI stripped): `⠋ [████▌   ] 12.34% | 1.23 GB / 4.80 GB | 38.00 MB/s | ETA 01:34`.
# Sizes are `formatSizeBytes1000` (decimal units).
_LMS_BAR_RE = re.compile(
    r"(?P<pct>\d+(?:\.\d+)?)%"
    r"(?:\s*\|\s*(?P<done>\d+(?:\.\d+)?)\s*(?P<du>[KMGT]?B)\s*/\s*(?P<total>\d+(?:\.\d+)?)\s*(?P<tu>[KMGT]?B)"
    r"\s*\|\s*(?P<speed>\d+(?:\.\d+)?)\s*(?P<su>[KMGT]?B)/s(?:\s*\|\s*ETA\s*(?P<eta>[\d:]+))?)?"
)


def _size(value: str, unit: str) -> int:
    return int(float(value) * _SIZE_UNITS.get(unit.upper(), 1))


class _LmsGetProgress:
    """`lms get` output -> `DownloadProgress`.

    `lms get` draws a progress bar on stdout (no TTY check) with
    `downloadedBytes / totalBytes | speed/s | ETA`, from LM Studio's own
    `onProgress` updates; `Finalizing download...` marks the move into the
    library. Everything else it prints is a message line.
    """

    def __init__(self) -> None:
        self.seen_bytes = False
        self.total: Optional[int] = None

    def feed(self, text: str) -> Optional[DownloadProgress]:
        match = None
        for match in _LMS_BAR_RE.finditer(text):
            pass
        if match is None or "[" not in text:
            return None
        pct = float(match.group("pct"))
        if match.group("done") is None:
            return DownloadProgress(status=DownloadStatus.DOWNLOADING, message="lms get: starting the transfer", percent=pct, phase="downloading" if pct > 0 else "resolving")
        done = _size(match.group("done"), match.group("du"))
        total = _size(match.group("total"), match.group("tu"))
        self.seen_bytes = True
        self.total = total
        return DownloadProgress(
            status=DownloadStatus.DOWNLOADING,
            message=f"lms get: {match.group('done')} {match.group('du')} of {match.group('total')} {match.group('tu')}",
            percent=pct,
            downloaded_bytes=done,
            total_bytes=total,
            phase="downloading",
            size_unknown=False,
        )

    def line(self, line: str) -> Optional[DownloadProgress]:
        low = line.lower()
        if "finalizing" in low:
            return DownloadProgress(status=DownloadStatus.DOWNLOADING, message=line, phase="installing")
        if "download completed" in low:
            return DownloadProgress(status=DownloadStatus.DOWNLOADING, message=line, phase="installing")
        if not self.seen_bytes:
            # "Searching for models...", "Downloading <name>": still resolving
            # until the first byte count arrives.
            return DownloadProgress(status=DownloadStatus.STARTING, message=line, phase="resolving")
        return None


def _lms_graceful_cancel(proc: subprocess.Popen) -> None:
    """Stop `lms get` the way its own Ctrl-C does, so LM Studio stops too.

    On SIGINT `lms get` asks "Continue to download in the background?"; `N`
    aborts the download in LM Studio ("Download canceled."), while `Y` -- or
    simply killing the CLI -- leaves LM Studio downloading on its own. So:
    SIGINT, answer `N`, and only then escalate (SIGTERM, then SIGKILL).
    """

    import signal
    import time as _time

    try:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            _time.sleep(0.3)
            if proc.stdin is not None:
                try:
                    proc.stdin.write(b"N\n")
                    proc.stdin.flush()
                except Exception:
                    pass
        for _ in range(30):
            if proc.poll() is not None:
                return
            _time.sleep(0.1)
        _terminate_group(proc, signal.SIGTERM)
        for _ in range(50):
            if proc.poll() is not None:
                return
            _time.sleep(0.1)
        _terminate_group(proc, signal.SIGKILL)
    except Exception:
        pass


def _terminate_group(proc: subprocess.Popen, sig: int) -> None:
    try:
        if os.name != "nt":
            os.killpg(proc.pid, sig)
        else:  # pragma: no cover - windows only
            proc.kill()
    except Exception:
        try:
            proc.send_signal(sig)
        except Exception:
            pass


class _LmStudioDiskWatch:
    """The fallback when `lms get` shows no bar: bytes landing in LM Studio's library.

    LM Studio's REST download status is not reachable from `lms get`; when
    the CLI prints no progress, the honest number is what is on disk: files
    under the models root modified since the download began.
    """

    def __init__(self, parser: _LmsGetProgress, emit: ProgressCallback, expected_bytes: Optional[int] = None):
        import time as _time

        self.parser = parser
        self.emit = emit
        self.root = _lmstudio_models_root()
        self.since = _time.time() - 1.0
        self.expected = expected_bytes if isinstance(expected_bytes, int) and expected_bytes > 0 else None
        self.stop = threading.Event()

    def bytes_on_disk(self) -> int:
        found = 0
        try:
            for dirpath, _dirs, names in os.walk(self.root):
                for name in names:
                    try:
                        st = os.stat(os.path.join(dirpath, name))
                    except OSError:
                        continue
                    if st.st_mtime >= self.since:
                        found += st.st_size
        except Exception:
            return 0
        return found

    def run(self) -> None:
        while not self.stop.wait(1.0):
            if self.parser.seen_bytes:
                continue
            got = self.bytes_on_disk()
            note = f"LM Studio reports no progress; {format_bytes(got)} on disk so far"
            self.emit(
                DownloadProgress(
                    status=DownloadStatus.DOWNLOADING,
                    message=note,
                    downloaded_bytes=got,
                    total_bytes=self.expected,
                    phase="downloading" if got > 0 else None,
                    size_unknown=self.expected is None,
                    size_note=None if self.expected else "LM Studio reports no progress",
                )
            )



# ---------------------------------------------------------------------------
# The recommended journey
# ---------------------------------------------------------------------------


def recommended_downloads(host: Optional[Dict[str, Any]] = None) -> List[Dict[str, str]]:
    """This host's recommended downloads as a stable list, route key included.

    `capability_defaults.recommended_model_downloads()`: the portable set, with
    the text row following the Apple-silicon memory tiers on a Mac.
    """

    out: List[Dict[str, str]] = []
    for route_key, spec in recommended_model_downloads(host).items():
        provider = str(spec.get("provider") or "").strip()
        artifact = str(spec.get("artifact") or "").strip()
        if not provider or not artifact:
            continue
        out.append({"route": route_key, "provider": provider, "artifact": artifact})
    return out


def recommended_plan(*, base_urls: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """What `--recommended` WOULD fetch, and what is already here.

    This is the payload behind both the fresh-install banner and the dry run:
    one probe per recommended artifact, no downloads, no hub contact.
    """

    from .model_catalog import recommended_text_model

    urls = {k.lower(): v for k, v in (base_urls or {}).items()}
    # The text row says WHY it is the pick and whether it fits this host
    # (`recommended_text_model`): a tier the fit estimate doubts stays the
    # tier, with its `warning` for the surface to show.
    text_pick = recommended_text_model()
    rows: List[Dict[str, Any]] = []
    with presence_sweep():
        for item in recommended_downloads():
            presence = probe(item["provider"], item["artifact"], base_url=urls.get(item["provider"].lower()))
            row = dict(item)
            row.update(presence.to_dict())
            if item["route"] == "input.text" and (item["provider"], item["artifact"]) == (text_pick["provider"], text_pick["artifact"]):
                row.update(
                    catalog_id=text_pick["catalog_id"],
                    basis=text_pick["basis"],
                    tier=text_pick["tier"],
                    fit_verdict=(text_pick.get("fit") or {}).get("verdict"),
                    fits=text_pick["fits"],
                    warning=text_pick["warning"],
                )
            rows.append(row)
    missing = [r for r in rows if r.get("status") == PRESENCE_ABSENT]
    unknown = [r for r in rows if r.get("status") == PRESENCE_UNKNOWN]
    present = [r for r in rows if r.get("status") == PRESENCE_INSTALLED]
    return {
        "recommended": rows,
        "total": len(rows),
        "installed": len(present),
        "absent": len(missing),
        "unknown": len(unknown),
        "would_download": [{"provider": r["provider"], "artifact": r["artifact"], "route": r["route"]} for r in missing],
    }


def route_is_answered(row: Any) -> bool:
    """Does this capability-default row have something serving it, by ANY lane?

    A value of its own is the obvious lane. The other three come from the
    hierarchy decoration (`manager._decorate_route_hierarchy`): `covered_by`
    (the text model handles this input modality), `covered_by_tasks` (a parent
    whose task rows are all set) and `inherits_broad` (a task row whose parent
    is set). All four mean the same thing to a recommendation: there is nothing
    here to fix.
    """

    if not isinstance(row, dict):
        return False
    if row.get("provider") and row.get("model"):
        return True
    return bool(row.get("covered_by") or row.get("covered_by_tasks") or row.get("inherits_broad"))


def mark_recommended_route_gaps(plan: Dict[str, Any], routes: Iterable[Any]) -> Dict[str, Any]:
    """Split a `recommended_plan()` into ADVICE and GAPS, in place.

    THE STARTER KIT IS ADVICE FOR AN EMPTY ROUTE, NOT A STANDING DEBT.
    `recommended_plan()` answers exactly one question -- "is the fresh-install
    model on this disk?" -- and every surface that rendered that answer raw told
    an operator who had deliberately routed `input.text` at their own model that
    a model was MISSING, with a download command to run. It could not be
    cleared except by installing the model they had chosen against, so it never
    cleared; a status line that cries wolf on a healthy host teaches an operator
    to stop reading it.

    `gaps` is the subset of `would_download` whose route nothing else answers --
    the only part a surface may present as work to do. The counts and
    `would_download` are left exactly as they were: `--dry-run` and
    `models download --recommended` ask what the recommendation WOULD fetch,
    which is a different question with a different right answer.
    """

    if not isinstance(plan, dict):
        return plan
    answered = {
        str(row.get("key") or "").strip().lower()
        for row in routes
        if isinstance(row, dict) and route_is_answered(row)
    }
    answered.discard("")

    def _mark(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        item["route_answered"] = str(item.get("route") or "").strip().lower() in answered
        return bool(item["route_answered"])

    for item in plan.get("recommended") or []:
        _mark(item)
    plan["gaps"] = [dict(item) for item in (plan.get("would_download") or []) if not _mark(item)]
    plan["routes_unanswered"] = len(plan["gaps"])
    return plan


# ---------------------------------------------------------------------------
# Grid annotation -- the shape every console renders
# ---------------------------------------------------------------------------


def annotate_route_availability(routes: Iterable[Any]) -> List[Dict[str, Any]]:
    """Annotate capability-default rows with weight availability.

    Input is the route-row shape `list_capability_defaults()` produces (dicts
    with `key`/`provider`/`model`). Output is one row per input row carrying
    `availability` (a `ModelPresence` dict) plus, when the route is one of the
    recommended three, the `download_artifact` that would actually fetch it --
    which is NOT the row's model id whenever quantization is pinned.

    A row with no provider/model is left at `unknown` with no instruction: an
    unconfigured route has no weights to be missing.
    """

    recommended_by_route = {item["route"]: item for item in recommended_downloads()}
    # `output.text` is the canonical read of the `input.text` storage key; the
    # recommendation is stored under the latter, so both rows answer.
    text_alias = recommended_by_route.get("input.text")

    rows_in = [dict(raw) for raw in (routes or []) if isinstance(raw, dict)]

    def _recommendation_for(key: str) -> Optional[Dict[str, str]]:
        return recommended_by_route.get(key) or (text_alias if key == "output.text" else None)

    # A COVERED ROW FETCHES WHAT ITS COVERING ROW FETCHES. `input.image` served
    # by the text model is the same weights as `input.text`, so it must resolve
    # to the same artifact -- including the quantization. Resolving it on its
    # own produced `lms get qwen/qwen3.5-9b` next to `lms get
    # qwen/qwen3.5-9b@4bit` for one set of files, i.e. two instructions for one
    # download, one of them naming no quant at all.
    covering_artifact: Dict[str, str] = {}
    for row in rows_in:
        key = str(row.get("key") or "").strip()
        rec = _recommendation_for(key)
        if rec and _norm(row.get("provider")) == _norm(rec["provider"]) and _matches_installed_id(
            row.get("model"), rec["artifact"]
        ):
            covering_artifact[key] = rec["artifact"]

    seen_cache: Dict[Tuple[str, str, str], ModelPresence] = {}
    out: List[Dict[str, Any]] = []
    with presence_sweep():
        for row in rows_in:
            key = str(row.get("key") or "").strip()
            provider = str(row.get("provider") or "").strip()
            model = str(row.get("model") or "").strip()
            base_url = str(row.get("base_url") or "").strip() or None

            rec = _recommendation_for(key)
            artifact = model
            if rec and _matches_installed_id(model, rec["artifact"]) and _norm(provider) == _norm(rec["provider"]):
                # The route stores the served id; the recommendation names the
                # exact weights. Fetch what the recommendation names.
                artifact = rec["artifact"]
            else:
                covered_by = str(row.get("covered_by") or row.get("derived_from") or "").strip()
                inherited = covering_artifact.get(covered_by)
                if inherited and _matches_installed_id(model, inherited):
                    artifact = inherited

            if not provider or not artifact:
                row["availability"] = ModelPresence(
                    provider, artifact, PRESENCE_UNKNOWN, evidence="route not configured"
                ).to_dict()
                out.append(row)
                continue

            cache_key = (_provider_id(provider), _norm(artifact), base_url or "")
            presence = seen_cache.get(cache_key)
            if presence is None:
                presence = probe(provider, artifact, base_url=base_url)
                seen_cache[cache_key] = presence
            row["availability"] = presence.to_dict()
            if artifact != model:
                row["download_artifact"] = artifact
            if rec:
                row["recommended_artifact"] = rec["artifact"]
            out.append(row)
    return out


def route_key_for(kind: Any, modality: Any, task: Any = None) -> str:
    """Re-exported so surfaces build route keys the one supported way."""

    return capability_route_key(kind, modality, task)


# ---------------------------------------------------------------------------
# Job hooks (cooperative cancel) -- see `config.host_jobs`
# ---------------------------------------------------------------------------


def _job_control() -> Any:
    """The running host job's cancel handle, or None outside a job."""

    try:
        from .host_jobs import current_job_control

        return current_job_control()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Hugging Face artifact references: `org/repo`, `org/repo:QUANT`
# ---------------------------------------------------------------------------

_GGUF_QUANT_RE = re.compile(r"^(?:ud[-_])?(?:i?q\d[\w]*|f16|bf16|f32|fp16|mxfp4)$", re.IGNORECASE)


def hf_artifact_parts(artifact: Any) -> Tuple[str, Optional[str], Optional[List[str]]]:
    """`"unsloth/Qwen3-8B-GGUF:Q4_K_M"` -> `(repo_id, "Q4_K_M", ["*Q4_K_M*.gguf", ...])`.

    The `repo:QUANT` form is how the catalog names ONE GGUF quant inside a
    multi-quant repo (Hugging Face repo ids never contain `:`). The `@quant`
    form (`split_artifact`) is accepted too. Only GGUF-shaped quants produce
    `allow_patterns`; an MLX repo is one quant already and is fetched whole.
    """

    raw = str(artifact or "").strip()
    quant: Optional[str] = None
    repo = raw
    head, sep, tail = raw.partition(":")
    if sep and "/" in head and tail.strip():
        repo, quant = head.strip(), tail.strip()
    else:
        repo, quant = split_artifact(raw)
    patterns: Optional[List[str]] = None
    if quant and _GGUF_QUANT_RE.match(quant):
        variants: List[str] = []
        for v in (quant, quant.upper(), quant.lower()):
            if v not in variants:
                variants.append(v)
        patterns = [f"*{v}*.gguf" for v in variants]
    return repo, quant, patterns


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    import fnmatch

    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _snapshot_has_matching_file(snapshot: Path, patterns: List[str]) -> bool:
    try:
        for path in snapshot.rglob("*"):
            rel = path.relative_to(snapshot).as_posix()
            if _matches_any(rel, patterns) and path.exists():
                return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------
# list_installed(): every model on this machine, per engine, with sizes
# ---------------------------------------------------------------------------

MODELS_INSTALLED_SCHEMA = "models_installed_v1"
INSTALLED_PROVIDERS: Tuple[str, ...] = ("ollama", "lmstudio", "mlx", "huggingface")

_BLOCKER_LOADED = "loaded"
_BLOCKER_REMOTE = "remote_engine"
_BLOCKER_SHARED = "shared_cache:mlx,huggingface"
_BLOCKER_NOT_RUNNING = "engine_not_running"
_BLOCKER_UNKNOWN_LOCATION = "unknown_location"


def _display(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        rel = Path(path).resolve().relative_to(Path.home().resolve())
        return "~/" + rel.as_posix()
    except Exception:
        return str(path)


def _installed_row(
    provider: str,
    artifact: str,
    *,
    quant: Optional[str] = None,
    size_bytes: Optional[int] = None,
    params_total: Optional[int] = None,
    location: Optional[str] = None,
    loaded: Optional[bool] = None,
    blockers: Optional[List[str]] = None,
    deletable: bool = True,
    **extra: Any,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "provider": provider,
        "artifact": artifact,
        "quant": (str(quant).lower() if quant else None),
        "size_bytes": int(size_bytes) if isinstance(size_bytes, int) else None,
        "params_total": int(params_total) if isinstance(params_total, int) else None,
        "location": location,
        "loaded": loaded,
        "catalog_id": None,
        "deletable": bool(deletable),
        "delete_blockers": list(blockers or []),
        # What the artifact IS and what it can DO, from LOCAL evidence only
        # (`_hf_local_kind_tasks`, the engine's own listing). `kind`:
        # model | adapter | encoder | embedding, or None when the local files
        # cannot say; `tasks` is [] then. `tasks_source` names the evidence.
        "kind": None,
        "tasks": [],
        "tasks_source": None,
    }
    row.update(extra)
    return row


def _parse_params(value: Any) -> Tuple[Optional[int], Optional[int]]:
    try:
        from ..utils.model_fit import parse_param_count, parse_params_from_name
    except Exception:  # pragma: no cover
        return None, None
    text = str(value or "").strip()
    if not text:
        return None, None
    total, active = parse_params_from_name("x-" + text.replace(" ", ""))
    if total is None:
        total = parse_param_count(text)
    return total, active


# --- ollama -----------------------------------------------------------------


def _ollama_models_dir() -> Path:
    try:
        from ..utils.host_profile import default_model_store_paths

        return Path(default_model_store_paths()["ollama"])
    except Exception:
        return Path.home() / ".ollama" / "models"


def _ollama_manifest_rows(models_dir: Path) -> List[Dict[str, Any]]:
    """Installed tags read from Ollama's on-disk manifests (server not running).

    Layout: `manifests/<registry>/<namespace>/<model>/<tag>`; the library
    namespace on the default registry is the bare `model:tag` users type.
    """

    rows: List[Dict[str, Any]] = []
    root = models_dir / "manifests"
    try:
        files = [p for p in root.rglob("*") if p.is_file()]
    except Exception:
        return rows
    for path in files:
        try:
            parts = path.relative_to(root).parts
        except Exception:
            continue
        if len(parts) < 4 or parts[-1].startswith("."):
            continue
        registry, namespace, model, tag = parts[0], "/".join(parts[1:-2]), parts[-2], parts[-1]
        if registry == "registry.ollama.ai" and namespace == "library":
            name = f"{model}:{tag}"
        elif registry == "registry.ollama.ai":
            name = f"{namespace}/{model}:{tag}"
        else:
            name = f"{registry}/{namespace}/{model}:{tag}"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            size = sum(int(layer.get("size") or 0) for layer in manifest.get("layers") or [])
            size += int((manifest.get("config") or {}).get("size") or 0)
        except Exception:
            size = None
        total, _active = _parse_params(tag)
        rows.append(
            _installed_row(
                "ollama",
                name,
                size_bytes=size,
                params_total=total,
                location=_display(models_dir),
                loaded=None,
                blockers=[_BLOCKER_NOT_RUNNING],
                deletable=False,
                size_source="manifest",
            )
        )
    rows.sort(key=lambda r: r["artifact"])
    return rows


def _ollama_loaded_names(host: str) -> Optional[set]:
    payload, _error = _http_json(f"{host}/api/ps")
    if not isinstance(payload, dict):
        return None
    names = set()
    for row in payload.get("models") or []:
        if isinstance(row, dict):
            for key in ("name", "model"):
                value = str(row.get(key) or "").strip().lower()
                if value:
                    names.add(value)
    return names


def _installed_ollama(base_url: Optional[str], include_loaded: bool) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    host = _ollama_base_url(base_url)
    local = _ollama_is_this_machine(host)
    payload, error = _http_json(f"{host}/api/tags")
    if not isinstance(payload, dict):
        if local:
            rows = _ollama_manifest_rows(_ollama_models_dir())
            return rows, f"unreachable ({error}); listed {len(rows)} tag(s) from on-disk manifests, delete needs the server"
        return [], f"unreachable ({error})"
    loaded = _ollama_loaded_names(host) if include_loaded else None
    rows: List[Dict[str, Any]] = []
    for item in payload.get("models") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("model") or "").strip()
        if not name:
            continue
        details = item.get("details") if isinstance(item.get("details"), dict) else {}
        total, _active = _parse_params(details.get("parameter_size"))
        is_loaded: Optional[bool] = None
        if loaded is not None:
            is_loaded = any(_ollama_tag_match(n, name) for n in loaded)
        blockers: List[str] = []
        if is_loaded:
            blockers.append(_BLOCKER_LOADED)
        if not local:
            blockers.append(_BLOCKER_REMOTE)
        size = item.get("size")
        rows.append(
            _installed_row(
                "ollama",
                name,
                quant=details.get("quantization_level") or None,
                size_bytes=size if isinstance(size, int) else None,
                params_total=total,
                location=_display(_ollama_models_dir()) if local else host,
                loaded=is_loaded,
                blockers=blockers,
                size_source="engine",
                format=details.get("format") or None,
                family=details.get("family") or None,
                digest=item.get("digest") or None,
                modified_at=item.get("modified_at") or None,
            )
        )
    return rows, None


# --- lmstudio ---------------------------------------------------------------


def _lms_listing() -> Tuple[Optional[List[Dict[str, Any]]], str]:
    return _cached_listing("lms:ls:full", _read_lms_listing)


def _read_lms_listing() -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """The full `lms ls --json` rows (not just ids), or why we have none."""

    cli = _lms_cli()
    if not cli:
        return None, "the `lms` CLI is not installed"
    try:
        proc = subprocess.run([cli, "ls", "--json"], capture_output=True, text=True, timeout=_CLI_PROBE_TIMEOUT)
    except Exception as exc:
        return None, f"`lms ls` failed: {exc}"
    if proc.returncode != 0:
        return None, f"`lms ls` exited {proc.returncode}: {(proc.stderr or '').strip()[:200]}"
    try:
        payload = json.loads(proc.stdout or "[]")
    except Exception as exc:
        return None, f"`lms ls --json` returned unparseable output: {exc}"
    if not isinstance(payload, list):
        return None, f"`lms ls --json` returned a {type(payload).__name__}, not a list of models"
    return [row for row in payload if isinstance(row, dict)], ""


def _lms_loaded_keys() -> Optional[set]:
    """Identifiers `lms ps --json` reports as loaded; None when it cannot tell."""

    cli = _lms_cli()
    if not cli:
        return None
    try:
        proc = subprocess.run([cli, "ps", "--json"], capture_output=True, text=True, timeout=_CLI_PROBE_TIMEOUT)
        payload = json.loads(proc.stdout or "[]") if proc.returncode == 0 else None
    except Exception:
        return None
    if not isinstance(payload, list):
        return None
    keys = set()
    for row in payload:
        if not isinstance(row, dict):
            continue
        for key in ("identifier", "modelKey", "path", "indexedModelIdentifier"):
            value = str(row.get(key) or "").strip().lower()
            if value:
                keys.add(value)
    return keys


def _lmstudio_models_root() -> Path:
    try:
        from ..utils.model_cache import default_lmstudio_model_dirs

        dirs = default_lmstudio_model_dirs()
        if dirs:
            return Path(dirs[0])
    except Exception:
        pass
    return Path.home() / ".lmstudio" / "models"


def _lmstudio_hub_root() -> Path:
    return Path.home() / ".lmstudio" / "hub" / "models"


def _ci_child(parent: Path, name: str) -> Optional[Path]:
    direct = parent / name
    if direct.exists():
        return direct
    try:
        for entry in parent.iterdir():
            if entry.name.lower() == name.lower():
                return entry
    except Exception:
        return None
    return None


def _ci_path(root: Path, rel: str) -> Optional[Path]:
    current = root
    for part in [p for p in str(rel).replace("\\", "/").split("/") if p]:
        nxt = _ci_child(current, part)
        if nxt is None:
            return None
        current = nxt
    return current if current != root else None


def _lmstudio_locate(row: Dict[str, Any]) -> Tuple[Optional[Path], List[Path]]:
    """`(weights_path, extra_paths)` for one `lms ls` row.

    `path` is either a location under the models root (a repo directory, or a
    single `.gguf` file inside a multi-quant repo) or an LM Studio HUB id
    (`qwen/qwen3.8-27b`) whose manifest under `~/.lmstudio/hub/models` names
    the Hugging Face repo that holds the weights. Hub models return the
    manifest directory as an extra path, so a delete removes the entry too.
    """

    ref = str(row.get("path") or "").strip()
    root = _lmstudio_models_root()
    if ref:
        found = _ci_path(root, ref)
        if found is not None:
            return found, []
        hub_dir = _ci_path(_lmstudio_hub_root(), ref)
        manifest = hub_dir / "manifest.json" if hub_dir is not None else None
        if manifest is not None and manifest.is_file():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:
                data = {}
            for dep in data.get("dependencies") or []:
                for source in (dep or {}).get("sources") or []:
                    if (source or {}).get("type") != "huggingface":
                        continue
                    user, repo = source.get("user"), source.get("repo")
                    if user and repo:
                        weights = _ci_path(root, f"{user}/{repo}")
                        if weights is not None:
                            return weights, [hub_dir]
            return None, [hub_dir]
    return None, []


def _lmstudio_row_artifact(row: Dict[str, Any]) -> str:
    return str(row.get("selectedVariant") or row.get("modelKey") or row.get("path") or "").strip()


def _lmstudio_row_matches(row: Dict[str, Any], artifact: str) -> bool:
    candidates = [
        row.get("selectedVariant"),
        row.get("modelKey"),
        row.get("indexedModelIdentifier"),
        row.get("path"),
        *(row.get("variants") or []),
    ]
    return any(_matches_installed_id(c, artifact) for c in candidates if isinstance(c, str) and c.strip())


def _installed_lmstudio(include_loaded: bool) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    listing, error = _lms_listing()
    if listing is None:
        return [], error or "the `lms` CLI is unavailable"
    loaded_keys = _lms_loaded_keys() if include_loaded else None
    rows: List[Dict[str, Any]] = []
    for item in listing:
        artifact = _lmstudio_row_artifact(item)
        if not artifact:
            continue
        quantization = item.get("quantization") if isinstance(item.get("quantization"), dict) else {}
        total, active = _parse_params(item.get("paramsString"))
        weights, extra = _lmstudio_locate(item)
        is_loaded: Optional[bool] = None
        if loaded_keys is not None:
            names = {str(item.get(k) or "").strip().lower() for k in ("modelKey", "path", "indexedModelIdentifier")}
            names.discard("")
            is_loaded = bool(names & loaded_keys)
        blockers: List[str] = []
        if is_loaded:
            blockers.append(_BLOCKER_LOADED)
        if weights is None:
            blockers.append(_BLOCKER_UNKNOWN_LOCATION)
        size = item.get("sizeBytes")
        rows.append(
            _installed_row(
                "lmstudio",
                artifact,
                quant=quantization.get("name") or None,
                size_bytes=size if isinstance(size, int) else None,
                params_total=total,
                location=_display(weights),
                loaded=is_loaded,
                blockers=blockers,
                deletable=weights is not None,
                size_source="engine",
                params_active=active,
                model_key=item.get("modelKey"),
                lms_path=item.get("path"),
                type=item.get("type"),
                format=item.get("format"),
                display_name=item.get("displayName"),
                max_context=item.get("maxContextLength"),
                **_lmstudio_kind_tasks(item),
            )
        )
    return rows, None


# --- huggingface cache (mlx + huggingface) ------------------------------------


def _classify_hf_repo(repo_id: str, repo_path: Optional[Path]) -> str:
    try:
        from ..providers.mlx_model_rules import is_mlx_model

        return "mlx" if is_mlx_model(repo_id, local_path=repo_path) else "huggingface"
    except Exception:
        return "mlx" if "mlx" in repo_id.lower() else "huggingface"


_NAME_QUANT_RE = re.compile(r"(?:^|[-_.])(\d+bit|[34568]bit|fp8|bf16|fp16|mxfp4|q\d_k_[msl]|q\d_\d|q8_0)(?:$|[-_.])", re.IGNORECASE)


def _hf_repos() -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """`[{repo_id, repo_path, size_bytes, revisions, nb_files, cache_dir}]` for model repos."""

    repos: List[Dict[str, Any]] = []
    try:
        from huggingface_hub import scan_cache_dir  # type: ignore
    except Exception:
        scan_cache_dir = None  # type: ignore[assignment]
    error: Optional[str] = None
    for cache_dir in _hf_cache_dirs():
        if scan_cache_dir is not None:
            try:
                info = scan_cache_dir(cache_dir)
            except Exception as exc:
                error = f"scan_cache_dir({cache_dir}) failed: {exc}"
                continue
            for repo in info.repos:
                if getattr(repo, "repo_type", "model") != "model":
                    continue
                repos.append(
                    {
                        "repo_id": repo.repo_id,
                        "repo_path": Path(repo.repo_path),
                        "size_bytes": int(repo.size_on_disk),
                        "revisions": [rev.commit_hash for rev in repo.revisions],
                        "nb_files": int(repo.nb_files),
                        "cache_dir": Path(cache_dir),
                    }
                )
            continue
        # No huggingface_hub: the cache layout is simple enough to read.
        try:
            for folder in Path(cache_dir).glob("models--*"):
                repo_id = folder.name[len("models--"):].replace("--", "/", 1)
                size = sum(p.stat().st_size for p in (folder / "blobs").glob("*") if p.is_file())
                revs = [p.name for p in (folder / "snapshots").glob("*") if p.is_dir()]
                repos.append(
                    {
                        "repo_id": repo_id,
                        "repo_path": folder,
                        "size_bytes": size,
                        "revisions": revs,
                        "nb_files": None,
                        "cache_dir": Path(cache_dir),
                    }
                )
        except Exception as exc:
            error = f"cache scan of {cache_dir} failed: {exc}"
    return repos, error


# Hub `pipeline_tag` -> AbstractCore task names. An unmapped tag passes
# through with `-` -> `_` rather than being dropped.
_PIPELINE_TASKS: Dict[str, Tuple[str, ...]] = {
    "text-generation": ("text_generation",),
    "text2text-generation": ("text_generation",),
    "image-text-to-text": ("text_generation", "image_to_text"),
    "image-to-text": ("image_to_text",),
    "feature-extraction": ("text_embedding",),
    "sentence-similarity": ("text_embedding",),
    "automatic-speech-recognition": ("speech_to_text",),
    "text-to-speech": ("text_to_speech",),
    "text-to-image": ("text_to_image",),
    "image-to-image": ("image_to_image",),
    "text-to-video": ("text_to_video",),
    "image-to-video": ("image_to_video",),
}
_ENCODER_MODEL_TYPES = frozenset(
    {"bert", "roberta", "xlm-roberta", "distilbert", "deberta", "deberta-v2", "electra", "modernbert", "nomic_bert", "mpnet", "camembert", "albert"}
)
_ENCODER_PIPELINES = frozenset(
    {"fill-mask", "text-classification", "token-classification", "zero-shot-classification", "image-feature-extraction", "image-classification"}
)
_CARD_READ_BYTES = 64 * 1024


def _hf_card_front_matter(path: Path) -> Dict[str, Any]:
    """`pipeline_tag` / `library_name` / `tags` from a README's YAML front
    matter, read with a tiny line parser (no YAML dependency, first 64 KB)."""

    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read(_CARD_READ_BYTES)
    except Exception:
        return {}
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    out: Dict[str, Any] = {"tags": []}
    in_tags = False
    for line in lines[1:]:
        if line.strip() == "---":
            break
        stripped = line.strip()
        if in_tags and stripped.startswith("- "):
            out["tags"].append(stripped[2:].strip().strip("'\"").lower())
            continue
        in_tags = False
        key, sep, value = line.partition(":")
        if not sep or line[:1].isspace():
            continue
        key, value = key.strip(), value.strip().strip("'\"")
        if key == "tags":
            if value.startswith("[") and value.endswith("]"):
                out["tags"].extend(v.strip().strip("'\"").lower() for v in value[1:-1].split(",") if v.strip())
            else:
                in_tags = True
        elif key in ("pipeline_tag", "library_name") and value:
            out[key] = value.lower()
    return out


def _pipeline_tasks(tag: str) -> List[str]:
    return list(_PIPELINE_TASKS.get(tag) or (tag.replace("-", "_"),))


def _hf_snapshot_for_kind(repo_path: Optional[Path]) -> Optional[Path]:
    """The snapshot `refs/main` points at, else the newest one on disk."""

    if repo_path is None:
        return None
    snaps = repo_path / "snapshots"
    try:
        ref = (repo_path / "refs" / "main").read_text(encoding="utf-8").strip()
        if ref and (snaps / ref).is_dir():
            return snaps / ref
    except Exception:
        pass
    try:
        dirs = [p for p in snaps.iterdir() if p.is_dir()]
    except Exception:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime) if dirs else None


def _hf_local_kind_tasks(repo_path: Optional[Path]) -> Dict[str, Any]:
    """`kind` / `tasks` / `tasks_source` for a cached HF repo -- LOCAL FILES ONLY.

    No card fetch: the model card counts only if `README.md` was downloaded
    (mlx-lm's own fetch skips it). Evidence, strongest first:
      kind  -- `adapter_config.json`, or a LoRA-tagged card / `*lora*.safetensors`
               with no base config -> adapter; sentence-transformers files or an
               embedding pipeline tag -> embedding; an encoder-only architecture
               -> encoder; a generative architecture, `model_index.json` or a
               card pipeline tag -> model.
      tasks -- the card's `pipeline_tag` (else a pipeline name among its tags);
               else `config.json` (`*ForCausalLM` -> text_generation, `+
               vision_config` -> image_to_text, whisper -> speech_to_text).
    Whatever the files cannot say stays `None` / `[]`.
    """

    unknown: Dict[str, Any] = {"kind": None, "tasks": [], "tasks_source": None}
    snap = _hf_snapshot_for_kind(repo_path)
    if snap is None:
        return unknown
    try:
        names = {p.name for p in snap.iterdir()}
    except Exception:
        return unknown
    card = _hf_card_front_matter(snap / "README.md") if "README.md" in names else {}
    tags = [t for t in card.get("tags") or [] if isinstance(t, str)]
    config: Dict[str, Any] = {}
    if "config.json" in names:
        try:
            loaded = json.loads((snap / "config.json").read_text(encoding="utf-8"))
            config = loaded if isinstance(loaded, dict) else {}
        except Exception:
            config = {}
    archs = [str(a) for a in (config.get("architectures") or []) if isinstance(a, str)]
    model_type = str(config.get("model_type") or "").lower()

    tasks: List[str] = []
    source: Optional[str] = None
    pipeline = card.get("pipeline_tag")
    if not pipeline:
        pipeline = next((t for t in tags if t in _PIPELINE_TASKS), None)
    if pipeline:
        tasks, source = _pipeline_tasks(pipeline), "model_card"

    generative = any(a.endswith(("ForCausalLM", "LMHeadModel", "ForConditionalGeneration", "ForSpeechSeq2Seq")) for a in archs)
    encoder = model_type in _ENCODER_MODEL_TYPES or (
        bool(archs) and not generative and all(a.endswith(("Model", "ForMaskedLM", "ForSequenceClassification", "ForTokenClassification")) for a in archs)
    )
    sentence_tf = bool({"modules.json", "config_sentence_transformers.json"} & names) or card.get("library_name") == "sentence-transformers"
    lora_weights = any(n.lower().endswith(".safetensors") and "lora" in n.lower() for n in names)

    kind: Optional[str] = None
    if "adapter_config.json" in names:
        kind = "adapter"
        if not tasks:
            try:
                adapter = json.loads((snap / "adapter_config.json").read_text(encoding="utf-8"))
            except Exception:
                adapter = {}
            if isinstance(adapter, dict) and str(adapter.get("task_type") or "").upper() == "CAUSAL_LM":
                tasks, source = ["text_generation"], "adapter_config"
    elif ("lora" in tags or lora_weights) and "config.json" not in names and "model_index.json" not in names:
        kind = "adapter"
    elif sentence_tf or pipeline in ("feature-extraction", "sentence-similarity"):
        kind = "embedding"
        if not tasks:
            tasks, source = ["text_embedding"], "files"
    elif encoder and (not pipeline or pipeline in _ENCODER_PIPELINES):
        # A generative pipeline tag outranks the architecture: a diffusers or
        # audio repo's top-level config often names its (encoder) sub-model.
        kind = "encoder"
    elif generative or encoder or pipeline or "model_index.json" in names or any(n.lower().endswith(".gguf") for n in names):
        kind = "model"

    if not tasks and kind == "model" and archs:
        if model_type == "whisper" or any(a.endswith("ForSpeechSeq2Seq") for a in archs):
            tasks = ["speech_to_text"]
        elif generative:
            tasks = ["text_generation"] + (["image_to_text"] if isinstance(config.get("vision_config"), dict) else [])
        if tasks:
            source = "config"
    return {"kind": kind, "tasks": tasks, "tasks_source": source}


def _lmstudio_kind_tasks(item: Dict[str, Any]) -> Dict[str, Any]:
    """From `lms ls --json` itself: `type` is llm | embedding, `vision` a bool."""

    kind_type = str(item.get("type") or "").strip().lower()
    if kind_type == "embedding":
        return {"kind": "embedding", "tasks": ["text_embedding"], "tasks_source": "engine"}
    if kind_type == "llm":
        tasks = ["text_generation"] + (["image_to_text"] if item.get("vision") is True else [])
        return {"kind": "model", "tasks": tasks, "tasks_source": "engine"}
    return {"kind": None, "tasks": [], "tasks_source": None}


def _installed_hf(wanted: Iterable[str]) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    wanted_set = set(wanted)
    repos, error = _hf_repos()
    rows: List[Dict[str, Any]] = []
    for repo in repos:
        if not repo["revisions"]:
            continue  # a resolved-but-empty entry holds no weights
        provider = _classify_hf_repo(repo["repo_id"], repo["repo_path"])
        if provider not in wanted_set:
            continue
        total, active = _parse_params(repo["repo_id"].rsplit("/", 1)[-1])
        m = _NAME_QUANT_RE.search(repo["repo_id"].rsplit("/", 1)[-1])
        interrupted, _bytes, _where = _hf_interrupted_downloads(repo["repo_id"])
        rows.append(
            _installed_row(
                provider,
                repo["repo_id"],
                quant=m.group(1) if m else None,
                size_bytes=repo["size_bytes"],
                params_total=total,
                location=_display(repo["repo_path"]),
                loaded=None,
                blockers=[],
                size_source="engine",
                params_active=active,
                revisions=len(repo["revisions"]),
                nb_files=repo["nb_files"],
                incomplete_files=interrupted,
                **_hf_local_kind_tasks(repo["repo_path"]),
            )
        )
    rows.sort(key=lambda r: (r["provider"], r["artifact"].lower()))
    return _attach_companion_rows(rows), error


def _attach_companion_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A companion is listed WITH its model, never as a stray row.

    Every MLX row gets `companions: [{artifact, role, installed, size_bytes,
    location, ...}]` (the registry's companions for it, installed or not).
    A cached companion used by at least one listed model disappears from the
    top level; one whose models are all gone stays, marked `role:
    "mtp_companion"` with `companion_of: []`, so it can still be found and
    deleted. A companion shared by several models is attached to each.
    """

    from ..providers.speculation import mlx_registry_drafters

    drafters = mlx_registry_drafters()
    by_id = {_norm(r["artifact"]): r for r in rows}
    used: Dict[str, List[str]] = {}
    for row in rows:
        if row["provider"] != "mlx" or _norm(row["artifact"]) in drafters:
            continue
        attached = []
        for comp in companion_artifacts("mlx", row["artifact"]):
            found = by_id.get(_norm(comp))
            entry = dict(found) if found is not None else {"provider": "mlx", "artifact": comp, "size_bytes": None, "location": None}
            entry.update(role=COMPANION_ROLE, installed=found is not None)
            attached.append(entry)
            if found is not None:
                used.setdefault(_norm(comp), []).append(row["artifact"])
        row["companions"] = attached
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = _norm(row["artifact"])
        if key in used:
            continue  # shown under its model(s)
        if key in drafters:
            row = dict(row, role=COMPANION_ROLE, companion_of=[])
        out.append(row)
    for row in out:
        for entry in row.get("companions") or []:
            entry["companion_of"] = list(used.get(_norm(entry["artifact"]), []))
    return out


def _annotate_catalog_ids(rows: List[Dict[str, Any]]) -> None:
    try:
        from .model_catalog import catalog_id_for
    except Exception:
        return
    for row in rows:
        try:
            row["catalog_id"] = catalog_id_for(row["provider"], row["artifact"])
        except Exception:
            row["catalog_id"] = None


def list_installed(
    provider: Optional[str] = None,
    *,
    base_urls: Optional[Dict[str, str]] = None,
    include_loaded: bool = True,
) -> Dict[str, Any]:
    """Contract D (`models_installed_v1`): every installed model, with sizes.

    One row per artifact an engine holds on this machine: Ollama tags
    (`/api/tags`: `size`, `details.parameter_size`, `details.quantization_level`),
    LM Studio models (`lms ls --json`: `sizeBytes`, `paramsString`,
    `quantization.name`), and the Hugging Face cache split into `mlx` and
    `huggingface` rows by `mlx_model_rules.is_mlx_model`. An engine that cannot
    be read lands in `errors`, never as an empty "nothing installed".
    """

    from ..utils.host_profile import utc_now_iso

    urls = {k.lower(): v for k, v in (base_urls or {}).items()}
    pid = _provider_id(provider) if provider else ""
    wanted = [pid] if pid else list(INSTALLED_PROVIDERS)
    rows: List[Dict[str, Any]] = []
    errors: Dict[str, str] = {}
    probed: List[str] = []

    with presence_sweep():
        for name in wanted:
            if name not in INSTALLED_PROVIDERS:
                if _is_relay(name):
                    errors[name] = "remote engine: models are served remotely, nothing is installed here"
                else:
                    errors[name] = (
                        f"no installed-model listing for {name!r}; supported: " + ", ".join(INSTALLED_PROVIDERS)
                    )
                continue
            if name in {"mlx", "huggingface"}:
                if "mlx" in probed or "huggingface" in probed:
                    continue
                hf_wanted = [n for n in wanted if n in {"mlx", "huggingface"}]
                got, err = _installed_hf(hf_wanted)
                probed.extend(hf_wanted)
                if err:
                    for n in hf_wanted:
                        errors[n] = err
            elif name == "ollama":
                got, err = _installed_ollama(urls.get("ollama"), include_loaded)
                probed.append(name)
                if err:
                    errors[name] = err
            else:
                got, err = _installed_lmstudio(include_loaded)
                probed.append(name)
                if err:
                    errors[name] = err
            rows.extend(got)

    _annotate_catalog_ids(rows)
    return {
        "schema": MODELS_INSTALLED_SCHEMA,
        "rows": rows,
        "engines_probed": probed,
        "errors": errors,
        "totals": {
            "count": len(rows),
            # A companion shared by two models is on disk once: count each once.
            "size_bytes": sum(r["size_bytes"] for r in rows if isinstance(r.get("size_bytes"), int))
            + sum(
                c["size_bytes"]
                for c in {
                    _norm(c["artifact"]): c
                    for r in rows
                    for c in (r.get("companions") or [])
                    if c.get("installed") and isinstance(c.get("size_bytes"), int)
                }.values()
            ),
        },
        "generated_at": utc_now_iso(),
    }


# ---------------------------------------------------------------------------
# delete_artifact(): the one delete verb
# ---------------------------------------------------------------------------


def _find_installed_row(provider: str, artifact: str, base_url: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    payload = list_installed(provider, base_urls={provider: base_url} if base_url else None)
    rows = payload["rows"]
    error = payload["errors"].get(provider)
    if provider == "ollama":
        hit = next((r for r in rows if _ollama_tag_match(r["artifact"], artifact)), None)
    elif provider == "lmstudio":
        hit = next((r for r in rows if _norm(r["artifact"]) == _norm(artifact)), None)
        if hit is None:
            listing, _ = _lms_listing()
            for item in listing or []:
                if _lmstudio_row_matches(item, artifact):
                    hit = next((r for r in rows if r["artifact"] == _lmstudio_row_artifact(item)), None)
                    break
    else:
        repo_id, _quant, _patterns = hf_artifact_parts(artifact)
        hit = next((r for r in rows if _norm(r["artifact"]) == _norm(repo_id)), None)
    return hit, error


def delete_blockers(provider: Any, artifact: Any, *, base_url: Optional[str] = None) -> Dict[str, Any]:
    """What would stop a delete: `{found, row, delete_blockers, error}`.

    Surfaces call this BEFORE starting a delete job so a refusal is an
    immediate answer (HTTP 409 / exit 2) naming the blockers, not a job that
    fails a second later.
    """

    pid = _provider_id(provider)
    ref = str(artifact or "").strip()
    if _is_relay(pid):
        return {"found": False, "row": None, "delete_blockers": [_BLOCKER_REMOTE], "error": f"{pid} serves models remotely"}
    lookup = pid
    if pid in _HF_BACKED:
        lookup = "__hf__"
    if lookup == "__hf__":
        payload = list_installed(None)
        repo_id, _q, _p = hf_artifact_parts(ref)
        rows = [r for r in payload["rows"] if r["provider"] in {"mlx", "huggingface"}]
        rows = rows + [
            {k: v for k, v in c.items() if k not in ("installed",)}
            for r in rows
            for c in (r.get("companions") or [])
            if c.get("installed")
        ]
        row = next((r for r in rows if _norm(r["artifact"]) == _norm(repo_id)), None)
        error = payload["errors"].get("huggingface") or payload["errors"].get("mlx")
        if row is not None and row["provider"] != pid and pid in {"mlx", "huggingface"}:
            row = dict(row)
            row["delete_blockers"] = list(row["delete_blockers"]) + [_BLOCKER_SHARED]
        elif row is not None and pid not in {"mlx", "huggingface"}:
            # mlx-gen / diffusers / mlx-vlm read the same cache.
            row = dict(row)
            row["delete_blockers"] = list(row["delete_blockers"]) + [f"shared_cache:{pid},{row['provider']}"]
    elif pid in {"ollama", "lmstudio"}:
        row, error = _find_installed_row(pid, ref, base_url)
    else:
        return {"found": False, "row": None, "delete_blockers": [], "error": f"no delete verb for provider {pid!r}"}
    return {
        "found": row is not None,
        "row": row,
        "delete_blockers": list((row or {}).get("delete_blockers") or []),
        "error": error,
    }


def _safe_remove(path: Path, roots: Iterable[Path]) -> Tuple[bool, str]:
    """Remove a file or directory ONLY when it sits strictly inside a known root."""

    target = Path(path)
    try:
        resolved = target.resolve()
    except Exception as exc:
        return False, f"cannot resolve {target}: {exc}"
    inside = False
    for root in roots:
        try:
            root_resolved = Path(root).resolve()
        except Exception:
            continue
        if resolved != root_resolved and root_resolved in resolved.parents:
            inside = True
            break
    if not inside:
        return False, f"refusing to delete {target}: not inside a known model store"
    try:
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        else:
            return False, f"{target} does not exist"
    except Exception as exc:
        return False, f"could not delete {target}: {exc}"
    return True, ""


def _http_request(url: str, method: str, body: Dict[str, Any], timeout: float = 30.0) -> Tuple[Optional[int], str]:
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - local engine daemon
            return int(response.status), response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        try:
            text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            text = ""
        return int(exc.code), text
    except Exception as exc:
        return None, str(exc)


def delete_artifact(
    provider: Any,
    artifact: Any,
    *,
    dry_run: bool = False,
    force: bool = False,
    base_url: Optional[str] = None,
    with_companions: Optional[bool] = None,
) -> Dict[str, Any]:
    """Delete one installed artifact with the engine's own mechanism.

    COMPANIONS (an MLX model's MTP drafter, see `companion_artifacts`): a
    model's delete OFFERS its companions -- `companion_offer: [{artifact,
    size_bytes, shared_with, command}]` -- and removes them only when
    `with_companions=True`. A companion another installed model still uses
    (`shared_with` non-empty) is never removed with this one.

      ollama       `DELETE /api/delete` (a loaded model is unloaded first with --force)
      lmstudio     `lms unload` when loaded (--force), then remove the model's
                   files under the LM Studio models root (+ its hub manifest)
      mlx / huggingface   `scan_cache_dir().delete_revisions(...).execute()`

    REFUSES (status `refused`, `delete_blockers` named) when the model is
    loaded, lives on a remote engine, or sits in the shared Hugging Face cache
    under another engine's classification -- unless `force`. Never removes a
    path outside a known model store. `dry_run` reports what would go.
    """

    pid = _provider_id(provider)
    ref = str(artifact or "").strip()
    out: Dict[str, Any] = {
        "provider": pid,
        "artifact": ref,
        "ok": False,
        "status": "failed",
        "message": "",
        "freed_bytes": None,
        "paths": [],
        "command": [],
        "delete_blockers": [],
        "forced": bool(force),
        "dry_run": bool(dry_run),
    }
    if not pid or not ref:
        out["message"] = "a provider and an artifact are required"
        return out
    if _is_relay(pid):
        out.update(status="not_applicable", message=f"{pid} serves models remotely; there is nothing to delete here")
        return out

    check = delete_blockers(pid, ref, base_url=base_url)
    row = check.get("row")
    if row is None:
        detail = check.get("error")
        out.update(
            status="not_found",
            message=f"{ref} is not installed for {pid}" + (f" ({detail})" if detail else ""),
        )
        return out
    blockers = list(check.get("delete_blockers") or [])
    out["delete_blockers"] = blockers
    out["freed_bytes"] = row.get("size_bytes")
    out["location"] = row.get("location")
    hard = [b for b in blockers if b in {_BLOCKER_UNKNOWN_LOCATION, _BLOCKER_NOT_RUNNING}]
    if hard:
        out.update(status="refused", message=f"cannot delete {ref}: " + ", ".join(hard))
        return out
    if blockers and not force:
        out.update(
            status="refused",
            message=f"refusing to delete {ref}: " + ", ".join(blockers) + " (pass force to override)",
        )
        return out

    if pid == "ollama":
        return _delete_ollama(row, out, base_url, dry_run)
    if pid == "lmstudio":
        return _delete_lmstudio(row, out, dry_run)
    out = _delete_hf(row, out, dry_run)
    return _companion_delete(row, out, dry_run=dry_run, with_companions=with_companions)


def _companion_delete(row: Dict[str, Any], out: Dict[str, Any], *, dry_run: bool, with_companions: Optional[bool]) -> Dict[str, Any]:
    """Offer (or, when asked, remove) a deleted model's companions."""

    installed = [c for c in (row.get("companions") or []) if c.get("installed")]
    if not installed or out.get("status") not in ("deleted", "planned"):
        return out
    offer = []
    for comp in installed:
        shared = [m for m in (comp.get("companion_of") or []) if _norm(m) != _norm(row["artifact"])]
        offer.append(
            {
                "artifact": comp["artifact"],
                "role": COMPANION_ROLE,
                "size_bytes": comp.get("size_bytes"),
                "shared_with": shared,
                "command": cli_equivalent_delete_companion(comp["artifact"]),
            }
        )
    out["companion_offer"] = offer
    if not with_companions:
        return out
    removed = []
    for entry in offer:
        if entry["shared_with"]:
            removed.append({"artifact": entry["artifact"], "ok": False, "status": "kept",
                            "message": f"kept: still used by {', '.join(entry['shared_with'])}"})
            continue
        sub = {"provider": "mlx", "artifact": entry["artifact"], "ok": False, "status": "failed", "message": "",
               "freed_bytes": entry.get("size_bytes"), "paths": [], "command": []}
        removed.append(_delete_hf({"artifact": entry["artifact"]}, sub, dry_run))
    out["companions_deleted"] = removed
    freed = [r.get("freed_bytes") for r in removed if r.get("ok")]
    if isinstance(out.get("freed_bytes"), int):
        out["freed_bytes"] = out["freed_bytes"] + sum(int(f) for f in freed if isinstance(f, int))
    names = [r["artifact"] for r in removed if r.get("ok")]
    if names:
        out["message"] = f"{out.get('message') or ''}; " + ("would also delete" if dry_run else "also deleted") + " its MTP companion " + ", ".join(names)
    return out


def cli_equivalent_delete_companion(artifact: str) -> str:
    return f"abstractcore models delete mlx {artifact} --yes"


def _delete_ollama(row: Dict[str, Any], out: Dict[str, Any], base_url: Optional[str], dry_run: bool) -> Dict[str, Any]:
    host = _ollama_base_url(base_url)
    name = row["artifact"]
    out["command"] = ["DELETE", f"{host}/api/delete", name]
    out["paths"] = [row.get("location")] if row.get("location") else []
    if dry_run:
        out.update(ok=True, status="planned", message=f"would delete {name} from Ollama at {host}")
        return out
    if row.get("loaded"):
        # Unload first: `keep_alive: 0` is Ollama's documented unload.
        _http_request(f"{host}/api/generate", "POST", {"model": name, "keep_alive": 0})
    code, text = _http_request(f"{host}/api/delete", "DELETE", {"model": name, "name": name})
    if code == 200:
        out.update(ok=True, status="deleted", message=f"deleted {name} from Ollama")
    elif code == 404:
        out.update(status="not_found", message=f"Ollama has no model {name}: {text.strip()[:200]}")
    else:
        out.update(status="failed", message=f"Ollama DELETE /api/delete returned {code}: {text.strip()[:300]}")
    return out


def _delete_lmstudio(row: Dict[str, Any], out: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    listing, _error = _lms_listing()
    item = next((i for i in listing or [] if _lmstudio_row_artifact(i) == row["artifact"]), None)
    if item is None:
        out.update(status="not_found", message=f"{row['artifact']} is no longer listed by `lms ls`")
        return out
    weights, extra = _lmstudio_locate(item)
    if weights is None:
        out.update(status="refused", message=f"cannot locate the files of {row['artifact']} under {_lmstudio_models_root()}")
        return out
    paths = [weights] + list(extra)
    out["paths"] = [str(p) for p in paths]
    cli = _lms_cli() or "lms"
    model_key = str(item.get("modelKey") or row["artifact"])
    commands: List[List[str]] = []
    if row.get("loaded"):
        commands.append([cli, "unload", model_key])
    out["command"] = [c for cmd in commands for c in cmd] or ["rm", "-r", str(weights)]
    if dry_run:
        out.update(ok=True, status="planned", message=f"would remove {', '.join(out['paths'])}")
        return out
    for cmd in commands:
        try:
            subprocess.run(cmd, capture_output=True, text=True, timeout=60)  # noqa: S603 - fixed argv
        except Exception as exc:
            out.update(status="failed", message=f"`{' '.join(cmd)}` failed: {exc}")
            return out
    roots = [_lmstudio_models_root(), _lmstudio_hub_root()]
    removed: List[str] = []
    for path in paths:
        ok, why = _safe_remove(path, roots)
        if not ok:
            out.update(status="failed", message=why, paths=removed or out["paths"])
            return out
        removed.append(str(path))
        # A single-quant file removed from a multi-file repo dir: drop the dir
        # only when nothing but OS litter is left.
        parent = path.parent
        try:
            leftovers = [p for p in parent.iterdir() if p.name not in {".DS_Store"}]
            if path.suffix == ".gguf" and not leftovers:
                _safe_remove(parent, [_lmstudio_models_root()])
        except Exception:
            pass
    out.update(ok=True, status="deleted", message=f"removed {row['artifact']} ({len(removed)} path(s))")
    return out


def _delete_hf(row: Dict[str, Any], out: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    repo_id = row["artifact"]
    repos, _error = _hf_repos()
    repo = next((r for r in repos if _norm(r["repo_id"]) == _norm(repo_id)), None)
    if repo is None:
        out.update(status="not_found", message=f"{repo_id} is not in the Hugging Face cache")
        return out
    out["paths"] = [str(repo["repo_path"])]
    out["command"] = ["huggingface_hub.scan_cache_dir().delete_revisions", *repo["revisions"]]
    try:
        from huggingface_hub import scan_cache_dir  # type: ignore
    except Exception:
        scan_cache_dir = None  # type: ignore[assignment]
    if scan_cache_dir is not None:
        try:
            info = scan_cache_dir(repo["cache_dir"])
            strategy = info.delete_revisions(*repo["revisions"])
            out["freed_bytes"] = int(getattr(strategy, "expected_freed_size", 0) or 0) or out.get("freed_bytes")
        except Exception as exc:
            out.update(status="failed", message=f"cannot plan the cache delete: {exc}")
            return out
        if dry_run:
            out.update(ok=True, status="planned", message=f"would delete {repo_id} from {repo['cache_dir']}")
            return out
        try:
            strategy.execute()
        except Exception as exc:
            out.update(status="failed", message=f"cache delete failed: {exc}")
            return out
        # The strategy deletes revisions; a repo whose revisions are all gone
        # can keep `.incomplete` blobs -- remove the folder when that is all.
        if Path(repo["repo_path"]).exists():
            _safe_remove(Path(repo["repo_path"]), [repo["cache_dir"]])
        out.update(ok=True, status="deleted", message=f"deleted {repo_id} from the Hugging Face cache")
        return out
    out["command"] = ["rm", "-r", str(repo["repo_path"])]
    if dry_run:
        out.update(ok=True, status="planned", message=f"would delete {repo['repo_path']}")
        return out
    ok, why = _safe_remove(Path(repo["repo_path"]), [repo["cache_dir"]])
    out.update(ok=ok, status="deleted" if ok else "failed", message=why or f"deleted {repo_id}")
    return out
