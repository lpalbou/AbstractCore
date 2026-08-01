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
    RECOMMENDED_MODEL_DOWNLOADS,
    capability_route_key,
)

# Reuse, do not reinvent: `abstractcore.download` already defines the progress
# vocabulary the async download API speaks. The materializer is the SYNC lane
# (a CLI streaming lines, a Gateway background job posting updates), and it
# speaks the same words so a surface that renders one renders the other.
from ..download import DownloadProgress, DownloadStatus

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
            return _probe_huggingface(pid, artifact)
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
        return [p for p in (Path.home() / ".cache" / "huggingface" / "hub",) if p.is_dir()]


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
        try:
            if not blobs.is_dir():
                continue
            for path in blobs.iterdir():
                if not path.name.endswith(".incomplete"):
                    continue
                count += 1
                where = where or blobs
                try:
                    size += path.stat().st_size
                except Exception:
                    pass
        except Exception:
            continue
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


def _probe_huggingface(provider: str, artifact: str) -> ModelPresence:
    repo_id, _quant = split_artifact(artifact)
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
) -> DownloadOutcome:
    """Fetch one artifact with the provider's own tool. Only on explicit request.

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
        return _download(pid, ref, emit, base_url=base_url, dry_run=dry_run)
    finally:
        _sweep.reset(_outer_sweep)


def _download(
    pid: str,
    ref: str,
    emit: ProgressCallback,
    *,
    base_url: Optional[str] = None,
    dry_run: bool = False,
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

    if dry_run:
        return DownloadOutcome(
            pid,
            ref,
            True,
            "planned",
            message=f"would download {ref} with {pid}",
            command=_planned_command(pid, ref),
        )

    emit(DownloadProgress(status=DownloadStatus.STARTING, message=f"{pid}: fetching {ref}"))
    try:
        return handler(ref, emit, base_url)
    except Exception as exc:  # pragma: no cover - handlers convert their own failures
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=str(exc)))
        return DownloadOutcome(pid, ref, False, "failed", message=str(exc))


def _planned_command(provider: str, artifact: str) -> List[str]:
    if provider == "lmstudio":
        return [_lms_cli() or "lms", "get", artifact, "--yes"]
    if provider == "ollama":
        return ["ollama", "pull", artifact]
    if provider == "supertonic":
        return ["python", "-m", "abstractvoice", "download", "--supertonic"]
    repo_id, _ = split_artifact(artifact)
    return ["huggingface_hub.snapshot_download", repo_id]


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
    outcome = _run_streaming(cmd, "lmstudio", artifact, emit)
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
    try:
        with urllib.request.urlopen(request, timeout=None) as response:  # noqa: S310 - localhost daemon
            for raw in response:
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
                total = event.get("total")
                completed = event.get("completed")
                if isinstance(total, int) and isinstance(completed, int) and total > 0:
                    percent = completed / total * 100.0
                    emit(
                        DownloadProgress(
                            status=DownloadStatus.DOWNLOADING,
                            message=status or "downloading",
                            percent=percent,
                            downloaded_bytes=completed,
                            total_bytes=total,
                        )
                    )
                elif status and status != last_status:
                    kind = DownloadStatus.VERIFYING if "verif" in status.lower() else DownloadStatus.DOWNLOADING
                    emit(DownloadProgress(status=kind, message=status))
                if status and status != last_status:
                    lines.append(status)
                    last_status = status
    except urllib.error.URLError as exc:
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
        message = str(exc)
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=message))
        return DownloadOutcome("ollama", artifact, False, "failed", message=message, output="\n".join(lines))

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


# --- supertonic download ----------------------------------------------------


def _download_supertonic(artifact: str, emit: ProgressCallback, base_url: Optional[str]) -> DownloadOutcome:
    try:
        from abstractvoice.supertonic.runtime import prefetch_supertonic  # type: ignore
    except Exception as exc:
        return DownloadOutcome(
            "supertonic",
            artifact,
            False,
            "failed",
            message=f"abstractvoice is not importable: {exc}",
            instruction='pip install "abstractvoice[supertonic]"',
        )
    emit(DownloadProgress(status=DownloadStatus.DOWNLOADING, message="fetching Supertonic 3 ONNX assets"))
    try:
        root = prefetch_supertonic()
    except Exception as exc:
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=str(exc)))
        return DownloadOutcome(
            "supertonic",
            artifact,
            False,
            "failed",
            message=str(exc),
            instruction="Retry, or fetch manually with `python -m abstractvoice download --supertonic`.",
        )
    emit(DownloadProgress(status=DownloadStatus.COMPLETE, message=f"Supertonic ready at {root}", percent=100.0))
    return DownloadOutcome(
        "supertonic",
        artifact,
        True,
        "completed",
        message="Supertonic 3 assets cached",
        location=str(root),
        command=["abstractvoice.prefetch_supertonic"],
    )


# --- huggingface download ---------------------------------------------------


def _download_huggingface(artifact: str, emit: ProgressCallback, base_url: Optional[str]) -> DownloadOutcome:
    repo_id, _quant = split_artifact(artifact)
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        return DownloadOutcome(
            "huggingface",
            artifact,
            False,
            "failed",
            message=f"huggingface_hub is not installed: {exc}",
            instruction='pip install "abstractcore[huggingface]"',
        )

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    emit(DownloadProgress(status=DownloadStatus.DOWNLOADING, message=f"snapshot_download {repo_id}"))

    # Real progress without a hub-version-specific callback API: the bytes on
    # disk are the ground truth, and polling them costs one stat walk a second.
    stop = threading.Event()
    watcher = threading.Thread(target=_watch_hf_cache, args=(repo_id, emit, stop), daemon=True)
    watcher.start()
    try:
        resolved = snapshot_download(repo_id=repo_id, token=token)
    except Exception as exc:
        stop.set()
        message = str(exc)
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=message))
        return DownloadOutcome(
            "huggingface",
            artifact,
            False,
            "failed",
            message=message,
            output=message,
            instruction=(
                f"If {repo_id} is gated, accept its licence on huggingface.co and export HF_TOKEN, "
                "then retry."
            ),
        )
    finally:
        stop.set()

    emit(DownloadProgress(status=DownloadStatus.COMPLETE, message=f"cached at {resolved}", percent=100.0))
    return DownloadOutcome(
        "huggingface",
        artifact,
        True,
        "completed",
        message=f"snapshot cached at {resolved}",
        location=str(resolved),
        command=["huggingface_hub.snapshot_download", repo_id],
    )


def _watch_hf_cache(repo_id: str, emit: ProgressCallback, stop: threading.Event) -> None:
    folder = "models--" + repo_id.replace("/", "--")
    while not stop.wait(2.0):
        total = 0
        for base in _hf_cache_dirs():
            path = base / folder
            try:
                if not path.is_dir():
                    continue
                total += sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            except Exception:
                continue
        if total > 0:
            emit(
                DownloadProgress(
                    status=DownloadStatus.DOWNLOADING,
                    message=f"{repo_id}: {total / 1_000_000:.0f} MB on disk",
                    downloaded_bytes=total,
                )
            )


_DOWNLOADERS: Dict[str, Callable[[str, ProgressCallback, Optional[str]], DownloadOutcome]] = {
    "lmstudio": _download_lmstudio,
    "ollama": _download_ollama,
    "supertonic": _download_supertonic,
}


def _run_streaming(cmd: List[str], provider: str, artifact: str, emit: ProgressCallback) -> DownloadOutcome:
    """Run a provider CLI and forward its lines verbatim (rule 4)."""

    lines: List[str] = []
    try:
        proc = subprocess.Popen(  # noqa: S603 - argv, never a shell string
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
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
    assert proc.stdout is not None
    for raw in proc.stdout:
        text = raw.rstrip("\n").rstrip("\r")
        if not text.strip():
            continue
        lines.append(text)
        emit(DownloadProgress(status=DownloadStatus.DOWNLOADING, message=text.strip()))
    code = proc.wait()
    output = "\n".join(lines)
    if code != 0:
        emit(DownloadProgress(status=DownloadStatus.ERROR, message=f"{cmd[0]} exited {code}"))
        return DownloadOutcome(
            provider,
            artifact,
            False,
            "failed",
            message=f"{cmd[0]} exited {code}",
            output=output,
            command=list(cmd),
            instruction=f"Run `{' '.join(cmd)}` yourself to see the tool's full prompt.",
        )
    emit(DownloadProgress(status=DownloadStatus.COMPLETE, message=f"{provider}: {artifact} downloaded", percent=100.0))
    return DownloadOutcome(
        provider,
        artifact,
        True,
        "completed",
        message=f"{artifact} downloaded",
        output=output,
        command=list(cmd),
    )


# ---------------------------------------------------------------------------
# The recommended journey
# ---------------------------------------------------------------------------


def recommended_downloads() -> List[Dict[str, str]]:
    """`RECOMMENDED_MODEL_DOWNLOADS` as a stable list, route key included."""

    out: List[Dict[str, str]] = []
    for route_key, spec in RECOMMENDED_MODEL_DOWNLOADS.items():
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

    urls = {k.lower(): v for k, v in (base_urls or {}).items()}
    rows: List[Dict[str, Any]] = []
    with presence_sweep():
        for item in recommended_downloads():
            presence = probe(item["provider"], item["artifact"], base_url=urls.get(item["provider"].lower()))
            row = dict(item)
            row.update(presence.to_dict())
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
