"""The model browser: what can be downloaded, is it here, and will it fit.

Contract C (`model_catalog_v1`). One payload joins five sources that already
exist elsewhere and must not be re-derived by any surface:

  curated seed      assets/model_downloads_catalog.json (per-engine artifact ids)
  capabilities      assets/model_capabilities.json via the registry alias lookup
  presence          model_materializer.probe(), inside ONE presence_sweep()
  fit               utils.model_fit.estimate_fit() against utils.host_profile()
  hub (opt-in)      Hugging Face sizes/params (+ free-text search rows)

THE HUB IS OPT-IN AND CACHED. `hub=False` (the default) never touches the
network: sizes come from the seed (observed on real engine stores) or are
labelled `estimate` from parameters x bits. `hub=True` asks the Hugging Face
API for exact file sizes and parameter counts, caches every answer on disk
for 24 hours under the AbstractCore config directory, stops at the first 429
(rate limit) and reports it in `hub.errors`, and degrades to the offline
answer when the hub is unreachable. A browser that hammered the hub on every
keystroke would be rate-limited within minutes.
"""

from __future__ import annotations

import functools
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "MODEL_CATALOG_SCHEMA",
    "SEED_SCHEMA",
    "load_seed",
    "validate_catalog",
    "catalog",
    "search",
    "catalog_id_for",
    "hub_cache_path",
]

MODEL_CATALOG_SCHEMA = "model_catalog_v1"
SEED_SCHEMA = "model_downloads_catalog_v1"
_HUB_TTL_S = 24 * 3600
_HUB_SEARCH_LIMIT = 20
_SEED_PROVIDERS = ("ollama", "lmstudio", "mlx", "huggingface", "mlx-gen", "mlx-vlm", "diffusers", "supertonic")
_EXACT_SIZE_SOURCES = ("catalog", "hf_api", "engine")
_HF_REPO_PROVIDERS = ("mlx", "huggingface", "mlx-gen", "mlx-vlm", "diffusers")

# Which engine (contract B id) runs an artifact of this provider.
_PROVIDER_ENGINE = {
    "ollama": "ollama",
    "lmstudio": "lmstudio",
    "mlx": "mlx",
    "mlx-gen": "mlx",
    "mlx-vlm": "mlx",
    "huggingface": "huggingface",
    "diffusers": "huggingface",
}

# The build an engine fetches when the reference names no quant.
_ENGINE_DEFAULT_QUANT = {"ollama": "q4_k_m", "lmstudio": "4bit"}

# Pre-selection order per accelerator (principle 3: fewest clicks).
_HOST_PREFERENCE = {
    "metal": ("lmstudio", "mlx", "ollama", "huggingface", "mlx-gen", "supertonic"),
    "cuda": ("ollama", "lmstudio", "huggingface", "supertonic"),
    "rocm": ("ollama", "lmstudio", "huggingface", "supertonic"),
    "none": ("ollama", "lmstudio", "huggingface", "supertonic"),
}


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def _seed_path() -> Path:
    return Path(__file__).resolve().parent.parent / "assets" / "model_downloads_catalog.json"


@functools.lru_cache(maxsize=1)
def _load_seed_cached() -> Dict[str, Any]:
    with open(_seed_path(), "r", encoding="utf-8") as fh:
        return json.load(fh)


def load_seed() -> Dict[str, Any]:
    """The curated seed (a deep copy: callers may mutate it)."""

    return json.loads(json.dumps(_load_seed_cached()))


def validate_catalog(data: Any) -> List[str]:
    """Structural validation of a seed; `[]` when valid.

    Mirrors `model_downloads_catalog.schema.json` so the check runs without
    the optional `jsonschema` package (the test suite also runs the real
    schema when `jsonschema` is importable).
    """

    errors: List[str] = []
    if not isinstance(data, dict):
        return ["catalog is not an object"]
    if data.get("schema") != SEED_SCHEMA:
        errors.append(f"schema must be {SEED_SCHEMA!r}")
    if not isinstance(data.get("version"), str):
        errors.append("version must be a YYYY-MM-DD string")
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        return errors + ["rows must be a non-empty list"]
    required_row = ("id", "family", "display_name", "vendor", "params_total", "params_active", "license",
                    "capabilities_key", "tags", "starter", "artifacts")
    required_art = ("provider", "artifact", "quant", "download_bytes", "size_source", "verified")
    allowed_row = set(required_row) | {"notes", "capabilities_override"}
    allowed_art = set(required_art) | {"recommended"}
    seen: set = set()
    for i, row in enumerate(rows):
        where = f"rows[{i}]"
        if not isinstance(row, dict):
            errors.append(f"{where} is not an object")
            continue
        for key in required_row:
            if key not in row:
                errors.append(f"{where} missing {key!r}")
        for key in row:
            if key not in allowed_row:
                errors.append(f"{where} has unknown field {key!r}")
        rid = row.get("id")
        if not isinstance(rid, str) or not rid or rid != rid.lower() or " " in rid:
            errors.append(f"{where}.id must be a lowercase id")
        elif rid in seen:
            errors.append(f"{where}.id {rid!r} is duplicated")
        seen.add(rid)
        for key in ("params_total", "params_active"):
            value = row.get(key)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value <= 0):
                errors.append(f"{where}.{key} must be a positive integer or null")
        if not isinstance(row.get("tags"), list):
            errors.append(f"{where}.tags must be a list")
        if not isinstance(row.get("starter"), bool):
            errors.append(f"{where}.starter must be a boolean")
        arts = row.get("artifacts")
        if not isinstance(arts, list) or not arts:
            errors.append(f"{where}.artifacts must be a non-empty list")
            continue
        for j, art in enumerate(arts):
            aw = f"{where}.artifacts[{j}]"
            if not isinstance(art, dict):
                errors.append(f"{aw} is not an object")
                continue
            for key in required_art:
                if key not in art:
                    errors.append(f"{aw} missing {key!r}")
            for key in art:
                if key not in allowed_art:
                    errors.append(f"{aw} has unknown field {key!r}")
            if art.get("provider") not in _SEED_PROVIDERS:
                errors.append(f"{aw}.provider {art.get('provider')!r} is not one of {', '.join(_SEED_PROVIDERS)}")
            if not isinstance(art.get("artifact"), str) or not art.get("artifact"):
                errors.append(f"{aw}.artifact must be a non-empty string")
            size = art.get("download_bytes")
            if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size <= 0):
                errors.append(f"{aw}.download_bytes must be a positive integer or null")
            if art.get("size_source") not in ("catalog", "unknown"):
                errors.append(f"{aw}.size_source must be 'catalog' or 'unknown'")
            if (size is None) != (art.get("size_source") == "unknown"):
                errors.append(f"{aw}: size_source must be 'unknown' exactly when download_bytes is null")
            if not isinstance(art.get("verified"), bool):
                errors.append(f"{aw}.verified must be a boolean")
    return errors


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


@functools.lru_cache(maxsize=1)
def _artifact_index() -> Dict[Tuple[str, str], str]:
    index: Dict[Tuple[str, str], str] = {}
    for row in _load_seed_cached().get("rows") or []:
        for art in row.get("artifacts") or []:
            index[(_norm(art.get("provider")), _norm(art.get("artifact")))] = row["id"]
    return index


def catalog_id_for(provider: Any, artifact: Any) -> Optional[str]:
    """The curated row id an installed artifact belongs to, or None.

    Tolerant the same way presence is: `qwen/qwen3.5-9b` (the id LM Studio
    reports) matches the catalog's `qwen/qwen3.5-9b@4bit`, `gemma3:1b:latest`
    matches `gemma3:1b`, and an HF repo matches its `repo:QUANT` artifacts.
    """

    from . import model_materializer as mm

    pid = _norm(provider)
    ref = _norm(artifact)
    index = _artifact_index()
    if (pid, ref) in index:
        return index[(pid, ref)]
    for (p, a), rid in index.items():
        if p != pid:
            continue
        if pid == "ollama" and mm._ollama_tag_match(a, ref):
            return rid
        if pid == "lmstudio" and mm._matches_installed_id(ref, a):
            return rid
        if pid in _HF_REPO_PROVIDERS and _norm(mm.hf_artifact_parts(a)[0]) == ref:
            return rid
    return None


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def _capabilities(key: Optional[str], override: Optional[Mapping[str, Any]], tags: Iterable[str]) -> Dict[str, Any]:
    caps: Dict[str, Any] = {
        "text": None,
        "vision": None,
        "audio": None,
        "tools": None,
        "thinking": None,
        "max_tokens": None,
        "embedding": "embedding" in set(tags),
        "source": None,
    }
    registry: Optional[Dict[str, Any]] = None
    if key:
        try:
            from ..architectures.detection import lookup_registry_model_capabilities

            registry = lookup_registry_model_capabilities(key)
        except Exception:
            registry = None
    if registry:
        routes = registry.get("capability_routes") or {}
        outputs = routes.get("output") if isinstance(routes, dict) else None
        caps["text"] = ("text" in outputs) if isinstance(outputs, list) else True
        caps["vision"] = bool(registry.get("vision_support"))
        caps["audio"] = bool(registry.get("audio_support"))
        tool = str(registry.get("tool_support") or "").strip().lower()
        caps["tools"] = tool if tool in {"native", "prompted", "none"} else None
        thinking = registry.get("thinking_support")
        caps["thinking"] = bool(thinking) if thinking is not None else None
        max_tokens = registry.get("max_tokens")
        caps["max_tokens"] = int(max_tokens) if isinstance(max_tokens, int) and max_tokens > 0 else None
        if str(registry.get("model_type") or "").lower() == "embedding":
            caps["embedding"] = True
        caps["source"] = "model_capabilities.json"
    if caps["embedding"]:
        caps["text"] = False if caps["text"] is None else caps["text"]
    for k, v in dict(override or {}).items():
        caps[k] = v
    return caps


# ---------------------------------------------------------------------------
# Hugging Face hub (opt-in, cached)
# ---------------------------------------------------------------------------


def hub_cache_path() -> Path:
    """`<abstractcore config dir>/cache/hf_hub_catalog.json`."""

    base = str(os.getenv("ABSTRACTCORE_CONFIG_DIR") or "").strip()
    root = Path(base).expanduser() if base else Path.home() / ".abstractcore" / "config"
    return root / "cache" / "hf_hub_catalog.json"


_hub_lock = threading.Lock()


class _HubCache:
    def __init__(self, path: Path):
        self.path = path
        self.data: Dict[str, Any] = {"models": {}, "searches": {}}
        self.dirty = False
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.data["models"] = dict(loaded.get("models") or {})
                self.data["searches"] = dict(loaded.get("searches") or {})
        except Exception:
            pass

    def get(self, bucket: str, key: str) -> Optional[Dict[str, Any]]:
        entry = self.data[bucket].get(key)
        if not isinstance(entry, dict):
            return None
        if time.time() - float(entry.get("fetched_at") or 0) > _HUB_TTL_S:
            return None
        return entry

    def put(self, bucket: str, key: str, value: Dict[str, Any]) -> None:
        self.data[bucket][key] = dict(value, fetched_at=time.time())
        self.dirty = True

    def save(self) -> None:
        if not self.dirty:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(f".tmp{os.getpid()}")
            tmp.write_text(json.dumps(self.data), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception:
            pass


class _RateLimited(Exception):
    pass


def _status_code(exc: BaseException) -> Optional[int]:
    response = getattr(exc, "response", None)
    code = getattr(response, "status_code", None)
    if isinstance(code, int):
        return code
    text = str(exc)
    return 429 if "429" in text and "Too Many" in text else None


def _default_hf_api() -> Any:
    from huggingface_hub import HfApi  # type: ignore

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
    return HfApi(token=token)


def _hub_model_facts(api: Any, cache: _HubCache, repo_id: str) -> Dict[str, Any]:
    """`{siblings: [[name, size], ...], params_total, gated}` for one repo (cached)."""

    hit = cache.get("models", repo_id)
    if hit is not None:
        if hit.get("error"):
            raise LookupError(hit["error"])
        return hit
    try:
        info = api.model_info(repo_id, files_metadata=True)
    except Exception as exc:
        if _status_code(exc) == 429:
            raise _RateLimited(str(exc)) from exc
        cache.put("models", repo_id, {"error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        raise LookupError(str(exc)) from exc
    siblings = []
    for s in getattr(info, "siblings", None) or []:
        name = getattr(s, "rfilename", None)
        size = getattr(s, "size", None)
        if name:
            siblings.append([str(name), int(size) if isinstance(size, int) else None])
    params = None
    for attr in ("safetensors", "gguf"):
        block = getattr(info, attr, None)
        total = block.get("total") if isinstance(block, dict) else getattr(block, "total", None)
        if isinstance(total, int) and total > 0:
            params = total
            break
    facts = {"siblings": siblings, "params_total": params, "gated": bool(getattr(info, "gated", False))}
    cache.put("models", repo_id, facts)
    return facts


def _size_for(facts: Mapping[str, Any], patterns: Optional[List[str]]) -> Optional[int]:
    from .model_materializer import _matches_any

    total = 0
    counted = False
    for name, size in facts.get("siblings") or []:
        if patterns and not _matches_any(str(name), patterns):
            continue
        if isinstance(size, int):
            total += size
            counted = True
    return total if counted else None


def _hub_search(api: Any, cache: _HubCache, query: str, engine: Optional[str], accelerator: str) -> List[Dict[str, Any]]:
    """Free-text hub search -> `[{repo_id, provider, downloads}]` (cached)."""

    lanes: List[Tuple[str, str]] = []
    if engine in (None, "", "mlx") and (accelerator == "metal" or engine == "mlx"):
        lanes.append(("mlx", "mlx"))
    if engine in (None, "", "huggingface", "llamacpp"):
        lanes.append(("gguf", "huggingface"))
    out: List[Dict[str, Any]] = []
    for tag, provider in lanes:
        key = f"{tag}:{query.lower()}"
        hit = cache.get("searches", key)
        if hit is None:
            try:
                results = api.list_models(search=query, filter=tag, sort="downloads", limit=_HUB_SEARCH_LIMIT)
                rows = [
                    {"repo_id": str(getattr(m, "id", None) or getattr(m, "modelId", "")), "downloads": getattr(m, "downloads", None)}
                    for m in results
                ]
            except Exception as exc:
                if _status_code(exc) == 429:
                    raise _RateLimited(str(exc)) from exc
                raise LookupError(str(exc)) from exc
            cache.put("searches", key, {"results": rows})
            hit = {"results": rows}
        for r in hit.get("results") or []:
            if r.get("repo_id"):
                out.append({"repo_id": r["repo_id"], "provider": provider, "downloads": r.get("downloads")})
    return out


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------


def _disk_free_for(provider: str, host: Mapping[str, Any]) -> Optional[int]:
    disk = host.get("disk") or {}
    key = {"ollama": "ollama", "lmstudio": "lmstudio"}.get(provider, "hf_cache")
    entry = disk.get(key) if isinstance(disk, dict) else None
    value = (entry or {}).get("free_bytes") if isinstance(entry, dict) else None
    return int(value) if isinstance(value, int) else None


def _local_geometry(provider: str, artifact: str) -> Optional[Dict[str, Any]]:
    if provider not in ("mlx", "huggingface"):
        return None
    try:
        from ..utils.context_estimate import _config_geometry_from_dir, _resolve_snapshot_dir
        from .model_materializer import hf_artifact_parts

        snapshot = _resolve_snapshot_dir(hf_artifact_parts(artifact)[0])
        return _config_geometry_from_dir(snapshot) if snapshot is not None else None
    except Exception:
        return None


def _engine_support(host: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    try:
        from .engines import _support

        return {
            eid: {"supported": _support(eid, str(host.get("os")), str(host.get("arch")), host.get("accelerator"))[0]}
            for eid in ("ollama", "lmstudio", "mlx", "huggingface", "llamacpp")
        }
    except Exception:
        return {}


def _engine_for(provider: str, quant: Optional[str]) -> Optional[str]:
    if provider == "huggingface" and quant:
        from .model_materializer import _GGUF_QUANT_RE

        if _GGUF_QUANT_RE.match(str(quant)):
            return "llamacpp"
    return _PROVIDER_ENGINE.get(provider)


def _build_artifact(
    row: Mapping[str, Any],
    art: Mapping[str, Any],
    caps: Mapping[str, Any],
    host: Mapping[str, Any],
    support: Mapping[str, Mapping[str, Any]],
    hub: Optional[Dict[str, Any]],
    installed_sizes: Mapping[Tuple[str, str], int],
) -> Dict[str, Any]:
    from ..utils.model_fit import bits_for_quant, estimate_fit
    from . import model_materializer as mm

    provider = str(art["provider"])
    artifact = str(art["artifact"])
    quant = art.get("quant")
    size = art.get("download_bytes")
    size_source = "catalog" if isinstance(size, int) else "unknown"
    params = row.get("params_total")
    params_source = "catalog"

    presence = mm.probe(provider, artifact)
    installed_size = installed_sizes.get((provider, _norm(artifact)))
    if presence.status == mm.PRESENCE_INSTALLED and isinstance(installed_size, int):
        size, size_source = installed_size, "engine"

    if hub is not None and provider in _HF_REPO_PROVIDERS and size_source not in ("engine",):
        repo, _q, patterns = mm.hf_artifact_parts(artifact)
        facts = hub["facts"].get(repo)
        if facts:
            hub_size = _size_for(facts, patterns)
            if isinstance(hub_size, int):
                size, size_source = hub_size, "hf_api"
            if params is None and isinstance(facts.get("params_total"), int):
                params, params_source = facts["params_total"], "hf_api"

    if params is None:
        from ..utils.model_fit import parse_params_from_name

        guessed, _active = parse_params_from_name(mm.hf_artifact_parts(artifact)[0] if "/" in artifact else artifact.replace(":", "-"))
        if guessed:
            params, params_source = guessed, "name"

    bits = bits_for_quant(quant)
    fit_quant = quant
    assumed_note: Optional[str] = None
    if bits is None and not quant and provider in _ENGINE_DEFAULT_QUANT:
        # No quant in the reference: the engine picks its default build
        # (Ollama tags are Q4_K_M; LM Studio picks a 4-bit MLX/GGUF build).
        fit_quant = _ENGINE_DEFAULT_QUANT[provider]
        bits = bits_for_quant(fit_quant)
        params_source = "name" if params_source == "name" else "assumed_quant"
        assumed_note = f"no quant in the reference; assuming {provider}'s default ({fit_quant})"
    if not isinstance(size, int) and params and bits:
        size = int(params * bits / 8 * 1.03)
        size_source = "estimate"

    text_like = bool(caps.get("text")) or caps.get("text") is None
    context: Optional[int] = None
    if caps.get("embedding"):
        context = 512
    elif not text_like:
        context = 1  # image / voice engines: no KV cache worth budgeting

    fit = estimate_fit(
        host=host,
        params_total=params,
        params_source="name" if params_source in ("name", "assumed_quant") else params_source,
        quant=fit_quant,
        weight_bytes=size if size_source in _EXACT_SIZE_SOURCES else None,
        download_bytes=size if isinstance(size, int) else None,
        geometry=_local_geometry(provider, artifact) if presence.status == mm.PRESENCE_INSTALLED else None,
        context=context,
        max_tokens=caps.get("max_tokens"),
        disk_free_bytes=_disk_free_for(provider, host),
    )
    if assumed_note:
        fit["notes"] = [assumed_note] + list(fit.get("notes") or [])
    if presence.status == mm.PRESENCE_INSTALLED:
        fit["disk_ok"] = True
        fit["notes"] = list(fit.get("notes") or []) + ["already installed"]

    engine = _engine_for(provider, quant)
    supported = bool((support.get(engine) or {}).get("supported", True)) if engine else True
    if not supported:
        fit["notes"] = list(fit.get("notes") or []) + [f"the {engine} engine does not run on this host"]
    downloadable = supported and provider in mm.supported_providers() and presence.status != mm.PRESENCE_NOT_APPLICABLE

    return {
        "provider": provider,
        "artifact": artifact,
        "engine": engine,
        "quant": _norm(quant) or None,
        "bits": bits,
        "download_bytes": size if isinstance(size, int) else None,
        "size_source": size_source,
        "presence": {
            "status": presence.status,
            "location": presence.location,
            "evidence": presence.evidence or presence.detail or None,
        },
        "fit": fit,
        "supported_on_host": supported,
        "downloadable": bool(downloadable),
        "recommended": bool(art.get("recommended")),
        "verified": bool(art.get("verified")),
        "cli_download": f"abstractcore models download {provider} {artifact}",
    }


def _pick_recommended(row: Dict[str, Any], accelerator: str, installed_engines: Mapping[str, bool]) -> None:
    """Exactly one artifact per row gets `recommended: true`: the pre-selection.

    A curated recommendation (the fresh-install starter) wins when this host
    can run it; otherwise the host's preferred engine order decides, with an
    installed engine and a `fits` verdict preferred over the rest.
    """

    arts = row["artifacts"]
    curated = [a for a in arts if a.get("recommended") and a.get("supported_on_host")]
    for a in arts:
        a["recommended"] = False
    if curated:
        curated[0]["recommended"] = True
        return
    order = _HOST_PREFERENCE.get(accelerator, _HOST_PREFERENCE["none"])
    verdict_rank = {"fits": 0, "tight": 1, "partial_offload": 2, "unknown": 3, "too_large": 4}

    def rank(a: Mapping[str, Any]) -> Tuple[int, int, int, int]:
        installed = 0 if (a.get("presence") or {}).get("status") == "installed" else 1
        engine_ok = 0 if installed_engines.get(str(a.get("engine"))) else 1
        pos = order.index(a["provider"]) if a["provider"] in order else len(order)
        return (verdict_rank.get((a.get("fit") or {}).get("verdict", "unknown"), 3), installed, engine_ok, pos)

    candidates = [a for a in arts if a.get("supported_on_host") and a.get("downloadable")]
    if candidates:
        min(candidates, key=rank)["recommended"] = True


def _matches_query(row: Mapping[str, Any], query: str) -> bool:
    if not query:
        return True
    hay = " ".join(
        [
            str(row.get("id") or ""),
            str(row.get("family") or ""),
            str(row.get("display_name") or ""),
            str(row.get("vendor") or ""),
            " ".join(row.get("tags") or []),
            " ".join(str(a.get("artifact") or "") for a in row.get("artifacts") or []),
        ]
    ).lower()
    return all(tok in hay for tok in query.lower().split())


def _artifact_matches_engine(art: Mapping[str, Any], engine: str) -> bool:
    engine = engine.lower()
    return art.get("provider") == engine or art.get("engine") == engine


def catalog(
    q: Optional[str] = None,
    *,
    engine: Optional[str] = None,
    fits: bool = False,
    hub: bool = False,
    host: Optional[Dict[str, Any]] = None,
    hf_api: Any = None,
    hub_cache: Optional[Path] = None,
    tags: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Contract C: the `model_catalog_v1` payload.

    q        free text (all tokens must match id/family/name/vendor/tags/artifacts)
    engine   keep only artifacts for this provider/engine (ollama, lmstudio,
             mlx, huggingface, llamacpp, ...)
    fits     keep only artifacts whose verdict is `fits` or `tight` on this host
    hub      enrich HF-hosted artifacts with exact sizes from the Hugging Face
             API and, when `q` is given, append `hf_search` rows
    tags     keep only rows carrying every tag (e.g. ["embedding"])
    """

    from ..utils.host_profile import host_profile, utc_now_iso
    from . import model_materializer as mm

    profile = host or host_profile()
    accelerator = str(profile.get("accelerator") or "none")
    query = str(q or "").strip()
    eng = _norm(engine) or None
    want_tags = [_norm(t) for t in (tags or []) if _norm(t)]
    seed = load_seed()
    support = _engine_support(profile)

    seed_rows = [r for r in seed["rows"] if _matches_query(r, query)]
    if want_tags:
        seed_rows = [r for r in seed_rows if all(t in [_norm(x) for x in r.get("tags") or []] for t in want_tags)]

    hub_block: Optional[Dict[str, Any]] = None
    hub_ctx: Optional[Dict[str, Any]] = None
    search_hits: List[Dict[str, Any]] = []
    if hub:
        hub_block = {"enabled": True, "ok": True, "errors": [], "cache": None, "fetched": 0, "cached": 0}
        hub_ctx = {"facts": {}}
        with _hub_lock:
            cache = _HubCache(hub_cache or hub_cache_path())
            hub_block["cache"] = str(cache.path)
            try:
                api = hf_api if hf_api is not None else _default_hf_api()
            except Exception as exc:
                api = None
                hub_block.update(ok=False)
                hub_block["errors"].append(f"huggingface_hub unavailable: {exc}")
            if api is not None:
                repos: List[str] = []
                for r in seed_rows:
                    for a in r["artifacts"]:
                        if a["provider"] in _HF_REPO_PROVIDERS and (eng is None or _artifact_matches_engine(a, eng) or a["provider"] == eng):
                            repo = mm.hf_artifact_parts(a["artifact"])[0]
                            if repo not in repos:
                                repos.append(repo)
                try:
                    if query:
                        search_hits = _hub_search(api, cache, query, eng, accelerator)
                        for hit in search_hits:
                            if hit["repo_id"] not in repos:
                                repos.append(hit["repo_id"])
                    for repo in repos:
                        was_cached = cache.get("models", repo) is not None
                        try:
                            hub_ctx["facts"][repo] = _hub_model_facts(api, cache, repo)
                            hub_block["cached" if was_cached else "fetched"] += 1
                        except LookupError as exc:
                            hub_block["errors"].append(f"{repo}: {str(exc)[:160]}")
                except _RateLimited as exc:
                    hub_block.update(ok=False)
                    hub_block["errors"].append(f"rate limited by the Hugging Face API (429); showing cached/offline data: {str(exc)[:160]}")
                except LookupError as exc:
                    hub_block.update(ok=False)
                    hub_block["errors"].append(f"hub search failed (offline?): {str(exc)[:200]}")
                cache.save()

    try:
        from .engines import engine_inventory

        inventory = engine_inventory(probe=False, host=profile)
        installed_engines = {e["id"]: bool(e.get("installed")) for e in inventory["engines"]}
    except Exception:
        installed_engines = {}

    installed_sizes: Dict[Tuple[str, str], int] = {}
    rows_out: List[Dict[str, Any]] = []
    with mm.presence_sweep():
        try:
            inst = mm.list_installed(None, include_loaded=False)
            for r in inst["rows"]:
                if isinstance(r.get("size_bytes"), int):
                    installed_sizes[(r["provider"], _norm(r["artifact"]))] = r["size_bytes"]
        except Exception:
            pass
        for seed_row in seed_rows:
            caps = _capabilities(seed_row.get("capabilities_key"), seed_row.get("capabilities_override"), seed_row.get("tags") or [])
            arts = [
                _build_artifact(seed_row, a, caps, profile, support, hub_ctx, installed_sizes)
                for a in seed_row["artifacts"]
                if eng is None or a["provider"] == eng or _engine_for(a["provider"], a.get("quant")) == eng
            ]
            row = {
                "id": seed_row["id"],
                "family": seed_row["family"],
                "display_name": seed_row["display_name"],
                "vendor": seed_row["vendor"],
                "params_total": seed_row.get("params_total"),
                "params_active": seed_row.get("params_active"),
                "license": seed_row.get("license"),
                "capabilities": caps,
                "source": "curated",
                "tags": list(seed_row.get("tags") or []),
                "starter": bool(seed_row.get("starter")),
                "notes": seed_row.get("notes"),
                "artifacts": arts,
            }
            if arts:
                _pick_recommended(row, accelerator, installed_engines)
            rows_out.append(row)

        seen_repos = {
            mm.hf_artifact_parts(a["artifact"])[0].lower()
            for r in rows_out
            for a in r["artifacts"]
            if a["provider"] in _HF_REPO_PROVIDERS
        }
        for hit in search_hits:
            if hit["repo_id"].lower() in seen_repos:
                continue
            seen_repos.add(hit["repo_id"].lower())
            provider = hit["provider"]
            artifact = hit["repo_id"]
            quant = None
            facts = (hub_ctx or {}).get("facts", {}).get(artifact) or {}
            if provider == "huggingface":
                quant = _pick_gguf_quant(facts)
                if quant:
                    artifact = f"{artifact}:{quant}"
                else:
                    continue  # a GGUF repo with no recognisable quant file is not one click
            elif provider == "mlx":
                import re as _re

                m = _re.search(r"(\d+)bit", hit["repo_id"], _re.IGNORECASE)
                quant = f"{m.group(1)}bit" if m else None
            tags_hit = ["hub"]
            pseudo = {"params_total": facts.get("params_total")}
            caps = _capabilities(None, None, tags_hit)
            art = _build_artifact(
                pseudo,
                {"provider": provider, "artifact": artifact, "quant": quant, "download_bytes": None},
                caps,
                profile,
                support,
                hub_ctx,
                installed_sizes,
            )
            name = hit["repo_id"].rsplit("/", 1)[-1]
            row = {
                "id": "hf:" + hit["repo_id"].lower(),
                "family": name.lower(),
                "display_name": name,
                "vendor": hit["repo_id"].split("/", 1)[0],
                "params_total": facts.get("params_total"),
                "params_active": None,
                "license": None,
                "capabilities": caps,
                "source": "hf_search",
                "tags": tags_hit,
                "starter": False,
                "notes": f"Hugging Face search result ({hit.get('downloads') or 0} downloads); capabilities unknown until installed.",
                "downloads": hit.get("downloads"),
                "artifacts": [art],
            }
            _pick_recommended(row, accelerator, installed_engines)
            rows_out.append(row)

    if fits:
        for row in rows_out:
            row["artifacts"] = [
                a for a in row["artifacts"]
                if a.get("supported_on_host") and (a.get("fit") or {}).get("verdict") in ("fits", "tight")
            ]
    rows_out = [r for r in rows_out if r["artifacts"]]

    return {
        "schema": MODEL_CATALOG_SCHEMA,
        "host_profile": profile,
        "query": {"q": query or None, "engine": eng, "fits": bool(fits), "hub": bool(hub), "tags": want_tags or None},
        "hub": hub_block,
        "seed_version": seed.get("version"),
        "counts": {
            "rows": len(rows_out),
            "artifacts": sum(len(r["artifacts"]) for r in rows_out),
            "installed": sum(1 for r in rows_out for a in r["artifacts"] if a["presence"]["status"] == "installed"),
        },
        "rows": rows_out,
        "generated_at": utc_now_iso(),
    }


def _pick_gguf_quant(facts: Mapping[str, Any]) -> Optional[str]:
    """The best single quant a GGUF repo offers: Q4_K_M, then Q4_K_S/Q5_K_M/Q8_0."""

    names = [str(n) for n, _s in facts.get("siblings") or [] if str(n).lower().endswith(".gguf")]
    for q in ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0", "Q4_0"):
        if any(q.lower() in n.lower() for n in names):
            return q
    return None


def search(q: str, **kwargs: Any) -> Dict[str, Any]:
    """`catalog(q=...)` -- the `models search` verb."""

    return catalog(q, **kwargs)
