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
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

__all__ = [
    "MODEL_CATALOG_SCHEMA",
    "SEED_SCHEMA",
    "QUANT_CLASSES",
    "MTP_RECOMMENDED",
    "APPLE_TEXT_TIERS",
    "load_seed",
    "validate_catalog",
    "catalog",
    "search",
    "catalog_id_for",
    "hub_cache_path",
    "quant_class",
    "recommended_text_model",
]

MODEL_CATALOG_SCHEMA = "model_catalog_v1"
# The verdicts the "fits this computer" filter keeps. `needs_gpu_limit` is in:
# a model that fits once the operator runs the command in `fit.gpu_limit`
# (the 128 GiB tier's recommendation, backlog 0947) must not vanish from the
# filter that recommends it.
FITS_FILTER_VERDICTS = ("fits", "tight", "needs_gpu_limit")
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

# Pre-selection order per accelerator (principle 3: fewest clicks). On Apple
# silicon MLX is the recommended lane (operator ruling 2026-09-24); LM Studio
# and Ollama artifacts stay listed and downloadable, just not pre-selected.
_HOST_PREFERENCE = {
    "metal": ("mlx", "lmstudio", "ollama", "huggingface", "mlx-gen", "supertonic"),
    "cuda": ("ollama", "lmstudio", "huggingface", "supertonic"),
    "rocm": ("ollama", "lmstudio", "huggingface", "supertonic"),
    "none": ("ollama", "lmstudio", "huggingface", "supertonic"),
}


# ---------------------------------------------------------------------------
# The recommended text model: ONE function, every surface
# ---------------------------------------------------------------------------
#
# Operator rulings 2026-09-24 and 2026-09-28 (the second one after the
# operator's own measurement on a 24 GB Mac mini: Qwen3.8 27B 4-bit runs out
# of the box with a small context, and with about 30k tokens of context once
# `sudo sysctl iogpu.wired_limit_mb=20480` raises the GPU memory limit). On
# Apple silicon (accelerator `metal`) the recommended text artifact is chosen
# by unified memory, in GiB as the host probe reports it
# (`ram_bytes / 2**30`):
#
#     memory <  24          -> Qwen3.5 9B        (qwen3.5-9b): fits from
#                              16 GB; on an 8 GB Mac the pick stays, with the
#                              fit warning
#     24 <= memory < 128    -> Qwen3.8 27B       (qwen3.8-27b): on a 24 GB
#                              Mac it fits with a small context at the
#                              default limit (`tight`, `small_context`), and
#                              the fit's `raised_limit` gives the command
#                              for more context
#     memory >= 128         -> Qwen3.8 Flash-Next (qwen3.8-flash-next): on a
#                              128 GB Mac it fits once the GPU memory limit is
#                              raised (`needs_gpu_limit`, operator-accepted)
#
# Every other host keeps the portable default (`RECOMMENDED_MODEL_DOWNLOADS`
# in capability_defaults) -- except where the portable default's engine has
# no build: LM Studio does not run on Intel Macs (engines support matrix), so
# there the same model's Ollama build (`_PORTABLE_TEXT_ENGINE_FALLBACK`) is
# the pick, `basis: portable_engine_fallback`. The catalog's `recommended`/`starter` flags, the
# fresh-install seed, `apply-recommended`, `models download --recommended` and
# the Gateway's guide tiles all read `recommended_text_model()`; nothing else
# may hold a tier table.

# Whether each tier recommends its MTP (native multi-token prediction) build
# instead of the plain 4-bit one. W2 2026-09-24: NO-GO until the companion
# download and the mlx-lm fallback are fixed (mission CC).
MTP_RECOMMENDED = False

# The MTP builds carry their route options in the seed (artifact `options`:
# `speculation = {mode: native_mtp, num_draft_tokens: 2, require_acceleration:
# false}`, the keys providers/speculation.py accepts); the pick copies them.

# (upper bound in GiB, exclusive; None = no bound), catalog row, plain, MTP
# (None: the row has no MTP build, the plain one is recommended either way,
# and the route carries no MTP `speculation` policy).
# Every artifact named here must be a seed artifact of that row carrying an
# `upstream` verification record (enforced by tests/config/test_model_catalog.py);
# tests/config/test_recommended_text_tiers.py enforces the fit rule above.
APPLE_TEXT_TIERS: Tuple[Dict[str, Any], ...] = (
    {"below_gib": 24, "row": "qwen3.5-9b",
     "plain": "mlx-community/Qwen3.5-9B-MLX-4bit", "mtp": "mlx-works/Qwen3.5-9B-oQ4e-mtp"},
    {"below_gib": 128, "row": "qwen3.8-27b",
     "plain": "mlx-community/Qwen3.8-27B-4bit", "mtp": "Jundot/Qwen3.8-27B-oQ4e-mtp"},
    {"below_gib": None, "row": "qwen3.8-flash-next",
     "plain": "mlx-community/Qwen3.8-Flash-Next-4bit", "mtp": "Jundot/Qwen3.8-Flash-Next-oQ4e-mtp"},
)
_TIER_PROVIDER = "mlx"

# The portable text model on an engine that runs where LM Studio does not
# (Intel Macs). Same catalog row as the portable default; Ollama's bare tag
# is its default Q4_K_M build, the 4-bit intent of `qwen/qwen3.5-9b@4bit`.
# Must be a seed artifact of the portable row (a missing one raises).
_PORTABLE_TEXT_ENGINE_FALLBACK = {"provider": "ollama", "artifact": "qwen3.5:9b"}


def _tier_artifact(tier: Mapping[str, Any], mtp: Optional[bool] = None) -> str:
    use_mtp = MTP_RECOMMENDED if mtp is None else bool(mtp)
    return str(tier["mtp"] if use_mtp and tier["mtp"] else tier["plain"])


def _memory_gib(host: Mapping[str, Any]) -> Optional[float]:
    ram = host.get("ram_bytes")
    if isinstance(ram, bool) or not isinstance(ram, (int, float)) or ram <= 0:
        return None
    return float(ram) / float(1024**3)


def _apple_tier(memory_gib: Optional[float]) -> Tuple[Dict[str, Any], str]:
    """The tier for this much unified memory, and the rule that chose it."""

    if memory_gib is None:
        # No memory reading: the smallest tier, said out loud (never a guess
        # dressed up as a measurement).
        return APPLE_TEXT_TIERS[0], "unified memory unknown: smallest Apple silicon tier"
    lower = 0
    for tier in APPLE_TEXT_TIERS:
        bound = tier["below_gib"]
        if bound is None or memory_gib < bound:
            rule = f"{lower} <= memory < {bound} GiB" if bound is not None else f"memory >= {lower} GiB"
            return tier, rule
        lower = bound
    raise AssertionError("APPLE_TEXT_TIERS must end with an unbounded tier")


def _seed_row_and_artifact(row_id: str, provider: str, artifact: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """The seed row + artifact a recommendation names. Raises when absent:
    a recommendation the catalog cannot show is a broken seed, not a fallback."""

    for row in _load_seed_cached().get("rows") or []:
        if row.get("id") != row_id:
            continue
        for art in row.get("artifacts") or []:
            if art.get("provider") == provider and art.get("artifact") == artifact:
                return row, art
    raise LookupError(f"recommended artifact {provider}:{artifact} is not in catalog row {row_id!r}")


def _portable_text_row_id() -> str:
    portable = _portable_text_default()
    row_id = catalog_id_for(portable["provider"], portable["artifact"])
    if row_id is None:
        raise LookupError(f"portable text default {portable['provider']}:{portable['artifact']} is not in the catalog")
    return row_id


def _portable_text_default() -> Dict[str, Any]:
    from .capability_defaults import RECOMMENDED_CAPABILITY_DEFAULT_ROUTES, RECOMMENDED_MODEL_DOWNLOADS

    route = RECOMMENDED_CAPABILITY_DEFAULT_ROUTES["input.text"]
    download = RECOMMENDED_MODEL_DOWNLOADS["input.text"]
    return {
        "provider": download["provider"],
        "artifact": download["artifact"],
        "model": route.model,
        "options": json.loads(json.dumps(route.options or {})),
    }


def _fit_for_seed_artifact(row: Mapping[str, Any], art: Mapping[str, Any], host: Mapping[str, Any]) -> Dict[str, Any]:
    """The catalog's own fit verdict for a not-yet-installed seed artifact."""

    from ..utils.model_fit import bits_for_quant, estimate_fit

    caps = _capabilities(row.get("capabilities_key"), row.get("capabilities_override"), row.get("tags") or [])
    text_like = bool(caps.get("text")) or caps.get("text") is None
    size = art.get("download_bytes") if isinstance(art.get("download_bytes"), int) else None
    _repos, companion_bytes = _companions(str(art.get("provider")), str(art.get("artifact")))
    if isinstance(size, int) and isinstance(companion_bytes, int):
        size += companion_bytes  # the drafter downloads with it and stays resident
    quant = art.get("quant")
    if bits_for_quant(quant) is None and not quant and art.get("provider") in _ENGINE_DEFAULT_QUANT:
        quant = _ENGINE_DEFAULT_QUANT[str(art.get("provider"))]
    resident = _resident_bytes(art)
    fit = estimate_fit(
        host=host,
        params_total=row.get("params_total"),
        params_source="catalog",
        quant=quant,
        weight_bytes=resident if resident is not None else size,
        download_bytes=size,
        geometry=_seed_geometry(row),
        context=512 if caps.get("embedding") else (None if text_like else 1),
        max_tokens=caps.get("max_tokens"),
        disk_free_bytes=_disk_free_for(str(art.get("provider")), host),
    )
    note = _resident_note(art)
    if note:
        fit["notes"] = list(fit.get("notes") or []) + [note]
    return fit


def recommended_artifact_fit(provider: str, artifact: str, host: Mapping[str, Any]) -> Dict[str, Any]:
    """`{row, artifact, fit}` for one seed artifact a recommendation names, on `host`.

    The memory gate of a route whose model may not fit at all (the
    fit-gated rows: image, video, music; recommended or configured). Works on
    the LIGHT host reading too (import-time seed): with no measured ceiling it
    uses the host probe's own fallback basis, 75% of RAM
    (`host_profile._FALLBACK_CEILING_FRACTION`), so the seed, the plan and the
    catalog compare against the same number on such a host. Raises
    when the artifact is not in the seed (a broken recommendation, never a
    silent pass).
    """

    row_id = catalog_id_for(provider, artifact)
    if row_id is None:
        raise LookupError(f"recommended artifact {provider}:{artifact} is not in the catalog")
    row, art = _seed_row_and_artifact(row_id, provider, artifact)
    return {"row": row, "artifact": art, "fit": recommended_artifact_fit_for(row, art, host)}


def recommended_artifact_fit_for(row: Mapping[str, Any], art: Mapping[str, Any], host: Mapping[str, Any]) -> Dict[str, Any]:
    """`_fit_for_seed_artifact` on `host`, with the host probe's fallback
    ceiling (75% of RAM) where the reading has none (`recommended_artifact_fit`)."""

    from ..utils.host_profile import _FALLBACK_CEILING_FRACTION

    profile = dict(host)
    ram = profile.get("ram_bytes")
    if not isinstance(profile.get("ceiling_bytes"), (int, float)) and isinstance(ram, (int, float)) and ram > 0:
        profile["ceiling_bytes"] = int(_FALLBACK_CEILING_FRACTION * ram)
    return _fit_for_seed_artifact(row, art, profile)


def recommended_text_model(
    host: Optional[Mapping[str, Any]] = None, *, mtp: Optional[bool] = None, fit: bool = True
) -> Dict[str, Any]:
    """THE recommended text model for a host (default: this machine).

    Returns `{provider, artifact, model, options, catalog_id, basis, tier,
    memory_gib, mtp, fit, fits, warning}`:

      artifact   the exact download reference (what `models download` fetches)
      model      the id the route stores (the served id; for MLX the repo id)
      options    route options: the portable route's MTP policy
                 (`speculation`), overlaid with the artifact's own options
      basis      `apple_silicon_tiers`, `portable_default`,
                 `portable_engine_fallback` (the portable engine has no build
                 on this host, e.g. LM Studio on an Intel Mac), or
                 `no_supported_engine` (neither has one, e.g. FreeBSD: the
                 portable pick, which the routes report unavailable)
      tier       the rule that chose it (`24 <= memory < 128 GiB`; for the
                 engine fallback, why the portable engine was skipped)
      fit        the catalog's fit block for this artifact on this host
      companions repos that must be downloaded with the artifact (an MTP
                 build's drafter), `[]` for most

    A TIER THAT DOES NOT FIT IS STILL THE TIER. When the catalog's fit logic
    says the tier's model is `too_large` (or only `partial_offload`s) on this
    host, the pick does not silently fall to a smaller tier: it stays, with
    `fits: False` and a plain-language `warning` every surface shows. The
    memory rule is the operator's; the estimate is advice about it.
    """

    from ..utils.host_profile import host_profile

    # `fit=False` (the route/download tables, the import-time config seed)
    # needs only accelerator + memory: the LIGHT host reading, which never
    # loads mlx/torch or runs a GPU tool. The fit block needs the full probe.
    profile = dict(host) if host is not None else host_profile(light=not fit)
    accelerator = str(profile.get("accelerator") or "none")
    memory = _memory_gib(profile)
    use_mtp = MTP_RECOMMENDED if mtp is None else bool(mtp)
    if accelerator == "metal":
        tier, rule = _apple_tier(memory)
        provider, artifact = _TIER_PROVIDER, _tier_artifact(tier, use_mtp)
        row, art = _seed_row_and_artifact(str(tier["row"]), provider, artifact)
        # The route's MTP POLICY (`speculation`) is host-wide and the same as
        # the portable route's: it asks for MTP wherever the loaded artifact
        # can do it. An MTP build's own options overlay it. A tier whose model
        # has no MTP build (`mtp: None`) gets no policy: nothing
        # is written for a capability the model does not have.
        policy = dict(_portable_text_default()["options"])
        if not tier["mtp"]:
            policy.pop("speculation", None)
        out: Dict[str, Any] = {
            "provider": provider,
            "artifact": artifact,
            "model": artifact,  # an MLX route serves the repo id it downloads
            "options": dict(policy, **json.loads(json.dumps(art.get("options") or {}))),
            "catalog_id": row["id"],
            "basis": "apple_silicon_tiers",
            "tier": rule,
            "mtp": use_mtp and bool(tier["mtp"]),
        }
    else:
        from .capability_defaults import recommended_route_unavailable_reason

        portable = _portable_text_default()
        row_id = _portable_text_row_id()
        unsupported = recommended_route_unavailable_reason(portable["provider"], profile)
        fallback = _PORTABLE_TEXT_ENGINE_FALLBACK
        fallback_unsupported = (
            recommended_route_unavailable_reason(fallback["provider"], profile) if unsupported else None
        )
        if unsupported and fallback_unsupported:
            # Neither engine runs here (FreeBSD, a 32-bit ARM board, ...): the
            # pick stays the portable one and says why, and the route layer
            # reports input.text unavailable with LM Studio's reason -- never a
            # fallback that cannot run either.
            row, art = _seed_row_and_artifact(row_id, portable["provider"], portable["artifact"])
            out = dict(
                portable,
                catalog_id=row_id,
                basis="no_supported_engine",
                tier=f"{unsupported}; {fallback_unsupported}",
                mtp=False,
            )
        elif unsupported:
            row, art = _seed_row_and_artifact(row_id, fallback["provider"], fallback["artifact"])
            out = {
                "provider": fallback["provider"],
                "artifact": fallback["artifact"],
                "model": fallback["artifact"],  # Ollama serves the tag it pulls
                # The host-wide MTP policy, as on every other host.
                "options": portable["options"],
                "catalog_id": row_id,
                "basis": "portable_engine_fallback",
                "tier": f"{unsupported}: the portable model's {fallback['provider']} build",
                "mtp": False,
            }
        else:
            row, art = _seed_row_and_artifact(row_id, portable["provider"], portable["artifact"])
            out = dict(portable, catalog_id=row_id, basis="portable_default", tier=None, mtp=False)
    out["memory_gib"] = round(memory, 2) if memory is not None else None
    if not fit:
        out.update(fit=None, fits=None, companions=None, companion_bytes=None, warning=None)
        return out
    fit = _fit_for_seed_artifact(row, art, profile)
    out["fit"] = fit
    verdict = fit.get("verdict")
    out["fits"] = verdict in ("fits", "tight") if verdict != "unknown" else None
    out["companions"], out["companion_bytes"] = _companions(str(out["provider"]), str(out["artifact"]))
    out["warning"] = _fit_warning(row, fit)
    return out


def _gib(value: int, digits: int) -> float:
    return round(value / 1024**3, digits)


def _gib_digits(*pairs: Tuple[int, int]) -> int:
    """Decimals (1 or 2) at which every `(a, b)` pair keeps its ORDER.

    "needs 16 GiB, can give 16 GiB" under a `too_large` verdict reads as a
    contradiction; one more decimal keeps the comparison the verdict made.
    """

    for digits in (1, 2):
        if all((_gib(a, digits) > _gib(b, digits)) == (a > b) and (_gib(a, digits) < _gib(b, digits)) == (a < b) for a, b in pairs):
            return digits
    return 2


def _fit_amounts(fit: Mapping[str, Any]) -> str:
    """The numbers the verdict compared: the total need vs the usable memory.

    Each is split into its parts (weights + working memory; the ceiling minus
    the system reserve) and the parts are printed so they ADD UP on screen.
    """

    need = fit.get("need_bytes")
    usable = fit.get("usable_bytes")
    ceiling = fit.get("ceiling_bytes")
    weights = fit.get("weight_bytes")
    if not (isinstance(need, int) and isinstance(usable, int)):
        return ""
    d = _gib_digits((need, usable))
    need_g, usable_g = _gib(need, d), _gib(usable, d)
    split = ""
    if isinstance(weights, int) and 0 < weights < need:
        w_g = _gib(weights, d)
        split = f" ({w_g:.{d}f} GiB of weights plus {need_g - w_g:.{d}f} GiB for its working memory and cache)"
    if fit.get("accelerator") == "metal" and isinstance(ceiling, int) and ceiling > usable:
        # Apple silicon: say the GPU memory limit itself (what macOS reports:
        # 17.8 GB = 16.6 GiB measured on a 24 GB Mac mini; 75% of RAM when it
        # cannot be read), then what is left after MLX's working buffers --
        # never the remainder alone as "what this computer can give".
        c_g = _gib(ceiling, d)
        which = (
            "the GPU memory limit raised on this Mac"
            if fit.get("ceiling_source") == "metal_wired_limit"
            else "macOS's GPU memory limit on this Mac"
        )
        return (
            f" It needs about {need_g:.{d}f} GiB in total{split}; {which} is {c_g:.{d}f} GiB, and after "
            f"{c_g - usable_g:.{d}f} GiB of working buffers about {usable_g:.{d}f} GiB is left for a model."
        )
    reserve = ""
    if isinstance(ceiling, int) and ceiling > usable:
        c_g = _gib(ceiling, d)
        reserve = (
            f" (the most this computer lets a model use is {c_g:.{d}f} GiB, and {c_g - usable_g:.{d}f} GiB "
            "of that is kept free for the system)"
        )
    return (
        f" It needs about {need_g:.{d}f} GiB in total{split}; this computer can give a model about "
        f"{usable_g:.{d}f} GiB{reserve}."
    )


# MEASURED context at a raised GPU memory limit, per catalog row and
# `iogpu.wired_limit_mb`: the only token counts any sentence states (an
# estimate at these margins is below the working reserve's own error).
MEASURED_CONTEXT: Dict[Tuple[str, int], Dict[str, Any]] = {
    ("qwen3.8-27b", 20480): {
        "tokens": 30000,
        "where": "a 24 GB Mac mini",
        "source": "operator measurement 2026-09-28 (Qwen3.8 27B 4-bit, mlx-lm/Ollama/LM Studio/AbstractFramework)",
    },
}


def measured_context_note(row_id: Any, raised: Optional[Mapping[str, Any]]) -> Optional[str]:
    """`about 30k tokens after `<command>` (measured on a 24 GB Mac mini)` for a
    raised limit with a measured context, else None."""

    if not isinstance(raised, Mapping):
        return None
    got = MEASURED_CONTEXT.get((str(row_id or ""), int(raised.get("required_mb") or 0)))
    if not got:
        return None
    return f"about {_tokens(got['tokens'])} tokens after `{raised['command']}` (measured on {got['where']})"


def raised_limit_instruction(fit: Mapping[str, Any], row: Optional[Mapping[str, Any]] = None) -> Optional[str]:
    """The sentence for a `tight` Apple silicon fit that has a `raised_limit`:
    the command for more context, with the context it gives where it was
    measured (`MEASURED_CONTEXT`). None when the fit has no `raised_limit`."""

    rl = fit.get("raised_limit")
    if not isinstance(rl, Mapping):
        return None
    gib = int(rl["required_mb"]) / 1024
    got = MEASURED_CONTEXT.get((str((row or {}).get("id") or ""), int(rl["required_mb"])))
    more = f" (about {_tokens(got['tokens'])} tokens, measured on {got['where']})" if got else ""
    return (
        f"For more context{more}, let the GPU use {gib:.0f} GiB: run `{rl['command']}` in a terminal "
        "(asks for your password; lasts until the Mac restarts)."
    )


def _tokens(n: int) -> str:
    if n >= 10_000:
        return f"{round(n / 1000):.0f}k"
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def gpu_limit_instruction(fit: Mapping[str, Any]) -> str:
    """The sentence every surface shows for a `needs_gpu_limit` verdict:
    the exact command, the value, and what it costs (admin, until restart)."""

    gl = fit.get("gpu_limit") or {}
    gib = int(gl["required_mb"]) / 1024
    return (
        f"It fits once macOS lets the GPU use {gib:.0f} GiB: run `{gl['command']}` in a terminal "
        "(asks for your password; lasts until the Mac restarts), then load it."
    )


def _fit_warning(row: Mapping[str, Any], fit: Mapping[str, Any]) -> Optional[str]:
    """One sentence per verdict that deserves one: `too_large` /
    `partial_offload` (may not fit) and `tight` (fits, little headroom).

    The amounts are EXACTLY the two the verdict compared (`estimate_fit`):
    the total need (weights + KV cache + overhead) and the usable memory (the
    ceiling minus the system reserve) -- never the weights alone against the
    raw ceiling, which read "needs 16 GiB, can give 18 GiB: may not fit".
    """

    verdict = fit.get("verdict")
    if verdict not in ("too_large", "partial_offload", "tight", "needs_gpu_limit"):
        return None
    name = row.get("display_name") or row.get("id")
    amounts = _fit_amounts(fit)
    if verdict == "needs_gpu_limit":
        return f"{name} is the recommendation for this computer's memory.{amounts} {gpu_limit_instruction(fit)}"
    if verdict == "tight":
        raised = raised_limit_instruction(fit, row)
        if fit.get("small_context"):
            # Runs, with a small context: the weights against the limit
            # itself, and no token count (operator ruling 2026-09-28).
            weights, ceiling = fit.get("weight_bytes"), fit.get("ceiling_bytes")
            room = ""
            if isinstance(weights, int) and isinstance(ceiling, int):
                which = (
                    "the GPU memory limit raised on this Mac"
                    if fit.get("ceiling_source") == "metal_wired_limit"
                    else "macOS's GPU memory limit on this Mac"
                )
                room = (
                    f" Its weights ({_gib(weights, 1):.1f} GiB) fit {which} ({_gib(ceiling, 1):.1f} GiB), with "
                    "little room left for context."
                )
            return (
                f"{name} is the recommendation for this computer's memory. Tight: it runs with a small context "
                f"by default; close other apps first.{room}"
                + (f" {raised}" if raised else "")
            )
        return (
            f"{name} is the recommendation for this computer's memory and AbstractCore's estimate says it "
            f"fits, but tightly.{amounts} Close other models before loading it."
            + (f" {raised}" if raised else "")
        )
    if verdict == "partial_offload":
        return (
            f"{name} is the recommendation for this computer's memory, but AbstractCore's estimate says it "
            f"does not fit in the graphics memory and will run partly on the processor, slowly.{amounts}"
        )
    return (
        f"{name} is the recommendation for this computer's memory, but AbstractCore's estimate says it "
        f"may not fit.{amounts} It can fail to load or run slowly."
    )


# ---------------------------------------------------------------------------
# quant_class: one normalized quantization family per artifact
# ---------------------------------------------------------------------------

QUANT_CLASSES = ("2bit", "3bit", "4bit", "5bit", "6bit", "8bit", "16bit", "full", "unknown")

# Label -> class, checked in order against the lowercased quant label
# (`-`/space -> `_`). The label is authoritative; `bits` is consulted only when
# the label came from the artifact itself (never the engine-default ASSUMPTION
# the fit logic makes for an Ollama tag with no quant), and anything else is
# `unknown` -- never a silent guess.
#   N bit / Nbit / N_bit (MLX)        -> Nbit        (4bit, 8bit, 4bit_dwq)
#   [ud_][i]qN... (GGUF)              -> Nbit        (q4_k_m, q4_0, iq4_xs, q8_0)
#   oqN... (oMLX oQ mixed precision)  -> Nbit        (oq4e: base bits 4)
#   mxfp4 / nvfp4 / int4              -> 4bit
#   fp8 / f8 / int8                   -> 8bit
#   f16 / fp16 / bf16                 -> 16bit
#   f32 / fp32                        -> full
_QUANT_CLASS_PATTERNS: Tuple[Tuple[str, Optional[str]], ...] = (
    (r"^(\d+)_?bits?(?:_.*)?$", None),
    (r"^(?:ud_)?i?q(\d)(?:_.*)?$", None),
    (r"^oq(\d)[a-z0-9_]*$", None),
    (r"^(?:mxfp4|nvfp4|int4)$", "4bit"),
    (r"^(?:fp8|f8|int8)$", "8bit"),
    (r"^(?:f16|fp16|bf16)$", "16bit"),
    (r"^(?:f32|fp32)$", "full"),
)


def quant_class(quant: Any, bits: Any = None) -> str:
    """`q4_k_m` -> `4bit`, `Q8_0` -> `8bit`, `bf16` -> `16bit`, None -> `unknown`.

    `bits` (effective bits/weight) is a fallback for a label this table does
    not parse but the caller measured: [N, N+1) -> Nbit for N in 2..6,
    [8, 9) -> 8bit, [16, 17) -> 16bit, >= 32 -> full; else `unknown`.
    """

    import re as _re

    raw = str(quant or "").strip().lower().replace("-", "_").replace(" ", "")
    if raw:
        for pattern, fixed in _QUANT_CLASS_PATTERNS:
            m = _re.match(pattern, raw)
            if not m:
                continue
            if fixed is not None:
                return fixed
            n = int(m.group(1))
            if n >= 32:
                return "full"
            return f"{n}bit" if f"{n}bit" in QUANT_CLASSES else "unknown"
    if isinstance(bits, bool) or not isinstance(bits, (int, float)) or bits <= 0:
        return "unknown"
    b = float(bits)
    for n in (2, 3, 4, 5, 6, 8, 16):
        if n <= b < n + 1:
            return f"{n}bit"
    return "full" if b >= 32 else "unknown"


def quant_class_for(provider: Any, quant: Any) -> Tuple[str, Optional[str]]:
    """`(quant_class, quant_class_source)` for one catalog artifact.

      stated   the reference names its quant (`q4_k_m`, `@4bit`, `-8bit` repo)
      assumed  a bare Ollama tag / LM Studio id: the class of the build the
               engine fetches by default (`_ENGINE_DEFAULT_QUANT`), which the
               fit estimate assumes too -- a console labels it "assumed"
      None     no quant information at all -> `unknown`
    """

    from ..utils.model_fit import bits_for_quant

    if str(quant or "").strip():
        cls = quant_class(quant, bits_for_quant(quant))
        return cls, ("stated" if cls != "unknown" else None)
    default = _ENGINE_DEFAULT_QUANT.get(str(provider or ""))
    if default:
        cls = quant_class(default, bits_for_quant(default))
        if cls != "unknown":
            return cls, "assumed"
    return "unknown", None


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
    sizes = data.get("companion_sizes")
    if sizes is not None:
        if not isinstance(sizes, dict):
            errors.append("companion_sizes must be an object")
        else:
            for repo, entry in sizes.items():
                cw = f"companion_sizes[{repo!r}]"
                if not isinstance(repo, str) or repo.count("/") != 1:
                    errors.append(f"{cw}: the key must be a repo id")
                if not isinstance(entry, dict) or set(entry) - {"download_bytes", "upstream"}:
                    errors.append(f"{cw} must be {{download_bytes, upstream}}")
                    continue
                errors.extend(
                    _validate_upstream(cw, dict(entry, provider="mlx", size_source="catalog"), entry.get("download_bytes"))
                )
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        return errors + ["rows must be a non-empty list"]
    required_row = ("id", "family", "display_name", "vendor", "params_total", "params_active", "license",
                    "capabilities_key", "tags", "starter", "artifacts")
    required_art = ("provider", "artifact", "quant", "download_bytes", "size_source", "verified")
    allowed_row = set(required_row) | {"notes", "capabilities_override", "kv_geometry"}
    allowed_art = set(required_art) | {"recommended", "options", "upstream", "note", "resident"}
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
        if "kv_geometry" in row:
            errors.extend(_validate_kv_geometry(where, row.get("kv_geometry")))
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
            if "options" in art:
                errors.extend(_validate_artifact_options(aw, art.get("options")))
            if "upstream" in art:
                errors.extend(_validate_upstream(aw, art, size))
            if "note" in art and (not isinstance(art.get("note"), str) or not art.get("note")):
                errors.append(f"{aw}.note must be a non-empty string")
            if "resident" in art:
                errors.extend(_validate_resident(aw, art.get("resident")))
    return errors


def _validate_resident(aw: str, resident: Any) -> List[str]:
    """`{bytes, source[, measured_with, smaller_canvases, larger_canvases]}`: a
    MEASURED run-time memory need at the engine's default canvas, never a guess
    -- `source` must say where and at which profile it was measured;
    `measured_with` is the short label every sentence quotes (engine and output
    size), so the figure reads as that engine's, not as the model's own
    minimum. The optional canvas lists hold measured needs at other output
    sizes, `[{canvas, bytes, source}]`: `smaller_canvases` largest first, each
    below the default's (the size a model still runs at where its default does
    not fit); `larger_canvases` smallest first, each above the default's (the
    bigger sizes a computer with room to spare can ask for)."""

    if not isinstance(resident, dict):
        return [f"{aw}.resident must be an object"]
    errors: List[str] = []
    for key in resident:
        if key not in ("bytes", "source", "measured_with", "smaller_canvases", "larger_canvases"):
            errors.append(f"{aw}.resident has unknown field {key!r}")
    if "measured_with" in resident and (not isinstance(resident["measured_with"], str) or not resident["measured_with"]):
        errors.append(f"{aw}.resident.measured_with must name the engine and output size it was measured with")
    value = resident.get("bytes")
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{aw}.resident.bytes must be a positive integer")
        value = None
    if not isinstance(resident.get("source"), str) or not resident.get("source"):
        errors.append(f"{aw}.resident.source must say where the memory was measured")
    for field, direction in (("smaller_canvases", "below"), ("larger_canvases", "above")):
        entries = resident.get(field)
        if entries is None:
            continue
        if not isinstance(entries, list) or not entries:
            errors.append(f"{aw}.resident.{field} must be a non-empty list")
            continue
        previous = value
        for i, entry in enumerate(entries):
            ew = f"{aw}.resident.{field}[{i}]"
            if not isinstance(entry, dict) or set(entry) != {"canvas", "bytes", "source"}:
                errors.append(f"{ew} must be {{canvas, bytes, source}}")
                continue
            if not isinstance(entry["canvas"], str) or not re.fullmatch(r"\d+x\d+x\d+", entry["canvas"]):
                errors.append(f"{ew}.canvas must be WIDTHxHEIGHTxFRAMES")
            b = entry["bytes"]
            if not isinstance(b, int) or isinstance(b, bool) or b <= 0:
                errors.append(f"{ew}.bytes must be a positive integer")
            elif previous is not None and (b >= previous if direction == "below" else b <= previous):
                errors.append(f"{ew}.bytes must be {direction} the previous canvas's (the default's first)")
            else:
                previous = b
            if not isinstance(entry["source"], str) or not entry["source"]:
                errors.append(f"{ew}.source must say where the memory was measured")
    return errors


def _resident_bytes(art: Mapping[str, Any]) -> Optional[int]:
    """The measured run-time memory of a seed artifact (`resident.bytes`), or None.

    Video generation needs far more memory than its file size (the activations
    of every frame), so for those artifacts the fit compares THIS against the
    host, and the download size only against the disk."""

    resident = art.get("resident")
    value = resident.get("bytes") if isinstance(resident, Mapping) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def smaller_canvas_fit(provider: str, artifact: str, host: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The largest measured smaller canvas (`resident.smaller_canvases`) at
    which a seed artifact fits `host` (`fits` or `tight`), as `{canvas,
    bytes, fit}`, or None. For a video model whose default canvas does not
    fit: the size at which it still runs on this computer, measured."""

    row_id = catalog_id_for(provider, artifact)
    if row_id is None:
        raise LookupError(f"artifact {provider}:{artifact} is not in the catalog")
    row, art = _seed_row_and_artifact(row_id, provider, artifact)
    resident = art.get("resident") or {}
    for entry in resident.get("smaller_canvases") or []:
        variant = dict(art, resident={"bytes": entry["bytes"], "source": entry["source"]})
        fit = recommended_artifact_fit_for(row, variant, host)
        if fit.get("verdict") in ("fits", "tight"):
            return {"canvas": entry["canvas"], "bytes": entry["bytes"], "fit": fit}
    return None


def larger_canvas_fit(provider: str, artifact: str, host: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The largest measured larger canvas (`resident.larger_canvases`) at
    which a seed artifact also fits `host` (`fits` or `tight`), as `{canvas,
    bytes, fit}`, or None. For a video model whose default canvas fits: the
    bigger size this computer can ask for, measured."""

    row_id = catalog_id_for(provider, artifact)
    if row_id is None:
        raise LookupError(f"artifact {provider}:{artifact} is not in the catalog")
    row, art = _seed_row_and_artifact(row_id, provider, artifact)
    resident = art.get("resident") or {}
    for entry in reversed(resident.get("larger_canvases") or []):
        variant = dict(art, resident={"bytes": entry["bytes"], "source": entry["source"]})
        fit = recommended_artifact_fit_for(row, variant, host)
        if fit.get("verdict") in ("fits", "tight"):
            return {"canvas": entry["canvas"], "bytes": entry["bytes"], "fit": fit}
    return None


def _resident_note(art: Mapping[str, Any]) -> Optional[str]:
    resident = art.get("resident")
    if _resident_bytes(art) is None or not isinstance(resident, Mapping):
        return None
    return f"memory need is measured, not the file size: {resident.get('source')}"


def _validate_kv_geometry(where: str, geo: Any) -> List[str]:
    """`{n_layers, n_kv_heads, head_dim, source}`: the KV-CACHED attention
    geometry (a hybrid model counts only its full-attention layers)."""

    if not isinstance(geo, dict):
        return [f"{where}.kv_geometry must be an object"]
    errors: List[str] = []
    for key in geo:
        if key not in ("n_layers", "n_kv_heads", "head_dim", "source"):
            errors.append(f"{where}.kv_geometry has unknown field {key!r}")
    for key in ("n_layers", "n_kv_heads", "head_dim"):
        value = geo.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"{where}.kv_geometry.{key} must be a positive integer")
    if not isinstance(geo.get("source"), str) or not geo.get("source"):
        errors.append(f"{where}.kv_geometry.source must say where the geometry was read")
    return errors


def _companions(provider: str, artifact: str) -> Tuple[List[str], Optional[int]]:
    """`(repos, bytes)`: the companion repos an artifact downloads with it --
    FROM THE MLX DRAFTER REGISTRY (`model_materializer.companion_artifacts`
    -> `speculation.mlx_companion_repos`), never a hand-typed list -- and their
    total verified size from the seed's `companion_sizes` (None when any
    companion's size is not recorded: never a guess)."""

    from . import model_materializer as mm

    repos = list(mm.companion_artifacts(provider, artifact))
    if not repos:
        return [], None
    table = _load_seed_cached().get("companion_sizes") or {}
    sizes = [((table.get(r) or {}).get("download_bytes")) for r in repos]
    if all(isinstance(x, int) and x > 0 for x in sizes):
        return repos, int(sum(sizes))
    return repos, None


def _seed_geometry(row: Mapping[str, Any]) -> Optional[Dict[str, int]]:
    geo = row.get("kv_geometry")
    if not isinstance(geo, Mapping):
        return None
    return {k: int(geo[k]) for k in ("n_layers", "n_kv_heads", "head_dim")}


# How an `upstream` record proves an id exists and how big it is:
#   hf_api           HfApi().model_info(files_metadata=True); bytes = the weight
#                    files (*.safetensors, or the repo:QUANT *.gguf files)
#   ollama_registry  GET registry.ollama.ai/v2/library/<name>/manifests/<tag>;
#                    bytes = sum of the manifest layers
_UPSTREAM_METHODS = ("hf_api", "ollama_registry")


def _validate_upstream(aw: str, art: Mapping[str, Any], size: Any) -> List[str]:
    """An artifact that claims upstream verification must carry the size the
    verification read: a row without a verifiable size is rejected."""

    import re as _re

    up = art.get("upstream")
    if not isinstance(up, dict):
        return [f"{aw}.upstream must be an object"]
    errors: List[str] = []
    for key in up:
        if key not in ("method", "checked", "revision"):
            errors.append(f"{aw}.upstream has unknown field {key!r}")
    if up.get("method") not in _UPSTREAM_METHODS:
        errors.append(f"{aw}.upstream.method must be one of {', '.join(_UPSTREAM_METHODS)}")
    if not isinstance(up.get("checked"), str) or not _re.fullmatch(r"\d{4}-\d{2}-\d{2}", up.get("checked") or ""):
        errors.append(f"{aw}.upstream.checked must be a YYYY-MM-DD string")
    if up.get("method") == "hf_api" and art.get("provider") not in _HF_REPO_PROVIDERS:
        errors.append(f"{aw}.upstream.method hf_api needs a Hugging Face provider")
    if up.get("method") == "ollama_registry" and art.get("provider") != "ollama":
        errors.append(f"{aw}.upstream.method ollama_registry needs provider ollama")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0 or art.get("size_source") != "catalog":
        errors.append(f"{aw}: an upstream-verified artifact must carry the verified download_bytes (size_source 'catalog')")
    return errors


def _validate_artifact_options(aw: str, options: Any) -> List[str]:
    if not isinstance(options, dict):
        return [f"{aw}.options must be an object"]
    errors: List[str] = []
    for key in options:
        if key != "speculation":
            errors.append(f"{aw}.options has unknown field {key!r}")
    if "speculation" in options:
        from ..providers.speculation import normalize_speculation_request

        try:
            normalize_speculation_request(options["speculation"])
        except ValueError as exc:
            errors.append(f"{aw}.options.speculation: {exc}")
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

    # COMPANIONS (an MLX build's MTP drafter, from the registry): they download
    # with the artifact and stay resident while it decodes, so a catalog-sized
    # artifact's `download_bytes` (and the fit's weights) include them;
    # `companion_bytes` is that part. An engine-reported size is left as the
    # engine reports it.
    companions, companion_bytes = _companions(provider, artifact)
    if companions and isinstance(size, int) and size_source in ("catalog", "hf_api"):
        if isinstance(companion_bytes, int):
            size += companion_bytes

    if params is None:
        from ..utils.model_fit import parse_params_from_name

        guessed, _active = parse_params_from_name(mm.hf_artifact_parts(artifact)[0] if "/" in artifact else artifact.replace(":", "-"))
        if guessed:
            params, params_source = guessed, "name"

    bits = bits_for_quant(quant)
    # quant_class: the artifact's OWN label when it states one (`stated`);
    # else the build the engine fetches for a bare reference (`assumed`, the
    # same assumption the fit estimate makes below); else `unknown`.
    qclass, qclass_source = quant_class_for(provider, quant)
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

    # A measured run-time memory need (video generation) replaces the file
    # size as the weights the fit compares; the download size stays the disk need.
    resident = _resident_bytes(art)
    fit = estimate_fit(
        host=host,
        params_total=params,
        params_source="name" if params_source in ("name", "assumed_quant") else params_source,
        quant=fit_quant,
        weight_bytes=resident if resident is not None else (size if size_source in _EXACT_SIZE_SOURCES else None),
        download_bytes=size if isinstance(size, int) else None,
        # The seed's KV geometry (hybrid models: full-attention layers only)
        # beats a local config read, which counts every layer as KV-cached.
        geometry=_seed_geometry(row) or (_local_geometry(provider, artifact) if presence.status == mm.PRESENCE_INSTALLED else None),
        context=context,
        max_tokens=caps.get("max_tokens"),
        disk_free_bytes=_disk_free_for(provider, host),
    )
    if assumed_note:
        fit["notes"] = [assumed_note] + list(fit.get("notes") or [])
    if _resident_note(art):
        fit["notes"] = list(fit.get("notes") or []) + [_resident_note(art)]
    if companions and companion_bytes is None:
        fit["notes"] = list(fit.get("notes") or []) + [
            f"companion size not recorded ({', '.join(companions)}): not included in download_bytes"
        ]
    elif companions:
        fit["notes"] = list(fit.get("notes") or []) + [f"includes the MTP companion {', '.join(companions)}"]
    if presence.status == mm.PRESENCE_INSTALLED:
        fit["disk_ok"] = True
        fit["notes"] = list(fit.get("notes") or []) + ["already installed"]

    engine = _engine_for(provider, quant)
    supported = bool((support.get(engine) or {}).get("supported", True)) if engine else True
    if not supported:
        fit["notes"] = list(fit.get("notes") or []) + [f"the {engine} engine does not run on this host"]
    downloadable = supported and provider in mm.supported_providers() and presence.status != mm.PRESENCE_NOT_APPLICABLE

    # The default canvas does not fit but a measured smaller one does (video):
    # the size at which it still runs here.
    smaller: Optional[Dict[str, Any]] = None
    if supported and fit.get("verdict") not in ("fits", "tight") and (art.get("resident") or {}).get("smaller_canvases"):
        got = smaller_canvas_fit(provider, artifact, host)
        if got is not None:
            smaller = {"canvas": got["canvas"], "need_bytes": got["fit"]["need_bytes"], "verdict": got["fit"]["verdict"]}
            fit["notes"] = list(fit.get("notes") or []) + [
                f"at {got['canvas']} (width x height x frames) it needs about "
                f"{got['fit']['need_bytes'] / 1024**3:.1f} GiB (measured) and fits"
            ]

    # The default canvas fits and a measured larger one does too (video): the
    # bigger size this computer can ask for.
    larger: Optional[Dict[str, Any]] = None
    if supported and fit.get("verdict") in ("fits", "tight") and (art.get("resident") or {}).get("larger_canvases"):
        got = larger_canvas_fit(provider, artifact, host)
        if got is not None:
            larger = {"canvas": got["canvas"], "need_bytes": got["fit"]["need_bytes"], "verdict": got["fit"]["verdict"]}
            fit["notes"] = list(fit.get("notes") or []) + [
                f"at {got['canvas']} (width x height x frames) it needs about "
                f"{got['fit']['need_bytes'] / 1024**3:.1f} GiB (measured) and fits too"
            ]

    return {
        "provider": provider,
        "artifact": artifact,
        "engine": engine,
        "quant": _norm(quant) or None,
        "bits": bits,
        "quant_class": qclass,
        "quant_class_source": qclass_source,
        "companions": companions,
        "companion_bytes": companion_bytes,
        "note": art.get("note"),
        "options": json.loads(json.dumps(art.get("options") or {})),
        "download_bytes": size if isinstance(size, int) else None,
        "size_source": size_source,
        # The measured run-time memory (`resident` in the seed), or None.
        "resident_bytes": resident,
        # `{canvas, need_bytes, verdict}` when only a measured smaller canvas
        # fits this host (`resident.smaller_canvases`), else None.
        "smaller_canvas": smaller,
        # `{canvas, need_bytes, verdict}`: the largest measured larger canvas
        # that also fits this host (`resident.larger_canvases`), else None.
        "larger_canvas": larger,
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


def _pick_recommended(
    row: Dict[str, Any],
    accelerator: str,
    installed_engines: Mapping[str, bool],
    tier_artifact: Optional[Tuple[str, str]] = None,
) -> None:
    """Exactly one artifact per row gets `recommended: true`: the pre-selection.

    On Apple silicon a row that is one of the text tiers pre-selects its tier
    artifact (`tier_artifact`, from `APPLE_TEXT_TIERS` + `MTP_RECOMMENDED`).
    Otherwise a curated recommendation (the portable fresh-install starter)
    wins when this host can run it -- except on Apple silicon when the row has
    an MLX artifact (MLX is the recommended lane there); then the host's
    preferred engine order decides, with
    an installed engine and a `fits` verdict preferred over the rest.
    """

    arts = row["artifacts"]
    curated = [a for a in arts if a.get("recommended") and a.get("supported_on_host")]
    if accelerator == "metal" and any(a["provider"] == "mlx" and a.get("supported_on_host") for a in arts):
        curated = []  # the MLX lane beats a curated LM Studio/Ollama pick on a Mac
    if tier_artifact is not None:
        forced = [a for a in arts if (a["provider"], a["artifact"]) == tier_artifact]
        if not forced:
            raise LookupError(f"tier artifact {tier_artifact} missing from catalog row {row.get('id')!r}")
        curated = forced
    for a in arts:
        a["recommended"] = False
    if curated:
        curated[0]["recommended"] = True
        return
    order = _HOST_PREFERENCE.get(accelerator, _HOST_PREFERENCE["none"])
    verdict_rank = {"fits": 0, "tight": 1, "partial_offload": 2, "needs_gpu_limit": 2, "unknown": 3, "too_large": 4}

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
    # A token matches at the START of a word (words split on separators, `.`
    # kept inside versions): "8b" finds "qwen3-8b" but not "qwen3.5-0.8b".
    import re as _re

    words = [w for w in _re.split(r"[\s/:@_,-]+", hay) if w]
    return all(any(w.startswith(tok) for w in words) for tok in query.lower().split())


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
    fits     keep only artifacts that fit this host (`FITS_FILTER_VERDICTS`:
             `fits`, `tight`, or `needs_gpu_limit` -- fits once the Mac's GPU
             memory limit is raised, the command in `fit.gpu_limit`)
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
    # THE recommended text model for this host: drives the tier rows'
    # pre-selection and which text row is the `starter` (one function, see
    # `recommended_text_model`).
    text_pick = recommended_text_model(profile)
    text_rows = {str(t["row"]) for t in APPLE_TEXT_TIERS} | {text_pick["catalog_id"], _portable_text_row_id()}
    # A recommended row this host cannot run (Apple-only image generation on
    # Linux/Windows/Intel Macs) is not part of THIS host's starter kit.
    from .capability_defaults import RECOMMENDED_MODEL_DOWNLOADS, recommended_unavailable_routes

    unavailable_starter_rows = {
        catalog_id_for(RECOMMENDED_MODEL_DOWNLOADS[key]["provider"], RECOMMENDED_MODEL_DOWNLOADS[key]["artifact"])
        for key in recommended_unavailable_routes(profile)
        if key in RECOMMENDED_MODEL_DOWNLOADS
    }
    tier_by_row = {str(t["row"]): t for t in APPLE_TEXT_TIERS}

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
                if repos and not (hub_block["fetched"] or hub_block["cached"]):
                    hub_block["ok"] = False  # every lookup failed: offline or blocked
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
                # No text engine runs here (`no_supported_engine`): no text
                # starter either, like every other row this host cannot run.
                "starter": (seed_row["id"] == text_pick["catalog_id"] and text_pick["basis"] != "no_supported_engine") if seed_row["id"] in text_rows else (bool(seed_row.get("starter")) and seed_row["id"] not in unavailable_starter_rows),
                "notes": seed_row.get("notes"),
                "artifacts": arts,
            }
            if arts:
                tier = tier_by_row.get(seed_row["id"]) if accelerator == "metal" else None
                forced = (_TIER_PROVIDER, _tier_artifact(tier)) if tier is not None else None
                if text_pick["basis"] == "portable_engine_fallback" and seed_row["id"] == text_pick["catalog_id"]:
                    forced = (text_pick["provider"], text_pick["artifact"])
                if forced is not None and not any((a["provider"], a["artifact"]) == forced for a in arts):
                    forced = None  # an engine filter (`engine=ollama`) removed the tier artifact
                _pick_recommended(row, accelerator, installed_engines, forced)
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
                if a.get("supported_on_host") and (a.get("fit") or {}).get("verdict") in FITS_FILTER_VERDICTS
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
