"""The recommended model for every capability, per host and per machine class.

`recommended_models(host)` is THE per-host answer for every capability the
framework recommends a model for: text, image input (vision), speech output,
speech input, image generation, video generation and music. It adds no
recommendation of its own; it reads the ones that exist:

    text           `model_catalog.recommended_text_model()` (Apple silicon:
                   the unified-memory tier; elsewhere the portable default)
    vision         covered by the text model when it reads images
                   (`manager.model_supports_input`), which the recommended
                   text models do
    everything else `capability_defaults.RECOMMENDED_MODELS`, filtered per
                   host by the same rules the writers use (engine support,
                   and the memory gate of fit-gated rows)

`recommendation_matrix()` evaluates `recommended_models` on reference machines
(`MACHINE_CLASSES`: Apple silicon at every unified-memory size Apple ships,
merged into bands where every answer is the same; Linux or Windows with an
NVIDIA GPU; processor-only Linux or Windows; Intel Macs). It is what
`abstractcore models recommendations --json` exports and what the docs tables
(`docs/recommended-models.md`, `scripts/update_recommended_models_doc.py`) and
the AbstractFramework website render, so no table of models is ever edited by
hand.

The matrix is deterministic: reference hosts are synthetic `host_profile_v1`
dicts, nothing on this machine is probed, and no network is used.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "APPLE_MEMORY_SIZES_GIB",
    "MACHINE_CLASSES",
    "RECOMMENDATION_CAPABILITIES",
    "RECOMMENDATIONS_SCHEMA",
    "add_recommendations_parser",
    "handle_recommendations",
    "host_recommendations",
    "iter_entries",
    "recommendation_matrix",
    "recommended_models",
    "render_markdown",
]

RECOMMENDATIONS_SCHEMA = "model_recommendations_v1"

# (id, route key, label, tasks). The order is the order every table shows.
RECOMMENDATION_CAPABILITIES: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    ("text", "input.text", "Text and chat", ("text_generation",)),
    ("vision", "input.image", "Image input (vision)", ("image_understanding",)),
    ("speech_output", "output.voice", "Speech output (text to speech)", ("text_to_speech",)),
    ("speech_input", "input.voice", "Speech input (speech to text)", ("speech_to_text",)),
    ("image", "output.image", "Image generation", ("text_to_image",)),
    ("video", "output.video", "Video generation", ("text_to_video", "image_to_video")),
    ("music", "output.music", "Music generation", ("text_to_music",)),
)

# Unified memory sizes (GiB) Apple ships Apple silicon Macs with. The matrix
# evaluates every size and merges neighbours whose answers are identical.
APPLE_MEMORY_SIZES_GIB: Tuple[int, ...] = (8, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512)

_GIB = 1024**3

# The GPU memory ceiling the matrix assumes on Apple silicon: the host probe's
# own fallback basis (`host_profile._FALLBACK_CEILING_FRACTION`, 75% of
# unified memory), which `recommended_artifact_fit` also uses. `--host`
# evaluates this Mac's measured ceiling instead.
_APPLE_CEILING_SOURCE = "ram_75pct"

# Reference machines of the classes that are not Apple silicon. `variants` are
# the other platforms the class label covers: tests assert each gives the same
# answers as the reference, so the label never over-claims.
MACHINE_CLASSES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "nvidia",
        "label": "Linux or Windows with an NVIDIA GPU",
        "reference": "x86_64, NVIDIA GPU with 24 GB of memory, 64 GB of RAM",
        "host": {"os": "linux", "arch": "x86_64", "accelerator": "cuda", "unified_memory": False,
                 "ram_bytes": 64 * _GIB, "vram_bytes": 24 * _GIB, "ceiling_bytes": 24 * _GIB,
                 "ceiling_source": "cuda_total", "gpu_count": 1},
        "variants": ({"os": "windows", "arch": "x86_64"},),
    },
    {
        "id": "cpu",
        "label": "Linux or Windows, processor only",
        "reference": "x86_64, no supported GPU, 16 GB of RAM",
        "host": {"os": "linux", "arch": "x86_64", "accelerator": "none", "unified_memory": False,
                 "ram_bytes": 16 * _GIB, "vram_bytes": None, "ceiling_bytes": 12 * _GIB,
                 "ceiling_source": "ram_75pct", "gpu_count": 0},
        "variants": ({"os": "windows", "arch": "x86_64"}, {"os": "linux", "arch": "arm64"}),
    },
    {
        "id": "intel_mac",
        "label": "Intel Mac",
        "reference": "macOS x86_64, 16 GB of RAM",
        "host": {"os": "darwin", "arch": "x86_64", "accelerator": "none", "unified_memory": False,
                 "ram_bytes": 16 * _GIB, "vram_bytes": None, "ceiling_bytes": 12 * _GIB,
                 "ceiling_source": "ram_75pct", "gpu_count": 0},
        "variants": (),
    },
)

# What runs each recommended provider, in words.
_ENGINE_LABEL = {
    "mlx": "MLX (AbstractCore)",
    "lmstudio": "LM Studio",
    "ollama": "Ollama",
    "supertonic": "Supertonic on ONNX Runtime (AbstractVoice)",
    "mlx-gen": "MLX-Gen (AbstractVision)",
    "faster-whisper": "faster-whisper on CTranslate2 (AbstractVoice)",
    "acestep": "ACE-Step on Diffusers and PyTorch (AbstractMusic)",
}


def _device(provider: str, accelerator: str) -> str:
    """Where the engine computes on this kind of host (each engine's own rule)."""

    gpu = {"metal": "Apple GPU (Metal)", "cuda": "NVIDIA GPU (CUDA)"}.get(accelerator)
    if provider in ("mlx", "mlx-gen"):
        return "Apple GPU (Metal)"
    if provider in ("lmstudio", "ollama"):
        return gpu or "processor"
    if provider == "supertonic":
        # abstractvoice runs Supertonic on ONNX Runtime's CPUExecutionProvider.
        return "processor"
    if provider == "faster-whisper":
        # abstractvoice `best_faster_whisper_device()`: CUDA or CPU, never MPS.
        return "NVIDIA GPU (CUDA)" if accelerator == "cuda" else "processor"
    if provider == "acestep":
        # abstractmusic acestep: CUDA, then MPS (bfloat16), then CPU (float32).
        return {"metal": "Apple GPU (MPS, bfloat16)", "cuda": "NVIDIA GPU (CUDA)"}.get(accelerator, "processor (float32)")
    raise ValueError(f"recommended provider {provider!r} has no device rule")


def _device_notes(provider: str, accelerator: str) -> List[str]:
    """Facts about running there that the numbers alone do not say."""

    if provider == "acestep" and accelerator not in ("metal", "cuda"):
        return ["On the processor AbstractMusic runs it in float32: about twice the memory shown."]
    return []


def _artifact_facts(
    provider: str, artifact: str, host: Mapping[str, Any], fit: Optional[Mapping[str, Any]], *, text: bool = False
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """`(facts, row, fit)` for one recommended artifact: the entry's catalog
    fields (name, sizes, fit verdict), its seed row and its fit block (the
    catalog's own, `recommended_artifact_fit`, unless the caller has it).
    `text`: a text model, whose tight Apple silicon fit also carries the
    raised-limit command and the context it gives (`context`)."""

    from .model_catalog import (
        _companions,
        _seed_row_and_artifact,
        catalog_id_for,
        measured_context_note,
        recommended_artifact_fit,
    )

    row_id = catalog_id_for(provider, artifact)
    if row_id is None:
        raise LookupError(f"recommended artifact {provider}:{artifact} is not in the catalog")
    row, art = _seed_row_and_artifact(row_id, provider, artifact)
    if fit is None:
        fit = recommended_artifact_fit(provider, artifact, host)["fit"]
    size = art.get("download_bytes") if isinstance(art.get("download_bytes"), int) else None
    _repos, companion_bytes = _companions(provider, artifact)
    if isinstance(size, int) and isinstance(companion_bytes, int):
        size += companion_bytes
    verdict = fit.get("verdict")
    raised = (
        fit.get("raised_limit")
        if text and verdict == "tight" and isinstance(fit.get("raised_limit"), Mapping)
        else None
    )
    if verdict == "needs_gpu_limit":
        command = (fit.get("gpu_limit") or {}).get("command")
    else:
        # A tight Apple silicon fit: the highest safe GPU memory limit, for
        # more context (`model_fit.raised_limit`).
        command = raised.get("command") if raised else None
    context = None
    if text and (raised or fit.get("small_context")):
        # No estimated token count: only a MEASURED one (`MEASURED_CONTEXT`).
        context = {
            "small": bool(fit.get("small_context")),
            "measured": measured_context_note(row_id, raised),
        }
    facts = {
        "catalog_id": row_id,
        "display_name": row.get("display_name") or row_id,
        "download_bytes": size,
        "memory_need_bytes": fit.get("need_bytes") if isinstance(fit.get("need_bytes"), int) else None,
        "memory_need_source": "measured" if isinstance((art.get("resident") or {}).get("bytes"), int) else "estimated",
        "fit": verdict,
        "gpu_limit_command": command,
        "context": context,
    }
    return facts, row, dict(fit)


def _entry(cap: Tuple[str, str, str, Tuple[str, ...]], **fields: Any) -> Dict[str, Any]:
    cid, route, label, tasks = cap
    out: Dict[str, Any] = {
        "capability": cid,
        "label": label,
        "route": route,
        "tasks": list(tasks),
        "status": None,
        "starter": False,
        "provider": None,
        "engine": None,
        "device": None,
        "model": None,
        "artifact": None,
        "download_provider": None,
        "catalog_id": None,
        "display_name": None,
        "download_bytes": None,
        "memory_need_bytes": None,
        "memory_need_source": None,
        "fit": None,
        "gpu_limit_command": None,
        "context": None,
        "covered_by": None,
        "smaller_canvas": None,
        "reason": None,
        "warning": None,
        "notes": [],
    }
    out.update(fields)
    return out


def _host_or_probe(host: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if host is not None:
        return dict(host)
    from ..utils.host_profile import host_profile

    return host_profile()


def recommended_models(host: Optional[Mapping[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """THE recommendation for every capability on `host` (default: this machine).

    Returns `{capability_id: entry}` in `RECOMMENDATION_CAPABILITIES` order.
    Every entry has a `status`:

      recommended  run `provider`/`model` (download `artifact` with
                   `download_provider`); `starter` says whether the
                   fresh-install seed, `apply-recommended` and `models download
                   --recommended` act on it
      covered      served by another capability's model (`covered_by`): image
                   input is read by the recommended text model
      unavailable  nothing recommended runs on this host; `reason` says why and
                   what to use instead. The fields still name the model the
                   recommendation would be, so a listing can say what is missing

    `fit` is the catalog's verdict for this host (`fits`, `tight`, `too_large`,
    `partial_offload`, `needs_gpu_limit`, `unknown`), with the admin command in
    `gpu_limit_command` for `needs_gpu_limit`; `warning` is the sentence a
    surface shows for a verdict that deserves one. Text keeps its rule: a tier
    that may not fit is still the tier, with its warning. Fit-gated rows
    (image, video, music) are `unavailable` where they do not fit.
    """

    from .capability_defaults import (
        RECOMMENDED_MODELS,
        _full_recommendation,
        _unavailable_reasons,
        image_input_unavailable_reason,
        recommended_route_unavailable_reason,
    )
    from .manager import model_supports_input
    from .model_catalog import _fit_warning, recommended_text_model, smaller_canvas_fit

    profile = _host_or_probe(host)
    accelerator = str(profile.get("accelerator") or "none")
    routes, downloads = _full_recommendation(profile, starter_only=False)
    unavailable = _unavailable_reasons(routes, downloads, profile)
    caps = {cap[0]: cap for cap in RECOMMENDATION_CAPABILITIES}
    out: Dict[str, Dict[str, Any]] = {}

    text = recommended_text_model(profile)
    text_reason = unavailable.get("input.text")
    text_facts, _row, _fit = _artifact_facts(text["provider"], text["artifact"], profile, text.get("fit") or {}, text=True)
    out["text"] = _entry(
        caps["text"],
        status="unavailable" if text_reason else "recommended",
        starter=True,
        provider=text["provider"],
        engine=_ENGINE_LABEL[text["provider"]],
        device=_device(text["provider"], accelerator),
        model=text["model"],
        artifact=text["artifact"],
        download_provider=text["provider"],
        reason=text_reason,
        warning=text.get("warning"),
        basis=text["basis"],
        tier=text["tier"],
        **text_facts,
    )

    if model_supports_input(str(text["model"]), "image"):
        out["vision"] = _entry(
            caps["vision"],
            status="unavailable" if text_reason else "covered",
            starter=True,
            covered_by="text",
            provider=text["provider"],
            engine=out["text"]["engine"],
            device=out["text"]["device"],
            model=text["model"],
            artifact=text["artifact"],
            download_provider=text["provider"],
            reason=text_reason,
            **{k: text_facts[k] for k in ("catalog_id", "display_name", "download_bytes", "memory_need_bytes",
                                          "memory_need_source", "fit", "gpu_limit_command", "context")},
        )
    else:
        out["vision"] = _entry(
            caps["vision"],
            status="unavailable",
            reason=image_input_unavailable_reason(routes["input.text"], None),
        )

    for cid, route_key, _label, _tasks in RECOMMENDATION_CAPABILITIES:
        if cid in out:
            continue
        rec = RECOMMENDED_MODELS[route_key]
        route, download = routes[route_key], downloads[route_key]
        provider = str(route.provider)
        facts, row, fit = _artifact_facts(download["provider"], download["artifact"], profile, None)
        reason = unavailable.get(route_key)
        smaller = None
        if reason and facts["fit"] not in ("fits", "tight") and not recommended_route_unavailable_reason(provider, profile, route_key):
            # The default canvas does not fit, the engine runs: the measured
            # smaller size it still runs at, if any.
            got = smaller_canvas_fit(download["provider"], download["artifact"], profile)
            if got is not None:
                smaller = {"canvas": got["canvas"], "memory_need_bytes": got["fit"]["need_bytes"],
                           "memory_need_source": "measured", "fit": got["fit"]["verdict"]}
        out[cid] = _entry(
            caps[cid],
            status="unavailable" if reason else "recommended",
            starter=rec.starter,
            provider=provider,
            engine=_ENGINE_LABEL[provider],
            device=_device(provider, accelerator),
            model=route.model,
            artifact=download["artifact"],
            download_provider=download["provider"],
            reason=reason,
            smaller_canvas=smaller,
            warning=None if reason else _fit_warning(row, fit),
            notes=_device_notes(provider, accelerator),
            **facts,
        )
    return {cap[0]: out[cap[0]] for cap in RECOMMENDATION_CAPABILITIES}


# ---------------------------------------------------------------------------
# The matrix: every machine class x every capability
# ---------------------------------------------------------------------------


def _apple_host(gib: int) -> Dict[str, Any]:
    ram = gib * _GIB
    return {
        "os": "darwin", "arch": "arm64", "accelerator": "metal", "unified_memory": True,
        "ram_bytes": ram, "vram_bytes": None, "ceiling_bytes": int(0.75 * ram),
        "ceiling_source": _APPLE_CEILING_SOURCE, "gpu_count": 1,
    }


def _signature(entries: Mapping[str, Mapping[str, Any]]) -> Tuple[Any, ...]:
    """What must be equal for two memory sizes to share one band."""

    return tuple(
        (cid, e["status"], e["provider"], e["artifact"], e["fit"], e["gpu_limit_command"],
         tuple(sorted((e["context"] or {}).items())),
         (e["smaller_canvas"] or {}).get("canvas"), (e["smaller_canvas"] or {}).get("fit"))
        for cid, e in entries.items()
    )


def _gib_list(sizes: Sequence[int]) -> str:
    words = [f"{s} GB" for s in sizes]
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " or " + words[-1]


def _platform(host: Mapping[str, Any]) -> Dict[str, Any]:
    return {"os": host["os"], "arch": host["arch"], "accelerator": host["accelerator"]}


def recommendation_matrix() -> Dict[str, Any]:
    """Every machine class x every capability: the `model_recommendations_v1` export.

    `classes[]` lists the Apple silicon bands first (one per run of memory
    sizes with identical answers, `memory_gib` = the sizes it covers), then
    `MACHINE_CLASSES`. Each class carries `entries` (`recommended_models`
    on its reference host) keyed by capability id.
    """

    from .. import __version__

    # Apple silicon: evaluate every size, then merge runs of identical answers.
    runs: List[Tuple[List[int], Dict[str, Any], Dict[str, Dict[str, Any]]]] = []
    for gib in APPLE_MEMORY_SIZES_GIB:
        host = _apple_host(gib)
        entries = recommended_models(host)
        if runs and _signature(runs[-1][2]) == _signature(entries):
            runs[-1][0].append(gib)
            continue
        runs.append(([gib], host, entries))
    classes: List[Dict[str, Any]] = []
    for i, (sizes, host, entries) in enumerate(runs):
        classes.append(
            {
                "id": f"apple_silicon_{sizes[0]}gb",
                "family": "apple_silicon",
                "label": f"Apple silicon Mac, {_gib_list(sizes)}",
                "reference": (
                    f"unified memory {_gib_list(sizes)}; fit verdicts assume a GPU memory limit of 75% of "
                    "unified memory (`models recommendations --host` reads this Mac's own)"
                ),
                "platform": _platform(host),
                "memory_gib": sizes,
                "memory_gib_min": sizes[0],
                # The next band's first size: this band answers every Mac with
                # memory_gib_min <= memory < memory_gib_below (None: no bound).
                "memory_gib_below": runs[i + 1][0][0] if i + 1 < len(runs) else None,
                "entries": entries,
            }
        )
    for spec in MACHINE_CLASSES:
        host = dict(spec["host"])
        classes.append(
            {
                "id": spec["id"],
                "family": spec["id"],
                "label": spec["label"],
                "reference": spec["reference"],
                "platform": _platform(host),
                "variants": [dict(v) for v in spec["variants"]],
                "entries": recommended_models(host),
            }
        )
    return {
        "schema": RECOMMENDATIONS_SCHEMA,
        "abstractcore_version": __version__,
        "capabilities": [
            {"id": cid, "route": route, "label": label, "tasks": list(tasks)}
            for cid, route, label, tasks in RECOMMENDATION_CAPABILITIES
        ],
        "classes": classes,
    }


def host_recommendations(host: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """`recommended_models` for this machine (the full host probe), as an export."""

    from .. import __version__
    from .model_catalog import _memory_gib

    profile = _host_or_probe(host)
    memory = _memory_gib(profile)
    return {
        "schema": RECOMMENDATIONS_SCHEMA,
        "abstractcore_version": __version__,
        "host": dict(
            _platform({"os": profile.get("os"), "arch": profile.get("arch"), "accelerator": profile.get("accelerator")}),
            memory_gib=round(memory, 2) if memory is not None else None,
            ceiling_bytes=profile.get("ceiling_bytes"),
            ceiling_source=profile.get("ceiling_source"),
        ),
        "entries": recommended_models(profile),
    }


# ---------------------------------------------------------------------------
# Markdown rendering (docs and website share it)
# ---------------------------------------------------------------------------


def _gb(value: Any) -> str:
    if not isinstance(value, int):
        return "unknown"
    if value < 1024**3:
        return f"{value / 1024**2:.0f} MiB"
    return f"{value / 1024**3:.1f} GiB"


_FIT_WORDS = {
    "fits": "fits",
    "tight": "fits, tightly",
    "too_large": "may not fit",
    "partial_offload": "partly on the processor",
    "needs_gpu_limit": "fits after raising the GPU memory limit",
    "unknown": "unknown",
    None: "",
}


def _cell_model(e: Mapping[str, Any]) -> str:
    if e["status"] == "covered":
        return f"the text model (`{e['model']}`)"
    return f"`{e['model']}`" if e["model"] == e["artifact"] else f"`{e['model']}` (download `{e['artifact']}`)"


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|")


def render_markdown(matrix: Mapping[str, Any]) -> str:
    """The docs tables: one per capability, one row per machine class."""

    lines: List[str] = []
    for cap in matrix["capabilities"]:
        cid = cap["id"]
        lines.append(f"### {cap['label']}")
        lines.append("")
        lines.append(f"Route `{cap['route']}` ({', '.join(t.replace('_', ' ') for t in cap['tasks'])}).")
        lines.append("")
        lines.append("| Machine | Recommended model | Engine, device | Download | Memory need | Fit |")
        lines.append("|---|---|---|---|---|---|")
        notes: List[str] = []
        for cls in matrix["classes"]:
            e = cls["entries"][cid]
            machine = _md_escape(cls["label"])
            if e["status"] == "unavailable" and e["smaller_canvas"]:
                sc = e["smaller_canvas"]
                w, h, frames = sc["canvas"].split("x")
                lines.append(
                    f"| {machine} | At the default canvas: not available, it needs more memory. At {w}x{h} "
                    f"({frames} frames): {_cell_model(e)}, set it yourself | {_md_escape(e['engine'])}, {e['device']} | "
                    f"{_gb(e['download_bytes'])} | {_gb(sc['memory_need_bytes'])} at {w}x{h} (measured with AbstractVision/mlx-gen) | "
                    f"{_FIT_WORDS.get(sc['fit'], sc['fit'])} at {w}x{h} |"
                )
                continue
            if e["status"] == "unavailable":
                lines.append(f"| {machine} | Not available: {_md_escape(e['reason'])} | | | | |")
                continue
            fit = _FIT_WORDS.get(e["fit"], str(e["fit"]))
            ctx = e.get("context") or {}
            if ctx.get("small"):
                fit = "tight: runs with a small context by default"
            if e["gpu_limit_command"] and e["fit"] == "tight":
                fit += f"; {ctx['measured']}" if ctx.get("measured") else f"; more context after `{e['gpu_limit_command']}`"
            elif e["gpu_limit_command"]:
                fit = f"{fit}: `{e['gpu_limit_command']}`"
            need = _gb(e["memory_need_bytes"])
            if e["memory_need_source"] == "measured":
                # The engine's measured figure, not the model's minimum.
                need += " (measured with AbstractVision/mlx-gen)"
            lines.append(
                f"| {machine} | {_cell_model(e)} | {_md_escape(e['engine'])}, {e['device']} | "
                f"{_gb(e['download_bytes'])} | {need} | {fit} |"
            )
            notes.extend(f"- {cls['label']}: {n}" for n in e.get("notes") or [])
        lines.append("")
        if notes:
            lines.extend(notes)
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def iter_entries(matrix: Mapping[str, Any]) -> Iterable[Tuple[Mapping[str, Any], str, Mapping[str, Any]]]:
    """`(class, capability_id, entry)` for every cell of a matrix."""

    for cls in matrix["classes"]:
        for cid, entry in cls["entries"].items():
            yield cls, cid, entry


# ---------------------------------------------------------------------------
# `abstractcore models recommendations`
# ---------------------------------------------------------------------------


def add_recommendations_parser(sub: Any) -> None:
    """`abstractcore models recommendations [--host] [--json|--markdown] [--output PATH]`."""

    parser = sub.add_parser(
        "recommendations",
        help="The recommended model for every capability, per machine class (or for this machine)",
        description=(
            "Print the recommended model for every capability (text, image input, speech output, "
            "speech input, image, video, music) on every machine class: Apple silicon by unified "
            "memory, Linux or Windows with an NVIDIA GPU, processor-only Linux or Windows, Intel Macs. "
            "Unavailable cells say why. Reads only: nothing is probed but the catalog, nothing is "
            "downloaded. --host answers for this machine instead."
        ),
    )
    parser.add_argument("--host", action="store_true", help="Answer for this machine (full host probe) instead of the matrix")
    fmt = parser.add_mutually_exclusive_group()
    fmt.add_argument("--json", action="store_true", help="Emit the model_recommendations_v1 JSON")
    fmt.add_argument("--markdown", action="store_true", help="Emit the Markdown tables the docs embed")
    parser.add_argument("--output", default=None, help="Write to this file instead of standard output")
    parser.set_defaults(func=handle_recommendations)


def _text_lines(entries: Mapping[str, Mapping[str, Any]]) -> List[str]:
    lines: List[str] = []
    for e in entries.values():
        if e["status"] == "unavailable":
            lines.append(f"  {e['label']}: not available ({e['reason']})")
            continue
        what = f"covered by the text model ({e['model']})" if e["status"] == "covered" else f"{e['provider']} {e['model']}"
        fit = f", {_FIT_WORDS.get(e['fit'], e['fit'])}" if e["fit"] else ""
        if (e.get("context") or {}).get("small"):
            fit = ", tight: runs with a small context by default"
        lines.append(f"  {e['label']}: {what} [{e['device']}; download {_gb(e['download_bytes'])}; needs {_gb(e['memory_need_bytes'])}{fit}]")
        if e.get("warning"):
            lines.append(f"    {e['warning']}")
    return lines


def handle_recommendations(args: Any) -> int:
    import json

    payload = host_recommendations() if getattr(args, "host", False) else recommendation_matrix()
    if getattr(args, "json", False):
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    elif getattr(args, "markdown", False):
        if getattr(args, "host", False):
            raise ValueError("--markdown renders the machine-class matrix; use --json with --host")
        text = render_markdown(payload)
    elif getattr(args, "host", False):
        h = payload["host"]
        mem = f", {h['memory_gib']} GiB" if h.get("memory_gib") is not None else ""
        text = "\n".join([f"This machine ({h['os']} {h['arch']}, {h['accelerator']}{mem}):", *_text_lines(payload["entries"])]) + "\n"
    else:
        blocks = [f"{cls['label']}:\n" + "\n".join(_text_lines(cls["entries"])) for cls in payload["classes"]]
        text = "\n\n".join(blocks) + "\n"
    output = getattr(args, "output", None)
    if output:
        from pathlib import Path

        Path(output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0
