"""Shared capability routing defaults.

This module defines the small, JSON-safe contract used by AbstractCore and
AbstractGateway to describe default provider/model routing for framework
capabilities.  It intentionally does not know how to load a model or invoke a
plugin; it only normalizes the configuration shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


CAPABILITY_DEFAULTS_VERSION = 1

CAPABILITY_KINDS = ("input", "output", "embedding", "rerank")
CAPABILITY_MODALITIES = ("text", "image", "video", "voice", "sound", "music", "scene3d")
CAPABILITY_ROUTE_TASKS = (
    "text_to_image",
    "image_to_image",
    "image_upscale",
    "text_to_video",
    "image_to_video",
    "text_to_scene3d",
    "image_to_scene3d",
)

_KIND_ALIASES = {
    "in": "input",
    "inputs": "input",
    "understand": "input",
    "understanding": "input",
    "out": "output",
    "outputs": "output",
    "generate": "output",
    "generation": "output",
    "embed": "embedding",
    "embeddings": "embedding",
    "vector": "embedding",
    "vectors": "embedding",
    "rank": "rerank",
    "ranking": "rerank",
    "reranker": "rerank",
    "rerankers": "rerank",
}

_MODALITY_ALIASES = {
    "speech": "voice",
    "tts": "voice",
    "stt": "voice",
    "sfx": "sound",
    "sound_effect": "sound",
    "sound_effects": "sound",
    "audio": "sound",
    "3d": "scene3d",
    "3d_scene": "scene3d",
    "scene_3d": "scene3d",
    "scene-3d": "scene3d",
    "scene": "scene3d",
}

_TASK_ALIASES = {
    "t2i": "text_to_image",
    "image_generation": "text_to_image",
    "generate_image": "text_to_image",
    "i2i": "image_to_image",
    "image_edit": "image_to_image",
    "edit_image": "image_to_image",
    "upscale": "image_upscale",
    "upscaler": "image_upscale",
    "upscale_image": "image_upscale",
    "image_upscaling": "image_upscale",
    "t2v": "text_to_video",
    "video_generation": "text_to_video",
    "generate_video": "text_to_video",
    "i2v": "image_to_video",
    "video_from_image": "image_to_video",
    "image_video": "image_to_video",
    "t23d": "text_to_scene3d",
    "text2scene3d": "text_to_scene3d",
    "text_to_3d": "text_to_scene3d",
    "i23d": "image_to_scene3d",
    "image2scene3d": "image_to_scene3d",
    "image_to_3d": "image_to_scene3d",
    "image_to_scene": "image_to_scene3d",
}


@dataclass(frozen=True)
class CapabilityDefaultSpec:
    """One routable framework capability row."""

    key: str
    kind: str
    modality: str
    label: str
    task: str
    package_hint: Optional[str] = None
    option_examples: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "key": self.key,
            "kind": self.kind,
            "modality": self.modality,
            "label": self.label,
            "task": self.task,
        }
        if self.package_hint:
            out["package_hint"] = self.package_hint
        if self.option_examples:
            out["option_examples"] = dict(self.option_examples)
        return out


@dataclass
class CapabilityRouteDefault:
    """Default routing target for one capability route."""

    provider: Optional[str] = None
    model: Optional[str] = None
    base_url: Optional[str] = None
    reasoning: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

    def configured(self) -> bool:
        return bool(self.provider or self.model or self.base_url or self.reasoning or self.options)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.provider:
            out["provider"] = self.provider
        if self.model:
            out["model"] = self.model
        if self.base_url:
            out["base_url"] = self.base_url
        if self.reasoning:
            out["reasoning"] = self.reasoning
        if self.options:
            out["options"] = dict(self.options)
        return out


@dataclass
class CapabilityDefaultsConfig:
    """Versioned collection of capability routing defaults."""

    version: int = CAPABILITY_DEFAULTS_VERSION
    routes: Dict[str, CapabilityRouteDefault] = field(default_factory=dict)
    # Provenance marker for routes written by the fresh-install seed
    # (`RECOMMENDED_CAPABILITY_DEFAULT_ROUTES`). Informational only: it never
    # gates behaviour (file existence gates the seed), it lets surfaces say
    # "recommended default" instead of implying an operator chose the value.
    seeded: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "version": int(self.version or CAPABILITY_DEFAULTS_VERSION),
            "routes": {key: route.to_dict() for key, route in sorted(self.routes.items()) if route.configured()},
        }
        if self.seeded:
            out["seeded"] = str(self.seeded)
        return out


def normalize_kind(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    raw = _KIND_ALIASES.get(raw, raw)
    if raw not in CAPABILITY_KINDS:
        raise ValueError(
            f"Unknown capability route kind: {value!r}. "
            "Expected input, output, embedding, or rerank."
        )
    return raw


def normalize_modality(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    raw = _MODALITY_ALIASES.get(raw, raw)
    if raw not in CAPABILITY_MODALITIES:
        raise ValueError(
            f"Unknown capability modality: {value!r}. "
            "Expected text, image, video, voice, sound, music, or scene3d."
        )
    return raw


def normalize_task(value: Any) -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    raw = _TASK_ALIASES.get(raw, raw)
    if raw not in CAPABILITY_ROUTE_TASKS:
        raise ValueError(
            f"Unknown capability route task: {value!r}. "
            "Expected text_to_image, image_to_image, image_upscale, text_to_video, image_to_video, "
            "text_to_scene3d, or image_to_scene3d."
        )
    return raw


def capability_route_key(kind: Any, modality: Any, task: Any = None) -> str:
    base = f"{normalize_kind(kind)}.{normalize_modality(modality)}"
    if task is None or str(task or "").strip() == "":
        return base
    return f"{base}.{normalize_task(task)}"


def split_capability_route(value: Any, modality: Any = None) -> Tuple[str, str]:
    if modality is not None:
        return normalize_kind(value), normalize_modality(modality)

    raw = str(value or "").strip()
    if "." in raw:
        left, right = raw.split(".", 1)
        return normalize_kind(left), normalize_modality(right)
    if ":" in raw:
        left, right = raw.split(":", 1)
        return normalize_kind(left), normalize_modality(right)
    raise ValueError("Capability route must be written as kind.modality, for example output.text.")


def split_capability_default_route(value: Any, modality: Any = None, task: Any = None) -> Tuple[str, str, Optional[str]]:
    """Split a persisted capability default route.

    Defaults may be broad (`output.image`) or task-specific
    (`output.image.image_to_image`). Model capability routes intentionally keep
    using `split_capability_route` so static model metadata stays broad.
    """

    if modality is not None:
        normalized_task = normalize_task(task) if task is not None and str(task or "").strip() else None
        return normalize_kind(value), normalize_modality(modality), normalized_task

    raw = str(value or "").strip()
    separator = "." if "." in raw else ":" if ":" in raw else ""
    if not separator:
        raise ValueError("Capability route must be written as kind.modality, for example output.text.")
    parts = [part.strip() for part in raw.replace(":", ".").split(".") if part.strip()]
    if len(parts) == 2:
        return normalize_kind(parts[0]), normalize_modality(parts[1]), None
    if len(parts) == 3:
        return normalize_kind(parts[0]), normalize_modality(parts[1]), normalize_task(parts[2])
    raise ValueError(
        "Capability default route must be written as kind.modality or kind.modality.task, "
        "for example output.image.image_to_image."
    )


def clean_capability_route_default(value: Any) -> CapabilityRouteDefault:
    if isinstance(value, CapabilityRouteDefault):
        return value
    data = value if isinstance(value, Mapping) else {}
    provider = _clean_optional_string(data.get("provider") or data.get("provider_id"))
    model = _clean_optional_string(data.get("model") or data.get("model_id"))
    base_url = _clean_optional_string(data.get("base_url"))
    reasoning = _clean_optional_string(data.get("reasoning"))
    options_raw = data.get("options")
    options = dict(options_raw) if isinstance(options_raw, Mapping) else {}

    # Backward-compatible convenience: unknown scalar fields become options so
    # plugin-specific defaults such as voice/profile are not lost.
    for key, raw in data.items():
        if key in {
            "provider",
            "provider_id",
            "model",
            "model_id",
            "base_url",
            "reasoning",
            "options",
            "key",
            "source",
            "kind",
            "modality",
            "task",
            "label",
            "package_hint",
            "option_examples",
        }:
            continue
        if isinstance(key, str) and key.strip():
            options.setdefault(key.strip(), raw)

    return CapabilityRouteDefault(
        provider=provider,
        model=model,
        base_url=base_url,
        reasoning=reasoning,
        options=options,
    )


RECOMMENDED_SEED_VERSION = "recommended-v2"

# What each seed version added to the one before it, applied once to a store
# an earlier seed wrote (`upgrade_recommended_seed`). v2 (AbstractCore 2.19.2):
# speech input, so a fresh install transcribes locally instead of falling
# through to OpenAI without a key.
RECOMMENDED_SEED_ADDITIONS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("recommended-v2", ("input.voice",)),
)

# THE RECOMMENDED MODEL PER CAPABILITY: one table, every surface.
#
# `RECOMMENDED_MODELS` holds, per capability route, the recommended route
# (provider/model, what the route stores), the download that fetches its
# weights, and whether it belongs to the fresh-install STARTER set. It is the
# full recommendation, NOT a host answer: every reader goes through the
# host-aware functions below (`recommended_capability_default_routes(host)`,
# `recommended_model_downloads(host)`, `recommended_unavailable_routes(host)`),
# and `recommendations.recommended_models(host)` / `recommendation_matrix()`
# render all of it per machine class (`abstractcore models recommendations`).
#
# Text is the exception to the table: its per-host pick (Apple silicon: the
# unified-memory tier) belongs to `model_catalog.recommended_text_model()`;
# the row below is the portable default that function starts from. Image
# input (vision) is not a row either: it is covered by `input.text` where the
# host's recommended text model reads images (every recommended text model
# does today), and reported unavailable where it does not, or where no text
# engine runs (`image_input_unavailable_reason`).
#
# STARTER rows (`starter=True`) are the fresh-install set (operator ruling
# 2026-08-01): a new install should WORK out of the box on the framework's
# recommended local stack rather than refuse until configured. The seed writes
# them ONLY when the config file does not exist yet -- never merged into an
# existing store (an operator who cleared a route meant it), and never on the
# corrupt-file fallback. `apply-recommended` and `models download
# --recommended` act on the same starter set: text, voice output, speech
# input, image and video. Rows with `starter=False` (music) are
# recommendations every listing shows, which no writer applies on its own: the operator sets them (`abstractcore config
# set-default`). Text stores at input.text (the canonical storage key;
# output.text derives).
#
# PER-ACCELERATOR PICKS (`by_accelerator`): a row whose best engine differs by
# the host's accelerator carries the other pick IN THE ROW, keyed by the host
# probe's `accelerator` (`cuda`, `metal`, ...). `route`/`download` stay the
# portable pick every other host starts from; `_full_recommendation` swaps in
# the accelerator's pick BEFORE the engine-support filter and the memory gate,
# so the pick is judged by the same rules as every other row. The row stays
# the one place its recommendation lives.
@dataclass(frozen=True)
class AcceleratorPick:
    """The route and download that replace a row's portable pick on one accelerator."""

    route: CapabilityRouteDefault
    download: Mapping[str, str]


@dataclass(frozen=True)
class RecommendedModel:
    """One capability's recommendation: the route, its download, starter or not."""

    route: CapabilityRouteDefault
    download: Mapping[str, str]
    starter: bool
    by_accelerator: Mapping[str, AcceleratorPick] = field(default_factory=dict)

    def pick_for(self, host: Mapping[str, Any]) -> Tuple[CapabilityRouteDefault, Mapping[str, str]]:
        """`(route, download)` for `host`: its accelerator's pick, else the portable one."""

        pick = self.by_accelerator.get(str(host.get("accelerator") or ""))
        return (pick.route, pick.download) if pick is not None else (self.route, self.download)


RECOMMENDED_MODELS: Dict[str, RecommendedModel] = {
    # Text: the 4-BIT quantized build (operator ruling 2026-08-01). The ROUTE
    # stores the bare LM Studio id because that is what the server serves when
    # a single quant is installed; the 4-bit choice is pinned by the download
    # artifact reference, which is what actually fetches the weights. The
    # variant is LM Studio's GGUF Q4_K_M: LM Studio is recommended only off
    # Apple silicon (Linux, Windows), where its catalog has no `4bit` (MLX)
    # variant -- `lms get qwen/qwen3.5-9b@4bit` fails there with "Cannot find
    # variant 4bit" (measured on Linux + NVIDIA, framework backlog 0989).
    "input.text": RecommendedModel(
        route=CapabilityRouteDefault(
            provider="lmstudio", model="qwen/qwen3.5-9b",
            options={"speculation": {"mode": "native_mtp", "num_draft_tokens": 2,
                                     "require_acceleration": False}},
        ),
        download={"provider": "lmstudio", "artifact": "qwen/qwen3.5-9b@q4_k_m"},
        starter=True,
    ),
    "output.voice": RecommendedModel(
        route=CapabilityRouteDefault(provider="supertonic", model="supertonic-3"),
        download={"provider": "supertonic", "artifact": "supertonic-3"},
        starter=True,
    ),
    # Image: FLUX.2 [klein] 4B. Apple silicon (and the portable pick): the
    # 8-bit MLX-Gen build. An NVIDIA GPU (`cuda`): the same model's Diffusers
    # repo on AbstractVision's `diffusers` backend (the route provider the
    # image lane maps to that backend; device `auto` resolves to CUDA, float16).
    # Its 14.9 GiB of float16 weights do not fit a 16 GB card whole, so
    # AbstractVision (>= 0.3.32, `cpu_offload="auto"`) loads it with model CPU
    # offload there: measured 768x768 in ~17 s with an 8.3 GiB GPU peak on a
    # Quadro RTX 5000 16 GB (framework backlog 0989; the catalog artifact's
    # `resident` is that figure, so the memory gate judges it). Processor-only
    # hosts keep the portable pick and report it unavailable.
    "output.image": RecommendedModel(
        route=CapabilityRouteDefault(provider="mlx-gen", model="AbstractFramework/flux.2-klein-4b-8bit"),
        download={"provider": "mlx-gen", "artifact": "AbstractFramework/flux.2-klein-4b-8bit"},
        starter=True,
        by_accelerator={
            "cuda": AcceleratorPick(
                route=CapabilityRouteDefault(provider="diffusers", model="black-forest-labs/FLUX.2-klein-4B"),
                download={"provider": "diffusers", "artifact": "black-forest-labs/FLUX.2-klein-4B"},
            ),
        },
    ),
    # Video: Wan2.2 TI2V-5B, ONE checkpoint for text-to-video AND
    # image-to-video, so the modality cell answers both tasks. It is the only
    # video model AbstractVision serves that is not a 40 GB A14B package; its
    # engine (MLX-Gen) is Apple silicon only and it needs ~16.6 GiB of MLX
    # memory at AbstractVision's default canvas (832x480x121, measured MLX
    # allocator peak for image-to-video, the larger of its two tasks;
    # text-to-video peaks at 16.3 GiB), so it is fit-gated (`_FIT_GATED_ROUTES`).
    "output.video": RecommendedModel(
        route=CapabilityRouteDefault(provider="mlx-gen", model="AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"),
        download={"provider": "mlx-gen", "artifact": "AbstractFramework/wan2.2-ti2v-5b-diffusers-8bit"},
        starter=True,
    ),
    # Speech input: AbstractVoice's faster-whisper engine (CTranslate2: CUDA on
    # an NVIDIA GPU, the processor elsewhere -- it has no Metal backend) with
    # AbstractVoice's own default model, `base`. faster-whisper resolves
    # `base` to the Hugging Face repo below in the standard Hugging Face cache,
    # which is where `models download huggingface <repo>` puts it. A STARTER
    # row (2.19.2, framework rehearsal 0.6.3): without it a fresh install's
    # speech-to-text fell through to OpenAI and failed for want of an API key,
    # although the installer ships faster-whisper (`abstractvoice[stt]`).
    "input.voice": RecommendedModel(
        route=CapabilityRouteDefault(provider="faster-whisper", model="base"),
        download={"provider": "huggingface", "artifact": "Systran/faster-whisper-base"},
        starter=True,
    ),
    # Music: AbstractMusic's `acestep` backend (Diffusers AceStepPipeline on
    # PyTorch: CUDA, Apple MPS in bfloat16, or the processor in float32) with
    # the one checkpoint AbstractMusic marks recommended and validated,
    # ACE-Step 1.5 XL turbo. AbstractMusic loads it from the local Hugging
    # Face cache only, so it must be downloaded first. Fit-gated like video.
    "output.music": RecommendedModel(
        route=CapabilityRouteDefault(provider="acestep", model="ACE-Step/acestep-v15-xl-turbo-diffusers"),
        download={"provider": "diffusers", "artifact": "ACE-Step/acestep-v15-xl-turbo-diffusers"},
        starter=False,
    ),
}

# The STARTER views of the table: what the fresh-install seed,
# `apply-recommended` and `models download --recommended` act on. Derived,
# never edited: `RECOMMENDED_MODELS` is the one place a recommendation lives.
# Like the table, never read directly for a host answer: use the host-aware
# functions below. They hold the PORTABLE picks (no `by_accelerator` swap).
RECOMMENDED_CAPABILITY_DEFAULT_ROUTES: Dict[str, CapabilityRouteDefault] = {
    key: rec.route for key, rec in RECOMMENDED_MODELS.items() if rec.starter
}
RECOMMENDED_MODEL_DOWNLOADS: Dict[str, Dict[str, str]] = {
    key: dict(rec.download) for key, rec in RECOMMENDED_MODELS.items() if rec.starter
}


# HOST-AWARE VIEWS. The two tables above are the full recommended stack; no
# host gets them verbatim. Per host:
#   - the text row is chosen by `model_catalog.recommended_text_model()` (its
#     ONE owner): the unified-memory tier on Apple silicon (operator ruling
#     2026-09-24), the portable LM Studio build elsewhere, and the same model's
#     Ollama build where LM Studio has no build (Intel Macs);
#   - a row with a pick for the host's accelerator (`by_accelerator`) uses it:
#     image generation on an NVIDIA GPU is Diffusers, not MLX-Gen;
#   - every row whose engine cannot run on the host is DROPPED, never written
#     as a route that fails at first use. It is reported instead, with the
#     reason, by `recommended_unavailable_routes()` -- the grid then shows the
#     row unset with that reason and apply-recommended reports it;
#   - a FIT-GATED row (`_FIT_GATED_ROUTES`: image, video, music) is also
#     dropped, the same way, where the catalog's fit estimate says its model
#     does not fit the host's memory (a CONFIGURED route on those rows is
#     judged by the same verdict: `configured_routes_unavailable`). Text is
#     deliberately NOT gated (a tier that does not fit is still the tier, with
#     a warning -- `recommended_text_model`): every install needs a text model,
#     while a generation model is an extra that is only worth writing where it
#     can run at all.
# Every writer and every download surface reads these functions, never the
# tables directly, so a host seeds, applies, downloads and displays the same
# pick.


# Where the engine behind each RECOMMENDED provider runs: the key into the ONE
# host-support matrix (`engines._support`, which cites each vendor's builds).
# A provider the recommendation names but this table does not know raises: a
# silent "runs everywhere" default is exactly how an Apple-only engine reached
# Linux hosts.
#   mlx         MLX (Metal): Apple silicon only.
#   mlx-gen     MLX-Gen runs on MLX: Apple silicon only, same rule.
#   lmstudio    LM Studio: x86_64/arm64 Linux and Windows, Apple-silicon macOS.
#   ollama      Ollama: x86_64/arm64 Linux and Windows, macOS.
#   supertonic  abstractvoice's Supertonic 3 runtime is ONNX Runtime on
#               `CPUExecutionProvider` (abstractvoice/supertonic/runtime.py):
#               x86_64/arm64 Linux and Windows, macOS.
#   faster-whisper  abstractvoice's speech input runs on CTranslate2
#               (abstractvoice/adapters/stt_faster_whisper.py): macOS,
#               x86_64/arm64 Linux, x86_64 Windows.
#   acestep     abstractmusic's ACE-Step backend is a Diffusers pipeline on
#               PyTorch (abstractmusic/backends/acestep.py): Apple-silicon
#               macOS, x86_64/arm64 Linux, x86_64 Windows.
#   diffusers   abstractvision's Diffusers backend is PyTorch
#               (abstractvision/backends/huggingface_diffusers.py), the same
#               builds as acestep. Recommended on `cuda` hosts only
#               (`RECOMMENDED_MODELS["output.image"].by_accelerator`).
_RECOMMENDED_PROVIDER_ENGINE = {
    "mlx": "mlx",
    "mlx-gen": "mlx",
    "lmstudio": "lmstudio",
    "ollama": "ollama",
    "supertonic": "onnxruntime",
    "faster-whisper": "ctranslate2",
    "acestep": "torch",
    "diffusers": "torch",
}
# How a provider that runs ON another engine words the engine's refusal.
_ENGINE_VIA = {
    "supertonic": "Supertonic voice runs on ONNX Runtime (CPU), and ",
    "faster-whisper": "faster-whisper speech input runs on CTranslate2, and ",
    "acestep": "ACE-Step music generation runs on PyTorch, and ",
    "diffusers": "Diffusers image generation runs on PyTorch, and ",
}

# CONFIGURED routes whose engine runs INSIDE this process (AbstractCore's MLX
# provider, AbstractVision's MLX-Gen, AbstractVoice's Supertonic): such a route
# can only run on this computer, so a host without the engine makes it fail at
# first use, and every grid flags it (`route_unavailable`). Server providers
# (LM Studio, Ollama, vLLM, OpenAI-compatible) are deliberately absent: their
# route may address a server on another machine (a route or provider-profile
# `base_url`), so this host's builds say nothing about whether it runs. Cloud
# providers run anywhere.
_IN_PROCESS_PROVIDERS = frozenset({"mlx", "mlx-gen", "supertonic"})

# What an operator can do instead, per route, appended to the reason. Provider
# ids are abstractvision's own (`diffusers`, `sdcpp`). An NVIDIA GPU gets the
# Diffusers image pick itself (`by_accelerator`); a processor-only host does not
# (no measured processor run vouches for it), so the sentence names the engines.
# Video has NO local alternative off Apple silicon today: abstractvision's
# Diffusers text-to-video is disabled for its only model (CogVideoX-2b,
# `_TEMPORARILY_DISABLED_LOCAL_DIFFUSERS_TASKS`) and has no image-to-video,
# stable-diffusion.cpp raises for both, and its registry marks LTX-2,
# HunyuanVideo, CogVideoX 1.5, Mochi and SVD `backend: not_supported`. The
# one remaining path is abstractvision's OpenAI-compatible backend pointed at
# a video endpoint (its text_to_video/image_to_video paths). No built-in cloud
# video provider exists yet.
_VIDEO_NO_LOCAL_ENGINE = (
    "no other local engine in AbstractFramework generates video today (abstractvision's Diffusers video path "
    "is disabled, stable-diffusion.cpp has none); the remaining option is an OpenAI-compatible video endpoint "
    "(abstractvision openai-compatible backend)"
)
_UNAVAILABLE_NEXT_STEP = {
    # Holds by construction: the text pick is LM Studio's build only where LM
    # Studio runs or where Ollama does not run either
    # (`model_catalog.recommended_text_model`, basis `no_supported_engine`).
    "input.text": (
        "no local text engine AbstractFramework recommends (LM Studio, Ollama) runs on this host; set "
        "input.text to a cloud provider or to a text server on another machine (lmstudio, ollama or "
        "openai-compatible, with its base_url)"
    ),
    "output.video": _VIDEO_NO_LOCAL_ENGINE,
    # Engine ids are abstractvoice's (`transformers-asr`, `openai`).
    "input.voice": (
        "set input.voice to another speech-to-text engine: transformers-asr (AbstractVoice, PyTorch) "
        "or a cloud provider (openai)"
    ),
    # Backend ids are abstractmusic's (`acemusic`, `elevenlabs-music`).
    "output.music": "set output.music to a cloud music backend (acemusic or elevenlabs-music, with its API key)",
}
# output.image names only the three settings (operator ruling 2026-09-29). Apple
# silicon never reaches this (MLX-Gen runs there), nor does an NVIDIA GPU (its
# pick is Diffusers, which runs there). Linux: abstractcore[gpu] ships
# both local image engines; Windows x86_64: abstractcore[gpu] ships diffusers
# (stable-diffusion.cpp is a source build there, marked out; backlog 0988). An Intel
# Mac or Windows on ARM has no setting that installs them (`uv pip compile` of
# abstractcore[gpu] for x86_64-apple-darwin fails on torch). One sentence covers every
# non-Apple host so a machine-class row ("Linux or Windows ...") stays exact.
_UNAVAILABLE_NEXT_STEP["output.image"] = (
    "set output.image to a local image engine: diffusers, included with abstractcore[gpu] on Linux and "
    "Windows, or sdcpp (stable-diffusion.cpp), included with abstractcore[gpu] on Linux (on an Intel Mac or "
    "Windows on ARM they are not available with AbstractCore's install settings), or to a cloud image provider"
)


def _unavailable_next_step(key: Optional[str], host: Mapping[str, Any]) -> Optional[str]:
    """What an operator can do instead of an unavailable recommended row."""

    return _UNAVAILABLE_NEXT_STEP.get(key or "")


# Next step when the engine runs but the model does not fit (fit-gated rows).
_TOO_LARGE_NEXT_STEP = {
    "output.image": "use a Mac with more unified memory, or a cloud image provider",
    "output.video": (
        "use an Apple silicon Mac with more unified memory, or an OpenAI-compatible video endpoint "
        "(abstractvision openai-compatible backend)"
    ),
    "output.music": (
        "use a computer with more memory, or a cloud music backend (acemusic or elevenlabs-music, "
        "with its API key)"
    ),
}
# The same, where the host's accelerator words it differently (the key is
# `(accelerator, route)`): an NVIDIA GPU's image pick is judged against GPU
# memory, not a Mac's unified memory.
_TOO_LARGE_NEXT_STEP_BY_ACCELERATOR = {
    ("cuda", "output.image"): "use an NVIDIA GPU with more memory, or a cloud image provider",
}
# Recommended rows written only where the catalog's fit estimate (the same one
# the model browser's "fits this computer" filter uses: `fits` or `tight`)
# says the model fits this host's memory.
# Image joined 2026-09-28 (operator ruling: a recommendation must fit): FLUX.2
# klein 4B needs ~8.5 GiB, which an 8 GB Mac cannot give a model.
_FIT_GATED_ROUTES = frozenset({"output.image", "output.video", "output.music"})
# What each mlx-gen route generates, for the reason sentence.
_MLX_GEN_WORK = {"output.image": "image generation", "output.video": "video generation"}


def _host_platform(host: Mapping[str, Any]) -> Tuple[str, str, Optional[str]]:
    accelerator = host.get("accelerator")
    os_id = str(host.get("os") or "").strip().lower()
    arch = str(host.get("arch") or "").strip().lower()
    if accelerator == "metal":
        # `metal` IS Apple silicon (host_profile sets it for darwin/arm64 only).
        os_id, arch = os_id or "darwin", arch or "arm64"
    return os_id or "unknown", arch or "unknown", accelerator if isinstance(accelerator, str) else None


def recommended_route_unavailable_reason(
    provider: Any, host: Mapping[str, Any], key: Optional[str] = None
) -> Optional[str]:
    """Why a recommended provider cannot run on `host`, or None when it can.

    `key` (the route) only words the sentence ("MLX-Gen video generation").
    Engine support only: the memory gate of fit-gated routes is
    `recommended_unavailable_routes`.
    """

    from .engines import _support

    pid = str(provider or "").strip().lower()
    os_id, arch, accelerator = _host_platform(host)
    engine = _RECOMMENDED_PROVIDER_ENGINE.get(pid)
    if engine is None:
        raise ValueError(f"recommended provider {provider!r} has no host-support rule")
    ok, reason = _support(engine, os_id, arch, accelerator)
    if ok:
        return None
    if pid == "mlx-gen":
        route_key = str(key or "")
        work = _MLX_GEN_WORK.get(route_key) or _MLX_GEN_WORK.get(
            capability_route_broad_key(route_key) or "", "image generation"
        )
        return f"MLX-Gen {work} needs MLX, and {reason}"
    return f"{_ENGINE_VIA.get(pid, '')}{reason}"


def _gib_text(value: Any) -> str:
    return f"{float(value) / 1024**3:.1f} GiB"


def _fit_gate_reason(key: str, download: Mapping[str, str], host: Mapping[str, Any]) -> Optional[str]:
    """Why a fit-gated recommendation does not fit `host`, or None when it does.

    The verdict is the catalog's own (`model_catalog.recommended_artifact_fit`:
    measured run-time memory against the host ceiling), so the plan, the seed
    and the model browser's fit filter agree. An `unknown` verdict (no memory
    reading) is not a fit: nothing is written that cannot be vouched for.
    """

    return _fit_gate(key, download, host)[1]


def _fit_gate(key: str, download: Mapping[str, str], host: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
    """`(verdict, reason)` of the fit gate: the catalog verdict for `download`
    on `host`, and the reason sentence (None for `fits` / `tight`). ONE fit
    rule for the recommendation (`_fit_gate_reason`) and for configured routes
    (`configured_routes_unavailable`, which also accepts `needs_gpu_limit`)."""

    from .model_catalog import recommended_artifact_fit

    got = recommended_artifact_fit(download["provider"], download["artifact"], host)
    verdict = str(got["fit"].get("verdict") or "unknown")
    return verdict, _fit_gate_sentence(key, got, download, host)


def _fit_gate_sentence(
    key: str, got: Mapping[str, Any], download: Mapping[str, str], host: Mapping[str, Any]
) -> Optional[str]:
    fit = got["fit"]
    verdict = fit.get("verdict")
    if verdict in ("fits", "tight"):
        return None
    name = got["row"].get("display_name") or got["row"].get("id")
    step = _TOO_LARGE_NEXT_STEP_BY_ACCELERATOR.get((str(host.get("accelerator") or ""), key)) or _TOO_LARGE_NEXT_STEP.get(key)
    next_step = f"; {step}" if step else ""
    if verdict == "unknown" or not isinstance(fit.get("need_bytes"), int) or not isinstance(fit.get("usable_bytes"), int):
        return f"{name} needs a lot of memory and this computer's memory could not be measured{next_step}"
    # "measured" only where the seed carries a measured run-time peak
    # (`resident`, the video rows); otherwise the need is the estimate from
    # the download (music). A measured figure is the ENGINE's (`measured_with`:
    # AbstractVision/mlx-gen keeps the text encoder and VAE in memory), never
    # worded as the model's own minimum (operator ruling 2026-09-28).
    resident = (got.get("artifact") or {}).get("resident")
    if isinstance(resident, dict) and resident.get("measured_with"):
        basis = f"measured with {resident['measured_with']}; this engine's figure, not the model's minimum"
    else:
        basis = "measured" if isinstance(resident, dict) else "estimated"
    at_default = (
        " at its default canvas"
        if isinstance(resident, dict) and resident.get("smaller_canvases") and not resident.get("measured_with")
        else ""
    )
    if fit.get("accelerator") == "metal" and isinstance(fit.get("ceiling_bytes"), int):
        # Apple silicon: the GPU memory limit itself, then what is left after
        # MLX's working buffers (`model_fit`), never the remainder alone.
        room = (
            f"and macOS's GPU memory limit on this Mac is about {_gib_text(fit['ceiling_bytes'])}, about "
            f"{_gib_text(fit['usable_bytes'])} of it left for a model after working buffers"
        )
    else:
        room = f"and this computer can give a model about {_gib_text(fit['usable_bytes'])}"
    reason = f"{name} needs about {_gib_text(fit['need_bytes'])} of memory while it generates{at_default} ({basis}), {room}"
    if verdict == "needs_gpu_limit":
        # Not written (it needs an admin command first), but never a bare
        # "too large": the reason carries the command that makes it fit.
        from .model_catalog import gpu_limit_instruction

        return f"{reason}. {gpu_limit_instruction(fit)}"
    # Not written either (a route runs at the default canvas unless the
    # caller asks for less), but the smaller size it still runs at is said,
    # measured, so the operator can choose it.
    from .model_catalog import smaller_canvas_fit

    smaller = smaller_canvas_fit(download["provider"], download["artifact"], host)
    if smaller is not None:
        w, h, frames = smaller["canvas"].split("x")
        reason += (
            f"; at {w}x{h} ({frames} frames) it needs about {_gib_text(smaller['fit']['need_bytes'])} (measured with the same engine) and "
            f"fits this computer: set {key} to {download['provider']}/{download['artifact']} yourself and generate "
            f"at {w}x{h}"
        )
    return f"{reason}{next_step}"


def _unavailable_reasons(
    routes: Mapping[str, CapabilityRouteDefault], downloads: Mapping[str, Mapping[str, str]], host: Mapping[str, Any]
) -> Dict[str, str]:
    """`{key: reason}` for every recommended row this host cannot run: its
    engine has no build here, or (fit-gated rows) its model does not fit."""

    out: Dict[str, str] = {}
    for key, route in routes.items():
        reason = recommended_route_unavailable_reason(route.provider, host, key)
        if reason:
            next_step = _unavailable_next_step(key, host)
            if next_step:
                reason = f"{reason}; {next_step}"
            out[key] = reason
            continue
        if key in _FIT_GATED_ROUTES:
            reason = _fit_gate_reason(key, downloads[key], host)
            if reason:
                out[key] = reason
    return out


def _host_or_probe(host: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if host is not None:
        return dict(host)
    from ..utils.host_profile import host_profile

    # The LIGHT reading first: os / arch / Apple silicon / RAM, no GPU tool,
    # no torch, no mlx. It answers Apple silicon whole. Off Apple silicon it
    # reads every GPU as `none`, and there the accelerator decides a
    # recommendation (an NVIDIA GPU's image pick, `by_accelerator`), so the
    # FULL probe answers (cached for a few seconds; nvidia-smi). On the light
    # reading alone apply-recommended, the grid and `models download
    # --recommended` would treat a CUDA host as processor-only and never
    # offer its image pick (framework backlog 0989). The import-time seed never comes here: it passes the light reading
    # itself (`seed_recommended_capability_defaults`).
    light = host_profile(light=True)
    if light.get("accelerator") == "metal":
        return light
    return host_profile()


def _full_recommendation(
    host: Mapping[str, Any], *, starter_only: bool = True
) -> Tuple[Dict[str, CapabilityRouteDefault], Dict[str, Dict[str, str]]]:
    """(routes, downloads) with the host's text pick, BEFORE the support filter.

    `starter_only` (the writers' view) keeps the fresh-install starter rows;
    `False` is every row of `RECOMMENDED_MODELS` (`recommendations`).
    """

    from .model_catalog import recommended_text_model

    table = {key: rec.pick_for(host) for key, rec in RECOMMENDED_MODELS.items() if rec.starter or not starter_only}
    routes = {
        key: CapabilityRouteDefault(
            provider=route.provider, model=route.model, base_url=route.base_url,
            reasoning=route.reasoning, options=dict(route.options),
        )
        for key, (route, _download) in table.items()
    }
    downloads = {key: dict(download) for key, (_route, download) in table.items()}
    pick = recommended_text_model(host, fit=False)
    routes["input.text"] = CapabilityRouteDefault(
        provider=pick["provider"], model=pick["model"], options=dict(pick.get("options") or {})
    )
    downloads["input.text"] = {"provider": pick["provider"], "artifact": pick["artifact"]}
    return routes, downloads


def image_input_unavailable_reason(
    text_route: CapabilityRouteDefault, text_reason: Optional[str]
) -> Optional[str]:
    """Why `input.image` has no recommendation on a host, or None.

    Image input is read by the host's recommended text model
    (`recommendations`: `covered` by `input.text`), so it is unavailable when
    text is (`text_reason`), or when that model does not read images. One
    sentence for the grid, the plan and
    `abstractcore models recommendations`.
    """

    if text_reason:
        return text_reason
    from .manager import model_supports_input

    model = str(text_route.model or "")
    if model_supports_input(model, "image"):
        return None
    return (
        f"the recommended text model ({model}) does not read images; set input.image to a "
        "vision-capable model on another machine or a cloud provider"
    )


def recommended_unavailable_routes(host: Optional[Mapping[str, Any]] = None) -> Dict[str, Dict[str, str]]:
    """Recommended rows this host cannot run: `{key: {provider, model, reason}}`.

    On every host that is not Apple silicon `output.video` is here (its only
    recommended engine, MLX-Gen, is Apple silicon only), and so is
    `output.image` except on an NVIDIA GPU, whose image pick is Diffusers
    (`by_accelerator`; there it is here only where the catalog says it does
    not fit the GPU's memory). On Apple
    silicon the memory-gated rows (`_FIT_GATED_ROUTES`: image, video) are here
    where the catalog says their model does not fit: `output.image` on an 8 GB
    Mac (FLUX.2 klein 4B needs ~8.5 GiB), `output.video` below 32 GiB of
    unified memory (a 24 GiB Mac only after raising the GPU memory limit). Such a route stays UNSET with this reason rather than
    seeded with a route that fails at first use.

    `input.image` is here when the host's recommended text model, which covers
    it, does not read images (`image_input_unavailable_reason`; provider and
    model name that text model).
    """

    profile = _host_or_probe(host)
    routes, downloads = _full_recommendation(profile)
    reasons = _unavailable_reasons(routes, downloads, profile)
    out = {
        key: {"provider": str(routes[key].provider or ""), "model": str(routes[key].model or ""), "reason": reason}
        for key, reason in reasons.items()
    }
    text = routes["input.text"]
    image_reason = image_input_unavailable_reason(text, reasons.get("input.text"))
    if image_reason:
        out["input.image"] = {"provider": str(text.provider or ""), "model": str(text.model or ""), "reason": image_reason}
    return out


def _recommended_key_for(key: str) -> Optional[str]:
    """The recommended row that answers `key`: itself, or its modality cell."""

    if key in RECOMMENDED_CAPABILITY_DEFAULT_ROUTES:
        return key
    broad = capability_route_broad_key(key)
    return broad if broad in RECOMMENDED_CAPABILITY_DEFAULT_ROUTES else None


def configured_routes_unavailable(
    routes: Mapping[str, Any], host: Optional[Mapping[str, Any]] = None
) -> Dict[str, Dict[str, str]]:
    """CONFIGURED routes this host cannot run: `{key: {provider, model, reason}}`.

    The shape of `recommended_unavailable_routes`, for what the operator (or an
    older seed) actually stored. Only in-process providers are judged
    (`_IN_PROCESS_PROVIDERS`); a route to a server or a cloud API may run from
    here whatever this host's builds. A route is unavailable when its engine
    has no build here, or -- on a memory-gated capability (image, video) --
    when the catalog says its model does not fit this host's memory, with the
    recommendation's own verdict and sentence (`_configured_route_fit_reason`;
    `tight` and `needs_gpu_limit` are not unavailable). The reason ends with
    what to do: the host's own recommendation for that row when it has one,
    otherwise the row's next step. Only a report: the stored routes are never
    changed here.

    Field reports behind it: Linux, Windows and Intel-Mac installs seeded
    before the host-aware recommendation still carry `output.image:
    mlx-gen/...`, which fails at first use and read as fine (2026-09-27); an
    8 GB Mac's saved FLUX.2 klein 4B image route (~8.5 GiB) read as fine and
    ran out of memory at first use (2026-09-28).
    """

    judged: Dict[str, Tuple[str, str]] = {}
    for key, value in routes.items():
        route = clean_capability_route_default(value)
        provider = str(route.provider or "").strip()
        if provider.lower() in _IN_PROCESS_PROVIDERS:
            judged[str(key)] = (provider, str(route.model or ""))
    if not judged:
        return {}
    profile = _host_or_probe(host)
    out: Dict[str, Dict[str, str]] = {}
    recommended: Optional[Dict[str, CapabilityRouteDefault]] = None
    for key, (provider, model) in judged.items():
        reason = recommended_route_unavailable_reason(provider, profile, key)
        if not reason:
            reason = _configured_route_fit_reason(key, provider, model, profile)
            if not reason:
                continue
            # The memory reason already ends with what to do (`_TOO_LARGE_NEXT_STEP`);
            # name this host's own pick only when one runs here.
            if recommended is None:
                recommended = recommended_capability_default_routes(profile)
            rec_key = _recommended_key_for(key)
            pick = recommended.get(rec_key) if rec_key else None
            if pick is not None and (pick.provider, pick.model) != (provider, model):
                reason = f"{reason}; this computer's recommended route is {pick.provider}/{pick.model}"
            out[key] = {"provider": provider, "model": model, "reason": reason}
            continue
        if recommended is None:
            recommended = recommended_capability_default_routes(profile)
        rec_key = _recommended_key_for(key)
        pick = recommended.get(rec_key) if rec_key else None
        if pick is not None:
            reason = f"{reason}; this computer's recommended route is {pick.provider}/{pick.model}"
        elif _unavailable_next_step(rec_key, profile):
            reason = f"{reason}; {_unavailable_next_step(rec_key, profile)}"
        out[key] = {"provider": provider, "model": model, "reason": reason}
    return out


def _configured_route_fit_reason(key: str, provider: str, model: str, host: Mapping[str, Any]) -> Optional[str]:
    """Why a CONFIGURED route's model does not fit `host`, or None.

    Judged only on the memory-gated capabilities (`_FIT_GATED_ROUTES`, a task
    row by its modality) and only for a model the catalog knows (its memory
    need is the catalog's): the same verdict and the same sentence as the
    recommendation (`_fit_gate`). Unavailable means the catalog says it does
    not fit (`too_large`, `partial_offload`) at any measured canvas; `fits`,
    `tight`, `needs_gpu_limit` (fits after the `sysctl`), `unknown` (no memory
    reading) and a model with a measured smaller canvas that fits are not.
    """

    from .model_catalog import FITS_FILTER_VERDICTS

    broad = key if key in _FIT_GATED_ROUTES else capability_route_broad_key(key)
    if broad not in _FIT_GATED_ROUTES:
        return None
    try:
        verdict, reason = _fit_gate(broad, {"provider": provider, "artifact": model}, host)
    except LookupError:
        return None  # not a catalog artifact: no memory need to judge
    if verdict in FITS_FILTER_VERDICTS or verdict == "unknown":
        return None
    # A measured smaller canvas this Mac runs it at (for example T2V-A14B at
    # 640x352 where its 832x480 default does not fit): the route is the one our
    # own reason tells the operator to set -- never flagged, never cleared by
    # `--force`.
    from .model_catalog import smaller_canvas_fit

    if smaller_canvas_fit(provider, model, host) is not None:
        return None
    return reason


def recommended_capability_default_routes(
    host: Optional[Mapping[str, Any]] = None,
) -> Dict[str, CapabilityRouteDefault]:
    """The recommended routes this host can run (default host: this machine)."""

    profile = _host_or_probe(host)
    routes, downloads = _full_recommendation(profile)
    unavailable = _unavailable_reasons(routes, downloads, profile)
    return {key: route for key, route in routes.items() if key not in unavailable}


def recommended_model_downloads(host: Optional[Mapping[str, Any]] = None) -> Dict[str, Dict[str, str]]:
    """The downloads behind `recommended_capability_default_routes(host)`."""

    profile = _host_or_probe(host)
    routes, downloads = _full_recommendation(profile)
    unavailable = _unavailable_reasons(routes, downloads, profile)
    return {key: spec for key, spec in downloads.items() if key not in unavailable}


# The `--only` vocabulary: the words the operator says ("text", "voice",
# "image", "video") mapped to the route keys the recommendation actually
# writes. One table so the CLI, the Gateway endpoint and both console-TUIs
# offer the same words and can never disagree about which row each one means.
RECOMMENDED_SELECTORS: Dict[str, str] = {
    "text": "input.text",
    "voice": "output.voice",
    "stt": "input.voice",
    "image": "output.image",
    "video": "output.video",
}


def recommended_selector_for_route(key: str) -> str:
    """The `--only` word for a recommended route key (`""` if it has none)."""
    for selector, route_key in RECOMMENDED_SELECTORS.items():
        if route_key == key:
            return selector
    return ""


def plan_recommended_capability_defaults(
    routes: Mapping[str, CapabilityRouteDefault],
    *,
    only: Optional[Iterable[str]] = None,
    force: bool = False,
    host: Optional[Mapping[str, Any]] = None,
) -> Tuple[Dict[str, Any], ...]:
    """What `apply-recommended` WOULD do to `routes`, one entry per route.

    The recommendation is this host's (`recommended_capability_default_routes`):
    on Apple silicon the text route follows the unified-memory tiers, and a
    route whose recommended engine cannot run here is reported as
    `unavailable` with its `reason` (never written, even with `force`).

    THE SEED WILL NOT DO THIS. `seed_recommended_capability_defaults` runs only
    when the store file has never existed, deliberately: an operator who
    cleared a route meant it. That safety left a hole an operator fell into --
    "I asked for qwen3.5-9b everywhere and I see qwen3-0.6b" (2026-08-01) --
    because nothing in the product could say "make this machine match the
    recommendation". This plan is that action, and it stays honest about the
    difference between filling a gap and overruling a choice:

      `apply`      the route is empty -> the recommendation is written
      `already`    the route already names the recommended provider/model
      `kept`       the operator configured something else -> UNTOUCHED unless
                   `force`, and reported so they can see what was skipped
      `overwrite`  `force`, and the operator's provider/model is replaced
      `unavailable` nothing recommended runs on this host (`reason`); a
                   configured route is kept
      `cleared`    `force`, nothing recommended runs here, AND the configured
                   route cannot run here either: it is removed, whole

    A configured route this host cannot run (`configured_routes_unavailable`)
    also carries `route_unavailable: {provider, model, reason}` on its entry,
    whatever the action, so a broken route is never reported as fine. The
    route an entry leaves in place carries `engine_missing: {engine, name,
    reason, install}` when this host can run its in-process engine but the
    engine is not installed (`route_engines.route_engine_missing`).

    FIELD-PRESERVING like every other writer here: only `provider` and `model`
    come from the recommendation. A pinned `base_url`, a reasoning effort and
    plugin options (`{voice: M2}`) belong to the operator's machine, not to the
    recommendation, and survive every outcome above.
    """

    wanted: Optional[set] = None
    if only is not None:
        wanted = set()
        for token in only:
            name = str(token or "").strip().lower()
            if not name:
                continue
            key = RECOMMENDED_SELECTORS.get(name, name)
            if key not in RECOMMENDED_CAPABILITY_DEFAULT_ROUTES:
                raise ValueError(
                    f"Unknown recommended selector: {token!r}. "
                    f"Expected one of {', '.join(sorted(RECOMMENDED_SELECTORS))}."
                )
            wanted.add(key)

    from .route_engines import route_engine_missing

    profile = _host_or_probe(host)
    recommended_routes = recommended_capability_default_routes(profile)
    recommended_downloads = recommended_model_downloads(profile)
    unavailable = recommended_unavailable_routes(profile)
    broken = configured_routes_unavailable(
        {key: routes[key] for key in RECOMMENDED_CAPABILITY_DEFAULT_ROUTES if key in routes}, profile
    )
    plan: list = []
    for key in RECOMMENDED_CAPABILITY_DEFAULT_ROUTES:
        if wanted is not None and key not in wanted:
            continue
        current = routes.get(key)
        before = current.to_dict() if isinstance(current, CapabilityRouteDefault) else {}
        if key in unavailable:
            # Nothing this host can run is recommended: never written. A route
            # the operator configured is kept (not even `force` replaces a
            # working choice with nothing) -- unless it cannot run here either:
            # then `force` clears it, and without `force` it is flagged.
            cleared = force and key in broken
            entry = {
                "key": key,
                "selector": recommended_selector_for_route(key),
                "action": "cleared" if cleared else "unavailable",
                "changed": cleared,
                "recommended": {},
                "before": before,
                "after": {} if cleared else dict(before),
                "download": {},
                "reason": unavailable[key]["reason"],
            }
            if key in broken:
                entry["route_unavailable"] = dict(broken[key])
            plan.append(entry)
            continue
        recommended = recommended_routes[key]
        configured = bool(before.get("provider") or before.get("model"))
        matches = (
            str(before.get("provider") or "") == str(recommended.provider or "")
            and str(before.get("model") or "") == str(recommended.model or "")
        )
        if matches:
            action = "already"
        elif not configured:
            action = "apply"
        elif force:
            action = "overwrite"
        else:
            action = "kept"

        after = dict(before)
        if action in {"apply", "overwrite"}:
            after["provider"] = recommended.provider
            after["model"] = recommended.model
            after = {k: v for k, v in after.items() if v not in (None, "", {})}
        plan.append(
            {
                "key": key,
                "selector": recommended_selector_for_route(key),
                "action": action,
                "changed": action in {"apply", "overwrite"},
                "recommended": recommended.to_dict(),
                "before": before,
                "after": after,
                "download": dict(recommended_downloads.get(key, {})),
                # The configured route cannot run on this host: `kept` rows
                # say so (only `force` replaces them), `overwrite` rows say
                # what was fixed.
                **({"route_unavailable": dict(broken[key])} if key in broken else {}),
            }
        )
    for entry in plan:
        # The route this entry LEAVES in place, when its in-process engine is
        # not installed here (`route_engines`): "apply" writes a route whose
        # engine still has to be installed, and says so with the command. A
        # route this host cannot run at all is `route_unavailable` instead.
        after = entry["after"]
        if not after.get("provider") or ("route_unavailable" in entry and after == entry["before"]):
            continue
        flag = route_engine_missing(after.get("provider"), after.get("model"), entry["key"])
        if flag is not None:
            entry["engine_missing"] = flag
    return tuple(plan)


def seed_recommended_capability_defaults(
    config: CapabilityDefaultsConfig, *, host: Optional[Mapping[str, Any]] = None
) -> CapabilityDefaultsConfig:
    """Apply the fresh-install recommendation to an empty defaults config.

    Only fills routes that are not already configured (defensive — the caller
    gates on file absence, so in practice all three are empty) and stamps the
    provenance marker so surfaces can label the values as recommended. The
    recommendation is this host's (Apple silicon: the unified-memory tier); a
    route this host cannot run is left unset (`recommended_unavailable_routes`).

    With no `host` it reads the LIGHT host profile only: it runs when the
    config store is first created, on import, where no GPU tool or engine may
    load. The light reading sees no CUDA, so a fresh NVIDIA install gets no
    image route here; `apply-recommended` (full probe) writes it.
    """
    if host is None:
        from ..utils.host_profile import host_profile

        host = host_profile(light=True)
    for key, route in recommended_capability_default_routes(host).items():
        existing = config.routes.get(key)
        if existing is None or not existing.configured():
            config.routes[key] = CapabilityRouteDefault(
                provider=route.provider, model=route.model,
                base_url=route.base_url, reasoning=route.reasoning,
                options=dict(route.options),
            )
    config.seeded = RECOMMENDED_SEED_VERSION
    return config


def upgrade_recommended_seed(
    config: CapabilityDefaultsConfig, *, host: Optional[Mapping[str, Any]] = None
) -> bool:
    """Add the rows later seed versions introduced to a store an earlier seed wrote.

    Only a store stamped by an earlier seed (`seeded` is an older
    `recommended-v*`) is touched, and only the rows the newer versions added
    (`RECOMMENDED_SEED_ADDITIONS`) that are EMPTY in it: a route the operator
    set is never replaced, and a store that was never seeded is never
    changed. The recommendation is this host's (a row this host cannot run is
    left unset), read from the light host profile like the seed itself. The
    result lives in memory, stamped with the current version, exactly like a
    fresh seed: the next save of the store persists it. Returns whether the
    config changed.
    """

    seeded = str(config.seeded or "")
    if not seeded.startswith("recommended-v") or seeded == RECOMMENDED_SEED_VERSION:
        return False
    try:
        current = int(seeded[len("recommended-v"):])
    except ValueError:
        return False
    keys = [
        key
        for version, added in RECOMMENDED_SEED_ADDITIONS
        if int(version[len("recommended-v"):]) > current
        for key in added
    ]
    if not keys:
        return False
    if host is None:
        from ..utils.host_profile import host_profile

        host = host_profile(light=True)
    runnable = recommended_capability_default_routes(host)
    for key in keys:
        route = runnable.get(key)
        existing = config.routes.get(key)
        if route is None or (existing is not None and existing.configured()):
            continue
        config.routes[key] = CapabilityRouteDefault(
            provider=route.provider, model=route.model,
            base_url=route.base_url, reasoning=route.reasoning,
            options=dict(route.options),
        )
    config.seeded = RECOMMENDED_SEED_VERSION
    return True


def capability_defaults_from_dict(value: Any) -> CapabilityDefaultsConfig:
    if isinstance(value, CapabilityDefaultsConfig):
        return value
    data = value if isinstance(value, Mapping) else {}
    routes_raw = data.get("routes") if isinstance(data.get("routes"), Mapping) else data
    routes: Dict[str, CapabilityRouteDefault] = {}
    for key_raw, route_raw in dict(routes_raw or {}).items():
        try:
            kind, modality, task = split_capability_default_route(key_raw)
            key = capability_route_key(kind, modality, task)
            route = clean_capability_route_default(route_raw)
        except Exception:
            continue
        if route.configured():
            routes[key] = route
    version = data.get("version", CAPABILITY_DEFAULTS_VERSION)
    try:
        version_i = int(version)
    except Exception:
        version_i = CAPABILITY_DEFAULTS_VERSION
    seeded_raw = data.get("seeded")
    seeded = str(seeded_raw).strip() if isinstance(seeded_raw, str) and seeded_raw.strip() else None
    return CapabilityDefaultsConfig(version=version_i, routes=routes, seeded=seeded)


def iter_capability_default_specs() -> Iterable[CapabilityDefaultSpec]:
    specs = [
        ("input", "text", "Text Input", "text_understanding", None, {}),
        ("input", "image", "Image Input", "image_understanding", "abstractvision or a vision-capable LLM", {}),
        ("input", "video", "Video Input", "video_understanding", "abstractvideo or a video-capable LLM", {}),
        ("input", "voice", "Voice Input", "speech_to_text", "abstractvoice", {"language": "en"}),
        ("input", "sound", "Sound Input", "audio_understanding", "abstractsound or abstractmusic", {}),
        ("input", "music", "Music Input", "music_understanding", "abstractmusic or a music-capable LLM", {}),
        ("input", "scene3d", "3D Scene Input", "scene3d_understanding", "abstract3d", {}),
        ("output", "text", "Text Output", "text_generation", None, {}),
        ("output", "image", "Image Output", "image_generation", "abstractvision", {}),
        ("output", "image.text_to_image", "Image Generation", "text_to_image", "abstractvision", {}),
        ("output", "image.image_to_image", "Image Edit", "image_to_image", "abstractvision", {}),
        ("output", "image.image_upscale", "Image Restore / Upscale", "image_upscale", "abstractvision", {"resolution": "2x", "softness": 0.25}),
        ("output", "video", "Video Output", "video_generation", "abstractvideo or abstractvision", {}),
        ("output", "video.text_to_video", "Video Generation", "text_to_video", "abstractvideo or abstractvision", {}),
        ("output", "video.image_to_video", "Image To Video", "image_to_video", "abstractvideo or abstractvision", {}),
        ("output", "voice", "Voice Output", "text_to_speech", "abstractvoice", {"voice": "default"}),
        ("output", "sound", "Sound Effects Output", "sound_generation", "abstractsound or abstractmusic", {}),
        ("output", "music", "Music Output", "music_generation", "abstractmusic", {}),
        ("output", "scene3d", "3D Scene Output", "scene3d_generation", "abstract3d", {}),
        ("output", "scene3d.text_to_scene3d", "Text To 3D", "text_to_scene3d", "abstract3d", {}),
        ("output", "scene3d.image_to_scene3d", "Image To 3D", "image_to_scene3d", "abstract3d", {}),
        ("embedding", "text", "Text Embeddings", "text_embedding", "abstractcore.embeddings", {}),
        ("embedding", "image", "Image Embeddings", "image_embedding", "abstractcore.embeddings or abstractvision", {}),
        ("rerank", "text", "Text Rerank", "text_rerank", "future reranker manager", {}),
    ]
    for kind, modality_raw, label, task, package_hint, option_examples in specs:
        modality, route_task = (
            str(modality_raw).split(".", 1) if "." in str(modality_raw) else (str(modality_raw), None)
        )
        yield CapabilityDefaultSpec(
            key=capability_route_key(kind, modality, route_task),
            kind=kind,
            modality=modality,
            label=label,
            task=task,
            package_hint=package_hint,
            option_examples=option_examples,
        )


def capability_default_specs_dict() -> Dict[str, Dict[str, Any]]:
    return {spec.key: spec.to_dict() for spec in iter_capability_default_specs()}


# ---------------------------------------------------------------------------
# THE HIERARCHY, DERIVED ONCE
# ---------------------------------------------------------------------------
#
# `output.image` is not a remnant and not a sibling of `output.image.*` -- it is
# their PARENT: the answer for every image task that has no row of its own.
# Setting it is the simple path (one model for generate/edit/upscale), and it is
# what the fresh-install seed writes. A `.task` row overrides it for that task.
#
# Four surfaces render this grid (web console, both console-TUIs, the CLI) and
# every one of them used to draw the parent as a flat sibling ABOVE its own
# children with a red "not configured" -- which is what made an operator ask
# whether the row was dead code. The parent/child facts are derived HERE, once,
# and travel on the payload, so no surface re-derives them and they cannot drift.


def capability_route_broad_key(key: Any) -> Optional[str]:
    """The modality-cell key a task row falls back to, or ``None``.

    ``output.image.image_upscale`` -> ``output.image``; a 2-part key is already
    the modality cell and has no parent.
    """

    parts = [part for part in str(key or "").strip().split(".") if part]
    if len(parts) < 3:
        return None
    return f"{parts[0]}.{parts[1]}"


def capability_route_task_keys(key: Any) -> Tuple[str, ...]:
    """The task rows that override one modality cell, in grid order.

    ``output.image`` -> the three `output.image.*` rows. Empty for a modality
    with no persistable sub-task (voice/sound/music, every input route): those
    cells ARE the primary key, which is why the row shape can never be deleted.
    """

    parent = str(key or "").strip()
    if not parent or capability_route_broad_key(parent) is not None:
        return ()
    return tuple(
        spec.key
        for spec in iter_capability_default_specs()
        if capability_route_broad_key(spec.key) == parent
    )


def capability_route_tasks_cover_broad(key: Any, routes: Any) -> bool:
    """True when every task row under ``key`` is configured, so broad is unreachable.

    PROVABLE, not cosmetic: `_OUTPUT_ROUTE_TABLE`'s 3-part keys for a modality
    are exactly that modality's task rows, so once all of them carry a route,
    `capability_route_key_for_output` can never return the modality cell for
    that modality and nothing reads it. The grid may then say "not needed"
    instead of flagging an unset parent as a problem.

    ``routes`` is any mapping of route key -> row in the JSON-safe shape
    `list_capability_defaults()` produces (a row marked
    ``source: "not_configured"`` counts as unset), or route key ->
    `CapabilityRouteDefault`.
    """

    task_keys = capability_route_task_keys(key)
    if not task_keys or not isinstance(routes, Mapping):
        return False
    return all(_route_row_is_configured(routes.get(task_key)) for task_key in task_keys)


def _route_row_is_configured(row: Any) -> bool:
    if isinstance(row, CapabilityRouteDefault):
        return row.configured()
    if not isinstance(row, Mapping):
        return False
    if str(row.get("source") or "").strip() == "not_configured":
        return False
    if "configured" in row:
        return bool(row.get("configured"))
    return bool(
        row.get("provider")
        or row.get("model")
        or row.get("base_url")
        or row.get("reasoning")
        or row.get("options")
    )


# ---------------------------------------------------------------------------
# THE ONE TABLE: generation-task vocabulary -> capability default route key
# ---------------------------------------------------------------------------
#
# TWO ENTRY POINTS, ONE STORE. AbstractCore owns the only store of per-modality
# provider/model defaults; AbstractGateway is a CRUD surface over it. Both entry
# points therefore have to agree on ONE question: given a generation request,
# WHICH route key holds its default? That question is answered here and nowhere
# else. `abstractcore.core.generate_contract` (the execution path) and
# AbstractRuntime's LLM client (the stream lane) both delegate to this table --
# each used to carry its own copy, and the copies had already drifted.
#
# WHY THE input.X / output.X GRID (and not a flat per-task namespace):
#   `kind` is the direction of the MEDIA the route handles, not the direction of
#   the user's intent. That is the only reading under which every modality lands
#   in exactly one cell:
#       input.image   image understanding (vision-in)
#       output.image  image generation / edit / upscale
#       input.voice   speech-to-text  -- the AUDIO is the input; the text result
#                     is not a generated-media output at all, which is why
#                     `text/transcription` maps to NO output route below and is
#                     routed through `input.voice` by `_input_route_keys`
#       output.voice  text-to-speech
#       input.sound / input.music / input.video   understanding
#       output.sound / output.music / output.video / output.scene3d  generation
#   The grid also matches the multimodal COVERAGE logic already in the manager
#   ("input.image covered by input.text" when the text model is vision-capable),
#   which only makes sense on an input/output grid.
#
# BROAD vs TASK-SPECIFIC -- A PARENT AND ITS OVERRIDES, NOT TWO FLAT NAMESPACES.
# A modality cell (`output.image`) is always valid and is the fallback: it is the
# answer for EVERY image task that has no row of its own, which makes setting it
# the simple path ("one model for generate/edit/upscale") and is why the
# fresh-install seed writes `output.image` rather than three task rows. A `.task`
# suffix overrides it for that task, WHOLESALE (a row is one coherent backend
# identity -- never field-merged with its parent), but ONLY for the seven tasks in
# `CAPABILITY_ROUTE_TASKS` -- those are the ones the store can actually persist.
# Anything else (tts, stt, music_generation, ...) resolves at the modality cell.
# Emitting a `<kind>.<modality>.<task>` key for a task outside that tuple mints a
# key the store can never hold; the tuple below is the guard against that.
#
# FOUR of the seven output modalities have NO task rows at all (voice, sound,
# music, and every `input.*`/`embedding.*`/`rerank.*` cell): for them the modality
# cell is the PRIMARY key, not a fallback. Only image/video/scene3d carry both
# levels. That is why the broad row shape cannot be deleted -- deleting it would
# delete `output.voice`.
#
# A MODALITY-LEVEL QUESTION ("which image backend does this host use?") must
# resolve the SAME WAY execution does -- canonical task row first, modality cell
# second -- or advertising and execution disagree. Ask
# `capability_route_keys_for_output(modality)` for that pair rather than reading
# the modality cell directly; two advertising readers did the latter and reported
# `openai` for a host whose three image task rows all named mlx-gen.
#
# `has_source_image` picks the image/video/scene3d variant a bare request means:
# with a source image attached, "generate" is really an edit/i2v/i23d.
#
# Rows are matched IN ORDER; `_ANY_TASK` is a catch-all alias for "every task not
# claimed by an earlier row of the same modality".
_ANY_TASK = "*"

_OUTPUT_ROUTE_TABLE: Tuple[Tuple[str, Tuple[str, ...], str, Optional[str]], ...] = (
    # (modality, accepted task aliases, route key, route key when a source image is attached)
    # `transcription` must be listed BEFORE the text catch-all: a transcription
    # is not a text GENERATION, and claiming `output.text` for it would hand the
    # STT call the operator's chat model.
    ("text", ("transcription",), "", None),
    ("text", (_ANY_TASK,), "output.text", None),
    ("image", ("image_upscale", "upscale_image", "image_upscaling", "upscale"), "output.image.image_upscale", None),
    ("image", ("image_edit", "image_to_image", "i2i", "edit_image"), "output.image.image_to_image", None),
    ("image", ("", "image_generation", "text_to_image", "t2i"), "output.image.text_to_image", "output.image.image_to_image"),
    ("video", ("image_to_video", "i2v", "video_from_image", "video_edit"), "output.video.image_to_video", None),
    ("video", ("", "video_generation", "text_to_video", "t2v"), "output.video.text_to_video", "output.video.image_to_video"),
    # Voice/sound/music have no persistable sub-task, so both directions resolve
    # at the modality cell. `stt` appears here because a caller may label the
    # spec `modality=voice`; the canonical STT spec is `text/transcription`
    # above, whose default comes from the `input.voice` INPUT route.
    ("voice", ("stt", "transcribe", "transcription", "speech_to_text", "asr"), "input.voice", None),
    # `voice_clone` is the task `_infer_output_specs` assigns to a bare
    # `output="voice"` request that carries reference audio, and it is in the
    # public output vocabulary (`OUTPUT_TASK_MODALITIES`). It SYNTHESISES voice,
    # so it is an output.voice route exactly like tts -- the reference audio is
    # a conditioning input, not a second modality. (Adversary catch: the first
    # cut of this table omitted it, which silently dropped the operator's voice
    # default for every clone request and handed it back to the plugin's
    # env-or-openai fallback -- the dm#28 429 incident.)
    ("voice", ("", "tts", "text_to_speech", "speech", "speak", "voice_clone", "clone"), "output.voice", None),
    ("music", ("text_to_audio", "sound_generation", "text_to_sound", "sfx", "sound_effect"), "output.sound", None),
    ("music", ("", "music_generation", "text_to_music", "t2m"), "output.music", None),
    ("sound", ("", "sound_generation", "text_to_sound", "sfx", "sound_effect"), "output.sound", None),
    ("scene3d", ("image_to_scene3d", "i23d", "image_to_3d"), "output.scene3d.image_to_scene3d", None),
    (
        "scene3d",
        ("", "scene3d_generation", "text_to_scene3d", "t23d", "text_to_3d"),
        "output.scene3d.text_to_scene3d",
        "output.scene3d.image_to_scene3d",
    ),
)


def capability_route_key_for_output(
    modality: Any,
    task: Any = None,
    *,
    has_source_image: bool = False,
) -> Optional[str]:
    """Return the capability default route key for one generation output spec.

    THE ONE TABLE (see `_OUTPUT_ROUTE_TABLE` above). Returns ``None`` when the
    spec names no routable generation -- notably `text/transcription`, whose
    default lives on the `input.voice` INPUT route because the audio, not the
    text, is what the route provisions.
    """

    modality_s = str(modality or "").strip().lower().replace("-", "_")
    task_s = str(task or "").strip().lower().replace("-", "_")
    for row_modality, aliases, route_key, source_image_key in _OUTPUT_ROUTE_TABLE:
        if row_modality != modality_s:
            continue
        if task_s not in aliases and _ANY_TASK not in aliases:
            continue
        if has_source_image and source_image_key:
            return source_image_key
        return route_key or None
    return None


def capability_route_keys_for_output(
    modality: Any,
    task: Any = None,
    *,
    has_source_image: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """The (exact, broad-fallback) route-key pair for one generation output spec.

    The broad key is the modality cell (`output.image`); it is ``None`` when the
    exact key already IS the modality cell, so callers never look the same key
    up twice.
    """

    exact = capability_route_key_for_output(modality, task, has_source_image=has_source_image)
    if not exact:
        return None, None
    parts = [part for part in exact.split(".") if part]
    if len(parts) < 3:
        return exact, None
    return exact, f"{parts[0]}.{parts[1]}"


# THE TEXT-GENERATION ROUTE, BY NAME. `output.text` is the canonical read; the
# store canonicalizes it to `input.text`, which stays readable so a config that
# carries only the storage key still resolves. Both keys name the same cell.
TEXT_ROUTE_KEY = "output.text"
TEXT_ROUTE_STORAGE_KEY = "input.text"
TEXT_ROUTE_KEYS: Tuple[str, ...] = (TEXT_ROUTE_KEY, TEXT_ROUTE_STORAGE_KEY)


def capability_default_reasoning(routes: Any) -> Optional[str]:
    """The configured reasoning effort for the text-generation route.

    `routes` is a mapping of route key -> route row, in the JSON-safe shape
    `list_capability_defaults()` and `get_capability_default()` produce. The
    canonical key answers first, the storage key second; a row explicitly marked
    ``source: "not_configured"`` never answers.

    This is the ONE definition of where a reasoning default lives, so callers
    resolve it by asking rather than by re-deriving the key order.
    """

    if not isinstance(routes, Mapping):
        return None
    for key in TEXT_ROUTE_KEYS:
        row = routes.get(key)
        if not isinstance(row, Mapping):
            continue
        if str(row.get("source") or "").strip() == "not_configured":
            continue
        value = _clean_optional_string(row.get("reasoning"))
        if value:
            return value.lower()
    return None


def capability_default_speculation(routes: Any) -> Any:
    """Text-route MTP policy; absence inherits, False is an explicit override.

    This is configured intent, not a capability claim. Only the execution host
    may negotiate it against the selected artifact/backend/loaded instance.
    """
    from ..providers.speculation import normalize_speculation_value

    if not isinstance(routes, Mapping):
        return None
    for key in TEXT_ROUTE_KEYS:
        row = routes.get(key)
        if not isinstance(row, Mapping) or row.get("source") == "not_configured":
            continue
        options = row.get("options")
        if isinstance(options, Mapping) and "speculation" in options:
            return normalize_speculation_value(options["speculation"])
    return None


def _clean_optional_string(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None
