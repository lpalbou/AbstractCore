"""One rule for 'is this local repository an MLX artifact?'.

Both local-weight providers scan the same directories -- the HuggingFace hub
cache and the LM Studio store -- and each one has to answer the same question
about every entry it finds: does this repo run on MLX, or on Transformers /
llama.cpp? The answer used to be spelled out independently in three places
(``MLXProvider.list_available_models``, ``HuggingFaceProvider.list_available_models``
and the ``create_llm`` re-route), all of them a bare ``"mlx" in name`` substring
test. Three copies of a naming heuristic drift, and the two listings can
disagree, which is how a repo ends up in both dropdowns or in neither.

This module is the single rule they all call, so the MLX list and the
HuggingFace list are complements of each other by construction.

The rule, in order (first match wins):

1. ``ABSTRACTCORE_NON_MLX_MODEL_PATTERNS`` -- operator override, never MLX.
2. ``ABSTRACTCORE_MLX_MODEL_PATTERNS`` -- operator override, always MLX.
3. A GGUF marker in the name -- llama.cpp weights, which MLX cannot load.
4. ``mlx`` anywhere in the name (``mlx-community/*``, ``*-MLX-4bit``, ...).
5. A publisher that ships MLX format by construction (:data:`MLX_PUBLISHERS`).
6. An MLX-only quantizer tag in the name, currently oMLX's ``oQ<n>`` mixed
   precision marker (``Qwen3.8-27B-oQ4e-mtp``).
7. The on-disk signature: ``library_name: mlx`` in the model card, or an MLX
   ``quantization`` block in ``config.json``. A card naming a generation
   library (``mlx-gen``, ``mflux``) is MLX format but belongs to the media
   backends, so it is excluded here rather than listed as an LLM.

Rules 4-6 read the name alone, which is what makes them usable for a cache
entry whose weights are not downloaded yet -- ``~/.cache/huggingface/hub``
holds ``refs/`` for repos that were only resolved, and there is no
``config.json`` to inspect for those. Rule 7 is the authoritative one whenever
the weights are actually present.

Operator overrides (rules 1-2) take one or more patterns separated by commas
or ``os.pathsep``. A pattern containing ``*``, ``?`` or ``[`` is matched
against the whole lowercased ``org/model`` handle with :mod:`fnmatch`;
anything else is a plain substring test::

    export ABSTRACTCORE_MLX_MODEL_PATTERNS="acme/*-mlxq,someorg/one-model"
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

__all__ = [
    "MLX_PUBLISHERS",
    "MLX_GENERATION_LIBRARIES",
    "ENV_FORCE_MLX_PATTERNS",
    "ENV_FORCE_NON_MLX_PATTERNS",
    "is_mlx_model",
    "explain_mlx_model",
    "is_mlx_quantization_config",
    "card_library_name",
    "repo_mlx_signature",
    "resolve_repo_dir",
]


#: Publishers whose text-model repositories are MLX format by construction, so
#: a repo of theirs is MLX even when the handle never says so.
#:
#: * ``mlx-community`` -- the reference MLX conversion org.
#: * ``jundot`` -- author of oMLX/oQ; publishes MLX safetensors only.
MLX_PUBLISHERS = frozenset({"mlx-community", "jundot"})

#: Environment variable holding operator patterns that force MLX classification.
ENV_FORCE_MLX_PATTERNS = "ABSTRACTCORE_MLX_MODEL_PATTERNS"

#: Environment variable holding operator patterns that force non-MLX classification.
ENV_FORCE_NON_MLX_PATTERNS = "ABSTRACTCORE_NON_MLX_MODEL_PATTERNS"

# oMLX (https://github.com/jundot/omlx) writes MLX safetensors and tags the
# repo `-oQ4e-`, `-oQ6-`, ... The leading `o` is load-bearing: it is what keeps
# this off llama.cpp's `Q4_K_M` / `q8_0` GGUF tags, which mean the opposite.
_MLX_QUANTIZER_TAG_RE = re.compile(r"(?:^|[-_.])oq\d+[a-z]*(?:[-_.]|$)", re.IGNORECASE)

# GGUF is llama.cpp's container. MLX cannot load it, so the marker is decisive
# even for a publisher that otherwise ships MLX (lmstudio-community ships both).
_GGUF_NAME_RE = re.compile(r"(?:^|[-_./])gguf(?:[-_.]|$)|\.gguf$", re.IGNORECASE)

# `library_name:` in model-card front matter. Read the DECLARED library, not a
# `tags: [- mlx]` entry: mlx-gen/mflux image and video repos tag themselves
# `mlx` too, and those belong to the media generation lane, not to the `mlx`
# text provider. `AbstractFramework/*-8bit` are exactly that case.
_CARD_LIBRARY_RE = re.compile(
    r"^library_name:\s*[\"']?([A-Za-z0-9._-]+)[\"']?\s*$", re.IGNORECASE | re.MULTILINE
)

#: Card libraries that are MLX but belong to a generation backend, not here.
MLX_GENERATION_LIBRARIES = frozenset({"mlx-gen", "mflux", "diffusionkit"})

_CARD_FRONT_MATTER_MAX_BYTES = 4096


def _patterns_from_env(var_name: str) -> Tuple[str, ...]:
    raw = os.environ.get(var_name, "")
    if not raw.strip():
        return ()
    chunks: list[str] = []
    for part in raw.replace(os.pathsep, ",").split(","):
        cleaned = part.strip().strip("\"'").lower()
        if cleaned:
            chunks.append(cleaned)
    return tuple(chunks)


def _matches_any_pattern(model_lower: str, patterns: Sequence[str]) -> bool:
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            if fnmatch.fnmatch(model_lower, pattern):
                return True
        elif pattern in model_lower:
            return True
    return False


def _publisher(model_name: str) -> str:
    text = str(model_name or "").strip().replace("\\", "/")
    if "/" not in text:
        return ""
    return text.split("/", 1)[0].strip().lower()


def is_mlx_quantization_config(quantization_config: Optional[Mapping[str, Any]]) -> bool:
    """True when a ``quantization`` block is MLX's, not Transformers'.

    MLX writes ``{"bits": 4, "group_size": 64, "mode": "affine"}`` with no
    ``quant_method``; every Transformers quantizer (awq, gptq, bnb,
    compressed-tensors, fp8) names itself in ``quant_method``. This is the same
    signature ``HuggingFaceProvider`` already refuses to load, kept here so the
    listing and the loader cannot disagree about what an MLX checkpoint is.
    """
    if not isinstance(quantization_config, Mapping) or not quantization_config:
        return False
    if str(quantization_config.get("quant_method") or "").strip():
        return False
    if "bits" not in quantization_config or "group_size" not in quantization_config:
        return False
    mode = str(quantization_config.get("mode") or "affine").strip().lower()
    return mode in {"affine", "mxfp4", "nf4"}


def resolve_repo_dir(path: Any) -> Optional[Path]:
    """Return the directory that holds ``config.json`` for a local repo.

    Accepts either a HuggingFace hub cache entry (``.../models--org--name``,
    whose files live under ``snapshots/<revision>/``) or a plain model
    directory such as LM Studio's ``~/.lmstudio/models/<org>/<model>``. Returns
    ``None`` when nothing is materialized on disk -- a hub entry that only ever
    got a ``refs/`` pointer has no files to look at.
    """
    if path is None:
        return None
    try:
        base = Path(path)
        if not base.is_dir():
            return None
        if (base / "config.json").is_file() or (base / "README.md").is_file():
            return base
        snapshots = base / "snapshots"
        if not snapshots.is_dir():
            return None
        revisions = [item for item in snapshots.iterdir() if item.is_dir()]
        if not revisions:
            return None
        # Prefer the revision `refs/main` points at; fall back to newest.
        head = base / "refs" / "main"
        if head.is_file():
            try:
                wanted = head.read_text(encoding="utf-8").strip()
            except Exception:
                wanted = ""
            for revision in revisions:
                if revision.name == wanted:
                    return revision
        return max(revisions, key=lambda item: item.stat().st_mtime)
    except Exception:
        return None


#: Weight files a local repo must actually contain to be loadable.
_WEIGHT_PATTERNS = ("*.safetensors", "*.npz", "*.bin", "*.pt", "*.pth")


def has_local_weights(path: Any) -> bool:
    """True when the repo has weight files on disk, not just a cache entry.

    A HuggingFace hub directory exists as soon as a repo is RESOLVED --
    `models--Org--Name/refs/main` and nothing else, 4KB total. Listing one of
    those offers a model the loader then refuses with "Model not found for MLX
    provider", while printing that same model in its own "available models"
    list: the operator is told to pick from a list containing the thing they
    just picked. `MLXProvider._load_model` requires weights (its `_has_weights`
    check, deliberately mirrored here) so the list must require them too.
    """
    repo_dir = resolve_repo_dir(path)
    if repo_dir is None:
        return False
    for pattern in _WEIGHT_PATTERNS:
        try:
            if any(repo_dir.glob(pattern)):
                return True
        except Exception:
            continue
    # Diffusers-style repos keep weights in per-component subdirectories
    # (transformer/, text_encoder/, vae/) rather than at the top level.
    try:
        for child in repo_dir.iterdir():
            if not child.is_dir():
                continue
            for pattern in _WEIGHT_PATTERNS:
                if any(child.glob(pattern)):
                    return True
    except Exception:
        pass
    return False


def card_library_name(path: Any) -> str:
    """The ``library_name`` a local model card declares, lowercased, or ``""``."""
    repo_dir = resolve_repo_dir(path)
    if repo_dir is None:
        return ""
    card_path = repo_dir / "README.md"
    if not card_path.is_file():
        return ""
    try:
        with card_path.open("r", encoding="utf-8", errors="ignore") as handle:
            head = handle.read(_CARD_FRONT_MATTER_MAX_BYTES)
    except Exception:
        return ""
    stripped = head.lstrip()
    if not stripped.startswith("---"):
        return ""
    body = stripped[3:]
    end = body.find("\n---")
    front_matter = body if end < 0 else body[:end]
    match = _CARD_LIBRARY_RE.search(front_matter)
    return match.group(1).strip().lower() if match else ""


def repo_mlx_signature(path: Any) -> bool:
    """True when the files on disk identify the repo as an MLX text/VLM model."""
    repo_dir = resolve_repo_dir(path)
    if repo_dir is None:
        return False

    library = card_library_name(repo_dir)
    if library in MLX_GENERATION_LIBRARIES:
        # MLX format, but an image/video generation backend owns it. Claiming
        # it here would put every diffusers repack in the LLM model picker.
        return False
    if library == "mlx":
        return True

    config_path = repo_dir / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            config = None
        if isinstance(config, Mapping):
            # mlx_lm/mlx_vlm write a top-level `quantization` block; a repo that
            # carries one is MLX even when `quantization_config` mirrors it.
            if is_mlx_quantization_config(config.get("quantization")):
                return True
            if is_mlx_quantization_config(config.get("quantization_config")):
                return True

    return False


def explain_mlx_model(model_name: str, *, local_path: Any = None) -> Tuple[bool, str]:
    """Classify ``model_name`` and say which rule decided it.

    Returns ``(is_mlx, reason)``. ``reason`` is a short stable slug -- it is
    what tests assert on and what the CLI prints when a user asks why a model
    landed in one provider's list rather than the other's.
    """
    text = str(model_name or "").strip()
    if not text:
        return False, "empty-name"
    model_lower = text.lower()

    if _matches_any_pattern(model_lower, _patterns_from_env(ENV_FORCE_NON_MLX_PATTERNS)):
        return False, f"env:{ENV_FORCE_NON_MLX_PATTERNS}"
    if _matches_any_pattern(model_lower, _patterns_from_env(ENV_FORCE_MLX_PATTERNS)):
        return True, f"env:{ENV_FORCE_MLX_PATTERNS}"

    if _GGUF_NAME_RE.search(model_lower):
        return False, "name:gguf"

    if "mlx" in model_lower:
        return True, "name:mlx"

    if _publisher(model_lower) in MLX_PUBLISHERS:
        return True, "publisher"

    if _MLX_QUANTIZER_TAG_RE.search(model_lower):
        return True, "name:oq-quantizer-tag"

    if local_path is not None and repo_mlx_signature(local_path):
        return True, "repo-signature"

    return False, "no-match"


def is_mlx_model(model_name: str, *, local_path: Any = None) -> bool:
    """True when ``model_name`` names an MLX artifact.

    ``local_path`` is the on-disk repo (a hub cache entry or a model
    directory). Pass it whenever the caller has it: without it only the name
    rules apply, and a locally converted MLX repo with a neutral name is
    indistinguishable from a Transformers one.
    """
    verdict, _reason = explain_mlx_model(model_name, local_path=local_path)
    return verdict
