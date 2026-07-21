"""Model-config geometry identity for KV-artifact validity (backlog 0817, axis 3).

A saved KV artifact encodes K/V tensors computed under one attention/position
GEOMETRY: RoPE rotations baked into cached K at fill time (`rope_theta`,
`rope_scaling`, ...), the sliding-window layout that decides which positions a
cache retains per layer (`sliding_window`, `layer_types`, ...), and the
positional envelope (`max_position_embeddings`, ...). A `config.json` edit
under the SAME model id — a rope_theta retune, a longrope block, a window
change — leaves no textual trace (`rendered_recipe_sha256` hashes rendered
TEXT) and no tokenizer trace (axis 2 sees text→ids only), so a reused artifact
holds positionally-wrong KV with no error anywhere: the silently-wrong-cache
class again, one config file over.

Fingerprint subject: a CURATED set of geometry-relevant keys, not the whole
config. Hashing the entire config would false-invalidate the corpus on
irrelevant churn (`transformers_version` bumps, name metadata, quantization
bookkeeping); the curated set moves exactly when the cached tensors' meaning
moves. Keys absent from a config hash as an empty geometry — a stable value
that CHANGES the moment such a key appears (adding `rope_scaling` to a config
that had none is precisely an edit this axis must catch). "" is reserved for
"config unreachable" (model not loaded): validators must treat it as
"cannot verify", never as "matches".
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Optional

from .tokenizer_fingerprint import check_tokenizer_fingerprint as _shared_verdict

__all__ = ["model_config_fingerprint_for", "check_model_config_fingerprint"]


# Position/attention-geometry keys whose change invalidates cached K/V under
# the same model id. Grouped by WHY they are load-bearing:
_GEOMETRY_KEYS = (
    # RoPE family — cached K vectors carry the rotation applied at fill time;
    # a different base/scaling re-rotates nothing retroactively.
    "rope_theta",
    "rope_scaling",
    "rope_parameters",
    "partial_rotary_factor",
    "rotary_dim",
    "rotary_emb_base",
    "rope_local_base_freq",
    "rope_traditional",
    # Window/attention layout — which positions the cache physically retains,
    # per layer (mixed KVCache/RotatingKVCache stacks serialize differently).
    "sliding_window",
    "use_sliding_window",
    "sliding_window_pattern",
    "layer_types",
    "attention_chunk_size",
    "global_attention_every_n_layers",
    # Positional envelope — longrope/yarn scale against the ORIGINAL length.
    "max_position_embeddings",
    "original_max_position_embeddings",
    "position_embedding_type",
    # Architecture family sanity: same id, different model_type is never the
    # same geometry.
    "model_type",
)

# Multimodal configs nest the text model's geometry one level down.
_NESTED_SECTIONS = ("text_config",)


def _config_as_dict(config: Any) -> Optional[Dict[str, Any]]:
    """Duck-typed view of a model config as a plain dict.

    Accepts a dict (config.json content), a transformers ``PretrainedConfig``
    (``to_dict()``), or a dataclass-like object such as mlx_lm's ``ModelArgs``
    (``vars()``). None when no dict view is reachable.
    """
    if config is None:
        return None
    if isinstance(config, dict):
        return config
    to_dict = getattr(config, "to_dict", None)
    if callable(to_dict):
        try:
            data = to_dict()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    try:
        data = vars(config)
        if isinstance(data, dict) and data:
            return dict(data)
    except TypeError:
        pass
    return None


def _canonical_value(value: Any) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return str(value)


def _geometry_view(config_dict: Dict[str, Any]) -> Dict[str, str]:
    """The curated geometry keys present in a config, canonicalized.

    Sections are prefixed so a top-level key and the same key under
    ``text_config`` never collide (both are hashed when both exist).
    """
    view: Dict[str, str] = {}
    sections: Dict[str, Any] = {"": config_dict}
    for name in _NESTED_SECTIONS:
        nested = config_dict.get(name)
        nested_dict = _config_as_dict(nested)
        if nested_dict:
            sections[name] = nested_dict
    for prefix, section in sections.items():
        if not isinstance(section, dict):
            continue
        for key in _GEOMETRY_KEYS:
            if key in section:
                label = f"{prefix}.{key}" if prefix else key
                view[label] = _canonical_value(section[key])
    return view


def model_config_fingerprint_for(config: Any) -> str:
    """Return a stable fingerprint of a model config's KV-geometry identity.

    "" when no config dict is reachable (unverifiable — validators must treat
    it as "cannot verify", never as "matches"). A reachable config with ZERO
    curated keys hashes as an empty geometry: a stable constant that changes
    the moment a geometry key appears.
    """
    config_dict = _config_as_dict(config)
    if config_dict is None:
        return ""
    view = _geometry_view(config_dict)
    digest = hashlib.sha256()
    for key in sorted(view):
        digest.update(key.encode("utf-8", errors="replace"))
        digest.update(b"\x01")
        digest.update(view[key].encode("utf-8", errors="replace"))
        digest.update(b"\x00")
    return f"model-config:sha256:{digest.hexdigest()[:24]}"


def check_model_config_fingerprint(stored: Any, current: Any) -> str:
    """Three-way verdict shared by every gate that consumes the fingerprint.

    Same contract as axis 2 (one shared implementation — the verdict logic is
    identical by design):

    - "mismatch": both known and different — the artifact's KV geometry can no
      longer be trusted; refuse (recompile or raise), never reload.
    - "unverified_stored": the artifact predates this axis — reuse is allowed
      but must be LABELED, never silent.
    - "unverified_current": the current config is unavailable (model not
      loaded at validation time) — comparison abstains; a later gate that has
      the config must re-check.
    - "ok": both known and equal.
    """
    return _shared_verdict(stored, current)
