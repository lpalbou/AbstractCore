"""mlx-vlm as a vision ENCODER for the mlx-lm decoder.

`mlx_lm` deliberately discards the vision tower at load (see
`mlx_lm/models/qwen3_5.py::Model.sanitize`), so an MLX checkpoint that ships one
is loaded blind. This module supplies the missing half WITHOUT replacing the
runtime: `mlx_lm` still owns the decoder, the KV cache, the sampler and the
tokenizer. `mlx_vlm` is used only to turn pixels into embeddings the decoder can
consume, via `mlx_lm.generate_step(input_embeddings=...)` -- an upstream-supported
entry point. One forward pass, one KV cache.

Why the decoder is never mlx-vlm's: `make_prompt_cache` on an mlx-vlm wrapper
silently builds an all-`KVCache` stack where Qwen3.5 needs
``ArraysCache x 48 + KVCache x 16`` -- 75% of layers wrong, with no exception.
Keeping `self.llm` an mlx-lm model makes that impossible by construction.

Family support is decided by MEASUREMENT, not by an allowlist. Two probes:

* an embedding-convention check (see `_convention_compatible`), because handing
  one library's embeddings to another's decoder is only valid if they agree on
  what an embedding is -- Gemma pre-multiplies by ``embed_scale`` and would
  arrive double-scaled;
* a side-channel check, because some families return more than embeddings and
  `input_embeddings` can only carry the one.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..utils.structured_logging import get_logger

logger = get_logger(__name__)

# Weight-name prefixes that mark a multimodal module. Mirrors the vocabulary
# mlx-vlm itself tolerates (`mlx_vlm/utils.py::skip_multimodal_module`) -- the
# single string "vision_tower" is NOT enough: Qwen3-VL stores its tower under
# `model.visual`, and llava/pixtral/paligemma add `multi_modal_projector`.
VISION_WEIGHT_PREFIXES = (
    "vision_tower",
    "vision_model",
    "visual",
    "multi_modal_projector",
    "vl_connector",
    "img_projector",
)

# Extra outputs `get_input_embeddings` may produce that `input_embeddings` cannot
# carry. Populating one is NOT a refusal -- the model still reads the image -- but
# it is a fidelity loss, so it is annotated on the delivery record rather than
# absorbed silently (ADR 0001).
DROPPABLE_SIDE_CHANNELS = ("attention_mask_4d",)

# Outputs the decoder structurally needs. A family that populates one of these
# cannot be served, because the information has nowhere to go.
REQUIRED_SIDE_CHANNELS = ("per_layer_inputs", "cross_attention_states")

# Named refusal reasons. These land in `response.metadata["media_dropped"]`, so
# they are part of the observable contract -- keep them stable.
REASON_NOT_INSTALLED = "mlx_vlm_not_installed"
REASON_FAMILY_UNSUPPORTED = "vision_family_unsupported"
REASON_NO_VISION_WEIGHTS = "vision_weights_absent"
REASON_ENCODE_FAILED = "vision_encode_failed"
REASON_CONVENTION_MISMATCH = "vision_embedding_convention_mismatch"


class VisionAddOnUnavailable(Exception):
    """The lane cannot serve this checkpoint. Carries a named `reason`."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


def _is_vision_key(key: str) -> bool:
    """Match a weight name against the multimodal prefix vocabulary.

    Matched over the first two dotted segments, not `key.split(".")[0]`:
    Qwen3-VL stores its tower under `model.visual.*`, so a first-segment check
    reads a real VLM as having zero vision tensors.
    """
    return any(seg in VISION_WEIGHT_PREFIXES for seg in key.split(".")[:2])


def _read_config(model_dir: str) -> Dict[str, Any]:
    try:
        return json.loads((Path(model_dir) / "config.json").read_text())
    except Exception:
        return {}


def _has_vision_weights(model_dir: str) -> bool:
    """True if vision-module tensors are actually present on disk.

    A `vision_config` with no matching weights is a language-only re-export, so
    the config alone would lie. Handles BOTH the sharded layout (an index file)
    and the single-shard layout, since a real VLM may ship one safetensors file.
    """
    d = Path(model_dir)
    index = d / "model.safetensors.index.json"
    if index.exists():
        try:
            weight_map = json.loads(index.read_text()).get("weight_map", {})
            return any(_is_vision_key(k) for k in weight_map)
        except Exception:
            return False
    for shard in d.glob("*.safetensors"):
        try:
            from safetensors import safe_open

            with safe_open(str(shard), framework="numpy") as f:
                if any(_is_vision_key(k) for k in f.keys()):
                    return True
        except Exception:
            continue
    return False


def vision_status(model_dir: str) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """Could this checkpoint be served by the add-on lane?

    Config and weight-header reads only -- no model load, no mlx-vlm import, safe
    to call during `_load_model`. This answers "is it worth trying"; the
    authoritative answer comes from actually building the add-on, which runs the
    runtime probes.

    Returns ``(worth_trying, reason, info)``. ``reason`` is a named literal when
    a sighted checkpoint is refused, and ``None`` when the checkpoint simply is
    not a VLM (nothing to report).
    """
    cfg = _read_config(model_dir)
    info: Dict[str, Any] = {"model_type": cfg.get("model_type")}
    vision_cfg = cfg.get("vision_config")
    if not vision_cfg or cfg.get("language_model_only") is True:
        return False, None, info
    if not _has_vision_weights(model_dir):
        return False, REASON_NO_VISION_WEIGHTS, info

    # The placeholder AbstractCore renders into its own prompt. Qwen wraps the
    # image token in vision_start/vision_end; Gemma declares only the image token.
    # Use whichever the checkpoint actually declares -- the image token is the one
    # the processor expands, and it is the only one strictly required.
    ids = [
        i
        for i in (
            cfg.get("vision_start_token_id"),
            cfg.get("image_token_id"),
            cfg.get("vision_end_token_id"),
        )
        if i is not None
    ]
    if cfg.get("image_token_id") is None:
        # Without it we cannot mark where the image goes, and AbstractCore owns
        # prompt rendering.
        return False, REASON_FAMILY_UNSUPPORTED, info

    info["placeholder_ids"] = ids
    # Only M-RoPE families lose positional fidelity on this lane: mlx-lm
    # recognises "mrope" and substitutes plain 1-D RoPE. Gemma uses ordinary
    # RoPE, so claiming the substitution for it would be a false annotation.
    # Both spellings occur in the wild for the SAME behaviour: Qwen3.5 writes
    # `rope_parameters`, Qwen3-VL writes `rope_scaling`. Reading only one gives a
    # false negative on the family where the substitution actually costs the most.
    text_cfg = cfg.get("text_config") or cfg
    rope = text_cfg.get("rope_parameters") or text_cfg.get("rope_scaling") or {}
    info["uses_mrope"] = bool(isinstance(rope, dict) and rope.get("mrope_section"))
    info["image_token_id"] = cfg.get("image_token_id")
    info["video_token_id"] = cfg.get("video_token_id")
    return True, None, info


class _DeepStackLayer:
    """A decoder layer that adds the multi-scale visual residual after itself.

    Qwen3-VL's vision tower returns intermediate features from several ViT depths
    in addition to the merged embedding. Its own runtime adds those into the
    hidden states at image-token positions after each of the first N decoder
    layers (`mlx_vlm/models/qwen3_vl/language.py`) -- an additive residual, which
    is why it can be reproduced by composing a wrapper around the layer objects
    of the mlx-lm model we already hold, rather than patching anything upstream.

    Dropping it is not free: measured on Qwen3-VL-4B, the word MARMALADE reads as
    "MARMARMALMALADEADE" without this and "MARMALADE" with it.
    """

    def __init__(self, inner: Any, embeds: Any, indices: Any, seq_len: int):
        self._inner = inner
        self._embeds = embeds
        self._indices = indices
        self._seq_len = seq_len
        self._max_index = int(indices.max().item()) if indices.size else -1

    def __call__(self, x, *args, **kwargs):
        h = self._inner(x, *args, **kwargs)
        # Only the prompt pass carries image tokens; decode steps feed one token
        # at a time and must be left alone.
        #
        # Do NOT test for an exact `seq_len` match: `generate_step` holds the
        # last prompt token back for the first sampling step, so the prompt pass
        # is one shorter than the sequence. Requiring equality silently never
        # fired. Bounds-checking the indices is both correct and self-evident.
        if h.ndim == 3 and h.shape[0] == 1 and h.shape[1] > 1 and self._max_index < h.shape[1]:
            merged = h[0]
            merged = merged.at[self._indices].add(self._embeds.astype(h.dtype))
            h = merged[None]
        return h

    def __getattr__(self, name):
        return getattr(self._inner, name)


@contextmanager
def deepstack_layers(text_model: Any, embeds: Any, pos_masks: Any, seq_len: int):
    """Install the visual residual for ONE forward, then always restore.

    The wrappers hold this request's features, so they must never outlive it --
    a later request with a different image would otherwise inherit them.
    """
    import numpy as np

    import mlx.core as mx

    container = text_model.language_model.model
    original = list(container.layers)
    if embeds is None or pos_masks is None:
        yield
        return
    try:
        idx = mx.array(np.where(np.array(pos_masks[0], copy=False))[0], dtype=mx.uint32)
        n = min(len(embeds), len(original))
        container.layers = [
            _DeepStackLayer(original[i], embeds[i], idx, seq_len) for i in range(n)
        ] + original[n:]
        yield
    finally:
        container.layers = original


class MLXVisionAddOn:
    """Turns (rendered prompt, images) into embeddings the mlx-lm decoder eats.

    Construct lazily -- only when an image actually arrives -- so a text-only
    session never imports mlx-vlm.
    """

    _PROBE_TEXT = "The quick brown fox."

    def __init__(self, model_dir: str, tokenizer: Any, info: Dict[str, Any], text_model: Any):
        try:
            from mlx_vlm import load as vlm_load
        except ImportError as exc:  # pragma: no cover - depends on the extra
            raise VisionAddOnUnavailable(REASON_NOT_INSTALLED, str(exc)) from exc

        import mlx.core as mx

        self._mx = mx
        self.model_type = info.get("model_type")

        # `lazy=True` leaves parameters unevaluated, so nothing is materialised
        # until it is used. Measured on a 27B: the add-on costs +0.000 GB here and
        # +0.859 GB once the tower is evaluated, against a resident decoder.
        self._model, self.processor = vlm_load(model_dir, lazy=True)
        mx.eval(self._model.vision_tower.parameters())

        if not hasattr(self._model, "get_input_embeddings"):
            raise VisionAddOnUnavailable(
                REASON_FAMILY_UNSUPPORTED,
                f"{self.model_type}: no get_input_embeddings entry point",
            )

        self._embed_factor = self._calibrate_embedding_convention(tokenizer, text_model)

        self.image_token_id = info["image_token_id"]
        self.video_token_id = info.get("video_token_id")
        self.placeholder = "".join(tokenizer.decode([i]) for i in info["placeholder_ids"])

    def _calibrate_embedding_convention(self, tokenizer: Any, text_model: Any) -> float:
        """Reconcile the two libraries' embedding conventions for this family.

        Handing mlx-vlm's embeddings to mlx-lm's decoder is only valid if both
        agree on what an embedding IS. Gemma does not agree: mlx-vlm multiplies by
        ``embed_scale`` inside `get_input_embeddings`, while
        `mlx_lm/models/gemma4_text.py` applies ``h = h * self.embed_scale``
        UNCONDITIONALLY -- including when embeddings are supplied. The vectors
        therefore arrive scaled twice (measured: x73.32 = sqrt(5376) on
        gemma-4-31b) and the model emits `000000000...`.

        Rather than refuse, measure the disagreement and correct it. Returns the
        factor to divide encoder output by before handing it over; 1.0 when the
        conventions already agree (every Qwen family). Refuses only when the
        disagreement is NOT a uniform scalar, i.e. when no correction exists.

        Measured at runtime rather than encoded as a per-family table, so an
        upstream convention change is corrected or caught, never mis-served.
        """
        mx = self._mx
        ids = mx.array([tokenizer.encode(self._PROBE_TEXT)])
        try:
            feats = self._model.get_input_embeddings(input_ids=ids)
            theirs = getattr(feats, "inputs_embeds", feats)
            ours = text_model.language_model.model.embed_tokens(ids)
            mx.eval(theirs, ours)
        except Exception as exc:
            raise VisionAddOnUnavailable(
                REASON_ENCODE_FAILED, f"embedding-convention probe failed: {exc}"
            ) from exc

        if theirs.shape != ours.shape:
            raise VisionAddOnUnavailable(
                REASON_CONVENTION_MISMATCH,
                f"{self.model_type}: encoder returned {theirs.shape}, decoder "
                f"expects {ours.shape}",
            )

        # Norm ratio rather than an elementwise divide: embeddings contain zeros.
        ours_norm = float(mx.sqrt((ours.astype(mx.float32) ** 2).sum()).item())
        theirs_norm = float(mx.sqrt((theirs.astype(mx.float32) ** 2).sum()).item())
        if ours_norm <= 0.0:
            raise VisionAddOnUnavailable(
                REASON_CONVENTION_MISMATCH, f"{self.model_type}: null decoder embeddings"
            )
        factor = theirs_norm / ours_norm

        # A scalar factor is only a valid correction if it actually reconciles the
        # two elementwise. If it does not, the disagreement is structural.
        residual = float(mx.abs(theirs / factor - ours).max().item())
        # Scaled to the actual magnitude of an embedding entry, not to a flat
        # floor: a `max(1.0, ...)` floor made the tolerance ~100x the mean entry,
        # so a factor wrong by 1% -- or a purely additive offset -- passed as a
        # clean scalar correction.
        scale = float(mx.abs(ours).mean().item())
        tolerance = max(1e-3 * scale, 1e-6)
        if residual > tolerance:
            raise VisionAddOnUnavailable(
                REASON_CONVENTION_MISMATCH,
                f"{self.model_type}: encoder and decoder embeddings differ by more "
                f"than a scalar (residual {residual:.4g} after factor {factor:.4g})",
            )
        if abs(factor - 1.0) > 1e-3:
            logger.info(
                f"mlx vision add-on: correcting {self.model_type} embedding "
                f"convention by 1/{factor:.5f} (encoder pre-scales; the mlx-lm "
                "decoder re-applies the same scale)"
            )
        return factor

    def compute_embeddings(
        self, text_model: Any, rendered_prompt: str, image_paths: List[str]
    ) -> Tuple[Any, Any, int, Tuple[str, ...], Dict[str, Any]]:
        """Return ``(input_ids, merged_embeddings, n_image_tokens, fidelity, side)``.

        ``side`` carries the multi-scale features the decoder needs installed for
        the forward pass; see `deepstack_layers`.

        `rendered_prompt` is AbstractCore's own prompt, already carrying one vision
        placeholder per image; the processor expands each into as many tokens as
        the image grid requires. AbstractCore keeps ownership of rendering,
        thinking control and tool transcripts -- mlx-vlm's chat template is never
        invoked.
        """
        from mlx_vlm.utils import prepare_inputs

        mx = self._mx
        inputs = prepare_inputs(
            self.processor,
            images=image_paths,
            prompts=rendered_prompt,
            image_token_index=self.image_token_id,
        )
        input_ids = inputs["input_ids"]

        # One uniform entry point across every family, rather than a per-family
        # merge adapter. mlx-vlm's own merge signatures differ in argument ORDER
        # between families that share an arity, so a positional adapter would
        # mis-bind silently; this avoids the hazard rather than guarding it.
        passthrough = {
            k: v
            for k, v in inputs.items()
            if k
            in (
                "pixel_values",
                "pixel_values_videos",
                "image_grid_thw",
                "video_grid_thw",
                "attention_mask",
            )
        }
        if "attention_mask" in passthrough:
            passthrough["mask"] = passthrough.pop("attention_mask")
        feats = self._model.get_input_embeddings(input_ids=input_ids, **passthrough)
        merged = getattr(feats, "inputs_embeds", feats)

        blocking = [
            name for name in REQUIRED_SIDE_CHANNELS if getattr(feats, name, None) is not None
        ]
        if blocking:
            raise VisionAddOnUnavailable(
                REASON_FAMILY_UNSUPPORTED,
                f"{self.model_type}: the decoder needs {', '.join(blocking)}, which "
                "input_embeddings cannot carry",
            )
        fidelity = tuple(
            name for name in DROPPABLE_SIDE_CHANNELS if getattr(feats, name, None) is not None
        )

        side = {
            "deepstack_visual_embeds": getattr(feats, "deepstack_visual_embeds", None),
            "visual_pos_masks": getattr(feats, "visual_pos_masks", None),
            "seq_len": int(input_ids.shape[-1]),
        }
        n_image_tokens = int((input_ids == self.image_token_id).sum().item())
        if n_image_tokens < 2:
            # The processor did not expand the placeholder. Downstream this would
            # still "succeed" while the model answered from text alone.
            raise VisionAddOnUnavailable(
                REASON_ENCODE_FAILED,
                f"the image placeholder expanded to {n_image_tokens} token(s)",
            )
        # Undo the encoder's own embedding scaling where the decoder re-applies
        # it (see _calibrate_embedding_convention). 1.0 for every Qwen family.
        if abs(self._embed_factor - 1.0) > 1e-3:
            merged = merged / self._embed_factor
        mx.eval(merged)
        return input_ids[0], merged[0], n_image_tokens, fidelity, side

    def close(self) -> None:
        self._model = None
        self.processor = None
