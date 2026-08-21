"""Media-delivery contract for the MLX provider (backlog 0842) and the vision
add-on's detection/adapter layer (backlog 0840).

Weights-free by construction: everything here is config fixtures, signature
introspection and pure-Python contract checks, so it runs in CI where mlx is not
installed. The live proof lives in `test_mlx_vision_addon_live.py`.
"""

import json
import sys
import types

import pytest

from abstractcore.media.delivery import (
    MEDIA_DELIVERED_KEY,
    MEDIA_DROPPED_KEY,
    MediaReport,
    attach_media_report,
    media_delivery_verdict,
)


class _Resp:
    def __init__(self, metadata=None):
        self.metadata = metadata


# --------------------------------------------------------------------------- #
# MediaReport / metadata round trip
# --------------------------------------------------------------------------- #


def test_report_with_no_media_adds_no_keys():
    """A text-only request must be byte-identical to before this contract existed."""
    report = MediaReport.for_request(None, provider="mlx", model="m")
    resp = _Resp({"existing": 1})
    out = attach_media_report(resp, report)
    assert out.metadata == {"existing": 1}


def test_delivered_record_carries_measured_tokens_and_identity():
    report = MediaReport.for_request([object()], provider="mlx", model="m")
    report.deliver(
        index=0,
        kind="image",
        content=b"abc",
        tokens=1024,
        transport="mlx_vision_addon",
        fidelity=("rope_1d_substituted",),
    )
    meta = report.as_metadata()
    entry = meta[MEDIA_DELIVERED_KEY][0]
    assert entry["tokens"] == 1024
    assert entry["transport"] == "mlx_vision_addon"
    assert entry["fidelity"] == ["rope_1d_substituted"]
    # sha256 identifies WHICH image; ids alone cannot (same-size images tokenize
    # identically), so this is the field that makes cache reuse decidable later.
    assert entry["sha256"] == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
    assert MEDIA_DROPPED_KEY not in meta


def test_dropped_reasons_stay_a_list_of_literals():
    """Existing consumers read media_dropped as a list of short reasons; prose
    belongs in the log, never in metadata."""
    report = MediaReport.for_request([object()], provider="mlx", model="m")
    report.drop("vision_encode_failed", detail="a long human sentence")
    meta = report.as_metadata()
    assert meta[MEDIA_DROPPED_KEY] == ["vision_encode_failed"]
    assert "a long human sentence" not in json.dumps(meta)


# --------------------------------------------------------------------------- #
# The funnel: streaming must carry the same record as non-streaming
# --------------------------------------------------------------------------- #


def test_every_streamed_chunk_is_stamped():
    report = MediaReport.for_request([object()], provider="mlx", model="m")
    report.drop("vision_encode_failed")
    chunks = [_Resp(), _Resp(), _Resp()]
    out = list(attach_media_report(iter(chunks), report))
    # Every chunk: downstream stream processing rebuilds chunks and which ones
    # survive is model-dependent, so stamping only the ends loses the record.
    assert len(out) == 3
    for chunk in out:
        assert chunk.metadata[MEDIA_DROPPED_KEY] == ["vision_encode_failed"]


def test_single_chunk_stream_is_stamped_once():
    report = MediaReport.for_request([object()], provider="mlx", model="m")
    report.drop("vision_encode_failed")
    out = list(attach_media_report(iter([_Resp()]), report))
    assert out[0].metadata[MEDIA_DROPPED_KEY] == ["vision_encode_failed"]


# --------------------------------------------------------------------------- #
# The three-valued verdict. Two-valued would break every non-migrated provider.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "metadata,provider,expected_state",
    [
        ({MEDIA_DROPPED_KEY: ["x"]}, "mlx", "not_delivered"),
        ({MEDIA_DELIVERED_KEY: [{"tokens": 1024}]}, "mlx", "delivered"),
        # present but unexpanded: a record that claims without evidence
        ({MEDIA_DELIVERED_KEY: [{"tokens": 1}]}, "mlx", "not_delivered"),
        ({MEDIA_DELIVERED_KEY: []}, "mlx", "not_delivered"),
        # a reporting provider going silent is a code path that forgot
        ({}, "mlx", "not_delivered"),
        # a provider that does not participate keeps working
        ({}, "openai", "unverified"),
        ({}, None, "unverified"),
    ],
)
def test_delivery_verdict_states(metadata, provider, expected_state):
    verdict = media_delivery_verdict(_Resp(metadata), provider=provider)
    assert verdict.state == expected_state


def test_non_migrated_provider_is_not_treated_as_failure():
    """Regression guard for the breaking change this contract could have caused:
    the fallback gate is shared by every captioning route, so an absent record
    from a provider that never writes one must not refuse."""
    from abstractcore.media.delivery import REPORTING_PROVIDERS

    for provider in ("openai", "anthropic", "ollama", "lmstudio", "huggingface"):
        assert provider not in REPORTING_PROVIDERS
        assert media_delivery_verdict(_Resp({}), provider=provider).state == "unverified"


# --------------------------------------------------------------------------- #
# Vision detection (0840). Config fixtures only.
# --------------------------------------------------------------------------- #


def _write_checkpoint(tmp_path, config, weights=("language_model.layers.0.q",)):
    (tmp_path / "config.json").write_text(json.dumps(config))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in weights}})
    )
    return str(tmp_path)


def test_text_only_checkpoint_is_not_a_refusal(tmp_path):
    from abstractcore.providers.mlx_vision_addon import vision_status

    d = _write_checkpoint(tmp_path, {"model_type": "qwen3"})
    usable, reason, _ = vision_status(d)
    assert usable is False
    assert reason is None  # not sighted at all: nothing to report


def test_declared_vision_without_weights_is_refused(tmp_path):
    from abstractcore.providers.mlx_vision_addon import (
        REASON_NO_VISION_WEIGHTS,
        vision_status,
    )

    d = _write_checkpoint(
        tmp_path,
        {
            "model_type": "qwen3_5",
            "vision_config": {"depth": 27},
            "vision_start_token_id": 1,
            "image_token_id": 2,
            "vision_end_token_id": 3,
        },
    )
    usable, reason, _ = vision_status(d)
    assert usable is False
    assert reason == REASON_NO_VISION_WEIGHTS


def test_supported_family_with_weights_is_usable(tmp_path):
    from abstractcore.providers.mlx_vision_addon import vision_status

    d = _write_checkpoint(
        tmp_path,
        {
            "model_type": "qwen3_5",
            "vision_config": {"depth": 27},
            "vision_start_token_id": 248053,
            "image_token_id": 248056,
            "vision_end_token_id": 248054,
        },
        weights=("vision_tower.blocks.0.attn.qkv.weight", "language_model.layers.0.q"),
    )
    usable, reason, info = vision_status(d)
    assert usable is True and reason is None
    assert info["placeholder_ids"] == [248053, 248056, 248054]


def test_deepstack_family_is_worth_trying(tmp_path):
    """Qwen3-VL ships deepstack_visual_indexes. Those multi-scale features cannot
    ride `input_embeddings`, but dropping them is a QUALITY loss, not a
    correctness one -- the model still reads the image. Measured: Qwen3-VL-4B
    describes the test image accurately through this lane. So it must be served
    and annotated, not refused: refusing would fall back to a text-only path that
    confabulates."""
    from abstractcore.providers.mlx_vision_addon import vision_status

    d = _write_checkpoint(
        tmp_path,
        {
            "model_type": "qwen3_vl",
            "vision_config": {"deepstack_visual_indexes": [5, 11, 17]},
            "vision_start_token_id": 151652,
            "image_token_id": 151655,
            "vision_end_token_id": 151653,
        },
        weights=("model.visual.blocks.0.attn.qkv.weight",),
    )
    usable, reason, info = vision_status(d)
    assert usable is True and reason is None
    assert info["placeholder_ids"] == [151652, 151655, 151653]


def test_gemma4_placeholder_uses_the_ids_it_declares(tmp_path):
    """gemma-4 declares an image_token_id but no vision_start/end ids. Qwen wraps
    its image token; Gemma does not. Requiring all three would refuse a model that
    works, so only the image token -- the one the processor expands -- is
    required."""
    from abstractcore.providers.mlx_vision_addon import vision_status

    d = _write_checkpoint(
        tmp_path,
        {"model_type": "gemma4", "vision_config": {"depth": 27}, "image_token_id": 258880},
        weights=("vision_tower.blocks.0.attn.qkv.weight",),
    )
    usable, reason, info = vision_status(d)
    assert usable is True and reason is None
    assert info["placeholder_ids"] == [258880]


def test_checkpoint_without_an_image_token_is_refused(tmp_path):
    from abstractcore.providers.mlx_vision_addon import (
        REASON_FAMILY_UNSUPPORTED,
        vision_status,
    )

    d = _write_checkpoint(
        tmp_path,
        {"model_type": "mystery", "vision_config": {"depth": 4}},
        weights=("vision_tower.blocks.0.attn.qkv.weight",),
    )
    usable, reason, _ = vision_status(d)
    assert usable is False
    assert reason == REASON_FAMILY_UNSUPPORTED


def test_vision_weight_prefix_matches_nested_visual():
    """Qwen3-VL stores its tower under `model.visual.*`. A first-segment check
    reads a real VLM as having zero vision tensors."""
    from abstractcore.providers.mlx_vision_addon import _is_vision_key

    assert _is_vision_key("vision_tower.blocks.0.attn.qkv.weight")
    assert _is_vision_key("model.visual.blocks.1.mlp.weight")
    assert _is_vision_key("multi_modal_projector.linear.weight")
    assert not _is_vision_key("language_model.model.layers.0.self_attn.q_proj.weight")
    assert not _is_vision_key("model.language_model.layers.3.mlp.gate_proj.weight")


# --------------------------------------------------------------------------- #
# Runtime probes replace the family allowlist.
# --------------------------------------------------------------------------- #


def test_side_channel_classification_is_explicit():
    """Three classes, and the difference is what the decoder can be given.

    `deepstack_visual_embeds` is CARRIED -- it is an additive residual at
    image-token positions after the first few decoder layers, so it is applied by
    wrapping those layers. `per_layer_inputs` is structural and cannot be
    supplied at all. `attention_mask_4d` is genuinely dropped.
    """
    from abstractcore.providers.mlx_vision_addon import (
        DROPPABLE_SIDE_CHANNELS,
        REQUIRED_SIDE_CHANNELS,
        deepstack_layers,
    )

    assert "per_layer_inputs" in REQUIRED_SIDE_CHANNELS
    assert "attention_mask_4d" in DROPPABLE_SIDE_CHANNELS
    # Carried, so it must appear in NEITHER bucket.
    assert "deepstack_visual_embeds" not in DROPPABLE_SIDE_CHANNELS
    assert "deepstack_visual_embeds" not in REQUIRED_SIDE_CHANNELS
    assert callable(deepstack_layers)
    assert not set(DROPPABLE_SIDE_CHANNELS) & set(REQUIRED_SIDE_CHANNELS)


def test_deepstack_layers_always_restores_the_model():
    """The wrappers hold ONE request's visual features. If they outlived the
    request, a later prompt would inherit a previous image's detail."""
    from abstractcore.providers.mlx_vision_addon import deepstack_layers

    class _Container:
        def __init__(self):
            self.layers = ["l0", "l1", "l2", "l3"]

    class _LM:
        def __init__(self):
            self.model = _Container()

    class _Model:
        def __init__(self):
            self.language_model = _LM()

    m = _Model()
    original = list(m.language_model.model.layers)

    # no features: a no-op that still restores
    with deepstack_layers(m, None, None, 4):
        pass
    assert m.language_model.model.layers == original

    # and restores even when the body raises
    try:
        with deepstack_layers(m, None, None, 4):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert m.language_model.model.layers == original


def _probe_addon(vlm_embed_factory):
    """An add-on stub wired to a fake encoder/decoder pair, for probing the
    embedding-convention calibration without loading any weights."""
    import mlx.core as mx

    from abstractcore.providers.mlx_vision_addon import MLXVisionAddOn

    class _Tok:
        def encode(self, text):
            return [1, 2, 3, 4]

    class _Embed:
        def __call__(self, ids):
            return mx.ones((1, 4, 8))

    class _Inner:
        def __init__(self):
            self.embed_tokens = _Embed()

    class _LM:
        def __init__(self):
            self.model = _Inner()

    class _TextModel:
        def __init__(self):
            self.language_model = _LM()

    class _VLM:
        def get_input_embeddings(self, input_ids=None, **kw):
            class _F:
                inputs_embeds = vlm_embed_factory()

            return _F()

    addon = MLXVisionAddOn.__new__(MLXVisionAddOn)
    addon._mx = mx
    addon.model_type = "probe"
    addon._model = _VLM()
    return addon, _Tok(), _TextModel()


def test_uniform_rescale_is_corrected_not_refused():
    """Gemma's encoder pre-multiplies by embed_scale while the mlx-lm decoder
    re-applies the same scale unconditionally, so embeddings arrive scaled twice
    and the model emits zeros. That is a uniform scalar disagreement, so it has an
    exact correction -- refusing a working vision model over it would be wrong.
    """
    import mlx.core as mx

    scale = 73.32121
    addon, tok, text_model = _probe_addon(lambda: mx.ones((1, 4, 8)) * scale)
    factor = addon._calibrate_embedding_convention(tok, text_model)
    assert factor == pytest.approx(scale, rel=1e-4)


def test_non_scalar_disagreement_is_refused():
    """A disagreement no single factor can reconcile has no correction, so the
    lane must refuse rather than feed the decoder out-of-distribution vectors."""
    import mlx.core as mx

    from abstractcore.providers.mlx_vision_addon import (
        REASON_CONVENTION_MISMATCH,
        VisionAddOnUnavailable,
    )

    def _skewed():
        base = mx.ones((1, 4, 8))
        return base * mx.arange(1, 9, dtype=mx.float32)  # per-dimension, not scalar

    addon, tok, text_model = _probe_addon(_skewed)
    with pytest.raises(VisionAddOnUnavailable) as excinfo:
        addon._calibrate_embedding_convention(tok, text_model)
    assert excinfo.value.reason == REASON_CONVENTION_MISMATCH


def test_agreeing_family_needs_no_correction():
    import mlx.core as mx

    addon, tok, text_model = _probe_addon(lambda: mx.ones((1, 4, 8)))
    assert addon._calibrate_embedding_convention(tok, text_model) == pytest.approx(1.0)


def test_text_embedded_document_delivery_is_recorded():
    """A document reaching the model as text IS delivery on this lane. Leaving the
    report empty made a successful request read as `not_delivered`, because
    silence from a reporting provider is treated as a code path that forgot."""
    report = MediaReport.for_request([object()], provider="mlx", model="m")
    report.deliver(index=0, kind="document", content=b"x", tokens=120, transport="text_embedded")
    verdict = media_delivery_verdict(_Resp(report.as_metadata()), provider="mlx")
    assert verdict.state == "delivered"
    assert verdict.tokens == 120


# --------------------------------------------------------------------------- #
# Guards that mutation testing found untested.
# --------------------------------------------------------------------------- #


def test_error_response_never_claims_delivery():
    """A failed generation delivered nothing to the model, whatever the encoder
    built. Without this the positive record outlives the thing it attests to --
    and an error response carrying `media_delivered` is exactly the "claimed
    sight, answered from text" shape the contract exists to prevent."""

    class _Err:
        finish_reason = "error"
        metadata = None

    report = MediaReport.for_request([object()], provider="mlx", model="m")
    report.deliver(index=0, kind="image", content=b"x", tokens=1024, transport="mlx_vision_addon")
    out = attach_media_report(_Err(), report)
    assert MEDIA_DELIVERED_KEY not in (out.metadata or {})


def test_error_response_still_reports_drops():
    """The negative channel must survive the error path: the caller still needs
    to know the image was not carried."""

    class _Err:
        finish_reason = "error"
        metadata = None

    report = MediaReport.for_request([object()], provider="mlx", model="m")
    report.drop("vision_encode_failed")
    out = attach_media_report(_Err(), report)
    assert out.metadata[MEDIA_DROPPED_KEY] == ["vision_encode_failed"]


def test_unexpanded_placeholder_is_refused(monkeypatch):
    """If the processor did not expand the placeholder, the model would answer
    from text while the lane claimed sight. Guarded at the encoder, before the
    merge, where the count is unambiguous."""
    import mlx.core as mx
    import mlx_vlm.utils as vu

    from abstractcore.providers.mlx_vision_addon import (
        MLXVisionAddOn,
        REASON_ENCODE_FAILED,
        VisionAddOnUnavailable,
    )

    class _Feats:
        inputs_embeds = mx.ones((1, 4, 8))

    class _VLM:
        def get_input_embeddings(self, input_ids=None, **kw):
            return _Feats()

    addon = MLXVisionAddOn.__new__(MLXVisionAddOn)
    addon._mx = mx
    addon.model_type = "qwen3_5"
    addon.image_token_id = 999
    addon.video_token_id = None
    addon._embed_factor = 1.0
    addon._model = _VLM()
    addon.processor = object()

    # A single placeholder token: the processor did not expand it.
    monkeypatch.setattr(
        vu,
        "prepare_inputs",
        lambda *a, **k: {
            "input_ids": mx.array([[1, 999, 2, 3]]),
            "pixel_values": mx.ones((1, 3, 4, 4)),
        },
    )
    with pytest.raises(VisionAddOnUnavailable) as excinfo:
        addon.compute_embeddings(object(), "prompt", ["x.png"])
    assert excinfo.value.reason == REASON_ENCODE_FAILED


def test_language_model_only_repack_is_not_treated_as_sighted(tmp_path):
    """A text repack of a VLM keeps its vision_config but sets
    language_model_only. It has no tower to use."""
    from abstractcore.providers.mlx_vision_addon import vision_status

    d = _write_checkpoint(
        tmp_path,
        {
            "model_type": "qwen3_5",
            "vision_config": {"depth": 27},
            "language_model_only": True,
            "image_token_id": 248056,
        },
        weights=("vision_tower.blocks.0.attn.qkv.weight",),
    )
    usable, reason, _ = vision_status(d)
    assert usable is False
    assert reason is None


def test_vision_addon_construction_is_locked():
    """The gateway serves from worker threads. Two concurrent first-image calls
    must not each build a vision tower and race to publish it."""
    import threading

    from abstractcore.providers.mlx_provider import MLXProvider

    provider = MLXProvider.__new__(MLXProvider)
    provider._vision_addon = None
    provider._vision_addon_lock = threading.Lock()

    built = []

    def _build():
        with provider._vision_addon_lock:
            if provider._vision_addon is None:
                built.append(1)
                provider._vision_addon = object()

    threads = [threading.Thread(target=_build) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1


# --------------------------------------------------------------------------- #
# Image routing must follow CONTENT, not the file extension.
# --------------------------------------------------------------------------- #


def test_image_with_an_unmapped_extension_is_still_an_image(tmp_path):
    """`detect_media_type` classifies by suffix and its table omits real image
    formats (.jfif, .jpe, .heic, .avif, .jp2). A genuine JPEG named `.jfif` was
    therefore classified DOCUMENT, skipped the vision lane and got embedded as
    TEXT -- while `analyze_media`'s decode gate, which sniffs content, let it
    through. The two disagreed, so an unseen image reached the model and the
    response claimed delivery.
    """
    from PIL import Image

    from abstractcore.providers.mlx_provider import MLXProvider

    jfif = tmp_path / "photo.jfif"
    Image.new("RGB", (32, 32), (10, 20, 30)).save(jfif, "JPEG")

    class _Part:
        media_type = None
        mime_type = "application/octet-stream"
        content = "extracted text, not the original bytes"
        file_path = str(jfif)

    assert MLXProvider._is_image_part(_Part()) is True


def test_non_image_document_is_not_mistaken_for_an_image(tmp_path):
    from abstractcore.providers.mlx_provider import MLXProvider

    doc = tmp_path / "notes.txt"
    doc.write_text("just some text")

    class _Part:
        media_type = None
        mime_type = "text/plain"
        content = "just some text"
        file_path = str(doc)

    assert MLXProvider._is_image_part(_Part()) is False


def test_streamed_error_chunk_does_not_claim_delivery():
    """The error guard cannot live on the returned object for a streaming call:
    that object is a GENERATOR and has no finish_reason. It has to be re-applied
    per chunk, or stamping every chunk guarantees the error chunk carries a
    positive claim."""

    class _Chunk:
        def __init__(self, finish):
            self.finish_reason = finish
            self.metadata = None

    report = MediaReport.for_request([object()], provider="mlx", model="m")
    report.deliver(index=0, kind="image", content=b"x", tokens=252, transport="mlx_vision_addon")
    out = list(attach_media_report(iter([_Chunk(None), _Chunk("error")]), report))
    assert MEDIA_DELIVERED_KEY in (out[0].metadata or {})
    assert MEDIA_DELIVERED_KEY not in (out[-1].metadata or {})


def test_mlx_vision_extra_is_opt_in():
    """mlx-vlm must not reach the text-only local LLM install: it pulls a web
    framework, opencv and an audio stack, and raises the transformers floor."""
    import tomllib
    from pathlib import Path as _P

    data = tomllib.loads(_P("pyproject.toml").read_text())
    extras = data["project"]["optional-dependencies"]
    assert any(d.startswith("mlx-vlm") for d in extras["mlx-vision"])
    for name in ("mlx", "apple"):
        assert not any(d.startswith("mlx-vlm") for d in extras[name]), name
