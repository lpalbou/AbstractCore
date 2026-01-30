from __future__ import annotations

import pytest


@pytest.mark.basic
def test_merge_enrichment_metadata_appends_list() -> None:
    from abstractcore.media.enrichment import MEDIA_ENRICHMENT_KEY, build_enrichment_item, merge_enrichment_metadata

    item = build_enrichment_item(
        status="used",
        input_modality="image",
        summary_kind="caption",
        policy="caption",
        backend={"kind": "llm", "provider": "stub", "model": "stub"},
        input_index=1,
        input_name="example.png",
        injected_text="A red square.",
    )

    md = merge_enrichment_metadata(None, [item])
    assert isinstance(md, dict)
    assert isinstance(md.get(MEDIA_ENRICHMENT_KEY), list)
    assert md[MEDIA_ENRICHMENT_KEY][0]["status"] == "used"

    md2 = merge_enrichment_metadata(md, [item])
    assert len(md2[MEDIA_ENRICHMENT_KEY]) == 2


@pytest.mark.basic
def test_local_media_handler_records_skipped_when_vision_not_configured(monkeypatch, sample_media_files) -> None:
    from abstractcore.media.handlers import LocalMediaHandler
    from abstractcore.media.types import ContentFormat, MediaContent, MediaType
    from abstractcore.media.vision_fallback import VisionFallbackHandler, VisionNotConfiguredError

    def raise_not_configured(self, image_path, user_prompt=None):
        raise VisionNotConfiguredError("Vision fallback is disabled")

    monkeypatch.setattr(VisionFallbackHandler, "create_description_with_trace", raise_not_configured)

    handler = LocalMediaHandler("openrouter", {"vision_support": False}, model_name="qwen/qwen3-next-80b")

    image = MediaContent(
        media_type=MediaType.IMAGE,
        content=str(sample_media_files["png"]),
        content_format=ContentFormat.FILE_PATH,
        mime_type="image/png",
        file_path=str(sample_media_files["png"]),
        metadata={"file_name": sample_media_files["png"].name},
    )

    msg = handler.create_multimodal_message("What is in this image?", [image])
    assert isinstance(msg, str)

    enrichments = getattr(handler, "media_enrichment", None)
    assert isinstance(enrichments, list)
    assert enrichments
    entry = enrichments[0]
    assert entry.get("status") == "skipped"
    assert entry.get("input_modality") == "image"
    assert entry.get("summary_kind") == "caption"

