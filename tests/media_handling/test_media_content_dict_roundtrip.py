from __future__ import annotations


def test_media_content_from_dict_parses_basic_shape() -> None:
    from abstractcore.media.types import ContentFormat, MediaContent, MediaType

    raw = {
        "media_type": "image",
        "content": "ZGF0YQ==",
        "content_format": "base64",
        "mime_type": "image/png",
        "metadata": {"source": "test"},
    }

    mc = MediaContent.from_dict(raw)
    assert mc.media_type == MediaType.IMAGE
    assert mc.content_format == ContentFormat.BASE64
    assert mc.mime_type == "image/png"
    assert mc.metadata.get("source") == "test"


def test_media_content_to_dict_is_json_safe() -> None:
    from abstractcore.media.types import ContentFormat, MediaContent, MediaType

    mc = MediaContent(
        media_type=MediaType.DOCUMENT,
        content=b"hello",
        content_format=ContentFormat.BINARY,
        mime_type="text/plain",
        metadata={"k": "v"},
    )

    d = mc.to_dict()
    assert isinstance(d, dict)
    assert d.get("media_type") == "document"
    assert d.get("content_format") == "binary"
    assert isinstance(d.get("content"), str)  # base64-encoded

