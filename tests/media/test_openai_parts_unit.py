"""CPU-only transport tests for native OpenAI-compatible image parts."""

import base64
import copy
import subprocess
import sys

import pytest

from abstractcore.media import openai_parts as module
from abstractcore.media.openai_parts import extract_native_chat_media
from abstractcore.media.types import ContentFormat, MediaType


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aO2sAAAAASUVORK5CYII="
)
GIF = base64.b64decode("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7")


def image_part(data=PNG, mime="image/png", **fields):
    return {"type": "image_url", "image_url": {
        "url": "data:" + mime + ";base64," + base64.b64encode(data).decode("ascii"),
        **fields,
    }}


def user(*parts):
    return [{"role": "user", "content": list(parts)}]


def forbid_decoding(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid or over-limit requests must be rejected before decoding")
    monkeypatch.setattr(module.base64, "b64decode", forbidden)


def test_text_history_is_preserved_and_deep_copied():
    messages = [
        {"role": "system", "content": "Be concise.", "metadata": {"labels": ["system"]}},
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "call1", "function": {"name": "inspect"}}]},
        {"role": "tool", "tool_call_id": "call1", "content": "Result"},
        {"role": "user", "name": "user1"},
    ]
    before = copy.deepcopy(messages)
    copied, media = extract_native_chat_media(messages)
    assert copied == before
    assert not media
    assert copied is not messages
    copied[0]["metadata"]["labels"].append("changed")
    copied[2]["tool_calls"][0]["function"]["name"] = "changed"
    assert messages == before


def test_text_arrays_are_joined_and_images_keep_input_order_without_mutation():
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Instructions"}]},
        {"role": "user", "content": [
            {"type": "text", "text": "First image:"}, image_part(detail="high"),
            {"type": "text", "text": "Second image:"}, image_part(GIF, "image/gif"),
            {"type": "text", "text": "Compare."},
        ], "metadata": {"labels": ["original"]}},
    ]
    before = copy.deepcopy(messages)
    copied, media = extract_native_chat_media(messages)
    assert copied[0]["content"] == "Instructions"
    assert copied[1]["content"] == "First image:\nSecond image:\nCompare."
    assert [item.content for item in media] == [PNG, GIF]
    assert [item.mime_type for item in media] == ["image/png", "image/gif"]
    assert all(item.media_type is MediaType.IMAGE for item in media)
    assert all(item.content_format is ContentFormat.BINARY for item in media)
    assert all(item.file_path is None for item in media)
    assert media[0].metadata == {"detail": "high"}
    assert media[1].metadata == {}
    copied[1]["metadata"]["labels"].append("changed")
    media[0].metadata["detail"] = "low"
    assert messages == before


@pytest.mark.parametrize("mime,data", [
    ("image/png", PNG), ("image/gif", GIF), ("image/gif", b"GIF87a"),
    ("image/jpeg", b"\xff\xd8\xff\xe0"),
    ("image/webp", b"RIFF\x04\x00\x00\x00WEBP"),
])
def test_supported_mime_signatures_leave_actual_bitmap_decoding_to_provider(mime, data):
    copied, media = extract_native_chat_media(user(image_part(data, mime)))
    assert copied == [{"role": "user", "content": ""}]
    assert media[0].content == data
    assert media[0].mime_type == mime


@pytest.mark.parametrize("detail", [None, "auto", "low", "high"])
def test_valid_detail_is_metadata_only(detail):
    _, media = extract_native_chat_media(user(image_part(detail=detail)))
    assert media[0].metadata == ({"detail": detail} if detail else {})


def test_empty_message_list_and_empty_content_array():
    assert extract_native_chat_media([]) == ([], [])
    assert extract_native_chat_media(user()) == ([{"role": "user", "content": ""}], [])


def test_case_insensitive_data_scheme_and_mime():
    part = image_part()
    part["image_url"]["url"] = part["image_url"]["url"].replace(
        "data:image/png;base64,", "DATA:IMAGE/PNG;BASE64,"
    )
    _, media = extract_native_chat_media(user(part))
    assert media[0].mime_type == "image/png"


@pytest.mark.parametrize("url", [
    "https://secret.example/private.png?token=secret",
    "http://127.0.0.1:8080/private", "file:///private/secret.png",
    "/private/secret.png", "ftp://secret.example/a.png", "//secret.example/a.png",
])
def test_remote_and_file_urls_are_rejected_without_echoing_the_url(url, monkeypatch):
    forbid_decoding(monkeypatch)
    with pytest.raises(ValueError, match="inline base64") as exc:
        extract_native_chat_media(user({"type": "image_url", "image_url": {"url": url}}))
    assert url not in str(exc.value)
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize("role", ["assistant", "system", "tool", None])
def test_images_require_final_user_role(role, monkeypatch):
    forbid_decoding(monkeypatch)
    with pytest.raises(ValueError, match="final user message"):
        extract_native_chat_media([{"role": role, "content": [image_part()]}])


def test_image_bearing_history_is_rejected_not_silently_moved(monkeypatch):
    forbid_decoding(monkeypatch)
    with pytest.raises(ValueError, match="conversation history"):
        extract_native_chat_media(user(image_part()) + [{"role": "user", "content": "Now compare"}])


@pytest.mark.parametrize("messages", [
    None, {}, (), [None], ["message"], [{"role": "user", "content": 7}],
    [{"role": "user", "content": {"text": "not an array"}}],
])
def test_malformed_messages_raise_value_error(messages):
    with pytest.raises(ValueError):
        extract_native_chat_media(messages)


@pytest.mark.parametrize("part", [
    "text", None, {}, {"type": "audio", "audio": "secret"},
    {"type": "text"}, {"type": "text", "text": 4},
    {"type": "text", "text": "ok", "unknown": True},
    {"type": "image_url"}, {"type": "image_url", "image_url": "data:image/png;base64,AAAA"},
    {"type": "image_url", "image_url": {"url": "", "unknown": True}},
    {"type": "image_url", "image_url": {"url": ""}, "unknown": True},
])
def test_unknown_or_malformed_parts_are_rejected(part, monkeypatch):
    forbid_decoding(monkeypatch)
    with pytest.raises(ValueError):
        extract_native_chat_media(user(part))


@pytest.mark.parametrize("detail", ["medium", 1, True, {}, []])
def test_unsupported_detail_is_rejected(detail, monkeypatch):
    forbid_decoding(monkeypatch)
    with pytest.raises(ValueError, match="detail"):
        extract_native_chat_media(user(image_part(detail=detail)))


@pytest.mark.parametrize("header", [
    "data:image/svg+xml;base64,", "data:application/octet-stream;base64,",
    "data:image/jpg;base64,", "data:image/png;charset=utf-8;base64,",
    "data:image/png,", "data:;base64,", "data:image/png;base64",
    "data:" + "x" * 100 + ";base64,",
])
def test_unsupported_mime_or_data_url_headers_fail_before_decode(header, monkeypatch):
    forbid_decoding(monkeypatch)
    with pytest.raises(ValueError):
        extract_native_chat_media(user({"type": "image_url", "image_url": {
            "url": header + base64.b64encode(PNG).decode("ascii"),
        }}))


@pytest.mark.parametrize("encoded", [
    "", "AAAA\n", " AAAA", "AA-A", "AA_A", "AAAA!AAA", "AA=AAAA=",
    "AAAA====", "AA==AAAA", "AA", "A===", "éAAA", "AAAA\x00",
])
def test_nonstandard_or_malformed_base64_is_rejected_before_decode(encoded, monkeypatch):
    forbid_decoding(monkeypatch)
    with pytest.raises(ValueError):
        extract_native_chat_media(user({"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + encoded,
        }}))


def test_noncanonical_pad_bits_are_rejected():
    # All four variants decode to the same 8-byte PNG signature. Only the
    # encoder-produced ending "o=" is canonical (the low pad bits must be zero).
    with pytest.raises(ValueError, match="canonical"):
        extract_native_chat_media(user({"type": "image_url", "image_url": {
            "url": "data:image/png;base64,iVBORw0KGgp=",
        }}))


@pytest.mark.parametrize("mime,data", [
    ("image/jpeg", PNG), ("image/png", GIF), ("image/gif", PNG),
    ("image/webp", b"RIFFxxxxNOPE"), ("image/png", b"plain text"),
])
def test_mime_signature_mismatch_is_rejected(mime, data):
    with pytest.raises(ValueError, match="declared MIME"):
        extract_native_chat_media(user(image_part(data, mime)))


def test_image_size_exact_boundary_is_allowed(monkeypatch):
    monkeypatch.setattr(module, "MAX_IMAGE_BYTES", len(PNG))
    _, media = extract_native_chat_media(user(image_part()))
    assert len(media[0].content) == len(PNG)


@pytest.mark.parametrize("limit", [1, 8, len(PNG) - 1])
def test_per_image_limit_checked_before_decode(monkeypatch, limit):
    monkeypatch.setattr(module, "MAX_IMAGE_BYTES", limit)
    forbid_decoding(monkeypatch)
    with pytest.raises(ValueError, match="per-image limit"):
        extract_native_chat_media(user(image_part()))


def test_aggregate_limit_checked_before_decoding_any_image(monkeypatch):
    monkeypatch.setattr(module, "MAX_TOTAL_IMAGE_BYTES", len(PNG) + len(GIF) - 1)
    forbid_decoding(monkeypatch)
    with pytest.raises(ValueError, match="total decoded limit"):
        extract_native_chat_media(user(image_part(), image_part(GIF, "image/gif")))


def test_aggregate_exact_boundary_is_allowed(monkeypatch):
    monkeypatch.setattr(module, "MAX_TOTAL_IMAGE_BYTES", len(PNG) + len(GIF))
    _, media = extract_native_chat_media(user(image_part(), image_part(GIF, "image/gif")))
    assert len(media) == 2


def test_image_count_limit_checked_before_decoding_any_image(monkeypatch):
    forbid_decoding(monkeypatch)
    with pytest.raises(ValueError, match="at most 16"):
        extract_native_chat_media(user(*(image_part() for _ in range(17))))


def test_exactly_sixteen_images_are_allowed():
    _, media = extract_native_chat_media(user(*(image_part() for _ in range(16))))
    assert len(media) == 16
    assert len({id(item.metadata) for item in media}) == 16


def test_later_invalid_part_rejects_whole_request_before_decode(monkeypatch):
    messages = user(image_part(), {"type": "input_audio", "data": "secret"})
    before = copy.deepcopy(messages)
    forbid_decoding(monkeypatch)
    with pytest.raises(ValueError, match="unsupported"):
        extract_native_chat_media(messages)
    assert messages == before


def test_helper_does_not_import_mlx_or_image_decoders():
    # A fresh interpreter avoids interference from unrelated test modules.
    result = subprocess.run([
        sys.executable, "-c",
        "import sys; import abstractcore.media.openai_parts; "
        "assert not any(n == 'mlx' or n.startswith('mlx.') or "
        "n == 'PIL' or n.startswith('PIL.') for n in sys.modules)",
    ], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
