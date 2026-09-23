"""Bounded inline-image normalization for native OpenAI-compatible chat.

This adapter never fetches a URL or opens a file. Native providers receive
copied text messages and binary MediaContent instead of JSON-stringified image
parts. Only the final user message may contain images: attaching history images
to the latest turn would change their meaning. Image decoding remains the native
processor's responsibility; this module validates transport and MIME signatures.
"""

from __future__ import annotations

import base64
import binascii
import copy
import re
from dataclasses import dataclass

from .types import ContentFormat, MediaContent, MediaType

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 32 * 1024 * 1024
MAX_IMAGES = 16
_IMAGE_MIMES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
_BASE64 = re.compile(r"[A-Za-z0-9+/]+={0,2}\Z")


@dataclass(frozen=True)
class _InlineImage:
    mime_type: str
    encoded: str
    size: int
    detail: str | None
    location: str


def _parse_image_url(value, location: str) -> _InlineImage:
    if not isinstance(value, dict) or set(value) - {"url", "detail"}:
        raise ValueError(f"{location}: image_url must contain url and optional detail only")
    url = value.get("url")
    if not isinstance(url, str) or url[:5].lower() != "data:":
        raise ValueError(f"{location}: native vision accepts inline base64 image data URLs only; HTTP and file URLs are unsupported")
    detail = value.get("detail")
    if detail is not None and detail not in ("auto", "low", "high"):
        raise ValueError(f"{location}: image detail must be auto, low, or high")
    # Search a bounded header before slicing the potentially large payload.
    comma = url.find(",", 0, 80)
    if comma < 0:
        raise ValueError(f"{location}: malformed image data URL header")
    header = url[5:comma].lower()
    if not header.endswith(";base64") or header[:-7] not in _IMAGE_MIMES:
        raise ValueError(f"{location}: expected base64 PNG, JPEG, WebP, or GIF image MIME type")
    encoded_length = len(url) - comma - 1
    if not encoded_length or encoded_length % 4:
        raise ValueError(f"{location}: image payload must use padded standard base64")
    if encoded_length > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError(f"{location}: decoded image exceeds the 10 MiB per-image limit")
    encoded = url[comma + 1:]
    if _BASE64.fullmatch(encoded) is None:
        raise ValueError(f"{location}: image payload is not strict standard base64")
    padding = 2 if encoded.endswith("==") else 1 if encoded.endswith("=") else 0
    size = encoded_length // 4 * 3 - padding
    if size <= 0 or size > MAX_IMAGE_BYTES:
        raise ValueError(f"{location}: decoded image exceeds the 10 MiB per-image limit")
    return _InlineImage(header[:-7], encoded, size, detail, location)


def _matches_mime(data: bytes, mime: str) -> bool:
    if mime == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime == "image/gif":
        return data.startswith((b"GIF87a", b"GIF89a"))
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def extract_native_chat_media(messages: list[dict]) -> tuple[list[dict], list[MediaContent]]:
    """Return independent text messages and ordered binary inline images.

    String/None content and all message-level fields retain their values.
    Text-part arrays become newline-joined text. Images are accepted only in
    the final message when its role is ``user``. Optional OpenAI ``detail`` is
    retained as metadata; image resizing is still owned by the native processor.
    Limits cover decoded bytes and are checked for the whole request before any
    base64 decoding. No MLX, Pillow, network or filesystem access is involved.

    Raises ValueError for unsupported or malformed request content. Exceptions
    never include the image payload or a supplied remote URL.
    """
    if not isinstance(messages, list):
        raise ValueError("Native chat messages must be a list of message objects")
    copied, images = [], []
    total_bytes = 0
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"Message {message_index}: expected a message object")
        content = message.get("content")
        if content is None or isinstance(content, str):
            copied.append(copy.deepcopy(message))
            continue
        if not isinstance(content, list):
            raise ValueError(f"Message {message_index}: content must be text, null, or a list of supported parts")
        text_parts = []
        for part_index, part in enumerate(content):
            location = f"Message {message_index}, part {part_index}"
            if not isinstance(part, dict):
                raise ValueError(f"{location}: content parts must be typed objects")
            kind = part.get("type")
            if kind == "text":
                if set(part) - {"type", "text"} or not isinstance(part.get("text"), str):
                    raise ValueError(f"{location}: text parts must contain a string text field only")
                text_parts.append(part["text"])
            elif kind == "image_url":
                if set(part) - {"type", "image_url"}:
                    raise ValueError(f"{location}: image parts must contain an image_url field only")
                if message_index != len(messages) - 1 or message.get("role") != "user":
                    raise ValueError(f"{location}: native images are supported only in the final user message, not in conversation history")
                if len(images) >= MAX_IMAGES:
                    raise ValueError("Native chat supports at most 16 images per request")
                image = _parse_image_url(part.get("image_url"), location)
                total_bytes += image.size
                if total_bytes > MAX_TOTAL_IMAGE_BYTES:
                    raise ValueError("Native chat images exceed the 32 MiB total decoded limit")
                images.append(image)
            else:
                raise ValueError(f"{location}: unsupported native content part type; expected text or image_url")
        out = copy.deepcopy(message)
        out["content"] = "\n".join(text_parts)
        copied.append(out)

    media = []
    for image in images:
        try:
            data = base64.b64decode(image.encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError(f"{image.location}: invalid base64 image payload") from exc
        if len(data) != image.size or base64.b64encode(data).decode("ascii") != image.encoded:
            raise ValueError(f"{image.location}: image base64 encoding is not canonical")
        if not _matches_mime(data, image.mime_type):
            raise ValueError(f"{image.location}: image bytes do not match the declared MIME type")
        media.append(MediaContent(
            media_type=MediaType.IMAGE, content=data, content_format=ContentFormat.BINARY,
            mime_type=image.mime_type,
            metadata={"detail": image.detail} if image.detail is not None else {},
        ))
    return copied, media
