from __future__ import annotations

import os
from io import BytesIO

import pytest
from PIL import Image, ImageDraw, ImageFont

from abstractcore.media.types import ContentFormat, MediaContent, MediaType
from abstractcore.providers.anthropic_provider import AnthropicProvider


def _make_test_image_bytes(text: str) -> bytes:
    img = Image.new("RGB", (520, 220), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 64)
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (img.size[0] - w) // 2
    y = (img.size[1] - h) // 2
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


@pytest.mark.e2e
def test_e2e_anthropic_vision_prompt_empty_injects_media() -> None:
    """Level C: real Anthropic call; verifies media is sent even when prompt=''."""
    if os.environ.get("ABSTRACT_E2E_ANTHROPIC") != "1":
        pytest.skip("Set ABSTRACT_E2E_ANTHROPIC=1 to run this test.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not configured.")

    model = os.environ.get("ANTHROPIC_E2E_MODEL", "claude-haiku-4-5-20251001")
    provider = AnthropicProvider(model=model, api_key=os.environ.get("ANTHROPIC_API_KEY"))

    expected = "K9X3Q7T1"
    img_bytes = _make_test_image_bytes(expected)
    media = [
        MediaContent(
            media_type=MediaType.IMAGE,
            content=img_bytes,
            content_format=ContentFormat.BASE64,
            mime_type="image/png",
            file_path="e2e.png",
        )
    ]

    prompt = (
        "Read the exact text shown in the image and output it verbatim.\n"
        "Output only that text. If you cannot see any image, output NOIMAGE."
    )

    resp = provider.generate(
        prompt="",
        messages=[{"role": "user", "content": prompt}],
        media=media,
        temperature=0.0,
        max_output_tokens=24,
    )

    out = (resp.content or "").strip()
    assert "NOIMAGE" not in out, f"Model reported missing image: {out!r}"

    def norm(s: str) -> str:
        return "".join(ch for ch in s.upper() if ch.isalnum())

    def edit_distance(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        dp = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            prev = dp[0]
            dp[0] = i
            for j, cb in enumerate(b, start=1):
                cur = dp[j]
                cost = 0 if ca == cb else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = cur
        return dp[-1]

    out_token = norm(out.splitlines()[0] if out else "")
    dist = edit_distance(out_token, norm(expected))
    assert dist <= 2, f"Expected output close to {expected!r} (edit distance <=2), got: {out!r} (norm={out_token!r}, dist={dist})"
