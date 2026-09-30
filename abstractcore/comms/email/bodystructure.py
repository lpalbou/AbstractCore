"""IMAP BODYSTRUCTURE (RFC 3501 section 7.4.2): the MIME tree of a message without its content.

Reading a message fetches its structure first, then only the parts it needs by section
(`BODY.PEEK[1.2]`): the text/plain and text/html bodies for `read_email`, one part for an
attachment download. Attachments are listed from the structure (name, type, size) without
downloading them.

The parser follows the RFC grammar (parenthesised lists, strings, NIL); it never looks for
words in free text. Section numbers follow RFC 3501 section 6.4.5: the parts of a multipart
body are numbered 1, 2, ... (nested: 2.1, 2.2); a message that is not multipart has its body at
section 1; the parts inside an attached message (message/rfc822 at section 2) are 2.1, 2.2, ...
(its body is 2.1 when it is not multipart).
"""

from __future__ import annotations

import base64
import binascii
import email.header
import email.utils
import quopri
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple


@dataclass(frozen=True)
class MimePart:
    """One node of the structure. `size` is the part's size on the wire (encoded octets)."""

    section: str
    content_type: str
    params: Dict[str, str] = field(default_factory=dict)
    content_id: str = ""
    encoding: str = ""
    size: int = 0
    disposition: str = ""
    disposition_params: Dict[str, str] = field(default_factory=dict)
    children: Tuple["MimePart", ...] = ()
    # message/rfc822 only: the body of the attached message.
    message_body: Optional["MimePart"] = None

    @property
    def multipart(self) -> bool:
        return self.content_type.startswith("multipart/")

    @property
    def charset(self) -> str:
        return self.params.get("charset", "")

    @property
    def filename(self) -> str:
        return self.disposition_params.get("filename") or self.params.get("name") or ""

    @property
    def is_attachment(self) -> bool:
        """Same rule as reading the whole message did: disposition `attachment`, or a file name."""

        if self.multipart:
            return False
        return self.disposition == "attachment" or bool(self.filename)


def _text(tok: Any) -> str:
    if tok is None:
        return ""
    if isinstance(tok, (bytes, bytearray)):
        return bytes(tok).decode("utf-8", errors="replace")
    return str(tok)


def _decode_words(value: str) -> str:
    """RFC 2047 encoded words (`=?utf-8?B?...?=`) inside a parameter value, if any."""

    if "=?" not in value:
        return value
    try:
        return str(email.header.make_header(email.header.decode_header(value)))
    except (ValueError, LookupError, UnicodeDecodeError, binascii.Error):
        return value


def _params(tok: Any) -> Dict[str, str]:
    """A parameter list `("NAME" "value" ...)` -> {name: decoded value}.

    RFC 2231 forms (`filename*`, continuations `filename*0*`) are decoded with the standard
    library; RFC 2047 encoded words are decoded too.
    """

    if not isinstance(tok, list):
        return {}
    pairs: List[Tuple[str, str]] = []
    for i in range(0, len(tok) - 1, 2):
        key = _text(tok[i]).strip().lower()
        if key:
            value = _text(tok[i + 1])
            if not key.endswith("*"):
                # decode_params unquotes; keep the value as the server sent it.
                value = '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
            pairs.append((key, value))
    if not pairs:
        return {}
    try:
        decoded = email.utils.decode_params([("", "")] + pairs)[1:]
    except Exception:  # noqa: BLE001 - a malformed parameter list reads as "no parameters"
        decoded = [(k.rstrip("*"), v.strip('"')) for k, v in pairs]
    out: Dict[str, str] = {}
    for key, value in decoded:
        if isinstance(value, tuple):
            charset, language, encoded = value
            text = email.utils.collapse_rfc2231_value((charset, language, email.utils.unquote(encoded)))
        else:
            text = _decode_words(email.utils.unquote(str(value)))
        out[str(key).lower()] = str(text)
    return out


def _disposition(tok: Any) -> Tuple[str, Dict[str, str]]:
    if isinstance(tok, list) and tok:
        return _text(tok[0]).strip().lower(), _params(tok[1] if len(tok) > 1 else None)
    return "", {}


def _int(tok: Any) -> int:
    try:
        return int(_text(tok).strip())
    except ValueError:
        return 0


def _part_at(tok: Any, section: str) -> MimePart:
    """The part described by `tok`, located at `section`."""

    if not isinstance(tok, list) or not tok:
        raise ValueError("BODYSTRUCTURE part is not a list")
    if isinstance(tok[0], list):
        # multipart: (part)(part)... "SUBTYPE" [params [disposition [language [location]]]]
        children: List[MimePart] = []
        i = 0
        while i < len(tok) and isinstance(tok[i], list):
            children.append(_part_at(tok[i], f"{section}.{len(children) + 1}" if section else str(len(children) + 1)))
            i += 1
        subtype = _text(tok[i]).lower() if i < len(tok) else "mixed"
        params = _params(tok[i + 1]) if i + 1 < len(tok) else {}
        disp, dparams = _disposition(tok[i + 2]) if i + 2 < len(tok) else ("", {})
        return MimePart(
            section=section,
            content_type=f"multipart/{subtype}",
            params=params,
            disposition=disp,
            disposition_params=dparams,
            children=tuple(children),
        )
    # single part: type subtype params id description encoding size ...
    if len(tok) < 7:
        raise ValueError("BODYSTRUCTURE part is too short")
    ctype = f"{_text(tok[0]).lower()}/{_text(tok[1]).lower()}"
    rest = list(tok[7:])
    message_body: Optional[MimePart] = None
    if ctype == "message/rfc822" and len(rest) >= 3:
        # envelope, body, lines
        message_body = _message_body(rest[1], section)
        rest = rest[3:]
    elif ctype.startswith("text/") and rest:
        rest = rest[1:]  # lines
    # extension data: md5 disposition language location
    disp, dparams = _disposition(rest[1]) if len(rest) > 1 else ("", {})
    return MimePart(
        section=section,
        content_type=ctype,
        params=_params(tok[2]),
        content_id=_text(tok[3]).strip(),
        encoding=_text(tok[5]).strip().lower(),
        size=_int(tok[6]),
        disposition=disp,
        disposition_params=dparams,
        message_body=message_body,
    )


def _message_body(tok: Any, prefix: str) -> MimePart:
    """The body of a message (the top level, or an attached message at section `prefix`)."""

    if isinstance(tok, list) and tok and isinstance(tok[0], list):
        return _part_at(tok, prefix)
    return _part_at(tok, f"{prefix}.1" if prefix else "1")


def parse_bodystructure(tok: Any) -> MimePart:
    """The root of a message's structure from the parsed `BODYSTRUCTURE` value."""

    return _message_body(tok, "")


def walk(part: MimePart) -> Iterator[MimePart]:
    """Depth-first, in document order. An attached message (message/rfc822 with disposition
    `attachment`) is one attachment; an inline one is walked into."""

    yield part
    for child in part.children:
        yield from walk(child)
    if part.message_body is not None and not part.is_attachment:
        yield from walk(part.message_body)


def body_parts(root: MimePart) -> List[MimePart]:
    """The text/plain and text/html parts that are not attachments, in document order."""

    return [
        p
        for p in walk(root)
        if not p.multipart
        and p.message_body is None
        and not p.is_attachment
        and p.content_type in ("text/plain", "text/html")
    ]


def attachment_parts(root: MimePart) -> List[MimePart]:
    return [p for p in walk(root) if p.is_attachment]


def decode_transfer(data: bytes, encoding: str) -> bytes:
    """Undo the Content-Transfer-Encoding of a part's wire bytes."""

    enc = (encoding or "").strip().lower()
    if enc == "base64":
        try:
            return base64.b64decode(data, validate=False)
        except (binascii.Error, ValueError):
            return base64.b64decode(data + b"=" * (-len(data) % 4), validate=False)
    if enc == "quoted-printable":
        return quopri.decodestring(data)
    return bytes(data)


def decode_text(data: bytes, part: MimePart) -> str:
    raw = decode_transfer(data, part.encoding)
    charset = part.charset or "utf-8"
    try:
        return raw.decode(charset, errors="replace")
    except LookupError:
        return raw.decode("utf-8", errors="replace")


__all__ = [
    "MimePart",
    "attachment_parts",
    "body_parts",
    "decode_text",
    "decode_transfer",
    "parse_bodystructure",
    "walk",
]
