"""IMAP wire helpers: quoting, modified UTF-7 mailbox names, and a response tokenizer.

These parse the IMAP4rev1 grammar (RFC 3501): quoted strings, atoms, parenthesised lists,
literals and NIL. They never look for words in free text.
"""

from __future__ import annotations

import base64
from typing import Any, List, Optional, Sequence, Tuple, Union

Token = Union[str, bytes, list, None]


# ---------------------------------------------------------------------------------------
# Quoting
# ---------------------------------------------------------------------------------------


def quote(value: str) -> str:
    """An IMAP quoted string. Refuses CR/LF/NUL (they cannot be quoted)."""

    text = str(value)
    if any(c in text for c in "\r\n\x00"):
        raise ValueError("IMAP strings cannot contain line breaks")
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


# ---------------------------------------------------------------------------------------
# Modified UTF-7 (RFC 3501 section 5.1.3)
# ---------------------------------------------------------------------------------------


def _b64_modified_encode(chunk: str) -> str:
    raw = chunk.encode("utf-16-be")
    return base64.b64encode(raw).decode("ascii").rstrip("=").replace("/", ",")


def encode_mailbox(name: str) -> str:
    out: List[str] = []
    pending: List[str] = []

    def flush() -> None:
        if pending:
            out.append("&" + _b64_modified_encode("".join(pending)) + "-")
            pending.clear()

    for ch in str(name):
        code = ord(ch)
        if 0x20 <= code <= 0x7E:
            flush()
            out.append("&-" if ch == "&" else ch)
        else:
            pending.append(ch)
    flush()
    return "".join(out)


def decode_mailbox(name: Union[str, bytes]) -> str:
    text = name.decode("ascii", errors="replace") if isinstance(name, (bytes, bytearray)) else str(name)
    out: List[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch != "&":
            out.append(ch)
            i += 1
            continue
        end = text.find("-", i + 1)
        if end == -1:
            out.append(text[i:])
            break
        segment = text[i + 1 : end]
        if not segment:
            out.append("&")
        else:
            b64 = segment.replace(",", "/")
            b64 += "=" * (-len(b64) % 4)
            try:
                out.append(base64.b64decode(b64).decode("utf-16-be"))
            except Exception:
                out.append(text[i : end + 1])
        i = end + 1
    return "".join(out)


# ---------------------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------------------


class _Reader:
    """Reads tokens from segments: text with `{n}` literal markers, each followed by its bytes."""

    def __init__(self, segments: Sequence[Tuple[bytes, Optional[bytes]]]) -> None:
        self.segments = list(segments)
        self.seg = 0
        self.pos = 0

    def _text(self) -> bytes:
        return self.segments[self.seg][0] if self.seg < len(self.segments) else b""

    def peek(self) -> Optional[int]:
        while self.seg < len(self.segments):
            text = self._text()
            if self.pos < len(text):
                return text[self.pos]
            # end of this text; a pending literal is consumed by read_literal only
            if self.segments[self.seg][1] is not None:
                return None
            self.seg += 1
            self.pos = 0
        return None

    def skip_spaces(self) -> None:
        while True:
            c = self.peek()
            if c is None or c not in (0x20, 0x09, 0x0D, 0x0A):
                return
            self.pos += 1

    def at_literal_end(self) -> bool:
        return self.seg < len(self.segments) and self.pos >= len(self._text()) and self.segments[self.seg][1] is not None

    def take_literal(self) -> bytes:
        data = self.segments[self.seg][1] or b""
        self.seg += 1
        self.pos = 0
        return data

    def token(self) -> Token:
        self.skip_spaces()
        if self.at_literal_end():
            return self.take_literal()
        c = self.peek()
        if c is None:
            raise ValueError("unexpected end of IMAP data")
        text = self._text()
        if c == 0x28:  # (
            self.pos += 1
            items: list = []
            while True:
                self.skip_spaces()
                if not self.at_literal_end() and self.peek() == 0x29:  # )
                    self.pos += 1
                    return items
                items.append(self.token())
        if c == 0x22:  # "
            self.pos += 1
            buf = bytearray()
            while self.pos < len(text):
                b = text[self.pos]
                if b == 0x5C and self.pos + 1 < len(text):  # backslash escape
                    buf.append(text[self.pos + 1])
                    self.pos += 2
                    continue
                if b == 0x22:
                    self.pos += 1
                    return bytes(buf).decode("utf-8", errors="replace")
                buf.append(b)
                self.pos += 1
            raise ValueError("unterminated IMAP quoted string")
        if c == 0x7B:  # {n} literal marker at the end of this text segment
            end = text.find(b"}", self.pos)
            if end == -1:
                raise ValueError("malformed IMAP literal")
            self.pos = end + 1
            self.skip_spaces()
            if self.at_literal_end():
                return self.take_literal()
            raise ValueError("IMAP literal marker without data")
        start = self.pos
        depth = 0
        while self.pos < len(text):
            b = text[self.pos]
            if b == 0x5B:  # [ inside an atom like BODY[HEADER.FIELDS (FROM)]
                depth += 1
            elif b == 0x5D:
                depth -= 1
            elif depth <= 0 and b in (0x20, 0x28, 0x29, 0x0D, 0x0A):
                break
            self.pos += 1
        atom = text[start : self.pos].decode("utf-8", errors="replace")
        if atom.upper() == "NIL":
            return None
        return atom

    def done(self) -> bool:
        self.skip_spaces()
        return self.peek() is None and not self.at_literal_end()


def segments_from_response(data: Sequence[Any]) -> List[List[Tuple[bytes, Optional[bytes]]]]:
    """Group an imaplib response list into one segment list per response line.

    imaplib returns plain `bytes` for a line without literals and `(text, literal)` tuples
    for each literal, followed by a `bytes` continuation of the same line.
    """

    groups: List[List[Tuple[bytes, Optional[bytes]]]] = []
    current: Optional[List[Tuple[bytes, Optional[bytes]]]] = None
    for item in data or []:
        if isinstance(item, tuple) and len(item) >= 2:
            if current is None:
                current = []
                groups.append(current)
            current.append((bytes(item[0] or b""), bytes(item[1] or b"")))
        elif isinstance(item, (bytes, bytearray)):
            if current is not None:
                current.append((bytes(item), None))
                current = None
            else:
                groups.append([(bytes(item), None)])
        elif item is None:
            continue
    return groups


def parse_segments(segments: Sequence[Tuple[bytes, Optional[bytes]]]) -> List[Token]:
    reader = _Reader(segments)
    out: List[Token] = []
    while not reader.done():
        out.append(reader.token())
    return out


def parse_fetch_group(segments: Sequence[Tuple[bytes, Optional[bytes]]]) -> dict:
    """`<seq> (NAME value NAME value ...)` -> {NAME.upper(): value}."""

    tokens = parse_segments(segments)
    attrs = None
    for tok in tokens:
        if isinstance(tok, list):
            attrs = tok
            break
    out: dict = {}
    if not attrs:
        return out
    i = 0
    while i + 1 < len(attrs):
        name = attrs[i]
        value = attrs[i + 1]
        if isinstance(name, str):
            out[name.upper()] = value
        i += 2
    return out


def parse_list_line(segments: Sequence[Tuple[bytes, Optional[bytes]]]) -> Optional[dict]:
    """One LIST response: `(flags) "delim" name` -> {flags, delimiter, name}."""

    tokens = parse_segments(segments)
    if len(tokens) < 3 or not isinstance(tokens[0], list):
        return None
    flags = [str(f) for f in tokens[0] if f is not None]
    delim = tokens[1]
    name = tokens[2]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    if name is None:
        return None
    return {
        "flags": flags,
        "delimiter": "" if delim is None else str(delim),
        "name": decode_mailbox(str(name)),
        "raw_name": str(name),
    }
