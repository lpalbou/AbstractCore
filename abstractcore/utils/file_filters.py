"""Shared file-family and extension filtering helpers.

These helpers keep Gateway browse/list routes and Runtime workspace nodes on
the same filtering contract without introducing a second extension taxonomy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from abstractcore.media.types import MediaType, detect_media_type, get_supported_extensions_by_type

_ARCHIVE_EXTENSIONS = {
    "7z",
    "bz2",
    "gz",
    "rar",
    "tar",
    "tgz",
    "xz",
    "zip",
}

_CODE_EXTENSIONS = {
    "bash",
    "c",
    "cc",
    "clj",
    "cljs",
    "cmake",
    "conf",
    "containerfile",
    "cpp",
    "cs",
    "css",
    "cxx",
    "dart",
    "dockerfile",
    "env",
    "erl",
    "ex",
    "exs",
    "fish",
    "go",
    "gradle",
    "h",
    "hpp",
    "hrl",
    "html",
    "htm",
    "hxx",
    "ini",
    "java",
    "jl",
    "js",
    "json",
    "jsonl",
    "jsx",
    "kt",
    "less",
    "lua",
    "m",
    "make",
    "md",
    "php",
    "pl",
    "pm",
    "properties",
    "py",
    "pyw",
    "pyx",
    "r",
    "rb",
    "rs",
    "sass",
    "scala",
    "scss",
    "sh",
    "sql",
    "swift",
    "toml",
    "ts",
    "tsx",
    "vue",
    "xml",
    "yaml",
    "yml",
    "zsh",
}

_JSON_EXTENSIONS = {"json", "jsonl", "ndjson"}

_MEDIA_TYPE_BY_FAMILY = {
    "image": MediaType.IMAGE,
    "video": MediaType.VIDEO,
    "audio": MediaType.AUDIO,
    "document": MediaType.DOCUMENT,
    "text": MediaType.TEXT,
}


def normalize_extensions(value: Optional[Iterable[str] | str]) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        items = value.replace("\n", ",").split(",")
    else:
        items = list(value)
    out: set[str] = set()
    for item in items:
        text = str(item or "").strip().lower()
        if not text:
            continue
        if text.startswith("."):
            text = text[1:]
        if text:
            out.add(text)
    return out


def extensions_for_family(family: str) -> set[str]:
    name = str(family or "").strip().lower()
    if not name or name == "any":
        return set()
    if name == "archive":
        return set(_ARCHIVE_EXTENSIONS)
    if name == "code":
        return set(_CODE_EXTENSIONS)
    if name == "json":
        return set(_JSON_EXTENSIONS)
    media_type = _MEDIA_TYPE_BY_FAMILY.get(name)
    if media_type is None:
        return set()
    return set(get_supported_extensions_by_type(media_type))


def guess_file_family(path: str | Path) -> str:
    candidate = Path(path)
    suffix = candidate.suffix.lower().lstrip(".")
    if suffix in _ARCHIVE_EXTENSIONS:
        return "archive"
    if suffix in _JSON_EXTENSIONS:
        return "json"
    if suffix in _CODE_EXTENSIONS:
        return "code"
    try:
        media_type = detect_media_type(candidate)
    except Exception:
        media_type = None
    if media_type == MediaType.IMAGE:
        return "image"
    if media_type == MediaType.VIDEO:
        return "video"
    if media_type == MediaType.AUDIO:
        return "audio"
    if media_type == MediaType.DOCUMENT:
        return "document"
    if media_type == MediaType.TEXT:
        return "text"
    return "other"


def file_matches_filters(
    path: str | Path,
    *,
    family: str = "any",
    extensions: Optional[Iterable[str] | str] = None,
) -> bool:
    suffix = Path(path).suffix.lower().lstrip(".")
    family_name = str(family or "").strip().lower() or "any"
    ext_filter = normalize_extensions(extensions)
    if ext_filter and suffix not in ext_filter:
        return False
    if family_name in {"", "any"}:
        return True
    if family_name == "other":
        return guess_file_family(path) == "other"
    if family_name == "archive":
        return suffix in _ARCHIVE_EXTENSIONS
    if family_name == "json":
        return suffix in _JSON_EXTENSIONS
    if family_name == "code":
        return suffix in _CODE_EXTENSIONS
    return guess_file_family(path) == family_name


__all__ = [
    "extensions_for_family",
    "file_matches_filters",
    "guess_file_family",
    "normalize_extensions",
]
