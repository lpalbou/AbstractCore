"""Framework and application identity for "About" screens.

One canonical descriptor (``identity/abstractframework.json`` in the
AbstractFramework repository) is vendored here as package data so that an
installed application can show, without any network access, what it is part
of, who wrote it, its licence, and where its source, documentation, issue
tracker and feedback channel live.

Every AbstractFramework application renders the same facts:

* the application name and version;
* the framework name and website;
* the author and copyright line;
* links to the source repository, the documentation, "report an issue" and
  "give feedback".

Use :func:`app_identity` to look an application up by its distribution name
and :func:`about_lines` / :func:`about_html` to render it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata, resources
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

_ASSET = "abstractframework_identity.json"


@dataclass(frozen=True)
class FrameworkIdentity:
    name: str
    website: str
    github_org: str
    author: str
    years: str
    license: str
    copyright: str
    contact_email: str


@dataclass(frozen=True)
class AppIdentity:
    id: str
    name: str
    version: str
    website: str
    repo: str
    docs: str
    issues: str
    feedback: str

    @property
    def framework(self) -> FrameworkIdentity:
        return framework_identity()


@lru_cache(maxsize=1)
def _descriptor() -> Dict[str, object]:
    text = resources.files("abstractcore.assets").joinpath(_ASSET).read_text(encoding="utf-8")
    return json.loads(text)


@lru_cache(maxsize=1)
def framework_identity() -> FrameworkIdentity:
    raw = dict(_descriptor()["framework"])  # type: ignore[arg-type]
    return FrameworkIdentity(**raw)


def known_app_ids() -> Tuple[str, ...]:
    return tuple(sorted(_descriptor()["apps"].keys()))  # type: ignore[union-attr]


def installed_version(distribution: str) -> str:
    """The installed version of a distribution, or ``"unknown"``."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def app_identity(app_id: str, version: Optional[str] = None) -> AppIdentity:
    """Identity of one application.

    ``app_id`` is the distribution name in lower case (``"abstractassistant"``).
    ``version`` defaults to the installed distribution version.
    Raises ``KeyError`` for an application the descriptor does not know:
    a caller must not invent identity facts.
    """
    apps: Mapping[str, Mapping[str, str]] = _descriptor()["apps"]  # type: ignore[assignment]
    raw = apps[app_id]
    return AppIdentity(
        id=app_id,
        name=raw["name"],
        version=version if version is not None else installed_version(app_id),
        website=raw["website"],
        repo=raw["repo"],
        docs=raw["docs"],
        issues=raw["issues"],
        feedback=raw["feedback"],
    )


def about_fields(identity: AppIdentity, extra: Optional[Mapping[str, str]] = None) -> List[Tuple[str, str]]:
    """Ordered (label, value) pairs every About screen shows.

    ``extra`` appends application-specific rows (for example the versions of
    the gateway packages the application talks to).
    """
    fw = identity.framework
    rows: List[Tuple[str, str]] = [
        ("Application", f"{identity.name} {identity.version}"),
        ("Part of", f"{fw.name} — {fw.website}"),
        ("Author", f"{fw.author} ({fw.years})"),
        ("Copyright", fw.copyright),
        ("Website", identity.website),
        ("Source", identity.repo),
        ("Documentation", identity.docs),
        ("Report an issue", identity.issues),
        ("Give feedback", identity.feedback),
        ("Contact", fw.contact_email),
    ]
    for label, value in (extra or {}).items():
        rows.append((str(label), str(value)))
    return rows


def about_lines(identity: AppIdentity, extra: Optional[Mapping[str, str]] = None) -> List[str]:
    """Plain-text About lines, one ``label: value`` per row."""
    return [f"{label}: {value}" for label, value in about_fields(identity, extra)]


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def about_html(identity: AppIdentity, extra: Optional[Mapping[str, str]] = None) -> str:
    """The same rows as HTML, with URLs rendered as links (Qt rich text safe)."""
    parts: List[str] = []
    for label, value in about_fields(identity, extra):
        if value.startswith("http://") or value.startswith("https://"):
            rendered = f'<a href="{_escape(value)}">{_escape(value)}</a>'
        elif "@" in value and " " not in value:
            rendered = f'<a href="mailto:{_escape(value)}">{_escape(value)}</a>'
        else:
            rendered = _escape(value)
        parts.append(f"<b>{_escape(label)}:</b> {rendered}")
    return "<br>".join(parts)


__all__ = [
    "AppIdentity",
    "FrameworkIdentity",
    "about_fields",
    "about_html",
    "about_lines",
    "app_identity",
    "framework_identity",
    "installed_version",
    "known_app_ids",
]
