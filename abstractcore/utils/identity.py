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
import re
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
    """The installed version of a distribution.

    Raises ``importlib.metadata.PackageNotFoundError`` when the distribution is
    not installed: an About screen never shows a made-up version.
    """
    return metadata.version(distribution)


def app_identity(app_id: str, version: Optional[str] = None) -> AppIdentity:
    """Identity of one application.

    ``app_id`` is the distribution name in lower case (``"abstractassistant"``).
    ``version`` defaults to the installed distribution version and raises
    ``importlib.metadata.PackageNotFoundError`` when there is none: a caller
    passes the version it knows or gets an error, never a silent "unknown".
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


GatewayAboutPayload = Mapping[str, object]


def gateway_version_rows(payload: Optional[GatewayAboutPayload], error: Optional[str] = None) -> List[Tuple[str, str]]:
    """Rows describing the gateway an application talks to.

    Mirrors ``gatewayVersionRows`` in ``@abstractframework/ui-kit`` so every
    About screen prints the same lines from ``GET /api/gateway/about``
    (``{abstractframework, abstractgateway, packages}``):

    * ``Gateway`` → ``AbstractGateway <version>``
    * ``Gateway framework`` → ``AbstractFramework <version>`` or
      ``not installed on the gateway host``
    * ``Gateway package <name>`` → ``<version>`` for every other package, sorted

    On an error, or a payload without a gateway version, exactly one row:
    ``Gateway`` → ``unavailable (<reason>)``.
    """
    if error is not None:
        return [("Gateway", f"unavailable ({error.strip() or 'unknown error'})")]
    gateway = _version_text((payload or {}).get("abstractgateway"))
    if not gateway:
        return [("Gateway", "unavailable (the gateway did not report its version)")]
    rows: List[Tuple[str, str]] = [("Gateway", f"AbstractGateway {gateway}")]
    framework_text = _version_text((payload or {}).get("abstractframework"))
    rows.append(("Gateway framework", f"AbstractFramework {framework_text}" if framework_text else "not installed on the gateway host"))
    packages = (payload or {}).get("packages")
    if isinstance(packages, Mapping):
        for name in sorted(packages):
            if name in ("abstractgateway", "abstractframework"):
                continue
            text = _version_text(packages[name])
            if text:
                rows.append((f"Gateway package {name}", text))
    return rows


def _version_text(value: object) -> str:
    """Only a non-empty string is a version; numbers, booleans and None are 'not reported'."""
    return value.strip() if isinstance(value, str) else ""


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


_URL_RE = re.compile(r"https?://[^\s<>\"]+")


def _render_value(label: str, value: str) -> str:
    """Escape a value; URLs inside it become links; only the Contact row is a mailto."""
    if label == "Contact":
        return f'<a href="mailto:{_escape(value)}">{_escape(value)}</a>'
    out: List[str] = []
    last = 0
    for match in _URL_RE.finditer(value):
        out.append(_escape(value[last:match.start()]))
        url = match.group(0)
        out.append(f'<a href="{_escape(url)}">{_escape(url)}</a>')
        last = match.end()
    out.append(_escape(value[last:]))
    return "".join(out)


def about_html(identity: AppIdentity, extra: Optional[Mapping[str, str]] = None) -> str:
    """The same rows as HTML, with URLs rendered as links (Qt rich text safe)."""
    parts: List[str] = []
    for label, value in about_fields(identity, extra):
        parts.append(f"<b>{_escape(label)}:</b> {_render_value(label, value)}")
    return "<br>".join(parts)


__all__ = [
    "AppIdentity",
    "FrameworkIdentity",
    "about_fields",
    "about_html",
    "about_lines",
    "app_identity",
    "framework_identity",
    "gateway_version_rows",
    "installed_version",
    "known_app_ids",
]
