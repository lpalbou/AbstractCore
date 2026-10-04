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
from importlib import metadata
from pathlib import Path
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
    # A path next to the package, like the other asset readers: on Python 3.9
    # importlib.resources.files() fails on abstractcore.assets (no __init__.py).
    text = (Path(__file__).resolve().parents[1] / "assets" / _ASSET).read_text(encoding="utf-8")
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
    "about_card_html",
    "about_fields",
    "about_html",
    "about_lines",
    "about_links",
    "about_version_facts",
    "app_identity",
    "framework_identity",
    "gateway_version_rows",
    "installed_version",
    "known_app_ids",
]


# --- Compact About card (ui-kit 0.7.0 `AfAbout` twin) -------------------------
#
# Content rule (operator, round 5): the app's name and version, the framework
# version, the gateway version, the links (website, source, docs, issues,
# feedback, contact) and ONE author/licence line. Never a package list.


def about_version_facts(
    framework: Optional[str],
    gateway: Optional[str],
    framework_note: str = "",
    gateway_note: str = "",
) -> List[Tuple[str, str]]:
    """The two version facts of an About card, as the kit's `aboutVersionFacts`."""
    fw = framework.strip() if isinstance(framework, str) else ""
    gw = gateway.strip() if isinstance(gateway, str) else ""
    return [
        ("AbstractFramework", fw or (framework_note.strip() or "not reported")),
        ("AbstractGateway", gw or (gateway_note.strip() or "not connected")),
    ]


def about_links(identity: AppIdentity) -> List[Tuple[str, str, str]]:
    """``(id, label, href)`` of an About card's links, as the kit's `aboutLinks`."""
    fw = framework_identity()
    return [
        ("website", "Website", identity.website),
        ("source", "Source", identity.repo),
        ("docs", "Docs", identity.docs),
        ("issues", "Issues", identity.issues),
        ("feedback", "Feedback", identity.feedback),
        ("contact", "Contact", f"mailto:{fw.contact_email}"),
    ]


def about_card_html(
    identity: AppIdentity,
    framework: Optional[str],
    gateway: Optional[str],
    *,
    framework_note: str = "",
    gateway_note: str = "",
    title_id: str = "",
    action_html: str = "",
) -> str:
    """The compact About card markup, identical in structure and class names
    to the kit's `AfAbout` (styled by the kit's vendored ``af-about`` CSS).
    ``action_html`` (trusted markup, e.g. a Close button) ends the heading row."""
    fw = framework_identity()
    tid = f' id="{_escape(title_id)}"' if title_id else ""
    facts = "".join(
        f'<div class="af-about-card__fact"><dt>{_escape(label)}</dt><dd>{_escape(text)}</dd></div>'
        for label, text in about_version_facts(framework, gateway, framework_note, gateway_note)
    )
    links = []
    for link_id, label, href in about_links(identity):
        title = href[len("mailto:"):] if href.startswith("mailto:") else href
        extra = "" if href.startswith("mailto:") else ' target="_blank" rel="noopener noreferrer"'
        links.append(
            f'<a class="af-about-card__link" data-link="{link_id}" href="{_escape(href)}" title="{_escape(title)}"{extra}>{_escape(label)}</a>'
        )
    return (
        f'<div class="af-about-card" data-app="{_escape(identity.id)}">'
        f'<div class="af-about-card__head"><h2 class="af-about-card__name"{tid}>{_escape(identity.name)} '
        f'<span class="af-about-card__version">{_escape(identity.version)}</span></h2>{action_html}</div>'
        f'<dl class="af-about-card__versions">{facts}</dl>'
        f'<nav class="af-about-card__links" aria-label="{_escape(identity.name)} links">{"".join(links)}</nav>'
        f'<p class="af-about-card__legal">{_escape(fw.copyright)}</p>'
        "</div>"
    )
