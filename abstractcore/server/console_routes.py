"""Web console routes for ``abstractcore serve``.

- ``GET /console``: the standalone console page (public HTML; the data it
  loads goes through the normal API auth boundary, and the page prompts for
  the bearer token when the API answers 401).
- ``GET /console/fragment/{kind}``: the embeddable Models/Engines fragment
  (``{"kind", "html", "js", "css"}``) for hosts such as abstractgateway that
  fetch it over HTTP instead of importing ``abstractcore.console.web``.

Both are static code, never data, so the auth middleware exempts them
(``_request_is_auth_exempt`` in ``app.py``).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

from ..console.web import FRAGMENT_KINDS, fragment, render_console_html

router = APIRouter(tags=["console"])

_NO_STORE = {"Cache-Control": "no-store"}


@router.get("/console", include_in_schema=False)
async def console_page() -> HTMLResponse:
    return HTMLResponse(render_console_html(api_base="/acore"), headers=_NO_STORE)


@router.get("/console/fragment/{kind}", summary="Embeddable web console fragment")
async def console_fragment(kind: str) -> JSONResponse:
    """Return the self-contained Models or Engines tab body (html, js, css)."""
    if kind not in FRAGMENT_KINDS:
        raise HTTPException(status_code=404, detail=f"Unknown console fragment {kind!r}; expected one of {list(FRAGMENT_KINDS)}")
    payload = {"kind": kind, **fragment(kind)}
    return JSONResponse(payload, headers=_NO_STORE)
