"""``POST /acore/session/claim``: trade a one-time claim code for the bearer token.

Only for a server started by ``abstractcore serve`` on a loopback address with
the generated token (see ``first_run.py``). The request must come straight
from a loopback peer: a request that carries any forwarding header
(``X-Forwarded-For``, ``Forwarded``, ``X-Real-IP``, ...) is refused, whatever
the header says, and so is a ``Host`` that is not a loopback name (a
DNS-rebinding page). The path is auth-exempt in ``app.py``.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from . import first_run

router = APIRouter(tags=["session"])


class ClaimBody(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"code": "rOjpWV8YAnaS6MJKoCTku78_yxqEwWJE"}]})

    code: str = Field("", description="The one-time code from the `/console#claim=<code>` link.")

CLAIM_PATH = "/acore/session/claim"
_FORWARDING_HEADERS = (
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-real-ip",
    "forwarded",
    "true-client-ip",
    "cf-connecting-ip",
)
_NO_STORE = {"Cache-Control": "no-store"}


def _refuse(status_code: int, reason: str, message: str) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"ok": False, "reason": reason, "message": message, "error": {"message": message, "type": f"claim_{reason}"}},
        headers=_NO_STORE,
    )


def _host_header_name(value: str) -> str:
    raw = str(value or "").strip()
    if raw.startswith("["):
        return raw[1 : raw.find("]")] if "]" in raw else raw
    return raw.rsplit(":", 1)[0] if raw.count(":") == 1 else raw


def peer_is_direct_loopback(request: Request) -> bool:
    """The socket peer is loopback AND nothing claims the request was forwarded."""

    if any(request.headers.get(h) for h in _FORWARDING_HEADERS):
        return False
    client = request.client
    if client is None or not first_run.is_loopback_host(client.host):
        return False
    return first_run.is_loopback_host(_host_header_name(request.headers.get("host", "")))


@router.post(CLAIM_PATH, summary="Redeem a one-time console claim code (loopback only)")
async def claim_session(request: Request, body: ClaimBody) -> JSONResponse:
    if not first_run.claims_enabled():
        return _refuse(
            404,
            "not_enabled",
            "Claim codes are only issued by a loopback `abstractcore serve` using its generated token.",
        )
    if not peer_is_direct_loopback(request):
        return _refuse(403, "not_loopback", "Claim codes can only be redeemed from this machine (a direct loopback connection).")
    result = await run_in_threadpool(first_run.redeem_claim, body.code)
    if not result.ok:
        return _refuse(
            403,
            result.reason,
            "This console link was already used, has expired or is unknown. "
            "Run `abstractcore serve --claim-url` for a fresh one.",
        )
    return JSONResponse(
        {"ok": True, "token_type": "bearer", "token": first_run.env_token()},
        headers=_NO_STORE,
    )
