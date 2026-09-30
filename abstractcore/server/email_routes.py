"""`/acore/email` — the email account of this AbstractCore install, over HTTP.

The Email page of the web console uses these; the terminal console uses the same operations
through `abstractcore email ... --json`. Every route calls `EmailAccountStore` and returns its
`email_settings_v1` document (`public()`), so both consoles render one shape.

    GET    /acore/email                      settings + status (+ notices)
    PUT    /acore/email                      connect: test, then store (password in the body; never echoed)
    POST   /acore/email/test                 sign in to IMAP and SMTP with the stored account
    DELETE /acore/email                      disconnect: delete the credentials and account settings
    PUT    /acore/email/policy               {mode, entries}   recipient policy (replaces the entries)
    POST   /acore/email/policy/check         {addresses}       would they be allowed?
    PUT    /acore/email/limits               {per_hour, per_day}
    PUT    /acore/email/enabled              {enabled}
    PUT    /acore/email/agent-tools          {enabled}         "Agent email tools" (default off)
    PUT    /acore/email/registered-address   {address}
    POST   /acore/email/oauth/start          begin an OAuth2 sign-in (device code or loopback browser flow)
    POST   /acore/email/oauth/finish         {flow_id, wait_s}  complete it (pending until approved)
    POST   /acore/email/oauth/cancel         {flow_id}          drop a pending sign-in (closes its listener)

Every route needs the server principal (the same rule as the host routes): the email account
is the install owner's.

Errors are `{ok: false, error: {code, cause, fix, retryable, details}}` with 400 (invalid
input), 404 (no account), 409 (turned off / credentials missing) or 422 (the mail server
refused the connection test).
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

router = APIRouter(tags=["email"])

_STATUS_BY_CODE = {
    "email_invalid_settings": 400,
    "email_invalid_message": 400,
    "email_policy_refused": 400,
    "email_not_configured": 404,
    "email_disabled": 409,
    "email_secret_unavailable": 409,
    "email_rate_limited": 429,
}


def _principal(request: Request) -> None:
    from .host_routes import _require_host_principal

    _require_host_principal(request)


def _store():
    from ..comms.email import EmailAccountStore

    return EmailAccountStore()


def _error(err: Any) -> JSONResponse:
    status = _STATUS_BY_CODE.get(err.code, 422)
    return JSONResponse(status_code=status, content={"ok": False, "error": {**err.to_dict(include_details=True), "message": err.message}})


async def _call(fn) -> Any:
    from ..comms.email import EmailError

    try:
        return await run_in_threadpool(fn)
    except EmailError as err:
        return _error(err)


class ServerBody(BaseModel):
    host: str = ""
    port: Optional[int] = None
    security: str = "ssl"
    folder: str = "INBOX"
    ca_file: str = ""


class ConnectBody(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "address": "me@example.com",
                    "password": "app-password",
                    "imap": {"host": "imap.example.com", "port": 993, "security": "ssl"},
                    "smtp": {"host": "smtp.example.com", "port": 587, "security": "starttls"},
                }
            ]
        }
    )

    address: str
    display_name: str = ""
    username: str = ""
    password: str = Field("", description="Password or app password; stored encrypted, never returned")
    imap: Optional[ServerBody] = None
    smtp: Optional[ServerBody] = None
    registered_address: Optional[str] = None
    test: bool = True


class PolicyBody(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"mode": "allowlist", "entries": ["me@example.com", "example.org"]}]})

    mode: str = "allowlist"
    entries: List[str] = Field(default_factory=list)


class CheckBody(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"addresses": ["colleague@example.org"]}]})

    addresses: List[str] = Field(default_factory=list)


class LimitsBody(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"per_hour": 20, "per_day": 100}]})

    per_hour: Optional[int] = None
    per_day: Optional[int] = None


class EnabledBody(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"enabled": True}]})

    enabled: bool


class AddressBody(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"address": "me@example.com"}]})

    address: str = ""


class OAuthStartBody(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"address": "me@outlook.com", "provider": "microsoft", "client_id": "00000000-0000-0000-0000-000000000000", "flow": "device"}]}
    )

    address: str
    provider: str = Field(..., description="google | microsoft | custom")
    client_id: str = Field("", description="Your own OAuth client id; empty = the built-in AbstractFramework client when one is registered")
    client_secret: str = Field("", description="Your own OAuth client secret; stored encrypted, never returned")
    tenant: str = ""
    flow: str = Field("", description="device | loopback (default: device for Microsoft, loopback for Google)")
    display_name: str = ""
    imap: Optional[ServerBody] = None
    smtp: Optional[ServerBody] = None
    token_endpoint: str = ""
    authorization_endpoint: str = ""
    device_authorization_endpoint: str = ""
    scopes: List[str] = Field(default_factory=list)
    registered_address: Optional[str] = None
    ca_file: str = Field("", description="PEM file of a private CA, for the sign-in endpoints and for mail servers without their own ca_file")


class OAuthFinishBody(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"flow_id": "flow-id-from-start", "wait_s": 20}]})

    flow_id: str
    wait_s: float = 20.0


def _servers(imap_body: Optional[ServerBody], smtp_body: Optional[ServerBody], preset: Dict[str, Any]):
    from ..comms.email import ImapSettings, SmtpSettings

    imap_raw = imap_body.model_dump() if imap_body and imap_body.host else dict(preset.get("imap") or {})
    smtp_raw = smtp_body.model_dump() if smtp_body and smtp_body.host else dict(preset.get("smtp") or {})
    imap = ImapSettings.build(
        imap_raw["host"], port=imap_raw.get("port"), security=imap_raw.get("security") or "ssl",
        folder=imap_raw.get("folder") or "INBOX", ca_file=imap_raw.get("ca_file") or "",
    ) if imap_raw.get("host") else None
    smtp = SmtpSettings.build(
        smtp_raw["host"], port=smtp_raw.get("port"), security=smtp_raw.get("security") or "ssl",
        ca_file=smtp_raw.get("ca_file") or "",
    ) if smtp_raw.get("host") else None
    return imap, smtp


@router.get("/acore/email", summary="Email account settings and status")
async def email_get(request: Request) -> Any:
    _principal(request)

    def run() -> Dict[str, Any]:
        store = _store()
        notices = store.ensure_legacy_imported()
        return {**store.public(), "notices": notices}

    return await _call(run)


@router.put("/acore/email", summary="Connect the email account (test, then store)")
async def email_put(body: ConnectBody, request: Request) -> Any:
    _principal(request)
    from ..comms.email import EmailAccount, EmailSecret

    def run() -> Dict[str, Any]:
        imap, smtp = _servers(body.imap, body.smtp, {})
        account = EmailAccount.build(
            address=body.address, username=body.username, imap=imap, smtp=smtp, display_name=body.display_name
        )
        return {"ok": True, **_store().connect(account, EmailSecret(body.password), test=body.test, registered_address=body.registered_address)}

    return await _call(run)


@router.post("/acore/email/test", summary="Test the stored email account")
async def email_test(request: Request) -> Any:
    _principal(request)
    return await _call(lambda: _store().test())


@router.delete("/acore/email", summary="Disconnect the email account")
async def email_delete(request: Request) -> Any:
    _principal(request)
    return await _call(lambda: {"ok": True, **_store().disconnect()})


@router.put("/acore/email/policy", summary="Set the recipient policy")
async def email_policy(body: PolicyBody, request: Request) -> Any:
    _principal(request)
    return await _call(lambda: {"ok": True, **_store().set_policy(mode=body.mode, add=body.entries, clear=True)})


@router.post("/acore/email/policy/check", summary="Check recipients against the policy")
async def email_policy_check(body: CheckBody, request: Request) -> Any:
    _principal(request)
    from ..comms.email import EmailInvalidMessage, evaluate, parse_recipients

    def run() -> Dict[str, Any]:
        try:
            addrs = parse_recipients(list(body.addresses))
        except ValueError as exc:
            raise EmailInvalidMessage(f"A recipient is not valid: {exc}.", "Give recipients as name@example.com.") from None
        return evaluate(_store().settings().policy, to=addrs).to_dict()

    return await _call(run)


@router.put("/acore/email/limits", summary="Set the send limits")
async def email_limits(body: LimitsBody, request: Request) -> Any:
    _principal(request)
    return await _call(lambda: {"ok": True, **_store().set_limits(per_hour=body.per_hour, per_day=body.per_day)})


@router.put("/acore/email/enabled", summary="Turn email on or off")
async def email_enabled(body: EnabledBody, request: Request) -> Any:
    _principal(request)
    return await _call(lambda: {"ok": True, **_store().set_enabled(body.enabled)})


@router.put("/acore/email/agent-tools", summary="Turn \"Agent email tools\" on or off")
async def email_agent_tools(body: EnabledBody, request: Request) -> Any:
    """Whether agents may use the email tools with this account (default off). The response's
    `agent_tools` is `{enabled, active, reason}`: `active` also needs a connected, turned-on
    account."""

    _principal(request)
    return await _call(lambda: {"ok": True, **_store().set_agent_tools(body.enabled)})


@router.put("/acore/email/registered-address", summary="Set the registered (own) address")
async def email_registered(body: AddressBody, request: Request) -> Any:
    _principal(request)
    return await _call(lambda: {"ok": True, **_store().set_registered_address(body.address)})


# ---------------------------------------------------------------------------
# OAuth2 sign-in (in-memory pending flows, 15 minutes)
# ---------------------------------------------------------------------------

_FLOWS: Dict[str, Dict[str, Any]] = {}
_FLOWS_LOCK = threading.Lock()
_FLOW_TTL_S = 900.0


def _prune_flows() -> None:
    now = time.time()
    with _FLOWS_LOCK:
        for fid in [k for k, v in _FLOWS.items() if now - v["created"] > _FLOW_TTL_S]:
            flow = _FLOWS.pop(fid)
            loop = flow.get("loopback")
            if loop is not None:
                loop.close()


@router.post("/acore/email/oauth/start", summary="Begin an OAuth2 sign-in")
async def email_oauth_start(body: OAuthStartBody, request: Request) -> Any:
    _principal(request)
    from ..comms.email import (
        EmailAccount,
        EmailInvalidSettings,
        LoopbackAuthorization,
        OAuthSettings,
        OAuthTokenClient,
        provider_preset,
        resolve_oauth_client,
        tls_context,
    )

    def run() -> Dict[str, Any]:
        _prune_flows()
        preset = provider_preset(body.provider, tenant=body.tenant) if body.provider != "custom" else {}
        chosen = resolve_oauth_client(body.provider, body.client_id, body.client_secret)
        oauth = OAuthSettings.build(
            body.provider, chosen["client_id"], client_source=chosen["source"], token_endpoint=body.token_endpoint,
            authorization_endpoint=body.authorization_endpoint,
            device_authorization_endpoint=body.device_authorization_endpoint,
            scopes=body.scopes or None, tenant=body.tenant,
        )
        imap, smtp = _servers(body.imap, body.smtp, preset)
        if body.ca_file:
            imap = replace(imap, ca_file=imap.ca_file or body.ca_file) if imap else None
            smtp = replace(smtp, ca_file=smtp.ca_file or body.ca_file) if smtp else None
        account = EmailAccount.build(
            address=body.address, username=body.address, imap=imap, smtp=smtp,
            display_name=body.display_name, auth_kind="oauth2", oauth=oauth,
        )
        try:
            verify = tls_context(body.ca_file) if body.ca_file else None
        except (OSError, ValueError):
            raise EmailInvalidSettings(
                f"The CA file {body.ca_file!r} could not be loaded.",
                "Give the path of a readable PEM file, or leave it empty to use the system trust store.",
            ) from None
        client = OAuthTokenClient(oauth, client_secret=chosen["client_secret"], verify=verify)
        flow_kind = body.flow or ("device" if oauth.provider == "microsoft" else "loopback")
        fid = secrets.token_urlsafe(16)
        entry: Dict[str, Any] = {
            "created": time.time(), "account": account, "client": client, "kind": flow_kind,
            "client_secret": chosen["client_secret"], "registered_address": body.registered_address,
        }
        if flow_kind == "device":
            device = client.start_device_authorization()
            entry["device"] = device
            out = {"flow_id": fid, "flow": "device", **device.public()}
        else:
            loop = LoopbackAuthorization(client, login_hint=body.address)
            url = loop.start()
            entry["loopback"] = loop
            out = {"flow_id": fid, "flow": "loopback", "authorization_url": url}
        with _FLOWS_LOCK:
            _FLOWS[fid] = entry
        return out

    return await _call(run)


@router.post("/acore/email/oauth/finish", summary="Complete an OAuth2 sign-in")
async def email_oauth_finish(body: OAuthFinishBody, request: Request) -> Any:
    _principal(request)
    from ..comms.email import EmailError, EmailOAuthFailed, EmailOAuthPending, EmailSecret

    def run() -> Dict[str, Any]:
        with _FLOWS_LOCK:
            entry = _FLOWS.get(body.flow_id)
        if entry is None:
            raise EmailOAuthFailed("This sign-in is unknown or expired.", "Start the sign-in again.")
        wait = max(0.0, min(float(body.wait_s), 60.0))
        client = entry["client"]
        try:
            if entry["kind"] == "device":
                device = entry["device"]
                deadline = time.time() + wait
                tokens = None
                while tokens is None:
                    if time.time() >= device.expires_at:
                        raise EmailOAuthFailed("The sign-in code expired before it was approved.", "Start the sign-in again.")
                    try:
                        tokens = client.poll_device_once(device)
                    except EmailOAuthPending:
                        if time.time() + device.interval > deadline:
                            return {"ok": False, "pending": True, "flow_id": body.flow_id}
                        time.sleep(device.interval)
            else:
                loop = entry["loopback"]
                if not loop.wait(wait):
                    return {"ok": False, "pending": True, "flow_id": body.flow_id}
                tokens = loop.finish(timeout_s=0.1)
        except EmailError:
            with _FLOWS_LOCK:
                _FLOWS.pop(body.flow_id, None)
            raise
        with _FLOWS_LOCK:
            _FLOWS.pop(body.flow_id, None)
        secret = EmailSecret(
            refresh_token=tokens.refresh_token, access_token=tokens.access_token,
            expires_at=tokens.expires_at, client_secret=entry["client_secret"],
        )
        return {"ok": True, **_store().connect(entry["account"], secret, test=True, registered_address=entry["registered_address"])}

    return await _call(run)


class OAuthCancelBody(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"flow_id": "flow-id-from-start"}]})

    flow_id: str


@router.post("/acore/email/oauth/cancel", summary="Cancel a pending OAuth2 sign-in")
async def email_oauth_cancel(body: OAuthCancelBody, request: Request) -> Any:
    _principal(request)

    def run() -> Dict[str, Any]:
        with _FLOWS_LOCK:
            entry = _FLOWS.pop(body.flow_id, None)
        if entry is not None and entry.get("loopback") is not None:
            entry["loopback"].close()
        return {"ok": True, "cancelled": entry is not None}

    return await _call(run)
