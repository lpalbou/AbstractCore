"""`/acore/host|engines|models|jobs` -- the models & engines contracts over HTTP.

Thin by design: every handler calls one config-layer function and returns
its payload unchanged, so the server, the CLI (`--json`) and the gateway
(which re-exposes the same functions through its own seam) emit ONE shape.

    GET  /acore/host/profile                         contract A
    GET  /acore/engines?probe=                       contract B
    GET  /acore/engines/{id}                         one contract-B row
    POST /acore/engines/{id}/install {dry_run,force} contract E job   (403 / 409)
    GET  /acore/models/catalog?q=&engine=&fits=&hub=&tag=   contract C
    GET  /acore/models/installed?provider=           contract D
    POST /acore/models/download {provider,artifact,dry_run} contract E job
    POST /acore/models/delete {provider,artifact,dry_run,force} job (409 on blockers)
    GET  /acore/jobs?kind=&status=                   contract E list (newest first)
    GET  /acore/jobs/{id}                            contract E
    POST /acore/jobs/{id}/cancel                     contract E

HOST MUTATIONS NEED A REAL PRINCIPAL. The global middleware already enforces
the server token; on top of it, every POST here refuses a request that got
through only on the "explicit upstream provider key" exemption (that header
authorizes calls to a model provider, not deleting files on this host).
Engine installs additionally need `ABSTRACTCORE_ALLOW_ENGINE_INSTALL`, which
defaults to ON only when the server is bound to loopback.
"""

from __future__ import annotations

import ipaddress
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

router = APIRouter(tags=["models & engines"])

BIND_HOST_ENV = "ABSTRACTCORE_SERVER_BIND_HOST"
ALLOW_ENGINE_INSTALL_ENV = "ABSTRACTCORE_ALLOW_ENGINE_INSTALL"


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


def _is_loopback_host(host: Optional[str]) -> bool:
    raw = str(host or "").strip().strip("[]").lower()
    if not raw:
        return False
    if raw == "localhost":
        return True
    try:
        return ipaddress.ip_address(raw).is_loopback
    except ValueError:
        return False


def server_allows_engine_install() -> bool:
    """`ABSTRACTCORE_ALLOW_ENGINE_INSTALL` when set; else True only on a loopback bind.

    The bind host is recorded by `run_server_with_args` in
    `ABSTRACTCORE_SERVER_BIND_HOST`; an unknown bind counts as NOT loopback.
    """

    raw = str(os.getenv(ALLOW_ENGINE_INSTALL_ENV) or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return _is_loopback_host(os.getenv(BIND_HOST_ENV))


def _require_host_principal(request: Request) -> None:
    from . import app as server_app

    if server_app._server_auth_enabled():
        if not server_app._request_has_server_auth(request):
            raise HTTPException(status_code=401, detail="server authentication required")
        return
    if not server_app._server_allows_unauthenticated():
        raise HTTPException(
            status_code=403,
            detail=(
                "host actions need a server principal: set ABSTRACTCORE_AUTH_TOKEN (or, for local "
                "development only, ABSTRACTCORE_SERVER_ALLOW_UNAUTHENTICATED=1)"
            ),
        )


def _refusal(status_code: int, status: str, message: str, **extra: Any) -> JSONResponse:
    """A structured refusal: machine fields at the top level, plus the
    server's usual `error` envelope so generic clients still show a message."""

    body: Dict[str, Any] = {"ok": False, "status": status, "message": message}
    body.update(extra)
    body["error"] = {"message": message, "type": f"host_action_{status}"}
    return JSONResponse(status_code=status_code, content=body)


# ---------------------------------------------------------------------------
# Bodies
# ---------------------------------------------------------------------------


class ModelDownloadBody(BaseModel):
    provider: str = Field(..., description="ollama | lmstudio | mlx | huggingface | mlx-gen | supertonic ...")
    artifact: str = Field(..., description="Exact artifact reference (quant included): qwen3:8b, qwen/qwen3.5-9b@4bit, org/Repo-GGUF:Q4_K_M")
    dry_run: bool = False
    expected_bytes: Optional[int] = Field(None, description="Known download size (catalog) for the disk pre-check")


class ModelDeleteBody(BaseModel):
    provider: str
    artifact: str
    dry_run: bool = False
    force: bool = False


class EngineInstallBody(BaseModel):
    dry_run: bool = False
    force: bool = False


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


@router.get("/acore/host/profile", summary="Host profile (contract A)")
async def acore_host_profile(refresh: bool = Query(False)) -> Dict[str, Any]:
    from ..utils.host_profile import host_profile

    return await run_in_threadpool(host_profile, refresh=refresh)


@router.get("/acore/engines", summary="Local engines status (contract B)")
async def acore_engines(probe: bool = Query(False, description="GET each local server once")) -> Dict[str, Any]:
    from ..config.engines import engine_inventory

    payload = await run_in_threadpool(engine_inventory, probe)
    payload["install_allowed"] = server_allows_engine_install()
    return payload


@router.get("/acore/engines/{engine_id}", summary="One engine (contract B row)")
async def acore_engine(engine_id: str, probe: bool = Query(False)) -> Dict[str, Any]:
    from ..config.engines import engine_status

    try:
        row = await run_in_threadpool(engine_status, engine_id, probe=probe)
    except KeyError as exc:
        return _refusal(404, "not_found", str(exc.args[0] if exc.args else exc))
    row["install_allowed"] = server_allows_engine_install()
    return row


@router.get("/acore/models/catalog", summary="Downloadable model catalog with presence and fit (contract C)")
async def acore_models_catalog(
    q: Optional[str] = Query(None),
    engine: Optional[str] = Query(None),
    fits: bool = Query(False),
    hub: bool = Query(False, description="Enrich from the Hugging Face API (cached 24 h)"),
    tag: Optional[List[str]] = Query(None),
) -> Dict[str, Any]:
    from ..config.model_catalog import catalog

    return await run_in_threadpool(lambda: catalog(q, engine=engine, fits=fits, hub=hub, tags=tag))


@router.get("/acore/models/installed", summary="Installed models per engine, with sizes (contract D)")
async def acore_models_installed(provider: Optional[str] = Query(None)) -> Dict[str, Any]:
    from ..config.model_materializer import list_installed

    return await run_in_threadpool(list_installed, provider)


def _all_jobs(kind: Optional[str], status: Optional[str]) -> List[Dict[str, Any]]:
    from ..config import host_jobs

    by_id: Dict[str, Dict[str, Any]] = {}
    registry = host_jobs.default_registry()
    if registry.persist_dir is not None:
        for job in host_jobs.read_persisted_jobs(registry.persist_dir):
            by_id[job["job_id"]] = job
    for job in registry.list():
        by_id[job["job_id"]] = job
    jobs = sorted(by_id.values(), key=lambda j: str(j.get("started_at") or ""), reverse=True)
    return [j for j in jobs if (not kind or j.get("kind") == kind) and (not status or j.get("status") == status)]


@router.get("/acore/jobs", summary="Host jobs, newest first (contract E)")
async def acore_jobs(kind: Optional[str] = Query(None), status: Optional[str] = Query(None)) -> Dict[str, Any]:
    from ..utils.host_profile import utc_now_iso

    jobs = await run_in_threadpool(_all_jobs, kind, status)
    return {"schema": "host_jobs_v1", "jobs": jobs, "generated_at": utc_now_iso()}


@router.get("/acore/jobs/{job_id}", summary="One host job (contract E)")
async def acore_job(job_id: str) -> Dict[str, Any]:
    from ..config import host_jobs

    registry = host_jobs.default_registry()
    job = registry.get(job_id)
    if job is None and registry.persist_dir is not None:
        job = host_jobs.read_persisted_job(job_id, registry.persist_dir)
    if job is None:
        return _refusal(404, "not_found", f"no job {job_id}")
    return job


# ---------------------------------------------------------------------------
# Mutations
# ---------------------------------------------------------------------------


@router.post("/acore/jobs/{job_id}/cancel", summary="Cancel a host job (contract F)")
async def acore_job_cancel(job_id: str, request: Request) -> Dict[str, Any]:
    from ..config import host_jobs

    _require_host_principal(request)
    registry = host_jobs.default_registry()
    job = registry.cancel(job_id)
    if job is None and registry.persist_dir is not None:
        job = host_jobs.request_cancel(job_id, registry.persist_dir)
    if job is None:
        return _refusal(404, "not_found", f"no job {job_id}")
    return job


@router.post("/acore/models/download", summary="Download one artifact as a job (contract F)")
async def acore_models_download(body: ModelDownloadBody, request: Request) -> Dict[str, Any]:
    from ..config import host_jobs

    _require_host_principal(request)
    try:
        return await run_in_threadpool(
            lambda: host_jobs.start_download_job(
                body.provider,
                body.artifact,
                dry_run=body.dry_run,
                expected_bytes=body.expected_bytes,
                run_inline=body.dry_run,
            )
        )
    except ValueError as exc:
        return _refusal(400, "invalid", str(exc))


@router.post("/acore/models/delete", summary="Delete one installed artifact as a job (contract F)")
async def acore_models_delete(body: ModelDeleteBody, request: Request) -> Dict[str, Any]:
    from ..config import host_jobs
    from ..config.model_materializer import delete_blockers

    _require_host_principal(request)
    check = await run_in_threadpool(delete_blockers, body.provider, body.artifact)
    if not check.get("found"):
        blockers = check.get("delete_blockers") or []
        return _refusal(
            409 if blockers else 404,
            "refused" if blockers else "not_found",
            f"{body.artifact} is not installed for {body.provider}" + (f" ({check['error']})" if check.get("error") else ""),
            delete_blockers=blockers,
        )
    blockers = check.get("delete_blockers") or []
    hard = [b for b in blockers if b in {"unknown_location", "engine_not_running"}]
    if hard or (blockers and not body.force):
        return _refusal(
            409,
            "refused",
            "refusing to delete: " + ", ".join(blockers) + ("" if hard else " (send force=true to override)"),
            delete_blockers=blockers,
        )
    try:
        return await run_in_threadpool(
            lambda: host_jobs.start_delete_job(
                body.provider, body.artifact, dry_run=body.dry_run, force=body.force, run_inline=body.dry_run
            )
        )
    except ValueError as exc:
        return _refusal(400, "invalid", str(exc))


@router.post("/acore/engines/{engine_id}/install", summary="Install an engine as a job (contract F)")
async def acore_engine_install(engine_id: str, request: Request, body: Optional[EngineInstallBody] = None) -> Dict[str, Any]:
    from ..config import host_jobs
    from ..config.engines import EngineInstallRefused, engine_install

    _require_host_principal(request)
    body = body or EngineInstallBody()
    allowed = server_allows_engine_install()
    try:
        return await run_in_threadpool(
            lambda: engine_install(
                engine_id, dry_run=body.dry_run, force=body.force, allow=allowed, run_inline=body.dry_run
            )
        )
    except EngineInstallRefused as exc:
        code = {"not_allowed": 403, "unknown_engine": 404}.get(exc.reason, 409)
        return _refusal(code, "refused", str(exc), reason=exc.reason, install=exc.plan)
    except host_jobs.JobBusy as exc:
        return _refusal(409, "busy", str(exc), reason="busy", job=exc.job)
