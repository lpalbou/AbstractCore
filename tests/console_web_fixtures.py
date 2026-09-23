"""Contract fixtures + a stub router for the web console tests.

The real `/acore/host|engines|models|jobs` routes are built by another
workstream against the same contracts (host_profile_v1, engines_status_v1,
model_catalog_v1, models_installed_v1, host_job_v1). These fixtures follow
those shapes so the console can be exercised without the backend.
Not a test module (no `test_` prefix): imported by tests/test_console_web.py
and by the subprocess smoke server.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException

NOW = "2026-09-23T12:00:00Z"
GIB = 1024**3

HOST_PROFILE: Dict[str, Any] = {
    "schema": "host_profile_v1",
    "os": "darwin",
    "arch": "arm64",
    "accelerator": "metal",
    "gpu_name": "Apple M5 Max",
    "unified_memory": True,
    "ram_bytes": 128 * GIB,
    "vram_bytes": None,
    "ceiling_bytes": 96 * GIB,
    "ceiling_source": "metal_wired_limit",
    "free_now_bytes": 57 * GIB,
    "disk": {
        "hf_cache": {"path": "~/.cache/huggingface/hub", "free_bytes": 800 * GIB},
        "lmstudio": {"path": "~/.lmstudio/models", "free_bytes": 800 * GIB},
        "ollama": {"path": "~/.ollama/models", "free_bytes": 800 * GIB},
    },
    "python": "3.12.13",
    "generated_at": NOW,
}

ENGINES: Dict[str, Any] = {
    "schema": "engines_status_v1",
    "engines": [
        {
            "id": "ollama", "name": "Ollama", "kind": "local_server",
            "supported_on_host": True, "unsupported_reason": None,
            "installed": False, "version": None, "install_location": None,
            "running": False, "base_url": "http://127.0.0.1:11434", "reachable": False, "models_count": None,
            "install": {"available": True, "method": "brew", "argv": ["brew", "install", "ollama"],
                        "url": "https://ollama.com/download", "requires_confirmation": True,
                        "estimated_bytes": None, "notes": "Installs the Ollama CLI and background service."},
            "docs_url": "https://ollama.com",
        },
        {
            "id": "lmstudio", "name": "LM Studio", "kind": "local_server",
            "supported_on_host": True, "unsupported_reason": None,
            "installed": True, "version": "0.3.30", "install_location": "/Applications/LM Studio.app",
            "running": True, "base_url": "http://127.0.0.1:1234/v1", "reachable": True, "models_count": 7,
            "install": {"available": True, "method": "download_page", "argv": None,
                        "url": "https://lmstudio.ai/download", "requires_confirmation": True,
                        "estimated_bytes": None, "notes": "GUI app; install by hand."},
            "docs_url": "https://lmstudio.ai/docs",
        },
        {
            "id": "mlx", "name": "MLX", "kind": "local_engine",
            "supported_on_host": True, "unsupported_reason": None,
            "installed": True, "version": "0.29.1", "install_location": None,
            "running": None, "base_url": None, "reachable": None, "models_count": 3,
            "install": {"available": True, "method": "pip", "argv": ["/usr/bin/python3", "-m", "pip", "install", "mlx-lm"],
                        "url": None, "requires_confirmation": True, "estimated_bytes": 120_000_000, "notes": None},
            "docs_url": None,
        },
        {
            "id": "vllm", "name": "vLLM", "kind": "local_server",
            "supported_on_host": False, "unsupported_reason": "vLLM needs Linux with CUDA",
            "installed": False, "version": None, "install_location": None,
            "running": None, "base_url": None, "reachable": None, "models_count": None,
            "install": {"available": False, "method": None, "argv": None, "url": None,
                        "requires_confirmation": True, "estimated_bytes": None, "notes": None},
            "docs_url": "https://docs.vllm.ai",
        },
    ],
    "generated_at": NOW,
}


def _fit(verdict: str, need: int) -> Dict[str, Any]:
    return {"verdict": verdict, "need_bytes": need, "ceiling_bytes": 96 * GIB, "free_now_bytes": 57 * GIB,
            "fits_now": need < 57 * GIB, "disk_ok": True, "max_context": 32768,
            "confidence": "estimated", "notes": ["weights + 8k context KV"]}


CATALOG: Dict[str, Any] = {
    "schema": "model_catalog_v1",
    "host_profile": HOST_PROFILE,
    "rows": [
        {
            "id": "qwen3-8b", "family": "qwen3", "display_name": "Qwen3 8B", "vendor": "Qwen",
            "params_total": 8_200_000_000, "params_active": None, "license": "apache-2.0",
            "capabilities": {"text": True, "vision": False, "audio": False, "tools": "native", "thinking": True, "max_tokens": 32768, "embedding": False},
            "source": "curated", "tags": ["chat", "coding"],
            "artifacts": [
                {"provider": "ollama", "artifact": "qwen3:8b", "quant": "q4_k_m", "bits": 4.85,
                 "download_bytes": 5_200_000_000, "size_source": "catalog",
                 "presence": {"status": "absent", "location": None, "evidence": "ollama list"},
                 "fit": _fit("fits", 7 * GIB), "downloadable": True, "recommended": True},
                {"provider": "lmstudio", "artifact": "qwen/qwen3-8b@4bit", "quant": "4bit", "bits": 4.5,
                 "download_bytes": 4_600_000_000, "size_source": "hf_api",
                 "presence": {"status": "installed", "location": "~/.lmstudio/models/qwen/qwen3-8b", "evidence": "lms ls"},
                 "fit": _fit("fits", 6 * GIB), "downloadable": True, "recommended": False},
            ],
        },
        {
            "id": "llama-3.1-405b", "family": "llama3.1", "display_name": "Llama 3.1 405B", "vendor": "Meta",
            "params_total": 405_000_000_000, "params_active": None, "license": "llama3.1",
            "capabilities": {"text": True, "vision": False, "audio": False, "tools": "prompted", "thinking": False, "max_tokens": 8192, "embedding": False},
            "source": "curated", "tags": ["chat"],
            "artifacts": [
                {"provider": "ollama", "artifact": "llama3.1:405b", "quant": "q4_0", "bits": 4.5,
                 "download_bytes": 231_000_000_000, "size_source": "catalog",
                 "presence": {"status": "absent", "location": None, "evidence": "ollama list"},
                 "fit": _fit("too_large", 240 * GIB), "downloadable": True, "recommended": False},
            ],
        },
        {
            "id": "gpt-5", "family": "gpt-5", "display_name": "GPT-5 <remote>", "vendor": "OpenAI",
            "params_total": None, "params_active": None, "license": None,
            "capabilities": {"text": True, "vision": True, "audio": False, "tools": "native", "thinking": True, "max_tokens": None, "embedding": False},
            "source": "curated", "tags": [],
            "artifacts": [
                {"provider": "openai", "artifact": "gpt-5", "quant": None, "bits": None,
                 "download_bytes": None, "size_source": "unknown",
                 "presence": {"status": "not_applicable", "location": None, "evidence": "remote API"},
                 "fit": {"verdict": "unknown", "need_bytes": None, "ceiling_bytes": None, "free_now_bytes": None,
                         "fits_now": None, "disk_ok": None, "max_context": None, "confidence": "unknown", "notes": []},
                 "downloadable": False, "recommended": False},
            ],
        },
    ],
    "generated_at": NOW,
}

INSTALLED: Dict[str, Any] = {
    "schema": "models_installed_v1",
    "rows": [
        {"provider": "lmstudio", "artifact": "qwen/qwen3-8b@4bit", "quant": "4bit", "size_bytes": 4_600_000_000,
         "params_total": 8_200_000_000, "location": "~/.lmstudio/models/qwen/qwen3-8b", "loaded": True,
         "catalog_id": "qwen3-8b", "deletable": True, "delete_blockers": ["loaded"]},
        {"provider": "mlx", "artifact": "mlx-community/Qwen3-4B-4bit", "quant": "4bit", "size_bytes": 2_300_000_000,
         "params_total": None, "location": "~/.cache/huggingface/hub/models--mlx-community--Qwen3-4B-4bit", "loaded": None,
         "catalog_id": None, "deletable": True, "delete_blockers": ["shared_cache:mlx,huggingface"]},
    ],
    "engines_probed": ["ollama", "lmstudio", "mlx", "huggingface"],
    "errors": {"ollama": "unreachable"},
    "generated_at": NOW,
}


def make_job(job_id: str, kind: str, **fields: Any) -> Dict[str, Any]:
    job = {
        "schema": "host_job_v1", "job_id": job_id, "kind": kind, "status": "running",
        "provider": None, "artifact": None, "engine": None,
        "percent": 0.0, "downloaded_bytes": 0, "total_bytes": 0, "message": "starting",
        "log_tail": [], "command": None, "dry_run": False, "started_at": NOW, "finished_at": None,
        "error": None, "joined": 0, "cli_equivalent": None,
    }
    job.update(fields)
    return job


def build_stub_router(prefix: str = "/acore") -> APIRouter:
    """Register fixture responses for every contract route the console calls.

    Jobs advance 50 percentage points per poll, so a download completes on
    the second GET /jobs/{id}."""
    router = APIRouter()
    jobs: Dict[str, Dict[str, Any]] = {}
    counter = itertools.count(1)
    calls: Dict[str, list] = {"catalog_queries": [], "posts": []}
    router.calls = calls  # type: ignore[attr-defined]

    @router.get(f"{prefix}/host/profile")
    def host_profile() -> Dict[str, Any]:
        return copy.deepcopy(HOST_PROFILE)

    @router.get(f"{prefix}/engines")
    def engines(probe: bool = False) -> Dict[str, Any]:
        return copy.deepcopy(ENGINES)

    @router.get(f"{prefix}/models/catalog")
    def catalog(q: str = "", engine: str = "", fits: bool = False, hub: bool = False) -> Dict[str, Any]:
        calls["catalog_queries"].append({"q": q, "engine": engine, "fits": fits, "hub": hub})
        data = copy.deepcopy(CATALOG)
        if q:
            data["rows"] = [r for r in data["rows"] if q.lower() in r["display_name"].lower()]
        return data

    @router.get(f"{prefix}/models/installed")
    def installed(provider: str = "") -> Dict[str, Any]:
        return copy.deepcopy(INSTALLED)

    def _new(kind: str, body: Dict[str, Any], **fields: Any) -> Dict[str, Any]:
        calls["posts"].append({"kind": kind, **body})
        job_id = f"{kind[:2]}_{next(counter):04d}"
        dry = bool(body.get("dry_run"))
        job = make_job(job_id, kind, dry_run=dry, **fields)
        if dry:
            job.update(status="completed", percent=100.0, message="dry run: nothing executed", finished_at=NOW)
        jobs[job_id] = job
        return copy.deepcopy(job)

    @router.post(f"{prefix}/models/download")
    def download(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        p, a = body.get("provider"), body.get("artifact")
        return _new("download", body, provider=p, artifact=a, command=["ollama", "pull", a],
                    total_bytes=5_200_000_000, cli_equivalent=f"abstractcore models download {p} {a}")

    @router.post(f"{prefix}/models/delete")
    def delete(body: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        p, a = body.get("provider"), body.get("artifact")
        return _new("delete", body, provider=p, artifact=a, command=["lms", "rm", a],
                    cli_equivalent=f"abstractcore models delete {p} {a} --yes")

    @router.post(f"{prefix}/engines/{{engine_id}}/install")
    def install(engine_id: str, body: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        return _new("engine_install", {"engine": engine_id, **body}, engine=engine_id,
                    command=["brew", "install", engine_id], cli_equivalent=f"abstractcore engines install {engine_id} --yes")

    @router.get(f"{prefix}/jobs/{{job_id}}")
    def job(job_id: str) -> Dict[str, Any]:
        j = jobs.get(job_id)
        if j is None:
            raise HTTPException(status_code=404, detail="unknown job")
        if j["status"] == "running":
            j["percent"] = min(100.0, j["percent"] + 50.0)
            j["downloaded_bytes"] = int(j["total_bytes"] * j["percent"] / 100)
            j["message"] = f"pulling {j['percent']:.0f}%"
            j["log_tail"] = j["log_tail"] + [j["message"]]
            if j["percent"] >= 100.0:
                j.update(status="completed", message="done", finished_at=NOW)
        return copy.deepcopy(j)

    @router.post(f"{prefix}/jobs/{{job_id}}/cancel")
    def cancel(job_id: str) -> Dict[str, Any]:
        j = jobs.get(job_id)
        if j is None:
            raise HTTPException(status_code=404, detail="unknown job")
        if j["status"] in {"queued", "running"}:
            j.update(status="cancelled", message="cancelled by user", finished_at=NOW)
        return copy.deepcopy(j)

    return router
