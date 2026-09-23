"""CLI verbs for the models & engines contracts (A-F).

    abstractcore host profile [--json]
    abstractcore models list|catalog|search|delete|jobs|cancel ...
    abstractcore engines status|install|open ...

Wired from `config/main.py`. Every verb has `--json` and the same exit
codes: 0 ok, 1 error, 2 refused (policy, blockers, or a destructive action
without `--yes`; the refusal JSON -- `status`, `message`, `reason` /
`delete_blockers` -- is still printed). The JSON is the exact payload the
core server routes return, so the terminal console's CLI transport and the
web consoles render one shape.

Two verbs are LONG and stream instead of printing one document:
`models download <provider> <artifact> --json` and `engines install <id>
--yes --json` print one compact `host_job_v1` object per line while the job
runs (NDJSON) and end with the final job. SIGTERM/SIGINT cancel the job,
terminate the vendor tool's process group, and the last line says
`cancelled`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_REFUSED = 2


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# NDJSON job streaming (the terminal console's CLI transport reads this)
# ---------------------------------------------------------------------------

_STREAM_KEYS = ("status", "percent", "downloaded_bytes", "total_bytes", "message")
_CANCEL_GRACE_S = 10.0


def stream_job_ndjson(
    job_id: str,
    registry: Any,
    *,
    final_extra: Optional[Dict[str, Any]] = None,
    poll_s: float = 0.2,
    out: Any = None,
) -> Dict[str, Any]:
    """Print one compact `host_job_v1` JSON object per line while the job runs,
    then the final object (plus `final_extra`). Returns the final job.

    SIGTERM / SIGINT cancel the job -- which terminates the vendor tool's
    whole process group -- and the last line reports `cancelled`. A job that
    does not stop within a grace period (a Hugging Face transfer between
    progress ticks) is reported `cancelled` anyway and abandoned with the
    process.
    """

    import signal

    stream = out or sys.stdout
    state = {"cancel_at": None, "signals": 0}

    def _on_signal(signum: int, _frame: Any) -> None:
        state["signals"] += 1
        if state["cancel_at"] is None:
            state["cancel_at"] = time.monotonic()
            registry.cancel(job_id)
        elif state["signals"] >= 3:
            raise SystemExit(130)

    previous: Dict[int, Any] = {}
    for sig in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGINT", None)):
        if sig is None:
            continue
        try:
            previous[sig] = signal.signal(sig, _on_signal)
        except Exception:
            pass  # not the main thread: no handler, the job still runs

    def _emit(job: Dict[str, Any]) -> None:
        stream.write(json.dumps(job, sort_keys=True, default=str) + "\n")
        stream.flush()

    last_key: Any = None
    try:
        while True:
            job = registry.get(job_id) or {}
            finished = job.get("status") not in ("queued", "running")
            key = tuple(job.get(k) for k in _STREAM_KEYS)
            if finished:
                break
            if state["cancel_at"] is not None and time.monotonic() - state["cancel_at"] > _CANCEL_GRACE_S:
                job = dict(job, status="cancelled", message="cancelled (the transfer was abandoned on exit)", error=None)
                break
            if key != last_key:
                _emit(job)
                last_key = key
            time.sleep(poll_s)
    finally:
        for sig, handler in previous.items():
            try:
                signal.signal(sig, handler)
            except Exception:
                pass
    final = dict(job)
    final.update(final_extra or {})
    _emit(final)
    return final


def _gb(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "?"
    if value >= 1e9:
        return f"{value / 1e9:.1f} GB"
    return f"{value / 1e6:.0f} MB"


def _confirm(prompt: str) -> bool:
    if not sys.stdin.isatty():
        return False
    try:
        answer = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return answer in {"y", "yes"}


# ---------------------------------------------------------------------------
# host
# ---------------------------------------------------------------------------


def handle_host(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(prog="abstractcore host", description="Describe this machine for local models.")
    sub = parser.add_subparsers(dest="cmd")
    profile = sub.add_parser("profile", help="OS, accelerator, memory ceiling, free memory and disk per engine store")
    profile.add_argument("--json", action="store_true", help="Emit host_profile_v1 JSON")
    args = parser.parse_args(argv)
    if args.cmd != "profile":
        parser.print_help()
        return EXIT_ERROR
    from ..utils.host_profile import host_profile

    payload = host_profile(refresh=True)
    if args.json:
        _print_json(payload)
        return EXIT_OK
    print("Host profile")
    print(f"  os/arch        {payload['os']} / {payload['arch']}   python {payload['python']}")
    gpu = payload.get("gpu_name") or "-"
    print(f"  accelerator    {payload['accelerator']} ({gpu}){'  unified memory' if payload.get('unified_memory') else ''}")
    print(f"  RAM            {_gb(payload.get('ram_bytes'))}" + (f"   VRAM {_gb(payload.get('vram_bytes'))}" if payload.get("vram_bytes") else ""))
    print(f"  model ceiling  {_gb(payload.get('ceiling_bytes'))}  ({payload.get('ceiling_source') or 'unknown'})")
    print(f"  free now       {_gb(payload.get('free_now_bytes'))}")
    for name, entry in (payload.get("disk") or {}).items():
        print(f"  disk {name:<9} {_gb(entry.get('free_bytes'))} free  {entry.get('path')}")
    for note in payload.get("notes") or []:
        print(f"  note: {note}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# models (new verbs; status/download stay in main.py)
# ---------------------------------------------------------------------------


def add_models_subparsers(sub: Any) -> None:
    """Register `list`, `catalog`, `search`, `delete`, `jobs`, `cancel`."""

    listing = sub.add_parser("list", help="List installed models per engine, with sizes")
    listing.add_argument("--provider", default=None, help="ollama | lmstudio | mlx | huggingface")
    listing.add_argument("--json", action="store_true", help="Emit models_installed_v1 JSON")
    listing.set_defaults(func=_handle_list)

    cat = sub.add_parser("catalog", help="Browse the curated download catalog with presence and fit verdicts")
    _catalog_args(cat)
    cat.set_defaults(func=_handle_catalog, query=None)

    search = sub.add_parser("search", help="Search the catalog (and, with --hub, Hugging Face)")
    search.add_argument("query", help="Free text, e.g. 'qwen3 8b' or 'embedding'")
    _catalog_args(search)
    search.set_defaults(func=_handle_catalog)

    delete = sub.add_parser("delete", help="Delete one installed model (refuses when loaded unless --force)")
    delete.add_argument("provider", help="ollama | lmstudio | mlx | huggingface")
    delete.add_argument("artifact", help="The installed artifact, as `models list` prints it")
    delete.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    delete.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    delete.add_argument("--force", action="store_true", help="Delete even when loaded / shared (unloads first)")
    delete.add_argument("--json", action="store_true", help="Emit the host_job_v1 JSON")
    delete.set_defaults(func=_handle_delete)

    jobs = sub.add_parser("jobs", help="List download/delete/engine-install jobs (or show one)")
    jobs.add_argument("job_id", nargs="?", default=None, help="Show one job")
    jobs.add_argument("--kind", default=None, choices=["download", "delete", "engine_install"])
    jobs.add_argument("--status", default=None, choices=["queued", "running", "completed", "failed", "cancelled"])
    jobs.add_argument("--json", action="store_true", help="Emit JSON")
    jobs.set_defaults(func=_handle_jobs)

    cancel = sub.add_parser("cancel", help="Cancel a running job")
    cancel.add_argument("job_id")
    cancel.add_argument("--json", action="store_true", help="Emit JSON")
    cancel.set_defaults(func=_handle_cancel)


def _catalog_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", default=None, help="Only artifacts for this engine/provider (ollama, lmstudio, mlx, huggingface, llamacpp)")
    parser.add_argument("--fits", action="store_true", help="Only artifacts that fit this machine (fits or tight)")
    parser.add_argument("--hub", action="store_true", help="Ask Hugging Face for exact sizes (and search results); cached 24 h")
    parser.add_argument("--tag", action="append", default=None, help="Only rows with this tag (repeatable): chat, coding, vision, embedding...")
    parser.add_argument("--json", action="store_true", help="Emit model_catalog_v1 JSON")


def _handle_list(args: argparse.Namespace) -> int:
    from .model_materializer import list_installed

    payload = list_installed(getattr(args, "provider", None))
    if args.json:
        _print_json(payload)
        return EXIT_OK
    rows = payload["rows"]
    print(f"Installed models ({len(rows)}, {_gb(payload['totals']['size_bytes'])})")
    for row in rows:
        flags = []
        if row.get("loaded"):
            flags.append("loaded")
        if row.get("delete_blockers"):
            flags.append("blocked:" + ",".join(row["delete_blockers"]))
        print(
            f"  {row['provider']:<11} {row['artifact']:<58} {str(row.get('quant') or '-'):<8} "
            f"{_gb(row.get('size_bytes')):>9}  {' '.join(flags)}"
        )
    for name, error in (payload.get("errors") or {}).items():
        print(f"  ! {name}: {error}")
    return EXIT_OK


def _handle_catalog(args: argparse.Namespace) -> int:
    from .model_catalog import catalog

    payload = catalog(
        getattr(args, "query", None),
        engine=args.engine,
        fits=bool(args.fits),
        hub=bool(args.hub),
        tags=args.tag,
    )
    if args.json:
        _print_json(payload)
        return EXIT_OK
    host = payload["host_profile"]
    print(
        f"Model catalog -- {host.get('accelerator')} {host.get('gpu_name') or ''}, "
        f"ceiling {_gb(host.get('ceiling_bytes'))}, free now {_gb(host.get('free_now_bytes'))}"
    )
    for row in payload["rows"]:
        print(f"\n{row['display_name']}  [{row['id']}]  {', '.join(row.get('tags') or [])}")
        for art in row["artifacts"]:
            fit = art.get("fit") or {}
            mark = "*" if art.get("recommended") else " "
            status = art["presence"]["status"]
            size = _gb(art.get("download_bytes")) + ("~" if art.get("size_source") == "estimate" else "")
            disk = "" if fit.get("disk_ok") in (True, None) else "  NO DISK SPACE"
            print(f"  {mark} {art['provider']:<11} {art['artifact']:<52} {size:>10}  {status:<9} {fit.get('verdict')}{disk}")
    hub = payload.get("hub")
    if hub:
        for err in hub.get("errors") or []:
            print(f"  ! hub: {err}")
    print("\n* = pre-selected for this machine.  Download: abstractcore models download <provider> <artifact>")
    return EXIT_OK


def _handle_delete(args: argparse.Namespace) -> int:
    from . import host_jobs
    from .model_materializer import delete_blockers

    check = delete_blockers(args.provider, args.artifact)
    if not check.get("found"):
        payload = {
            "ok": False,
            "status": "not_found",
            "provider": args.provider,
            "artifact": args.artifact,
            "message": f"{args.artifact} is not installed for {args.provider}" + (f" ({check['error']})" if check.get("error") else ""),
            "delete_blockers": check.get("delete_blockers") or [],
            "reason": "blocked" if check.get("delete_blockers") else "not_found",
        }
        _print_json(payload) if args.json else print(f"❌ {payload['message']}")
        return EXIT_REFUSED if payload["delete_blockers"] else EXIT_ERROR
    blockers = check.get("delete_blockers") or []
    if blockers and not args.force:
        payload = {
            "ok": False,
            "status": "refused",
            "provider": args.provider,
            "artifact": args.artifact,
            "delete_blockers": blockers,
            "message": "refusing to delete: " + ", ".join(blockers) + " (use --force to override)",
            "reason": "blocked",
        }
        _print_json(payload) if args.json else print(f"⛔ {payload['message']}")
        return EXIT_REFUSED
    row = check.get("row") or {}
    if not args.yes and not args.dry_run:
        if args.json or not _confirm(f"Delete {args.provider} {args.artifact} ({_gb(row.get('size_bytes'))}) at {row.get('location')}?"):
            payload = {
                "ok": False,
                "status": "refused",
                "reason": "not_confirmed",
                "provider": args.provider,
                "artifact": args.artifact,
                "delete_blockers": [],
                "message": "not confirmed: pass --yes to delete without a prompt",
            }
            _print_json(payload) if args.json else print(f"⛔ {payload['message']}")
            return EXIT_REFUSED
    job = host_jobs.start_delete_job(
        args.provider, args.artifact, dry_run=bool(args.dry_run), force=bool(args.force), run_inline=True
    )
    result = job.get("result") or {}
    if args.json:
        _print_json(job)
    else:
        glyph = "✅" if job["status"] == "completed" else "❌"
        print(f"{glyph} {result.get('message') or job.get('message')}")
        for path in result.get("paths") or []:
            print(f"   {path}")
        if result.get("freed_bytes"):
            print(f"   {'would free' if args.dry_run else 'freed'} {_gb(result['freed_bytes'])}")
    if result.get("status") == "refused":
        return EXIT_REFUSED
    return EXIT_OK if job["status"] == "completed" else EXIT_ERROR


def _all_jobs() -> List[Dict[str, Any]]:
    from . import host_jobs

    by_id: Dict[str, Dict[str, Any]] = {j["job_id"]: j for j in host_jobs.read_persisted_jobs()}
    for job in host_jobs.default_registry().list():
        by_id[job["job_id"]] = job
    return sorted(by_id.values(), key=lambda j: str(j.get("started_at") or ""), reverse=True)


def _handle_jobs(args: argparse.Namespace) -> int:
    from . import host_jobs

    if args.job_id:
        job = host_jobs.default_registry().get(args.job_id) or host_jobs.read_persisted_job(args.job_id)
        if job is None:
            _print_json({"ok": False, "message": f"no job {args.job_id}"}) if args.json else print(f"❌ no job {args.job_id}")
            return EXIT_ERROR
        if args.json:
            _print_json(job)
        else:
            _print_job(job)
            for line in job.get("log_tail") or []:
                print(f"   | {line}")
        return EXIT_OK
    jobs = [
        j
        for j in _all_jobs()
        if (args.kind is None or j.get("kind") == args.kind) and (args.status is None or j.get("status") == args.status)
    ]
    if args.json:
        _print_json({"schema": "host_jobs_v1", "jobs": jobs})
        return EXIT_OK
    if not jobs:
        print("No jobs.")
    for job in jobs:
        _print_job(job)
    return EXIT_OK


def _print_job(job: Dict[str, Any]) -> None:
    target = job.get("engine") or f"{job.get('provider')} {job.get('artifact')}"
    pct = f" {job['percent']:.0f}%" if isinstance(job.get("percent"), (int, float)) else ""
    print(f"  {job['job_id']:<18} {job['kind']:<15} {job['status']:<10}{pct:>5}  {target}  {job.get('message') or ''}")


def _handle_cancel(args: argparse.Namespace) -> int:
    from . import host_jobs

    job = host_jobs.default_registry().cancel(args.job_id) or host_jobs.request_cancel(args.job_id)
    if job is None:
        _print_json({"ok": False, "message": f"no job {args.job_id}"}) if args.json else print(f"❌ no job {args.job_id}")
        return EXIT_ERROR
    if args.json:
        _print_json(job)
    else:
        print(f"cancel requested for {job['job_id']} ({job.get('status')})")
    return EXIT_OK


# ---------------------------------------------------------------------------
# engines
# ---------------------------------------------------------------------------


def handle_engines(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="abstractcore engines",
        description="Detect, install and open the download pages of local inference engines.",
    )
    sub = parser.add_subparsers(dest="cmd")
    status = sub.add_parser("status", help="Which engines are installed / running here")
    status.add_argument("--probe", action="store_true", help="Also ask each local server whether it answers (one GET each)")
    status.add_argument("--json", action="store_true", help="Emit engines_status_v1 JSON")

    install = sub.add_parser("install", help="Install an engine with its vendor's command (shown first)")
    install.add_argument("engine", help="ollama | lmstudio | mlx | llamacpp | vllm | huggingface")
    install.add_argument("--yes", action="store_true", help="Do not ask for confirmation")
    install.add_argument("--dry-run", action="store_true", help="Only show the exact command")
    install.add_argument("--force", action="store_true", help="Run even when the engine is already installed")
    install.add_argument("--detach", action="store_true", help="Run in the background; print the job and exit")
    install.add_argument("--json", action="store_true", help="Emit the host_job_v1 JSON")

    open_ = sub.add_parser("open", help="Print (and open) an engine's download page")
    open_.add_argument("engine")
    open_.add_argument("--no-browser", action="store_true", help="Only print the URL")
    open_.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    if not args.cmd:
        args = parser.parse_args([*argv, "status"])
    try:
        if args.cmd == "status":
            return _engines_status(args)
        if args.cmd == "install":
            return _engines_install(args)
        if args.cmd == "open":
            return _engines_open(args)
    except KeyError as exc:
        print(f"❌ {exc.args[0] if exc.args else exc}")
        return EXIT_ERROR
    parser.print_help()
    return EXIT_ERROR


def _engines_status(args: argparse.Namespace) -> int:
    from .engines import engine_inventory

    payload = engine_inventory(probe=bool(args.probe))
    if args.json:
        _print_json(payload)
        return EXIT_OK
    print("Local engines" + (" (probed)" if args.probe else ""))
    for e in payload["engines"]:
        if not e["supported_on_host"]:
            state = f"not for this host: {e['unsupported_reason']}"
        elif e["installed"]:
            state = f"installed {e.get('version') or ''}".strip()
            if e.get("running") is True:
                state += f", running at {e.get('base_url')}" + (f" ({e['models_count']} models)" if e.get("models_count") is not None else "")
            elif e.get("running") is False:
                state += ", not running"
        else:
            plan = e["install"]
            state = "not installed" + (f" -> abstractcore engines install {e['id']}" if plan.get("available") else "")
        print(f"  {e['name']:<28} {state}")
    if not payload.get("install_allowed"):
        print("  (engine installs are disabled on this host: ABSTRACTCORE_ALLOW_ENGINE_INSTALL=0)")
    return EXIT_OK


def _engines_install(args: argparse.Namespace) -> int:
    from . import host_jobs
    from .engines import EngineInstallRefused, engine_install, engine_install_allowed, engine_status

    eid = args.engine.strip().lower()
    status = engine_status(eid, probe=False)
    plan = status["install"]
    refused: Optional[str] = None
    reason = ""
    if not status["supported_on_host"]:
        refused, reason = status.get("unsupported_reason") or "not supported on this host", "unsupported"
    elif not plan.get("available"):
        refused, reason = plan.get("notes") or "no install command for this host", "no_plan"
    elif not args.dry_run and not engine_install_allowed():
        refused, reason = "engine installs are disabled on this host (ABSTRACTCORE_ALLOW_ENGINE_INSTALL=0)", "not_allowed"
    if refused:
        payload = {"ok": False, "status": "refused", "reason": reason, "engine": eid, "message": refused, "install": plan}
        _print_json(payload) if args.json else print(f"⛔ {refused}\n   download page: {plan.get('url')}")
        return EXIT_REFUSED

    if not args.json:
        print(f"{status['name']}: {'installed ' + (status.get('version') or '') if status['installed'] else 'not installed'}")
        print(f"  command : {' '.join(plan['argv'])}")
        print(f"  runs on : this machine ({'needs sudo/admin' if plan.get('requires_admin') else 'no admin needed'})")
        print(f"  notes   : {plan.get('notes')}")
        if plan.get("url"):
            print(f"  page    : {plan['url']}")
    if not args.yes and not args.dry_run:
        if args.json or not _confirm("Run this command now?"):
            payload = {"ok": False, "status": "refused", "reason": "not_confirmed", "engine": eid, "message": "not confirmed: pass --yes to install without a prompt", "install": plan}
            _print_json(payload) if args.json else print(f"⛔ {payload['message']}")
            return EXIT_REFUSED

    if args.detach and not args.dry_run:
        job = host_jobs.spawn_detached({"kind": "engine_install", "engine": eid, "force": bool(args.force)})
        _print_json(job) if args.json else print(f"started {job['job_id']} -- follow it with: abstractcore models jobs {job['job_id']}")
        return EXIT_OK

    try:
        if args.dry_run:
            job = engine_install(eid, dry_run=True, force=bool(args.force), run_inline=True)
        elif args.json:
            started = engine_install(eid, force=bool(args.force))
            job = stream_job_ndjson(started["job_id"], host_jobs.default_registry())
            return EXIT_OK if job["status"] == "completed" else EXIT_ERROR
        else:
            job = engine_install(eid, force=bool(args.force))
            job = _follow(job["job_id"])
    except EngineInstallRefused as exc:
        payload = {"ok": False, "status": "refused", "engine": eid, "reason": exc.reason, "message": str(exc)}
        _print_json(payload) if args.json else print(f"⛔ {exc}")
        return EXIT_REFUSED
    except host_jobs.JobBusy as exc:
        payload = {"ok": False, "status": "refused", "engine": eid, "reason": "busy", "message": str(exc), "job": exc.job}
        _print_json(payload) if args.json else print(f"⛔ {exc}")
        return EXIT_REFUSED
    if args.json:
        _print_json(job)
    else:
        result = job.get("result") or {}
        glyph = "✅" if job["status"] == "completed" else "❌"
        print(f"{glyph} {result.get('message') or job.get('message')}")
    return EXIT_OK if job["status"] == "completed" else EXIT_ERROR


def _follow(job_id: str) -> Dict[str, Any]:
    """Stream a running job's new log lines until it finishes (foreground CLI)."""

    from . import host_jobs

    registry = host_jobs.default_registry()
    printed = 0
    while True:
        job = registry.get(job_id) or {}
        tail = job.get("log_tail") or []
        if len(tail) > printed:
            for line in tail[printed:]:
                print(f"   | {line}", flush=True)
            printed = len(tail)
        if job.get("status") not in ("queued", "running"):
            return job
        try:
            time.sleep(0.2)
        except KeyboardInterrupt:
            registry.cancel(job_id)


def _engines_open(args: argparse.Namespace) -> int:
    from .engines import engine_download_url

    url = engine_download_url(args.engine)
    opened = False
    if not args.no_browser:
        try:
            import webbrowser

            opened = bool(webbrowser.open(url))
        except Exception:
            opened = False
    if args.json:
        _print_json({"engine": args.engine.lower(), "url": url, "opened": opened})
    else:
        print(url)
    return EXIT_OK
