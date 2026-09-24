"""Offline fakes for the models & engines tests.

Nothing here touches a real engine, a real model store or the network:

- `FakeOllama`       a threading HTTP server speaking /api/tags, /api/ps,
                     /api/version, /api/pull (ndjson stream), /api/delete,
                     /api/generate (keep_alive: 0 unload)
- `install_fake_lms` a `lms` executable on PATH answering `ls --json`,
                     `ps --json`, `unload`, `get`, `version` from a JSON file
- `make_hf_repo`     a Hugging Face hub cache entry laid out exactly like
                     huggingface_hub writes it (blobs + snapshot symlinks)
- `FakeHfApi`        `model_info(files_metadata=True)` / `list_models`
- `isolate_host`     HOME, caches, config and PATH pointed at tmp_path

Field names mirror what the real tools emit (verified 2026-09-23 on an
M5 Max: `lms ls --json` rows carry type/modelKey/format/displayName/
publisher/path/sizeBytes/indexedModelIdentifier/deviceIdentifier/
architecture/quantization{name,bits}/maxContextLength and, for some,
paramsString/vision/trainedForToolUse/variants/selectedVariant).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Host isolation
# ---------------------------------------------------------------------------


def isolate_host(tmp_path: Path, monkeypatch: Any) -> Dict[str, Path]:
    """Point every store the models code reads at tmp_path.

    PATH keeps only system dirs (no /usr/local/bin, no Homebrew) plus a fake
    bin dir, so a real `ollama`/`lms`/`brew` on the developer's machine is
    never invoked.
    """

    home = tmp_path / "home"
    home.mkdir(parents=True, exist_ok=True)
    hf = home / ".cache" / "huggingface" / "hub"
    hf.mkdir(parents=True)
    lms_models = home / ".lmstudio" / "models"
    lms_models.mkdir(parents=True)
    ollama_models = home / ".ollama" / "models"
    ollama_models.mkdir(parents=True)
    config = tmp_path / "config"
    config.mkdir()
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HF_HUB_CACHE", str(hf))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_CACHE", raising=False)
    monkeypatch.delenv("DIFFUSERS_CACHE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("LMSTUDIO_MODELS_DIR", str(lms_models))
    monkeypatch.setenv("OLLAMA_MODELS", str(ollama_models))
    monkeypatch.setenv("ABSTRACTCORE_CONFIG_DIR", str(config))
    monkeypatch.setenv("ABSTRACTCORE_JOBS_DIR", str(config / "jobs"))
    # A port nothing listens on: an unreachable engine is the default state.
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("ABSTRACTCORE_LMS_CLI", str(fakebin / "lms-not-installed"))
    monkeypatch.delenv("ABSTRACTCORE_ALLOW_ENGINE_INSTALL", raising=False)
    monkeypatch.setenv("PATH", os.pathsep.join([str(fakebin), "/usr/bin", "/bin", "/usr/sbin", "/sbin"]))

    from abstractcore.config import engines, host_jobs

    engines._reset_caches_for_tests()
    host_jobs.set_default_registry(host_jobs.HostJobRegistry(persist_dir=config / "jobs"))
    return {
        "home": home,
        "hf": hf,
        "lms_models": lms_models,
        "ollama_models": ollama_models,
        "config": config,
        "fakebin": fakebin,
    }


def synthetic_host(kind: str, *, disk_free: int = 500 * 10**9) -> Dict[str, Any]:
    """host_profile_v1 dicts for the reference machines (+ any `metal<GiB>`)."""

    GiB = 1024**3
    disk = {
        name: {"path": f"~/{name}", "abs_path": f"/tmp/{name}", "exists": True, "free_bytes": disk_free}
        for name in ("hf_cache", "lmstudio", "ollama")
    }
    base = {
        "schema": "host_profile_v1",
        "python": "3.12.0",
        "cpu_count": 8,
        "notes": [],
        "disk": disk,
        "generated_at": "2026-09-23T00:00:00Z",
        "gpu_count": 1,
    }
    if kind == "metal128":
        return dict(
            base,
            os="darwin",
            arch="arm64",
            accelerator="metal",
            gpu_name="Apple M5 Max",
            unified_memory=True,
            ram_bytes=128 * GiB,
            vram_bytes=None,
            ceiling_bytes=96 * GiB,
            ceiling_source="ram_75pct",
            free_now_bytes=60 * GiB,
        )
    if kind == "cuda24":
        return dict(
            base,
            os="linux",
            arch="x86_64",
            accelerator="cuda",
            gpu_name="NVIDIA RTX 4090",
            unified_memory=False,
            ram_bytes=64 * GiB,
            vram_bytes=24 * GiB,
            ceiling_bytes=24 * GiB,
            ceiling_source="cuda_total",
            free_now_bytes=22 * GiB,
        )
    if kind == "cpu16":
        return dict(
            base,
            os="linux",
            arch="x86_64",
            accelerator="none",
            gpu_name=None,
            gpu_count=0,
            unified_memory=False,
            ram_bytes=16 * GiB,
            vram_bytes=None,
            ceiling_bytes=12 * GiB,
            ceiling_source="ram_75pct",
            free_now_bytes=8 * GiB,
        )
    if kind.startswith("metal"):
        # Any Apple silicon size: `metal24`, `metal23.9`, `metal192` (GiB of
        # unified memory). Ceiling = 75% (the host probe's fallback basis),
        # free now = half.
        ram = int(float(kind[len("metal"):]) * GiB)
        return dict(
            base,
            os="darwin",
            arch="arm64",
            accelerator="metal",
            gpu_name="Apple M-series",
            unified_memory=True,
            ram_bytes=ram,
            vram_bytes=None,
            ceiling_bytes=int(0.75 * ram),
            ceiling_source="ram_75pct",
            free_now_bytes=ram // 2,
        )
    if kind == "rocm32":
        return dict(
            base,
            os="linux",
            arch="x86_64",
            accelerator="rocm",
            gpu_name=None,
            unified_memory=False,
            ram_bytes=32 * GiB,
            vram_bytes=None,
            ceiling_bytes=24 * GiB,
            ceiling_source="ram_75pct",
            free_now_bytes=16 * GiB,
        )
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# Fake Ollama
# ---------------------------------------------------------------------------


def ollama_tag(name: str, size: int, parameter_size: str, quant: str, family: str = "qwen3") -> Dict[str, Any]:
    """One `/api/tags` row with Ollama's documented field names."""

    return {
        "name": name,
        "model": name,
        "modified_at": "2026-09-01T10:00:00.000000+02:00",
        "size": size,
        "digest": hashlib.sha256(name.encode()).hexdigest(),
        "details": {
            "parent_model": "",
            "format": "gguf",
            "family": family,
            "families": [family],
            "parameter_size": parameter_size,
            "quantization_level": quant,
        },
    }


class FakeOllama:
    def __init__(self, models: Optional[List[Dict[str, Any]]] = None, loaded: Optional[List[str]] = None):
        self.models = list(models or [])
        self.loaded = list(loaded or [])
        self.requests: List[Dict[str, Any]] = []
        self.pull_chunks = 5
        self.pull_delay = 0.0
        self.pull_error: Optional[str] = None
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:  # silence
                pass

            def _json(self, code: int, payload: Any) -> None:
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self) -> Dict[str, Any]:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    return json.loads(raw or b"{}")
                except Exception:
                    return {}

            def do_GET(self) -> None:
                fake.requests.append({"method": "GET", "path": self.path})
                if self.path == "/api/tags":
                    return self._json(200, {"models": fake.models})
                if self.path == "/api/ps":
                    rows = [m for m in fake.models if m["name"] in fake.loaded]
                    return self._json(200, {"models": rows})
                if self.path == "/api/version":
                    return self._json(200, {"version": "0.20.2"})
                return self._json(404, {"error": "not found"})

            def do_DELETE(self) -> None:
                body = self._body()
                fake.requests.append({"method": "DELETE", "path": self.path, "body": body})
                name = body.get("model") or body.get("name")
                before = len(fake.models)
                fake.models = [m for m in fake.models if m["name"] != name]
                if len(fake.models) == before:
                    return self._json(404, {"error": f"model '{name}' not found"})
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_POST(self) -> None:
                body = self._body()
                fake.requests.append({"method": "POST", "path": self.path, "body": body})
                if self.path == "/api/generate":
                    if body.get("keep_alive") == 0 and body.get("model") in fake.loaded:
                        fake.loaded.remove(body["model"])
                    return self._json(200, {"done": True})
                if self.path == "/api/pull":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    name = body.get("name") or body.get("model")
                    if fake.pull_error:
                        self.wfile.write((json.dumps({"error": fake.pull_error}) + "\n").encode())
                        return
                    total = 1000
                    self.wfile.write((json.dumps({"status": "pulling manifest"}) + "\n").encode())
                    for i in range(1, fake.pull_chunks + 1):
                        line = {"status": "pulling 3f2b", "digest": "sha256:3f2b", "total": total, "completed": total * i // fake.pull_chunks}
                        try:
                            self.wfile.write((json.dumps(line) + "\n").encode())
                            self.wfile.flush()
                        except Exception:
                            return
                        if fake.pull_delay:
                            time.sleep(fake.pull_delay)
                    self.wfile.write((json.dumps({"status": "verifying sha256 digest"}) + "\n").encode())
                    self.wfile.write((json.dumps({"status": "success"}) + "\n").encode())
                    fake.models.append(ollama_tag(name, 1000, "8.2B", "Q4_K_M"))
                    return
                return self._json(404, {"error": "not found"})

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "FakeOllama":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()


# ---------------------------------------------------------------------------
# Fake `lms`
# ---------------------------------------------------------------------------

REAL_SHAPED_LMS_ROWS: List[Dict[str, Any]] = [
    {
        "type": "llm",
        "modelKey": "qwen/qwen3.8-27b",
        "format": "gguf",
        "displayName": "Qwen3.8 27B",
        "publisher": "qwen",
        "path": "qwen/qwen3.8-27b",
        "sizeBytes": 17742039110,
        "indexedModelIdentifier": "qwen/qwen3.8-27b",
        "deviceIdentifier": None,
        "paramsString": "27B",
        "architecture": "qwen35",
        "quantization": {"name": "Q4_K_M", "bits": 4},
        "variants": ["qwen/qwen3.8-27b@q4_k_m"],
        "selectedVariant": "qwen/qwen3.8-27b@q4_k_m",
        "vision": True,
        "trainedForToolUse": True,
        "maxContextLength": 262144,
    },
    {
        "type": "llm",
        "modelKey": "llama-3.2-1b-instruct",
        "format": "safetensors",
        "displayName": "Llama 3.2 1B Instruct",
        "publisher": "mlx-community",
        "path": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "sizeBytes": 712575975,
        "indexedModelIdentifier": "mlx-community/Llama-3.2-1B-Instruct-4bit",
        "deviceIdentifier": None,
        "paramsString": "1B",
        "architecture": "llama",
        "quantization": {"name": "4bit", "bits": 4},
        "vision": False,
        "trainedForToolUse": True,
        "maxContextLength": 131072,
    },
    {
        "type": "embedding",
        "modelKey": "text-embedding-qwen3-embedding-0.6b",
        "format": "gguf",
        "displayName": "Qwen3 Embedding 0.6B",
        "publisher": "Qwen",
        "path": "Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
        "sizeBytes": 639150592,
        "indexedModelIdentifier": "Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
        "deviceIdentifier": None,
        "paramsString": "0.6B",
        "architecture": "qwen3",
        "quantization": {"name": "Q8_0", "bits": 8},
        "maxContextLength": 32768,
    },
    {
        "type": "embedding",
        "modelKey": "text-embedding-nomic-embed-text-v1.5",
        "format": "gguf",
        "displayName": "Nomic Embed Text v1.5",
        "publisher": "nomic-ai",
        "path": "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf",
        "sizeBytes": 84106624,
        "indexedModelIdentifier": "nomic-ai/nomic-embed-text-v1.5-GGUF/nomic-embed-text-v1.5.Q4_K_M.gguf",
        "deviceIdentifier": None,
        "architecture": "nomic-bert",
        "quantization": {"name": "Q4_K_M", "bits": 4},
        "maxContextLength": 2048,
    },
]

_FAKE_LMS = r'''#!{python}
import json, os, sys
state_path = os.environ.get("FAKE_LMS_STATE") or {state!r}
state = json.load(open(state_path))
args = sys.argv[1:]
log = state.setdefault("calls", [])
log.append(args)
json.dump(state, open(state_path, "w"))
if args[:2] == ["ls", "--json"]:
    print(json.dumps(state["ls"]))
elif args[:2] == ["ps", "--json"]:
    print(json.dumps(state.get("ps", [])))
elif args[:1] == ["unload"]:
    state["ps"] = [r for r in state.get("ps", []) if r.get("modelKey") != args[1]]
    json.dump(state, open(state_path, "w"))
    print("Model unloaded")
elif args[:1] == ["version"]:
    print("lms is LM Studio's CLI utility for your models, server, and inference runtime.")
    print("CLI commit: 71bd99c")
elif args[:1] == ["get"]:
    print("Searching for models with the term " + args[1])
    print("Downloading " + args[1])
    print("Download completed")
    if state.get("get_adds"):
        state["ls"].append(state["get_adds"])
        json.dump(state, open(state_path, "w"))
else:
    print("unknown command", file=sys.stderr)
    sys.exit(1)
'''


def install_fake_lms(fakebin: Path, monkeypatch: Any, rows: List[Dict[str, Any]], ps: Optional[List[Dict[str, Any]]] = None) -> Path:
    """Put a `lms` on PATH (and in ABSTRACTCORE_LMS_CLI). Returns the state file."""

    state = fakebin / "lms_state.json"
    state.write_text(json.dumps({"ls": rows, "ps": ps or []}))
    exe = fakebin / "lms"
    exe.write_text(_FAKE_LMS.replace("{python}", sys.executable).replace("{state!r}", repr(str(state))))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    monkeypatch.setenv("ABSTRACTCORE_LMS_CLI", str(exe))
    monkeypatch.setenv("FAKE_LMS_STATE", str(state))
    return state


def lms_calls(state: Path) -> List[List[str]]:
    return json.loads(state.read_text()).get("calls", [])


def install_fake_tool(fakebin: Path, name: str, script: str) -> Path:
    exe = fakebin / name
    exe.write_text(f"#!{sys.executable}\n{script}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


# ---------------------------------------------------------------------------
# Fake Hugging Face cache + API
# ---------------------------------------------------------------------------


def make_hf_repo(cache: Path, repo_id: str, files: Dict[str, bytes], *, revision: str = "a" * 40, incomplete: int = 0) -> Path:
    """A cache entry the way huggingface_hub lays it out."""

    folder = cache / ("models--" + repo_id.replace("/", "--"))
    blobs = folder / "blobs"
    snap = folder / "snapshots" / revision
    blobs.mkdir(parents=True, exist_ok=True)
    snap.mkdir(parents=True, exist_ok=True)
    (folder / "refs").mkdir(exist_ok=True)
    (folder / "refs" / "main").write_text(revision)
    for name, data in files.items():
        digest = hashlib.sha256(data).hexdigest()
        blob = blobs / digest
        blob.write_bytes(data)
        target = snap / name
        target.parent.mkdir(parents=True, exist_ok=True)
        rel = os.path.relpath(blob, target.parent)
        if target.exists() or target.is_symlink():
            target.unlink()
        os.symlink(rel, target)
    for i in range(incomplete):
        (blobs / f"{'f' * 63}{i}.incomplete").write_bytes(b"x" * 10)
    return folder


class HubRateLimited(Exception):
    def __init__(self) -> None:
        super().__init__("429 Client Error: Too Many Requests")
        self.response = SimpleNamespace(status_code=429)


class FakeHfApi:
    def __init__(self, repos: Optional[Dict[str, Dict[str, Any]]] = None, search: Optional[Dict[str, List[str]]] = None):
        self.repos = dict(repos or {})
        self.search_results = dict(search or {})
        self.calls: List[tuple] = []
        self.fail_with: Optional[BaseException] = None

    def model_info(self, repo_id: str, files_metadata: bool = False, **_: Any) -> Any:
        self.calls.append(("model_info", repo_id))
        if self.fail_with is not None:
            raise self.fail_with
        if repo_id not in self.repos:
            raise LookupError(f"404 {repo_id}")
        spec = self.repos[repo_id]
        siblings = [SimpleNamespace(rfilename=n, size=s) for n, s in spec.get("files", {}).items()]
        st = SimpleNamespace(total=spec["params"]) if spec.get("params") else None
        return SimpleNamespace(id=repo_id, siblings=siblings, safetensors=st, gguf=None, gated=False)

    def list_models(self, search: str = "", filter: Any = None, sort: Any = None, limit: Any = None, **_: Any) -> Any:
        self.calls.append(("list_models", search, filter))
        if self.fail_with is not None:
            raise self.fail_with
        return [SimpleNamespace(id=r, downloads=1000) for r in self.search_results.get(f"{filter}:{search}", [])]
