"""
File "memory blocs": persistent extracted-text snapshots + per-model KV artifacts.

A "bloc" is an atomic, content-addressed unit of extracted file text stored on disk.

This module provides:
- A directory-backed store for extracted file text snapshots (`content.txt`) and metadata (`meta.json`).
- Optional JSON-LD metadata (`meta.jsonld`) for richer indexing/cataloging.
- A stable path convention for per-(provider, model) KV/prefix artifacts under `kv/`.

It intentionally stays at the storage/metadata layer; higher-level apps can compile/read KV
artifacts using provider-specific mechanisms.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import shutil
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


def default_blocs_root_dir() -> Path:
    return Path.home() / ".abstractcore" / "blocs"


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "will",
    "with",
}


def _first_nonempty_line(text: str) -> str:
    for line in (text or "").splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def heuristic_summary_and_keywords(text: str, *, max_keywords: int = 5) -> Tuple[str, List[str]]:
    """Fast, dependency-free metadata generation (best-effort)."""
    summary = _first_nonempty_line(text)
    if len(summary) > 180:
        summary = summary[:177].rstrip() + "..."

    words = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text or "")
    counts: Dict[str, int] = {}
    for w in words:
        key = w.lower()
        if key in _STOPWORDS:
            continue
        if len(key) < 4:
            continue
        counts[key] = counts.get(key, 0) + 1

    keywords = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    out = [k for k, _v in keywords[: max(0, int(max_keywords))]]
    return summary, out


def _safe_relpath(path: Path, *, base: Optional[Path]) -> Optional[str]:
    if base is None:
        return None
    try:
        rel = path.resolve().relative_to(base.resolve())
    except Exception:
        return None
    return str(rel)


def _slug_provider_model(provider_name: str, model_id: str) -> str:
    raw = f"{provider_name}__{model_id}".strip()
    import hashlib

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class FileBlocRecord:
    """A persisted, content-addressed file bloc record."""

    sha256: str
    bloc_id: Optional[int]
    path: str
    relpath: Optional[str]
    filename: str
    size_bytes: int
    mtime_ns: int
    media_type: str
    content_sha256: str
    content_length: int
    format: Optional[str] = None
    estimated_tokens: Optional[int] = None
    summary: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sha256": self.sha256,
            "bloc_id": int(self.bloc_id) if isinstance(self.bloc_id, int) and self.bloc_id > 0 else None,
            "path": self.path,
            "relpath": self.relpath,
            "filename": self.filename,
            "size_bytes": int(self.size_bytes),
            "mtime_ns": int(self.mtime_ns),
            "media_type": self.media_type,
            "content_sha256": self.content_sha256,
            "content_length": int(self.content_length),
            "format": self.format,
            "estimated_tokens": int(self.estimated_tokens) if isinstance(self.estimated_tokens, int) else None,
            "summary": self.summary,
            "keywords": list(self.keywords or ()),
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FileBlocRecord":
        bid_raw = data.get("bloc_id")
        bid: Optional[int] = None
        try:
            bid_i = int(bid_raw) if bid_raw is not None else None
        except Exception:
            bid_i = None
        if isinstance(bid_i, int) and bid_i > 0:
            bid = bid_i
        return cls(
            sha256=str(data.get("sha256") or ""),
            bloc_id=bid,
            path=str(data.get("path") or ""),
            relpath=str(data.get("relpath"))
            if isinstance(data.get("relpath"), str) and data.get("relpath")
            else None,
            filename=str(data.get("filename") or ""),
            size_bytes=int(data.get("size_bytes") or 0),
            mtime_ns=int(data.get("mtime_ns") or 0),
            media_type=str(data.get("media_type") or ""),
            content_sha256=str(data.get("content_sha256") or ""),
            content_length=int(data.get("content_length") or 0),
            format=str(data.get("format")) if isinstance(data.get("format"), str) and data.get("format") else None,
            estimated_tokens=int(data.get("estimated_tokens"))
            if isinstance(data.get("estimated_tokens"), int)
            else None,
            summary=str(data.get("summary")) if isinstance(data.get("summary"), str) and data.get("summary") else None,
            keywords=[str(k) for k in (data.get("keywords") or []) if isinstance(k, str)],
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0),
        )


class FileBlocStore:
    """A simple directory-backed store for file bloc records and artifacts."""

    def __init__(self, *, root_dir: Optional[Path] = None) -> None:
        self.root_dir = (root_dir or default_blocs_root_dir()).expanduser()
        self.files_dir = self.root_dir / "files"
        self._bloc_ids_loaded: bool = False
        self._sha_to_bloc_id: Dict[str, int] = {}
        self._bloc_id_to_sha: Dict[int, str] = {}
        self._next_bloc_id: Optional[int] = None

    def _bloc_dir(self, sha256: str) -> Path:
        sha = str(sha256 or "").strip().lower()
        return self.files_dir / sha

    def content_path(self, sha256: str) -> Path:
        return self._bloc_dir(sha256) / "content.txt"

    def meta_path(self, sha256: str) -> Path:
        return self._bloc_dir(sha256) / "meta.json"

    def meta_jsonld_path(self, sha256: str) -> Path:
        return self._bloc_dir(sha256) / "meta.jsonld"

    def kv_cache_path(self, sha256: str, *, provider: str, model: str, ext: str = ".safetensors") -> Path:
        slug = _slug_provider_model(str(provider or ""), str(model or ""))
        return self._bloc_dir(sha256) / "kv" / f"{slug}{ext}"

    def kv_cache_manifest_path(self, sha256: str, *, provider: str, model: str) -> Path:
        slug = _slug_provider_model(str(provider or ""), str(model or ""))
        return self._bloc_dir(sha256) / "kv" / f"{slug}.manifest.json"

    def bloc_id_counter_path(self) -> Path:
        """Path for stable, monotonic bloc ids (useful for durable selectors)."""
        return self.root_dir / "bloc_ids.json"

    def _bloc_id_lock_path(self) -> Path:
        return self.root_dir / ".bloc_ids.lock"

    @contextmanager
    def _bloc_id_lock(self, *, timeout_s: float = 5.0) -> Iterator[None]:
        """Best-effort cross-process lock via an exclusive lock file."""
        try:
            self.root_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        lock = self._bloc_id_lock_path()
        t0 = time.time()
        fd: Optional[int] = None
        while True:
            try:
                fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                try:
                    os.write(fd, f"{os.getpid()} {time.time():.3f}\n".encode("utf-8"))
                except Exception:
                    pass
                break
            except FileExistsError:
                if (time.time() - t0) >= float(timeout_s):
                    raise TimeoutError(f"timed out waiting for bloc id lock: {lock}")
                time.sleep(0.05)
            except Exception:
                # If the lock cannot be created, proceed without it.
                fd = None
                break
        try:
            yield
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
                try:
                    lock.unlink()
                except Exception:
                    pass

    def _read_next_bloc_id(self) -> int:
        p = self.bloc_id_counter_path()
        try:
            if not p.exists():
                return 1
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return 1
            v = data.get("next_id")
            n = int(v) if v is not None else 1
            return max(1, n)
        except Exception:
            return 1

    def _write_next_bloc_id(self, next_id: int) -> None:
        n = max(1, int(next_id))
        p = self.bloc_id_counter_path()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": 1, "next_id": n}, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)

    def ensure_bloc_ids(self) -> int:
        """Assign stable integer ids to all blocs (best-effort)."""
        if self._bloc_ids_loaded:
            return 0

        sha_dir_re = re.compile(r"^[0-9a-f]{64}$")
        assigned = 0

        with self._bloc_id_lock():
            self._sha_to_bloc_id = {}
            self._bloc_id_to_sha = {}
            max_id = 0
            missing: List[Tuple[float, str, Path]] = []

            try:
                if self.files_dir.exists():
                    for child in self.files_dir.iterdir():
                        if not child.is_dir():
                            continue
                        sha = str(child.name or "").strip().lower()
                        if not sha_dir_re.fullmatch(sha):
                            continue
                        mp = child / "meta.json"
                        if not mp.exists():
                            continue
                        try:
                            data = json.loads(mp.read_text(encoding="utf-8"))
                        except Exception:
                            continue
                        if not isinstance(data, dict):
                            continue
                        rec = FileBlocRecord.from_dict(data)
                        if not rec.sha256:
                            continue

                        bid = rec.bloc_id if isinstance(rec.bloc_id, int) and rec.bloc_id > 0 else None
                        if bid is not None:
                            if bid in self._bloc_id_to_sha and self._bloc_id_to_sha[bid] != rec.sha256:
                                bid = None
                            else:
                                self._sha_to_bloc_id[rec.sha256] = int(bid)
                                self._bloc_id_to_sha[int(bid)] = rec.sha256
                                max_id = max(max_id, int(bid))

                        if bid is None:
                            missing.append((float(rec.created_at or 0.0), rec.sha256, mp))
            except Exception:
                missing = []

            next_id = max(self._read_next_bloc_id(), int(max_id) + 1)
            if missing:
                missing.sort(key=lambda t: (t[0], t[1]))
                for _created_at, sha, mp in missing:
                    if sha in self._sha_to_bloc_id:
                        continue
                    bid = int(next_id)
                    next_id += 1
                    try:
                        data = json.loads(mp.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    if not isinstance(data, dict):
                        continue
                    data["bloc_id"] = bid
                    try:
                        mp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                    except Exception:
                        continue
                    self._sha_to_bloc_id[sha] = bid
                    self._bloc_id_to_sha[bid] = sha
                    assigned += 1

            try:
                self._write_next_bloc_id(int(next_id))
            except Exception:
                pass
            self._next_bloc_id = int(next_id)

        self._bloc_ids_loaded = True
        return int(assigned)

    def bloc_id(self, sha256: str) -> Optional[int]:
        sha = str(sha256 or "").strip().lower()
        if not sha:
            return None
        self.ensure_bloc_ids()
        bid = self._sha_to_bloc_id.get(sha)
        return int(bid) if isinstance(bid, int) and bid > 0 else None

    def sha_for_bloc_id(self, bloc_id: int) -> Optional[str]:
        try:
            bid = int(bloc_id)
        except Exception:
            return None
        if bid <= 0:
            return None
        self.ensure_bloc_ids()
        sha = self._bloc_id_to_sha.get(int(bid))
        return str(sha) if isinstance(sha, str) and sha else None

    def get_by_bloc_id(self, bloc_id: int) -> Optional[FileBlocRecord]:
        sha = self.sha_for_bloc_id(bloc_id)
        return self.get(sha) if sha else None

    def _allocate_bloc_id_for_sha(self, sha: str) -> Optional[int]:
        """Allocate and persist a new bloc id for `sha` (requires ids loaded)."""
        if not self._bloc_ids_loaded:
            return None
        s = str(sha or "").strip().lower()
        if not s:
            return None
        with self._bloc_id_lock():
            existing = self._sha_to_bloc_id.get(s)
            if isinstance(existing, int) and existing > 0:
                return int(existing)
            next_id = self._next_bloc_id if isinstance(self._next_bloc_id, int) and self._next_bloc_id > 0 else None
            if next_id is None:
                next_id = self._read_next_bloc_id()
            bid = int(next_id)
            next_id = bid + 1
            try:
                self._write_next_bloc_id(int(next_id))
            except Exception:
                pass
            self._next_bloc_id = int(next_id)
            self._sha_to_bloc_id[s] = bid
            self._bloc_id_to_sha[bid] = s
            return bid

    def upsert(
        self,
        *,
        file_meta: Dict[str, Any],
        content: str,
        relpath_base: Optional[Path] = None,
        summary: Optional[str] = None,
        keywords: Optional[Sequence[str]] = None,
    ) -> FileBlocRecord:
        """Create/update a record and persist its extracted content (best-effort)."""
        sha = str(file_meta.get("sha256") or "").strip().lower()
        if not sha:
            raise ValueError("file_meta.sha256 is required")

        path = str(file_meta.get("path") or "")
        filename = Path(path).name if path else ""
        relpath = _safe_relpath(Path(path), base=relpath_base) if path else None

        size_bytes = int(file_meta.get("size_bytes") or 0)
        mtime_ns = int(file_meta.get("mtime_ns") or 0)
        media_type = str(file_meta.get("media_type") or "")
        content_sha256 = str(file_meta.get("content_sha256") or "")
        fmt = str(file_meta.get("format")) if isinstance(file_meta.get("format"), str) and file_meta.get("format") else None
        estimated_tokens = int(file_meta.get("estimated_tokens")) if isinstance(file_meta.get("estimated_tokens"), int) else None
        content_length = int(file_meta.get("content_length") or len(content or ""))

        if summary is None or keywords is None:
            auto_summary, auto_keywords = heuristic_summary_and_keywords(content)
            if summary is None:
                summary = auto_summary or None
            if keywords is None:
                keywords = auto_keywords

        now = time.time()
        existing = self.get(sha)
        created_at = existing.created_at if existing else now
        bloc_id = existing.bloc_id if existing else None
        if bloc_id is None and self._bloc_ids_loaded:
            bloc_id = self._allocate_bloc_id_for_sha(sha)

        record = FileBlocRecord(
            sha256=sha,
            bloc_id=bloc_id,
            path=path,
            relpath=relpath,
            filename=filename,
            size_bytes=size_bytes,
            mtime_ns=mtime_ns,
            media_type=media_type,
            content_sha256=content_sha256,
            content_length=content_length,
            format=fmt,
            estimated_tokens=estimated_tokens,
            summary=str(summary) if isinstance(summary, str) and summary else None,
            keywords=[str(k) for k in (keywords or []) if isinstance(k, str)],
            created_at=float(created_at),
            updated_at=float(now),
        )

        bloc_dir = self._bloc_dir(sha)
        (bloc_dir / "kv").mkdir(parents=True, exist_ok=True)
        self.content_path(sha).write_text(str(content or ""), encoding="utf-8")
        self.meta_path(sha).write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    def get(self, sha256: str) -> Optional[FileBlocRecord]:
        path = self.meta_path(sha256)
        try:
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return None
            rec = FileBlocRecord.from_dict(data)
            return rec if rec.sha256 else None
        except Exception:
            return None

    def get_jsonld(self, sha256: str) -> Optional[Dict[str, Any]]:
        path = self.meta_jsonld_path(sha256)
        try:
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception:
            return None

    def write_jsonld(self, sha256: str, data: Dict[str, Any]) -> bool:
        """Persist JSON-LD bloc metadata (compact, single-line JSON)."""
        if not isinstance(data, dict):
            return False
        path = self.meta_jsonld_path(sha256)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            path.write_text(payload, encoding="utf-8")
            return True
        except Exception:
            return False

    def patch_jsonld_usage(
        self,
        sha256: str,
        *,
        access_count: Optional[int] = None,
        modified_count: Optional[int] = None,
    ) -> bool:
        """Update usage counters within `meta.jsonld` (best-effort)."""
        meta = self.get_jsonld(sha256)
        if not isinstance(meta, dict) or not meta:
            return False
        use = meta.get("use")
        if not isinstance(use, dict):
            use = {}
        changed = False

        if access_count is not None:
            try:
                v = int(access_count)
                if v < 0:
                    v = 0
                if use.get("acc") != v:
                    use["acc"] = v
                    changed = True
            except Exception:
                pass

        if modified_count is not None:
            try:
                v = int(modified_count)
                if v < 0:
                    v = 0
                if use.get("mcount") != v:
                    use["mcount"] = v
                    changed = True
            except Exception:
                pass

        if not changed:
            return True
        meta["use"] = use
        return bool(self.write_jsonld(sha256, meta))

    def patch_record(
        self,
        sha256: str,
        *,
        summary: Optional[str] = None,
        keywords: Optional[Sequence[str]] = None,
    ) -> Optional[FileBlocRecord]:
        """Update lightweight fields on an existing record (best-effort)."""
        rec = self.get(sha256)
        if rec is None or not rec.sha256:
            return None
        now = time.time()
        new_rec = FileBlocRecord(
            sha256=rec.sha256,
            bloc_id=rec.bloc_id,
            path=rec.path,
            relpath=rec.relpath,
            filename=rec.filename,
            size_bytes=rec.size_bytes,
            mtime_ns=rec.mtime_ns,
            media_type=rec.media_type,
            content_sha256=rec.content_sha256,
            content_length=rec.content_length,
            format=rec.format,
            estimated_tokens=rec.estimated_tokens,
            summary=str(summary).strip() if isinstance(summary, str) and str(summary).strip() else rec.summary,
            keywords=[str(k) for k in (keywords or rec.keywords or []) if isinstance(k, str) and str(k).strip()],
            created_at=rec.created_at,
            updated_at=float(now),
        )
        try:
            self.meta_path(sha256).write_text(
                json.dumps(new_rec.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return None
        return new_rec

    def list(self) -> List[FileBlocRecord]:
        out: List[FileBlocRecord] = []
        try:
            if not self.files_dir.exists():
                return []
            for child in sorted(self.files_dir.iterdir()):
                if not child.is_dir():
                    continue
                rec = self.get(child.name)
                if rec is not None:
                    out.append(rec)
        except Exception:
            return out
        return out

    def list_kv_artifacts(
        self,
        *,
        sha256: Optional[str] = None,
        bloc_id: Optional[int] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List manifest-backed KV artifacts with lightweight filters."""
        records: List[FileBlocRecord] = []
        if isinstance(sha256, str) and sha256.strip():
            rec = self.get(sha256.strip().lower())
            if rec is not None:
                records = [rec]
        elif bloc_id is not None:
            rec = self.get_by_bloc_id(bloc_id)
            if rec is not None:
                records = [rec]
        else:
            records = self.list()

        provider_s = str(provider or "").strip().lower()
        model_s = str(model or "").strip()
        out: List[Dict[str, Any]] = []
        for rec in records:
            kv_dir = self._bloc_dir(rec.sha256) / "kv"
            try:
                manifest_paths = sorted(kv_dir.glob("*.manifest.json"))
            except Exception:
                manifest_paths = []
            for manifest_path in manifest_paths:
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                if provider_s and str(data.get("provider") or "").strip().lower() != provider_s:
                    continue
                if model_s and str(data.get("model") or "").strip() != model_s:
                    continue
                artifact_filename = str(data.get("artifact_filename") or "").strip()
                artifact_path = manifest_path.with_name(artifact_filename) if artifact_filename else None
                artifact_exists = bool(artifact_path and artifact_path.exists())
                out.append(
                    {
                        "record": rec.to_dict(),
                        "manifest": dict(data),
                        "manifest_path": str(manifest_path),
                        "artifact_path": str(artifact_path) if artifact_path is not None else None,
                        "artifact_exists": artifact_exists,
                        "artifact_size_bytes": int(artifact_path.stat().st_size) if artifact_exists and artifact_path is not None else None,
                    }
                )
        return out

    def delete_kv_artifact_paths(self, *, artifact_path: Optional[Path], manifest_path: Optional[Path]) -> Dict[str, Any]:
        deleted: List[str] = []
        missing: List[str] = []
        for path in (artifact_path, manifest_path):
            if path is None:
                continue
            try:
                if path.exists():
                    path.unlink()
                    deleted.append(str(path))
                else:
                    missing.append(str(path))
            except FileNotFoundError:
                missing.append(str(path))
        return {"deleted_paths": deleted, "missing_paths": missing}

    def delete(self, sha256: str, *, delete_kv: bool = True) -> Dict[str, Any]:
        """Delete one bloc directory from disk."""
        rec = self.get(sha256)
        if rec is None:
            raise ValueError("bloc not found")
        bloc_dir = self._bloc_dir(rec.sha256)
        if not delete_kv:
            for path in (self.content_path(rec.sha256), self.meta_path(rec.sha256), self.meta_jsonld_path(rec.sha256)):
                try:
                    if path.exists():
                        path.unlink()
                except FileNotFoundError:
                    pass
            try:
                if bloc_dir.exists() and not any(bloc_dir.iterdir()):
                    bloc_dir.rmdir()
            except Exception:
                pass
        else:
            shutil.rmtree(bloc_dir, ignore_errors=False)
        try:
            self._bloc_ids_loaded = False
            self.ensure_bloc_ids()
        except Exception:
            pass
        return {"record": rec.to_dict(), "deleted_path": str(bloc_dir), "delete_kv": bool(delete_kv)}

    def has_kv_cache(self, sha256: str, *, provider: str, model: str) -> bool:
        try:
            return self.kv_cache_path(sha256, provider=provider, model=model).exists()
        except Exception:
            return False

    def has_kv_cache_manifest(self, sha256: str, *, provider: str, model: str) -> bool:
        try:
            return self.kv_cache_manifest_path(sha256, provider=provider, model=model).exists()
        except Exception:
            return False
