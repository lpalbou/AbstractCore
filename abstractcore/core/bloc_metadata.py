"""
Bloc metadata helpers (JSON-LD).

This module defines a compact JSON-LD schema for file blocs and provides a best-effort
metadata generator/parser. It is designed for:
- Robust parsing of imperfect model JSON outputs (truncation, fences, reasoning tags).
- Persisting metadata alongside `FileBlocStore` as `meta.jsonld`.

The generator can optionally reuse an existing prompt-cache/KV artifact (via a provider
prompt-cache control plane) so metadata backfill can be done without re-prefilling long content.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid
from typing import Any, Dict, Optional, Sequence, Tuple

from .file_blocs import FileBlocRecord, FileBlocStore


def _assets_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "assets"


_SCHEMA_CACHE: Optional[Dict[str, Any]] = None


def load_bloc_metadata_schema(*, filename: str = "bloc-schema.jsonld") -> Dict[str, Any]:
    """Load the bloc metadata schema from `abstractcore/assets/`."""
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return dict(_SCHEMA_CACHE)
    path = _assets_dir() / filename
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("@context"), dict):
        raise ValueError(f"invalid bloc metadata schema: {path}")
    _SCHEMA_CACHE = dict(data)
    return dict(data)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    try:
        return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()
    except Exception:
        return None


def _ts_iso(ts: Any) -> Optional[str]:
    try:
        t = float(ts)
    except Exception:
        return None
    if t <= 0:
        return None
    return _iso(datetime.fromtimestamp(t, tz=timezone.utc))


def _first_balanced_json_object(s: str) -> Optional[str]:
    """Return the first balanced {...} substring, respecting JSON strings."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escaped = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract a plausible metadata JSON object from a model response (best-effort)."""
    s = str(text or "").strip()
    if not s:
        return None

    def _looks_like_metadata_payload(obj: Dict[str, Any]) -> bool:
        keys = set(str(k) for k in obj.keys())
        if "@context" in keys and "@graph" in keys:
            return False
        expected = {
            "t",
            "d",
            "kw",
            "tp",
            "kind",
            "mod",
            "lang",
            "q",
            "title",
            "description",
            "keywords",
            "topics",
            "type",
            "modality",
            "quality",
            "language",
        }
        return bool(keys & expected)

    # Strip common reasoning wrappers.
    s_lower = s.lower()
    if "<think>" in s_lower:
        try:
            s = re.sub(r"<think>.*?</think>", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
        except Exception:
            pass
        if "<think>" in s.lower() and "</think>" not in s.lower():
            brace = s.find("{")
            if brace >= 0:
                s = s[brace:].strip()

    # If the model used fenced output, try to extract the inner payload.
    if "```" in s:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, flags=re.IGNORECASE | re.DOTALL)
        if m and isinstance(m.group(1), str):
            s = m.group(1).strip()

    # Direct parse.
    try:
        data = json.loads(s)
        return data if isinstance(data, dict) and _looks_like_metadata_payload(data) else None
    except Exception:
        pass

    # Best-effort repair for malformed/truncated JSON.
    fix_json = None
    try:
        from abstractcore.utils.self_fixes import fix_json as _fix_json

        fix_json = _fix_json
    except Exception:
        fix_json = None
    if callable(fix_json):
        try:
            fixed = fix_json(s)
        except Exception:
            fixed = None
        if isinstance(fixed, str) and fixed.strip():
            try:
                data = json.loads(fixed)
                if isinstance(data, dict) and _looks_like_metadata_payload(data):
                    return data
            except Exception:
                pass

        for trim in range(1, 201):
            trimmed = s[:-trim].rstrip()
            if not trimmed:
                break
            try:
                fixed2 = fix_json(trimmed)
            except Exception:
                fixed2 = None
            if not (isinstance(fixed2, str) and fixed2.strip()):
                continue
            try:
                data = json.loads(fixed2)
            except Exception:
                continue
            if isinstance(data, dict) and _looks_like_metadata_payload(data):
                return data

    cand = _first_balanced_json_object(s)
    if not cand:
        return None
    try:
        data = json.loads(cand)
        return data if isinstance(data, dict) and _looks_like_metadata_payload(data) else None
    except Exception:
        pass

    if callable(fix_json):
        try:
            fixed3 = fix_json(cand)
        except Exception:
            fixed3 = None
        if isinstance(fixed3, str) and fixed3.strip():
            try:
                data = json.loads(fixed3)
                return data if isinstance(data, dict) and _looks_like_metadata_payload(data) else None
            except Exception:
                return None
    return None


def _normalize_kind(kind: Any, allowed: Sequence[str]) -> Optional[str]:
    s = str(kind or "").strip()
    if not s:
        return None
    if s in allowed:
        return s
    s2 = s.replace("schema:", "s:").replace("schema.org/", "s:").strip()
    if s2 in allowed:
        return s2
    if ":" not in s2:
        for prefix in ("s", "dcat", "ac"):
            cand = f"{prefix}:{s2}"
            if cand in allowed:
                return cand
        try:
            cand = f"s:{''.join(w[:1].upper()+w[1:] for w in s2.replace('_',' ').split())}"
            if cand in allowed:
                return cand
        except Exception:
            pass
    return None


def _normalize_modality(modality: Any, allowed: Sequence[str]) -> Optional[str]:
    s = str(modality or "").strip().lower()
    if not s:
        return None
    return s if s in allowed else None


def _normalize_lang(lang: Any) -> Optional[str]:
    s = str(lang or "").strip().lower()
    if not s:
        return None
    if len(s) > 15:
        return None
    return s


def _coerce_float01(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except Exception:
        return None
    if f != f:
        return None
    if f < 0.0:
        f = 0.0
    if f > 1.0:
        f = 1.0
    return float(f)


def _build_metadata_prompt(schema: Dict[str, Any], *, record: FileBlocRecord) -> str:
    enums = schema.get("enums") if isinstance(schema.get("enums"), dict) else {}
    allowed_mod = list(enums.get("mod") or [])
    allowed_kind = list(enums.get("kind") or [])
    limits = schema.get("limits") if isinstance(schema.get("limits"), dict) else {}
    qdims = schema.get("quality_dims") if isinstance(schema.get("quality_dims"), list) else []

    tmax = int(limits.get("t_max_chars") or 140)
    dmax = int(limits.get("d_max_chars") or 420)
    kwmax = int(limits.get("kw_max") or 16)
    tpmax = int(limits.get("tp_max") or 12)

    kinds = ", ".join([f'"{k}"' for k in allowed_kind]) if allowed_kind else ""
    mods = ", ".join([f'"{m}"' for m in allowed_mod]) if allowed_mod else ""

    # Heuristic hinting based on filename.
    hints: list[str] = []
    try:
        ext = Path(str(record.filename or "")).suffix.lower()
    except Exception:
        ext = ""
    if ext in {".py", ".js", ".ts", ".rs", ".go", ".java", ".cpp", ".c", ".h", ".hpp"}:
        hints.append("Suggested: kind=s:SoftwareSourceCode mod=code lang=en")
    elif ext in {".md", ".txt", ".rst"}:
        hints.append("Suggested: kind=s:Report mod=text lang=en")

    hint_block = ""
    if hints:
        hint_block = "\nHints:\n" + "\n".join(hints) + "\n"

    q_template = ",".join([f"\"{k}\":0.0" for k in qdims])
    template = (
        f'{{"t":"...","d":"...","kind":"{allowed_kind[0] if allowed_kind else "s:Report"}",'
        f'"mod":"{allowed_mod[0] if allowed_mod else "text"}","lang":"en","q":{{{q_template}}},'
        '"tp":["..."],"kw":["..."]}'
    )

    return (
        "You are generating metadata for ONE bloc of text already in context.\n"
        "Return EXACTLY one line of JSON (no markdown, no code fences).\n"
        "No analysis, no preamble, no <think>.\n"
        "Use exactly these keys: t, d, kind, mod, lang, q, tp, kw.\n"
        "The FIRST character of your response MUST be '{'.\n"
        "q MUST be an object with ALL quality keys: snr, clar, coh, conc, struct, arg, evid.\n"
        "q values MUST NOT all be identical.\n"
        f"kw MUST be an array of <= {kwmax} short strings; choose only the most salient.\n"
        f"tp MUST be an array of <= {tpmax} short strings; choose only the most salient.\n"
        "Fill this template (all keys required):\n"
        f"{template}\n"
        f"Rules: t<= {tmax} chars; d<= {dmax} chars; kw<= {kwmax}; tp<= {tpmax}; kind∈[{kinds}]; mod∈[{mods}]; lang='en' unless clearly not; q values 0..1.\n"
        f"{hint_block}\nJSON only.\n"
    )


@dataclass(frozen=True)
class BlocMetadataResult:
    ok: bool
    jsonld: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def generate_bloc_metadata_jsonld(
    *,
    provider: Any,
    model_id: str,
    stable_cache_key: Optional[str],
    kv_path: Optional[str],
    record: FileBlocRecord,
    store: FileBlocStore,
    schema: Optional[Dict[str, Any]] = None,
    enabled: bool = False,
    debug: bool = False,
    max_output_tokens: int = 512,
) -> BlocMetadataResult:
    """Generate + persist compact JSON-LD metadata for a bloc (best-effort).

    This is optional by design (`enabled=False` by default). When enabled, it tries to reuse
    the bloc's cached KV/prefix context by forking or loading the KV artifact into a temporary
    cache key, then asking the model for a strict 1-line JSON payload.
    """
    if not enabled:
        return BlocMetadataResult(ok=True, jsonld=None)
    if provider is None:
        return BlocMetadataResult(ok=False, error="provider is required")
    if not isinstance(record, FileBlocRecord) or not record.sha256:
        return BlocMetadataResult(ok=False, error="invalid bloc record")

    schema = schema or load_bloc_metadata_schema()
    ctx = schema.get("@context")
    if not isinstance(ctx, dict):
        return BlocMetadataResult(ok=False, error="schema is missing @context")

    supports = getattr(provider, "prompt_cache_supports_operation", None)
    can_fork = callable(supports) and bool(supports("fork"))
    can_load = callable(supports) and bool(supports("load"))
    can_clear = callable(supports) and bool(supports("clear"))

    system_prompt = _build_metadata_prompt(schema, record=record)

    last_raw: str = ""
    last_error: str = ""

    def _write_raw_debug(*, raw_text: str, error: str) -> None:
        if not debug:
            return
        try:
            out_dir = store.meta_jsonld_path(record.sha256).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "meta.raw.txt").write_text(str(raw_text or ""), encoding="utf-8")
            (out_dir / "meta.error.txt").write_text(str(error or "").strip() + "\n", encoding="utf-8")
        except Exception:
            pass

    def _cleanup_debug() -> None:
        if not debug:
            return
        try:
            out_dir = store.meta_jsonld_path(record.sha256).parent
            for name in ("meta.raw.txt", "meta.error.txt"):
                p = out_dir / name
                if p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass
        except Exception:
            pass

    def _normalize_payload_keys(payload: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        out = dict(payload)
        if "t" not in out and isinstance(out.get("title"), str):
            out["t"] = out.get("title")
        if "d" not in out and isinstance(out.get("description"), str):
            out["d"] = out.get("description")
        if "d" not in out and isinstance(out.get("desc"), str):
            out["d"] = out.get("desc")
        if "kw" not in out and out.get("keywords") is not None:
            out["kw"] = out.get("keywords")
        if "tp" not in out and out.get("topics") is not None:
            out["tp"] = out.get("topics")
        if "lang" not in out and out.get("language") is not None:
            out["lang"] = out.get("language")
        if "mod" not in out and out.get("modality") is not None:
            out["mod"] = out.get("modality")
        if "kind" not in out and out.get("type") is not None:
            out["kind"] = out.get("type")
        if "q" not in out and out.get("quality") is not None:
            out["q"] = out.get("quality")
        return out

    def _fix_prompt(*, error: str, raw_text: str) -> str:
        snippet = str(raw_text or "").strip()
        if len(snippet) > 2000:
            snippet = snippet[:2000].rstrip() + "…"
        return (
            "Your previous output did not match the required JSON schema.\n"
            "Return EXACTLY one line of JSON. No analysis, no <think>.\n"
            f"Error: {error}\n"
            "Previous output (may be invalid):\n"
            f"{snippet}\n"
            "Return corrected JSON only.\n"
        )

    enums = schema.get("enums") if isinstance(schema.get("enums"), dict) else {}
    allowed_mod = list(enums.get("mod") or [])
    allowed_kind = list(enums.get("kind") or [])
    qdims = schema.get("quality_dims") if isinstance(schema.get("quality_dims"), list) else []
    limits = schema.get("limits") if isinstance(schema.get("limits"), dict) else {}
    tmax = int(limits.get("t_max_chars") or 140)
    dmax = int(limits.get("d_max_chars") or 420)
    kwmax = int(limits.get("kw_max") or 16)
    tpmax = int(limits.get("tp_max") or 12)

    repairs: list[str] = []

    for attempt in range(3):
        tmp_key = f"tmp:meta:{uuid.uuid4().hex[:12]}"
        try:
            if can_fork and stable_cache_key:
                provider.prompt_cache_fork(stable_cache_key, tmp_key, make_default=True)
            elif can_load and isinstance(kv_path, str) and kv_path:
                provider.prompt_cache_load(str(kv_path), key=tmp_key, make_default=True)
            else:
                return BlocMetadataResult(
                    ok=False,
                    error="provider does not support prompt-cache fork/load for metadata generation",
                )

            user_prompt = "Generate the JSON metadata now." if attempt == 0 else _fix_prompt(
                error=last_error or "invalid metadata json", raw_text=last_raw
            )
            attempt_max = int(max_output_tokens)
            if attempt == 1:
                attempt_max = max(attempt_max, 2048)
            elif attempt >= 2:
                attempt_max = max(attempt_max, 4096)

            resp = provider.generate(
                user_prompt,
                system_prompt=system_prompt,
                stream=False,
                thinking=False,
                max_output_tokens=int(attempt_max),
                temperature=0.0,
                top_p=1.0,
                prompt_cache_key=tmp_key,
            )
            last_raw = str(getattr(resp, "content", "") or "")

            data = _extract_json_object(last_raw)
            if not isinstance(data, dict):
                last_error = "model did not return a JSON object"
                repairs.append(last_error)
                _write_raw_debug(raw_text=last_raw, error=last_error)
                continue
            data = _normalize_payload_keys(data)

            title = str(data.get("t") or "").strip()
            desc = str(data.get("d") or "").strip()
            kw = data.get("kw")
            tp = data.get("tp")
            raw_kind = data.get("kind")
            raw_mod = data.get("mod")
            raw_lang = data.get("lang")
            q = data.get("q")

            if not title or not desc:
                last_error = "missing required fields: t/d"
                repairs.append(last_error)
                _write_raw_debug(raw_text=last_raw, error=last_error)
                continue

            if len(title) > tmax:
                title = title[:tmax].rstrip()
            if len(desc) > dmax:
                desc = desc[:dmax].rstrip()

            if kw is None:
                last_error = "missing kw"
                repairs.append(last_error)
                _write_raw_debug(raw_text=last_raw, error=last_error)
                continue
            if not isinstance(kw, list) or not all(isinstance(x, str) and x.strip() for x in kw):
                last_error = "invalid kw (expected array of strings)"
                repairs.append(last_error)
                _write_raw_debug(raw_text=last_raw, error=last_error)
                continue
            if len(kw) > kwmax:
                kw = kw[:kwmax]

            if tp is None:
                last_error = "missing tp"
                repairs.append(last_error)
                _write_raw_debug(raw_text=last_raw, error=last_error)
                continue
            if not isinstance(tp, list) or not all(isinstance(x, str) and x.strip() for x in tp):
                last_error = "invalid tp (expected array of strings)"
                repairs.append(last_error)
                _write_raw_debug(raw_text=last_raw, error=last_error)
                continue
            if len(tp) > tpmax:
                tp = tp[:tpmax]

            kind = _normalize_kind(raw_kind, allowed_kind) if allowed_kind else str(raw_kind or "").strip() or None
            if kind is None:
                last_error = "missing kind" if not str(raw_kind or "").strip() else "invalid kind (not in schema enum)"
                repairs.append(last_error)
                _write_raw_debug(raw_text=last_raw, error=last_error)
                continue

            modality = (
                _normalize_modality(raw_mod, allowed_mod) if allowed_mod else str(raw_mod or "").strip().lower() or None
            )
            if modality is None:
                last_error = "missing mod" if not str(raw_mod or "").strip() else "invalid mod (not in schema enum)"
                repairs.append(last_error)
                _write_raw_debug(raw_text=last_raw, error=last_error)
                continue

            lang = _normalize_lang(raw_lang)
            if lang is None:
                last_error = "missing lang" if not str(raw_lang or "").strip() else "invalid lang"
                repairs.append(last_error)
                _write_raw_debug(raw_text=last_raw, error=last_error)
                continue

            if q is None or not isinstance(q, dict):
                last_error = "missing q"
                repairs.append(last_error)
                _write_raw_debug(raw_text=last_raw, error=last_error)
                continue

            q_out: Dict[str, Any] = {}
            for k in qdims:
                key = str(k)
                val = _coerce_float01(q.get(key))
                if val is None:
                    last_error = f"quality dimension missing/invalid: {key}"
                    repairs.append(last_error)
                    _write_raw_debug(raw_text=last_raw, error=last_error)
                    q_out = {}
                    break
                q_out[key] = val
            if not q_out:
                continue

            try:
                rounded = {round(float(v), 2) for v in q_out.values()}
            except Exception:
                rounded = set()
            if len(rounded) <= 1:
                last_error = "q values are uninformative (all identical)"
                repairs.append(last_error)
                _write_raw_debug(raw_text=last_raw, error=last_error)
                continue

            now_iso = _iso(datetime.now(tz=timezone.utc))
            jsonld: Dict[str, Any] = {
                "@context": ctx,
                "id": f"ac:bloc-{record.sha256}",
                "typ": ["ac:Bloc", "dct:Text", kind],
                "kind": kind,
                "schema": int(schema.get("schema") or 1),
                "sha": str(record.sha256),
                "csha": str(record.content_sha256 or ""),
                "p": str(record.path or ""),
                "rp": str(record.relpath or ""),
                "fn": str(record.filename or ""),
                "format": str(record.media_type or ""),
                "tok": int(record.estimated_tokens) if isinstance(record.estimated_tokens, int) else None,
                "created": _ts_iso(record.created_at),
                "modified": _ts_iso(record.updated_at),
                "t": title,
                "d": desc,
                "kw": [str(x).strip() for x in kw if isinstance(x, str) and str(x).strip()],
                "tp": [str(x).strip() for x in tp if isinstance(x, str) and str(x).strip()],
                "mod": modality,
                "gen": {
                    "typ": "prov:Activity",
                    "model": str(model_id or ""),
                    "schema": int(schema.get("schema") or 1),
                    "at": now_iso,
                },
                "use": {"acc": 0, "mcount": 0},
                "q": q_out,
                "lang": lang,
            }
            if repairs:
                jsonld["qs"] = {"repairs": list(repairs)}

            store.write_jsonld(record.sha256, jsonld)
            store.patch_record(record.sha256, summary=title, keywords=jsonld.get("kw") or [])
            _cleanup_debug()
            return BlocMetadataResult(ok=True, jsonld=jsonld)
        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            repairs.append(last_error)
            _write_raw_debug(raw_text=last_raw, error=last_error)
        finally:
            if can_clear:
                try:
                    provider.prompt_cache_clear(tmp_key)
                except Exception:
                    pass

    return BlocMetadataResult(ok=False, error=last_error or "metadata generation failed")

