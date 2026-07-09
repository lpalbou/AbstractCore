"""
Shared PDF routing for byte-oriented callers such as web tools.

This module keeps local PDF extraction ownership inside the media layer while
allowing explicitly-authorized native LLM augmentation for small PDFs.
"""

from __future__ import annotations

import base64
import json
import os
import re
import tempfile
from dataclasses import dataclass
from importlib.util import find_spec
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from ..utils.truncation import preview_text
from .processors.pdf_processor import PDFProcessor


def _module_available(name: str) -> bool:
    return find_spec(name) is not None


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "") or "").strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _normalize_pdf_backend(value: Any) -> str:
    raw = str(value or "auto").strip().lower().replace("_", "-")
    if raw in {"", "auto"}:
        return "auto"
    if raw in {"native", "native-llm", "native_llm"}:
        return "native_llm"
    if raw in {"pymupdf", "fitz"}:
        return "pymupdf"
    if raw in {"pymupdf4llm", "pymupdf-4llm", "layout", "high-fidelity", "commercial"}:
        return "pymupdf4llm"
    if raw in {"pypdf", "default", "permissive", "basic"}:
        return "pypdf"
    raise ValueError(
        f"Unsupported PDF backend '{value}'. Use one of: auto, native_llm, pymupdf, pymupdf4llm, pypdf."
    )


def _extract_json_object(text: str) -> Optional[dict[str, Any]]:
    raw = str(text or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\s*```$", "", raw)
        raw = raw.strip()
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        pass

    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(raw[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
    return None


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines).strip()


def _has_meaningful_local_text(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    informative_lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if re.fullmatch(r"!\[[^\]]*\]\([^)]+\)", line):
            continue
        informative_lines.append(line)
    if not informative_lines:
        return False
    informative_text = "\n".join(informative_lines).strip()
    if len(re.sub(r"[^A-Za-z0-9]+", "", informative_text)) < 12:
        return False
    return True


def _guess_pdf_source_name(source_url: str, source_name: str) -> str:
    name = str(source_name or "").strip()
    if name:
        return name
    try:
        from urllib.parse import urlparse

        candidate = Path(urlparse(str(source_url or "")).path).name
        if candidate:
            return candidate
    except Exception:
        pass
    return "document.pdf"


def _peek_pdf_metadata(pdf_bytes: bytes) -> dict[str, Any]:
    result: dict[str, Any] = {"page_count": None, "title": "", "warnings": []}
    if not _module_available("pypdf"):
        return result
    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
        result["page_count"] = len(reader.pages)
        metadata = getattr(reader, "metadata", None) or {}
        title = str(getattr(metadata, "title", "") or "").strip() if metadata is not None else ""
        result["title"] = title
        result["encrypted"] = bool(getattr(reader, "is_encrypted", False))
    except Exception as exc:
        warnings = result.setdefault("warnings", [])
        warnings.append(f"PDF metadata probe failed: {exc}")
    return result


def _local_backend_available(backend: str) -> bool:
    if backend == "pymupdf4llm":
        return _module_available("pymupdf4llm")
    if backend == "pymupdf":
        return _module_available("pymupdf")
    if backend == "pypdf":
        return _module_available("pypdf")
    return False


def _local_backend_candidates(requested_backend: str) -> list[str]:
    requested = _normalize_pdf_backend(requested_backend)
    if requested == "pypdf":
        return ["pypdf"]
    if requested == "pymupdf4llm":
        return ["pymupdf4llm", "pymupdf", "pypdf"]
    if requested == "pymupdf":
        return ["pymupdf", "pypdf"]
    if requested == "native_llm":
        return ["pymupdf", "pypdf"]
    return ["pymupdf", "pypdf"]


def _extract_local_pdf_bytes(
    pdf_bytes: bytes,
    *,
    backend: str,
    include_full_content: bool,
) -> dict[str, Any]:
    suffix = ".pdf"
    temp_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(prefix="abstractcore_pdf_route_", suffix=suffix, delete=False) as tmp:
            tmp.write(pdf_bytes)
            temp_path = Path(tmp.name)

        processor = PDFProcessor(
            pdf_backend=backend,
            markdown_output=True,
            preserve_tables=True,
            extract_images=False,
        )
        result = processor.process_file(temp_path)
        if not result.success or result.media_content is None:
            raise RuntimeError(str(result.error_message or "PDF extraction failed"))

        content = _normalize_text(str(result.media_content.content or ""))
        metadata = dict(getattr(result.media_content, "metadata", None) or {})
        if not include_full_content and content:
            content = preview_text(content, max_chars=4_800)
        return {
            "backend": backend,
            "content": content,
            "metadata": metadata,
        }
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except Exception:
                pass


@dataclass
class NativePDFConfig:
    model: str
    api_key: str
    base_url: Optional[str]
    max_pages: int
    max_bytes: int
    timeout_s: int


def _native_pdf_config_from_env() -> Optional[NativePDFConfig]:
    model = str(os.getenv("ABSTRACTCORE_FETCH_URL_PDF_NATIVE_MODEL", "") or "").strip()
    if not model:
        model = "gpt-4.1-mini"

    base_url_raw = str(os.getenv("OPENAI_BASE_URL", "") or "").strip()
    base_url = base_url_raw or None

    api_key = str(os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key and base_url:
        api_key = "EMPTY"
    if not api_key:
        return None

    return NativePDFConfig(
        model=model,
        api_key=api_key,
        base_url=base_url,
        max_pages=max(1, _env_int("ABSTRACTCORE_FETCH_URL_PDF_NATIVE_MAX_PAGES", 4)),
        max_bytes=max(64 * 1024, _env_int("ABSTRACTCORE_FETCH_URL_PDF_NATIVE_MAX_BYTES", 4 * 1024 * 1024)),
        timeout_s=max(15, _env_int("ABSTRACTCORE_FETCH_URL_PDF_NATIVE_TIMEOUT_S", 120)),
    )


def _call_native_pdf_model(
    pdf_bytes: bytes,
    *,
    source_url: str,
    source_name: str,
    include_full_content: bool,
    config: NativePDFConfig,
) -> dict[str, Any]:
    if not _module_available("openai"):
        raise RuntimeError("openai package is not installed")

    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": config.api_key, "timeout": float(config.timeout_s)}
    if config.base_url:
        client_kwargs["base_url"] = config.base_url
    client = OpenAI(**client_kwargs)

    preview_chars = 7_000 if include_full_content else 2_400
    summary_chars = 1_200 if include_full_content else 800
    prompt = (
        "Read the attached PDF and return strict JSON only. "
        "Keys: title, summary, text_preview, page_count_detected, key_facts, warnings. "
        f"`summary` must stay under {summary_chars} characters. "
        f"`text_preview` must stay under {preview_chars} characters and should preserve visible fields, lists, and table rows when possible. "
        "Use the PDF pages directly, including rendered pages when the PDF is scanned or image-only. "
        "Do not use markdown fences. Do not invent values; prefer empty strings or empty lists when uncertain."
    )

    def _parse_native_output(output_text: str, *, transport: str, upload_name: str) -> dict[str, Any]:
        parsed = _extract_json_object(output_text)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Native PDF response was not valid JSON: {preview_text(output_text, max_chars=300)}")

        title = str(parsed.get("title") or "").strip()
        summary = str(parsed.get("summary") or "").strip()
        text_preview = _normalize_text(str(parsed.get("text_preview") or ""))
        page_count_detected = parsed.get("page_count_detected")
        if page_count_detected is not None:
            try:
                page_count_detected = int(page_count_detected)
            except Exception:
                page_count_detected = None
        key_facts = [
            str(item).strip()
            for item in (parsed.get("key_facts") or [])
            if str(item or "").strip()
        ]
        warnings = [
            str(item).strip()
            for item in (parsed.get("warnings") or [])
            if str(item or "").strip()
        ]
        return {
            "backend": "native_llm",
            "title": title,
            "summary": summary,
            "text_preview": text_preview,
            "page_count_detected": page_count_detected,
            "key_facts": key_facts,
            "warnings": warnings,
            "model": config.model,
            "base_url": config.base_url or "",
            "upload_name": upload_name,
            "transport": transport,
        }

    upload_name = _guess_pdf_source_name(source_url, source_name)
    upload_id = ""
    temp_path: Optional[Path] = None
    upload_error: Optional[Exception] = None
    try:
        with tempfile.NamedTemporaryFile(prefix="abstractcore_pdf_native_", suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            temp_path = Path(tmp.name)

        try:
            with temp_path.open("rb") as fh:
                upload = client.files.create(file=fh, purpose="user_data")
            upload_id = str(getattr(upload, "id", "") or "").strip()
            if not upload_id:
                raise RuntimeError("Native PDF upload returned no file id")

            response = client.responses.create(
                model=config.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_file", "file_id": upload_id},
                        ],
                    }
                ],
            )
            output_text = str(getattr(response, "output_text", "") or "").strip()
            return _parse_native_output(output_text, transport="file_id", upload_name=upload_name)
        except Exception as exc:
            upload_error = exc
        finally:
            if upload_id:
                try:
                    client.files.delete(upload_id)
                except Exception:
                    pass

        data_url = "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode("ascii")
        response = client.responses.create(
            model=config.model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_file",
                            "filename": upload_name,
                            "file_data": data_url,
                        },
                        {"type": "input_text", "text": prompt},
                    ],
                }
            ],
        )
        output_text = str(getattr(response, "output_text", "") or "").strip()
        return _parse_native_output(output_text, transport="data_url", upload_name=upload_name)
    except Exception as data_url_exc:
        if upload_error is not None:
            raise RuntimeError(
                f"Native PDF failed via file_id ({upload_error}) and data_url ({data_url_exc})"
            ) from data_url_exc
        raise
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except Exception:
                pass


def route_pdf_bytes(
    pdf_bytes: bytes,
    *,
    source_url: str = "",
    source_name: str = "",
    include_full_content: bool = False,
    preferred_backend: str = "auto",
) -> dict[str, Any]:
    """
    Route PDF extraction through native or local backends with explicit provenance.

    Returns a dict containing:
    - rendered
    - raw_text / normalized_text
    - title
    - page_count
    - text_backend
    - summary_backend
    - backend_attempts
    - warnings
    """

    requested_backend = _normalize_pdf_backend(preferred_backend)
    metadata_probe = _peek_pdf_metadata(pdf_bytes)
    page_count = metadata_probe.get("page_count")
    title = str(metadata_probe.get("title") or "").strip()
    warnings: list[str] = [str(item) for item in (metadata_probe.get("warnings") or []) if str(item or "").strip()]
    attempts: list[dict[str, Any]] = []

    native_config = _native_pdf_config_from_env()
    native_result: Optional[dict[str, Any]] = None
    if requested_backend in {"auto", "native_llm"}:
        if native_config is None:
            attempts.append({"backend": "native_llm", "status": "skipped", "reason": "not_configured"})
        else:
            if page_count is not None and int(page_count) > int(native_config.max_pages):
                attempts.append(
                    {
                        "backend": "native_llm",
                        "status": "skipped",
                        "reason": f"page_count>{native_config.max_pages}",
                    }
                )
            elif len(pdf_bytes) > int(native_config.max_bytes):
                attempts.append(
                    {
                        "backend": "native_llm",
                        "status": "skipped",
                        "reason": f"size_bytes>{native_config.max_bytes}",
                    }
                )
            else:
                try:
                    native_result = _call_native_pdf_model(
                        pdf_bytes,
                        source_url=source_url,
                        source_name=source_name,
                        include_full_content=include_full_content,
                        config=native_config,
                    )
                    attempts.append(
                        {
                            "backend": "native_llm",
                            "status": "used",
                            "model": str(native_result.get("model") or ""),
                            "base_url": str(native_result.get("base_url") or ""),
                            "transport": str(native_result.get("transport") or ""),
                        }
                    )
                except Exception as exc:
                    attempts.append(
                        {
                            "backend": "native_llm",
                            "status": "failed",
                            "reason": preview_text(str(exc), max_chars=200),
                        }
                    )
                    warnings.append(f"Native PDF extraction failed: {exc}")

    local_result: Optional[dict[str, Any]] = None
    for backend in _local_backend_candidates(requested_backend):
        if not _local_backend_available(backend):
            attempts.append({"backend": backend, "status": "skipped", "reason": "dependency_missing"})
            continue
        try:
            local_result = _extract_local_pdf_bytes(
                pdf_bytes,
                backend=backend,
                include_full_content=include_full_content,
            )
            attempts.append({"backend": backend, "status": "used"})
            break
        except Exception as exc:
            attempts.append(
                {
                    "backend": backend,
                    "status": "failed",
                    "reason": preview_text(str(exc), max_chars=200),
                }
            )
            warnings.append(f"{backend} extraction failed: {exc}")

    local_text = _normalize_text(str((local_result or {}).get("content") or ""))
    native_text_preview = _normalize_text(str((native_result or {}).get("text_preview") or ""))
    local_text_usable = _has_meaningful_local_text(local_text)
    used_native_text_fallback = (not local_text_usable) and bool(native_text_preview)
    raw_text = (local_text if local_text_usable else native_text_preview) or local_text or None
    normalized_text = raw_text

    if not title:
        title = str((native_result or {}).get("title") or "").strip()
    if not title:
        title = str(((local_result or {}).get("metadata") or {}).get("title") or "").strip()

    summary_backend = "native_llm" if native_result is not None else str((local_result or {}).get("backend") or "")
    result_page_count = page_count
    if result_page_count is None and native_result is not None:
        result_page_count = native_result.get("page_count_detected")
    if result_page_count is None and local_result is not None:
        result_page_count = ((local_result.get("metadata") or {}).get("page_count"))

    lines = ["📄 PDF Document Analysis", f"📊 Size: {len(pdf_bytes):,} bytes"]
    if pdf_bytes.startswith(b"%PDF-"):
        try:
            version_line = pdf_bytes[:20].decode("ascii", errors="ignore")
            lines.append(f"✅ Valid PDF format: {version_line.strip()}")
        except Exception:
            lines.append("✅ Valid PDF format detected")
    else:
        lines.append("⚠️  Invalid PDF format - missing PDF header")
    if title:
        lines.append(f"📰 Title: {preview_text(title, max_chars=180)}")
    if result_page_count is not None:
        lines.append(f"📚 Pages: {int(result_page_count)}")
    if summary_backend:
        lines.append(f"🧠 Summary Backend: {summary_backend}")
    if used_native_text_fallback:
        text_backend = "native_llm"
    else:
        text_backend = str((local_result or {}).get("backend") or ("native_llm" if native_text_preview else ""))
    if text_backend:
        lines.append(f"📝 Text Backend: {text_backend}")

    summary = str((native_result or {}).get("summary") or "").strip()
    if summary:
        lines.append("🧾 Summary:")
        lines.append(summary)
    key_facts = [
        str(item).strip()
        for item in ((native_result or {}).get("key_facts") or [])
        if str(item or "").strip()
    ]
    if key_facts:
        lines.append("🔎 Key Facts:")
        lines.extend([f"  • {preview_text(item, max_chars=240)}" for item in key_facts[:8]])
    if raw_text:
        lines.append("📄 Extracted Text:" if include_full_content else "📄 Extracted Text Preview:")
        lines.append(raw_text if include_full_content else preview_text(raw_text, max_chars=2_400))
        lines.append(f"📊 Extracted text length: {len(raw_text):,} characters")
    if warnings:
        lines.append("⚠️  Notes:")
        lines.extend([f"  • {preview_text(item, max_chars=240)}" for item in warnings[:8]])

    degraded = used_native_text_fallback
    if degraded:
        lines.append("⚠️  Local text extraction was empty or low-signal; using native PDF preview text as fallback evidence.")

    return {
        "rendered": "\n".join(lines),
        "raw_text": raw_text,
        "normalized_text": normalized_text,
        "title": title,
        "page_count": result_page_count,
        "text_backend": text_backend,
        "summary_backend": summary_backend,
        "backend_attempts": attempts,
        "warnings": warnings,
        "native_available": native_config is not None,
        "native_used": native_result is not None,
        "native_model": str((native_result or {}).get("model") or ""),
        "native_transport": str((native_result or {}).get("transport") or ""),
        "native_text_preview": native_text_preview or None,
        "degraded": degraded,
    }
