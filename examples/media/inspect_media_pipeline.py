#!/usr/bin/env python3
"""
Inspect the AbstractCore media pipeline (no LLM required).

What this demonstrates
- How `AutoMediaHandler` detects file type and selects a processor.
- What `MediaContent` looks like for different modalities:
  - text/documents -> extracted text
  - images -> base64-encoded image payload (+ metadata)
  - audio/video -> a file reference (v0: no transcription/captioning)

Run
  python examples/media/inspect_media_pipeline.py --demo
  python examples/media/inspect_media_pipeline.py /path/to/file.pdf /path/to/file.docx
  python examples/media/inspect_media_pipeline.py --dir /path/to/folder --glob \"**/*\"
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

from abstractcore.media import AutoMediaHandler


def _iter_paths_from_dir(root: Path, pattern: str) -> Iterable[Path]:
    for p in root.glob(pattern):
        if p.is_file():
            yield p


def _make_demo_files(tmp_dir: Path) -> List[Path]:
    paths: List[Path] = []

    txt = tmp_dir / "demo.txt"
    txt.write_text("Hello from AbstractCore.\nThis is a short text file.\n", encoding="utf-8")
    paths.append(txt)

    csv = tmp_dir / "demo.csv"
    csv.write_text("name,score\nAda,98\nLinus,95\n", encoding="utf-8")
    paths.append(csv)

    try:
        from PIL import Image, ImageDraw  # type: ignore

        img = Image.new("RGB", (320, 200), color=(20, 40, 80))
        draw = ImageDraw.Draw(img)
        draw.text((20, 80), "AbstractCore", fill=(240, 240, 240))
        png = tmp_dir / "demo.png"
        img.save(png)
        paths.append(png)
    except Exception:
        # Pillow is optional; skip demo image if unavailable.
        pass

    # Small WAV (silence) for demonstrating audio handling.
    try:
        import wave

        wav_path = tmp_dir / "demo.wav"
        with wave.open(str(wav_path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # 16-bit
            w.setframerate(16000)
            w.writeframes(b"\x00\x00" * 16000)  # 1 second of silence
        paths.append(wav_path)
    except Exception:
        pass

    # Small MP4 via ffmpeg (if available) for demonstrating video handling.
    mp4 = tmp_dir / "demo.mp4"
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=320x240:d=1",
                "-pix_fmt",
                "yuv420p",
                "-y",
                str(mp4),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if mp4.exists() and mp4.stat().st_size > 0:
            paths.append(mp4)
    except Exception:
        pass

    return paths


def _preview_text(s: str, n: int) -> str:
    s = str(s or "")
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    if len(s) <= n:
        return s
    return s[:n] + "…"


def _print_result(path: Path, *, res, show_content: bool, max_chars: int) -> None:
    print("-" * 100)
    print(str(path))
    if not res or not getattr(res, "success", False) or getattr(res, "media_content", None) is None:
        msg = str(getattr(res, "error_message", "") or "unknown error")
        print(f"status=error  message={msg}")
        return

    mc = res.media_content
    mt = getattr(mc, "media_type", None)
    fmt = getattr(mc, "content_format", None)
    mime = getattr(mc, "mime_type", None)
    print(f"status=ok  media_type={getattr(mt, 'value', mt)}  format={getattr(fmt, 'value', fmt)}  mime={mime}")

    meta = getattr(mc, "metadata", None)
    if isinstance(meta, dict) and meta:
        # Print a stable subset to keep output readable.
        keys = [k for k in ("processor", "extraction_method", "output_format", "content_length", "estimated_tokens") if k in meta]
        extra = {k: meta.get(k) for k in keys}
        if extra:
            print(f"metadata={extra}")

    if not show_content:
        return

    content = getattr(mc, "content", None)
    if isinstance(content, str):
        print("content_preview:")
        print(_preview_text(content, max_chars))
    else:
        print(f"content_preview: (non-text content type={type(content).__name__})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="Files to inspect.")
    ap.add_argument("--dir", default=None, help="Inspect a directory (use with --glob).")
    ap.add_argument("--glob", default="**/*", help="Glob pattern when using --dir (default: **/*).")
    ap.add_argument("--demo", action="store_true", help="Generate a few demo files and inspect them.")
    ap.add_argument("--show-content", action="store_true", help="Print extracted text (when available).")
    ap.add_argument("--max-chars", type=int, default=1200, help="Max chars to print for text previews.")
    args = ap.parse_args()

    selected: List[Path] = []
    if args.demo:
        tmp_dir = Path(tempfile.mkdtemp(prefix="abstractcore_media_demo_"))
        selected.extend(_make_demo_files(tmp_dir))

    if args.dir:
        root = Path(args.dir).expanduser()
        selected.extend(list(_iter_paths_from_dir(root, args.glob)))

    for raw in args.paths:
        selected.append(Path(raw).expanduser())

    # De-dupe while preserving order.
    seen = set()
    paths: List[Path] = []
    for p in selected:
        try:
            key = str(p.resolve())
        except Exception:
            key = str(p)
        if key in seen:
            continue
        seen.add(key)
        paths.append(p)

    if not paths:
        ap.print_help()
        print("\nTip: try `--demo` to generate sample files.")
        return 2

    handler = AutoMediaHandler(enable_glyph_compression=False)
    for p in paths:
        try:
            res = handler.process_file(p)
        except Exception as e:  # noqa: BLE001
            res = None
            print("-" * 100)
            print(str(p))
            print(f"status=error  exception={e}")
            continue
        _print_result(p, res=res, show_content=bool(args.show_content), max_chars=int(args.max_chars))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

