"""
Video frame extraction utilities (v0).

This module provides a small, dependency-light wrapper around ffmpeg/ffprobe
to sample a bounded number of frames from a video for downstream analysis.

Design goals:
- deterministic sampling (timestamp-based)
- bounded output (max_frames)
- actionable errors when ffmpeg/ffprobe are unavailable
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple


class VideoToolUnavailableError(RuntimeError):
    pass


def _which(cmd: str) -> Optional[str]:
    try:
        return shutil.which(cmd)
    except Exception:
        return None


def probe_duration_s(video_path: Path) -> Optional[float]:
    """Return best-effort duration (seconds) using ffprobe, or None."""
    ffprobe = _which("ffprobe")
    if not ffprobe:
        return None

    try:
        out = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=nk=1:nw=1",
                str(video_path),
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        if not out:
            return None
        return float(out)
    except Exception:
        return None


def probe_keyframe_timestamps_s(video_path: Path) -> List[float]:
    """Return keyframe timestamps (seconds) using ffprobe when available."""
    ffprobe = _which("ffprobe")
    if not ffprobe:
        return []

    try:
        out = subprocess.check_output(
            [
                ffprobe,
                "-v",
                "error",
                "-skip_frame",
                "nokey",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=pkt_pts_time",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except Exception:
        return []

    timestamps: List[float] = []
    for line in (out or "").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            t = float(s)
        except Exception:
            continue
        if t < 0:
            continue
        timestamps.append(t)

    # Deduplicate while preserving order (ffprobe can sometimes repeat values).
    seen = set()
    uniq: List[float] = []
    for t in timestamps:
        if t in seen:
            continue
        seen.add(t)
        uniq.append(t)
    return uniq


def _build_timestamps(duration_s: Optional[float], max_frames: int) -> List[float]:
    n = max(1, int(max_frames))
    if duration_s is None or duration_s <= 0:
        return [0.0]
    # Sample away from the extreme endpoints to avoid decode edge-cases.
    return [duration_s * (i + 1) / (n + 1) for i in range(n)]


def _pick_evenly_spaced(values: List[float], k: int) -> List[float]:
    if not values:
        return []
    n = len(values)
    k = max(1, int(k))
    if n <= k:
        return list(values)
    # Evenly spaced indices (include ends).
    idxs = [round(i * (n - 1) / (k - 1)) for i in range(k)] if k > 1 else [round((n - 1) / 2)]
    out: List[float] = []
    last = None
    for i in idxs:
        i = max(0, min(n - 1, int(i)))
        v = values[i]
        if last is not None and v == last:
            continue
        out.append(v)
        last = v
    return out


def extract_video_frames(
    video_path: Path,
    *,
    max_frames: int = 3,
    frame_format: str = "jpg",
    sampling_strategy: str = "uniform",
    max_side: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Tuple[List[Path], List[float]]:
    """
    Extract up to max_frames as image files and return (frame_paths, timestamps_s).

    Uses ffmpeg for extraction and ffprobe for duration (best-effort).
    """
    ffmpeg = _which("ffmpeg")
    if not ffmpeg:
        raise VideoToolUnavailableError("ffmpeg is required for video frame extraction. Install ffmpeg and ensure it is on PATH.")

    if not isinstance(video_path, Path):
        video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    fmt = str(frame_format or "jpg").strip().lower()
    if fmt == "jpeg":
        fmt = "jpg"
    if fmt not in {"jpg", "png"}:
        fmt = "jpg"

    out_dir = Path(output_dir) if output_dir is not None else Path(tempfile.mkdtemp(prefix="abstractcore_video_frames_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    duration_s = probe_duration_s(video_path)
    strategy = str(sampling_strategy or "uniform").strip().lower()
    if strategy == "keyframes":
        keyframes = probe_keyframe_timestamps_s(video_path)
        timestamps = _pick_evenly_spaced(keyframes, int(max_frames)) if keyframes else _build_timestamps(duration_s, max_frames=max_frames)
    else:
        timestamps = _build_timestamps(duration_s, max_frames=max_frames)

    frames: List[Path] = []
    for idx, ts in enumerate(timestamps):
        out_path = out_dir / f"frame_{idx+1:02d}.{fmt}"
        cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{ts:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
        ]
        if isinstance(max_side, int) and max_side > 0:
            ms = int(max_side)
            # Preserve aspect ratio, never upscale.
            # If width >= height: clamp width to ms, derive height. Else clamp height.
            vf = (
                f"scale="
                f"if(gt(iw\\,ih)\\,min(iw\\,{ms})\\,-2):"
                f"if(gt(iw\\,ih)\\,-2\\,min(ih\\,{ms}))"
            )
            cmd.extend(["-vf", vf])
        if fmt == "jpg":
            cmd.extend(["-q:v", "2"])
        cmd.append(str(out_path))

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError:
            continue

        if out_path.exists() and out_path.stat().st_size > 0:
            frames.append(out_path)

    return frames, timestamps
