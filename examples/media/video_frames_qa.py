#!/usr/bin/env python3
"""
Ask an LLM about a video by sampling frames (portable fallback).

What this demonstrates
- Passing `media=[path/to/video]` into `llm.generate(...)`.
- Using `video_policy="frames_caption"` so AbstractCore samples a bounded number of frames
  and feeds them as images to the model (requires ffmpeg + a vision-capable model).

Run
  python examples/media/video_frames_qa.py --provider ollama --model llama3.2-vision:11b --video /path/to.mp4
  python examples/media/video_frames_qa.py --provider lmstudio --model qwen2.5vl:7b --base-url http://localhost:1234/v1 --video /path/to.mov
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from abstractcore import create_llm


def _make_demo_video() -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="abstractcore_video_demo_"))
    out = tmp_dir / "demo.mp4"
    # 2s solid color clip (no external assets).
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:s=320x240:d=2",
            "-pix_fmt",
            "yuv420p",
            "-y",
            str(out),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama", help="e.g. ollama|lmstudio|openai|anthropic|huggingface")
    ap.add_argument("--model", default="llama3.2-vision:11b", help="Provider model id")
    ap.add_argument("--base-url", default=None, help="Optional base URL (lmstudio/openai-compatible).")
    ap.add_argument("--video", default=None, help="Path to a video file (mp4/mov/...).")
    ap.add_argument(
        "--prompt",
        default="What is happening in this video? Reply in 3 short bullet points.",
        help="Question to ask about the video.",
    )
    ap.add_argument("--max-frames", type=int, default=3, help="How many frames to sample (default: 3).")
    ap.add_argument(
        "--sampling",
        default="uniform",
        choices=["uniform", "keyframes"],
        help="Frame sampling strategy (default: uniform).",
    )
    ap.add_argument("--max-frame-side", type=int, default=1024, help="Max pixel side for extracted frames.")
    args = ap.parse_args()

    video_path: Optional[Path]
    if args.video:
        video_path = Path(args.video).expanduser()
    else:
        video_path = _make_demo_video()

    llm_kwargs: Dict[str, Any] = {"model": args.model}
    if args.base_url:
        llm_kwargs["base_url"] = args.base_url
    llm = create_llm(args.provider, **llm_kwargs)

    print("=" * 80)
    print(f"provider={args.provider} model={args.model}")
    print(f"video={video_path}")
    print(f"video_policy=frames_caption  max_frames={args.max_frames}  sampling={args.sampling}  max_frame_side={args.max_frame_side}")
    print("prompt:", args.prompt)
    print("=" * 80)

    resp = llm.generate(
        str(args.prompt),
        media=[str(video_path)],
        video_policy="frames_caption",
        video_max_frames=int(args.max_frames),
        video_sampling_strategy=str(args.sampling),
        video_max_frame_side=int(args.max_frame_side),
    )
    print(str(getattr(resp, "content", "") or "").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

