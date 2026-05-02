#!/usr/bin/env python3
"""
Audio input handling (v0) in AbstractCore.

What this demonstrates
- Passing `media=[path/to/audio]` into `llm.generate(...)`.
- The audio input policy guardrails:
  - By default (`audio_policy="native_only"`), AbstractCore will *error* if the model
    is not audio-capable (to avoid silent placeholder degradation).
  - You can request a transcription fallback (`audio_policy="speech_to_text"`) when
    an STT capability plugin is configured (e.g. abstractvoice).

This example does not require an audio-capable model to run — it’s useful even if it
errors, because the error message is the point.

Run
  python examples/media/audio_input_demo.py --provider lmstudio --model your-audio-model --audio /path/to.wav
  python examples/media/audio_input_demo.py --provider ollama --model llama3.2:3b --audio /path/to.wav
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from abstractcore import create_llm


def _make_demo_wav() -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="abstractcore_audio_demo_"))
    wav_path = tmp_dir / "demo.wav"
    import wave

    with wave.open(str(wav_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 16000)  # 1 second of silence
    return wav_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama", help="e.g. ollama|lmstudio|openai|anthropic|huggingface")
    ap.add_argument("--model", default="llama3.2:3b", help="Provider model id")
    ap.add_argument("--base-url", default=None, help="Optional base URL (lmstudio/openai-compatible).")
    ap.add_argument("--audio", default=None, help="Path to an audio file (wav/mp3/...).")
    ap.add_argument(
        "--audio-policy",
        default="native_only",
        choices=["native_only", "speech_to_text", "auto"],
        help="How to handle audio inputs (default: native_only).",
    )
    ap.add_argument("--prompt", default="Transcribe the attached audio.", help="Prompt to send along with the audio.")
    args = ap.parse_args()

    audio_path: Optional[Path]
    if args.audio:
        audio_path = Path(args.audio).expanduser()
    else:
        audio_path = _make_demo_wav()

    llm_kwargs: Dict[str, Any] = {"model": args.model}
    if args.base_url:
        llm_kwargs["base_url"] = args.base_url
    llm = create_llm(args.provider, **llm_kwargs)

    print("=" * 80)
    print(f"provider={args.provider} model={args.model}")
    print(f"audio={audio_path}")
    print(f"audio_policy={args.audio_policy}")
    print("prompt:", args.prompt)
    print("=" * 80)

    try:
        resp = llm.generate(
            str(args.prompt),
            media=[str(audio_path)],
            audio_policy=str(args.audio_policy),
        )
        print(str(getattr(resp, "content", "") or "").strip())
    except Exception as e:  # noqa: BLE001
        print(f"error: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

