#!/usr/bin/env python3
"""
Ask an LLM about an image (vision input).

What this demonstrates
- Passing `media=[path/to/image]` into `llm.generate(...)`.
- Letting AbstractCore normalize the image into the provider's expected payload.

Notes
- You need a vision-capable model for good results.
- For Ollama, try: `ollama pull llama3.2-vision:11b` or `ollama pull qwen2.5vl:7b`
- For LM Studio, load a vision model and start the OpenAI-compatible server.

Run
  python examples/media/vision_image_qa.py --provider ollama --model llama3.2-vision:11b --image /path/to.jpg
  python examples/media/vision_image_qa.py --provider lmstudio --model qwen2.5vl:7b --base-url http://localhost:1234/v1 --image /path/to.png
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from abstractcore import create_llm


def _make_demo_image() -> Path:
    tmp_dir = Path(tempfile.mkdtemp(prefix="abstractcore_vision_demo_"))
    img_path = tmp_dir / "demo.png"

    from PIL import Image, ImageDraw  # Pillow is an optional dependency

    img = Image.new("RGB", (420, 260), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, 400, 240], outline=(200, 200, 200), width=3)
    draw.text((40, 120), "AbstractCore vision demo", fill=(240, 240, 240))
    img.save(img_path)
    return img_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="ollama", help="e.g. ollama|lmstudio|openai|anthropic|huggingface")
    ap.add_argument("--model", default="llama3.2-vision:11b", help="Provider model id")
    ap.add_argument("--base-url", default=None, help="Optional base URL (lmstudio/openai-compatible).")
    ap.add_argument("--image", default=None, help="Path to an image file (png/jpg/webp/...).")
    ap.add_argument("--prompt", default="Describe this image in 3 bullet points.", help="Question to ask about the image.")
    args = ap.parse_args()

    image_path: Optional[Path]
    if args.image:
        image_path = Path(args.image).expanduser()
    else:
        image_path = _make_demo_image()

    llm_kwargs: Dict[str, Any] = {"model": args.model}
    if args.base_url:
        llm_kwargs["base_url"] = args.base_url
    llm = create_llm(args.provider, **llm_kwargs)

    print("=" * 80)
    print(f"provider={args.provider} model={args.model}")
    print(f"image={image_path}")
    print("prompt:", args.prompt)
    print("=" * 80)

    resp = llm.generate(str(args.prompt), media=[str(image_path)])
    print(str(getattr(resp, "content", "") or "").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

