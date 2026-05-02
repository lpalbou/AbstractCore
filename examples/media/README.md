# Media / Glyph Examples

## What This Folder Teaches

- How AbstractCore detects and processes local files (text + documents) through a single “media pipeline”.
- How “glyph” compression can reduce payload size for long text blocks (useful for transport/storage and some prompting workflows).

## Prereqs

- Media/file processing: `pip install "abstractcore[media]"`
  - Includes: Pillow (images), PyMuPDF4LLM + PyMuPDF (PDF), and `unstructured[...]` (DOCX/XLSX/PPTX).
- Glyph compression demos: `pip install "abstractcore[compression]"`

## Key AbstractCore Concepts

- Media detection: picking the right processor based on file type (text/document/media).
- `AutoMediaHandler`: the orchestrator that selects a processor and returns normalized `MediaContent`.
- Output formats: some processors can emit different representations (e.g. text vs markdown), depending on installed backends.

## How The Examples Work

These scripts are intentionally small and practical:
- inspect the media pipeline without an LLM,
- attach real media to an LLM call (vision/image + video frames),
- and show where v0 “file reference” handling exists (audio/video).

## Scripts

- `inspect_media_pipeline.py`
  - Demonstrates: running `AutoMediaHandler.process_file(...)` directly across multiple modalities.
  - How it works: you give it paths; it returns normalized `MediaContent` objects (text, base64 images, file refs).
  - Run: `python examples/media/inspect_media_pipeline.py --demo --show-content`
  - Takeaway: media processing is separate from generation; you can preflight/inspect before calling any LLM.

- `list_supported_formats.py`
  - Demonstrates: what processors/formats are available (based on installed optional deps).
  - Takeaway: media support is capability-driven; your environment determines what’s enabled.

- `vision_image_qa.py`
  - Demonstrates: asking a model about an image using `llm.generate(..., media=[...])`.
  - Run: `python examples/media/vision_image_qa.py --provider ollama --model llama3.2-vision:11b --image /path/to.png`
  - Takeaway: the media pipeline handles preprocessing; you focus on your prompt + model choice.

- `video_frames_qa.py`
  - Demonstrates: asking a model about a video by sampling frames (`video_policy="frames_caption"`).
  - Requires: `ffmpeg` on PATH + a vision-capable model.
  - Run: `python examples/media/video_frames_qa.py --provider ollama --model llama3.2-vision:11b --video /path/to.mp4`
  - Takeaway: portable video support is “video → sampled frames → vision”.

- `audio_input_demo.py`
  - Demonstrates: audio input policy guardrails (v0) and how to request transcription fallbacks.
  - Run: `python examples/media/audio_input_demo.py --provider ollama --model llama3.2:3b --audio /path/to.wav`
  - Takeaway: audio is not silently degraded; you must choose an audio-capable model or an STT fallback.

- `glyph_compression_demo.py`
  - Demonstrates: basic glyph compression workflow.
  - Takeaway: glyphs are a transport/storage optimization; you still decide when to compress.

- `glyph_complete_example.py`
  - Demonstrates: a fuller end-to-end glyph workflow (encode → persist → decode).
  - Takeaway: you can build deterministic “compressed artifacts” for large text blocks.

## Key Takeaways

- Prefer the media pipeline (`AutoMediaHandler`) over one-off per-file hacks: it keeps behavior consistent across formats.
- Keep compression as an explicit choice (opt-in) so it doesn’t surprise downstream consumers.
