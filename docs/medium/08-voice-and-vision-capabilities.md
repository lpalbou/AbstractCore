# AbstractVoice + AbstractVision: add voice and vision via capability plugins (without bloating core)

Most “LLM frameworks” either:
- ship heavy audio/vision dependencies in the default install, or
- hide modality behavior behind implicit conversions.

AbstractFramework takes a different stance:
**keep the core lightweight and deterministic**, and add multimodal capabilities through **explicit, optional packages**.

In practice:
- Install **AbstractVoice** → `llm.voice` / `llm.audio` (TTS/STT)
- Install **AbstractVision** → `llm.vision` (text→image, image→image, optional video)

![Capability plugins](assets/capabilities-voice-vision.svg)

## Why this design is useful

- Your default environment stays small and import-safe.
- Voice/vision behavior stays **explicit** (no “silent” conversions).
- Capability backends can evolve independently (OpenAI-compatible HTTP, local Diffusers, GGUF engines, …).

## AbstractVoice (TTS + STT)

AbstractVoice is a modular voice I/O library with:
- **TTS**: Piper (cross-platform, no system deps)
- **STT**: faster-whisper
- a Python API (`VoiceManager`) and a CLI/REPL for smoke testing

### Install

```bash
pip install abstractvoice
```

### Prefetch models explicitly (offline-first)

AbstractVoice avoids implicit downloads inside REPL flows. Prefetch once:

```bash
abstractvoice-prefetch --piper en
abstractvoice-prefetch --stt small
```

### Minimal Python

```python
from abstractvoice import VoiceManager

vm = VoiceManager()
vm.speak("Hello! This is AbstractVoice.")
```

## AbstractVision (generative vision)

AbstractVision provides:
- a small orchestration API (`VisionManager`)
- multiple backends (OpenAI-compatible HTTP, Diffusers, stable-diffusion.cpp / GGUF)
- optional artifact references for cross-process workflows

### Install

```bash
pip install abstractvision
```

### Minimal Python (OpenAI-compatible backend)

```python
from abstractvision import LocalAssetStore, VisionManager
from abstractvision.backends import OpenAICompatibleBackendConfig, OpenAICompatibleVisionBackend

backend = OpenAICompatibleVisionBackend(
    config=OpenAICompatibleBackendConfig(
        base_url="http://localhost:1234/v1",
        api_key="YOUR_KEY",      # optional for local servers
        model_id="REMOTE_MODEL", # optional (server-dependent)
    )
)

vm = VisionManager(backend=backend, store=LocalAssetStore())
ref = vm.generate_image("a cinematic photo of a red fox in snow")
print(ref)  # artifact ref dict (portable)
```

## Using voice/vision through AbstractCore

If you already use AbstractCore as your LLM layer, capability plugins show up under `llm.*`:

```python
from abstractcore import create_llm

llm = create_llm("openai", model="gpt-4o-mini")
print(llm.capabilities.status())  # availability + hints

# Voice/audio (from AbstractVoice)
wav_bytes = llm.voice.tts("Hello", format="wav")
text = llm.audio.transcribe("speech.wav")

# Vision (from AbstractVision)
# png_bytes = llm.vision.t2i("a red square")
```

This keeps a clean separation of concerns:
- AbstractCore remains the portable LLM + tools foundation.
- AbstractVoice/AbstractVision provide deterministic modality APIs.
- AbstractRuntime/Gateway can store artifacts and replay runs across clients.

## When to add these packages

Add **AbstractVoice** when you need:
- speech input (policy-driven STT fallback),
- text-to-speech output (local-first),
- or voice-enabled UIs (tray app / gateway UI).

Add **AbstractVision** when you need:
- text-to-image / image-to-image generation,
- vision tasks as tools,
- or artifact-ref outputs that travel across processes.

Next: the knowledge layer — [AbstractMemory + AbstractSemantics](09-memory-and-semantics.md).

---

### Evidence pointers (source of truth)

- AbstractVoice repo: https://github.com/lpalbou/abstractvoice
- AbstractVoice plugin wiring: https://github.com/lpalbou/abstractvoice/blob/main/abstractvoice/integrations/abstractcore_plugin.py
- AbstractVision repo: https://github.com/lpalbou/abstractvision
- AbstractVision managers/backends: https://github.com/lpalbou/abstractvision/blob/main/src/abstractvision/vision_manager.py and https://github.com/lpalbou/abstractvision/blob/main/src/abstractvision/backends/
- AbstractCore capability surface: https://github.com/lpalbou/abstractcore/blob/main/docs/capabilities.md
