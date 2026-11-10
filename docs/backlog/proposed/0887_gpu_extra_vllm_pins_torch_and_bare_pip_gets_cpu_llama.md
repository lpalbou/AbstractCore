# 0887 — The `gpu` extra: vLLM pins torch for everyone, and a bare pip gets the CPU llama.cpp wheel

> Type: packaging
> Created: 2026-11-10
> Priority: normal
> Labels: packaging, gpu, dependencies, installer
> Links: `pyproject.toml` (`gpu`, `vllm` extras), `scripts/install.sh` (CUDA wheel swap), abstractvoice 0.21.1 changelog

## Summary

Measured on 2026-11-10 (Ubuntu 24.04, Quadro RTX 5000, framework `gpu` matrix with the published
lower packages, isolated uv venv, Python 3.12):

1. `abstractcore[gpu]` on Linux pulls `vllm` (0.32.0 today), which pins `torch==2.13.0`,
   `torchaudio==2.11.0`, `torchvision==0.28.0` and its own CUDA sub-libraries. A voice-only or
   vision-only install resolves torch 2.15.0; the framework profile is downgraded two torch
   releases by a provider almost nobody runs through AbstractCore, and the install grows by about
   2 GB (vllm, flashinfer, xgrammar, cutlass-dsl for cu12 and cu13, ...). The downgrade also moved
   torch's cuDNN wheel to 9.20, which, combined with abstractvoice's `nvidia-cudnn-cu12` 9.28 in
   the same folder, broke every CUDA convolution (fixed on the voice side in 0.21.1; the core side
   remains a cost and a resolution hazard).
2. A plain `pip install abstractcore[gpu]` gets the CPU-only `llama-cpp-python` wheel from PyPI
   (`llama_supports_gpu_offload()` is False): GGUF models run on the processor on an NVIDIA host.
   `scripts/install.sh` swaps in the CUDA wheel on NVIDIA hosts; the bare pip path and the
   framework's own matrix install do not, and nothing tells the user.

## Decision

Proposed: (a) move `vllm` out of `gpu` into its own opt-in extra (`abstractcore[vllm]`, which today
is an alias of `gpu`), so `gpu` carries torch, transformers, llama.cpp and the vision/voice
engines only; state it in the changelog as a change of what `gpu` installs; (b) for llama.cpp on
NVIDIA, either document the CUDA wheel step in `docs/installation.md` next to the `gpu` extra
(the installer's exact command) or have the HuggingFace provider report "GGUF on the processor:
CPU build of llama-cpp-python" in its runtime info so the user sees it. Decide with the framework
seat, which owns the root `gpu` profile pins.

## Success criteria

- `pip install abstractcore[gpu]` on Linux resolves the torch that voice/vision resolve alone; no
  vllm unless `[vllm]` is asked for.
- An NVIDIA user who installs with bare pip can see that GGUF runs on the CPU and how to fix it.
