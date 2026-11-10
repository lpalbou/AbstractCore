### Fixed

- **`pip install "abstractcore[gpu]"` on a Linux NVIDIA host now runs GGUF models on the GPU**
  (`abstractcore/utils/llama_cuda_swap.py`). PyPI carries `llama-cpp-python` as a source build only, so pip
  resolved abetlen's prebuilt *processor* wheel and every GGUF model ran on the CPU while PyTorch saw the
  card (`llama_supports_gpu_offload()` False; measured on a Quadro RTX 5000 during the 0.14.0 profile
  proof: a 4B Q4_K_M answered in 9 s on the processor, 1 s on the card). The CUDA builds live on abetlen's
  own index, which a pip extra cannot name, and are too large for PyPI. The one-line installers already
  swapped the CUDA build in; AbstractCore now does the same swap itself, once per environment, before its
  first `import llama_cpp` on an x86_64 Linux host where `torch/version.py` says CUDA 12 or 13 and
  `nvidia-smi` reports a driver new enough (525 / 580): the installers' command (`pip`, or `uv pip` when the
  environment has no pip, `--no-index --no-deps --find-links <index>/<cu130|cu125>/llama-cpp-python/`, the
  installed version), one warning line before and after, the outcome written to
  `llama_cpp/.abstractcore-cuda-swap.json` so a failed swap is not retried at every load. A host set up by
  the installer has the CUDA build already and nothing runs. When the swap cannot happen (offline,
  read-only site-packages, no driver, `ABSTRACTCORE_LLAMA_CUDA_SWAP=0`) GGUF stays on the processor and the
  HuggingFace provider's runtime info (`llama_cuda_swap`) and a warning say why. Tests: the swap runs
  once with the right build for torch's CUDA major and never again; never on a CUDA build; never with
  the opt-out, without a CUDA torch, or with a driver too old; a failed swap is recorded and not retried.
