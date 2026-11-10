### Fixed

- **`pip install "abstractcore[gpu]"` on a Linux NVIDIA host now runs GGUF models on the GPU**
  (`abstractcore/utils/llama_cuda_swap.py`). PyPI carries `llama-cpp-python` as a source build only, so pip
  resolved abetlen's prebuilt *processor* wheel and every GGUF model ran on the CPU while PyTorch saw the
  card (`llama_supports_gpu_offload()` False; measured on a Quadro RTX 5000 during the 0.14.0 profile
  proof: a 4B Q4_K_M answered in 9 s on the processor, 0.7 s on the card). The CUDA builds live on abetlen's
  own index, which a pip extra cannot name, and are too large for PyPI. The one-line installers already
  swapped the CUDA build in; AbstractCore now does the same swap itself, once per environment, before its
  first `import llama_cpp` on an x86_64 Linux host where `torch/version.py` says CUDA 12 or 13 and
  `nvidia-smi` reports a driver new enough (525 / 580): the installers' command (`pip`, or `uv pip` when the
  environment has no pip, `--no-index --no-deps --find-links <index>/<cu130|cu125>/llama-cpp-python/`, the
  installed version), one warning line before and after, the outcome written to
  `llama_cpp/.abstractcore-cuda-swap.json` so a failed swap is not retried at every load. A host set up by
  the installer has the CUDA build already and nothing runs. When the swap cannot happen (offline,
  read-only site-packages, no driver, `ABSTRACTCORE_LLAMA_CUDA_SWAP=0`) GGUF stays on the processor and the
  HuggingFace provider's runtime info (`llama_cuda_swap`) and a warning say why. Proven from a bare `uv pip`
  install on the GPU host: the swap ran by itself in 0.8 s and the next answer came from the card; a second
  process ran no swap. Tests: the swap runs once with the right build for torch's CUDA major (both the
  plain and the annotated `cuda = '13.0'` forms of `torch/version.py`: torch 2.13 annotates, and the first
  real run skipped the swap because of it) and never again; never on a CUDA build; never with the opt-out,
  without a CUDA torch, or with a driver too old; a failed swap is recorded and not retried.
- **A process that embedded with a GGUF model no longer aborts at exit on Apple silicon**
  (`abstractcore/embeddings/manager.py`, `release_shared_gguf_embeddings_at_exit`). After every call had
  succeeded, the interpreter died in `exit()` with `ggml-metal-device.m: GGML_ASSERT([rsets->data count] ==
  0)` (exit code 134): the llama.cpp context was freed during module teardown, after the Metal device.
  The resident GGUF encoders are now closed by an `atexit` handler, before teardown. Found on the Mac
  during the 0.14.0 profile proof with `CompendiumLabs/bge-small-en-v1.5-gguf`; the same script exits 0.
- **Install hints on a processor-only Linux host say what works.** With no llama.cpp installed, the hint
  for GGUF models said `pip install "abstractcore[gpu]"`, which compiles llama-cpp-python from source
  (PyPI has no wheel) and, on a host without a card, installs the CUDA stack for nothing. On Linux the hint
  now names the AbstractCore installer, which adds the prebuilt build for the machine (then the CUDA swap
  above), and says that the `gpu` setting alone builds from source; still only the three settings as pip
  targets. `huggingface_hub` is part of the light set (`abstractcore models download` failed on a plain
  install with the same `gpu` hint; a missing one now reads as a broken install to repair).

### Changed

- **AbstractVoice 0.21.1 is the floor** (base dependency and the `all-apple` / `all-gpu` extras): 0.21.0's
  `gpu` extra added a second cuDNN wheel next to torch's on Linux and every CUDA convolution failed with
  `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH` once abstractcore's `gpu` extra (vllm, torch 2.13) was in the
  same environment; 0.21.1 keeps cuDNN Windows-only.
- **lxml up to 6.x** (`lxml>=4.9.0,<7.0.0`, backlog 0886): lxml 5 has no CPython 3.14 wheels, so a plain
  `pip install abstractcore` on Python 3.14 tried to build it from source and failed without the libxml2
  headers; 6.1 ships them. BeautifulSoup's `lxml` and `xml` parsers are unchanged.
- **vLLM on cards older than Ampere** (`docs/prerequisites.md`): measured on a Quadro RTX 5000 (compute
  capability 7.5) with vLLM 0.30.0, the version `abstractcore[gpu]` resolves: it falls back to float16 by
  itself and uses its Triton attention backend; 236 tokens/s on Qwen2.5-0.5B through `create_llm("vllm")`.
  No flag and no other version needed.
