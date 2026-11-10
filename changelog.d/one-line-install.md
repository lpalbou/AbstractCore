### Added

- **AbstractCore installs in one line, with no admin rights and no Python needed** (`scripts/install.sh`,
  `scripts/install.ps1`). `curl -LsSf https://abstractframework.ai/core/install.sh | sh` on macOS and
  Linux, `powershell -ExecutionPolicy ByPass -c "irm https://abstractframework.ai/core/install.ps1 | iex"`
  on Windows: the script gets uv when it is missing and Python 3.12 through uv, picks the install
  setting from the machine (Apple Silicon with macOS 14 or later: `abstractcore[apple]`; an NVIDIA
  card or ROCm: `abstractcore[gpu]`; anything else: the light install), and installs AbstractCore
  alone as a uv tool, pinned to this release, with the `abstractcore`, `abstractcore-config`,
  `abstractcore-chat` and `abstractcore-endpoint` commands and the apps on PATH. It never starts a
  compiler: llama.cpp comes from upstream's prebuilt Metal or CPU build, and the two optional
  packages without wheels (echo cancellation, stable-diffusion.cpp) are left out. With an NVIDIA
  card the `gpu` setting then gets llama.cpp's CUDA build matching PyTorch's CUDA (`cu130` or
  `cu125`; on Windows `cu130`, `cu132` or `cu125`), kept only when it loads the way AbstractCore
  loads it and `llama_supports_gpu_offload()` is True, else the CPU build with the reason (ROCm
  included). On Windows the `gpu` setting also gets PyTorch's CUDA build for the driver. Every step prints its command; `--print`
  shows the plan and changes nothing; running it again upgrades in place and keeps the setting;
  `--profile`, `--pin VERSION|latest` and `--from PATH` choose otherwise. On a machine where
  AbstractFramework's installer already provides the `abstractcore` command, it stops and says so.
  See [One-line install](docs/installation.md#one-line-install).
