#!/bin/sh
# =============================================================================
# AbstractCore standalone installer (macOS / Linux)
# =============================================================================
# One line, no admin rights, no system Python needed:
#
#   curl -LsSf https://abstractframework.ai/core/install.sh | sh
#   curl -LsSf https://abstractframework.ai/core/install.sh | sh -s -- --profile light
#
# It installs AbstractCore alone: the `abstractcore` command (config, models, engines, serve),
# `abstractcore-config`, `abstractcore-chat`, `abstractcore-endpoint` and the apps (summarizer,
# extractor, judge, intent, deepsearch), no gateway, no services. (AbstractFramework's own
# installer, https://abstractframework.ai/install.sh, already includes AbstractCore.)
#
# Steps (each prints its command; --print shows them and changes nothing): preflight (OS,
# arch, profile, disk, pypi.org); uv (https://docs.astral.sh/uv) into ~/.local/bin when missing;
# Python 3.12 through uv; `uv tool install --python 3.12 "abstractcore[<profile>]==<version>"`
# (isolated, user-scoped, prebuilt wheels only, below); `uv tool update-shell` when the command
# folder is not on PATH.
#
# Profiles (--profile auto|light|apple|gpu; auto takes the first that fits):
#   apple  Apple Silicon, macOS 14+: abstractcore[apple] (MLX, llama.cpp on Metal, local
#          embeddings, voice, image and music engines on the Apple GPU)
#   gpu    Linux with NVIDIA (nvidia-smi -L) or ROCm (rocminfo): abstractcore[gpu] (vLLM,
#          PyTorch, llama.cpp, local embeddings, voice, image and music engines); asked for
#          without a card, the same stack on the processor
#   light  anything else: abstractcore with no extra, which already carries every remote
#          provider (OpenAI, Anthropic, OpenRouter, Portkey, any OpenAI-compatible endpoint,
#          LM Studio, Ollama, a vLLM server), the built-in tools, media inputs and the HTTP server
#
# Prebuilt wheels only: no C compiler or Xcode tools (a fresh Mac never sees the Xcode dialog).
# Three packages of the apple and gpu trees publish no usable wheel on PyPI, so their install
# gets an overrides file (an override whose marker is never true drops the package) and
# --no-build-package for each, as in the framework installer:
#   aec-audio-processing         echo cancellation (optional, imported lazily): dropped
#   stable-diffusion-cpp-python  stable-diffusion.cpp (optional, imported lazily): dropped
#   llama-cpp-python             llama.cpp (GGUF models): PyPI has only the sdist, so it comes
#                                from upstream's prebuilt wheels (--find-links on abetlen's wheel
#                                index, pinned with --with): Metal 0.3.28 on Apple Silicon, CPU
#                                0.3.36 on glibc Linux x86_64/aarch64. Elsewhere, or when that
#                                install fails, it is dropped and the summary says so.
#
# Upgrade: run the same line again: "AbstractCore <old> found: upgrading to <new>" (or "already
# up to date"); the previous profile is kept unless --profile says otherwise.
# Remove: uv tool uninstall abstractcore
#
# Options (environment twins in brackets):
#   --profile auto|light|apple|gpu  install profile (default auto)              [AC_PROFILE]
#   --pin VERSION|latest     the AbstractCore version (default: the release this script
#                            ships with, below; latest = the newest on PyPI)    [AC_PIN]
#   --from PATH              install from a checkout, sdist or wheel (testing)  [AC_FROM]
#   --no-modify-path         do not run `uv tool update-shell`
#   --print, --dry-run       show the plan and the commands; change nothing
#   -v, --verbose            show the full output of every command
#   -h, --help               this help
# =============================================================================

if [ -n "${ZSH_VERSION:-}" ]; then emulate sh; fi
set -eu

# The release this script ships with = abstractcore/utils/version.py (tests/test_install_scripts.py
# fails on drift), so an install lands on the version the script was tested with.
AC_PIN_DEFAULT="2.35.0"
AC_PYTHON="3.12"
AC_SCRIPT_URL="https://abstractframework.ai/core/install.sh"
AC_DOCS="https://github.com/lpalbou/AbstractCore/blob/main/docs/installation.md#one-line-install"
# Every name of [project.scripts] in pyproject.toml (the test compares them): uv refuses the
# whole install when one of them already exists.
AC_COMMANDS="abstractcore abstractcore-config abstractcore-chat abstractcore-endpoint summarizer abstractcore-summarizer extractor abstractcore-extractor judge abstractcore-judge intent abstractcore-intent deepsearch abstractcore-deepsearch"
AC_NO_WHEEL="aec-audio-processing stable-diffusion-cpp-python"
AC_LLAMA_INDEX="https://abetlen.github.io/llama-cpp-python/whl"
AC_LLAMA_METAL_PIN="0.3.28"
AC_LLAMA_CPU_PIN="0.3.36"

PROFILE="${AC_PROFILE:-auto}"; PIN="${AC_PIN:-}"; FROM="${AC_FROM:-}"
NO_MODIFY_PATH=0; PRINT=0; VERBOSE=0
usage() {  # the header above when this runs as a file; a short form through curl | sh
    if [ -f "$0" ] && head -n 3 "$0" 2>/dev/null | grep -q "AbstractCore standalone installer"; then
        awk 'NR > 4 && /^# =+$/ { exit } NR > 4 { sub(/^# ?/, ""); print }' "$0"
    else printf 'Usage: install.sh [--profile auto|light|apple|gpu] [--pin VERSION|latest] [--from PATH] [--print]\nFull help: %s\n' "$AC_DOCS"; fi
}
need_arg() { [ $# -ge 2 ] && [ -n "$2" ] || { echo "ERROR: $1 needs a value" >&2; exit 2; }; }
while [ $# -gt 0 ]; do
    case "$1" in
        --profile) need_arg "$@"; PROFILE="$2"; shift ;;  --profile=*) PROFILE="${1#*=}" ;;
        --pin) need_arg "$@"; PIN="$2"; shift ;;          --pin=*) PIN="${1#*=}" ;;
        --from) need_arg "$@"; FROM="$2"; shift ;;        --from=*) FROM="${1#*=}" ;;
        --no-modify-path) NO_MODIFY_PATH=1 ;;
        --print|--dry-run|-n) PRINT=1 ;;
        -v|--verbose) VERBOSE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

C_B=""; C_D=""; C_R=""; C_G=""; C_Y=""; C_C=""; C_0=""
if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-}" != "dumb" ]; then E="$(printf '\033')"
    C_B="${E}[1m"; C_D="${E}[2m"; C_R="${E}[31m"; C_G="${E}[32m"; C_Y="${E}[33m"; C_C="${E}[36m"; C_0="${E}[0m"; fi
STEP=0; step() { STEP=$((STEP + 1)); printf '\n%s[%s] %s%s\n' "$C_B" "$STEP" "$1" "$C_0"; }
ok()   { printf '  %s✓%s %s\n' "$C_G" "$C_0" "$1"; }
info() { printf '  %s·%s %s\n' "$C_C" "$C_0" "$1"; }
warn() { printf '  %s!%s %s\n' "$C_Y" "$C_0" "$1"; }
die()  { printf '\n%sERROR: %s%s\n' "$C_R" "$1" "$C_0" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
q() {  # shell-quote one word for display
    case "$1" in
        ''|*[!A-Za-z0-9_./:=@,+%-]*) printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")" ;;
        *) printf '%s' "$1" ;;
    esac
}
show_cmd() { _l=""; for _w in "$@"; do _l="$_l $(q "$_w")"; done; printf '%s' "${_l# }"; }

# run DESCRIPTION CMD...: print the command, then run it (output in the log; live as well with
# RUN_LIVE=1 or -v). RUN_SOFT=1: a failure warns and returns 1 instead of stopping the install.
LOG=""; RUN_RC=0; RUN_SOFT=0; RUN_LIVE=0; RUN_SHOW=""
run() {
    _desc="$1"; shift
    _shown="${RUN_SHOW:-$(show_cmd "$@")}"; RUN_SHOW=""
    _soft="$RUN_SOFT"; _live="$RUN_LIVE"; RUN_SOFT=0; RUN_LIVE=0; RUN_RC=0
    printf '  %s$ %s%s\n' "$C_D" "$_shown" "$C_0"
    [ "$PRINT" = 1 ] && return 0
    printf '\n$ %s\n' "$_shown" >>"$LOG"
    if [ "$VERBOSE" = 1 ] || [ "$_live" = 1 ]; then
        echo 0 >"$LOG.rc"
        { "$@" 2>&1 || echo "$?" >"$LOG.rc"; } | tee -a "$LOG"
        RUN_RC="$(cat "$LOG.rc")"
    else "$@" >>"$LOG" 2>&1 || RUN_RC=$?; fi
    [ "$RUN_RC" = 0 ] && return 0
    if [ "$_soft" = 1 ]; then warn "$_desc did not succeed (exit $RUN_RC; log: $LOG)"; return 1; fi
    tail -n 15 "$LOG" | sed 's/^/    /' >&2
    die "$_desc failed (exit $RUN_RC). Log: $LOG
What to do: run the installer again (it repairs a half-finished install); if it stops at the same step, report it with that log ($AC_DOCS)."
}
http() {  # http URL: the body on stdout; non-zero unless the answer is 2xx
    if have curl; then curl -fsS --connect-timeout 5 --max-time 20 "$1" 2>/dev/null; else wget -qO- --timeout=20 "$1" 2>/dev/null; fi
}
UV=""; find_uv() {
    if have uv; then UV="$(command -v uv)"; return 0; fi
    for _c in "${UV_INSTALL_DIR:-}/uv" "${XDG_BIN_HOME:-}/uv" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
        if [ -x "$_c" ]; then UV="$_c"; return 0; fi
    done
    return 1
}
# NVIDIA, as the framework installer detects it: nvidia-smi answers -L. NV_NAME: the first GPU.
nvidia_info() {
    NV_OK=0; NV_NAME=""
    have nvidia-smi && nvidia-smi -L >/dev/null 2>&1 || return 0
    NV_OK=1; NV_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | sed 's/^ *//')" || NV_NAME=""
}

# --- 1. Preflight -------------------------------------------------------------
OS="$(uname -s)"; ARCH="$(uname -m)"
case "$OS" in
    Darwin) OS_ID=macos ;;
    Linux)  OS_ID=linux ;;
    MINGW*|MSYS*|CYGWIN*) die "Windows: use powershell -ExecutionPolicy ByPass -c \"irm https://abstractframework.ai/core/install.ps1 | iex\"" ;;
    *) die "unsupported OS: $OS (supported: macOS, Linux; Windows uses install.ps1)" ;;
esac
# A Terminal set to "Open using Rosetta" reports x86_64 on Apple Silicon: an Intel Python, no MLX.
if [ "$OS_ID" = macos ] && [ "$(sysctl -n sysctl.proc_translated 2>/dev/null || echo 0)" = 1 ]; then
    die "this Terminal runs in Intel (Rosetta) mode on an Apple Silicon Mac, so the installer would set up the
slow Intel version without the Apple Silicon engines. Run it natively: curl -LsSf $AC_SCRIPT_URL | arch -arm64 sh"
fi
MACOS_VERSION=""; MACOS_MAJOR=0
[ "$OS_ID" = macos ] && { MACOS_VERSION="$(sw_vers -productVersion 2>/dev/null || echo 0)"; MACOS_MAJOR="${MACOS_VERSION%%.*}"; }
MUSL=0; [ "$OS_ID" = linux ] && ldd --version 2>&1 | grep -qi musl && MUSL=1

printf '%sAbstractCore installer%s  %s%s%s\n' "$C_B" "$C_0" "$C_D" \
    "$([ "$PRINT" = 1 ] && echo '(--print: preflight only, nothing is installed)' || echo "$AC_SCRIPT_URL")" "$C_0"
step "Preflight"
ok "system: $OS_ID $ARCH${MACOS_VERSION:+ (macOS $MACOS_VERSION)}"

find_uv || true
TOOL_BIN=""; TOOL_DIR=""
[ -z "$UV" ] || { TOOL_BIN="$("$UV" tool dir --bin 2>/dev/null || true)"; TOOL_DIR="$("$UV" tool dir 2>/dev/null || true)"; }
[ -n "$TOOL_BIN" ] || TOOL_BIN="${UV_TOOL_BIN_DIR:-${XDG_BIN_HOME:-$HOME/.local/bin}}"
[ -n "$TOOL_DIR" ] || TOOL_DIR="${UV_TOOL_DIR:-${XDG_DATA_HOME:-$HOME/.local/share}/uv/tools}"
BEFORE=""; PREV_PROFILE=""
if [ -n "$UV" ]; then  # the existing install: its version, and its profile from the uv receipt
    BEFORE="$("$UV" tool list 2>/dev/null | sed -n 's/^abstractcore v\([^ ]*\).*/\1/p' | head -n 1)"
    _rcpt="$TOOL_DIR/abstractcore/uv-receipt.toml"
    if [ -n "$BEFORE" ] && grep -q 'name = "abstractcore"' "$_rcpt" 2>/dev/null; then
        _extras="$(sed -n 's/.*name = "abstractcore", extras = \[\([^]]*\)\].*/\1/p' "$_rcpt" 2>/dev/null | head -n 1)"
        case "$_extras" in
            *'"apple"'*|*'"all-apple"'*|*'"mlx"'*) PREV_PROFILE=apple ;;
            *'"gpu"'*|*'"all-gpu"'*|*'"vllm"'*) PREV_PROFILE=gpu ;;
            '') PREV_PROFILE=light ;;
        esac
    fi
fi

nvidia_info; HAS_NVIDIA="$NV_OK"
HAS_ROCM=0; have rocminfo && rocminfo >/dev/null 2>&1 && HAS_ROCM=1
APPLE_OK=0; [ "$OS_ID" = macos ] && [ "$ARCH" = arm64 ] && [ "$MACOS_MAJOR" -ge 14 ] 2>/dev/null && APPLE_OK=1
case "$PROFILE" in
    auto|"")
        if [ -n "$PREV_PROFILE" ]; then PROFILE="$PREV_PROFILE"; _why="kept from the previous install"
        elif [ "$APPLE_OK" = 1 ]; then PROFILE=apple; _why="Apple Silicon, macOS $MACOS_VERSION"
        elif [ "$OS_ID" = macos ] && [ "$ARCH" = arm64 ]; then PROFILE=light
            _why="macOS $MACOS_VERSION: the Apple Silicon engines (MLX) need macOS 14 or later; update macOS and run the installer again"
        elif [ "$HAS_NVIDIA" = 1 ]; then PROFILE=gpu; _why="nvidia-smi found ${NV_NAME:-a GPU}"
        elif [ "$HAS_ROCM" = 1 ]; then PROFILE=gpu; _why="rocminfo found a GPU"
        else PROFILE=light; _why="no GPU found, so remote providers and servers only"; fi
        ok "profile: $PROFILE ($_why; override with --profile)" ;;
    light) ok "profile: light (every remote provider, the tools and the HTTP server; no local engine)" ;;
    apple)
        [ "$OS_ID" = macos ] && [ "$ARCH" = arm64 ] || die "the apple profile needs an Apple Silicon Mac (this is $OS_ID $ARCH); use --profile light"
        [ "$MACOS_MAJOR" -ge 14 ] 2>/dev/null || die "the apple profile needs macOS 14 or later (this is $MACOS_VERSION); use --profile light"
        ok "profile: apple (MLX and the other local engines on the Apple GPU)" ;;
    gpu)
        [ "$OS_ID" = linux ] || die "the gpu profile targets Linux (Windows: install.ps1); on a Mac use --profile apple"
        [ "$HAS_NVIDIA" = 1 ] || [ "$HAS_ROCM" = 1 ] || warn "gpu profile requested but neither nvidia-smi nor rocminfo works: its engines will run on the processor"
        ok "profile: gpu (CUDA/ROCm engines when a card works, else the processor)" ;;
    *) die "unknown profile '$PROFILE' (expected auto, light, apple or gpu)" ;;
esac
case "$PROFILE" in apple) EXTRAS=apple ;; gpu) EXTRAS=gpu ;; *) EXTRAS="" ;; esac

# llama.cpp (GGUF): upstream's prebuilt wheel for this machine (see the top of this script).
GGUF=0; GGUF_PIN=""; GGUF_KIND=""; GGUF_LINKS=""; GGUF_RESULT="not applicable ($PROFILE profile)"
case "$PROFILE/$OS_ID/$ARCH" in
    apple/macos/arm64) GGUF_PIN="$AC_LLAMA_METAL_PIN"; GGUF_KIND=metal ;;
    gpu/linux/x86_64|gpu/linux/aarch64|gpu/linux/arm64) [ "$MUSL" = 1 ] || { GGUF_PIN="$AC_LLAMA_CPU_PIN"; GGUF_KIND=cpu; } ;;
esac
if [ -n "$GGUF_KIND" ]; then
    GGUF=1; GGUF_LINKS="$AC_LLAMA_INDEX/$GGUF_KIND/llama-cpp-python/"
    ok "llama.cpp (GGUF): llama-cpp-python $GGUF_PIN, upstream's prebuilt $GGUF_KIND wheel"
elif [ -n "$EXTRAS" ]; then
    GGUF_RESULT="skipped: no prebuilt llama.cpp wheel for $OS_ID $ARCH$([ "$MUSL" = 1 ] && echo ' (musl)'), and this installer never starts a compiler"
    info "llama.cpp (GGUF) $GGUF_RESULT"
fi

if [ -n "$FROM" ]; then
    [ -e "$FROM" ] || die "--from: no such file or folder: $FROM"
    FROM="$(cd "$(dirname "$FROM")" && pwd)/$(basename "$FROM")"; TARGET="(the build from $FROM)"; PIN_WHY="--from"
elif [ "$PIN" = latest ]; then
    _latest="$(http https://pypi.org/pypi/abstractcore/json | grep -o '"version": *"[^"]*"' | head -n 1 | sed 's/.*"\([^"]*\)"$/\1/')" || _latest=""
    TARGET="${_latest:-the newest release on PyPI}"; PIN_WHY="--pin latest"
else
    if [ -n "$PIN" ]; then PIN_WHY="--pin"; else PIN="$AC_PIN_DEFAULT"; PIN_WHY="the release this installer ships with"; fi; TARGET="$PIN"
fi
spec() {  # spec EXTRAS: the requirement given to uv
    if [ -n "$FROM" ]; then printf 'abstractcore%s @ file://%s' "${1:+[$1]}" "$FROM"
    elif [ "$PIN" = latest ]; then printf 'abstractcore%s' "${1:+[$1]}"
    else printf 'abstractcore%s==%s' "${1:+[$1]}" "$PIN"; fi
}
ok "package: $(spec "$EXTRAS")  ${C_D}(version: $PIN_WHY)${C_0}"
if [ -z "$BEFORE" ]; then ok "No AbstractCore install found: installing AbstractCore $TARGET"
elif [ -n "$FROM" ]; then ok "AbstractCore $BEFORE found: replacing it with the build from $FROM (--from)"
elif [ "$BEFORE" = "$TARGET" ]; then ok "AbstractCore $BEFORE found: already up to date (checked, repaired if needed)"
elif [ "$PIN" != latest ] && [ "$(printf '%s\n%s\n' "$BEFORE" "$PIN" | sort -V | tail -n 1)" = "$BEFORE" ]; then
    ok "AbstractCore $BEFORE found: going back to AbstractCore $TARGET (--pin)"
else ok "AbstractCore $BEFORE found: upgrading to AbstractCore $TARGET"; fi

# A command of the same name from another program (often AbstractFramework's gateway tool, which
# exposes AbstractCore's commands itself): uv would refuse the install, so stop and say why.
if [ -z "$BEFORE" ]; then
    for _n in $AC_COMMANDS; do
        [ -e "$TOOL_BIN/$_n" ] || [ -L "$TOOL_BIN/$_n" ] || continue
        _owner="$([ -z "$UV" ] || "$UV" tool list 2>/dev/null | awk -v n="$_n" '/^[^ -]/ { t = $1 } $1 == "-" && $2 == n { print t; exit }')"
        _msg="$TOOL_BIN/$_n already exists ($([ -n "$_owner" ] && echo "from the uv tool '$_owner'" || echo "another program's"))"
        [ "$_owner" != abstractgateway ] || _msg="$_msg: AbstractFramework is installed here and already includes AbstractCore; upgrade it with: curl -LsSf https://abstractframework.ai/install.sh | sh"
        [ "$PRINT" = 1 ] && { warn "$_msg (the real run stops here; nothing is changed)"; break; }
        die "$_msg
Nothing was changed."
    done
fi

# Measured with uv 0.13.7 (environment + uv cache + Python): gpu 13 GB on Ubuntu 24.04 x86_64
# (vLLM, PyTorch CUDA), light 1.1 GB; apple is the framework installer's estimate.
case "$PROFILE" in apple) NEED_MB=8000 ;; gpu) NEED_MB=14000 ;; *) NEED_MB=1500 ;; esac
FREE_MB="$(df -Pk "$HOME" 2>/dev/null | awk 'NR == 2 { print int($4 / 1024) }')" || FREE_MB=""
if [ -z "$FREE_MB" ]; then :
elif [ "$FREE_MB" -lt 500 ]; then die "only $FREE_MB MB free in $HOME (about $NEED_MB MB needed)"
elif [ "$FREE_MB" -lt "$NEED_MB" ]; then warn "only $FREE_MB MB free in $HOME; the $PROFILE profile needs about $NEED_MB MB (models need more)"
else ok "disk: $FREE_MB MB free (about $NEED_MB MB needed, models extra)"; fi
if http https://pypi.org/simple/pip/ >/dev/null; then ok "internet: pypi.org reachable"
elif [ "$PRINT" = 1 ]; then warn "pypi.org is not reachable (a real run stops here)"
else die "this computer cannot reach pypi.org, where AbstractCore is downloaded (check the connection or HTTPS_PROXY). Nothing was changed."; fi
if [ -z "$FROM" ] && [ "$PIN" != latest ] && ! http "https://pypi.org/pypi/abstractcore/$PIN/json" >/dev/null; then
    _m="abstractcore $PIN is not on PyPI (yet): use --pin latest for the newest published version"
    if [ "$PRINT" = 1 ]; then warn "$_m"; else die "$_m. Nothing was changed."; fi
fi
DL=""; if have curl; then DL="curl -LsSf"; elif have wget; then DL="wget -qO-"; fi
[ -n "$UV" ] || [ -n "$DL" ] || die "neither curl nor wget is installed, so uv cannot be downloaded"
WORK="${XDG_CACHE_HOME:-$HOME/.cache}/abstractcore/installer"
[ "$PRINT" = 1 ] || { mkdir -p "$WORK"; LOG="$WORK/install.log"; : >"$LOG"; }

# --- 2. uv and Python ---------------------------------------------------------
step "uv (Python toolchain manager)"
if [ -n "$UV" ]; then ok "uv found: $UV ($("$UV" --version 2>/dev/null | awk '{print $2}'))"
else
    info "installing uv from astral.sh into ~/.local/bin (no admin)"
    RUN_SHOW="$DL https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh"
    run "install uv" sh -c "$DL https://astral.sh/uv/install.sh | env UV_NO_MODIFY_PATH=1 sh"
    if [ "$PRINT" = 1 ]; then UV=uv
    else find_uv || die "uv was installed but cannot be found (looked in \$UV_INSTALL_DIR, \$XDG_BIN_HOME, ~/.local/bin)"
        ok "uv installed: $UV"; TOOL_BIN="$("$UV" tool dir --bin)"; TOOL_DIR="$("$UV" tool dir)"; fi
fi
step "Python $AC_PYTHON (managed by uv, isolated from any system Python)"
run "install Python $AC_PYTHON" "$UV" python install "$AC_PYTHON"

# --- 3. AbstractCore ----------------------------------------------------------
step "AbstractCore"
# Prebuilt wheels only (see the top of this script): the apple and gpu trees get the overrides
# file and --no-build-package; uv splits --overrides at whitespace, so the file is named relatively
# from its folder. The light tree has no such package and needs neither.
uv_overrides() {  # uv_overrides GGUF: one line per dropped package
    for _p in $AC_NO_WHEEL; do echo "$_p; sys_platform == 'never'"; done
    [ "$1" = 1 ] || echo "llama-cpp-python; sys_platform == 'never'"
}
install_tool() {  # install_tool GGUF SOFT [uv options]
    _gguf="$1"; _soft="$2"; shift 2
    set -- "$UV" tool install --python "$AC_PYTHON" "$@"
    if [ -n "$EXTRAS" ]; then
        if [ "$PRINT" = 1 ]; then info "uv-overrides.txt:"; uv_overrides "$_gguf" | sed 's/^/      /'
        else uv_overrides "$_gguf" >"$WORK/uv-overrides.txt"; fi
        set -- "$@" --overrides uv-overrides.txt
        for _p in $AC_NO_WHEEL llama-cpp-python; do set -- "$@" --no-build-package "$_p"; done
        [ "$_gguf" = 0 ] || set -- "$@" --with "llama-cpp-python==$GGUF_PIN" --find-links "$GGUF_LINKS"
    fi
    [ -z "$FROM" ] || set -- "$@" --reinstall-package abstractcore  # same version number, new code
    # --pin latest: without --upgrade, uv keeps an installed version that still satisfies.
    [ "$PIN" != latest ] || [ -n "$FROM" ] || set -- "$@" --upgrade
    set -- "$@" "$(spec "$EXTRAS")"
    RUN_SHOW="cd $(q "$WORK") && $(show_cmd "$@")"
    _cwd="$(pwd)"; [ "$PRINT" = 1 ] || cd "$WORK"
    _rc=0; RUN_LIVE=1 RUN_SOFT="$_soft" run "install $(spec "$EXTRAS")$([ "$_gguf" = 1 ] && echo " with llama.cpp's $GGUF_KIND wheel")" "$@" || _rc=1
    cd "$_cwd" 2>/dev/null || cd "$HOME"
    return "$_rc"
}
if [ "$GGUF" = 1 ]; then
    if install_tool 1 1; then
        GGUF_RESULT="llama-cpp-python $GGUF_PIN (upstream's prebuilt $GGUF_KIND wheel)"
        [ "$PRINT" = 0 ] || GGUF_RESULT="planned: $GGUF_RESULT; if it does not install, the install is retried without it"
    else
        GGUF=0; GGUF_RESULT="skipped: the prebuilt $GGUF_KIND wheel did not install here (log: $LOG); GGUF models need llama.cpp another way"
        warn "retrying without llama.cpp"; install_tool 0 0
    fi
else install_tool 0 0; fi
# Repair: uv can report the tool installed while its command is missing or cannot start (an
# interrupted install) and then does nothing, so check the command and reinstall in place.
if [ "$PRINT" = 0 ] && ! "$TOOL_BIN/abstractcore" --help >/dev/null 2>&1; then
    warn "the installed abstractcore does not start (an earlier install was interrupted or damaged): reinstalling it"
    install_tool "$GGUF" 0 --reinstall
    "$TOOL_BIN/abstractcore" --help >/dev/null 2>&1 || die "abstractcore still does not start after reinstalling it (log: $LOG)"
fi
AFTER="$BEFORE"
if [ "$PRINT" = 0 ]; then
    AFTER="$("$UV" tool list 2>/dev/null | sed -n 's/^abstractcore v\([^ ]*\).*/\1/p' | head -n 1)"
    if [ -z "$BEFORE" ]; then ok "installed abstractcore $AFTER"
    elif [ "$BEFORE" != "$AFTER" ]; then ok "abstractcore $BEFORE -> $AFTER"
    elif [ -n "$FROM" ]; then ok "abstractcore $AFTER reinstalled from $FROM"
    else ok "abstractcore $AFTER already installed"; fi
fi

# --- 4. PATH and summary ------------------------------------------------------
step "PATH"
case ":$PATH:" in
    *":$TOOL_BIN:"*) ok "$TOOL_BIN is on PATH" ;;
    *)
        _hand="add it yourself: export PATH=\"$TOOL_BIN:\$PATH\" in your shell's startup file"
        if grep -qsF "$TOOL_BIN" "$HOME/.zshenv" "$HOME/.zshrc" "$HOME/.bashrc" "$HOME/.bash_profile" \
            "$HOME/.profile" "${XDG_CONFIG_HOME:-$HOME/.config}/fish/conf.d/uv.env.fish"; then
            ok "$TOOL_BIN is already on PATH in your shell profile (new terminals have it)"
        elif [ "$NO_MODIFY_PATH" = 1 ]; then warn "$TOOL_BIN is not on PATH (--no-modify-path); $_hand"
        elif RUN_SOFT=1 run "add $TOOL_BIN to PATH in your shell profile" "$UV" tool update-shell; then
            info "open a new terminal for the 'abstractcore' command to be on PATH"
        else info "$_hand"; fi ;;
esac

printf '\n%s%s%s\n' "$C_B" "$([ "$PRINT" = 1 ] && echo 'Plan (--print: nothing was changed)' || echo 'AbstractCore is installed')" "$C_0"
_ver="${AFTER:-$TARGET}"; [ "$PRINT" = 1 ] && _ver="$TARGET"
echo "  version:   $_ver$([ "$PRINT" = 0 ] && [ -n "$BEFORE" ] && [ "$BEFORE" != "$AFTER" ] && echo " (was $BEFORE)")"
echo "  profile:   $PROFILE ($(spec "$EXTRAS"))"
echo "  llama.cpp: $GGUF_RESULT"
echo "  commands:  $TOOL_BIN/abstractcore, abstractcore-config, abstractcore-chat, abstractcore-endpoint"
echo "             and the apps summarizer, extractor, judge, intent, deepsearch (each also abstractcore-<app>)"
[ "$PRINT" = 1 ] || echo "  log:       $LOG"
echo
echo "Try it:"
echo "  abstractcore --help"
echo "  abstractcore config defaults"
echo "Docs: $AC_DOCS"
