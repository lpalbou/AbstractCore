"""Will this model fit on this machine? A verdict BEFORE the download.

`utils.context_estimate` answers the question for weights already on disk
(real geometry, real file sizes). A model browser has to answer it for weights
that are NOT here yet, from the only data obtainable without fetching them:
a parameter count, a quantization, sometimes an exact artifact size, sometimes
the model geometry. The formulas are the ones fixed in the models exploration
(explore-models.md section 7):

    W    weight bytes      exact (artifact size) or P x bits / 8 x 1.03
    KV   n x 2 x layers x kv_heads x head_dim x 2   (f16)
         unknown geometry -> n x 0.5 MiB x (P / 8e9), labelled rough
    O    overhead          max(0.5 GiB, 5% of W)
    C    host ceiling      host_profile.ceiling_bytes
    Ceff C - max(2 GiB, 5% of C)            (the context estimator's reserve)
                           returned as `usable_bytes` (and the reserve as
                           `reserve_bytes`): the verdict compares `need_bytes`
                           with THIS, never with the ceiling itself

    need = W + KV(n) + O
    fits       need <= 0.8 Ceff
    tight      need <= Ceff
    too_large  otherwise; on CUDA "partial_offload" when W <= VRAM + 0.75 RAM
    fits_now   need <= free_now_bytes
    disk_ok    download_bytes <= disk_free - 5 GiB
    max_ctx    (Ceff - W - O) // kv_bytes_per_token, clamped to max_tokens

MoE models use TOTAL parameters: every expert is resident even though only a
few are active per token (active parameters change speed, not memory).

ADVISORY ONLY. Nothing gates a download or a load on this verdict; it exists
so a human can choose before spending bytes. `unknown` is a legal verdict and
is returned whenever an input the formula needs is missing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

__all__ = [
    "FIT_VERDICTS",
    "QUANT_BITS",
    "bits_for_quant",
    "parse_param_count",
    "parse_params_from_name",
    "estimate_fit",
]

GIB = 1024**3
MIB = 1024**2

FIT_VERDICTS = ("fits", "tight", "too_large", "partial_offload", "unknown")
CONFIDENCE_ORDER = ("exact", "estimated", "rough", "unknown")

# Effective bits per weight INCLUDING quantization metadata (scales, zero
# points, the few layers kept at higher precision). These are the numbers
# that turn "8B at Q4_K_M" into bytes on disk; a bare "4" would under-count
# every quantized artifact by 10-20%.
QUANT_BITS: Dict[str, float] = {
    # MLX / generic bit-width labels (group-wise affine: +0.5 bit of scales)
    "2bit": 2.5,
    "3bit": 3.5,
    "4bit": 4.5,
    "5bit": 5.5,
    "6bit": 6.5,
    "8bit": 8.5,
    "mxfp4": 4.25,
    "nvfp4": 4.5,
    "dwq": 4.5,
    # llama.cpp GGUF
    "q2_k": 3.0,
    "q3_k_s": 3.5,
    "q3_k_m": 3.9,
    "q3_k_l": 4.3,
    "q4_0": 4.55,
    "q4_1": 5.0,
    "q4_k_s": 4.6,
    "q4_k_m": 4.85,
    "q4_k_xl": 4.95,
    "q5_0": 5.5,
    "q5_1": 6.0,
    "q5_k_s": 5.55,
    "q5_k_m": 5.7,
    "q6_k": 6.6,
    "q8_0": 8.5,
    "iq1_s": 1.6,
    "iq1_m": 1.75,
    "iq2_xxs": 2.1,
    "iq2_xs": 2.3,
    "iq2_s": 2.5,
    "iq2_m": 2.7,
    "iq3_xxs": 3.1,
    "iq3_xs": 3.3,
    "iq3_s": 3.45,
    "iq3_m": 3.65,
    "iq4_xs": 4.25,
    "iq4_nl": 4.5,
    # Float / integer formats
    "fp8": 8.0,
    "f8": 8.0,
    "int8": 8.0,
    "int4": 4.5,
    "fp16": 16.0,
    "f16": 16.0,
    "bf16": 16.0,
    "fp32": 32.0,
    "f32": 32.0,
}


def bits_for_quant(quant: Any) -> Optional[float]:
    """Effective bits/weight for a quant label, or None when unrecognised.

    Tolerant of the spellings engines actually print: `Q4_K_M`, `q4_k_m`,
    `4bit`, `4-bit`, `8BIT`, `fp8`, `MXFP4`, `UD-Q4_K_XL`.
    """

    raw = str(quant or "").strip().lower()
    if not raw:
        return None
    raw = raw.replace("-", "_").replace(" ", "")
    if raw.startswith("ud_"):
        raw = raw[3:]
    if raw in QUANT_BITS:
        return QUANT_BITS[raw]
    m = re.fullmatch(r"(\d+(?:\.\d+)?)_?bits?", raw)
    if m:
        key = f"{int(float(m.group(1)))}bit"
        if key in QUANT_BITS:
            return QUANT_BITS[key]
        return float(m.group(1)) + 0.5
    # A longer label that starts with a known one (`q4_k_m_imat`, `4bit_dwq`).
    for key in sorted(QUANT_BITS, key=len, reverse=True):
        if raw.startswith(key):
            return QUANT_BITS[key]
    return None


_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9, "t": 1e12}


def parse_param_count(value: Any) -> Optional[int]:
    """`"8.2B"` / `"30.5B"` / `"22M"` / `"1.6T"` / `8200000000` -> int, else None.

    `"35B-A3B"` returns the TOTAL (35B); use `parse_params_from_name` to also
    get the active count.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value) if value > 0 else None
    raw = str(value or "").strip().lower().replace(",", "")
    if not raw:
        return None
    m = re.match(r"^(\d+(?:\.\d+)?)\s*([kmbt])?", raw)
    if not m:
        return None
    number = float(m.group(1))
    unit = m.group(2)
    if unit is None:
        return int(number) if number >= 1e6 else None
    return int(round(number * _SUFFIX[unit]))


# `-8b`, `_0.6B`, `:30b`, `-35B-A3B`; never the `4b` of `4bit` and never the
# `e4b` of Gemma's "effective" sizes (a count of something else).
_NAME_TOTAL_RE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)([bm])(?![a-z0-9])", re.IGNORECASE)
_NAME_ACTIVE_RE = re.compile(r"(?<![a-z0-9])a(\d+(?:\.\d+)?)b(?![a-z0-9])", re.IGNORECASE)
_NAME_EXPERTS_RE = re.compile(r"(?<![a-z0-9])(\d+)x(\d+(?:\.\d+)?)b(?![a-z0-9])", re.IGNORECASE)


def parse_params_from_name(name: Any) -> Tuple[Optional[int], Optional[int]]:
    """`(total, active)` parameter counts read off a repo/tag name.

    `Qwen3-30B-A3B` -> (30e9, 3e9); `qwen3:8b` -> (8e9, None);
    `Mixtral-8x7B` -> (56e9, None) -- an upper bound, shared layers make the
    real count lower, which is why callers label name-derived counts `rough`.
    """

    text = str(name or "")
    if not text:
        return None, None
    tail = text.rsplit("/", 1)[-1]
    active: Optional[int] = None
    m_active = _NAME_ACTIVE_RE.search(tail)
    if m_active:
        active = int(round(float(m_active.group(1)) * 1e9))
    m_experts = _NAME_EXPERTS_RE.search(tail)
    if m_experts:
        return int(round(int(m_experts.group(1)) * float(m_experts.group(2)) * 1e9)), active
    for m in _NAME_TOTAL_RE.finditer(tail):
        start = m.start()
        if start > 0 and tail[start - 1].lower() == "a" and (start < 2 or not tail[start - 2].isalnum()):
            continue  # that's the active count
        unit = m.group(2).lower()
        return int(round(float(m.group(1)) * (1e9 if unit == "b" else 1e6))), active
    return None, active


def _worst(*levels: str) -> str:
    worst = 0
    for level in levels:
        try:
            worst = max(worst, CONFIDENCE_ORDER.index(level))
        except ValueError:
            worst = max(worst, CONFIDENCE_ORDER.index("unknown"))
    return CONFIDENCE_ORDER[worst]


def _kv_bytes_per_token(geometry: Optional[Mapping[str, Any]]) -> Optional[int]:
    if not isinstance(geometry, Mapping):
        return None
    try:
        layers = int(geometry.get("n_layers") or 0)
        kv_heads = int(geometry.get("n_kv_heads") or 0)
        head_dim = int(geometry.get("head_dim") or 0)
    except Exception:
        return None
    if layers <= 0 or kv_heads <= 0 or head_dim <= 0:
        return None
    return 2 * layers * kv_heads * head_dim * 2


def estimate_fit(
    *,
    host: Optional[Mapping[str, Any]] = None,
    params_total: Optional[int] = None,
    params_source: str = "catalog",
    bits: Optional[float] = None,
    quant: Optional[str] = None,
    weight_bytes: Optional[int] = None,
    download_bytes: Optional[int] = None,
    geometry: Optional[Mapping[str, Any]] = None,
    context: Optional[int] = None,
    max_tokens: Optional[int] = None,
    disk_free_bytes: Optional[int] = None,
) -> Dict[str, Any]:
    """The contract-C `fit` block for one artifact on one host.

    Inputs (all optional; missing ones lower the confidence or yield unknown):
      host            a `host_profile()` dict (ceiling/free/vram/ram/accelerator)
      params_total    P, total parameters (MoE: total, not active)
      params_source   "catalog" | "hf_api" | "engine" -> estimated; "name" -> rough
      bits / quant    effective bits per weight, or a quant label to look up
      weight_bytes    exact resident weight size when known (artifact size)
      download_bytes  bytes to fetch (defaults to weight_bytes) for the disk check
      geometry        {n_layers, n_kv_heads, head_dim} for exact KV math
      context         n tokens to budget KV for (default min(8192, max_tokens))
      max_tokens      the model's context window (clamps max_context)
      disk_free_bytes free space where the artifact will land
    """

    host = dict(host or {})
    notes: List[str] = []

    ceiling = host.get("ceiling_bytes")
    ceiling = int(ceiling) if isinstance(ceiling, (int, float)) and ceiling > 0 else None
    free_now = host.get("free_now_bytes")
    free_now = int(free_now) if isinstance(free_now, (int, float)) and free_now >= 0 else None

    # --- bits -----------------------------------------------------------------
    b: Optional[float] = float(bits) if isinstance(bits, (int, float)) and bits > 0 else None
    if b is None and quant:
        b = bits_for_quant(quant)
        if b is None:
            notes.append(f"unrecognised quantization {quant!r}")

    # --- W: weight bytes -------------------------------------------------------
    P = int(params_total) if isinstance(params_total, (int, float)) and params_total > 0 else None
    weights_conf = "unknown"
    W: Optional[int] = None
    if isinstance(weight_bytes, (int, float)) and weight_bytes > 0:
        W = int(weight_bytes)
        weights_conf = "exact"
    elif P is not None:
        if b is None:
            b = 16.0
            notes.append("quantization unknown; assuming 16-bit weights")
            weights_conf = "rough"
        else:
            weights_conf = "rough" if params_source == "name" else "estimated"
        W = int(P * b / 8 * 1.03)
        notes.append(f"weights estimated from {P / 1e9:.2f}B params x {b:g} bits")
    if P is None and W is not None:
        # Parameters back-derived from the artifact size, for the rough KV rule.
        P = int(W * 8 / (b or 4.85))

    # --- n: context budgeted ---------------------------------------------------
    mt = int(max_tokens) if isinstance(max_tokens, (int, float)) and max_tokens > 0 else None
    if isinstance(context, (int, float)) and context > 0:
        n = int(context)
    else:
        n = min(8192, mt) if mt else 8192

    # --- KV --------------------------------------------------------------------
    kv_per_token = _kv_bytes_per_token(geometry)
    kv_conf = "exact"
    if kv_per_token is None:
        if P is not None:
            kv_per_token = int(0.5 * MIB * (P / 8e9))
            kv_conf = "rough"
            notes.append("KV cache estimated without model geometry (0.5 MiB/token per 8B params)")
        else:
            kv_conf = "unknown"
    KV = int(n * kv_per_token) if kv_per_token else None

    base = {
        "verdict": "unknown",
        "need_bytes": None,
        "weight_bytes": W,
        "kv_bytes": KV,
        "context": n,
        "ceiling_bytes": ceiling,
        # What the verdict actually compares `need_bytes` against: the ceiling
        # minus the reserve kept for the system (`Ceff`), and the overhead
        # inside `need_bytes`. A sentence that states the ceiling while the
        # verdict used `usable_bytes` contradicts itself (mission KK).
        "usable_bytes": None,
        "reserve_bytes": None,
        "overhead_bytes": None,
        "free_now_bytes": free_now,
        "fits_now": None,
        "disk_ok": None,
        "max_context": None,
        "confidence": "unknown",
        "notes": notes,
    }

    # --- disk ------------------------------------------------------------------
    D_need = download_bytes if isinstance(download_bytes, (int, float)) and download_bytes > 0 else W
    if isinstance(disk_free_bytes, (int, float)) and D_need:
        base["disk_ok"] = bool(D_need <= disk_free_bytes - 5 * GIB)
        if not base["disk_ok"]:
            notes.append(
                f"not enough disk: needs {D_need / 1e9:.1f} GB + 5 GiB headroom, "
                f"{disk_free_bytes / 1e9:.1f} GB free"
            )

    if W is None:
        notes.append("no parameter count and no artifact size: cannot size the weights")
        return base
    if ceiling is None:
        notes.append("host memory ceiling unknown")
        return base

    O = int(max(0.5 * GIB, 0.05 * W))
    need = W + (KV or 0) + O
    c_eff = ceiling - max(2 * GIB, int(0.05 * ceiling))
    base["need_bytes"] = int(need)
    base["usable_bytes"] = int(c_eff)
    base["reserve_bytes"] = int(ceiling - c_eff)
    base["overhead_bytes"] = int(O)

    if need <= 0.8 * c_eff:
        verdict = "fits"
    elif need <= c_eff:
        verdict = "tight"
        notes.append("fits with little headroom; close other models first")
    else:
        verdict = "too_large"
        if host.get("accelerator") == "cuda":
            vram = host.get("vram_bytes") or ceiling
            ram = host.get("ram_bytes")
            if isinstance(ram, (int, float)) and W <= vram + 0.75 * ram:
                verdict = "partial_offload"
                notes.append("exceeds VRAM; runs with layers offloaded to system RAM (slower)")
    base["verdict"] = verdict

    if free_now is not None:
        base["fits_now"] = bool(need <= free_now)
        if verdict in ("fits", "tight") and not base["fits_now"]:
            notes.append("not enough free memory right now: unload something first")

    if kv_per_token:
        room = c_eff - W - O
        max_ctx = max(0, int(room // kv_per_token))
        if mt:
            max_ctx = min(max_ctx, mt)
        base["max_context"] = max_ctx

    base["confidence"] = _worst(weights_conf, kv_conf if kv_conf != "unknown" else "rough")
    return base
