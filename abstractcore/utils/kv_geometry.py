"""How much KV cache llama.cpp allocates for a GGUF at a given context.

ONE rule, used by the residency rows (`providers.hf_residency`), the context
sizing at load (`HuggingFaceProvider._load_gguf_model`) and the analytical
context-fit estimate (`utils.context_estimate`), so the three can never
disagree about the same model.

Why a layer walk and not `n_ctx * block_count * 2 * kv_heads * head_dim`: that
product counts every block as a full-attention layer. Measured on the Linux
test server (Quadro RTX 5000, 2026-10-29): `Qwen3.5-9B-Kimi-k3-Distilled`
(qwen35, 33 blocks, `full_attention_interval = 4`) at n_ctx 65,536 was shown
"+ 8.3 GiB cache" while only 8 blocks hold a KV cache (2.0 GiB); the GPU held
9.1 GiB in total for weights, KV and compute. The other 24 blocks are
Gated-DeltaNet layers with a fixed-size recurrent state.

The walk follows llama.cpp's own layer classification:

- `<arch>.full_attention_interval = k`: block `il` has a KV cache only when
  `(il + 1) % k == 0` (the others are recurrent: a fixed state per sequence).
- `<arch>.attention.sliding_window_pattern`: a per-layer bool array (True =
  sliding-window layer) or an int `n` (layer `il` slides when
  `il % n < n - 1`). A sliding layer uses `key_length_swa` / `value_length_swa`
  when present and holds `min(n_ctx, window * n_seq + n_ubatch)` cells, or
  `n_ctx` cells when llama.cpp keeps a full-size SWA cache (`swa_full`,
  llama-cpp-python's default: the log line "using full-size SWA cache").
- `<arch>.attention.head_count_kv` may be per layer.
- `<arch>.attention.shared_kv_layers = s`: the last `s` blocks reuse earlier
  caches; `<arch>.nextn_predict_layers = m`: the last `m` blocks are the
  multi-token-prediction head, not given a cache here.
- When flash attention is off and layers disagree on the V width, llama.cpp
  pads every V row to the widest ("padding V cache to ..."): `pad_v=True`.

K and V are f16 (llama.cpp's default cache type). The result is an ESTIMATE
from the file header, never a measurement, and callers label it so.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

KV_BYTES_PER_ELEMENT = 2  # f16 K and V (llama.cpp default cache type)
_RECURRENT_STATE_BYTES_PER_ELEMENT = 4  # recurrent conv/ssm states are f32


def _int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        out = int(str(value).strip())
    except Exception:
        return None
    return out


def _int_list(value: Any) -> Optional[List[int]]:
    if isinstance(value, (list, tuple)):
        out: List[int] = []
        for item in value:
            if isinstance(item, bool):
                out.append(int(item))
                continue
            v = _int(item)
            if v is None:
                return None
            out.append(v)
        return out or None
    return None


def geometry_from_metadata(metadata: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """The geometry dict (`read_gguf_geometry` keys) from llama-cpp-python's
    `Llama.metadata` (string values, `<arch>.`-prefixed keys). Per-layer arrays
    are not in that mapping; read the file header for them."""
    if not isinstance(metadata, Mapping):
        return None
    arch = str(metadata.get("general.architecture") or "").strip()
    if not arch:
        return None
    keys = {
        "block_count": "block_count",
        "head_count": "attention.head_count",
        "head_count_kv": "attention.head_count_kv",
        "key_length": "attention.key_length",
        "value_length": "attention.value_length",
        "embedding_length": "embedding_length",
        "context_length": "context_length",
        "full_attention_interval": "full_attention_interval",
        "sliding_window": "attention.sliding_window",
        "sliding_window_pattern": "attention.sliding_window_pattern",
        "key_length_swa": "attention.key_length_swa",
        "value_length_swa": "attention.value_length_swa",
        "shared_kv_layers": "attention.shared_kv_layers",
        "nextn_predict_layers": "nextn_predict_layers",
        "ssm_conv_kernel": "ssm.conv_kernel",
        "ssm_state_size": "ssm.state_size",
        "ssm_group_count": "ssm.group_count",
        "ssm_inner_size": "ssm.inner_size",
    }
    out: Dict[str, Any] = {"architecture": arch}
    for name, suffix in keys.items():
        v = _int(metadata.get(f"{arch}.{suffix}"))
        if v is not None:
            out[name] = v
    return out


def kv_cache_breakdown(
    geometry: Optional[Mapping[str, Any]],
    n_ctx: Optional[int],
    *,
    swa_full: bool = True,
    n_seq: int = 1,
    n_ubatch: int = 512,
    pad_v: bool = False,
    bytes_per_element: int = KV_BYTES_PER_ELEMENT,
) -> Optional[Dict[str, int]]:
    """`{"kv_bytes", "recurrent_bytes", "total_bytes", "kv_layers",
    "recurrent_layers", "n_ctx"}` for `n_ctx` tokens, or None when the
    geometry does not describe the layers. Never raises."""
    try:
        if not isinstance(geometry, Mapping) or not n_ctx or int(n_ctx) <= 0:
            return None
        n_ctx = int(n_ctx)
        n_layer = _int(geometry.get("block_count"))
        n_head = _int(geometry.get("head_count"))
        kv_layers_arr = _int_list(geometry.get("head_count_kv_layers"))
        n_head_kv = _int(geometry.get("head_count_kv")) or n_head
        k_len = _int(geometry.get("key_length"))
        if k_len is None:
            n_embd = _int(geometry.get("embedding_length"))
            k_len = (n_embd // n_head) if (n_embd and n_head) else None
        v_len = _int(geometry.get("value_length")) or k_len
        if not (n_layer and (n_head_kv or kv_layers_arr) and k_len and v_len):
            return None
        k_swa = _int(geometry.get("key_length_swa")) or k_len
        v_swa = _int(geometry.get("value_length_swa")) or v_len
        interval = _int(geometry.get("full_attention_interval"))
        window = _int(geometry.get("sliding_window"))
        pattern_arr = _int_list(geometry.get("sliding_window_pattern_layers"))
        pattern_n = _int(geometry.get("sliding_window_pattern"))
        shared = max(0, _int(geometry.get("shared_kv_layers")) or 0)
        nextn = max(0, _int(geometry.get("nextn_predict_layers")) or 0)
        last_cached = max(0, n_layer - max(shared, nextn))

        swa_cells = n_ctx
        if window and not swa_full:
            padded = window * max(1, int(n_seq)) + max(0, int(n_ubatch))
            padded = ((padded + 255) // 256) * 256
            swa_cells = min(n_ctx, padded)

        rows: List[tuple] = []  # (cells, k_width, v_width)
        recurrent_layers = 0
        for il in range(last_cached):
            if interval and interval > 0 and (il + 1) % interval != 0:
                recurrent_layers += 1
                continue
            heads = kv_layers_arr[il] if kv_layers_arr and il < len(kv_layers_arr) else n_head_kv
            if not heads:
                continue  # a layer with no KV heads holds no cache
            if pattern_arr is not None:
                is_swa = bool(pattern_arr[il]) if il < len(pattern_arr) else False
            elif pattern_n and pattern_n > 1:
                is_swa = il % pattern_n < pattern_n - 1
            else:
                is_swa = False
            if is_swa and window:
                rows.append((swa_cells, heads * k_swa, heads * v_swa))
            else:
                rows.append((n_ctx, heads * k_len, heads * v_len))
        if not rows and not recurrent_layers:
            return None
        v_max = max((r[2] for r in rows), default=0)
        kv = 0
        for cells, k_w, v_w in rows:
            kv += cells * (k_w + (v_max if pad_v else v_w)) * int(bytes_per_element)

        recurrent = 0
        conv = _int(geometry.get("ssm_conv_kernel"))
        state = _int(geometry.get("ssm_state_size"))
        inner = _int(geometry.get("ssm_inner_size"))
        groups = _int(geometry.get("ssm_group_count")) or 1
        if recurrent_layers and conv and state and inner:
            per_layer = (max(0, conv - 1) * (inner + 2 * groups * state) + state * inner)
            recurrent = recurrent_layers * per_layer * _RECURRENT_STATE_BYTES_PER_ELEMENT * max(1, int(n_seq))
        return {
            "kv_bytes": int(kv),
            "recurrent_bytes": int(recurrent),
            "total_bytes": int(kv + recurrent),
            "kv_layers": len(rows),
            "recurrent_layers": int(recurrent_layers),
            "n_ctx": n_ctx,
        }
    except Exception:
        return None


def kv_cache_bytes(geometry: Optional[Mapping[str, Any]], n_ctx: Optional[int], **kw: Any) -> Optional[int]:
    """`kv_cache_breakdown(...)["total_bytes"]` (KV + recurrent state), or None."""
    out = kv_cache_breakdown(geometry, n_ctx, **kw)
    return int(out["total_bytes"]) if out else None


def kv_bytes_per_token(geometry: Optional[Mapping[str, Any]]) -> Optional[int]:
    """KV bytes one more token of context costs: every cached layer at full
    size (a full-size SWA cache grows with the context too). None when unknown."""
    one = kv_cache_breakdown(geometry, 1, swa_full=True)
    two = kv_cache_breakdown(geometry, 2, swa_full=True)
    if not one or not two:
        return None
    return int(two["kv_bytes"] - one["kv_bytes"]) or None
