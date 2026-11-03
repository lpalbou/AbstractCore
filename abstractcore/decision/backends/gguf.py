"""llama.cpp backend (GGUF) for decision models.

Needs a llama.cpp that knows decision models (ggml-org/llama.cpp PR #29818,
merged 2026-10-02): the converted GGUF carries the pointer head as one extra
``classifier.out_proj`` tensor and ``<arch>.decision.*`` metadata, and in
embeddings mode every token's output is its projected ``[q | k]`` vector
(``embedding_length_out`` floats). An older llama.cpp refuses the file at load
("wrong number of tensors"), which this backend reports with the version it
needs instead of letting the loader's message stand.

Rows: the state goes into sequence 0 once (chunked by ``n_batch``) and stays
there as the prefix. Each question is copied from that sequence
(``llama_memory_seq_cp``, which copies the attention KV and the DeltaNet
recurrent state alike), decoded on its own sequence, read at its ``</opt>`` and
``<decide>`` positions, and removed. Questions never see each other.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..encoding import EncodedQuestion
from ..head import PointerHead

MIN_LLAMA_CPP_NOTE = (
    "the installed llama-cpp-python bundles a llama.cpp without decision-model support "
    "(needs a build vendoring ggml-org/llama.cpp at or after 2026-10-02, PR #29818)"
)


@dataclass
class GGUFPrefix:
    state_ids: List[int]
    seq_id: int = 0


class GGUFDecisionBackend:
    name = "gguf"

    def __init__(
        self,
        model_path: str,
        *,
        n_ctx: int = 16_384,
        n_batch: int = 2_048,
        n_gpu_layers: int = -1,
        max_parallel_questions: int = 8,
        temperature: Optional[float] = None,
        verbose: bool = False,
    ):
        try:
            import llama_cpp  # type: ignore
            from llama_cpp import Llama  # type: ignore
        except ImportError as e:
            raise ImportError("GGUF decision models need llama-cpp-python") from e
        self.llama_cpp = llama_cpp
        self.model_path = model_path
        self.precision = _quant_from_filename(model_path)
        self.n_batch = int(n_batch)
        self.max_parallel = max(1, int(max_parallel_questions))
        # COMPAT (delete with abstractcore/decision/gguf_compat.py once abetlen/llama-cpp-python#2398
        # ships): a pre-#29818 llama.cpp refuses the official file, so run a derived backbone copy
        # and apply the pointer head in numpy from head.safetensors.
        self.compat_head: Optional[PointerHead] = None
        from ..gguf_compat import derive_backbone, gguf_has_pointer_head, llamacpp_has_decision_support

        if llamacpp_has_decision_support() is False and gguf_has_pointer_head(Path(model_path)):
            backbone, head_dir = derive_backbone(Path(model_path))
            self.compat_head = PointerHead.load(str(head_dir), temperature=temperature)
            self.source_path = model_path
            model_path = str(backbone)
        try:
            self.llm = Llama(
                model_path=model_path,
                embedding=True,
                pooling_type=llama_cpp.LLAMA_POOLING_TYPE_NONE,
                n_ctx=int(n_ctx),
                n_batch=int(n_batch),
                n_ubatch=int(n_batch),
                n_gpu_layers=int(n_gpu_layers),
                n_threads=None,
                kv_unified=True,
                n_seq_max=self.max_parallel + 1,
                verbose=verbose,
            )
        except Exception as e:  # llama.cpp reports "wrong number of tensors" on an old build
            message = str(e)
            if "number of tensors" in message or "cls" in message.lower():
                raise RuntimeError(f"cannot load {model_path}: {MIN_LLAMA_CPP_NOTE}: {message}") from e
            raise
        self.ctx = self.llm._ctx.ctx
        self.model = self.llm._model.model
        self.vocab = llama_cpp.llama_model_get_vocab(self.model)
        self.mem = llama_cpp.llama_get_memory(self.ctx)
        self.metadata = self._read_metadata()
        arch = self.metadata.get("general.architecture", "")
        decision_type = self.metadata.get(f"{arch}.decision.type")
        if self.compat_head is not None:
            # the derived backbone carries no decision metadata: the head and its temperature came from the file
            self.n_embd_out = int(llama_cpp.llama_model_n_embd(self.model))
            self.head_dim = self.compat_head.head_dim
            self.temperatures = {k: self.compat_head.temperature for k in ("choice", "noul", "score")}
        else:
            if decision_type != "kev":
                raise ValueError(
                    f"{model_path} is not a pointer-head decision GGUF ({arch}.decision.type={decision_type!r}); "
                    + (MIN_LLAMA_CPP_NOTE if decision_type is None else "only the 'kev' type is supported")
                )
            self.n_embd_out = int(llama_cpp.llama_model_n_embd_out(self.model))
            if self.n_embd_out % 2:
                raise ValueError(f"unexpected projected width {self.n_embd_out}: expected [q | k]")
            self.head_dim = self.n_embd_out // 2
            self.temperatures = {
                kind: float(self.metadata.get(f"{arch}.decision.temperature.{kind}", 1.0)) for kind in ("choice", "noul", "score")
            }
            if temperature is not None:
                self.temperatures = {k: float(temperature) for k in self.temperatures}
        self.n_ctx = int(llama_cpp.llama_n_ctx(self.ctx))
        self._resident: Optional[GGUFPrefix] = None
        self._embd_ptr_type = ctypes.POINTER(ctypes.c_float)

    # --- metadata -----------------------------------------------------------

    def _read_metadata(self) -> Dict[str, str]:
        lc = self.llama_cpp
        out: Dict[str, str] = {}
        count = lc.llama_model_meta_count(self.model)
        kbuf = ctypes.create_string_buffer(512)
        for i in range(count):
            lc.llama_model_meta_key_by_index(self.model, i, kbuf, 512)
            key = kbuf.value.decode("utf-8", "replace")
            if key.startswith("tokenizer.ggml.") and key not in ("tokenizer.ggml.model", "tokenizer.ggml.pre"):
                continue
            if "chat_template" in key:
                continue
            vbuf = ctypes.create_string_buffer(4096)
            lc.llama_model_meta_val_str_by_index(self.model, i, vbuf, 4096)
            out[key] = vbuf.value.decode("utf-8", "replace")
        return out

    # --- tokens -------------------------------------------------------------

    @property
    def tokenizer(self):
        from ..tokenizers import LlamaTokenizer

        if not hasattr(self, "_tokenizer"):
            self._tokenizer = LlamaTokenizer(self.llm)
        return self._tokenizer

    # --- decoding -----------------------------------------------------------

    def _decode_chunks(self, items: Sequence[Dict[str, Any]]) -> Dict[tuple, np.ndarray]:
        """Decode token runs on their sequences. ``items``: [{"seq": int, "tokens": [...],
        "pos0": int, "read": set(offsets)}]. Runs longer than n_batch are chunked in order.
        Returns {(seq, offset): projected vector} for every offset in ``read``."""
        lc = self.llama_cpp
        results: Dict[tuple, np.ndarray] = {}
        # Each item is decoded sequentially; several items may share one llama_decode
        # as long as the total stays within n_batch (whole runs only, so a run never
        # straddles two decodes on purpose: a long run is chunked by itself).
        pending: List[Dict[str, Any]] = []
        pending_tokens = 0

        def flush():
            nonlocal pending, pending_tokens
            if not pending:
                return
            batch = lc.llama_batch_init(pending_tokens, 0, 1)
            try:
                batch.n_tokens = 0
                index_of: Dict[tuple, int] = {}
                for item in pending:
                    for j, tok in enumerate(item["tokens"]):
                        i = batch.n_tokens
                        batch.token[i] = int(tok)
                        batch.pos[i] = int(item["pos0"] + j)
                        batch.n_seq_id[i] = 1
                        batch.seq_id[i][0] = int(item["seq"])
                        want = (item["offset0"] + j) in item["read"]
                        batch.logits[i] = 1 if want else 0
                        if want:
                            index_of[(item["seq"], item["offset0"] + j)] = i
                        batch.n_tokens += 1
                rc = lc.llama_decode(self.ctx, batch)
                if rc != 0:
                    raise RuntimeError(f"llama_decode failed with code {rc} (context {self.n_ctx} tokens; raise n_ctx?)")
                for key, i in index_of.items():
                    ptr = lc.llama_get_embeddings_ith(self.ctx, i)
                    if not ptr:
                        raise RuntimeError("llama.cpp returned no embedding for a requested token")
                    results[key] = np.ctypeslib.as_array(ptr, shape=(self.n_embd_out,)).astype(np.float32).copy()
            finally:
                lc.llama_batch_free(batch)
            pending, pending_tokens = [], 0

        for item in items:
            tokens = item["tokens"]
            if len(tokens) > self.n_batch:
                flush()
                for start in range(0, len(tokens), self.n_batch):
                    pending.append({**item, "tokens": tokens[start:start + self.n_batch],
                                    "pos0": item["pos0"] + start, "offset0": item.get("offset0", 0) + start})
                    pending_tokens = len(pending[-1]["tokens"])
                    flush()
                continue
            if pending_tokens + len(tokens) > self.n_batch:
                flush()
            pending.append({**item, "offset0": item.get("offset0", 0)})
            pending_tokens += len(tokens)
        flush()
        return results

    def prefix(self, state_ids: Sequence[int]) -> GGUFPrefix:
        state_ids = [int(t) for t in state_ids]
        if self._resident is not None and self._resident.state_ids == state_ids:
            return self._resident
        lc = self.llama_cpp
        lc.llama_memory_clear(self.mem, True)
        self._resident = None
        if len(state_ids) >= self.n_ctx:
            raise ValueError(f"state of {len(state_ids):,} tokens exceeds this context ({self.n_ctx:,} tokens)")
        self._decode_chunks([{"seq": 0, "tokens": state_ids, "pos0": 0, "read": set()}])
        self._resident = GGUFPrefix(state_ids=state_ids, seq_id=0)
        return self._resident

    def question_logits(self, prefix: GGUFPrefix, questions: Sequence[EncodedQuestion], kinds: Optional[Sequence[str]] = None) -> List[np.ndarray]:
        if self._resident is not prefix:
            self.prefix(prefix.state_ids)
        lc = self.llama_cpp
        n_state = len(prefix.state_ids)
        out: List[Optional[np.ndarray]] = [None] * len(questions)
        kinds = list(kinds) if kinds is not None else ["choice"] * len(questions)
        for start in range(0, len(questions), self.max_parallel):
            group = list(enumerate(questions))[start:start + self.max_parallel]
            budget = sum(len(q.ids) for _, q in group) + n_state
            if budget > self.n_ctx:
                raise ValueError(f"state plus questions need {budget:,} tokens, over this context ({self.n_ctx:,}); raise n_ctx")
            items = []
            for slot, (qi, q) in enumerate(group, start=1):
                lc.llama_memory_seq_cp(self.mem, 0, slot, -1, -1)
                items.append({"seq": slot, "tokens": q.ids, "pos0": n_state, "read": set(q.option_ends) | {q.decide}})
            vectors = self._decode_chunks(items)
            for slot, (qi, q) in enumerate(group, start=1):
                p_decide = vectors[(slot, q.decide)]
                p_opts = np.stack([vectors[(slot, e)] for e in q.option_ends])
                if self.compat_head is not None:
                    out[qi] = self.compat_head.logits(p_decide, p_opts)   # hidden states -> head in numpy (COMPAT)
                else:
                    temperature = self.temperatures.get(kinds[qi], 1.0)
                    head = _ProjectedHead(self.head_dim, temperature)
                    out[qi] = head.logits_from_projected(p_decide, p_opts)
                lc.llama_memory_seq_rm(self.mem, slot, -1, -1)
        return [z for z in out if z is not None]

    def release(self, prefix: Any) -> None:
        return None

    def unload(self) -> None:
        try:
            self.llm.close()
        except Exception:
            pass
        self.llm = None


class _ProjectedHead:
    """A pointer head whose projections already ran inside llama.cpp."""

    def __init__(self, head_dim: int, temperature: float):
        self.head_dim = head_dim
        self.temperature = float(temperature)
        self.scale = 1.0 / float(np.sqrt(head_dim))

    def logits_from_projected(self, p_decide: np.ndarray, p_options: np.ndarray) -> np.ndarray:
        q = p_decide[: self.head_dim]
        k = p_options[:, self.head_dim:]
        return (k @ q) * self.scale / self.temperature


def _quant_from_filename(path: str) -> str:
    base = os.path.basename(path).rsplit(".", 1)[0]
    for part in reversed(base.replace("-", "_").split("_")):
        if part.upper().startswith(("Q", "IQ", "F16", "BF16", "F32")) and any(ch.isdigit() for ch in part):
            upper = part.upper()
            # rebuild Q4_K_M-style names from the trailing tokens
            idx = base.upper().find(upper)
            return base[idx:] if idx >= 0 else upper
    return "gguf"


__all__ = ["GGUFDecisionBackend", "GGUFPrefix", "MIN_LLAMA_CPP_NOTE"]
