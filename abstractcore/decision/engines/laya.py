"""Laya (Convai Innovations, Apache-2.0): a bidirectional ModernBERT encoder with a
typed-decision head, run on transformers + torch.

One sequence per question: ``[CLS] <type> question: <instructions> [SEP] ([MASK] option)*
[SEP] <state> [SEP]`` (the head is capped at ``head_max_len`` tokens, each option at 48, the
whole sequence at ``max_len``, exactly as the checkpoint was trained). The encoder's hidden
states get a question-type embedding, pass through two post-norm transformer layers, and a
scorer reads one logit per option at its ``[MASK]`` marker. A temperature fitted per question
type and option count turns the logits into calibrated probabilities.

Multilingual and typed-decisions checkpoints of the same family load the same way (the
repository bundles them as subfolders; pass the subfolder as the model handle).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from ..render import answers_token_count, render as render_json, to_answers
from ..types import DecisionRequest, DecisionResponse, DecisionUsage

QTYPES = {"choice": 0, "score": 1, "noul": 2}
TEMP_MIN, TEMP_MAX = 0.5, 5.0   # a fitted temperature outside this band is not applied (sharpening is refused)


def _clamp_temperature(value: Any) -> float:
    try:
        t = float(value)
    except (TypeError, ValueError):
        return 1.0
    if t != t:
        return 1.0
    return min(TEMP_MAX, max(TEMP_MIN, t))


def temperature_bucket(qtype: str, n_options: int) -> str:
    size = "2" if n_options <= 2 else "3-5" if n_options <= 5 else "6-10" if n_options <= 10 else "11+"
    return f"{qtype}:{size}"


def render_criterion(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(qtype: str, criteria: Any) -> List[str]:
    """Option texts in label order; a noul question is always [false, true]."""
    if qtype == "choice":
        return [str(k) if v is None or v == "" else f"{k}: {render_criterion(v)}" for k, v in criteria.items()]
    if qtype == "score":
        return [f"level {i}: {render_criterion(c)}" for i, c in enumerate(criteria)]
    crit = criteria or {}
    false_crit, true_crit = crit.get("false"), crit.get("true")
    return [
        "false: " + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        "true: " + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def serialize_state(state: Any) -> str:
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def window_plan(n_state: int, room: int, window: Optional[int] = None, stride: Optional[int] = None) -> List[tuple]:
    """Overlapping (start, end) token windows over a state of ``n_state`` tokens, the window capped
    at ``room`` (what the questions leave inside the sequence) and the stride at half the window,
    as Laya's own long-state mode does. A state that fits is one window."""
    if room <= 0:
        raise ValueError("the questions leave no room for the state")
    w = min(int(window) if window else room, room)
    if n_state <= w:
        return [(0, n_state)]
    step = int(stride) if stride else max(1, w // 2)
    if step > w:
        raise ValueError("stride must not exceed the window: tokens between windows would be read by nothing")
    windows, start = [], 0
    while True:
        end = min(start + w, n_state)
        windows.append((start, end))
        if end >= n_state:
            break
        start += step
    return windows


def aggregate_windows(qtype: str, per_window: Sequence[np.ndarray]) -> np.ndarray:
    """Combine one question's per-window probabilities the way Laya's runtime does: a noul is the
    maximum P(true) over windows (the statement holds if any window supports it); a choice or score
    is the distribution of the single most confident window, so a localised signal is not outvoted
    by the neutral majority. The result is the deciding window's distribution, not a calibrated
    number for the whole document."""
    if len(per_window) == 1:
        return np.asarray(per_window[0], dtype=np.float32)
    if qtype == "noul":
        p_true = max(float(p[1]) for p in per_window)
        return np.asarray([1.0 - p_true, p_true], dtype=np.float32)
    best = max(per_window, key=lambda p: float(np.max(p)))
    return np.asarray(best, dtype=np.float32)


class LayaDecisionModel:
    kind = "laya"

    def __init__(self, model_dir: str, *, device: Optional[str] = None, dtype: str = "fp32", local_files_only: bool = True,
                 long_state: str = "windows", window: Optional[int] = None, stride: Optional[int] = None, max_windows_per_pass: int = 8):
        try:
            import torch  # type: ignore
            from torch import nn  # type: ignore
            from transformers import AutoConfig, AutoModel, AutoTokenizer  # type: ignore
        except ImportError as e:
            raise ImportError("Laya decision models need torch and transformers") from e
        from safetensors.torch import load_file

        self.torch = torch
        self.model_dir = model_dir
        self.model_id = model_dir
        self.provider = "huggingface"
        with open(os.path.join(model_dir, "rl_agent_config.json"), encoding="utf-8") as f:
            self.config = json.load(f)
        self.max_len = int(self.config.get("max_len", 512))
        self.head_max_len = int(self.config.get("head_max_len", 192))
        self.temperature = [_clamp_temperature(t) for t in self.config.get("temperature", [1.0, 1.0, 1.0])]
        self.temperature_by_options = {k: _clamp_temperature(v) for k, v in (self.config.get("temperature_by_options") or {}).items()}
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self._dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[dtype]

        tokenizer_dir = os.path.join(model_dir, "tokenizer") if os.path.isdir(os.path.join(model_dir, "tokenizer")) else model_dir
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=local_files_only)
        encoder_dir = os.path.join(model_dir, "encoder")
        encoder_config = AutoConfig.from_pretrained(encoder_dir, local_files_only=local_files_only)
        self.encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa" if self.device != "mps" else "eager")
        d = int(encoder_config.hidden_size)
        n_head_layers = int(self.config.get("head_layers", 2))
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, 0.0, batch_first=True, norm_first=True)
        self.head = nn.TransformerEncoder(layer, n_head_layers, enable_nested_tensor=False) if n_head_layers > 0 else None
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

        state = load_file(os.path.join(model_dir, "model.safetensors"))
        self._load_state(state)
        for module in (self.encoder, self.head, self.type_emb, self.scorer):
            if module is not None:
                module.to(self.device, dtype=self._dtype).eval()
        self.precision = dtype
        self.hidden_size = d
        if long_state not in ("windows", "truncate"):
            raise ValueError("long_state must be 'windows' (scan the whole state) or 'truncate' (read its start only)")
        self.long_state = long_state
        self.window = window
        self.stride = stride
        self.max_windows_per_pass = max(1, int(max_windows_per_pass))

    def _load_state(self, state: Dict[str, Any]) -> None:
        def sub(prefix: str) -> Dict[str, Any]:
            return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}

        missing, unexpected = self.encoder.load_state_dict(sub("encoder."), strict=False)
        unexpected = [k for k in unexpected if not k.startswith(("decoder", "head"))]
        missing = [k for k in missing if "rotary" not in k and "position" not in k]
        if missing or unexpected:
            raise ValueError(f"Laya encoder weights do not match the ModernBERT config (missing {missing[:3]}, unexpected {unexpected[:3]})")
        if self.head is not None:
            self.head.load_state_dict(sub("head."), strict=True)
        self.type_emb.load_state_dict(sub("type_emb."), strict=True)
        self.scorer.load_state_dict(sub("scorer."), strict=True)
        if "temperature" in state:
            file_temps = [float(x) for x in state["temperature"].tolist()]
            if len(file_temps) == 3 and not self.config.get("temperature"):
                self.temperature = [_clamp_temperature(t) for t in file_temps]

    # --- identity ------------------------------------------------------------------

    @property
    def backend_name(self) -> str:
        return "hf"

    # --- prompt ---------------------------------------------------------------------

    def _encode(self, text: str, **kwargs: Any) -> List[int]:
        return list(self.tokenizer(text, add_special_tokens=False, **kwargs)["input_ids"])

    def head_room(self, qtype: str, instructions: str, options: Sequence[str]) -> int:
        """State tokens one question leaves inside max_len."""
        ids, _, _ = self.build_sequence(qtype, instructions, options, [])
        return max(0, self.max_len - len(ids))

    def build_sequence(self, qtype: str, instructions: str, options: Sequence[str], state_ids: List[int]):
        tok = self.tokenizer
        mask = tok.mask_token
        head_ids = self._encode(f"{qtype} question: {str(instructions).replace(mask, ' ')}")
        opt_ids = [[tok.mask_token_id] + self._encode(" " + o.replace(mask, " "), truncation=True, max_length=48) for o in options]
        budget = self.head_max_len - sum(len(o) for o in opt_ids)
        if budget < 16:
            per = max(4, (self.head_max_len - 16) // max(1, len(opt_ids)))
            opt_ids = [o[:per] for o in opt_ids]
            budget = self.head_max_len - sum(len(o) for o in opt_ids)
        head_ids = head_ids[: max(8, budget)]
        ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
        markers = []
        for o in opt_ids:
            markers.append(len(ids))
            ids.extend(o)
        ids.append(tok.sep_token_id)
        room = max(0, self.max_len - len(ids) - 1)
        used = state_ids[:room]
        ids = ids + used + [tok.sep_token_id]
        ids, markers = ids[: self.max_len], [m for m in markers if m < self.max_len]
        return ids, markers, len(used)

    # --- scoring ---------------------------------------------------------------------

    def _temperature(self, qtype: str, n_options: int) -> float:
        return self.temperature_by_options.get(temperature_bucket(qtype, n_options), self.temperature[QTYPES[qtype]])

    def probabilities(self, request: DecisionRequest) -> List[np.ndarray]:
        """One probability vector per question (the same contract as the other engines)."""
        return self._probabilities_with_usage(request)[0]

    def _probabilities_with_usage(self, request: DecisionRequest):
        torch = self.torch
        state_text = serialize_state(request.state).replace(self.tokenizer.mask_token, " ")
        state_ids = self._encode(state_text)
        questions = [(q.type, render_json(q.instructions), render_options(q.type, q.criteria)) for q in request.questions.values()]
        if self.long_state == "windows" and questions:
            room = min(self.head_room(t, ins, opts) for t, ins, opts in questions)
            windows = window_plan(len(state_ids), room, self.window, self.stride)
        else:
            windows = [(0, len(state_ids))]
        self.last_windows = len(windows)
        # one sequence per (window, question); scored in passes of max_windows_per_pass windows
        per_question: List[List[np.ndarray]] = [[] for _ in questions]
        used_max = 0
        for w_start in range(0, len(windows), self.max_windows_per_pass):
            rows, meta = [], []
            for (a, b) in windows[w_start:w_start + self.max_windows_per_pass]:
                for qi, (t, ins, opts) in enumerate(questions):
                    ids, markers, used = self.build_sequence(t, ins, opts, state_ids[a:b])
                    if len(markers) != len(opts):
                        raise ValueError("a question does not fit the model's context: fewer options or shorter instructions")
                    rows.append(ids); meta.append((qi, markers, t)); used_max = max(used_max, a + used)
            for qi, p in zip([m[0] for m in meta], self._score_rows(rows, meta)):
                per_question[qi].append(p)
        probs = [aggregate_windows(t, pw) for (t, _, _), pw in zip(questions, per_question)]
        return probs, len(state_ids), min(used_max, len(state_ids))

    def _score_rows(self, rows: List[List[int]], meta: List[tuple]) -> List[np.ndarray]:
        torch = self.torch
        longest = max(len(r) for r in rows)
        pad_id = self.tokenizer.pad_token_id or 0
        input_ids = torch.full((len(rows), longest), pad_id, dtype=torch.long)
        attention = torch.zeros((len(rows), longest), dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r); attention[i, : len(r)] = 1
        input_ids, attention = input_ids.to(self.device), attention.to(self.device)
        qtype_idx = torch.tensor([QTYPES[m[2]] for m in meta], device=self.device)
        with torch.no_grad():
            h = self.encoder(input_ids=input_ids, attention_mask=attention).last_hidden_state
            h = h + self.type_emb(qtype_idx)[:, None, :]
            if self.head is not None:
                pad = ~attention.bool()
                for layer in self.head.layers:
                    h = layer(h, src_key_padding_mask=pad)
            out = []
            for i, (_, markers, kind) in enumerate(meta):
                picked = h[i, torch.tensor(markers, device=self.device)]
                logits = self.scorer(picked).squeeze(-1).float() / self._temperature(kind, len(markers))
                out.append(torch.softmax(logits, -1).cpu().numpy().astype(np.float32))
        return out

    def decide(self, state: Any = None, questions: Optional[Dict[str, Any]] = None, *, request: Optional[DecisionRequest] = None, truncate: Optional[bool] = None) -> DecisionResponse:
        if request is None:
            if questions is None:
                raise ValueError("decide() needs questions")
            request = DecisionRequest(state=state, questions=questions)
        from ..render import to_record

        _, meta = to_record(request)
        started = time.perf_counter()
        probs, state_tokens, state_used = self._probabilities_with_usage(request)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 1)
        answers = to_answers(probs, meta)
        return DecisionResponse(
            model=request.model or self.model_id, answers=answers,
            usage=DecisionUsage(input_tokens=state_tokens, output_tokens=answers_token_count(self._encode, answers),
                                state_tokens=state_tokens, state_tokens_used=state_used),
            latency_ms=latency_ms, truncated=state_used < state_tokens, provider=self.provider, backend="hf", precision=self.precision,
        )

    def unload(self) -> None:
        self.encoder = self.head = self.scorer = self.type_emb = None
        try:
            if self.device == "mps":
                self.torch.mps.empty_cache()
        except Exception:
            pass


__all__ = ["LayaDecisionModel", "render_options", "serialize_state", "temperature_bucket"]
