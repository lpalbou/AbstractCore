"""The decision-model kinds AbstractCore runs, and the prompt/readout facts each needs.

A *kind* is a readout mechanism, not a model: every checkpoint of a kind shares
the prompt layout and how probabilities are read, and differs only in weights
and calibration. The facts below are the ones llama.cpp's conversion writes
into a decision GGUF (``<arch>.decision.type``, the ``systemone`` chat template,
the temperatures), kept here so transformers and mlx-lm render the same prompt
as llama.cpp does.

- ``kev``: pointer head on a Qwen3.5 base; five delimiter tokens; see ``encoding.py``.
- ``lev``: LoRA on Qwen3.5-4B (Apache-2.0); chat prompt with lettered options,
  answer read from next-token logits of the label tokens; a choice is read in
  two option orders and averaged; a noul is a 0..8 rating read at the first
  nine labels.
- ``openjev``: same mechanism with one letter per option (A..Z a..z) and
  fixed temperatures; the published OpenJev weights are CC-BY-NC-4.0.
- ``laya``: ModernBERT encoder + marker scorer (Apache-2.0); see ``engines/laya.py``.
- ``systemone``: a remote endpoint; see ``engines/remote.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

LEV_N_RATINGS = 9


def _str_or_json(name: str) -> str:
    return "{{ " + name + " if " + name + " is string else " + name + " | tojson }}"


# The exact templates llama.cpp's converter stores for these checkpoints (conversion/lev.py and
# conversion/qwen.py of ggml-org/llama.cpp), so every runtime renders one prompt.
_LEV_SYSTEM = (
    "You are a System One decision model. You read the Evidence and answer each "
    "Criterion by choosing exactly one of the listed options. You never explain. "
    "You answer with the single option label only."
)
_LEV_OPTIONS = (
    "{{ '# Options\\n' }}{% for o in options %}{{ o.label }}. "
    "{% if type == 'score' %}(level {{ o.key }} of {{ options | length - 1 }}) " + _str_or_json("o.description")
    + "{% else %}{{ o.key }}{% if o.description %}: " + _str_or_json("o.description") + "{% endif %}{% endif %}"
    "{{ '\\n' }}{% endfor %}"
    "{{ '\\nRespond with only the letter of ' }}"
    "{% if type == 'score' %}the level that best matches.{% else %}the best option.{% endif %}"
)
_LEV_SCALE = "{{ '# Scale\\n0 = certainly no ... 8 = certainly yes\\n' }}" + "".join(
    "{% for o in options %}{% if o.key == '" + key + "' and o.description %}" + name + ": " + _str_or_json("o.description") + "{{ '\\n' }}{% endif %}{% endfor %}"
    for key, name in (("true", "yes"), ("false", "no"))
) + "{{ '\\nRespond with only a digit from 0 to 8.' }}"
LEV_TEMPLATE = (
    "<|im_start|>system\n" + _LEV_SYSTEM + "<|im_end|>\n"
    "<|im_start|>user\n# Evidence\n" + _str_or_json("state") + "\n\n# Criterion\n"
    "{% if instructions %}" + _str_or_json("instructions") + "{% else %}{{ id }}{% endif %}"
    "{{ '\\n\\n' }}{% if type == 'noul' %}" + _LEV_SCALE + "{% else %}" + _LEV_OPTIONS + "{% endif %}"
    "{{ '\\n<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
)

_OPENJEV_OPTION = (
    "{% if type != 'noul' %}{{ o.key }}: {% if o.description %}" + _str_or_json("o.description") + "{% endif %}"
    "{% elif o.key == 'true' %}yes: {% if o.description %}" + _str_or_json("o.description") + "{% else %}The statement is true.{% endif %}"
    "{% else %}no: {% if o.description %}" + _str_or_json("o.description") + "{% else %}The statement is false.{% endif %}{% endif %}"
)
OPENJEV_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
OPENJEV_TEMPLATE = (
    "{% set letters = '" + OPENJEV_LETTERS + "' %}"
    "<|im_start|>user\nState:\n" + _str_or_json("state") + "\n\nQuestion: " + _str_or_json("instructions")
    + "{% if type == 'score' %} Rate along the ordered levels below (lowest first).{% endif %}"
    "{{ '\\nOptions:\\n' }}"
    "{% for o in options %}[{{ letters[loop.index0] }}] " + _OPENJEV_OPTION + "{{ '\\n' }}{% endfor %}"
    "{{ '\\nAnswer with the letter of the best option only.<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' }}"
)


@dataclass
class LabelKindSpec:
    """How a label-logit kind renders its prompt and reads its answer."""

    kind: str
    template: str
    label_codes: List[str]                  # candidate label texts, in order; only single-token ones are used
    temperatures: Dict[str, float] = field(default_factory=dict)   # "choice", "choice.small", "noul", "score", ...
    noul_mode: str = "false_true"           # "false_true" (options false,true), "true_first" (true,false), "rating" (0..8 read at the first labels)
    choice_variants: int = 1                # 2 = also the reversed option order, averaged
    sort_json_keys: bool = False            # render JSON states/instructions with sorted keys
    max_options: int = 255
    adapter_scale: float = 1.0              # LoRA strength to fold in (Lev ships 0.25)
    lora_base: Optional[str] = None

    def bucket(self, qtype: str, n_options: int) -> str:
        if self.kind == "lev":
            return "small" if n_options <= 8 else "mid" if n_options <= 26 else "large"
        return "2" if n_options <= 2 else "3_5" if n_options <= 5 else "6_10" if n_options <= 10 else "11"

    def temperature(self, qtype: str, n_options: int) -> float:
        for name in (f"{qtype}.{self.bucket(qtype, n_options)}", qtype):
            if name in self.temperatures:
                return float(self.temperatures[name])
        return 1.0


def lev_label_codes() -> List[str]:
    codes = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    codes += [a + b for a in codes[:26] for b in codes[:26]]
    return codes


def lev_spec(calibration: Optional[Dict[str, float]] = None, adapter_scale: float = 0.25, base: Optional[str] = None) -> LabelKindSpec:
    temperatures: Dict[str, float] = {}
    for name, value in (calibration or {}).items():
        parts = name.split(":")
        if len(parts) >= 2 and parts[1] == "A":
            temperatures[".".join([parts[0]] + parts[2:])] = float(value)
    return LabelKindSpec(kind="lev", template=LEV_TEMPLATE, label_codes=lev_label_codes(), temperatures=temperatures,
                         noul_mode="rating", choice_variants=2, sort_json_keys=True, adapter_scale=adapter_scale, lora_base=base)


def openjev_spec(temperature: float = 0.85, noul_factor: float = 1.829074) -> LabelKindSpec:
    return LabelKindSpec(kind="openjev", template=OPENJEV_TEMPLATE, label_codes=list(OPENJEV_LETTERS),
                         temperatures={"choice": temperature, "score": temperature, "noul": temperature * noul_factor},
                         noul_mode="true_first", choice_variants=1, max_options=len(OPENJEV_LETTERS))


KINDS = {
    "kev": "pointer head on a Qwen3.5 base (Kev family)",
    "lev": "label logits of a LoRA chat model (Lev)",
    "openjev": "label logits of a chat model (OpenJev family; weights CC-BY-NC-4.0)",
    "laya": "ModernBERT encoder with a marker scorer (Laya)",
    "systemone": "a hosted System One endpoint (TypeSafe Jev and compatible hosts)",
}

__all__ = ["LabelKindSpec", "LEV_TEMPLATE", "OPENJEV_TEMPLATE", "OPENJEV_LETTERS", "LEV_N_RATINGS", "lev_spec", "openjev_spec", "lev_label_codes", "KINDS"]
