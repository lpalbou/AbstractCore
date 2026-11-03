"""Tokenizer adapters that satisfy ``encoding.DecisionTokenizer``.

Both tokenisers produce identical ids for the same text on the Qwen3.5
vocabulary; the GGUF path uses llama.cpp's own BPE so a GGUF-only install
never needs ``transformers``.
"""

from __future__ import annotations

from typing import Any, List, Optional

from .encoding import DELIMITERS


class HFTokenizer:
    """transformers tokenizer (loaded from a local snapshot directory or repo id)."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.pad_id = int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else 0
        ids = [tokenizer.convert_tokens_to_ids(t) for t in DELIMITERS]
        unknown = [t for t, i in zip(DELIMITERS, ids) if i is None or i == getattr(tokenizer, "unk_token_id", None)]
        self._delimiters = [] if unknown else [int(i) for i in ids]

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "HFTokenizer":
        from transformers import AutoTokenizer  # type: ignore

        return cls(AutoTokenizer.from_pretrained(path, **kwargs))

    def encode_text(self, text: str) -> List[int]:
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def encode_prompt(self, text: str) -> List[int]:
        """A rendered chat prompt: special tokens in the text are parsed."""
        return list(self.tokenizer(text, add_special_tokens=False).input_ids)

    def delimiter_ids(self) -> List[int]:
        if not self._delimiters:
            raise ValueError("tokenizer lacks the Kev delimiter tokens; this is not a Qwen-family vocabulary")
        return list(self._delimiters)


class LlamaTokenizer:
    """llama.cpp vocabulary of a loaded GGUF (``llama_cpp.Llama``)."""

    def __init__(self, llama: Any):
        self.llama = llama
        self.pad_id = 0
        self._delimiters = []
        for t in DELIMITERS:
            ids = llama.tokenize(t.encode("utf-8"), add_bos=False, special=True)
            if len(ids) != 1:
                self._delimiters = []
                break
            self._delimiters.append(int(ids[0]))

    def encode_text(self, text: str) -> List[int]:
        if not text:
            return []
        return [int(t) for t in self.llama.tokenize(text.encode("utf-8"), add_bos=False, special=False)]

    def encode_prompt(self, text: str) -> List[int]:
        if not text:
            return []
        return [int(t) for t in self.llama.tokenize(text.encode("utf-8"), add_bos=False, special=True)]

    def delimiter_ids(self) -> List[int]:
        if not self._delimiters:
            raise ValueError("GGUF vocabulary lacks the Kev delimiter tokens")
        return list(self._delimiters)


def load_hf_tokenizer(path: str, local_files_only: Optional[bool] = None) -> HFTokenizer:
    kwargs = {}
    if local_files_only is not None:
        kwargs["local_files_only"] = local_files_only
    return HFTokenizer.from_pretrained(path, **kwargs)


__all__ = ["HFTokenizer", "LlamaTokenizer", "load_hf_tokenizer"]
