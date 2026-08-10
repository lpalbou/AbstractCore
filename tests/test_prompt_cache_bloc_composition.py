"""Prompt-cache BLOC COMPOSABILITY: proof that N blocs compose token-for-token.

A bloc is an independently-keyed slice of ONE rendered conversation, not a
conversation of its own. The contract these tests pin down (see the long note
above `BaseProvider.prompt_cache_render_bloc_text` in
`abstractcore/providers/base.py`):

    for every k:  concat(fragment_1 .. fragment_k)
                  == tokenize(generate_prompt(union of blocs 1..k))[: boundary_k]

i.e. every bloc boundary is a TRUE TOKEN PREFIX of the prompt the provider's own
`generate()` would build for the same logical content. Anything less and the KV
state past the first boundary is unreachable — which is exactly the defect these
tests were written for: two blocs used to render two consecutive
`<|im_start|>system` blocks, bytes `generate()` never produces (measured 618 of
2148 prefix tokens reachable in the live agent, 375 of 668 on the fixture here).

Tokenizer + chat template only. NO model weights are loaded anywhere in this
file.
"""

from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, Iterator, List, Optional

import pytest

from abstractcore.core.types import GenerateResponse
from abstractcore.providers.base import BaseProvider, PromptCacheModule
from abstractcore.providers.mlx_provider import MLXProvider
from abstractcore.tools import UniversalToolHandler


# --------------------------------------------------------------------------
# Fixture content: one system bloc, one tools bloc, one history bloc.
# --------------------------------------------------------------------------

SYSTEM = (
    "You are a careful coding agent.\nAlways cite file paths.\n"
    + ("Filler line so the system bloc is not trivially short. " * 40)
)
TOOLS: List[Dict[str, Any]] = [
    {
        "name": "read_file",
        "description": "Read a file from disk",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write a file to disk",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "text": {"type": "string"}},
            "required": ["path", "text"],
        },
    },
]
HISTORY: List[Dict[str, Any]] = [
    {"role": "user", "content": "Find the bug in cache.py"},
    {"role": "assistant", "content": "Looking now."},
]

BLOC_SYSTEM = PromptCacheModule(module_id="system", system_prompt=SYSTEM).normalized()
BLOC_TOOLS = PromptCacheModule(module_id="tools", tools=TOOLS).normalized()
BLOC_HISTORY = PromptCacheModule(module_id="history", messages=HISTORY).normalized()

CHAINS = {
    1: [BLOC_SYSTEM],
    2: [BLOC_SYSTEM, BLOC_TOOLS],
    3: [BLOC_SYSTEM, BLOC_TOOLS, BLOC_HISTORY],
}


# --------------------------------------------------------------------------
# Tokenizers. Real local ones when present; a deterministic toy otherwise, so
# the invariant is always exercised even on a machine with no models.
# --------------------------------------------------------------------------


def _local_model_dir(repo: str) -> Optional[str]:
    """Directory of an ALREADY-INSTALLED local model. Never downloads."""
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    hits = sorted(
        glob.glob(os.path.join(home, "hub", f"models--{repo.replace('/', '--')}", "snapshots", "*", "tokenizer.json"))
    )
    if hits:
        return os.path.dirname(hits[0])
    lms = os.path.expanduser(f"~/.lmstudio/models/{repo}")
    if os.path.isfile(os.path.join(lms, "tokenizer.json")):
        return lms
    return None


def _load_runtime_tokenizer(repo: str):
    """Load a tokenizer that produces the SAME token stream the runtime does.

    `AutoTokenizer.from_pretrained` first — that is literally what the provider
    calls, so the fixture cannot drift from production. When it needs
    sentencepiece to convert a slow tokenizer, fall back to the serialized fast
    tokenizer PLUS the special tokens from `tokenizer_config.json`.

    Those special tokens are not cosmetic. `_encode_prompt_token_ids` decides
    `add_special_tokens` by testing whether the text starts with the BOS
    LITERAL; with `bos_token=None` it cannot, so the post-processor's BOS is
    added on top of the one the template already rendered. Measured on
    gemma-4-31b: a bare `PreTrainedTokenizerFast(tokenizer_file=...)` yields
    frag0 head `[2, 2, 105, 9731]` (BOS twice) where the runtime yields
    `[2, 105, 9731, 107]`. The invariant still holds internally — both sides use
    the same encoder — so a test built on the bare loader passes while
    validating a token stream that does not exist. Do not reintroduce it.
    """
    path = _local_model_dir(repo)
    if not path:
        return None
    try:
        from transformers import AutoTokenizer, PreTrainedTokenizerFast
    except Exception:
        return None
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception:
        pass
    try:
        cfg_path = os.path.join(path, "tokenizer_config.json")
        cfg = {}
        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
        specials = {
            k: cfg[k]
            for k in ("bos_token", "eos_token", "pad_token", "unk_token")
            if isinstance(cfg.get(k), str) and cfg.get(k)
        }
        return PreTrainedTokenizerFast(tokenizer_file=os.path.join(path, "tokenizer.json"), **specials)
    except Exception:
        return None


class ToyBPETokenizer:
    """Deterministic byte-ish tokenizer with SEAM-STRADDLING merges.

    Its whole reason to exist is the hard half of the invariant: character
    agreement is not token agreement. `"e" + "\\n\\n"` merges into one token
    here, so a naive cut that only compares strings produces a chain whose
    tokens do NOT concatenate to the single-shot tokenization. The planner has
    to back the boundary off a token for this tokenizer, and the tests below
    prove it does.
    """

    _MERGES = ("<|im_start|>", "<|im_end|>", "<|turn>", "<turn|>", "<bos>", "e\n\n", "\n\n", ". ")
    bos_token = None
    # Set on the subclass below: mimics a tokenizer POST-PROCESSOR that injects
    # a sequence-start token on every `encode(add_special_tokens=True)` — the
    # Llama-3 shape, where a per-bloc encode plants `<|begin_of_text|>` in the
    # middle of the sequence.
    auto_bos_id: Optional[int] = None

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {}

    def _id(self, piece: str) -> int:
        if piece not in self._vocab:
            self._vocab[piece] = len(self._vocab) + 1000
        return self._vocab[piece]

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        out: List[int] = []
        if add_special_tokens and self.auto_bos_id is not None:
            out.append(int(self.auto_bos_id))
        i = 0
        while i < len(text):
            for merge in self._MERGES:
                if text.startswith(merge, i):
                    out.append(self._id(merge))
                    i += len(merge)
                    break
            else:
                out.append(self._id(text[i]))
                i += 1
        return out

    def decode(self, ids: List[int]) -> str:
        rev = {v: k for k, v in self._vocab.items()}
        return "".join(rev.get(int(i), "?") for i in ids)


class ToyAutoBosTokenizer(ToyBPETokenizer):
    """Toy tokenizer that injects a BOS on every encode (the Llama-3 hazard)."""

    auto_bos_id = 999
    bos_token = None


# Real tokenizers. WHITELIST-BOUND (see the operator's model whitelist): only
# 4B+ models already installed locally, and NEVER a substitution across family
# or size — a 135M stand-in labelled "llama3" tells you nothing about the
# behaviour it claims to cover and its result is inadmissible. Each entry is a
# distinct real tokenizer, exercising a distinct hazard.
_REAL_TOKENIZER_CASES = [
    # ChatML + a folding template; no BOS at all.
    ("chatml/qwen3-4b", "im_start_end", "mlx-community/Qwen3-4B-Instruct-2507-4bit"),
    # gemma-turn: the template RENDERS a <bos> literal into the text, and the
    # tokenizer post-processor injects one too. 31B, whitelisted.
    ("gemma-turn/gemma4-31b", "gemma_turn", "mlx-community/gemma-4-31b-mxfp4"),
    # A third, independent real tokenizer on the ChatML branch (arch
    # qwen3_5_agentic): different vocab and merges over the same template, which
    # is where a seam merge would show up differently.
    ("chatml/ornith-9b", "im_start_end", "mlx-community/Ornith-1.0-9B-4bit"),
    # Hybrid 4B, different tokenizer revision again.
    ("chatml/qwen35-4b", "im_start_end", "mlx-community/Qwen3.5-4B-MLX-4bit"),
]

# NOT COVERED BY A REAL TOKENIZER: the `llama3_header` family. No Llama 4B+
# model is installed locally and the whitelist forbids downloading one or
# substituting a smaller sibling, so that cell is UNTESTED against a real
# tokenizer. The hazard it carries — a post-processor that injects a
# sequence-start token on EVERY encode — is covered deterministically by
# `ToyAutoBosTokenizer` below, which is a reproduction of the mechanism, not a
# claim about Llama-3's behaviour.


def _template_cases():
    """(label, message_format, tokenizer). Toy cases always; real cases from disk."""
    cases = [
        # Always present: no filesystem or network dependency, and each toy
        # tokenizer reproduces one seam hazard exactly.
        ("chatml/toy", "im_start_end", ToyBPETokenizer()),
        ("gemma-turn/toy", "gemma_turn", ToyBPETokenizer()),
        ("plain-fallback/toy", "llama3_header", ToyBPETokenizer()),
        ("autobos/toy", "llama3_header", ToyAutoBosTokenizer()),
    ]
    for label, message_format, repo in _REAL_TOKENIZER_CASES:
        tok = _load_runtime_tokenizer(repo)
        if tok is not None:
            cases.append((label, message_format, tok))
    return cases


TEMPLATE_CASES = _template_cases()
REAL_CASES = [c for c in TEMPLATE_CASES if "/toy" not in c[0]]

# HARD GUARD, not a soft preference. Without it the real cases are appended
# only `if tok is not None`, so on a box with no local models TEMPLATE_CASES
# silently shrinks to the four toy tokenizers and the suite still reports green
# — three real template families satisfied by environment coincidence rather
# than by the test. Fail loudly instead.
assert len(REAL_CASES) >= 3, (
    "bloc composition tests require at least 3 REAL local tokenizers; found "
    f"{[c[0] for c in REAL_CASES]}. Install the whitelisted models or fix "
    "_REAL_TOKENIZER_CASES — do NOT relax this guard."
)


# --------------------------------------------------------------------------
# Providers under test.
# --------------------------------------------------------------------------


def make_renderer(model: str, message_format: str, tokenizer) -> MLXProvider:
    """An MLXProvider wired for RENDERING ONLY.

    `__init__` is skipped deliberately: it resolves and loads a model, and this
    file must never touch weights. Everything the render/plan path reads is set
    explicitly below — if that path ever grows a new dependency this construction
    fails loudly rather than silently testing something else.
    """
    p = object.__new__(MLXProvider)
    p.model = model
    p.provider = "mlx"
    p.architecture_config = {"message_format": message_format}
    p.tool_handler = UniversalToolHandler(model)
    p.tokenizer = tokenizer
    return p


class FakeTokenCacheProvider(BaseProvider):
    """BaseProvider whose "KV cache" is the literal list of tokens fed into it.

    That makes `prompt_cache_prepare_modules` end-to-end checkable without a
    model: whatever the chain actually prefills is exactly what the assertions
    read back.
    """

    def __init__(self, renderer: MLXProvider, model: str = "fake", **kwargs):
        super().__init__(model, **kwargs)
        self._renderer = renderer
        self.tokenizer = renderer.tokenizer
        self.append_calls: List[Dict[str, Any]] = []

    # --- render/tokenize hooks: delegate to the real renderer ---
    def prompt_cache_render_bloc_text(self, **kwargs) -> Optional[str]:
        return self._renderer.prompt_cache_render_bloc_text(**kwargs)

    def prompt_cache_encode_bloc_text(self, text: str) -> Optional[List[int]]:
        return self._renderer.prompt_cache_encode_bloc_text(text)

    # --- backend hooks: token list as the cache value ---
    def supports_prompt_cache(self) -> bool:
        return True

    def _prompt_cache_backend_create(self) -> Any:
        return []

    def _prompt_cache_backend_clone(self, cache_value: Any) -> Any:
        return list(cache_value) if cache_value is not None else None

    def _prompt_cache_backend_token_count(self, cache_value: Any) -> Optional[int]:
        return len(cache_value) if isinstance(cache_value, list) else None

    def _prompt_cache_backend_append(self, cache_value: Any, **kwargs) -> bool:
        self.append_calls.append(dict(kwargs))
        planned = kwargs.get("bloc_token_ids")
        if isinstance(planned, (list, tuple)):
            cache_value.extend(int(t) for t in planned)
            return True
        # Legacy path: render THIS module standalone, exactly as the providers
        # did before bloc planning existed.
        text = self._renderer.prompt_cache_render_bloc_text(
            prompt=str(kwargs.get("prompt") or ""),
            messages=kwargs.get("messages"),
            system_prompt=kwargs.get("system_prompt"),
            tools=kwargs.get("tools"),
            add_generation_prompt=bool(kwargs.get("add_generation_prompt")),
        )
        cache_value.extend(self._renderer.prompt_cache_encode_bloc_text(text or "") or [])
        return True

    # --- BaseProvider abstract surface ---
    def _generate_internal(self, prompt: str, **kwargs) -> GenerateResponse:
        _ = (prompt, kwargs)
        return GenerateResponse(content="ok", model=self.model, finish_reason="stop")

    def get_capabilities(self) -> List[str]:
        return ["chat"]

    def unload_model(self, model_name: str) -> None:
        _ = model_name

    def list_available_models(self, **kwargs) -> List[str]:
        _ = kwargs
        return [self.model]

    def validate_config(self) -> bool:
        return True


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _union_generate_tokens(renderer: MLXProvider, blocs: List[PromptCacheModule]) -> List[int]:
    """Tokens of the prompt `generate()` builds for the union of `blocs`.

    Deliberately goes through `_build_prompt` (the real generate-side entry
    point) rather than the planner's own renderer, so the test compares the
    plan against the LIVE prompt path, not against itself.
    """
    union = BaseProvider._bloc_union(blocs)
    text = renderer._build_prompt(
        str(union.get("prompt") or ""),
        union.get("messages"),
        union.get("system_prompt"),
        union.get("tools"),
    )
    return renderer.prompt_cache_encode_bloc_text(text) or []


def _lcp(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# --------------------------------------------------------------------------
# The proofs
# --------------------------------------------------------------------------


@pytest.mark.parametrize("label,message_format,tokenizer", TEMPLATE_CASES, ids=[c[0] for c in TEMPLATE_CASES])
@pytest.mark.parametrize("n_blocs", [1, 2, 3])
def test_bloc_chain_is_a_token_prefix_of_generate(label, message_format, tokenizer, n_blocs):
    """render(blocs) == generate_prompt(same logical content), token for token."""
    renderer = make_renderer("test/model", message_format, tokenizer)
    blocs = CHAINS[n_blocs]

    plan = renderer.prompt_cache_plan_bloc_chain(blocs)
    assert plan is not None, f"{label}: no plan produced"
    assert plan.unsound_at is None, f"{label}: {plan.reason}"
    assert len(plan.fragments) == n_blocs

    chain_tokens: List[int] = []
    for k in range(n_blocs):
        chain_tokens.extend(plan.fragments[k])
        # EVERY boundary — not just the last — must be a token prefix of what
        # generate() builds for the union of blocs 0..k. That is what makes a
        # bloc individually reusable.
        gen_tokens = _union_generate_tokens(renderer, blocs[: k + 1])
        assert chain_tokens == gen_tokens[: len(chain_tokens)], (
            f"{label} n={n_blocs}: bloc {k} boundary is not a token prefix of generate(); "
            f"shared {_lcp(chain_tokens, gen_tokens)} of {len(chain_tokens)}"
        )
        assert plan.boundaries[k] == len(chain_tokens)

    # Every bloc must actually carry something, or the chain is a fiction.
    assert all(len(f) > 0 for f in plan.fragments), f"{label}: empty bloc fragment"


@pytest.mark.parametrize("label,message_format,tokenizer", TEMPLATE_CASES, ids=[c[0] for c in TEMPLATE_CASES])
def test_bloc_fragments_are_successor_independent(label, message_format, tokenizer):
    """Bloc k's tokens do not change when bloc k+1 changes or disappears.

    This is the property that makes a bloc's derived cache key an honest name
    for its bytes, and the only reason recombining blocs across sessions can
    ever be sound.
    """
    renderer = make_renderer("test/model", message_format, tokenizer)
    other_tools = PromptCacheModule(
        module_id="tools",
        tools=[{"name": "grep", "description": "search", "parameters": {"type": "object", "properties": {}}}],
    ).normalized()

    p1 = renderer.prompt_cache_plan_bloc_chain([BLOC_SYSTEM])
    p2 = renderer.prompt_cache_plan_bloc_chain([BLOC_SYSTEM, BLOC_TOOLS])
    p2b = renderer.prompt_cache_plan_bloc_chain([BLOC_SYSTEM, other_tools])
    p3 = renderer.prompt_cache_plan_bloc_chain([BLOC_SYSTEM, BLOC_TOOLS, BLOC_HISTORY])
    for p in (p1, p2, p2b, p3):
        assert p is not None and p.unsound_at is None

    assert p1.fragments[0] == p2.fragments[0] == p2b.fragments[0] == p3.fragments[0], (
        f"{label}: the `system` bloc changed when its successor changed"
    )
    assert p2.fragments[1] == p3.fragments[1], f"{label}: the `tools` bloc changed when a history bloc followed"


def _standalone_chain_tokens(renderer: MLXProvider, blocs: List[PromptCacheModule]) -> List[int]:
    """The OLD shape: one `_build_prompt_fragment` call per bloc, concatenated."""
    out: List[int] = []
    for mod in blocs:
        text = renderer.prompt_cache_render_bloc_text(
            prompt=str(mod.prompt or ""),
            messages=mod.messages,
            system_prompt=mod.system_prompt,
            tools=mod.tools,
        )
        out.extend(renderer.prompt_cache_encode_bloc_text(text or "") or [])
    return out


@pytest.mark.parametrize("label,message_format,tokenizer", TEMPLATE_CASES, ids=[c[0] for c in TEMPLATE_CASES])
def test_planned_cut_is_never_worse_than_standalone_rendering(label, message_format, tokenizer):
    """No template may regress: the planned chain is always fully reachable.

    The plain `role: content` fallback (where every non-ChatML, non-gemma
    architecture currently lands) happens to compose on its own — its bloc
    separator and its fold separator are both `\\n\\n`. That is luck, not
    design, and it is why the guarantee is asserted for every format rather
    than only for the ones that were visibly broken.
    """
    renderer = make_renderer("test/model", message_format, tokenizer)
    blocs = CHAINS[2]
    gen_tokens = _union_generate_tokens(renderer, blocs)

    standalone = _standalone_chain_tokens(renderer, blocs)
    plan = renderer.prompt_cache_plan_bloc_chain(blocs)
    planned = [t for frag in plan.fragments for t in frag]

    assert _lcp(planned, gen_tokens) == len(planned), f"{label}: planned chain is not a generate() prefix"
    assert _lcp(planned, gen_tokens) >= _lcp(standalone, gen_tokens), f"{label}: planned cut lost ground"


@pytest.mark.parametrize(
    "label,message_format,tokenizer",
    [c for c in TEMPLATE_CASES if c[1] in ("im_start_end", "gemma_turn")],
    ids=[c[0] for c in TEMPLATE_CASES if c[1] in ("im_start_end", "gemma_turn")],
)
def test_folding_templates_strand_a_standalone_bloc_chain(label, message_format, tokenizer):
    """The defect this design exists to fix, pinned as a regression test.

    ChatML and gemma-turn FOLD the tool instructions into the single system
    turn. Rendering each bloc as its own standalone conversation therefore
    opens a second system block — bytes `generate()` never produces — and
    everything past the first boundary is unreachable KV. The planned cut
    reaches the end.
    """
    renderer = make_renderer("test/model", message_format, tokenizer)
    blocs = CHAINS[2]
    gen_tokens = _union_generate_tokens(renderer, blocs)

    standalone = _standalone_chain_tokens(renderer, blocs)
    plan = renderer.prompt_cache_plan_bloc_chain(blocs)
    planned = [t for frag in plan.fragments for t in frag]

    assert _lcp(standalone, gen_tokens) < len(standalone), (
        f"{label}: expected standalone per-bloc rendering to diverge from generate()"
    )
    assert _lcp(planned, gen_tokens) == len(planned)
    assert _lcp(planned, gen_tokens) > _lcp(standalone, gen_tokens)
    # And the composability tax is small: the planned chain gives up only the
    # trailing tokens a successor could still rewrite (the system turn's
    # closing tag), never a whole bloc.
    merged_text = renderer.prompt_cache_render_bloc_text(system_prompt=SYSTEM, tools=TOOLS)
    merged = renderer.prompt_cache_encode_bloc_text(merged_text) or []
    assert len(merged) - len(planned) <= 4, f"{label}: composability tax {len(merged) - len(planned)} tokens"


@pytest.mark.parametrize("label,message_format,tokenizer", TEMPLATE_CASES, ids=[c[0] for c in TEMPLATE_CASES])
@pytest.mark.parametrize("n_blocs", [1, 2, 3])
def test_prepare_modules_feeds_exactly_the_planned_tokens(label, message_format, tokenizer, n_blocs):
    """End-to-end through `prompt_cache_prepare_modules`, still without weights.

    Also covers the warm path: preparing the same chain twice must not re-feed
    anything.
    """
    renderer = make_renderer("test/model", message_format, tokenizer)
    provider = FakeTokenCacheProvider(renderer)
    blocs = CHAINS[n_blocs]

    out = provider.prompt_cache_prepare_modules(namespace="test", modules=list(blocs))
    assert out["supported"] is True
    assert out["bloc_plan"]["composable"] is True
    assert len(out["modules"]) == n_blocs

    gen_tokens = _union_generate_tokens(renderer, blocs)
    for idx, entry in enumerate(out["modules"]):
        cached = provider._prompt_cache_store.get(entry["cache_key"])
        assert isinstance(cached, list) and cached, f"{label}: module {idx} cached nothing"
        expected = _union_generate_tokens(renderer, blocs[: idx + 1])
        assert cached == expected[: len(cached)], f"{label}: module {idx} cache is not a generate() prefix"
        meta = provider.prompt_cache_key_meta(entry["cache_key"])
        assert meta.get("bloc_boundary_tokens") == len(cached)

    final = provider._prompt_cache_store.get(out["final_cache_key"])
    assert final == gen_tokens[: len(final)]
    # Distinct keys per bloc: the separation is the deliverable, not an accident.
    assert len({e["cache_key"] for e in out["modules"]}) == n_blocs

    # Warm re-prepare: nothing is appended a second time.
    before = len(provider.append_calls)
    provider.prompt_cache_prepare_modules(namespace="test", modules=list(blocs))
    assert len(provider.append_calls) == before
    assert provider._prompt_cache_store.get(out["final_cache_key"]) == final


@pytest.mark.parametrize("label,message_format,tokenizer", TEMPLATE_CASES, ids=[c[0] for c in TEMPLATE_CASES])
def test_shared_system_bloc_is_reused_across_tool_sets(label, message_format, tokenizer):
    """Composability, concretely: two agents with different tools share the
    `system` bloc's cache key AND its bytes."""
    renderer = make_renderer("test/model", message_format, tokenizer)
    provider = FakeTokenCacheProvider(renderer)
    tools_b = PromptCacheModule(
        module_id="tools",
        tools=[{"name": "grep", "description": "search", "parameters": {"type": "object", "properties": {}}}],
    ).normalized()

    a = provider.prompt_cache_prepare_modules(namespace="test", modules=[BLOC_SYSTEM, BLOC_TOOLS])
    calls_after_a = len(provider.append_calls)
    b = provider.prompt_cache_prepare_modules(namespace="test", modules=[BLOC_SYSTEM, tools_b])

    # Guard against a vacuous pass: on a COLLAPSED chain neither key exists, and
    # an equality assertion over two absent keys proves nothing.
    assert a["bloc_plan"]["composable"] is True and b["bloc_plan"]["composable"] is True
    assert len(a["modules"]) == 2 and len(b["modules"]) == 2
    assert a["modules"][0]["cache_key"] == b["modules"][0]["cache_key"]
    assert a["modules"][1]["cache_key"] != b["modules"][1]["cache_key"]
    for entry in list(a["modules"]) + list(b["modules"]):
        assert provider._prompt_cache_store.get(entry["cache_key"]) is not None, (
            f"{label}: reported cache_key {entry['cache_key']} does not exist"
        )
    # The second chain rebuilds ONLY its tools bloc — the system prefill is reused.
    assert len(provider.append_calls) - calls_after_a == 1
    assert provider.append_calls[-1].get("tools")


@pytest.mark.parametrize("label,message_format,tokenizer", TEMPLATE_CASES, ids=[c[0] for c in TEMPLATE_CASES])
@pytest.mark.parametrize("padded", ["  SYS\n\n", "\nSYS  ", "SYS"])
def test_whitespace_padded_system_prompt_still_composes(label, message_format, tokenizer, padded):
    """`normalized()` strips; the renderer must strip identically.

    `PromptCacheModule.normalized()` strips `system_prompt` before it is
    fingerprinted AND before it is rendered into the bloc chain, while
    `generate()` receives the caller's raw string. The ChatML branch used to
    render raw — so a system prompt with any surrounding whitespace diverged at
    the FIRST token of the system text and the whole prefix cache was dead.
    Second independent cause of full-prefix loss; caught 2026-08-03.
    """
    renderer = make_renderer("test/model", message_format, tokenizer)
    bloc = PromptCacheModule(module_id="system", system_prompt=padded).normalized()
    plan = renderer.prompt_cache_plan_bloc_chain([bloc, BLOC_TOOLS])
    assert plan is not None and plan.unsound_at is None

    # generate() sees the RAW, unnormalized string — that is the whole point.
    raw_text = renderer._build_prompt("", None, padded, TOOLS)
    raw_tokens = renderer.prompt_cache_encode_bloc_text(raw_text) or []
    chain = [t for frag in plan.fragments for t in frag]
    assert chain == raw_tokens[: len(chain)], (
        f"{label} padded={padded!r}: bloc chain diverges from the raw generate() prompt; "
        f"shared {_lcp(chain, raw_tokens)} of {len(chain)}"
    )


def _sequence_start_ids(renderer: MLXProvider, message_format: str) -> List[int]:
    """Token ids that may legally appear ONLY at absolute position 0.

    Two independent sources: a tokenizer post-processor that injects one on
    every `encode` (Llama-3's `<|begin_of_text|>`), and a template that RENDERS
    a BOS literal into the text (gemma-turn). ChatML has neither — its
    `<|im_start|>` legitimately recurs at every turn — so this returns [] there
    and the test below is a no-op for it.
    """
    tok = renderer.tokenizer
    out: List[int] = []
    try:
        with_specials = tok.encode("x", add_special_tokens=True)
        without = tok.encode("x", add_special_tokens=False)
        if len(with_specials) > len(without) and with_specials[0] != without[0]:
            out.append(int(with_specials[0]))
    except Exception:
        pass
    if message_format == "gemma_turn":
        bos = str(getattr(tok, "bos_token", "") or "<bos>")
        ids = renderer.prompt_cache_encode_bloc_text(bos) or []
        if len(ids) == 1:
            out.append(int(ids[0]))
    return out


@pytest.mark.parametrize("label,message_format,tokenizer", TEMPLATE_CASES, ids=[c[0] for c in TEMPLATE_CASES])
def test_sequence_start_token_appears_exactly_once_in_the_whole_chain(label, message_format, tokenizer):
    """A BOS may appear at absolute position 0 and nowhere else — including in
    bloc 0.

    The whole-chain count is the assertion that matters. An earlier version only
    inspected fragments 1..n, which cannot see a DOUBLED BOS at the head: a
    template that renders a `<bos>` literal plus a tokenizer post-processor that
    injects one produces `[2, 2, ...]` at position 0, a token stream the runtime
    never emits, while every later fragment stays clean and the test passes.
    (Measured on gemma-4-31b with a special-token-less fixture loader,
    2026-08-03; the loader is fixed and this assertion is the guard.)
    """
    renderer = make_renderer("test/model", message_format, tokenizer)
    start_ids = _sequence_start_ids(renderer, message_format)
    if not start_ids:
        pytest.skip(f"{label}: template/tokenizer has no sequence-start token")

    plan = renderer.prompt_cache_plan_bloc_chain(CHAINS[3])
    assert plan is not None and plan.unsound_at is None
    chain = [t for frag in plan.fragments for t in frag]

    for sid in start_ids:
        assert chain.count(sid) == 1, (
            f"{label}: sequence-start token {sid} appears {chain.count(sid)}x in the chain "
            f"(expected exactly 1); head={chain[:6]}"
        )
        assert chain[0] == sid, f"{label}: sequence-start token {sid} is not at position 0"
    for idx, frag in enumerate(plan.fragments[1:], start=1):
        for sid in start_ids:
            assert sid not in frag, f"{label}: bloc {idx} re-injects sequence-start token {sid}"

    # The chain must also match what generate() actually sends, head included.
    gen_tokens = _union_generate_tokens(renderer, CHAINS[3])
    assert chain == gen_tokens[: len(chain)]

    # And the old shape genuinely did re-inject it — otherwise this test is
    # guarding nothing.
    standalone = _standalone_chain_tokens(renderer, CHAINS[3])
    assert any(standalone.count(sid) > 1 for sid in start_ids), (
        f"{label}: expected standalone per-bloc encoding to duplicate a sequence-start token"
    )


class HostileTokenizer(ToyBPETokenizer):
    """Tokenizer whose merges make NO cut inside the system turn token-safe.

    Every `<|im_start|>system\\n...` prefix is swallowed into a single token
    together with whatever follows it, so the backoff can never find an agreed
    boundary between the `system` and `tools` blocs. Exercises the plan-time
    seam check's `#FALLBACK` collapse — the path that must never silently
    produce a cache that claims more than it holds.
    """

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        _ = add_special_tokens
        # One token for everything up to and including the first close tag, so
        # any mid-turn cut lands inside an atomic token.
        for close in ("<|im_end|>", "<turn|>"):
            idx = text.find(close)
            if idx >= 0:
                head = text[: idx + len(close)]
                return [self._id(head)] + super().encode(text[idx + len(close) :])
        return [self._id(text)] if text else []


def test_seam_check_collapses_when_no_cut_is_token_safe():
    """(C3) asserted, not assumed: an unsafe seam degrades loudly and correctly."""
    renderer = make_renderer("test/model", "im_start_end", HostileTokenizer())
    plan = renderer.prompt_cache_plan_bloc_chain(CHAINS[2])

    assert plan is not None
    assert plan.collapsed is True, "expected the seam check to fail on this tokenizer"
    assert plan.reason
    assert len(plan.fragments) == 1

    # Collapsed or not, the cache content must still be a true token prefix of
    # generate()'s prompt — degradation may cost reuse, never correctness.
    gen_tokens = _union_generate_tokens(renderer, CHAINS[2])
    assert plan.fragments[0] == gen_tokens[: len(plan.fragments[0])]


def test_collapsed_chain_writes_only_the_final_key_and_stays_warm():
    renderer = make_renderer("test/model", "im_start_end", HostileTokenizer())
    provider = FakeTokenCacheProvider(renderer)

    out = provider.prompt_cache_prepare_modules(namespace="test", modules=list(CHAINS[2]))
    assert out["bloc_plan"]["collapsed"] is True
    assert out["bloc_plan"]["composable"] is False

    final = provider._prompt_cache_store.get(out["final_cache_key"])
    assert final, "collapsed chain cached nothing"
    gen_tokens = _union_generate_tokens(renderer, CHAINS[2])
    assert final == gen_tokens[: len(final)]
    # The intermediate key is deliberately NOT written: it would name a
    # boundary the tokenizer refused to give us. And the result must not ADVERTISE
    # it either — a caller forking from a reported key must never fork from
    # nothing.
    assert len(out["modules"]) == 1
    for entry in out["modules"]:
        assert provider._prompt_cache_store.get(entry["cache_key"]) is not None

    # ...and the next call must still find it warm, or a collapse would turn
    # into a full re-prefill on every single call.
    before = len(provider.append_calls)
    again = provider.prompt_cache_prepare_modules(namespace="test", modules=list(CHAINS[2]))
    assert len(provider.append_calls) == before
    assert again["final_cache_key"] == out["final_cache_key"]


def test_out_of_slot_order_chain_is_refused():
    """(C1) A bloc that renders into an already-cached region is not composable."""
    renderer = make_renderer("test/model", "im_start_end", ToyBPETokenizer())
    plan = renderer.prompt_cache_plan_bloc_chain([BLOC_HISTORY, BLOC_TOOLS])
    assert plan is None


def test_planning_is_unavailable_without_a_renderer():
    """A provider with no exact renderer gets None, not a guess."""

    class NoRenderer(FakeTokenCacheProvider):
        def prompt_cache_render_bloc_text(self, **kwargs):
            return None

    provider = NoRenderer(make_renderer("test/model", "im_start_end", ToyBPETokenizer()))
    assert provider.prompt_cache_plan_bloc_chain([BLOC_SYSTEM, BLOC_TOOLS]) is None
    # ...and prepare_modules still works, on the legacy per-module path.
    out = provider.prompt_cache_prepare_modules(namespace="test", modules=[BLOC_SYSTEM, BLOC_TOOLS])
    assert out["supported"] is True
    assert "bloc_plan" not in out


# --------------------------------------------------------------------------
# Composition position: token equality is necessary, NOT sufficient.
#
# Cached K tensors are rotary-position-encoded at compile time and nothing in
# this package re-applies them, so a bloc's KV is valid at exactly one absolute
# offset behind exactly one ordered prefix. These tests pin that the constraint
# is representable and refused, rather than silently violated.
# --------------------------------------------------------------------------


def _manifest(**over):
    from abstractcore.core.bloc_kv import BlocKVArtifactManifest

    base = dict(
        version=1,
        provider="mlx",
        model="m",
        model_resolved_id="/m",
        cache_backend="mlx",
        artifact_format="abstractcore-mlx-prompt-cache/v1",
        bloc_sha256="a" * 64,
        bloc_id=1,
        content_sha256="b" * 64,
        path_in_prompt="x.txt",
        recipe_id="attached_file_box",
        recipe_version=1,
        rendered_recipe_sha256="c" * 64,
        renderer_version=1,
        serializer_version="v1",
        artifact_filename="a.safetensors",
        artifact_sha256="d" * 64,
        quantization="fp",
        created_at="2026-08-03T00:00:00+00:00",
        token_count=100,
    )
    base.update(over)
    return BlocKVArtifactManifest(**base)


def test_composition_verdict_matches_same_offset_and_prefix():
    from abstractcore.core.bloc_kv import bloc_kv_composition_verdict

    m = _manifest(start_offset=512, prefix_chain=["sys-hash"])
    assert bloc_kv_composition_verdict(m, at_offset=512, prefix_chain=["sys-hash"])[0] == "match"


def test_composition_verdict_refuses_a_shifted_offset():
    from abstractcore.core.bloc_kv import bloc_kv_composition_verdict

    m = _manifest(start_offset=512, prefix_chain=["sys-hash"])
    verdict, detail = bloc_kv_composition_verdict(m, at_offset=0, prefix_chain=["sys-hash"])
    assert verdict == "mismatch"
    assert "offset" in detail


def test_composition_verdict_refuses_a_different_prefix_chain():
    from abstractcore.core.bloc_kv import bloc_kv_composition_verdict

    m = _manifest(start_offset=512, prefix_chain=["sys-hash"])
    assert bloc_kv_composition_verdict(m, at_offset=512, prefix_chain=["other"])[0] == "mismatch"
    # Reordering is a different prefix, not the same set.
    m2 = _manifest(start_offset=512, prefix_chain=["a", "b"])
    assert bloc_kv_composition_verdict(m2, at_offset=512, prefix_chain=["b", "a"])[0] == "mismatch"


def test_composition_verdict_abstains_for_pre_axis_artifacts():
    from abstractcore.core.bloc_kv import bloc_kv_composition_verdict

    verdict, detail = bloc_kv_composition_verdict(_manifest(), at_offset=0, prefix_chain=[])
    assert verdict == "abstain"
    assert "start_offset" in detail


def test_composition_fields_survive_a_manifest_roundtrip():
    from abstractcore.core.bloc_kv import BlocKVArtifactManifest

    m = _manifest(start_offset=7, prefix_chain=["p1", "p2"])
    back = BlocKVArtifactManifest.from_dict(m.to_dict())
    assert back.start_offset == 7
    assert back.prefix_chain == ["p1", "p2"]
    # The new axis must NOT change binding_id, or every pre-axis manifest on
    # disk would be rejected (same rule as the other reuse axes).
    assert back.binding_id == BlocKVArtifactManifest.from_dict(_manifest().to_dict()).binding_id


def test_toy_tokenizer_actually_straddles_the_seam():
    """Guard on the guard: if the toy tokenizer stops merging across the cut,
    `test_bloc_chain_is_a_token_prefix_of_generate` silently stops testing the
    token-alignment half of the invariant."""
    tok = ToyBPETokenizer()
    assert tok.encode("e\n\n") != tok.encode("e") + tok.encode("\n\n")


# --------------------------------------------------------------------------
# A tools bloc that holds no tools.
#
# Everything above is about token SEAMS. None of it can see the failure found
# on 2026-08-07: a tool set whose descriptions exceeded `ToolDefinition`'s
# 200-char cap was silently converted to zero definitions, so the renderers
# emitted no tool text and the planner returned a perfectly sound-looking chain
# whose "tools" bloc held 3 tokens of closing scaffolding and no tools. Every
# seam check passed, `collapsed` was False, `unsound_at` was None, and the
# model never saw a single tool. These two tests are the content half.
# --------------------------------------------------------------------------

# FIXTURE UPDATED 2026-08-07 (TOOL-desc). This list used to hold tools with
# >200-char descriptions, because that was then the easiest way to make the
# converter yield nothing. It no longer is: an over-long description on an
# EXTERNAL tool (a dict is external by construction) is now adapted to the cap
# and the tool renders normally, so the old fixture stopped producing the
# condition this test exists to catch. The assertions below are unchanged — only
# the way the empty bloc is manufactured. Malformed dicts still convert to
# nothing, which is the remaining real path to a tool-less tools bloc.
_UNRENDERABLE_TOOLS = [
    # Named (so `_bloc_tool_names` knows the bloc CLAIMS to carry it, which is
    # what makes the omission detectable) but carrying neither a `description`
    # nor an OpenAI-style `function` entry — so the converter cannot build a
    # ToolDefinition and the renderer emits nothing for it.
    {"name": f"tool_{i}", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}
    for i in range(3)
]


@pytest.mark.parametrize("label,message_format,tokenizer", REAL_CASES, ids=[c[0] for c in REAL_CASES])
def test_tools_bloc_that_renders_no_tools_is_reported(label, message_format, tokenizer):
    """The plan may be sound and still describe a prompt with no tools in it."""
    import warnings as _warnings

    renderer = make_renderer("mlx-community/Qwen3-4B-Instruct-2507-4bit", message_format, tokenizer)
    dropped = PromptCacheModule(module_id="tools", tools=_UNRENDERABLE_TOOLS).normalized()

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        plan = renderer.prompt_cache_plan_bloc_chain([BLOC_SYSTEM, dropped])

    assert plan is not None
    messages = [str(w.message) for w in caught if issubclass(w.category, RuntimeWarning)]
    assert any("do NOT appear in the rendered prompt" in m for m in messages), (
        f"{label}: a tools bloc whose tools never reached the render must be reported; got {messages[:2]}"
    )
    # And the bloc really is empty of tools — this is the condition, not a proxy.
    full = renderer.prompt_cache_render_bloc_text(system_prompt=SYSTEM, tools=_UNRENDERABLE_TOOLS) or ""
    assert "tool_0" not in full


@pytest.mark.parametrize("label,message_format,tokenizer", REAL_CASES, ids=[c[0] for c in REAL_CASES])
def test_a_real_tools_bloc_is_silent_and_carries_the_tools(label, message_format, tokenizer):
    """The healthy case must not warn, and the tools bloc must hold the tools."""
    import warnings as _warnings

    renderer = make_renderer("mlx-community/Qwen3-4B-Instruct-2507-4bit", message_format, tokenizer)

    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        plan = renderer.prompt_cache_plan_bloc_chain([BLOC_SYSTEM, BLOC_TOOLS])

    assert plan is not None and plan.unsound_at is None and not plan.collapsed
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)], f"{label}: unexpected warning"
    # The tools bloc is a real bloc, not template scaffolding: it must hold
    # substantially more than the closing tag the system bloc held back.
    assert len(plan.fragments[1]) > 20, (
        f"{label}: tools bloc holds only {len(plan.fragments[1])} tokens — tools did not reach the render"
    )
    system_only = renderer.prompt_cache_render_bloc_text(system_prompt=SYSTEM) or ""
    with_tools = renderer.prompt_cache_render_bloc_text(system_prompt=SYSTEM, tools=TOOLS) or ""
    assert len(with_tools) > len(system_only)
    assert "read_file" in with_tools and "read_file" not in system_only
