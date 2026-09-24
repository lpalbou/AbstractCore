"""Shared native VLM weights and lifecycle; no provider/request state lives here."""
from __future__ import annotations

import threading
import weakref
from pathlib import Path

_SESSIONS = weakref.WeakValueDictionary()
_LOAD_LOCK = threading.RLock()


def release_native_owner(session, owner_id):
    """Retire a collected owner without waiting on an in-flight GPU operation."""
    with session.config_lock:
        if session.runtime is not None and owner_id:
            # An explicit unload uses strict release and may reject busy work.
            # A weakref callback runs once: it must instead arrange eventual
            # retirement on the worker, never lose the lease after owner_busy.
            # The worker owns APC/cache closing, including flush failures.
            session.runtime.retire(owner_id, keepalive=session)
            return
        if not list(session.holders):
            if session.apc is not None:
                session.apc.close()
            session.runtime = None
            session.cache_store = None
            session.apc = None
            session.execution_config = None



# Measured on 2026-09-22 (see `NativeSession.prompt_cache`): how the snapshot
# budget is SPENT — on one prompt's near-identical intermediates, or on the last
# N distinct prompts of a conversation.
#
# Larger than any context window this provider serves, on purpose: mlx-vlm emits
# one intermediate checkpoint per `interval` tokens below the prompt's end, and
# `APCCoordinator.checkpoint_lengths` caps that count at `checkpoint_entries` —
# the SAME number that caps the whole snapshot store. An interval inside the
# prompt therefore makes a single prompt fill the store (mission A3; see the
# docstring below). An interval beyond it yields exactly one snapshot per call.
CHECKPOINT_INTERVAL_TOKENS = 1_048_576
# Lineage depth: how many distinct prompts of a session stay restorable. A tool
# loop of K LLM calls consumes K of these, so the conversation's own anchor (the
# first call of a run) survives a loop of up to ENTRIES-1 calls.
CHECKPOINT_ENTRIES = 8
# Where the one per-call snapshot lands: this many tokens BEFORE the prompt's
# end (mlx-vlm's `exact_cache_guard_tokens`, default 1). The next call's prompt
# re-renders the PREVIOUS last message differently whenever a payload repair
# merges a new user message into it (alternation-strict templates forbid
# user,user) — a parse-retry nudge or drained operator guidance after a loop
# tail or a tool-result carrier. The divergence then sits ~5 tokens before the
# old prompt's end (`<|im_end|>\n<|im_start|>assistant\n`), past a guard of 1,
# and the whole iteration re-prefilled (mission A3: 7,814 tokens on the 4B
# replay, turn 2 iteration 5). Eight tokens of margin absorb that plus a BPE
# re-merge of the content's last token, for ~7 extra fed tokens per call.
CHECKPOINT_GUARD_TOKENS = 8


class NativeSession:
    def __init__(self):
        self.model = None
        self.processor = None
        self.drafter = None
        self.draft_kind = None
        self.drafter_path = None
        self.drafter_weights_fingerprint = ""
        self.lock = threading.Lock()
        self.config_lock = threading.RLock()
        self.holders = weakref.WeakSet()
        self.execution_config = None
        self.runtime = None
        self.cache_store = None
        self.apc = None

    def prompt_cache(self, *, memory_max_gb=None):
        """The session's mlx-vlm prefix cache, built once.

        `memory_max_gb` is forwarded ONLY when the operator set it
        (`mlx_cache_memory_max_gb`). Absent, mlx-vlm sizes the budget from the
        machine: `min(8 GiB, recommended_working_set / 10)`. A flat 0.5 GiB
        override used to live here; mlx-vlm refuses to retain any single
        snapshot larger than the budget (`store_exact_cache`), and a hybrid
        model's snapshot grows ~27-50 KB per prompt token, so past ~10-19k
        tokens every store was silently skipped and every turn re-prefilled
        the whole conversation. The reserve is left to mlx-vlm (machine-relative).

        Checkpoint placement is SIZED HERE, by measurement (2026-09-22, 4B
        pair, 8 conversational turns, untracked/p3/replay_4b.py). Hybrid
        models can only restore an EXACT stored prefix, and mlx-vlm stores
        `entries - 1` intermediate checkpoints spaced `interval` tokens below
        the prompt's end plus the end itself (`APCCoordinator.checkpoint_lengths`).
        The framework's chat turns carry per-turn bytes INSIDE the last user
        message (the runtime grounding envelope and the react loop tail), so
        the end-of-prompt checkpoint never matches the next turn and reuse can
        only land on an intermediate one. Upstream's 2048 x 2 restored 4096 of
        a 4.9k-token prompt every turn (483-835 tokens re-prefilled and
        growing); 256 x 2 (the previous override) went COLD on 3 of 7 turns
        because its single intermediate often fell inside the changed tail;
        256 x 4 never went cold and re-prefilled 12-246 tokens beyond the new
        turn, at 4 resident snapshots (~37-45 KB per prompt token each, i.e.
        ~4.5 GB at a 25k-token conversation on a 27B-class hybrid, under the
        default 8 GiB budget). With a byte-stable last user turn the default
        2 x 2048 already restores the whole prefix (1 token short); this
        setting bounds the damage until that is fixed upstream of the provider.
        `APC_CHECKPOINT_INTERVAL_TOKENS` / `APC_CHECKPOINT_ENTRIES` still win
        when the operator sets them.

        RE-MEASURED 2026-09-22, after byte-stability landed for the react lane
        (abstractruntime `turn_grounding` stamps the grounding envelope into the
        DURABLE user turn; abstractagent stops merging loop chrome into it;
        `untracked/missionA/replay_runtime.py`, same 4B pair, 8 turns, driven
        through the REAL runtime stack instead of a synthetic wrapper). With a
        byte-stable last user turn the restore lands on the EXACT end-of-prompt
        snapshot of the previous turn, so the INTERMEDIATE checkpoints stop
        being load-bearing:

          chat, 8 turns    256 x 4  -> fed 105-121/turn, 874 MB resident
                           2048 x 2 -> fed 105-128/turn, 441 MB resident
          tool loop        identical fed at identical iteration indices
                           (4761 / 115 / 169), 818 MB vs 415 MB resident

        On that lane 256 x 4 now buys nothing and costs 2x the RAM.

        RE-SIZED 2026-09-22 (mission A3) — 1_048_576 x 8. The 256 x 4 shape was
        not merely wasteful: it made CONVERSATION reuse structurally impossible
        after any tool-using turn. mlx-vlm reads ONE number for two different
        jobs. `APCCoordinator.checkpoint_lengths` (apc_coordinator.py:207) caps
        the intermediates a SINGLE prompt emits at
        `budget = manager._exact_cache_max`, and `APCManager.store_exact_cache`
        (apc.py:3569) caps the WHOLE snapshot store at that same
        `_exact_cache_max`, evicting LRU and tenant-blind. So a single prompt
        always emits exactly as many snapshots as the store can hold, and the
        store can never contain more than the most recent prompt. Model-free
        proof, 4B pair, `checkpoint_lengths` called directly:

          interval=256   entries=4  prompt=28533 -> 4 stores, capacity 4
                                                   [27904, 28160, 28416, 28532]
          interval=2^20  entries=8  prompt=28533 -> 1 store,  capacity 8  [28532]

        Consequence, measured on the operator's live stack (runs 930d405a ->
        081d8daa -> 520d0e69, 2026-09-22 10:47-10:53) and reproduced on the 4B
        (`untracked/missionA3/before-tools.log`): after a tool loop, the four
        resident snapshots all sit inside the loop's discarded branch (28k-token
        prompts, 256 tokens apart), and the next turn's prompt — which still
        extends the run's FIRST prompt exactly — matches none of them. Every
        turn after a tool-using turn paid a full prefill: 51 s on the operator's
        model, 4752 restorable tokens thrown away on the 4B.

        The fix decouples the two jobs with the two knobs that exist. An interval
        past any context window makes `checkpoint_lengths` return `[final]`
        alone, so each call stores ONE snapshot — its end of prompt, the only one
        a byte-stable lane can ever use (measured above). `entries` then means
        what its name says: how many distinct prompts stay restorable. Eight
        covers a tool loop of seven LLM calls plus the conversation anchor.

        RAM: the budget is unchanged and still enforced by mlx-vlm
        (`min(8 GiB, working_set/10)`, overridable via `mlx_cache_memory_max_gb`);
        `entries` decides how it is SPENT, not how much is spent.
        `_make_room` (apc.py:3275) evicts snapshots LRU-first when the budget is
        exceeded, so the cost is bounded by the budget in both shapes. Measured
        on the 4B pair, ~35-44 KB per prompt token per snapshot: 256 x 4 held
        4 x 28.5k tokens = 3.91 GB at the peak of the tool-loop replay; 2^20 x 8
        holds eight END-of-prompt snapshots of the last eight calls: measured
        peak 4.90 GB on the same 2-turn x 4-iteration replay (28-30k-token
        prompts), 1.76 GB on the 8-turn chat, `memory_skips` 0 and `rejects` 0
        throughout (`untracked/missionA3/final-*.log`). Above the budget mlx-vlm
        evicts the OLDEST snapshot rather than skipping the store.

        Tradeoff, stated: intermediate checkpoints are gone, so a lane whose last
        user turn still changes between calls no longer lands on a 256-token grid
        — it goes cold instead of partial. React, CodeAct and MemAct all carry the
        byte-stable invariant as of mission A3, which leaves composed entity
        visits, visual `llm_call` nodes and direct AbstractCore callers. For them
        the previous shape restored 12-246 tokens beyond the new turn at 2x the
        RAM and destroyed cross-turn reuse for everyone else; that is a bad trade
        once the main lanes are stable. `APC_CHECKPOINT_INTERVAL_TOKENS` /
        `APC_CHECKPOINT_ENTRIES` still win when the operator sets them, so
        restoring 256 x 4 per process is one env var.

        The block pool stays at 256 x 256: checkpoint-mode (hybrid) models never
        touch it, and for dense models a 256-token block keeps the resident
        tensor count 16x below the 16-token default for the same 64k-token pool,
        well under Apple Metal's per-process resource ceiling that mlx-vlm
        guards with `APC_MAX_POOL_TENSORS`.
        """
        if self.apc is None:
            import os
            from mlx_vlm.apc import APCManager
            overrides = {}
            if memory_max_gb is not None:
                overrides["memory_max_gb"] = float(memory_max_gb)
            for key, env, measured in (
                ("checkpoint_interval_tokens", "APC_CHECKPOINT_INTERVAL_TOKENS", CHECKPOINT_INTERVAL_TOKENS),
                ("checkpoint_entries", "APC_CHECKPOINT_ENTRIES", CHECKPOINT_ENTRIES),
                ("checkpoint_guard_tokens", "APC_CHECKPOINT_GUARD_TOKENS", CHECKPOINT_GUARD_TOKENS),
            ):
                if env not in os.environ:
                    overrides[key] = measured
            self.apc = APCManager(num_blocks=256, block_size=256, overrides=overrides)
        return self.apc


def mtp_weight_keys(path) -> list:
    """The `mtp.` tensor names a local MLX checkpoint carries (empty: none).

    A cheap, LOCAL read -- the safetensors index's `weight_map`, or, for a
    single-file checkpoint, each safetensors file's JSON header (8-byte length
    + header; no tensor is read). Never the model name, never
    `mtp_num_hidden_layers` (present in configs whose weights hold no head).

    Why it matters: mlx-lm <= 0.31.3 `qwen3_5.Model.sanitize` reads ANY `mtp.`
    tensor as "raw Hugging Face checkpoint" and adds +1.0 to every RMSNorm
    weight, so an MTP-PRESERVING checkpoint that is already MLX-converted gets
    its norms shifted twice and generates garbage (fixed upstream on mlx-lm
    main by ml-explore/mlx-lm#1623, not released as of 0.31.3). mlx-vlm strips
    `mtp.` BEFORE that decision and loads the same files correctly. The MLX
    provider therefore never hands such a checkpoint to mlx-lm.

    The substring test (`"mtp." in key`) is mlx-lm's own, so this answers
    exactly the question mlx-lm's sanitize asks. An unreadable INDEX raises:
    a sharded checkpoint whose weights cannot be classified must not silently
    take the lane that corrupts it.
    """
    import json as _json
    import struct

    root = Path(path)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        document = _json.loads(index.read_text(encoding="utf-8"))
        weight_map = document.get("weight_map") if isinstance(document, dict) else None
        if not isinstance(weight_map, dict):
            raise ValueError(f"Invalid safetensors index {index}: weight_map must be an object")
        return sorted(k for k in weight_map if "mtp." in str(k))
    keys = []
    for shard in sorted(root.glob("*.safetensors")):
        # A file that is not valid safetensors cannot be loaded by EITHER
        # loader, so it cannot be mis-routed; skipping it lets the loader
        # report its own, precise error instead of this probe guessing one.
        try:
            with open(shard, "rb") as fh:
                raw = fh.read(8)
                if len(raw) != 8:
                    continue
                (size,) = struct.unpack("<Q", raw)
                if size <= 0 or size > 100 * 1024 * 1024:
                    continue
                header = _json.loads(fh.read(size).decode("utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            continue
        if isinstance(header, dict):
            keys.extend(k for k in header if k != "__metadata__" and "mtp." in k)
    return sorted(keys)


def resolve_native_drafter_path(path_or_repo: str) -> str:
    """Resolve aliases before sharing weights or deriving persistent identity.

    Cache first, never the network for a cached head. An explicitly named head
    that is NOT cached (mission U, 2026-09-24):
      * offline_first on (the default): refused with ModelNotFoundError naming
        the head and the download command -- loading never downloads, and a
        drafter is a model like any other;
      * offline_first off: downloaded through mlx-vlm's resolver, and logged.
    """
    path = Path(path_or_repo).expanduser()
    if path.is_dir():
        return str(path.resolve())
    from ..utils.model_cache import resolve_hf_snapshot_dir
    snapshot = resolve_hf_snapshot_dir(str(path_or_repo))
    if snapshot is None:
        from ..config.manager import get_config_manager
        name = str(path_or_repo)
        if get_config_manager().is_offline_first():
            from ..exceptions import ModelNotFoundError
            raise ModelNotFoundError(
                f"MLX drafter {name!r} is not in the local Hugging Face cache, and offline_first "
                f"is on, so loading never downloads it. Download it first: "
                f"`abstractcore models download mlx {name}` (or turn `offline.offline_first` off "
                "in the AbstractCore config to allow on-demand downloads)."
            )
        import logging
        logging.getLogger("abstractcore.providers.mlx").warning(
            "MLX drafter %r is not cached; offline_first is off, so it is being downloaded "
            "from the Hugging Face Hub now", name,
        )
        from mlx_vlm.utils import get_model_path
        snapshot = get_model_path(name)
    resolved = Path(snapshot).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Native MLX drafter directory is missing: {resolved}")
    return str(resolved)


def load_native_session(path: str, drafter_path=None):
    """Load one exact target/head pair once; callers configure ownership next."""
    target = str(Path(path).resolve())
    from .weights_fingerprint import weights_fingerprint_for_dir
    head = resolve_native_drafter_path(str(drafter_path)) if drafter_path else None
    head_fingerprint = weights_fingerprint_for_dir(head) if head else ""
    key = (target, head, head_fingerprint)
    with _LOAD_LOCK:
        session = _SESSIONS.get(key)
        if session is None:
            from mlx_vlm import load
            session = NativeSession()
            session.drafter_path = head
            session.drafter_weights_fingerprint = head_fingerprint
            if drafter_path:
                from mlx_vlm.speculative.drafters import load_drafter
                session.drafter, session.draft_kind = load_drafter(head)
                # Public depth is a fixed proposal count, not an adaptive cap.
                if hasattr(session.drafter, "__dict__"):
                    session.drafter.prefer_requested_block_size = True
            session.model, session.processor = load(target)
            _SESSIONS[key] = session
        return session
