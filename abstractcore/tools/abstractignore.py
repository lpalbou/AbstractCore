"""`.abstractignore` support for filesystem tools.

This module provides a small, runtime-enforced ignore policy for AbstractCore
filesystem tools (search/list/read/write/edit/analyze).

Goals:
- Avoid accidental scanning/editing of runtime artifacts (e.g. JsonFileRunStore
  directories ending in `.d/`).
- Allow users to define additional ignore rules via a `.abstractignore` file,
  inspired by `.gitignore` (not full parity).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import fnmatch
from typing import Iterable, List, Optional, Tuple


_DEFAULT_IGNORE_LINES: List[str] = [
    # VCS + caches
    ".git/",
    ".hg/",
    ".svn/",
    "__pycache__/",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    # Common build/env dirs
    "node_modules/",
    "dist/",
    "build/",
    ".venv/",
    "venv/",
    "env/",
    ".env/",
    # Editor/host artifacts
    ".cursor/",
    # AbstractFramework runtime stores (AbstractCode/Runtime file-backed stores)
    "*.d/",
]


@dataclass(frozen=True)
class AbstractIgnoreRule:
    pattern: str
    negate: bool = False
    dir_only: bool = False
    anchored: bool = False


def _parse_rules(lines: Iterable[str]) -> List[AbstractIgnoreRule]:
    rules: List[AbstractIgnoreRule] = []
    for raw in lines:
        line = str(raw or "").strip()
        if not line or line.startswith("#"):
            continue
        negate = False
        if line.startswith("!"):
            negate = True
            line = line[1:].strip()
        if not line:
            continue
        dir_only = line.endswith("/")
        if dir_only:
            line = line[:-1].strip()
        anchored = line.startswith("/")
        if anchored:
            line = line[1:].strip()
        if not line:
            continue
        rules.append(AbstractIgnoreRule(pattern=line, negate=negate, dir_only=dir_only, anchored=anchored))
    return rules


def _find_nearest_abstractignore(start: Path) -> Optional[Path]:
    """Find the nearest `.abstractignore` by walking upward from start."""
    cur = start if start.is_dir() else start.parent
    cur = cur.resolve()
    for p in (cur, *cur.parents):
        candidate = p / ".abstractignore"
        if candidate.is_file():
            return candidate
    return None


class AbstractIgnore:
    """A simple ignore matcher loaded from `.abstractignore` + defaults."""

    def __init__(self, *, root: Path, rules: List[AbstractIgnoreRule], source: Optional[Path] = None):
        self.root = root.resolve()
        self.rules = list(rules)
        self.source = source.resolve() if isinstance(source, Path) else None

    @classmethod
    def for_path(cls, path: Path) -> "AbstractIgnore":
        """Create an ignore matcher for a given path (file or dir)."""
        start = path if path.is_dir() else path.parent
        ignore_file = _find_nearest_abstractignore(start)
        root = ignore_file.parent if ignore_file is not None else start
        file_rules: List[AbstractIgnoreRule] = []
        if ignore_file is not None:
            try:
                file_rules = _parse_rules(ignore_file.read_text(encoding="utf-8").splitlines())
            except Exception:
                file_rules = []
        # Defaults first; user file rules can override via negation.
        rules = _parse_rules(_DEFAULT_IGNORE_LINES) + file_rules
        return cls(root=root, rules=rules, source=ignore_file)

    def _rel(self, path: Path) -> Tuple[str, List[str]]:
        p = path.resolve()
        try:
            rel = p.relative_to(self.root)
            rel_str = rel.as_posix()
            parts = list(rel.parts)
        except Exception:
            # If the target is outside root, fall back to absolute matching.
            rel_str = p.as_posix().lstrip("/")
            parts = list(p.parts)
        return rel_str, [str(x) for x in parts if str(x)]

    def is_ignored(self, path: Path, *, is_dir: Optional[bool] = None) -> bool:
        p = path.resolve()
        if is_dir is None:
            try:
                is_dir = p.is_dir()
            except Exception:
                is_dir = False

        rel_str, parts = self._rel(p)
        name = parts[-1] if parts else p.name
        dir_parts = parts if is_dir else parts[:-1]

        ignored = False
        for rule in self.rules:
            pat = rule.pattern
            if not pat:
                continue

            matched = False
            if rule.dir_only:
                # Match against any directory prefix.
                for i in range(1, len(dir_parts) + 1):
                    prefix = "/".join(dir_parts[:i])
                    if fnmatch.fnmatchcase(prefix, pat) or fnmatch.fnmatchcase(dir_parts[i - 1], pat):
                        matched = True
                        break
                if not matched and is_dir:
                    matched = fnmatch.fnmatchcase(rel_str, pat) or fnmatch.fnmatchcase(name, pat)
            else:
                if rule.anchored or ("/" in pat):
                    matched = fnmatch.fnmatchcase(rel_str, pat)
                else:
                    matched = fnmatch.fnmatchcase(name, pat)

            if matched:
                ignored = not rule.negate

        return ignored


__all__ = ["AbstractIgnore", "AbstractIgnoreRule"]


