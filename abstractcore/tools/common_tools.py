"""
Common shareable tools for AbstractCore applications.

This module provides a collection of utility tools for file operations,
web scraping, command execution, and user interaction.

Migrated from legacy system with enhanced decorator support.
"""

import os
import subprocess
import requests
from pathlib import Path
from typing import Optional, Dict, Any, Union
import platform
import re
import time
import json
import base64
import ast
from datetime import datetime
from urllib.parse import urlparse, urljoin
import mimetypes

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

try:
    import lxml  # noqa: F401
    BS4_PARSER = "lxml"
except ImportError:
    BS4_PARSER = "html.parser"

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import our enhanced tool decorator
from abstractcore.tools.core import tool
from abstractcore.utils.structured_logging import get_logger

logger = get_logger(__name__)


def _path_for_display(path: Path) -> str:
    """Best-effort absolute path for tool outputs (avoid CWD ambiguity)."""
    try:
        return str(path.expanduser().absolute())
    except Exception:
        try:
            return str(path.expanduser().resolve())
        except Exception:
            return str(path)


def _detect_code_language(path: Path, language: Optional[str]) -> Optional[str]:
    raw = str(language or "").strip().lower()
    if raw:
        if raw in {"py", "python"}:
            return "python"
        if raw in {"js", "javascript", "node"}:
            return "javascript"
        if raw in {"ts", "typescript"}:
            return "javascript"  # treat TS as JS for now (heuristic outline)
        return None

    ext = path.suffix.lower()
    if ext == ".py":
        return "python"
    if ext in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
        return "javascript"
    return None


def _format_line_range(start: Optional[int], end: Optional[int]) -> str:
    s = int(start or 0)
    e = int(end or 0)
    if s <= 0:
        return "?"
    if e <= 0 or e == s:
        return f"{s}"
    return f"{s}-{e}"


def _node_line_range(node: ast.AST) -> tuple[Optional[int], Optional[int]]:
    start = getattr(node, "lineno", None)
    end = getattr(node, "end_lineno", None)
    try:
        start_i = int(start) if start is not None else None
    except Exception:
        start_i = None
    try:
        end_i = int(end) if end is not None else start_i
    except Exception:
        end_i = start_i
    return start_i, end_i


def _safe_unparse(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node).strip()
    except Exception:
        return ""


def _format_python_function_signature(fn: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
    args = fn.args

    def _format_arg(a: ast.arg, default: Optional[ast.AST]) -> str:
        name = str(a.arg)
        ann = _safe_unparse(a.annotation)
        out = f"{name}: {ann}" if ann else name
        if default is not None:
            out += f"={_safe_unparse(default) or '…'}"
        return out

    pos_only = list(args.posonlyargs or [])
    pos_or_kw = list(args.args or [])
    kw_only = list(args.kwonlyargs or [])

    positional = pos_only + pos_or_kw
    defaults = list(args.defaults or [])
    default_start = len(positional) - len(defaults)
    default_by_index: Dict[int, ast.AST] = {}
    for i, d in enumerate(defaults):
        default_by_index[default_start + i] = d

    parts: list[str] = []
    for i, a in enumerate(positional):
        parts.append(_format_arg(a, default_by_index.get(i)))
        if pos_only and i == len(pos_only) - 1:
            parts.append("/")

    if args.vararg is not None:
        var = args.vararg
        ann = _safe_unparse(var.annotation)
        parts.append(("*" + var.arg + (f": {ann}" if ann else "")))
    elif kw_only:
        parts.append("*")

    kw_defaults = list(args.kw_defaults or [])
    for i, a in enumerate(kw_only):
        default = kw_defaults[i] if i < len(kw_defaults) else None
        parts.append(_format_arg(a, default))

    if args.kwarg is not None:
        kw = args.kwarg
        ann = _safe_unparse(kw.annotation)
        parts.append(("**" + kw.arg + (f": {ann}" if ann else "")))

    ret = _safe_unparse(fn.returns)
    prefix = "async " if isinstance(fn, ast.AsyncFunctionDef) else ""
    sig = f"{prefix}{fn.name}(" + ", ".join([p for p in parts if p]) + ")"
    if ret:
        sig += f" -> {ret}"
    return sig


def _collect_self_attributes(fn: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> list[str]:
    attrs: set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            for t in node.targets:
                _handle_target(t)
            self.generic_visit(node.value)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            _handle_target(node.target)
            self.generic_visit(node.value)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            _handle_target(node.target)
            self.generic_visit(node.value)

    def _handle_target(t: ast.AST) -> None:
        if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
            if isinstance(t.attr, str) and t.attr:
                attrs.add(t.attr)

    Visitor().visit(fn)
    return sorted(attrs)


def _collect_calls(fn: Union[ast.FunctionDef, ast.AsyncFunctionDef], *, local_functions: set[str], local_classes: set[str]) -> dict[str, list[tuple[str, int]]]:
    calls: list[tuple[str, int]] = []
    instantiates: list[tuple[str, int]] = []

    class Visitor(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call) -> None:
            name: Optional[str] = None
            if isinstance(node.func, ast.Name):
                name = node.func.id
                if name in local_classes:
                    instantiates.append((name, int(getattr(node, "lineno", 0) or 0)))
                elif name in local_functions:
                    calls.append((name, int(getattr(node, "lineno", 0) or 0)))
            self.generic_visit(node)

    Visitor().visit(fn)
    return {"calls": calls, "instantiates": instantiates}


def _brace_match_end_line(lines: list[str], *, start_line_index: int, start_col: int) -> Optional[int]:
    """Return 1-indexed end line for a JS/TS block starting at the given '{' position."""
    depth = 0
    in_single = False
    in_double = False
    in_template = False
    in_block_comment = False

    for i in range(start_line_index, len(lines)):
        line = lines[i]
        j = start_col if i == start_line_index else 0
        while j < len(line):
            ch = line[j]
            pair = line[j : j + 2]

            if in_block_comment:
                if pair == "*/":
                    in_block_comment = False
                    j += 2
                    continue
                j += 1
                continue

            if in_single:
                if ch == "\\":
                    j += 2
                    continue
                if ch == "'":
                    in_single = False
                j += 1
                continue

            if in_double:
                if ch == "\\":
                    j += 2
                    continue
                if ch == '"':
                    in_double = False
                j += 1
                continue

            if in_template:
                if ch == "\\":
                    j += 2
                    continue
                if ch == "`":
                    in_template = False
                j += 1
                continue

            # Not in string/comment.
            if pair == "/*":
                in_block_comment = True
                j += 2
                continue
            if pair == "//":
                break
            if ch == "'":
                in_single = True
                j += 1
                continue
            if ch == '"':
                in_double = True
                j += 1
                continue
            if ch == "`":
                in_template = True
                j += 1
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
            j += 1
    return None


@tool(
    description="Return a structured outline of a Python/JavaScript file (imports/classes/functions with line ranges) to guide precise edits.",
    when_to_use="Use before editing to locate the right block quickly; then read_file(start_line/end_line) around that block instead of re-reading the whole file.",
    examples=[
        {"description": "Outline a Python file", "arguments": {"file_path": "src/app.py"}},
        {"description": "Outline a JavaScript file", "arguments": {"file_path": "web/app.js"}},
        {"description": "Force language mode", "arguments": {"file_path": "script.txt", "language": "python"}},
    ],
)
def analyze_code(file_path: str, language: Optional[str] = None) -> str:
    """
    Return a structured outline of a Python/JavaScript code file with line ranges.

    IMPORTANT: Use this tool first for code navigation. Then use `read_file(start_line/end_line)`
    around the specific block you want to change, followed by `edit_file(...)` for bounded edits.

    Args:
        file_path: required; Path to the file to analyze (required; relative or absolute)
        language: Optional override for language detection ("python" or "javascript")

    Returns:
        A formatted outline including imports, classes, functions/methods, and (for JavaScript)
        resolved references to local modules.

    Examples:
        analyze_code(file_path="src/app.py")
        analyze_code(file_path="web/app.js")
        analyze_code(file_path="script.txt", language="python")
    """
    path = Path(file_path).expanduser()
    display_path = _path_for_display(path)
    # Runtime-enforced filesystem ignore policy (.abstractignore + defaults).
    from .abstractignore import AbstractIgnore

    ignore = AbstractIgnore.for_path(path)
    if ignore.is_ignored(path, is_dir=False):
        return f"Error: File '{display_path}' is ignored by .abstractignore policy"
    if not path.exists():
        return f"Error: File '{display_path}' does not exist"
    if not path.is_file():
        return f"Error: '{display_path}' is not a file"

    lang = _detect_code_language(path, language)
    if not lang:
        return f"Error: Unsupported code language for '{display_path}'. Supported: python, javascript"

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return f"Error: Cannot read '{display_path}' - file appears to be binary"
    except Exception as e:
        return f"Error reading file: {str(e)}"

    lines = text.splitlines()
    total_lines = len(lines)

    out: list[str] = [
        f"Code Analysis: {display_path} (language={lang}, lines={total_lines})",
        "Next step: use read_file(start_line/end_line) around the block you want to change, then edit_file(start_line/end_line) for a bounded edit.",
    ]

    if lang == "python":
        try:
            tree = ast.parse(text, filename=str(display_path))
        except SyntaxError as e:
            loc = f"line {getattr(e, 'lineno', '?')}"
            return f"Error: Python syntax error in '{display_path}' ({loc}): {str(e).strip()}"

        imports: list[str] = []
        module_assigns: list[str] = []
        functions: list[dict[str, Any]] = []
        classes: list[dict[str, Any]] = []

        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                start, end = _node_line_range(node)
                snippet = "\n".join(lines[(start or 1) - 1 : (end or start or 1)]).strip()
                imports.append(f"  - {_format_line_range(start, end)}: {snippet or _safe_unparse(node)}")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                start, end = _node_line_range(node)
                names: list[str] = []
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name):
                        names.append(t.id)
                if names:
                    module_assigns.append(f"  - {_format_line_range(start, end)}: {', '.join(sorted(set(names)))}")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start, end = _node_line_range(node)
                functions.append(
                    {
                        "name": node.name,
                        "sig": _format_python_function_signature(node),
                        "start": start,
                        "end": end,
                    }
                )
            elif isinstance(node, ast.ClassDef):
                start, end = _node_line_range(node)
                bases = [_safe_unparse(b) for b in (node.bases or []) if _safe_unparse(b)]
                methods: list[dict[str, Any]] = []
                self_attrs: set[str] = set()
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        ms, me = _node_line_range(item)
                        methods.append({"sig": _format_python_function_signature(item), "start": ms, "end": me, "name": item.name})
                        self_attrs.update(_collect_self_attributes(item))
                classes.append(
                    {
                        "name": node.name,
                        "bases": bases,
                        "start": start,
                        "end": end,
                        "methods": methods,
                        "self_attrs": sorted(self_attrs),
                    }
                )

        local_functions = {f["name"] for f in functions}
        local_classes = {c["name"] for c in classes}

        relationships: list[str] = []
        for c in classes:
            for m in c["methods"]:
                fn_node = None
                # Re-walk AST to find the matching node (cheap; file already parsed).
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and getattr(node, "name", None) == m["name"]:
                        # Best-effort: ensure we're inside the class range.
                        ns, ne = _node_line_range(node)
                        if ns and c["start"] and c["end"] and c["start"] <= ns <= c["end"]:
                            fn_node = node
                            break
                if fn_node is None:
                    continue
                rel = _collect_calls(fn_node, local_functions=local_functions, local_classes=local_classes)
                for name, ln in rel["instantiates"]:
                    relationships.append(f"  - instantiates: {c['name']}.{m['name']} -> {name} (line {ln})")
                for name, ln in rel["calls"]:
                    relationships.append(f"  - calls: {c['name']}.{m['name']} -> {name} (line {ln})")

        for f in functions:
            fn_node = None
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == f["name"]:
                    fn_node = node
                    break
            if fn_node is None:
                continue
            rel = _collect_calls(fn_node, local_functions=local_functions, local_classes=local_classes)
            for name, ln in rel["instantiates"]:
                relationships.append(f"  - instantiates: {f['name']} -> {name} (line {ln})")
            for name, ln in rel["calls"]:
                relationships.append(f"  - calls: {f['name']} -> {name} (line {ln})")

        out.append("language: python")
        out.append("imports:" if imports else "imports: []")
        out.extend(imports)
        out.append("module_assignments:" if module_assigns else "module_assignments: []")
        out.extend(module_assigns)

        out.append("classes:" if classes else "classes: []")
        for c in classes:
            bases = f" bases=[{', '.join(c['bases'])}]" if c["bases"] else ""
            out.append(f"  - {c['name']} (lines {_format_line_range(c['start'], c['end'])}){bases}")
            if c["methods"]:
                out.append("    methods:")
                for m in c["methods"]:
                    out.append(f"      - {_format_line_range(m['start'], m['end'])}: {m['sig']}")
            if c["self_attrs"]:
                out.append("    self_attributes_set: " + ", ".join(c["self_attrs"]))

        out.append("functions:" if functions else "functions: []")
        for f in functions:
            out.append(f"  - {_format_line_range(f['start'], f['end'])}: {f['sig']}")

        out.append("relationships:" if relationships else "relationships: []")
        out.extend(relationships[:50])
        if len(relationships) > 50:
            out.append(f"  - ... ({len(relationships) - 50} more)")

    else:
        # JavaScript/TypeScript (best-effort heuristic parsing).
        out.append("language: javascript")
        imports: list[str] = []
        classes: list[dict[str, Any]] = []
        functions: list[dict[str, Any]] = []
        module_assigns: list[str] = []
        refs: list[str] = []

        file_dir = path.parent.absolute()

        import_re = re.compile(r"^\s*import\s+(?:.+?\s+from\s+)?[\"'](?P<src>[^\"']+)[\"']\s*;?\s*$")
        import_from_re = re.compile(r"^\s*import\s+.+?\s+from\s+[\"'](?P<src>[^\"']+)[\"']\s*;?\s*$")
        require_re = re.compile(r"require\(\s*[\"'](?P<src>[^\"']+)[\"']\s*\)")

        class_re = re.compile(r"^\s*(?:export\s+)?class\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?:extends\s+(?P<base>[A-Za-z0-9_$.]+))?")
        func_re = re.compile(r"^\s*(?:export\s+)?function\s+(?P<name>[A-Za-z_$][\w$]*)\s*\((?P<params>[^)]*)\)")
        arrow_re = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(?(?P<params>[^)=]*)\)?\s*=>")
        var_re = re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\b")

        for i, raw in enumerate(lines, 1):
            line = raw.strip()
            if not line or line.startswith("//"):
                continue

            m = import_from_re.match(raw) or import_re.match(raw)
            if m:
                src = m.group("src")
                imports.append(f"  - {i}: import {src}")
                continue
            m = require_re.search(raw)
            if m:
                src = m.group("src")
                imports.append(f"  - {i}: require {src}")
                continue

        # Resolve local import paths (best-effort; only relative paths).
        def _resolve_js_ref(src: str) -> Optional[str]:
            if not src or not (src.startswith(".") or src.startswith("/")):
                return None
            base = Path(src)
            cand_base = (file_dir / base).absolute() if not base.is_absolute() else base
            candidates = []
            if cand_base.suffix:
                candidates.append(cand_base)
            else:
                for ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
                    candidates.append(Path(str(cand_base) + ext))
                candidates.append(cand_base / "index.js")
                candidates.append(cand_base / "index.ts")
            for c in candidates:
                try:
                    if c.exists() and c.is_file():
                        return str(c.absolute())
                except Exception:
                    continue
            return str(candidates[0].absolute()) if candidates else None

        for entry in imports:
            # entry looks like "  - <line>: import <src>" or "  - <line>: require <src>"
            parts = entry.split()
            src = parts[-1] if parts else ""
            resolved = _resolve_js_ref(src)
            if resolved:
                suffix = " (exists)" if Path(resolved).exists() else " (missing)"
                refs.append(f"  - {src} -> {resolved}{suffix}")

        # Classes + functions (brace matched).
        for idx, raw in enumerate(lines):
            line_no = idx + 1
            m = class_re.match(raw)
            if m:
                name = m.group("name")
                base = (m.group("base") or "").strip()
                open_pos = raw.find("{")
                if open_pos == -1:
                    # Find '{' on following lines.
                    for j in range(idx + 1, min(idx + 10, len(lines))):
                        pos = lines[j].find("{")
                        if pos != -1:
                            idx_open = j
                            open_pos = pos
                            break
                    else:
                        idx_open = idx
                        open_pos = 0
                else:
                    idx_open = idx

                end_line = _brace_match_end_line(lines, start_line_index=idx_open, start_col=open_pos) or line_no
                classes.append({"name": name, "base": base, "start": line_no, "end": end_line, "methods": []})
                continue

            m = func_re.match(raw)
            if m:
                name = m.group("name")
                params = (m.group("params") or "").strip()
                open_pos = raw.find("{")
                if open_pos != -1:
                    end_line = _brace_match_end_line(lines, start_line_index=idx, start_col=open_pos) or line_no
                else:
                    end_line = line_no
                functions.append({"name": name, "sig": f"{name}({params})", "start": line_no, "end": end_line})
                continue

            m = arrow_re.match(raw)
            if m:
                name = m.group("name")
                params = (m.group("params") or "").strip()
                open_pos = raw.find("{")
                if open_pos != -1:
                    end_line = _brace_match_end_line(lines, start_line_index=idx, start_col=open_pos) or line_no
                else:
                    end_line = line_no
                functions.append({"name": name, "sig": f"{name}({params}) =>", "start": line_no, "end": end_line})
                continue

            m = var_re.match(raw)
            if m:
                module_assigns.append(f"  - {line_no}: {m.group('name')}")

        out.append("imports:" if imports else "imports: []")
        out.extend(imports)
        out.append("module_assignments:" if module_assigns else "module_assignments: []")
        out.extend(module_assigns[:50])
        if len(module_assigns) > 50:
            out.append(f"  - ... ({len(module_assigns) - 50} more)")

        out.append("classes:" if classes else "classes: []")
        for c in classes:
            base = f" extends {c['base']}" if c["base"] else ""
            out.append(f"  - {c['name']} (lines {_format_line_range(c['start'], c['end'])}){base}")

        out.append("functions:" if functions else "functions: []")
        for f in functions:
            out.append(f"  - {_format_line_range(f['start'], f['end'])}: {f['sig']}")

        out.append("references:" if refs else "references: []")
        out.extend(refs[:50])
        if len(refs) > 50:
            out.append(f"  - ... ({len(refs) - 50} more)")
        out.append("notes: JavaScript parsing is best-effort (heuristic, not a full AST).")

    return "\n".join(out).rstrip()


# File Operations
@tool(
    description="List files/directories by name/path using glob patterns (case-insensitive). Does NOT search file contents; head_limit defaults to 10 results.",
    when_to_use="Use to find files by filename/path; prefer narrow patterns like '*.py|*.md' (avoid '*') and raise head_limit if needed. For file contents, use search_files().",
    examples=[
        {
            "description": "List Python + Markdown files in current directory",
            "arguments": {
                "directory_path": ".",
                "pattern": "*.py|*.md"
            }
        },
        {
            "description": "Find all Python files recursively",
            "arguments": {
                "directory_path": ".",
                "pattern": "*.py",
                "recursive": True
            }
        },
        {
            "description": "Find docs/config files recursively",
            "arguments": {
                "directory_path": ".",
                "pattern": "*.md|*.yml|*.yaml|*.json",
                "recursive": True
            }
        }
    ]
)
def list_files(directory_path: str = ".", pattern: str = "*", recursive: bool = False, include_hidden: bool = False, head_limit: Optional[int] = 10) -> str:
    """
    List files and directories in a specified directory with pattern matching (case-insensitive).

    IMPORTANT: Use 'directory_path' parameter (not 'file_path') to specify the directory to list.

    Args:
        directory_path: Path to the directory to list files from (default: "." for current directory)
        pattern: Glob pattern(s) to match files. Use "|" to separate multiple patterns (default: "*")
        recursive: Whether to search recursively in subdirectories (default: False)
        include_hidden: Whether to include hidden files/directories starting with '.' (default: False)
        head_limit: Maximum number of entries to return (default: 25, None for unlimited)

    Returns:
        Formatted string with file and directory listings or error message.
        When head_limit is applied, shows "showing X of Y files" in the header.

    Examples:
        list_files(directory_path="docs") - Lists files in the docs directory
        list_files(pattern="*.py") - Lists Python files (case-insensitive)
        list_files(pattern="*.py|*.js|*.md") - Lists Python, JavaScript, and Markdown files
        list_files(pattern="README*|*test*|config.*") - Lists README files, test files, and config files
        list_files(pattern="*TEST*", recursive=True) - Finds test files recursively (case-insensitive)
    """
    try:
        # Convert head_limit to int if it's a string (defensive programming)
        if isinstance(head_limit, str):
            try:
                head_limit = int(head_limit)
            except ValueError:
                head_limit = 25  # fallback to default
        
        # Expand home directory shortcuts like ~
        directory_input = Path(directory_path).expanduser()
        directory = directory_input.absolute()
        directory_display = str(directory)

        # Runtime-enforced filesystem ignore policy (.abstractignore + defaults).
        from .abstractignore import AbstractIgnore

        ignore = AbstractIgnore.for_path(directory)
        if ignore.is_ignored(directory, is_dir=True):
            return f"Error: Directory '{directory_display}' is ignored by .abstractignore policy"

        if not directory.exists():
            return f"Error: Directory '{directory_display}' does not exist"

        if not directory.is_dir():
            return f"Error: '{directory_display}' is not a directory"

        # Best-effort existence checks for clearer/no-surprises messaging.
        has_any_entries = False
        has_any_visible_entries = False
        try:
            for p in directory.iterdir():
                has_any_entries = True
                if include_hidden or not p.name.startswith("."):
                    has_any_visible_entries = True
                    break
        except Exception:
            # If we cannot enumerate entries (permissions, transient FS issues), fall back
            # to the existing "no matches" messaging below.
            pass

        # Split pattern by | to support multiple patterns
        patterns = [p.strip() for p in pattern.split('|')]

        # Get all entries first (files + directories), then apply case-insensitive pattern matching.
        #
        # NOTE: This tool is intentionally named `list_files` for historical reasons, but it
        # should list directories too. This is important for agent workflows that need to
        # confirm that `mkdir -p ...` succeeded even before any files exist.
        import fnmatch
        all_entries = []

        if recursive:
            for root, dirs, dir_files in os.walk(directory):
                # Prune hidden directories early unless explicitly requested.
                if not include_hidden:
                    dirs[:] = [d for d in dirs if not str(d).startswith(".")]
                # Prune ignored directories (including AbstractRuntime store dirs like `*.d/`).
                try:
                    dirs[:] = [d for d in dirs if not ignore.is_ignored(Path(root) / d, is_dir=True)]
                except Exception:
                    pass

                # Include directories (so empty folders still show up)
                for d in dirs:
                    if not include_hidden and str(d).startswith("."):
                        continue
                    p = Path(root) / d
                    if not ignore.is_ignored(p, is_dir=True):
                        all_entries.append(p)

                # Include files
                for f in dir_files:
                    if not include_hidden and str(f).startswith("."):
                        continue
                    p = Path(root) / f
                    if not ignore.is_ignored(p, is_dir=False):
                        all_entries.append(p)
        else:
            try:
                # Include both files and directories for better UX and agent correctness.
                all_entries = [p for p in directory.iterdir() if not ignore.is_ignored(p)]
            except PermissionError:
                pass

        # Apply case-insensitive pattern matching
        matched_files = []
        for entry_path in all_entries:
            filename = entry_path.name

            # Check if file matches any pattern (case-insensitive)
            for single_pattern in patterns:
                if fnmatch.fnmatch(filename.lower(), single_pattern.lower()):
                    matched_files.append(str(entry_path))
                    break

        files = matched_files

        if not files:
            if not has_any_entries:
                return f"Directory '{directory_display}' exists but is empty"
            if not include_hidden and not has_any_visible_entries:
                return f"Directory '{directory_display}' exists but contains only hidden entries (use include_hidden=True)"
            return f"Directory '{directory_display}' exists but no entries match pattern '{pattern}'"

        # Filter out hidden entries if include_hidden is False.
        if not include_hidden:
            filtered_files = []
            for file_path in files:
                path_obj = Path(file_path)
                # Check if any part of the path (after the directory_path) starts with '.'
                try:
                    relative_path = path_obj.relative_to(directory)
                except Exception:
                    relative_path = path_obj
                is_hidden = any(part.startswith(".") for part in relative_path.parts)
                if not is_hidden:
                    filtered_files.append(file_path)
            files = filtered_files

        if not files:
            hidden_note = " (hidden entries excluded)" if not include_hidden else ""
            if not has_any_entries:
                return f"Directory '{directory_display}' exists but is empty"
            if not include_hidden and not has_any_visible_entries:
                return f"Directory '{directory_display}' exists but contains only hidden entries (use include_hidden=True)"
            return f"Directory '{directory_display}' exists but no entries match pattern '{pattern}'{hidden_note}"

        # Remove duplicates and sort files by modification time (most recent first), then alphabetically
        unique_files = set(files)
        try:
            # Sort by modification time (most recent first) for better relevance
            files = sorted(unique_files, key=lambda f: (Path(f).stat().st_mtime if Path(f).exists() else 0), reverse=True)
        except Exception:
            # Fallback to alphabetical sorting if stat fails
            files = sorted(unique_files)

        # Apply head_limit if specified
        total_files = len(files)
        is_truncated = False
        if head_limit is not None and head_limit > 0 and len(files) > head_limit:
            files = files[:head_limit]
            limit_note = f" (showing {head_limit} of {total_files} entries)"
            is_truncated = True
        else:
            limit_note = ""

        hidden_note = " (hidden entries excluded)" if not include_hidden else ""
        output = [f"Entries in '{directory_display}' matching '{pattern}'{hidden_note}{limit_note}:"]

        for file_path in files:
            path_obj = Path(file_path)
            # Prefer relative paths for recursive listings; keeps results unambiguous.
            try:
                display_path = str(path_obj.relative_to(directory))
            except Exception:
                display_path = path_obj.name
            if path_obj.is_file():
                size = path_obj.stat().st_size
                size_str = f"{size:,} bytes"
                output.append(f"  {display_path} ({size_str})")
            elif path_obj.is_dir():
                # Ensure directories are visually distinct and easy to parse.
                suffix = "/" if not display_path.endswith("/") else ""
                output.append(f"  {display_path}{suffix}")

        # Add helpful hint when results are truncated
        if is_truncated:
            remaining = total_files - head_limit
            hint_args = [f'directory_path="{directory_display}"', f'pattern="{pattern}"']
            if recursive:
                hint_args.append("recursive=True")
            if include_hidden:
                hint_args.append("include_hidden=True")
            hint_args.append("head_limit=None")
            output.append(
                "\n"
                f"Note: {remaining} more entries available. "
                f"Next step: use list_files({', '.join(hint_args)}) to see all."
            )

        return "\n".join(output)

    except Exception as e:
        return f"Error listing files: {str(e)}"


@tool(
    description="Search INSIDE file contents for a text/code pattern (regex) and return matches with line numbers.",
    when_to_use="Use to find which files contain some text/code and where (line numbers). For filenames/paths, use list_files().",
    examples=[
        {
            "description": "Find files with function definitions containing 'search'",
            "arguments": {
                "pattern": "def.*search",
                "path": ".",
                "file_pattern": "*.py"
            }
        },
        {
            "description": "Count import statements with 're' module",
            "arguments": {
                "pattern": "import.*re",
                "path": ".",
                "output_mode": "count"
            }
        },
        {
            "description": "Show line-numbered context (±5 lines) around matches for precise editing",
            "arguments": {
                "pattern": "K_SPACE",
                "path": "game.py",
                "output_mode": "context",
                "context_lines": 5
            }
        }
    ]
)
def search_files(
    pattern: str,
    path: str = ".",
    output_mode: str = "content",
    context_lines: int = 0,
    head_limit: Optional[int] = 20,
    file_pattern: str = "*",
    case_sensitive: bool = False,
    multiline: bool = False,
    include_hidden: bool = False,
    ignore_dirs: Optional[str] = None,
) -> str:
    """
    Enhanced search tool with regex support and flexible output modes.

    Similar to grep functionality, this tool can search for patterns in files
    with various output formats and options.

    Args:
        pattern: required; Regular expression pattern to search for
        path: File or directory path to search in (default: current directory)
        output_mode: Output format - "content" (show matching lines), "context" (show ±N lines around matches), "files_with_matches" (show file paths with line numbers), "count" (show match counts) (default: "content")
        context_lines: When output_mode="context", show this many lines before/after each match (default: 5 when output_mode="context" and context_lines=0)
        head_limit: Limit output to first N entries (default: 20)
        file_pattern: Glob pattern(s) for files to search. Use "|" to separate multiple patterns (default: "*" for all files)
        case_sensitive: Whether search should be case sensitive (default: False)
        multiline: Enable multiline matching where pattern can span lines (default: False)

    Returns:
        Search results in the specified format or error message

    Examples:
        search_files("generate.*react|create_react_cycle", "abstractcore/session.py")  # Returns matching lines with content (default)
        search_files("def.*search", ".", file_pattern="*.py")  # Search Python files only, show content
        search_files("import.*re", ".", file_pattern="*.py|*.js")  # Search Python and JavaScript files, show content
        search_files("TODO|FIXME", ".", file_pattern="*.py|*.md|*.txt")  # Find TODO/FIXME in multiple file types, show content
        search_files("K_SPACE", "game.py", output_mode="context", context_lines=5)  # Show context for editing
        search_files("import.*re", ".", "files_with_matches")  # Show file paths with line numbers instead of content
        search_files("pattern", ".", "count")  # Count matches per file
    """
    try:
        output_mode = str(output_mode or "content").strip().lower()

        # Normalize head_limit (treat <= 0 as "no limit").
        if head_limit is not None:
            try:
                head_limit_int = int(head_limit)
            except (TypeError, ValueError):
                head_limit_int = 20  # fallback to default
            head_limit = head_limit_int if head_limit_int > 0 else None
        
        # Expand home directory shortcuts like ~
        search_path_input = Path(path).expanduser()
        search_path = search_path_input.absolute()
        search_path_display = str(search_path)

        # Runtime-enforced filesystem ignore policy (.abstractignore + defaults).
        from .abstractignore import AbstractIgnore

        ignore = AbstractIgnore.for_path(search_path)
        try:
            if ignore.is_ignored(search_path, is_dir=search_path.is_dir()):
                return f"Error: Path '{search_path_display}' is ignored by .abstractignore policy"
        except Exception:
            # Best-effort; continue without policy if filesystem queries fail.
            ignore = AbstractIgnore.for_path(Path.cwd())

        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE | re.DOTALL

        try:
            regex_pattern = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern '{pattern}': {str(e)}"

        # Context output defaults to ±5 lines unless explicitly set.
        try:
            ctx = int(context_lines or 0)
        except Exception:
            ctx = 0
        if ctx < 0:
            ctx = 0
        if output_mode == "context" and ctx == 0:
            ctx = 5

        def _append_context_blocks(file_path_for_display: Path, line_texts: list, match_lines: list) -> None:
            if not match_lines:
                return
            results.append(f"\n📄 {file_path_for_display}:")

            total_lines = len(line_texts)
            ranges = []
            for ln in match_lines:
                start = max(1, ln - ctx)
                end = min(total_lines, ln + ctx)
                ranges.append((start, end))
            ranges.sort()

            merged = []
            for start, end in ranges:
                if not merged:
                    merged.append([start, end])
                    continue
                if start <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])

            selected_set = set(match_lines)
            for block_index, (start, end) in enumerate(merged, 1):
                if block_index > 1:
                    results.append("    …")
                for ln in range(start, end + 1):
                    text = line_texts[ln - 1]
                    prefix = "  >" if ln in selected_set else "   "
                    results.append(f"{prefix} {ln}: {text}")

        # Determine if path is a file or directory
        if search_path.is_file():
            if ignore.is_ignored(search_path, is_dir=False):
                return f"Error: File '{search_path_display}' is ignored by .abstractignore policy"
            files_to_search = [search_path]
        elif search_path.is_dir():
            # Find files matching pattern in directory
            # Default directories to ignore for safety/performance (user home and projects)
            default_ignores = {
                ".git", ".hg", ".svn", "__pycache__", "node_modules", "dist", "build",
                ".DS_Store", ".Trash", ".cache", ".venv", "venv", "env", ".env",
                ".cursor", "Library", "Applications", "System", "Volumes"
            }
            extra_ignores = set()
            if ignore_dirs:
                extra_ignores = {d.strip() for d in ignore_dirs.split('|') if d.strip()}
            ignore_set = default_ignores | extra_ignores

            if file_pattern == "*":
                # Search all files recursively
                files_to_search = []
                for root, dirs, files in os.walk(search_path):
                    # Prune directories in-place
                    dirs[:] = [
                        d for d in dirs
                        if (include_hidden or not d.startswith('.'))
                        and d not in ignore_set
                        and not ignore.is_ignored(Path(root) / d, is_dir=True)
                    ]
                    for file in files:
                        file_path = Path(root) / file
                        # Skip hidden files unless allowed
                        if not include_hidden and file_path.name.startswith('.'):
                            continue
                        if ignore.is_ignored(file_path, is_dir=False):
                            continue
                        # Skip non-regular files (sockets, fifos, etc.) and symlinks
                        try:
                            if not file_path.is_file() or file_path.is_symlink():
                                continue
                        except Exception:
                            continue
                        # Skip binary files by checking if they're text files
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                f.read(1024)  # Try to read first 1KB
                            files_to_search.append(file_path)
                        except (UnicodeDecodeError, PermissionError, OSError):
                            continue  # Skip binary/inaccessible files
            else:
                # Support multiple patterns separated by |
                import fnmatch
                file_patterns = [p.strip() for p in file_pattern.split('|')]
                files_to_search = []

                for root, dirs, files in os.walk(search_path):
                    # Prune directories in-place
                    dirs[:] = [
                        d for d in dirs
                        if (include_hidden or not d.startswith('.'))
                        and d not in ignore_set
                        and not ignore.is_ignored(Path(root) / d, is_dir=True)
                    ]
                    for file in files:
                        file_path = Path(root) / file
                        filename = file_path.name
                        # Skip hidden files unless allowed
                        if not include_hidden and filename.startswith('.'):
                            continue
                        if ignore.is_ignored(file_path, is_dir=False):
                            continue
                        # Skip non-regular files (sockets, fifos, etc.) and symlinks
                        try:
                            if not file_path.is_file() or file_path.is_symlink():
                                continue
                        except Exception:
                            continue

                        # Check if file matches any pattern (case-insensitive)
                        matches_pattern = False
                        for single_pattern in file_patterns:
                            if fnmatch.fnmatch(filename.lower(), single_pattern.lower()):
                                matches_pattern = True
                                break

                        if matches_pattern:
                            # Skip binary files by checking if they're text files
                            try:
                                with open(file_path, 'r', encoding='utf-8') as f:
                                    f.read(1024)  # Try to read first 1KB
                                files_to_search.append(file_path)
                            except (UnicodeDecodeError, PermissionError, OSError):
                                continue  # Skip binary/inaccessible files
        else:
            return f"Error: Path '{search_path_display}' does not exist"

        if not files_to_search:
            return f"No files found to search in '{search_path_display}'"

        # Search through files
        results = []
        files_with_matches = []  # Will store (file_path, [line_numbers]) tuples
        match_counts = {}
        total_matches = 0
        global_content_lines_added = 0  # Track content lines across all files
        global_context_matches_added = 0  # Count match LINES rendered in context mode (not output lines)

        for file_path in files_to_search:
            if output_mode == "context" and head_limit is not None and global_context_matches_added >= head_limit:
                break

            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    if multiline:
                        # For multiline mode, we need to read the full content
                        # But we'll be more efficient about extracting matching lines
                        content = f.read()
                        matches = list(regex_pattern.finditer(content))

                        if matches:
                            # Pre-split content into lines for efficiency
                            lines = content.splitlines()
                            
                            # Collect line numbers and prepare content efficiently
                            line_numbers = []
                            file_header_added = False
                            context_match_lines = []
                            context_seen = set()
                            remaining_context = None
                            if output_mode == "context" and head_limit is not None:
                                remaining_context = max(0, head_limit - global_context_matches_added)

                            for match in matches:
                                line_num = content.count('\n', 0, match.start()) + 1
                                line_numbers.append(line_num)
                                
                                if output_mode == "content":
                                    # Check global head_limit before adding any content
                                    if head_limit and global_content_lines_added >= head_limit:
                                        break
                                        
                                    # Add file header only once and only if we're showing content
                                    if not file_header_added:
                                        results.append(f"\n📄 {file_path}:")
                                        file_header_added = True
                                    
                                    # Get only the specific matching line (efficient)
                                    if line_num <= len(lines):
                                        full_line = lines[line_num - 1]
                                        results.append(f"    {line_num}: {full_line}")
                                        global_content_lines_added += 1
                                        
                                        # Check global head_limit after adding content
                                        if head_limit and global_content_lines_added >= head_limit:
                                            break
                                elif output_mode == "context":
                                    if line_num not in context_seen:
                                        context_seen.add(line_num)
                                        context_match_lines.append(line_num)
                                        if remaining_context is not None and len(context_match_lines) >= remaining_context:
                                            break

                            file_display = _path_for_display(file_path)
                            files_with_matches.append((file_display, line_numbers))
                            match_counts[file_display] = len(matches)
                            total_matches += len(matches)

                            if output_mode == "context":
                                _append_context_blocks(Path(file_display), lines, context_match_lines)
                                global_context_matches_added += len(context_match_lines)
                    else:
                        # Non-multiline mode: process line by line (more efficient)
                        lines = f.readlines()
                        matching_lines = []
                        line_numbers = []
                        file_header_added = False
                        context_match_lines = []
                        remaining_context = None
                        if output_mode == "context" and head_limit is not None:
                            remaining_context = max(0, head_limit - global_context_matches_added)
                        
                        for line_num, line in enumerate(lines, 1):
                            line_content = line.rstrip()
                            matches = list(regex_pattern.finditer(line_content))

                            if matches:
                                line_numbers.append(line_num)
                                matching_lines.extend(matches)
                                
                                # For content mode, add lines as we find them (more efficient)
                                if output_mode == "content":
                                    # Check global head_limit before adding any content
                                    if head_limit and global_content_lines_added >= head_limit:
                                        break
                                        
                                    # Add file header only once when we find the first match
                                    if not file_header_added:
                                        results.append(f"\n📄 {_path_for_display(file_path)}:")
                                        file_header_added = True
                                    
                                    results.append(f"    {line_num}: {line_content}")
                                    global_content_lines_added += 1
                                    
                                    # Check global head_limit after adding content
                                    if head_limit and global_content_lines_added >= head_limit:
                                        break
                                elif output_mode == "context":
                                    context_match_lines.append(line_num)
                                    if remaining_context is not None and len(context_match_lines) >= remaining_context:
                                        break

                        if matching_lines:
                            file_display = _path_for_display(file_path)
                            files_with_matches.append((file_display, line_numbers))
                            match_counts[file_display] = len(matching_lines)
                            total_matches += len(matching_lines)
                            if output_mode == "context":
                                line_texts = [l.rstrip("\n").rstrip("\r") for l in lines]
                                _append_context_blocks(Path(file_display), line_texts, context_match_lines)
                                global_context_matches_added += len(context_match_lines)

            except Exception as e:
                if output_mode == "content":
                    results.append(f"\n⚠️  Error reading {file_path}: {str(e)}")
            
            # Break out of file loop if we've reached the global head_limit
            if head_limit and output_mode == "content" and global_content_lines_added >= head_limit:
                break
            if head_limit and output_mode == "context" and global_context_matches_added >= head_limit:
                break

        # Format output based on mode
        if output_mode == "files_with_matches":
            total_files_with_matches = len(files_with_matches)
            is_truncated = False

            if head_limit and len(files_with_matches) > head_limit:
                files_with_matches = files_with_matches[:head_limit]
                is_truncated = True

            if files_with_matches:
                header = f"Files matching pattern '{pattern}':"
                formatted_results = [header]

                for file_path, line_numbers in files_with_matches:
                    # Format line numbers - show ALL line numbers since that's the main value of this mode
                    if len(line_numbers) == 1:
                        line_info = f"line {line_numbers[0]}"
                    else:
                        line_info = f"lines {', '.join(map(str, line_numbers))}"

                    formatted_results.append(f"{file_path} ({line_info})")

                # Add helpful hint when results are truncated
                if is_truncated:
                    remaining = total_files_with_matches - head_limit
                    case_hint = "" if case_sensitive else ", case_sensitive=False"
                    multiline_hint = ", multiline=True" if multiline else ""
                    file_pattern_hint = f", file_pattern='{file_pattern}'" if file_pattern != "*" else ""
                    formatted_results.append(
                        f"\n💡 {remaining} more files with matches available. "
                        f"Use search_files('{pattern}', '{search_path_display}', head_limit=None{case_hint}{multiline_hint}{file_pattern_hint}) to see all."
                    )

                return "\n".join(formatted_results)
            else:
                return f"No files found matching pattern '{pattern}'"

        elif output_mode == "count":
            all_count_items = list(match_counts.items())
            is_count_truncated = False

            if head_limit and len(all_count_items) > head_limit:
                count_items = all_count_items[:head_limit]
                is_count_truncated = True
            else:
                count_items = all_count_items

            if count_items:
                header = f"Match counts for pattern '{pattern}':"
                count_results = [header]
                for file_path, count in count_items:
                    count_results.append(f"{count:3d} {file_path}")
                count_results.append(f"\nTotal: {total_matches} matches in {len(files_with_matches)} files")

                # Add helpful hint when results are truncated
                if is_count_truncated:
                    remaining = len(all_count_items) - head_limit
                    case_hint = "" if case_sensitive else ", case_sensitive=False"
                    multiline_hint = ", multiline=True" if multiline else ""
                    file_pattern_hint = f", file_pattern='{file_pattern}'" if file_pattern != "*" else ""
                    count_results.append(
                        f"\n💡 {remaining} more files with matches available. "
                        f"Use search_files('{pattern}', '{search_path_display}', 'count', head_limit=None{case_hint}{multiline_hint}{file_pattern_hint}) to see all."
                    )

                return "\n".join(count_results)
            else:
                return f"No matches found for pattern '{pattern}'"

        elif output_mode == "context":
            if not results:
                return f"No matches found for pattern '{pattern}'"

            file_count = len([r for r in results if r.startswith("\n📄")])
            header = f"Search context for pattern '{pattern}' under '{search_path_display}' in {file_count} files (±{ctx} lines):"

            # Head-limit note (cap is on number of matches, not output lines).
            result_text = header + "\n" + "\n".join(results)
            if head_limit and global_context_matches_added >= head_limit:
                result_text += f"\n\n... (showing context for first {head_limit} matches)"
            return result_text

        else:  # content mode
            if not results:
                return f"No matches found for pattern '{pattern}'"

            # Count files with matches for header
            file_count = len([r for r in results if r.startswith("\n📄")])
            header = f"Search results for pattern '{pattern}' under '{search_path_display}' in {file_count} files:"

            # Apply head_limit to final output if specified
            final_results = results
            if head_limit:
                content_lines = [r for r in results if re.match("^\\s+\\d+:", r)]
                if len(content_lines) > head_limit:
                    # Keep file headers and trim content lines
                    trimmed_results = []
                    content_count = 0
                    for line in results:
                        if re.match("^\\s+\\d+:", line):
                            if content_count < head_limit:
                                trimmed_results.append(line)
                                content_count += 1
                        else:
                            trimmed_results.append(line)
                    final_results = trimmed_results
                    final_results.append(f"\n... (showing first {head_limit} matches)")

            # Add truncation notice if we hit the head_limit
            result_text = header + "\n" + "\n".join(final_results)
            if head_limit and global_content_lines_added >= head_limit:
                result_text += f"\n\n... (showing first {head_limit} matches)"
            
            return result_text

    except Exception as e:
        return f"Error performing search: {str(e)}"


@tool(
    description="Read a text file (line-numbered). Prefer analyze_code for code, then read_file(start_line/end_line); full reads may be refused if too large.",
    when_to_use="Use to inspect exact file contents. For code, prefer analyze_code first. Prefer bounded reads; if line numbers are unknown, use search_files(output_mode='context') first.",
    hide_args=["should_read_entire_file"],
    examples=[
        {
            "description": "Read entire file (only when it's small; large files are refused)",
            "arguments": {
                "file_path": "README.md"
            }
        },
        {
            "description": "Read specific line range",
            "arguments": {
                "file_path": "src/main.py",
                "start_line": 10,
                "end_line": 25
            }
        },
        {
            "description": "Read first 50 lines",
            "arguments": {
                "file_path": "large_file.txt",
                "end_line": 50
            }
        }
    ]
)
def read_file(
    file_path: str,
    should_read_entire_file: Optional[bool] = None,
    start_line: int = 1,
    end_line: Optional[int] = None,
) -> str:
    """
    Read the contents of a file with optional line range.

    Args:
        file_path: required; Path to the file to read
        start_line: Starting line number (1-indexed, default: 1)
        end_line: Ending line number (1-indexed, inclusive, optional)
        should_read_entire_file: Legacy/compatibility flag. If provided, overrides inference:
            - True  => attempt full read (or refuse if too large)
            - False => range mode (bounded by start_line/end_line)
            When omitted (recommended), mode is inferred:
            - no start/end hint => full read
            - start_line and/or end_line provided => range read

    Returns:
        File contents or error message
    """
    try:
        # Expand home directory shortcuts like ~
        path = Path(file_path).expanduser()
        display_path = _path_for_display(path)

        # Runtime-enforced filesystem ignore policy (.abstractignore + defaults).
        from .abstractignore import AbstractIgnore

        ignore = AbstractIgnore.for_path(path)
        if ignore.is_ignored(path, is_dir=False):
            return f"Error: File '{display_path}' is ignored by .abstractignore policy"

        if not path.exists():
            return f"Error: File '{display_path}' does not exist"

        if not path.is_file():
            return f"Error: '{display_path}' is not a file"

        # Guardrails: keep tool outputs bounded and avoid huge memory/time spikes.
        # These limits intentionally push agents toward: search_files(output_mode="context") → read_file(start_line/end_line) → edit_file(...)
        MAX_LINES_PER_CALL = 1000

        # Mode selection:
        # - Explicit legacy flag wins (for backwards compatibility).
        # - Otherwise infer: no range hint => full read; any range hint => slice read.
        try:
            inferred_start = int(start_line or 1)
        except Exception:
            inferred_start = 1
        if should_read_entire_file is True:
            read_entire = True
        elif should_read_entire_file is False:
            read_entire = False
        else:
            read_entire = end_line is None and inferred_start == 1

        with open(path, 'r', encoding='utf-8') as f:
            if read_entire:
                # Read entire file (bounded by MAX_LINES_PER_CALL). No truncation: either full content or refusal.
                raw_lines: list[str] = []
                for idx, line in enumerate(f, 1):
                    if idx > MAX_LINES_PER_CALL:
                        return (
                            f"Refused: File '{display_path}' is too large to read entirely "
                            f"(> {MAX_LINES_PER_CALL} lines).\n"
                            "Next step: use search_files(..., output_mode='context') to find the relevant line number(s), "
                            "then call read_file with start_line/end_line for a smaller range."
                        )
                    raw_lines.append(line.rstrip("\r\n"))

                line_count = len(raw_lines)
                num_width = max(1, len(str(line_count or 1)))
                numbered = "\n".join([f"{i:>{num_width}}: {line}" for i, line in enumerate(raw_lines, 1)])
                return f"File: {display_path} ({line_count} lines)\n\n{numbered}"
            else:
                # Read specific line range
                # Validate and convert to 0-indexed [start, end) slice with inclusive end.
                try:
                    start_line = int(start_line or 1)
                except Exception:
                    start_line = 1
                if start_line < 1:
                    return f"Error: start_line must be >= 1 (got {start_line})"

                end_line_value = None
                if end_line is not None:
                    try:
                        end_line_value = int(end_line)
                    except Exception:
                        return f"Error: end_line must be an integer (got {end_line})"
                    if end_line_value < 1:
                        return f"Error: end_line must be >= 1 (got {end_line_value})"

                if end_line_value is not None and start_line > end_line_value:
                    return f"Error: start_line ({start_line}) cannot be greater than end_line ({end_line_value})"

                if end_line_value is not None:
                    requested_lines = end_line_value - start_line + 1
                    if requested_lines > MAX_LINES_PER_CALL:
                        return (
                            f"Refused: Requested range would return {requested_lines} lines "
                            f"(> {MAX_LINES_PER_CALL} lines).\n"
                            "Next step: request a smaller range by narrowing end_line, "
                            "or use search_files(..., output_mode='context') to target the exact region."
                        )

                # Stream the file; collect only the requested lines.
                selected_lines: list[tuple[int, str]] = []
                last_line_seen = 0
                for line_no, line in enumerate(f, 1):
                    last_line_seen = line_no
                    if line_no < start_line:
                        continue
                    if end_line_value is not None and line_no > end_line_value:
                        break
                    selected_lines.append((line_no, line.rstrip("\r\n")))
                    if len(selected_lines) > MAX_LINES_PER_CALL:
                        return (
                            f"Refused: Requested range is too large to return in one call "
                            f"(> {MAX_LINES_PER_CALL} lines).\n"
                            "Next step: specify a smaller end_line, "
                            "or split the read into multiple smaller ranges."
                        )

                if last_line_seen < start_line:
                    return f"Error: Start line {start_line} exceeds file length ({last_line_seen} lines)"

                # Always include line numbers (1-indexed). Strip only line endings to preserve whitespace.
                end_width = selected_lines[-1][0] if selected_lines else start_line
                num_width = max(1, len(str(end_width)))
                result_lines = []
                for line_no, text in selected_lines:
                    result_lines.append(f"{line_no:>{num_width}}: {text}")

                header = f"File: {display_path} ({len(selected_lines)} lines)"
                return header + "\n\n" + "\n".join(result_lines)

    except UnicodeDecodeError:
        return f"Error: Cannot read '{_path_for_display(Path(file_path).expanduser())}' - file appears to be binary"
    except FileNotFoundError:
        return f"Error: File not found: {_path_for_display(Path(file_path).expanduser())}"
    except PermissionError:
        return f"Error: Permission denied reading file: {_path_for_display(Path(file_path).expanduser())}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool(
    description="Write full file content (create/overwrite/append). WARNING: mode='w' overwrites the entire file; for small edits, use edit_file().",
    when_to_use="Use to create new files or intentionally overwrite/append full content. For small edits, use edit_file().",
    hide_args=["create_dirs"],
    examples=[
        {
            "description": "Write a simple text file",
            "arguments": {
                "file_path": "output.txt",
                "content": "Hello, world!"
            }
        },
        {
            "description": "Overwrite an existing config file with complete new content (intentional whole-file rewrite)",
            "arguments": {
                "file_path": "config.json",
                "content": "{\n  \"api_key\": \"test\",\n  \"debug\": true\n}\n",
                "mode": "w",
            },
        },
        {
            "description": "Append to existing file",
            "arguments": {
                "file_path": "log.txt",
                "content": "\nNew log entry at 2025-01-01",
                "mode": "a"
            }
        },
    ]
)
def write_file(file_path: str, content: str, mode: str = "w", create_dirs: bool = True) -> str:
    """
    Write content to a file with robust error handling.

    This tool creates or overwrites a file with the specified content.
    It can optionally create parent directories if they don't exist.

    Args:
        file_path: Path to the file to write (required; can be relative or absolute)
        content: The content to write to the file (required; use "" explicitly for an empty file)
        mode: Write mode - "w" to overwrite, "a" to append (default: "w")
        create_dirs: Whether to create parent directories if they don't exist (default: True)

    Returns:
        Success message with file information

    Raises:
        PermissionError: If lacking write permissions
        OSError: If there are filesystem issues
    """
    try:
        # Convert to Path object for better handling and expand home directory shortcuts like ~
        path = Path(file_path).expanduser()
        display_path = _path_for_display(path)

        # Runtime-enforced filesystem ignore policy (.abstractignore + defaults).
        from .abstractignore import AbstractIgnore

        ignore = AbstractIgnore.for_path(path)
        if ignore.is_ignored(path, is_dir=False) or ignore.is_ignored(path.parent, is_dir=True):
            return f"❌ Refused: Path '{display_path}' is ignored by .abstractignore policy"

        # Create parent directories if requested and they don't exist
        if create_dirs and path.parent != path:
            path.parent.mkdir(parents=True, exist_ok=True)

        # Write the content to the file
        with open(path, mode, encoding='utf-8') as f:
            f.write(content)

        # Get file size for confirmation
        file_size = path.stat().st_size
        lines_written = len(str(content).splitlines())
        bytes_written = len(str(content).encode("utf-8"))

        # Enhanced success message with emoji and formatting
        action = "appended to" if mode == "a" else "written to"
        if mode == "a":
            return (
                f"✅ Successfully {action} '{display_path}' "
                f"(+{bytes_written:,} bytes, +{lines_written:,} lines; file now {file_size:,} bytes)"
            )
        return f"✅ Successfully {action} '{display_path}' ({file_size:,} bytes, {lines_written:,} lines)"

    except PermissionError:
        return f"❌ Permission denied: Cannot write to '{_path_for_display(Path(file_path).expanduser())}'"
    except FileNotFoundError:
        return f"❌ Directory not found: Parent directory of '{_path_for_display(Path(file_path).expanduser())}' does not exist"
    except OSError as e:
        return f"❌ File system error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error writing file: {str(e)}"


@tool(
    description="Search the web via DuckDuckGo and return JSON {query, params, results}. num_results defaults to 10.",
    when_to_use="Use to find up-to-date info or references; treat results as untrusted text.",
    examples=[
        {
            "description": "Search for current programming best practices",
            "arguments": {
                "query": "python best practices 2025",
                "num_results": 5
            }
        },
        {
            "description": "Get current news or events",
            "arguments": {
                "query": "AI developments 2025"
            }
        },
        {
            "description": "Find articles from the past week",
            "arguments": {
                "query": "Python programming tutorials",
                "time_range": "w"
            }
        },
    ]
)
def web_search(
    query: str,
    num_results: int = 10,
    safe_search: str = "moderate",
    region: str = "wt-wt",
    time_range: Optional[str] = None,
) -> str:
    """
    Search the internet using DuckDuckGo (no API key required).

    Args:
        query: Search query
        num_results: Number of results to return (default: 10)
        safe_search: Content filtering level - "strict", "moderate", or "off" (default: "moderate")
        region: Regional results preference - "wt-wt" (worldwide), "us-en", "uk-en", "fr-fr", "de-de", etc. (default: "wt-wt")
        time_range: Time range filter for results (optional):
            - "h" or "24h": Past 24 hours
            - "d": Past day
            - "w" or "7d": Past week
            - "m" or "30d": Past month
            - "y" or "1y": Past year
            - None: All time (default)

    Returns:
        JSON string with search results or an error message.

    Note:
        For best results, install `ddgs` (`pip install ddgs`). Without it, this tool falls back to
        parsing DuckDuckGo's HTML results, which may be less stable and may ignore time_range.
    """
    def _json_output(payload: Dict[str, Any]) -> str:
        try:
            return json.dumps(payload, ensure_ascii=False, indent=2)
        except Exception:
            return json.dumps({"error": "Failed to serialize search results", "query": query})

    def _normalize_time_range(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        v = str(value).strip().lower()
        if not v:
            return None
        return {
            "24h": "h",
            "7d": "w",
            "30d": "m",
            "1y": "y",
        }.get(v, v)

    try:
        normalized_time_range = _normalize_time_range(time_range)

        ddgs_error: Optional[str] = None

        # Preferred backend: ddgs (DuckDuckGo text search).
        try:
            from ddgs import DDGS  # type: ignore
        except Exception as e:
            DDGS = None  # type: ignore[assignment]
            ddgs_error = str(e)

        if DDGS is not None:
            try:
                with DDGS() as ddgs:
                    search_params: Dict[str, Any] = {
                        "keywords": query,
                        "max_results": num_results,
                        "region": region,
                        "safesearch": safe_search,
                    }
                    if normalized_time_range:
                        search_params["timelimit"] = normalized_time_range

                    search_results = list(ddgs.text(**search_params))

                return _json_output(
                    {
                        "engine": "duckduckgo",
                        "source": "duckduckgo.text",
                        "query": query,
                        "params": {
                            "num_results": num_results,
                            "safe_search": safe_search,
                            "region": region,
                            "time_range": normalized_time_range,
                            "backend": "ddgs.text",
                        },
                        "results": [
                            {
                                "rank": i,
                                "title": (result.get("title") or "").strip(),
                                "url": (result.get("href") or "").strip(),
                                "snippet": (result.get("body") or "").strip(),
                            }
                            for i, result in enumerate(search_results, 1)
                        ],
                    }
                )
            except Exception as e:
                ddgs_error = str(e)

        # Fallback backend: DuckDuckGo HTML results (best-effort).
        try:
            import html as html_lib

            url = "https://duckduckgo.com/html/"
            params: Dict[str, Any] = {"q": query, "kl": region}
            headers = {"User-Agent": "AbstractCore-WebSearch/1.0", "Accept-Language": region}
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            page = resp.text or ""

            # DuckDuckGo HTML results contain entries like:
            # <a class="result__a" href="...">Title</a>
            # <a class="result__snippet">Snippet</a>
            link_re = re.compile(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
            snippet_re = re.compile(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
            tag_re = re.compile(r"<[^>]+>")

            links = list(link_re.finditer(page))
            results: List[Dict[str, Any]] = []
            for i, m in enumerate(links, 1):
                if i > int(num_results or 0):
                    break
                href = html_lib.unescape((m.group(1) or "").strip())
                title_html = m.group(2) or ""
                title = html_lib.unescape(tag_re.sub("", title_html)).strip()

                # Try to find the snippet in the following chunk of HTML (best-effort).
                tail = page[m.end() : m.end() + 5000]
                sm = snippet_re.search(tail)
                snippet = ""
                if sm:
                    snippet_html = sm.group(1) or ""
                    snippet = html_lib.unescape(tag_re.sub("", snippet_html)).strip()

                results.append({"rank": i, "title": title, "url": href, "snippet": snippet})

            payload: Dict[str, Any] = {
                "engine": "duckduckgo",
                "source": "duckduckgo.text",
                "query": query,
                "params": {
                    "num_results": num_results,
                    "safe_search": safe_search,
                    "region": region,
                    "time_range": normalized_time_range,
                    "backend": "duckduckgo.html",
                },
                "results": results,
            }

            if not results:
                payload["error"] = "No results found from DuckDuckGo HTML endpoint."
                payload["hint"] = "Install `ddgs` for more reliable results."
                if ddgs_error:
                    payload["ddgs_error"] = ddgs_error

            return _json_output(payload)
        except Exception as e:
            payload: Dict[str, Any] = {
                "engine": "duckduckgo",
                "source": "duckduckgo.text",
                "query": query,
                "params": {
                    "num_results": num_results,
                    "safe_search": safe_search,
                    "region": region,
                    "time_range": normalized_time_range,
                },
                "results": [],
                "error": str(e),
                "hint": "Install `ddgs` for more reliable results: pip install ddgs",
            }
            if ddgs_error:
                payload["ddgs_error"] = ddgs_error
            return _json_output(payload)

    except Exception as e:
        return _json_output({
            "engine": "duckduckgo",
            "query": query,
            "error": str(e),
        })


@tool(
    description="Fetch a URL and parse common content types (HTML/JSON/text); supports previews and basic metadata.",
    when_to_use="Use to retrieve and analyze content from a specific URL (web page, API, document).",
    examples=[
        {
            "description": "Fetch and parse HTML webpage",
            "arguments": {
                "url": "https://example.com/article.html"
            }
        },
        {
            "description": "Fetch JSON API response",
            "arguments": {
                "url": "https://api.github.com/repos/python/cpython",
                "headers": {"Accept": "application/json"}
            }
        },
        {
            "description": "Fetch binary content with metadata",
            "arguments": {
                "url": "https://example.com/document.pdf",
                "include_binary_preview": True
            }
        }
    ]
)
def fetch_url(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    data: Optional[Union[Dict[str, Any], str]] = None,
    timeout: int = 30,
    max_content_length: int = 10485760,  # 10MB default
    follow_redirects: bool = True,
    include_binary_preview: bool = False,
    extract_links: bool = False,
    user_agent: str = "AbstractCore-FetchTool/1.0",
    include_full_content: bool = True,
) -> Dict[str, Any]:
    """
    Fetch and intelligently parse content from URLs with comprehensive content type detection.
    
    This tool automatically detects content types (HTML, JSON, XML, images, etc.) and provides
    appropriate parsing with metadata extraction including timestamps and response headers.
    
    Args:
        url: The URL to fetch content from
        method: HTTP method to use (default: "GET")
        headers: Optional custom headers to send with the request
        data: Optional data to send with POST/PUT requests (dict or string)
        timeout: Request timeout in seconds (default: 30)
        max_content_length: Maximum content length to fetch in bytes (default: 10MB)
        follow_redirects: Whether to follow HTTP redirects (default: True)
        include_binary_preview: Whether to include base64 preview for binary content (default: False)
        extract_links: Whether to extract links from HTML content (default: False)
        user_agent: User-Agent header to use (default: "AbstractCore-FetchTool/1.0")
        include_full_content: Whether to include full text/JSON/XML content (no preview truncation) (default: True)
    
    Returns:
        Formatted string with parsed content, metadata, and analysis or error message
        
    Examples:
        fetch_url("https://api.github.com/repos/python/cpython")  # Fetch and parse JSON API
        fetch_url("https://example.com", headers={"Accept": "text/html"})  # Fetch HTML with custom headers
        fetch_url("https://httpbin.org/post", method="POST", data={"test": "value"})  # POST request
        fetch_url("https://example.com/image.jpg", include_binary_preview=True)  # Fetch image with preview
    """
    try:
        # Validate URL
        parsed_url = urlparse(url)
        if not parsed_url.scheme or not parsed_url.netloc:
            rendered = f"❌ Invalid URL format: {url}"
            return {"success": False, "error": rendered.lstrip("❌").strip(), "url": url, "rendered": rendered}
        
        if parsed_url.scheme not in ['http', 'https']:
            rendered = f"❌ Unsupported URL scheme: {parsed_url.scheme}. Only HTTP and HTTPS are supported."
            return {
                "success": False,
                "error": rendered.lstrip("❌").strip(),
                "url": url,
                "scheme": str(parsed_url.scheme),
                "rendered": rendered,
            }
        
        # Prepare request headers
        request_headers = {
            'User-Agent': user_agent,
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        if headers:
            request_headers.update(headers)
        
        # Add data for POST/PUT requests
        if data and method.upper() in ['POST', 'PUT', 'PATCH']:
            if isinstance(data, dict):
                # Try JSON first, fallback to form data
                if request_headers.get('Content-Type', '').startswith('application/json'):
                    request_json = data
                    request_data = None
                else:
                    request_json = None
                    request_data = data
            else:
                request_json = None
                request_data = data
        else:
            request_json = None
            request_data = None
        
        # Record fetch timestamp
        fetch_timestamp = datetime.now().isoformat()

        def _decode_text_bytes(content: bytes, content_type_header: str) -> str:
            """Best-effort decode of text-based HTTP responses."""
            encoding = "utf-8"
            if "charset=" in (content_type_header or ""):
                try:
                    encoding = str(content_type_header).split("charset=")[1].split(";")[0].strip() or "utf-8"
                except Exception:
                    encoding = "utf-8"

            for enc in [encoding, "utf-8", "iso-8859-1", "windows-1252"]:
                try:
                    return content.decode(enc)
                except (UnicodeDecodeError, LookupError):
                    continue
            return content.decode("utf-8", errors="replace")

        def _normalize_text_for_evidence(*, raw_text: str, content_type_header: str, url: str) -> str:
            """Extract a readable text representation for evidence storage."""
            text = str(raw_text or "")
            if not text.strip():
                return ""

            main_type = str(content_type_header or "").split(";")[0].strip().lower()
            try:
                if main_type.startswith(("text/html", "application/xhtml+xml", "application/xhtml")):
                    # HTML: strip tags and normalize whitespace.
                    parser = _get_appropriate_parser(text)
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
                        soup = BeautifulSoup(text, parser)
                    return _normalize_text(soup.get_text(" ", strip=True))

                if main_type == "application/json":
                    data = json.loads(text)
                    return json.dumps(data, ensure_ascii=False, indent=2, separators=(",", ": "))
            except Exception:
                # Fall back to raw text on parse failures.
                pass

            return text
        
        # Make the request with session for connection reuse and keep it open while streaming
        with requests.Session() as session:
            session.headers.update(request_headers)
            with session.request(
                method=method.upper(),
                url=url,
                timeout=timeout,
                allow_redirects=follow_redirects,
                stream=True,
                json=request_json,
                data=request_data,
            ) as response:

                # Check response status
                if not response.ok:
                    rendered = (
                        f"❌ HTTP Error {response.status_code}: {response.reason}\n"
                        f"URL: {url}\n"
                        f"Timestamp: {fetch_timestamp}\n"
                        f"Response headers: {dict(response.headers)}"
                    )
                    return {
                        "success": False,
                        "error": f"HTTP Error {int(response.status_code)}: {str(response.reason)}",
                        "url": url,
                        "timestamp": fetch_timestamp,
                        "status_code": int(response.status_code),
                        "reason": str(response.reason),
                        "content_type": str(response.headers.get("content-type", "") or ""),
                        "rendered": rendered,
                    }

                # Get content info
                content_type = response.headers.get('content-type', '').lower()
                content_length = response.headers.get('content-length')
                if content_length:
                    content_length = int(content_length)

                # Check content length before downloading
                if content_length and content_length > max_content_length:
                    rendered = (
                        f"⚠️  Content too large: {content_length:,} bytes (max: {max_content_length:,})\n"
                        f"URL: {url}\n"
                        f"Content-Type: {content_type}\n"
                        f"Timestamp: {fetch_timestamp}\n"
                        "Use max_content_length parameter to increase limit if needed"
                    )
                    return {
                        "success": False,
                        "error": "Content too large",
                        "url": url,
                        "timestamp": fetch_timestamp,
                        "content_type": str(content_type or ""),
                        "content_length": int(content_length),
                        "max_content_length": int(max_content_length),
                        "rendered": rendered,
                    }

                # Download content with optimized chunking
                content_chunks = []
                downloaded_size = 0

                # Use larger chunks for better performance
                chunk_size = 32768 if 'image/' in content_type or 'video/' in content_type else 16384

                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        downloaded_size += len(chunk)
                        if downloaded_size > max_content_length:
                            rendered = (
                                f"⚠️  Content exceeded size limit during download: {downloaded_size:,} bytes (max: {max_content_length:,})\n"
                                f"URL: {url}\n"
                                f"Content-Type: {content_type}\n"
                                f"Timestamp: {fetch_timestamp}"
                            )
                            return {
                                "success": False,
                                "error": "Content exceeded size limit during download",
                                "url": url,
                                "timestamp": fetch_timestamp,
                                "content_type": str(content_type or ""),
                                "downloaded_size": int(downloaded_size),
                                "max_content_length": int(max_content_length),
                                "rendered": rendered,
                            }
                        content_chunks.append(chunk)

                content_bytes = b''.join(content_chunks)
                actual_size = len(content_bytes)

                # Detect content type and parse accordingly
                parsed_content = _parse_content_by_type(
                    content_bytes,
                    content_type,
                    url,
                    extract_links=extract_links,
                    include_binary_preview=include_binary_preview,
                    include_full_content=include_full_content,
                )

                # Build comprehensive response
                result_parts = []
                result_parts.append(f"🌐 URL Fetch Results")
                result_parts.append(f"📍 URL: {response.url}")  # Final URL after redirects
                if response.url != url:
                    result_parts.append(f"🔄 Original URL: {url}")
                result_parts.append(f"⏰ Timestamp: {fetch_timestamp}")
                result_parts.append(f"✅ Status: {response.status_code} {response.reason}")
                result_parts.append(f"📊 Content-Type: {content_type}")
                result_parts.append(f"📏 Size: {actual_size:,} bytes")

                # Add important response headers
                important_headers = ['server', 'last-modified', 'etag', 'cache-control', 'expires', 'location']
                response_metadata = []
                for header in important_headers:
                    value = response.headers.get(header)
                    if value:
                        response_metadata.append(f"  {header.title()}: {value}")

                if response_metadata:
                    result_parts.append(f"📋 Response Headers:")
                    result_parts.extend(response_metadata)

                # Add parsed content
                result_parts.append(f"\n📄 Content Analysis:")
                result_parts.append(parsed_content)

                rendered = "\n".join(result_parts)

                raw_text: Optional[str] = None
                normalized_text: Optional[str] = None
                try:
                    main_type = str(content_type or "").split(";")[0].strip().lower()
                    text_based_types = [
                        "text/",
                        "application/json",
                        "application/xml",
                        "application/javascript",
                        "application/rss+xml",
                        "application/atom+xml",
                        "application/xhtml+xml",
                    ]
                    is_text_based = any(main_type.startswith(t) for t in text_based_types)
                    if is_text_based:
                        raw_text = _decode_text_bytes(content_bytes, content_type)
                        normalized_text = _normalize_text_for_evidence(raw_text=raw_text, content_type_header=content_type, url=url)
                except Exception:
                    raw_text = None
                    normalized_text = None

                return {
                    "success": True,
                    "error": None,
                    "url": str(url),
                    "final_url": str(response.url),
                    "timestamp": str(fetch_timestamp),
                    "status_code": int(response.status_code),
                    "reason": str(response.reason),
                    "content_type": str(content_type or ""),
                    "size_bytes": int(actual_size),
                    # Evidence-only fields (large). Higher layers should persist these as artifacts and drop them from
                    # tool outputs to keep run state/prompt size bounded.
                    "raw_text": raw_text,
                    "normalized_text": normalized_text,
                    # LLM-visible / UI-friendly rendering.
                    "rendered": rendered,
                }
        
    except requests.exceptions.Timeout:
        rendered = (
            f"⏰ Request timeout after {timeout} seconds\n"
            f"URL: {url}\n"
            "Consider increasing timeout parameter"
        )
        return {
            "success": False,
            "error": f"Request timeout after {int(timeout)} seconds",
            "url": str(url),
            "timeout_s": int(timeout),
            "rendered": rendered,
        }
    
    except requests.exceptions.ConnectionError as e:
        rendered = (
            f"🔌 Connection error: {str(e)}\n"
            f"URL: {url}\n"
            "Check network connectivity and URL validity"
        )
        return {
            "success": False,
            "error": f"Connection error: {str(e)}",
            "url": str(url),
            "rendered": rendered,
        }
    
    except requests.exceptions.TooManyRedirects:
        rendered = (
            "🔄 Too many redirects\n"
            f"URL: {url}\n"
            "Try setting follow_redirects=False to see redirect chain"
        )
        return {
            "success": False,
            "error": "Too many redirects",
            "url": str(url),
            "rendered": rendered,
        }
    
    except requests.exceptions.RequestException as e:
        rendered = f"❌ Request error: {str(e)}\nURL: {url}"
        return {"success": False, "error": str(e), "url": str(url), "rendered": rendered}
    
    except Exception as e:
        rendered = f"❌ Unexpected error fetching URL: {str(e)}\nURL: {url}"
        return {"success": False, "error": str(e), "url": str(url), "rendered": rendered}


def _parse_content_by_type(
    content_bytes: bytes,
    content_type: str,
    url: str,
    extract_links: bool = True,
    include_binary_preview: bool = False,
    include_full_content: bool = False,
) -> str:
    """
    Parse content based on detected content type with intelligent fallbacks.
    
    This function provides robust content type detection and parsing for various formats
    including HTML, JSON, XML, plain text, images, and other binary formats.
    """
    try:
        # Normalize content type
        main_type = content_type.split(';')[0].strip().lower()
        
        # Try to decode as text first for text-based formats
        text_content = None
        encoding = 'utf-8'
        
        # Detect encoding from content-type header
        if 'charset=' in content_type:
            try:
                encoding = content_type.split('charset=')[1].split(';')[0].strip()
            except:
                encoding = 'utf-8'
        
        # Attempt text decoding for text-based content types with better encoding detection
        text_based_types = [
            'text/', 'application/json', 'application/xml', 'application/javascript',
            'application/rss+xml', 'application/atom+xml', 'application/xhtml+xml'
        ]
        
        is_text_based = any(main_type.startswith(t) for t in text_based_types)
        
        if is_text_based:
            # Try multiple encoding strategies
            for enc in [encoding, 'utf-8', 'iso-8859-1', 'windows-1252']:
                try:
                    text_content = content_bytes.decode(enc)
                    break
                except (UnicodeDecodeError, LookupError):
                    continue
            else:
                # Final fallback with error replacement
                text_content = content_bytes.decode('utf-8', errors='replace')
        
        # Parse based on content type with fallback content detection
        if main_type.startswith('text/html') or main_type.startswith('application/xhtml'):
            return _parse_html_content(text_content, url, extract_links, include_full_content)
        
        elif main_type == 'application/json':
            return _parse_json_content(text_content, include_full_content)
        
        elif main_type in ['application/xml', 'text/xml', 'application/rss+xml', 'application/atom+xml', 'application/soap+xml']:
            return _parse_xml_content(text_content, include_full_content)
        
        elif main_type.startswith('text/'):
            # For generic text types, check if it's actually XML or JSON
            if text_content and text_content.strip():
                if _is_xml_content(text_content):
                    return _parse_xml_content(text_content, include_full_content)
                elif _is_json_content(text_content):
                    return _parse_json_content(text_content, include_full_content)
            return _parse_text_content(text_content, main_type, include_full_content)
        
        elif main_type.startswith('image/'):
            return _parse_image_content(content_bytes, main_type, include_binary_preview)
        
        elif main_type == 'application/pdf':
            return _parse_pdf_content(content_bytes, include_binary_preview)
        
        else:
            return _parse_binary_content(content_bytes, main_type, include_binary_preview)
    
    except Exception as e:
        return f"❌ Error parsing content: {str(e)}\n" \
               f"Content-Type: {content_type}\n" \
               f"Content size: {len(content_bytes):,} bytes"


def _is_xml_content(content: str) -> bool:
    """Detect if content is XML rather than HTML."""
    if not content:
        return False
    
    content_lower = content.lower().strip()
    
    # Check for XML declaration
    if content_lower.startswith('<?xml'):
        return True
    
    # Check for common XML root elements without HTML indicators
    xml_indicators = ['<rss', '<feed', '<urlset', '<sitemap', '<soap:', '<xml']
    html_indicators = ['<!doctype html', '<html', '<head>', '<body>', '<div', '<span', '<p>', '<a ']
    
    # Look at the first 1000 characters for indicators
    sample = content_lower[:1000]
    
    # If we find HTML indicators, it's likely HTML
    if any(indicator in sample for indicator in html_indicators):
        return False
    
    # If we find XML indicators without HTML indicators, it's likely XML
    if any(indicator in sample for indicator in xml_indicators):
        return True
    
    # Check if it starts with a root element that looks like XML
    import re
    root_match = re.search(r'<([^?\s/>]+)', content)
    if root_match:
        root_element = root_match.group(1).lower()
        # Common XML root elements that are not HTML
        xml_roots = ['rss', 'feed', 'urlset', 'sitemap', 'configuration', 'data', 'response']
        if root_element in xml_roots:
            return True
    
    return False


def _is_json_content(content: str) -> bool:
    """Detect if content is JSON."""
    if not content:
        return False
    
    content_stripped = content.strip()
    
    # Quick check for JSON structure
    if (content_stripped.startswith('{') and content_stripped.endswith('}')) or \
       (content_stripped.startswith('[') and content_stripped.endswith(']')):
        try:
            import json
            json.loads(content_stripped)
            return True
        except (json.JSONDecodeError, ValueError):
            pass
    
    return False


def _get_appropriate_parser(content: str) -> str:
    """Get the appropriate BeautifulSoup parser for the content."""
    # If lxml is available and content looks like XML, use xml parser
    if BS4_PARSER == "lxml" and _is_xml_content(content):
        return "xml"
    
    # Default to the configured parser (lxml or html.parser)
    return BS4_PARSER


def _parse_html_content(html_content: str, url: str, extract_links: bool = True, include_full_content: bool = False) -> str:
    """Parse HTML content and extract meaningful information."""
    if not html_content:
        return "❌ No HTML content to parse"
    
    # Detect if content is actually XML (fallback detection)
    if _is_xml_content(html_content):
        return _parse_xml_content(html_content, include_full_content)
    
    result_parts = []
    result_parts.append("🌐 HTML Document Analysis")
    
    try:
        # Choose appropriate parser based on content analysis
        parser = _get_appropriate_parser(html_content)

        # Suppress XML parsing warnings when using HTML parser on XML content
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
            soup = BeautifulSoup(html_content, parser)

        # Extract title
        title = soup.find("title")
        if title:
            result_parts.append(f"📰 Title: {title.get_text().strip()}")

        # Extract meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            desc = meta_desc["content"].strip()
            if not include_full_content and len(desc) > 200:
                desc = desc[:200] + "..."
            result_parts.append(f"📝 Description: {desc}")

        # Extract headings
        headings = []
        for i in range(1, 7):
            h_tags = soup.find_all(f"h{i}")
            for h in h_tags[:5]:  # Limit to first 5 of each level
                headings.append(f"H{i}: {h.get_text().strip()[:100]}")

        if headings:
            result_parts.append("📋 Headings (first 5 per level):")
            for heading in headings[:10]:  # Limit total headings
                result_parts.append(f"  • {heading}")

        # Extract links if requested
        if extract_links:
            links = []
            for a in soup.find_all("a", href=True)[:20]:  # Limit to first 20 links
                href = a["href"]
                text = a.get_text().strip()[:50]
                # Convert relative URLs to absolute
                if href.startswith("/"):
                    href = urljoin(url, href)
                elif not href.startswith(("http://", "https://")):
                    href = urljoin(url, href)
                links.append(f"{text} → {href}")

            if links:
                result_parts.append("🔗 Links (first 20):")
                for link in links:
                    result_parts.append(f"  • {link}")

        # Extract main text content with better cleaning
        # Remove script, style, nav, footer, header elements for cleaner content
        for element in soup(
            ["script", "style", "nav", "footer", "header", "aside", "noscript", "svg"]
        ):
            element.decompose()

        def _normalize_text(raw_text: str) -> str:
            return " ".join(str(raw_text or "").split())

        # Pick the most content-dense container (helps avoid menus/boilerplate).
        content_candidates = []
        content_selectors = [
            "main",
            "article",
            "[role='main']",
            "#mw-content-text",
            "#bodyContent",
            "#content",
            "#main",
            ".mw-parser-output",
            ".entry-content",
            ".post-content",
            ".article-content",
            ".page-content",
            ".content",
        ]
        try:
            selector_query = ", ".join(content_selectors)
            content_candidates.extend(soup.select(selector_query)[:25])
        except Exception:
            pass
        if soup.body:
            content_candidates.append(soup.body)
        content_candidates.append(soup)

        content_soup = None
        best_text_len = -1
        for candidate in content_candidates:
            candidate_text = _normalize_text(candidate.get_text(" ", strip=True))
            if len(candidate_text) > best_text_len:
                best_text_len = len(candidate_text)
                content_soup = candidate

        text = _normalize_text((content_soup or soup).get_text(" ", strip=True))

        if text:
            preview_length = None if include_full_content else 1000
            text_preview = text if preview_length is None else text[:preview_length]
            if preview_length is not None and len(text) > preview_length:
                text_preview += "..."
            result_parts.append("📄 Text Content:" if include_full_content else "📄 Text Content Preview:")
            result_parts.append(f"{text_preview}")
            result_parts.append(f"📊 Total text length: {len(text):,} characters")

    except Exception as e:
        result_parts.append(f"⚠️  BeautifulSoup parsing error: {str(e)}")
        result_parts.append("📄 Raw HTML Preview (first 1000 chars):")
        if include_full_content:
            result_parts.append(html_content)
        else:
            result_parts.append(html_content[:1000] + ("..." if len(html_content) > 1000 else ""))
    
    return "\n".join(result_parts)


def _parse_json_content(json_content: str, include_full_content: bool = False) -> str:
    """Parse JSON content and provide structured analysis."""
    if not json_content:
        return "❌ No JSON content to parse"
    
    result_parts = []
    result_parts.append("📊 JSON Data Analysis")
    
    try:
        data = json.loads(json_content)
        
        # Analyze JSON structure
        result_parts.append(f"📋 Structure: {type(data).__name__}")
        
        if isinstance(data, dict):
            result_parts.append(f"🔑 Keys ({len(data)}): {', '.join(list(data.keys())[:10])}")
            if len(data) > 10:
                result_parts.append(f"   ... and {len(data) - 10} more keys")
        elif isinstance(data, list):
            result_parts.append(f"📝 Array length: {len(data)}")
            if data and isinstance(data[0], dict):
                result_parts.append(f"🔑 First item keys: {', '.join(list(data[0].keys())[:10])}")
        
        # Pretty print JSON with smart truncation
        json_str = json.dumps(data, indent=2, ensure_ascii=False, separators=(',', ': '))
        preview_length = None if include_full_content else 1500  # Reduced for better readability
        if preview_length is not None and len(json_str) > preview_length:
            # Try to truncate at a logical point (end of object/array)
            truncate_pos = json_str.rfind('\n', 0, preview_length)
            if truncate_pos > preview_length - 200:  # If close to limit, use it
                json_preview = json_str[:truncate_pos] + "\n... (truncated)"
            else:
                json_preview = json_str[:preview_length] + "\n... (truncated)"
        else:
            json_preview = json_str
        
        result_parts.append(f"📄 JSON Content:")
        result_parts.append(json_preview)
        result_parts.append(f"📊 Total size: {len(json_content):,} characters")
    
    except json.JSONDecodeError as e:
        result_parts.append(f"❌ JSON parsing error: {str(e)}")
        result_parts.append(f"📄 Raw content preview (first 1000 chars):")
        if include_full_content:
            result_parts.append(json_content)
        else:
            result_parts.append(json_content[:1000] + ("..." if len(json_content) > 1000 else ""))
    
    return "\n".join(result_parts)


def _parse_xml_content(xml_content: str, include_full_content: bool = False) -> str:
    """Parse XML content including RSS/Atom feeds."""
    if not xml_content:
        return "❌ No XML content to parse"
    
    result_parts = []
    result_parts.append("📄 XML/RSS/Atom Analysis")
    
    try:
        # Try to detect if it's RSS/Atom
        if '<rss' in xml_content.lower() or '<feed' in xml_content.lower():
            result_parts.append("📡 Detected: RSS/Atom Feed")
        
        # Basic XML structure analysis
        import re
        
        # Find root element
        root_match = re.search(r'<([^?\s/>]+)', xml_content)
        if root_match:
            result_parts.append(f"🏷️  Root element: <{root_match.group(1)}>")
        
        # Count elements (basic)
        elements = re.findall(r'<([^/\s>]+)', xml_content)
        if elements:
            from collections import Counter
            element_counts = Counter(elements[:50])  # Limit analysis
            result_parts.append(f"📊 Top elements: {dict(list(element_counts.most_common(10)))}")
        
        # Show XML preview
        preview_length = None if include_full_content else 1500
        xml_preview = xml_content if preview_length is None else xml_content[:preview_length]
        if preview_length is not None and len(xml_content) > preview_length:
            xml_preview += "\n... (truncated)"
        
        result_parts.append("📄 XML Content:" if include_full_content else "📄 XML Content Preview:")
        result_parts.append(xml_preview)
        result_parts.append(f"📊 Total size: {len(xml_content):,} characters")
    
    except Exception as e:
        result_parts.append(f"❌ XML parsing error: {str(e)}")
        result_parts.append(f"📄 Raw content preview (first 1000 chars):")
        if include_full_content:
            result_parts.append(xml_content)
        else:
            result_parts.append(xml_content[:1000] + ("..." if len(xml_content) > 1000 else ""))
    
    return "\n".join(result_parts)


def _parse_text_content(text_content: str, content_type: str, include_full_content: bool = False) -> str:
    """Parse plain text content."""
    if not text_content:
        return "❌ No text content to parse"
    
    result_parts = []
    result_parts.append(f"📝 Text Content Analysis ({content_type})")
    
    # Basic text statistics
    lines = text_content.splitlines()
    words = text_content.split()
    
    result_parts.append(f"📊 Statistics:")
    result_parts.append(f"  • Lines: {len(lines):,}")
    result_parts.append(f"  • Words: {len(words):,}")
    result_parts.append(f"  • Characters: {len(text_content):,}")
    
    # Show text preview
    preview_length = None if include_full_content else 2000
    text_preview = text_content if preview_length is None else text_content[:preview_length]
    if preview_length is not None and len(text_content) > preview_length:
        text_preview += "\n... (truncated)"
    
    result_parts.append("📄 Content:" if include_full_content else "📄 Content Preview:")
    result_parts.append(text_preview)
    
    return "\n".join(result_parts)


def _parse_image_content(image_bytes: bytes, content_type: str, include_preview: bool = False) -> str:
    """Parse image content and extract metadata."""
    result_parts = []
    result_parts.append(f"🖼️  Image Analysis ({content_type})")
    
    result_parts.append(f"📊 Size: {len(image_bytes):,} bytes")
    
    # Try to get image dimensions (basic approach)
    try:
        if content_type.startswith('image/jpeg') or content_type.startswith('image/jpg'):
            # Basic JPEG header parsing for dimensions
            if image_bytes.startswith(b'\xff\xd8\xff'):
                result_parts.append("✅ Valid JPEG format detected")
        elif content_type.startswith('image/png'):
            # Basic PNG header parsing
            if image_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
                result_parts.append("✅ Valid PNG format detected")
        elif content_type.startswith('image/gif'):
            if image_bytes.startswith(b'GIF87a') or image_bytes.startswith(b'GIF89a'):
                result_parts.append("✅ Valid GIF format detected")
    except Exception:
        pass
    
    if include_preview:
        # Provide base64 preview for small images
        if len(image_bytes) <= 1048576:  # 1MB limit for preview
            b64_preview = base64.b64encode(image_bytes[:1024]).decode('ascii')  # First 1KB
            result_parts.append(f"🔍 Base64 Preview (first 1KB):")
            result_parts.append(f"{b64_preview}...")
        else:
            result_parts.append("⚠️  Image too large for base64 preview")
    
    result_parts.append("💡 Use image processing tools for detailed analysis")
    
    return "\n".join(result_parts)


def _parse_pdf_content(pdf_bytes: bytes, include_preview: bool = False) -> str:
    """Parse PDF content and extract basic metadata."""
    result_parts = []
    result_parts.append("📄 PDF Document Analysis")
    
    result_parts.append(f"📊 Size: {len(pdf_bytes):,} bytes")
    
    # Check PDF header
    if pdf_bytes.startswith(b'%PDF-'):
        try:
            version_line = pdf_bytes[:20].decode('ascii', errors='ignore')
            result_parts.append(f"✅ Valid PDF format: {version_line.strip()}")
        except:
            result_parts.append("✅ Valid PDF format detected")
    else:
        result_parts.append("⚠️  Invalid PDF format - missing PDF header")
    
    if include_preview:
        # Show hex preview of first few bytes
        hex_preview = ' '.join(f'{b:02x}' for b in pdf_bytes[:64])
        result_parts.append(f"🔍 Hex Preview (first 64 bytes):")
        result_parts.append(hex_preview)
    
    result_parts.append("💡 Use PDF processing tools for text extraction and detailed analysis")
    
    return "\n".join(result_parts)


def _parse_binary_content(binary_bytes: bytes, content_type: str, include_preview: bool = False) -> str:
    """Parse generic binary content."""
    result_parts = []
    result_parts.append(f"📦 Binary Content Analysis ({content_type})")
    
    result_parts.append(f"📊 Size: {len(binary_bytes):,} bytes")
    
    # Detect file type by magic bytes
    magic_signatures = {
        b'\x50\x4b\x03\x04': 'ZIP archive',
        b'\x50\x4b\x05\x06': 'ZIP archive (empty)',
        b'\x50\x4b\x07\x08': 'ZIP archive (spanned)',
        b'\x1f\x8b\x08': 'GZIP compressed',
        b'\x42\x5a\x68': 'BZIP2 compressed',
        b'\x37\x7a\xbc\xaf\x27\x1c': '7-Zip archive',
        b'\x52\x61\x72\x21\x1a\x07': 'RAR archive',
        b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a': 'PNG image',
        b'\xff\xd8\xff': 'JPEG image',
        b'\x47\x49\x46\x38': 'GIF image',
        b'\x25\x50\x44\x46': 'PDF document',
        b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1': 'Microsoft Office document',
        b'\x4d\x5a': 'Windows executable'
    }
    
    detected_type = None
    for signature, file_type in magic_signatures.items():
        if binary_bytes.startswith(signature):
            detected_type = file_type
            break
    
    if detected_type:
        result_parts.append(f"🔍 Detected format: {detected_type}")
    
    if include_preview:
        # Show hex preview
        hex_preview = ' '.join(f'{b:02x}' for b in binary_bytes[:64])
        result_parts.append(f"🔍 Hex Preview (first 64 bytes):")
        result_parts.append(hex_preview)
        
        # Try to show any readable ASCII strings
        try:
            ascii_preview = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in binary_bytes[:200])
            if ascii_preview.strip():
                result_parts.append(f"📝 ASCII Preview (first 200 bytes):")
                result_parts.append(ascii_preview)
        except:
            pass
    
    result_parts.append("💡 Use specialized tools for detailed binary analysis")
    
    return "\n".join(result_parts)


def _normalize_escape_sequences(text: str) -> str:
    """Convert literal escape sequences to actual control characters.

    Handles cases where LLMs send '\\n' (literal) instead of actual newlines.
    This is a common issue when LLM output is over-escaped in JSON.

    Args:
        text: Input string potentially containing literal escape sequences

    Returns:
        String with \\n, \\t, \\r converted to actual control characters
    """
    # Only convert if there are literal escape sequences
    if '\\n' in text or '\\t' in text or '\\r' in text:
        text = text.replace('\\n', '\n')
        text = text.replace('\\t', '\t')
        text = text.replace('\\r', '\r')
    return text


def _extract_pattern_tokens_for_diagnostics(pattern: str, *, max_tokens: int = 6) -> list[str]:
    """Extract human-meaningful tokens from a pattern for no-match diagnostics.

    This is intentionally heuristic and safe:
    - Only used to *suggest* likely locations (never to apply edits).
    - Prefers longer identifiers to reduce noise.
    """
    raw = str(pattern or "")
    if not raw:
        return []

    # Extract identifier-like tokens (e.g. pygame, draw, polygon, MyClass, render_foo).
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", raw)
    if not tokens:
        return []

    stop = {
        "self",
        "this",
        "true",
        "false",
        "null",
        "none",
        "return",
        "class",
        "def",
        "import",
        "from",
    }

    seen: set[str] = set()
    ordered: list[str] = []
    for t in tokens:
        tl = t.lower()
        if tl in stop:
            continue
        if tl in seen:
            continue
        seen.add(tl)
        ordered.append(t)

    if not ordered:
        return []

    ranked = sorted(enumerate(ordered), key=lambda pair: (-len(pair[1]), pair[0]))
    return [t for _, t in ranked[: max(1, int(max_tokens or 6))]]


def _pick_search_anchor_for_diagnostics(pattern: str) -> str:
    """Pick a concise anchor string for search_files() suggestions."""
    raw = str(pattern or "").strip()
    if not raw:
        return ""
    # Prefer dotted identifiers if present (common in Python/JS), else fall back to a token.
    dotted = re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+", raw)
    if dotted:
        return max(dotted, key=len)
    tokens = _extract_pattern_tokens_for_diagnostics(raw, max_tokens=1)
    if tokens:
        return tokens[0]
    return raw[:40]


def _find_candidate_lines_for_diagnostics(
    *,
    content: str,
    tokens: list[str],
    max_results: int = 5,
) -> list[tuple[int, str, int]]:
    if not content or not tokens:
        return []
    lines = content.splitlines()

    tokens_l = [t.lower() for t in tokens if isinstance(t, str) and t]
    if not tokens_l:
        return []

    scored: list[tuple[int, str, int]] = []
    for idx, line in enumerate(lines, 1):
        line_l = line.lower()
        score = 0
        for tok in tokens_l:
            if tok in line_l:
                score += 1
        if score <= 0:
            continue
        scored.append((idx, line, score))

    if not scored:
        return []

    scored.sort(key=lambda item: (-item[2], item[0]))
    return scored[: max(1, int(max_results or 5))]


def _format_edit_file_no_match_diagnostics(*, content: str, pattern: str, file_path: str) -> str:
    """Format compact diagnostics appended to edit_file no-match errors."""
    tokens = _extract_pattern_tokens_for_diagnostics(pattern)
    if not tokens:
        return ""

    candidates = _find_candidate_lines_for_diagnostics(content=content, tokens=tokens, max_results=5)
    if not candidates:
        return ""

    anchor = _pick_search_anchor_for_diagnostics(pattern)
    token_list = ", ".join(tokens[:3])

    def _truncate(line: str, limit: int = 200) -> str:
        s = "" if line is None else str(line)
        s = s.replace("\t", "    ")
        if len(s) <= limit:
            return s
        return s[: max(0, limit - 1)] + "…"

    out: list[str] = []
    if anchor:
        out.append(f"Tip: Use search_files(pattern=\"{anchor}\", path=\"{file_path}\") to locate the exact line(s).")
    out.append(f"Closest lines (token match: {token_list}):")
    for ln, text, _score in candidates:
        out.append(f"  {ln}: {_truncate(text)}")

    return "\n" + "\n".join(out)


def _flexible_whitespace_match(
    pattern: str,
    replacement: str,
    content: str,
    max_replacements: int
) -> Optional[tuple]:
    """
    Match pattern with flexible leading whitespace handling.

    Converts a multi-line pattern into a regex that:
    1. Normalizes line endings (\r\n -> \n)
    2. Matches any amount of leading whitespace on each line
    3. Preserves the non-whitespace content exactly

    Returns (updated_content, count) if matches found, None otherwise.
    """
    # Normalize line endings in both pattern and content
    pattern_normalized = pattern.replace('\r\n', '\n')
    content_normalized = content.replace('\r\n', '\n')

    # Split pattern into lines
    pattern_lines = pattern_normalized.split('\n')

    # Build regex parts for each line
    regex_parts = []
    for i, line in enumerate(pattern_lines):
        # Get leading whitespace and content
        stripped = line.lstrip()
        if stripped:
            # Escape special regex characters in the content
            escaped_content = re.escape(stripped)
            # Match any leading whitespace (spaces or tabs)
            regex_parts.append(r'[ \t]*' + escaped_content)
        else:
            # Empty line or whitespace-only - match any whitespace
            regex_parts.append(r'[ \t]*')

    # Join with flexible newline matching (handles \n or \r\n)
    flexible_pattern = r'\r?\n'.join(regex_parts)

    try:
        regex = re.compile(flexible_pattern, re.MULTILINE)
    except re.error:
        return None

    matches = list(regex.finditer(content_normalized))
    if not matches:
        return None

    # Apply replacements
    # For the replacement, we need to adjust indentation to match
    # the actual indentation found in the match

    def replacement_fn(match):
        """Adjust replacement to use the indentation from the matched text."""
        matched_text = match.group(0)
        matched_lines = matched_text.split('\n')

        # Normalize the replacement's line endings
        repl_normalized = replacement.replace('\r\n', '\n')
        repl_lines = repl_normalized.split('\n')

        if not repl_lines:
            return replacement

        # For each line in the replacement, use the corresponding matched line's
        # actual indentation. This preserves the file's indentation style exactly.
        adjusted_lines = []
        for j, repl_line in enumerate(repl_lines):
            repl_stripped = repl_line.lstrip()

            if j < len(matched_lines):
                # We have a corresponding matched line - use its actual indentation
                matched_line = matched_lines[j]
                actual_indent_str = matched_line[:len(matched_line) - len(matched_line.lstrip())]
                adjusted_lines.append(actual_indent_str + repl_stripped)
            else:
                # Extra lines in replacement - no matched counterpart
                # Use the indentation from the last matched line as reference
                if matched_lines:
                    last_matched = matched_lines[-1]
                    base_indent_str = last_matched[:len(last_matched) - len(last_matched.lstrip())]
                    # Add relative indentation from replacement
                    repl_indent_len = len(repl_line) - len(repl_stripped)
                    pattern_last_indent = len(pattern_lines[-1]) - len(pattern_lines[-1].lstrip()) if pattern_lines else 0
                    extra_spaces = max(0, repl_indent_len - pattern_last_indent)
                    adjusted_lines.append(base_indent_str + ' ' * extra_spaces + repl_stripped)
                else:
                    adjusted_lines.append(repl_line)

        return '\n'.join(adjusted_lines)

    # Apply the replacement
    if max_replacements == -1:
        updated = regex.sub(replacement_fn, content_normalized)
        count = len(matches)
    else:
        updated = regex.sub(replacement_fn, content_normalized, count=max_replacements)
        count = min(len(matches), max_replacements)

    # Restore original line endings if needed
    if '\r\n' in content and '\r\n' not in updated:
        updated = updated.replace('\n', '\r\n')

    return (updated, count)


_HUNK_HEADER_RE = re.compile(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@")


def _normalize_diff_path(raw: str) -> str:
    raw = raw.strip()
    raw = raw.split("\t", 1)[0].strip()
    raw = raw.split(" ", 1)[0].strip()
    if raw.startswith("a/") or raw.startswith("b/"):
        raw = raw[2:]
    return raw


def _path_parts(path_str: str) -> tuple[str, ...]:
    normalized = path_str.replace("\\", "/")
    parts = [p for p in normalized.split("/") if p and p != "."]
    return tuple(parts)


def _is_suffix_path(candidate: str, target: Path) -> bool:
    candidate_parts = _path_parts(candidate)
    if not candidate_parts:
        return False
    target_parts = tuple(target.as_posix().split("/"))
    return len(candidate_parts) <= len(target_parts) and target_parts[-len(candidate_parts) :] == candidate_parts


def _parse_unified_diff(patch: str) -> tuple[Optional[str], list[tuple[int, int, int, int, list[str]]], Optional[str]]:
    """Parse a unified diff for a single file."""
    lines = patch.splitlines()
    header_path: Optional[str] = None
    hunks: list[tuple[int, int, int, int, list[str]]] = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("--- "):
            old_path = _normalize_diff_path(line[4:])
            i += 1
            if i >= len(lines) or not lines[i].startswith("+++ "):
                return None, [], "Invalid unified diff: missing '+++ ' header after '--- '"
            new_path = _normalize_diff_path(lines[i][4:])
            if old_path != "/dev/null" and new_path != "/dev/null":
                if header_path is None:
                    header_path = new_path
                elif header_path != new_path:
                    return None, [], "Unified diff appears to reference multiple files"
            i += 1
            continue

        if line.startswith("@@"):
            m = _HUNK_HEADER_RE.match(line)
            if not m:
                return header_path, [], f"Invalid hunk header: {line}"

            old_start = int(m.group(1))
            old_len = int(m.group(2) or 1)
            new_start = int(m.group(3))
            new_len = int(m.group(4) or 1)

            i += 1
            hunk_lines: list[str] = []
            while i < len(lines):
                nxt = lines[i]
                if nxt.startswith("@@") or nxt.startswith("--- ") or nxt.startswith("diff --git "):
                    break
                hunk_lines.append(nxt)
                i += 1

            hunks.append((old_start, old_len, new_start, new_len, hunk_lines))
            continue

        i += 1

    if not hunks:
        return header_path, [], "No hunks found in diff (missing '@@ ... @@' sections)"

    return header_path, hunks, None


def _apply_unified_diff(original_text: str, hunks: list[tuple[int, int, int, int, list[str]]]) -> tuple[Optional[str], Optional[str]]:
    """Apply unified diff hunks to text."""
    ends_with_newline = original_text.endswith("\n")
    original_lines = original_text.splitlines()

    out: list[str] = []
    cursor = 0

    for old_start, _old_len, _new_start, _new_len, hunk_lines in hunks:
        hunk_start = max(old_start - 1, 0)
        if hunk_start > len(original_lines):
            return None, f"Hunk starts beyond end of file (start={old_start}, lines={len(original_lines)})"

        out.extend(original_lines[cursor:hunk_start])
        cursor = hunk_start

        for hl in hunk_lines:
            if hl == r"\ No newline at end of file":
                continue
            if not hl:
                return None, "Invalid diff line: empty line without prefix"

            prefix = hl[0]
            text = hl[1:]

            if prefix == " ":
                if cursor >= len(original_lines) or original_lines[cursor] != text:
                    got = original_lines[cursor] if cursor < len(original_lines) else "<EOF>"
                    return None, f"Context mismatch applying patch. Expected {text!r}, got {got!r}"
                out.append(text)
                cursor += 1
            elif prefix == "-":
                if cursor >= len(original_lines) or original_lines[cursor] != text:
                    got = original_lines[cursor] if cursor < len(original_lines) else "<EOF>"
                    return None, f"Remove mismatch applying patch. Expected {text!r}, got {got!r}"
                cursor += 1
            elif prefix == "+":
                out.append(text)
            else:
                return None, f"Invalid diff line prefix {prefix!r} (expected one of ' ', '+', '-')"

    out.extend(original_lines[cursor:])

    new_text = "\n".join(out)
    if ends_with_newline and not new_text.endswith("\n"):
        new_text += "\n"
    return new_text, None


def _render_edit_file_diff(*, path: Path, before: str, after: str) -> tuple[str, int, int]:
    """Render a compact, context-aware diff with per-line numbers.

    Output format is optimized for agent scratchpads and CLIs:
    - First line: `Edited <path> (+A -R)`
    - Then: unified diff hunks with 1 line of context, rendered with old/new line numbers.
    """
    import difflib
    import re

    old_lines = (before or "").splitlines()
    new_lines = (after or "").splitlines()

    diff_lines = list(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile=str(path),
            tofile=str(path),
            lineterm="",
            n=1,
        )
    )

    added = sum(1 for line in diff_lines if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff_lines if line.startswith("-") and not line.startswith("---"))

    kept: list[str] = []
    max_line = max(len(old_lines), len(new_lines), 1)
    width = max(1, len(str(max_line)))
    blank = " " * width

    old_no: int | None = None
    new_no: int | None = None
    hunk_re = re.compile(r"^@@ -(?P<o>\d+)(?:,(?P<oc>\d+))? \+(?P<n>\d+)(?:,(?P<nc>\d+))? @@")
    # Track per-hunk new-file line ranges to suggest bounded verification reads.
    hunk_ranges: list[tuple[int, int]] = []
    current_min_new: int | None = None
    current_max_new: int | None = None

    for line in diff_lines:
        if line.startswith(("---", "+++")):
            continue
        if line.startswith("@@"):
            if current_min_new is not None and current_max_new is not None:
                hunk_ranges.append((current_min_new, current_max_new))
            current_min_new = None
            current_max_new = None
            kept.append(line)
            m = hunk_re.match(line)
            if m:
                old_no = int(m.group("o"))
                new_no = int(m.group("n"))
            else:
                old_no = None
                new_no = None
            continue

        if not line:
            continue

        # Only annotate hunk body lines once we've seen a hunk header.
        if old_no is None or new_no is None:
            continue

        prefix = line[0]
        text = line[1:]

        if prefix == " ":
            # Context line: advances both old and new counters.
            if new_no is not None:
                current_min_new = new_no if current_min_new is None else min(current_min_new, new_no)
                current_max_new = new_no if current_max_new is None else max(current_max_new, new_no)
            kept.append(f" {old_no:>{width}} {new_no:>{width}} | {text}")
            old_no += 1
            new_no += 1
            continue
        if prefix == "-":
            # Deletion-only hunks still have a position in the new file; use the current new_no.
            if new_no is not None:
                current_min_new = new_no if current_min_new is None else min(current_min_new, new_no)
                current_max_new = new_no if current_max_new is None else max(current_max_new, new_no)
            kept.append(f"-{old_no:>{width}} {blank} | {text}")
            old_no += 1
            continue
        if prefix == "+":
            if new_no is not None:
                current_min_new = new_no if current_min_new is None else min(current_min_new, new_no)
                current_max_new = new_no if current_max_new is None else max(current_max_new, new_no)
            kept.append(f"+{blank} {new_no:>{width}} | {text}")
            new_no += 1
            continue

        # Fallback (rare): keep any other lines as-is (e.g. "\ No newline at end of file").
        kept.append(line)

    if current_min_new is not None and current_max_new is not None:
        hunk_ranges.append((current_min_new, current_max_new))

    body = "\n".join(kept).rstrip("\n")
    header = f"{_path_for_display(path)} (+{added} -{removed})"
    rendered = (f"Edited {header}\n{body}").rstrip()

    # Add a short, bounded verification hint so agents don't re-read entire files after small edits.
    if hunk_ranges:
        unique = []
        for start, end in hunk_ranges:
            if start <= 0 or end <= 0:
                continue
            unique.append((start, end))
        if unique:
            unique = sorted(set(unique))
            tips: list[str] = []
            abs_path = _path_for_display(path)
            for idx, (start, end) in enumerate(unique[:3], 1):
                a = max(1, start - 3)
                b = end + 3
                prefix = "Tip" if len(unique) == 1 else f"Tip (hunk {idx})"
                tips.append(
                    f"{prefix}: verify with read_file(file_path=\"{abs_path}\", start_line={a}, end_line={b})"
                )
            if len(unique) > 3:
                tips.append(f"Tip: {len(unique) - 3} more hunks not shown; use the diff above to choose ranges.")
            rendered = rendered + "\n\n" + "\n".join(tips)

    return (rendered, added, removed)


@tool(
    description="Surgically edit a text file via small find/replace (literal/regex) or a single-file unified diff patch.",
    when_to_use="Use for small, precise edits. Prefer search_files → read_file → edit_file with a small unique pattern; for whole-file rewrites, use write_file().",
    hide_args=["encoding", "flexible_whitespace"],
    examples=[
        {
            "description": "Surgical one-line replacement (bounded, safe)",
            "arguments": {
                "file_path": "config.py",
                "pattern": "debug = False",
                "replacement": "debug = True",
                "max_replacements": 1,
            },
        },
        {
            "description": "Update function definition using regex",
            "arguments": {
                "file_path": "script.py",
                "pattern": r"def old_function\\([^)]*\\):",
                "replacement": "def new_function(param1, param2):",
                "use_regex": True,
                "max_replacements": 1,
            },
        },
        {
            "description": "Preview changes before applying",
            "arguments": {
                "file_path": "test.py",
                "pattern": "class OldClass",
                "replacement": "class NewClass",
                "preview_only": True,
                "max_replacements": 1,
            },
        },
    ],
)
def edit_file(
    file_path: str,
    pattern: str,
    replacement: Optional[str] = None,
    use_regex: bool = False,
    max_replacements: int = -1,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    preview_only: bool = False,
    encoding: str = "utf-8",
    flexible_whitespace: bool = True,
) -> str:
    """
    Edit a UTF-8 text file.

    Two supported modes:
    1) **Find/replace mode** (recommended for small edits):
       - Provide `pattern` and `replacement` (optionally regex).
    2) **Unified diff mode** (recommended for precise multi-line edits):
       - Call `edit_file(file_path, patch)` with `replacement=None` and `pattern` set to a single-file unified diff.

    Finds patterns (text or regex) in files and replaces them with new content.
    For complex multi-line edits, prefer unified diff mode to avoid accidental partial matches.

    Args:
        file_path: Path to the file to edit
        pattern: Text or regex pattern to find
        replacement: Text to replace matches with
        use_regex: Whether to treat pattern as regex (default: False)
        max_replacements: Maximum number of replacements (-1 for unlimited, default: -1)
        start_line: Starting line number to limit search scope (1-indexed, optional)
        end_line: Ending line number to limit search scope (1-indexed, optional)
        preview_only: Show what would be changed without applying (default: False)
        encoding: File encoding (default: "utf-8")
        flexible_whitespace: Enable whitespace-flexible matching (default: True).
            When enabled, matches patterns even if indentation differs between
            the pattern and file content. Handles tabs vs spaces, different
            indentation levels, and line ending differences (\n vs \r\n).

    Returns:
        Success message with replacement details or error message

    Examples:
        edit_file("config.py", "debug = False", "debug = True")
        edit_file("script.py", r"def old_func\\([^)]*\\):", "def new_func():", use_regex=True)
        edit_file("document.txt", "TODO", "DONE", max_replacements=1)
        edit_file("test.py", "class OldClass", "class NewClass", preview_only=True)
        edit_file("app.py", \"\"\"--- a/app.py
+++ b/app.py
@@ -1,2 +1,2 @@
 print('hello')
-print('world')
+print('there')
\"\"\")
    """
    try:
        # Validate file exists and expand home directory shortcuts like ~
        path = Path(file_path).expanduser()
        display_path = _path_for_display(path)
        # Runtime-enforced filesystem ignore policy (.abstractignore + defaults).
        from .abstractignore import AbstractIgnore

        ignore = AbstractIgnore.for_path(path)
        if ignore.is_ignored(path, is_dir=False) or ignore.is_ignored(path.parent, is_dir=True):
            return f"❌ Refused: Path '{display_path}' is ignored by .abstractignore policy"
        if not path.exists():
            return f"❌ File not found: {display_path}"

        if not path.is_file():
            return f"❌ Path is not a file: {display_path}"

        # Read current content
        try:
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            return f"❌ Cannot decode file with encoding '{encoding}'. File may be binary."
        except Exception as e:
            return f"❌ Error reading file: {str(e)}"

        # Unified diff mode: treat `pattern` as a patch when `replacement` is omitted.
        if replacement is None:
            header_path, hunks, err = _parse_unified_diff(pattern)
            if err:
                return f"❌ Error: {err}"
            if header_path and not _is_suffix_path(header_path, path.resolve()):
                return (
                    "❌ Error: Patch file header does not match the provided path.\n"
                    f"Patch header: {header_path}\n"
                    f"Target path:  {path.resolve()}\n"
                    "Generate a unified diff targeting the exact file you want to edit."
                )

            updated, apply_err = _apply_unified_diff(content, hunks)
            if apply_err:
                return f"❌ Error: Patch did not apply cleanly: {apply_err}"

            assert updated is not None
            if updated == content:
                return "No changes applied (patch resulted in identical content)."

            rendered, _, _ = _render_edit_file_diff(path=path, before=content, after=updated)
            if preview_only:
                return rendered.replace("Edited ", "Preview ", 1)

            with open(path, "w", encoding=encoding) as f:
                f.write(updated)

            return rendered

        original_content = content

        # Normalize escape sequences - handles LLMs sending \\n instead of actual newlines
        pattern = _normalize_escape_sequences(pattern)
        replacement = _normalize_escape_sequences(replacement)

        if not isinstance(pattern, str) or not pattern:
            return "❌ Invalid pattern: pattern must be a non-empty string."

        # Handle line range targeting if specified
        search_content = content
        line_offset = 0
        if start_line is not None or end_line is not None:
            lines = content.splitlines(keepends=True)
            total_lines = len(lines)

            # Validate line range parameters
            if start_line is not None and (start_line < 1 or start_line > total_lines):
                return f"❌ Invalid start_line {start_line}. Must be between 1 and {total_lines}"

            if end_line is not None and (end_line < 1 or end_line > total_lines):
                return f"❌ Invalid end_line {end_line}. Must be between 1 and {total_lines}"

            if start_line is not None and end_line is not None and start_line > end_line:
                return f"❌ Invalid line range: start_line ({start_line}) cannot be greater than end_line ({end_line})"

            # Calculate actual line range (convert to 0-indexed)
            start_idx = (start_line - 1) if start_line is not None else 0
            end_idx = end_line if end_line is not None else total_lines

            # Extract target lines for search
            target_lines = lines[start_idx:end_idx]
            search_content = ''.join(target_lines)
            line_offset = start_idx  # Track where our search content starts in the original file


        # Perform pattern matching and replacement on targeted content
        matches_total: Optional[int] = None
        if use_regex:
            try:
                regex_pattern = re.compile(pattern, re.MULTILINE | re.DOTALL)
            except re.error as e:
                return f"❌ Invalid regex pattern '{pattern}': {str(e)}"

            # Count matches first
            matches = list(regex_pattern.finditer(search_content))
            matches_total = len(matches)
            if not matches:
                range_info = f" (lines {start_line}-{end_line})" if start_line is not None or end_line is not None else ""
                hint = ""
                if start_line is not None or end_line is not None:
                    hint = "\nHint: The match may exist outside the specified line range. Remove/widen start_line/end_line or re-read the file to confirm."
                diag = _format_edit_file_no_match_diagnostics(content=content, pattern=pattern, file_path=display_path)
                return f"❌ No matches found for regex pattern '{pattern}' in '{display_path}'{range_info}{hint}{diag}"

            # Apply replacements to search content
            if max_replacements == -1:
                updated_search_content = regex_pattern.sub(replacement, search_content)
                replacements_made = len(matches)
            else:
                updated_search_content = regex_pattern.sub(replacement, search_content, count=max_replacements)
                replacements_made = min(len(matches), max_replacements)
        else:
            # Simple text replacement on search content
            count = search_content.count(pattern)
            matches_total = count

            # If exact match fails and flexible_whitespace is enabled, try flexible matching
            if count == 0 and flexible_whitespace and '\n' in pattern:
                # Convert pattern to regex with flexible leading whitespace per line
                # Strategy: Replace each newline + whitespace with a regex that matches
                # any amount of leading whitespace
                flexible_result = _flexible_whitespace_match(
                    pattern, replacement, search_content, max_replacements
                )
                if flexible_result is not None:
                    updated_search_content, replacements_made = flexible_result
                else:
                    range_info = f" (lines {start_line}-{end_line})" if start_line is not None or end_line is not None else ""
                    hint = ""
                    if start_line is not None or end_line is not None:
                        hint = "\nHint: The match may exist outside the specified line range. Remove/widen start_line/end_line or re-read the file to confirm."
                    diag = _format_edit_file_no_match_diagnostics(content=content, pattern=pattern, file_path=display_path)
                    return f"❌ No occurrences of '{pattern}' found in '{display_path}'{range_info}{hint}{diag}"
            elif count == 0:
                range_info = f" (lines {start_line}-{end_line})" if start_line is not None or end_line is not None else ""
                hint = ""
                if start_line is not None or end_line is not None:
                    hint = "\nHint: The match may exist outside the specified line range. Remove/widen start_line/end_line or re-read the file to confirm."
                diag = _format_edit_file_no_match_diagnostics(content=content, pattern=pattern, file_path=display_path)
                return f"❌ No occurrences of '{pattern}' found in '{display_path}'{range_info}{hint}{diag}"
            else:
                # Exact match found
                def _idempotent_insert_replace_exact(
                    *,
                    search_content: str,
                    pattern: str,
                    replacement: str,
                    max_replacements: int,
                ) -> Optional[tuple[str, int]]:
                    """Idempotent insertion-oriented replace to prevent duplicate insertions.

                    Some edits are expressed as "keep the original text, but insert extra lines"
                    (e.g. replacement starts/ends with pattern). A naive `str.replace()` can
                    re-apply that insertion on subsequent identical calls because the pattern
                    remains present. This helper detects when the insertion is already present
                    around a match and skips it.
                    """
                    if not pattern or replacement == pattern:
                        return None

                    # Suffix insertion: replacement = pattern + suffix
                    if replacement.startswith(pattern):
                        suffix = replacement[len(pattern) :]
                        if not suffix:
                            return None
                        out: list[str] = []
                        i = 0
                        replaced = 0
                        while True:
                            pos = search_content.find(pattern, i)
                            if pos == -1:
                                out.append(search_content[i:])
                                break
                            out.append(search_content[i:pos])
                            after = pos + len(pattern)
                            if search_content.startswith(suffix, after):
                                out.append(pattern)
                            else:
                                if max_replacements != -1 and replaced >= max_replacements:
                                    out.append(pattern)
                                else:
                                    out.append(pattern + suffix)
                                    replaced += 1
                            i = after
                        return ("".join(out), replaced)

                    # Prefix insertion: replacement = prefix + pattern
                    if replacement.endswith(pattern):
                        prefix = replacement[: -len(pattern)]
                        if not prefix:
                            return None
                        out = []
                        i = 0
                        replaced = 0
                        plen = len(prefix)
                        while True:
                            pos = search_content.find(pattern, i)
                            if pos == -1:
                                out.append(search_content[i:])
                                break
                            out.append(search_content[i:pos])
                            already = pos >= plen and search_content[pos - plen : pos] == prefix
                            if already:
                                out.append(pattern)
                            else:
                                if max_replacements != -1 and replaced >= max_replacements:
                                    out.append(pattern)
                                else:
                                    out.append(prefix + pattern)
                                    replaced += 1
                            i = pos + len(pattern)
                        return ("".join(out), replaced)

                    return None

                idempotent_result = _idempotent_insert_replace_exact(
                    search_content=search_content,
                    pattern=pattern,
                    replacement=replacement,
                    max_replacements=max_replacements,
                )
                if idempotent_result is not None:
                    updated_search_content, replacements_made = idempotent_result
                else:
                    if max_replacements == -1:
                        updated_search_content = search_content.replace(pattern, replacement)
                        replacements_made = count
                    else:
                        updated_search_content = search_content.replace(pattern, replacement, max_replacements)
                        replacements_made = min(count, max_replacements)

        # Reconstruct the full file content if line ranges were used
        if start_line is not None or end_line is not None:
            lines = content.splitlines(keepends=True)
            start_idx = (start_line - 1) if start_line is not None else 0
            end_idx = end_line if end_line is not None else len(lines)

            # Rebuild the file with the updated targeted section
            updated_content = ''.join(lines[:start_idx]) + updated_search_content + ''.join(lines[end_idx:])
        else:
            updated_content = updated_search_content

        if updated_content == original_content:
            return "No changes would be applied." if preview_only else "No changes applied (resulted in identical content)."

        rendered, _, _ = _render_edit_file_diff(path=path, before=original_content, after=updated_content)
        rendered_lines = rendered.splitlines()
        if rendered_lines:
            if isinstance(matches_total, int) and matches_total > 0:
                rendered_lines[0] = f"{rendered_lines[0]} replacements={replacements_made}/{matches_total}"
            else:
                rendered_lines[0] = f"{rendered_lines[0]} replacements={replacements_made}"
        rendered = "\n".join(rendered_lines).rstrip()

        if (
            isinstance(matches_total, int)
            and matches_total > 0
            and isinstance(replacements_made, int)
            and 0 <= replacements_made < matches_total
            and max_replacements != -1
        ):
            remaining = matches_total - replacements_made
            rendered = (
                rendered
                + "\n\n"
                f"Note: {remaining} more match(es) remain. "
                "Next step: re-run edit_file with a higher max_replacements, or target the remaining occurrence(s) with start_line/end_line."
            )

        if preview_only:
            return rendered.replace("Edited ", "Preview ", 1)

        # Apply changes to file
        try:
            with open(path, "w", encoding=encoding) as f:
                f.write(updated_content)
        except Exception as e:
            return f"❌ Write failed: {str(e)}"

        return rendered

    except Exception as e:
        return f"❌ Error editing file: {str(e)}"


@tool(
    description="Execute shell commands safely with security controls and platform detection",
    when_to_use="When you need to run system commands, shell scripts, or interact with command-line tools",
    examples=[
        {
            "description": "List current directory contents",
            "arguments": {
                "command": "ls -la"
            }
        },
        {
            "description": "Search for a pattern in files (grep)",
            "arguments": {
                "command": "grep -R \"ActiveContextPolicy\" -n abstractruntime/src/abstractruntime | head"
            }
        },
        {
            "description": "Safe mode with confirmation",
            "arguments": {
                "command": "rm temp_file.txt",
                "require_confirmation": True
            }
        }
    ]
)
def execute_command(
    command: str,
    working_directory: str = None,
    timeout: int = 300,
    capture_output: bool = True,
    require_confirmation: bool = False,
    allow_dangerous: bool = False
) -> Dict[str, Any]:
    """
    Execute a shell command safely with comprehensive security controls.

    Args:
        command: The shell command to execute
        working_directory: Directory to run the command in (default: current directory)
        timeout: Maximum seconds to wait for command completion (default: 300)
        capture_output: Whether to capture and return command output (default: True)
        require_confirmation: Whether to ask for user confirmation before execution (default: False)
        allow_dangerous: Whether to allow potentially dangerous commands (default: False)

    Returns:
        Structured command execution result (JSON-safe).
    """
    try:
        # Platform detection
        current_platform = platform.system()

        def _truncate(text: str, *, limit: int) -> tuple[str, bool]:
            s = "" if text is None else str(text)
            if limit <= 0:
                return s, False
            if len(s) <= limit:
                return s, False
            return s[:limit], True

        # CRITICAL SECURITY VALIDATION - Dangerous commands MUST be blocked
        security_check = _validate_command_security(command, allow_dangerous)
        if not security_check["safe"]:
            rendered = (
                f"🚫 CRITICAL SECURITY BLOCK: {security_check['reason']}\n"
                f"BLOCKED COMMAND: {command}\n"
                f"⚠️  DANGER: This command could cause IRREVERSIBLE DAMAGE\n"
                f"Only use allow_dangerous=True with EXPRESS USER CONSENT\n"
                f"This safety mechanism protects your system and data"
            )
            return {
                "success": False,
                "error": str(security_check.get("reason") or "CRITICAL SECURITY BLOCK").strip(),
                "command": str(command),
                "platform": str(current_platform),
                "working_directory": str(working_directory or ""),
                "rendered": rendered,
            }

        # User confirmation for risky commands
        if require_confirmation:
            risk_level = _assess_command_risk(command)
            if risk_level != "low":
                logger.warning(f"Command execution simulated - {risk_level} risk command: {command}")
                logger.warning(f"Would normally ask for user confirmation before proceeding")

        # Working directory validation
        if working_directory:
            # Expand home directory shortcuts like ~ before resolving
            working_dir = Path(working_directory).expanduser().resolve()
            if not working_dir.exists():
                rendered = f"❌ Error: Working directory does not exist: {working_directory}"
                return {
                    "success": False,
                    "error": rendered.lstrip("❌").strip(),
                    "command": str(command),
                    "platform": str(current_platform),
                    "working_directory": str(working_directory),
                    "rendered": rendered,
                }
            if not working_dir.is_dir():
                rendered = f"❌ Error: Working directory path is not a directory: {working_directory}"
                return {
                    "success": False,
                    "error": rendered.lstrip("❌").strip(),
                    "command": str(command),
                    "platform": str(current_platform),
                    "working_directory": str(working_directory),
                    "rendered": rendered,
                }
        else:
            working_dir = None

        # Command execution
        start_time = time.time()

        try:
            # Execute command with security controls
            result = subprocess.run(
                command,
                shell=True,
                cwd=working_dir,
                timeout=timeout,
                capture_output=capture_output,
                text=True,
                check=False  # Don't raise exception on non-zero return code
            )

            execution_time = time.time() - start_time

            # Format results
            output_parts = []
            output_parts.append(f"🖥️  Command executed on {current_platform}")
            output_parts.append(f"💻 Command: {command}")
            output_parts.append(f"📁 Working directory: {working_dir or os.getcwd()}")
            output_parts.append(f"⏱️  Execution time: {execution_time:.2f}s")
            output_parts.append(f"🔢 Return code: {result.returncode}")

            stdout_full = result.stdout or ""
            stderr_full = result.stderr or ""

            stdout_preview = ""
            stderr_preview = ""
            stdout_truncated = False
            stderr_truncated = False

            if capture_output:
                if stdout_full:
                    # Keep the rendered preview bounded for LLM usability. Full output is still returned
                    # in structured fields so higher layers can store it durably as evidence.
                    stdout_preview, stdout_truncated = _truncate(stdout_full, limit=20000)
                    if stdout_truncated:
                        stdout_preview += f"\n... (output truncated, {len(stdout_full)} total chars)"
                    output_parts.append(f"\n📤 STDOUT:\n{stdout_preview}")

                if stderr_full:
                    stderr_preview, stderr_truncated = _truncate(stderr_full, limit=5000)
                    if stderr_truncated:
                        stderr_preview += f"\n... (error output truncated, {len(stderr_full)} total chars)"
                    output_parts.append(f"\n❌ STDERR:\n{stderr_preview}")

                if result.returncode == 0:
                    output_parts.append("\n✅ Command completed successfully")
                else:
                    output_parts.append(f"\n⚠️  Command completed with non-zero exit code: {result.returncode}")
            else:
                output_parts.append("📝 Output capture disabled")

            rendered = "\n".join(output_parts)
            ok = bool(result.returncode == 0)
            err = None if ok else f"Command completed with non-zero exit code: {int(result.returncode)}"
            return {
                "success": ok,
                "error": err,
                "command": str(command),
                "platform": str(current_platform),
                "working_directory": str(working_dir or os.getcwd()),
                "duration_s": float(execution_time),
                "return_code": int(result.returncode),
                "stdout": stdout_full if capture_output else "",
                "stderr": stderr_full if capture_output else "",
                "stdout_preview": stdout_preview,
                "stderr_preview": stderr_preview,
                "stdout_truncated": bool(stdout_truncated),
                "stderr_truncated": bool(stderr_truncated),
                "rendered": rendered,
            }

        except subprocess.TimeoutExpired:
            rendered = (
                f"⏰ Timeout: Command exceeded {timeout} seconds\n"
                f"Command: {command}\n"
                "Consider increasing timeout or breaking down the command"
            )
            return {
                "success": False,
                "error": f"Tool timeout after {int(timeout)}s",
                "command": str(command),
                "platform": str(current_platform),
                "working_directory": str(working_dir or os.getcwd()) if "working_dir" in locals() else str(working_directory or ""),
                "timeout_s": int(timeout),
                "rendered": rendered,
            }

        except subprocess.CalledProcessError as e:
            rendered = (
                "❌ Command execution failed\n"
                f"Command: {command}\n"
                f"Return code: {e.returncode}\n"
                f"Error: {e.stderr if e.stderr else 'No error details'}"
            )
            return {
                "success": False,
                "error": "Command execution failed",
                "command": str(command),
                "platform": str(current_platform),
                "working_directory": str(working_dir or os.getcwd()) if "working_dir" in locals() else str(working_directory or ""),
                "return_code": int(getattr(e, "returncode", -1) or -1),
                "stderr": str(getattr(e, "stderr", "") or ""),
                "rendered": rendered,
            }

    except Exception as e:
        rendered = f"❌ Execution error: {str(e)}\nCommand: {command}"
        return {
            "success": False,
            "error": str(e),
            "command": str(command),
            "platform": str(platform.system()),
            "working_directory": str(working_directory or ""),
            "rendered": rendered,
        }


def _validate_command_security(command: str, allow_dangerous: bool = False) -> dict:
    """
    CRITICAL SECURITY VALIDATION - Protects against destructive commands.

    This function implements multiple layers of protection:
    1. Regex pattern matching for known destructive commands
    2. Keyword scanning for dangerous operations
    3. Path analysis for system-critical locations
    4. Only bypassed with explicit allow_dangerous=True (requires express user consent)
    """

    if allow_dangerous:
        return {"safe": True, "reason": "DANGEROUS COMMANDS EXPLICITLY ALLOWED BY USER"}

    # Normalize command for analysis
    cmd_lower = command.lower().strip()

    # CRITICAL: Highly destructive commands (NEVER allow without express consent)
    critical_patterns = [
        r'\brm\s+(-rf?|--recursive|--force)',  # rm -rf, rm -r, rm -f
        r'\bdd\s+if=.*of=',  # dd operations (disk destruction)
        r'\bmkfs\.',         # filesystem formatting
        r'\bfdisk\b',        # partition management
        r'\bparted\b',       # partition editor
        r'\bshred\b',        # secure deletion
        r'\bwipe\b',         # disk wiping
        r'>\s*/dev/(sd[a-z]|nvme)',  # writing to disk devices
        r'\bchmod\s+777',    # overly permissive permissions
        r'\bsudo\s+(rm|dd|mkfs|fdisk)',  # sudo + destructive commands
        r'curl.*\|\s*(bash|sh|python)',  # piping downloads to interpreter
        r'wget.*\|\s*(bash|sh|python)',  # piping downloads to interpreter
        r'\bkill\s+-9\s+1\b',  # killing init process
        r'\binit\s+0',       # system shutdown
        r'\bshutdown\b',     # system shutdown
        r'\breboot\b',       # system reboot
        r'\bhalt\b',         # system halt
    ]

    for pattern in critical_patterns:
        if re.search(pattern, cmd_lower):
            return {
                "safe": False,
                "reason": f"CRITICAL DESTRUCTIVE PATTERN: {pattern} - Could cause IRREVERSIBLE system damage"
            }

    # System-critical paths (additional protection)
    critical_paths = ['/etc/', '/usr/', '/var/', '/opt/', '/boot/', '/sys/', '/proc/']
    if any(path in command for path in critical_paths):
        # Check if it's a destructive operation on critical paths
        destructive_ops_pattern = r'\b(rm|del|format)\s+.*(' + '|'.join(re.escape(p) for p in critical_paths) + ')'
        redirect_ops_pattern = r'.*(>|>>)\s*(' + '|'.join(re.escape(p) for p in critical_paths) + ')'

        if re.search(destructive_ops_pattern, cmd_lower) or re.search(redirect_ops_pattern, cmd_lower):
            return {
                "safe": False,
                "reason": "CRITICAL SYSTEM PATH MODIFICATION - Could corrupt operating system"
            }

    # High-risk keywords (warrant extreme caution)
    high_risk_keywords = [
        'format c:', 'format d:', 'del /f', 'deltree', 'destroy', 'wipe',
        'kill -9', ':(){:|:&};:', 'forkbomb'  # Include shell fork bomb
    ]
    for keyword in high_risk_keywords:
        if keyword in cmd_lower:
            return {
                "safe": False,
                "reason": f"HIGH-RISK KEYWORD: {keyword} - Requires EXPRESS user consent"
            }

    return {"safe": True, "reason": "Command passed comprehensive security validation"}


def _assess_command_risk(command: str) -> str:
    """Assess the risk level of a command for confirmation purposes."""

    cmd_lower = command.lower().strip()

    # High risk patterns
    high_risk = ['rm ', 'del ', 'format', 'fdisk', 'mkfs', 'dd ', 'shred']
    for pattern in high_risk:
        if pattern in cmd_lower:
            return "high"

    # Medium risk patterns
    medium_risk = ['chmod', 'chown', 'sudo', 'su ', 'passwd', 'crontab']
    for pattern in medium_risk:
        if pattern in cmd_lower:
            return "medium"

    # File system modification patterns
    if any(op in cmd_lower for op in ['>', '>>', '|', 'mv ', 'cp ', 'mkdir', 'touch']):
        return "medium"

    return "low"


# Export all tools for easy importing
__all__ = [
    'list_files',
    'search_files',
    'read_file',
    'write_file',
    'edit_file',
    'web_search',
    'fetch_url',
    'execute_command'
]
