"""
Common shareable tools for AbstractLLM applications.

This module provides a collection of utility tools for file operations,
web scraping, command execution, and user interaction.

Migrated from legacy system with enhanced decorator support.
"""

import os
import json
import subprocess
import requests
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import glob
import shutil
from urllib.parse import urljoin, urlparse
import logging
import platform
import re
import time

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import our enhanced tool decorator
from abstractllm.tools.core import tool

logger = logging.getLogger(__name__)

# File Operations
@tool(
    description="Find and list files and directories by their names/paths using glob patterns (case-insensitive, supports multiple patterns)",
    tags=["file", "directory", "listing", "filesystem"],
    when_to_use="When you need to find files by their names, paths, or file extensions (NOT for searching file contents)",
    examples=[
        {
            "description": "List all files in current directory",
            "arguments": {
                "directory_path": ".",
                "pattern": "*"
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
            "description": "Find all files with 'test' in filename (case-insensitive)",
            "arguments": {
                "directory_path": ".",
                "pattern": "*test*",
                "recursive": True
            }
        },
        {
            "description": "Find multiple file types using | separator",
            "arguments": {
                "directory_path": ".",
                "pattern": "*.py|*.js|*.md",
                "recursive": True
            }
        },
        {
            "description": "Complex multiple patterns - documentation, tests, and config files",
            "arguments": {
                "directory_path": ".",
                "pattern": "README*|*test*|config.*|*.yml",
                "recursive": True
            }
        },
        {
            "description": "List all files including hidden ones",
            "arguments": {
                "directory_path": ".",
                "pattern": "*",
                "include_hidden": True
            }
        }
    ]
)
def list_files(directory_path: str = ".", pattern: str = "*", recursive: bool = False, include_hidden: bool = False, head_limit: Optional[int] = 50) -> str:
    """
    List files and directories in a specified directory with pattern matching (case-insensitive).

    IMPORTANT: Use 'directory_path' parameter (not 'file_path') to specify the directory to list.

    Args:
        directory_path: Path to the directory to list files from (default: "." for current directory)
        pattern: Glob pattern(s) to match files. Use "|" to separate multiple patterns (default: "*")
        recursive: Whether to search recursively in subdirectories (default: False)
        include_hidden: Whether to include hidden files/directories starting with '.' (default: False)
        head_limit: Maximum number of files to return (default: 50, None for unlimited)

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
        directory = Path(directory_path)

        if not directory.exists():
            return f"Error: Directory '{directory_path}' does not exist"

        if not directory.is_dir():
            return f"Error: '{directory_path}' is not a directory"

        # Split pattern by | to support multiple patterns
        patterns = [p.strip() for p in pattern.split('|')]

        # Get all files first, then apply case-insensitive pattern matching
        import fnmatch
        all_files = []

        if recursive:
            for root, dirs, dir_files in os.walk(directory):
                for f in dir_files:
                    all_files.append(Path(root) / f)
        else:
            try:
                all_files = [f for f in directory.iterdir() if f.is_file()]
                if include_hidden:
                    # Add hidden files
                    hidden_files = [f for f in directory.iterdir() if f.name.startswith('.') and f.is_file()]
                    all_files.extend(hidden_files)
            except PermissionError:
                pass

        # Apply case-insensitive pattern matching
        matched_files = []
        for file_path in all_files:
            filename = file_path.name

            # Check if file matches any pattern (case-insensitive)
            for single_pattern in patterns:
                if fnmatch.fnmatch(filename.lower(), single_pattern.lower()):
                    matched_files.append(str(file_path))
                    break

        files = matched_files

        if not files:
            return f"No files found matching pattern '{pattern}' in '{directory_path}'"

        # Filter out hidden files if include_hidden is False (already handled in file collection above)
        if not include_hidden:
            filtered_files = []
            for file_path in files:
                path_obj = Path(file_path)
                # Check if any part of the path (after the directory_path) starts with '.'
                relative_path = path_obj.relative_to(directory) if directory != Path('.') else path_obj
                is_hidden = any(part.startswith('.') for part in relative_path.parts)
                if not is_hidden:
                    filtered_files.append(file_path)
            files = filtered_files

        if not files:
            hidden_note = " (hidden files excluded)" if not include_hidden else ""
            return f"No files found matching pattern '{pattern}' in '{directory_path}'{hidden_note}"

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
            limit_note = f" (showing {head_limit} of {total_files} files)"
            is_truncated = True
        else:
            limit_note = ""

        hidden_note = " (hidden files excluded)" if not include_hidden else ""
        output = [f"Files in '{directory_path}' matching '{pattern}'{hidden_note}{limit_note}:"]

        for file_path in files:
            path_obj = Path(file_path)
            if path_obj.is_file():
                size = path_obj.stat().st_size
                size_str = f"{size:,} bytes"
                output.append(f"  📄 {path_obj.name} ({size_str})")
            elif path_obj.is_dir():
                output.append(f"  📁 {path_obj.name}/")

        # Add helpful hint when results are truncated
        if is_truncated:
            remaining = total_files - head_limit
            recursive_hint = ", recursive=True" if recursive else ""
            hidden_hint = ", include_hidden=True" if include_hidden else ""
            output.append(f"\n💡 {remaining} more files available. Use list_files('{directory_path}', '{pattern}'{recursive_hint}{hidden_hint}, head_limit=None) to see all.")

        return "\n".join(output)

    except Exception as e:
        return f"Error listing files: {str(e)}"


@tool(
    description="Search for text patterns INSIDE files using regex (returns file paths with line numbers by default)",
    tags=["search", "content", "regex", "grep", "text"],
    when_to_use="When you need to find specific text, code patterns, or content INSIDE files (NOT for finding files by names)",
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
            "description": "Show content for specific patterns (limited results)",
            "arguments": {
                "pattern": "generate.*tools|create_react_cycle",
                "path": "abstractllm/session.py",
                "output_mode": "content",
                "head_limit": 5
            }
        }
    ]
)
def search_files(pattern: str, path: str = ".", output_mode: str = "files_with_matches", head_limit: Optional[int] = 20, file_pattern: str = "*", case_sensitive: bool = False, multiline: bool = False) -> str:
    """
    Enhanced search tool with regex support and flexible output modes.

    Similar to grep functionality, this tool can search for patterns in files
    with various output formats and options.

    Args:
        pattern: Regular expression pattern to search for
        path: File or directory path to search in (default: current directory)
        output_mode: Output format - "files_with_matches" (show file paths with line numbers), "content" (show matching lines), "count" (show match counts) (default: "files_with_matches")
        head_limit: Limit output to first N entries (default: 20)
        file_pattern: Glob pattern(s) for files to search. Use "|" to separate multiple patterns (default: "*" for all files)
        case_sensitive: Whether search should be case sensitive (default: False)
        multiline: Enable multiline matching where pattern can span lines (default: False)

    Returns:
        Search results in the specified format or error message

    Examples:
        search_files("generate.*react|create_react_cycle", "abstractllm/session.py")  # Returns file paths with line numbers (default)
        search_files("def.*search", ".", file_pattern="*.py")  # Search Python files only
        search_files("import.*re", ".", file_pattern="*.py|*.js")  # Search Python and JavaScript files
        search_files("TODO|FIXME", ".", file_pattern="*.py|*.md|*.txt")  # Find TODO/FIXME in multiple file types
        search_files("import.*re", ".", "content", 10)  # Show content with 10 match limit
        search_files("pattern", ".", "count")  # Count matches per file
    """
    try:
        search_path = Path(path)

        # Compile regex pattern
        flags = 0 if case_sensitive else re.IGNORECASE
        if multiline:
            flags |= re.MULTILINE | re.DOTALL

        try:
            regex_pattern = re.compile(pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern '{pattern}': {str(e)}"

        # Determine if path is a file or directory
        if search_path.is_file():
            files_to_search = [search_path]
        elif search_path.is_dir():
            # Find files matching pattern in directory
            if file_pattern == "*":
                # Search all files recursively
                files_to_search = []
                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        file_path = Path(root) / file
                        # Skip binary files by checking if they're text files
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                f.read(1024)  # Try to read first 1KB
                            files_to_search.append(file_path)
                        except (UnicodeDecodeError, PermissionError):
                            continue  # Skip binary/inaccessible files
            else:
                # Support multiple patterns separated by |
                import fnmatch
                file_patterns = [p.strip() for p in file_pattern.split('|')]
                files_to_search = []

                for root, dirs, files in os.walk(search_path):
                    for file in files:
                        file_path = Path(root) / file
                        filename = file_path.name

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
                            except (UnicodeDecodeError, PermissionError):
                                continue  # Skip binary/inaccessible files
        else:
            return f"Error: Path '{path}' does not exist"

        if not files_to_search:
            return f"No files found to search in '{path}'"

        # Search through files
        results = []
        files_with_matches = []  # Will store (file_path, [line_numbers]) tuples
        match_counts = {}
        total_matches = 0

        for file_path in files_to_search:
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    if multiline:
                        content = f.read()
                        matches = list(regex_pattern.finditer(content))

                        if matches:
                            # Collect line numbers for files_with_matches mode
                            line_numbers = []
                            for match in matches:
                                line_num = content[:match.start()].count('\n') + 1
                                line_numbers.append(line_num)

                            files_with_matches.append((str(file_path), line_numbers))
                            match_counts[str(file_path)] = len(matches)
                            total_matches += len(matches)

                            if output_mode == "content":
                                results.append(f"\n📄 {file_path}:")

                                # Convert content to lines for line number calculation
                                lines = content.splitlines()
                                for match in matches:
                                    # Find line number for match
                                    line_num = content[:match.start()].count('\n') + 1

                                    # Get the matched text
                                    matched_text = match.group()
                                    # If multiline match, show first line only
                                    if '\n' in matched_text:
                                        matched_text = matched_text.split('\n')[0] + "..."

                                    # Get the full line containing the match start
                                    if line_num <= len(lines):
                                        full_line = lines[line_num - 1]
                                        results.append(f"    Line {line_num}: {full_line}")

                                    # Apply head_limit for content mode
                                    if head_limit and len([r for r in results if r.startswith("    Line")]) >= head_limit:
                                        break
                    else:
                        lines = f.readlines()
                        file_matches = []

                        for line_num, line in enumerate(lines, 1):
                            line_content = line.rstrip()
                            matches = list(regex_pattern.finditer(line_content))

                            if matches:
                                file_matches.extend(matches)
                                if output_mode == "content":
                                    results.append(f"    Line {line_num}: {line_content}")

                        if file_matches:
                            # Collect line numbers for files_with_matches mode
                            line_numbers = []
                            for line_num, line in enumerate(lines, 1):
                                line_content = line.rstrip()
                                if regex_pattern.search(line_content):
                                    line_numbers.append(line_num)

                            files_with_matches.append((str(file_path), line_numbers))
                            match_counts[str(file_path)] = len(file_matches)
                            total_matches += len(file_matches)

                            if output_mode == "content" and file_matches:
                                # Insert file header before the lines we just added
                                file_header_position = len(results) - len(file_matches)
                                results.insert(file_header_position, f"\n📄 {file_path}:")

                                # Apply head_limit for content mode
                                if head_limit:
                                    content_lines = [r for r in results if r.startswith("    Line")]
                                    if len(content_lines) >= head_limit:
                                        break

            except Exception as e:
                if output_mode == "content":
                    results.append(f"\n⚠️  Error reading {file_path}: {str(e)}")

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
                    # Format line numbers nicely
                    if len(line_numbers) == 1:
                        line_info = f"line {line_numbers[0]}"
                    elif len(line_numbers) <= 5:
                        line_info = f"lines {', '.join(map(str, line_numbers))}"
                    else:
                        # Show first few line numbers and total count
                        first_lines = ', '.join(map(str, line_numbers[:3]))
                        line_info = f"lines {first_lines}... ({len(line_numbers)} total)"

                    formatted_results.append(f"{file_path} ({line_info})")

                # Add helpful hint when results are truncated
                if is_truncated:
                    remaining = total_files_with_matches - head_limit
                    case_hint = "" if case_sensitive else ", case_sensitive=False"
                    multiline_hint = ", multiline=True" if multiline else ""
                    file_pattern_hint = f", file_pattern='{file_pattern}'" if file_pattern != "*" else ""
                    formatted_results.append(f"\n💡 {remaining} more files with matches available. Use search_files('{pattern}', '{path}', head_limit=None{case_hint}{multiline_hint}{file_pattern_hint}) to see all.")

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
                    count_results.append(f"\n💡 {remaining} more files with matches available. Use search_files('{pattern}', '{path}', 'count', head_limit=None{case_hint}{multiline_hint}{file_pattern_hint}) to see all.")

                return "\n".join(count_results)
            else:
                return f"No matches found for pattern '{pattern}'"

        else:  # content mode
            if not results:
                return f"No matches found for pattern '{pattern}'"

            # Count files with matches for header
            file_count = len([r for r in results if r.startswith("\n📄")])
            header = f"Search results for pattern '{pattern}' in {file_count} files:"

            # Apply head_limit to final output if specified
            final_results = results
            if head_limit:
                content_lines = [r for r in results if r.startswith("    Line")]
                if len(content_lines) > head_limit:
                    # Keep file headers and trim content lines
                    trimmed_results = []
                    content_count = 0
                    for line in results:
                        if line.startswith("    Line"):
                            if content_count < head_limit:
                                trimmed_results.append(line)
                                content_count += 1
                        else:
                            trimmed_results.append(line)
                    final_results = trimmed_results
                    final_results.append(f"\n... (showing first {head_limit} matches)")

            return header + "\n" + "\n".join(final_results)

    except Exception as e:
        return f"Error performing search: {str(e)}"


@tool(
    description="Read the contents of a file with optional line range and hidden file access",
    tags=["file", "read", "content", "text"],
    when_to_use="When you need to read file contents, examine code, or extract specific line ranges from files",
    examples=[
        {
            "description": "Read entire file",
            "arguments": {
                "file_path": "README.md"
            }
        },
        {
            "description": "Read specific line range",
            "arguments": {
                "file_path": "src/main.py",
                "should_read_entire_file": False,
                "start_line_one_indexed": 10,
                "end_line_one_indexed_inclusive": 25
            }
        },
        {
            "description": "Read hidden file",
            "arguments": {
                "file_path": ".gitignore",
                "include_hidden": True
            }
        },
        {
            "description": "Read first 50 lines",
            "arguments": {
                "file_path": "large_file.txt",
                "should_read_entire_file": False,
                "end_line_one_indexed_inclusive": 50
            }
        }
    ]
)
def read_file(file_path: str, should_read_entire_file: bool = True, start_line_one_indexed: int = 1, end_line_one_indexed_inclusive: Optional[int] = None, include_hidden: bool = False) -> str:
    """
    Read the contents of a file with optional line range.

    Args:
        file_path: Path to the file to read
        should_read_entire_file: Whether to read the entire file (default: True)
        start_line_one_indexed: Starting line number (1-indexed, default: 1)
        end_line_one_indexed_inclusive: Ending line number (1-indexed, inclusive, optional)
        include_hidden: Whether to allow reading hidden files starting with '.' (default: False)

    Returns:
        File contents or error message
    """
    try:
        path = Path(file_path)

        if not path.exists():
            return f"Error: File '{file_path}' does not exist"

        if not path.is_file():
            return f"Error: '{file_path}' is not a file"

        # Check for hidden files (files starting with '.')
        if not include_hidden and path.name.startswith('.'):
            return f"Error: Access to hidden file '{file_path}' is not allowed. Use include_hidden=True to override."

        with open(path, 'r', encoding='utf-8') as f:
            if should_read_entire_file:
                # Read entire file
                content = f.read()
                line_count = len(content.splitlines())
                return f"File: {file_path} ({line_count} lines)\n\n{content}"
            else:
                # Read specific line range
                lines = f.readlines()
                total_lines = len(lines)

                # Convert to 0-indexed and validate
                start_idx = max(0, start_line_one_indexed - 1)
                end_idx = min(total_lines, end_line_one_indexed_inclusive or total_lines)

                if start_idx >= total_lines:
                    return f"Error: Start line {start_line_one_indexed} exceeds file length ({total_lines} lines)"

                selected_lines = lines[start_idx:end_idx]

                # Format without line numbers (as in legacy)
                result_lines = []
                for line in selected_lines:
                    result_lines.append(f"{line.rstrip()}")

                return "\n".join(result_lines)

    except UnicodeDecodeError:
        return f"Error: Cannot read '{file_path}' - file appears to be binary"
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except PermissionError:
        return f"Error: Permission denied reading file: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool(
    description="Write content to a file with robust error handling, creating directories if needed",
    tags=["file", "write", "create", "append", "content", "output"],
    when_to_use="When you need to create new files, save content, or append to existing files",
    examples=[
        {
            "description": "Write a simple text file",
            "arguments": {
                "file_path": "output.txt",
                "content": "Hello, world!"
            }
        },
        {
            "description": "Create a Python script",
            "arguments": {
                "file_path": "script.py",
                "content": "#!/usr/bin/env python3\nprint('Hello from Python!')"
            }
        },
        {
            "description": "Append to existing file",
            "arguments": {
                "file_path": "log.txt",
                "content": "\nNew log entry at 2025-01-01",
                "mode": "a"
            }
        },
        {
            "description": "Create file in nested directory",
            "arguments": {
                "file_path": "docs/api/endpoints.md",
                "content": "# API Endpoints\n\n## Authentication\n..."
            }
        },
        {
            "description": "Write JSON data",
            "arguments": {
                "file_path": "config.json",
                "content": "{\n  \"api_key\": \"test\",\n  \"debug\": true\n}"
            }
        }
    ]
)
def write_file(file_path: str, content: str = "", mode: str = "w", create_dirs: bool = True) -> str:
    """
    Write content to a file with robust error handling.

    This tool creates or overwrites a file with the specified content.
    It can optionally create parent directories if they don't exist.

    Args:
        file_path: Path to the file to write (relative or absolute)
        content: The content to write to the file (default: empty string)
        mode: Write mode - "w" to overwrite, "a" to append (default: "w")
        create_dirs: Whether to create parent directories if they don't exist (default: True)

    Returns:
        Success message with file information

    Raises:
        PermissionError: If lacking write permissions
        OSError: If there are filesystem issues
    """
    try:
        # Convert to Path object for better handling
        path = Path(file_path)

        # Create parent directories if requested and they don't exist
        if create_dirs and path.parent != path:
            path.parent.mkdir(parents=True, exist_ok=True)

        # Write the content to the file
        with open(path, mode, encoding='utf-8') as f:
            f.write(content)

        # Get file size for confirmation
        file_size = path.stat().st_size

        # Enhanced success message with emoji and formatting
        action = "appended to" if mode == "a" else "written to"
        return f"✅ Successfully {action} '{file_path}' ({file_size:,} bytes)"

    except PermissionError:
        return f"❌ Permission denied: Cannot write to '{file_path}'"
    except FileNotFoundError:
        return f"❌ Directory not found: Parent directory of '{file_path}' does not exist"
    except OSError as e:
        return f"❌ File system error: {str(e)}"
    except Exception as e:
        return f"❌ Unexpected error writing file: {str(e)}"


@tool(
    description="Search the web for real-time information using DuckDuckGo (no API key required)",
    tags=["web", "search", "internet", "information", "research"],
    when_to_use="When you need current information, research topics, or verify facts that might not be in your training data",
    examples=[
        {
            "description": "Search for current programming best practices",
            "arguments": {
                "query": "python best practices 2025",
                "num_results": 5
            }
        },
        {
            "description": "Research a technology or framework",
            "arguments": {
                "query": "semantic search embedding models comparison",
                "num_results": 3
            }
        },
        {
            "description": "Get current news or events",
            "arguments": {
                "query": "AI developments 2025"
            }
        },
        {
            "description": "Find documentation or tutorials",
            "arguments": {
                "query": "LanceDB vector database tutorial",
                "num_results": 4
            }
        },
        {
            "description": "Search with strict content filtering",
            "arguments": {
                "query": "machine learning basics",
                "safe_search": "strict"
            }
        },
        {
            "description": "Get UK-specific results",
            "arguments": {
                "query": "data protection regulations",
                "region": "uk-en"
            }
        }
    ]
)
def web_search(query: str, num_results: int = 5, safe_search: str = "moderate", region: str = "us-en") -> str:
    """
    Search the internet using DuckDuckGo (no API key required).

    Args:
        query: Search query
        num_results: Number of results to return (default: 5)
        safe_search: Content filtering level - "strict", "moderate", or "off" (default: "moderate")
        region: Regional results preference - "us-en", "uk-en", "ca-en", "au-en", etc. (default: "us-en")

    Returns:
        Search results or error message

    Note:
        DuckDuckGo Instant Answer API does not support time range filtering.
        For time-specific searches, include date terms in your query (e.g., "python best practices 2025").
    """
    try:
        # Simple DuckDuckGo instant answer API
        url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1',
            'no_redirect': '1',  # Faster responses
            'safe_search': safe_search,
            'region': region
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = []
        results.append(f"Search results for: '{query}'")

        # Abstract (main result)
        if data.get('Abstract'):
            results.append(f"\n📝 Summary: {data['Abstract']}")
            if data.get('AbstractURL'):
                results.append(f"Source: {data['AbstractURL']}")

        # Related topics
        if data.get('RelatedTopics'):
            results.append(f"\n🔗 Related Topics:")
            for i, topic in enumerate(data['RelatedTopics'][:num_results], 1):
                if isinstance(topic, dict) and 'Text' in topic:
                    text = topic['Text'][:200] + "..." if len(topic['Text']) > 200 else topic['Text']
                    results.append(f"{i}. {text}")
                    if 'FirstURL' in topic:
                        results.append(f"   URL: {topic['FirstURL']}")

        # Answer (if available)
        if data.get('Answer'):
            results.append(f"\n💡 Direct Answer: {data['Answer']}")

        if len(results) == 1:  # Only the header
            results.append("\nNo detailed results found. Try a more specific query.")

        return "\n".join(results)

    except Exception as e:
        return f"Error searching internet: {str(e)}"


# Export all tools for easy importing
__all__ = [
    'list_files',
    'search_files',
    'read_file',
    'write_file',
    'web_search'
]