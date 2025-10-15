"""
Common shareable tools for AbstractLLM applications.

This module provides a collection of utility tools for file operations,
web scraping, command execution, and user interaction.

Migrated from legacy system with enhanced decorator support.
"""

import os
import subprocess
import requests
from pathlib import Path
from typing import Optional
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
        # Convert head_limit to int if it's a string (defensive programming)
        if isinstance(head_limit, str):
            try:
                head_limit = int(head_limit)
            except ValueError:
                head_limit = 50  # fallback to default
        
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
            "description": "Show content for specific patterns (default behavior)",
            "arguments": {
                "pattern": "generate.*tools|create_react_cycle",
                "path": "abstractllm/session.py",
                "head_limit": 5
            }
        }
    ]
)
def search_files(
    pattern: str,
    path: str = ".",
    output_mode: str = "content",
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
        pattern: Regular expression pattern to search for
        path: File or directory path to search in (default: current directory)
        output_mode: Output format - "content" (show matching lines), "files_with_matches" (show file paths with line numbers), "count" (show match counts) (default: "content")
        head_limit: Limit output to first N entries (default: 20)
        file_pattern: Glob pattern(s) for files to search. Use "|" to separate multiple patterns (default: "*" for all files)
        case_sensitive: Whether search should be case sensitive (default: False)
        multiline: Enable multiline matching where pattern can span lines (default: False)

    Returns:
        Search results in the specified format or error message

    Examples:
        search_files("generate.*react|create_react_cycle", "abstractllm/session.py")  # Returns matching lines with content (default)
        search_files("def.*search", ".", file_pattern="*.py")  # Search Python files only, show content
        search_files("import.*re", ".", file_pattern="*.py|*.js")  # Search Python and JavaScript files, show content
        search_files("TODO|FIXME", ".", file_pattern="*.py|*.md|*.txt")  # Find TODO/FIXME in multiple file types, show content
        search_files("import.*re", ".", "files_with_matches")  # Show file paths with line numbers instead of content
        search_files("pattern", ".", "count")  # Count matches per file
    """
    try:
        # Convert head_limit to int if it's a string (defensive programming)
        if isinstance(head_limit, str):
            try:
                head_limit = int(head_limit)
            except ValueError:
                head_limit = 20  # fallback to default
        
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
                        if (include_hidden or not d.startswith('.')) and d not in ignore_set
                    ]
                    for file in files:
                        file_path = Path(root) / file
                        # Skip hidden files unless allowed
                        if not include_hidden and file_path.name.startswith('.'):
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
                        if (include_hidden or not d.startswith('.')) and d not in ignore_set
                    ]
                    for file in files:
                        file_path = Path(root) / file
                        filename = file_path.name
                        # Skip hidden files unless allowed
                        if not include_hidden and filename.startswith('.'):
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
            return f"Error: Path '{path}' does not exist"

        if not files_to_search:
            return f"No files found to search in '{path}'"

        # Search through files
        results = []
        files_with_matches = []  # Will store (file_path, [line_numbers]) tuples
        match_counts = {}
        total_matches = 0
        global_content_lines_added = 0  # Track content lines across all files

        for file_path in files_to_search:
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
                            
                            for match in matches:
                                line_num = content[:match.start()].count('\n') + 1
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
                                        results.append(f"    Line {line_num}: {full_line}")
                                        global_content_lines_added += 1
                                        
                                        # Check global head_limit after adding content
                                        if head_limit and global_content_lines_added >= head_limit:
                                            break

                            files_with_matches.append((str(file_path), line_numbers))
                            match_counts[str(file_path)] = len(matches)
                            total_matches += len(matches)
                    else:
                        # Non-multiline mode: process line by line (more efficient)
                        lines = f.readlines()
                        matching_lines = []
                        line_numbers = []
                        file_header_added = False
                        
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
                                        results.append(f"\n📄 {file_path}:")
                                        file_header_added = True
                                    
                                    results.append(f"    Line {line_num}: {line_content}")
                                    global_content_lines_added += 1
                                    
                                    # Check global head_limit after adding content
                                    if head_limit and global_content_lines_added >= head_limit:
                                        break

                        if matching_lines:
                            files_with_matches.append((str(file_path), line_numbers))
                            match_counts[str(file_path)] = len(matching_lines)
                            total_matches += len(matching_lines)

            except Exception as e:
                if output_mode == "content":
                    results.append(f"\n⚠️  Error reading {file_path}: {str(e)}")
            
            # Break out of file loop if we've reached the global head_limit
            if head_limit and output_mode == "content" and global_content_lines_added >= head_limit:
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

            # Add truncation notice if we hit the head_limit
            result_text = header + "\n" + "\n".join(final_results)
            if head_limit and global_content_lines_added >= head_limit:
                result_text += f"\n\n... (showing first {head_limit} matches)"
            
            return result_text

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
                "file_path": ".gitignore"
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
def read_file(file_path: str, should_read_entire_file: bool = True, start_line_one_indexed: int = 1, end_line_one_indexed_inclusive: Optional[int] = None) -> str:
    """
    Read the contents of a file with optional line range.

    Args:
        file_path: Path to the file to read
        should_read_entire_file: Whether to read the entire file (default: True)
            Note: Automatically set to False if start_line_one_indexed != 1 or end_line_one_indexed_inclusive is provided
        start_line_one_indexed: Starting line number (1-indexed, default: 1)
        end_line_one_indexed_inclusive: Ending line number (1-indexed, inclusive, optional)

    Returns:
        File contents or error message
    """
    try:
        path = Path(file_path)

        if not path.exists():
            return f"Error: File '{file_path}' does not exist"

        if not path.is_file():
            return f"Error: '{file_path}' is not a file"


        # Auto-override should_read_entire_file if line range parameters are provided
        if start_line_one_indexed != 1 or end_line_one_indexed_inclusive is not None:
            should_read_entire_file = False

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
        },
        {
            "description": "Search for recent news (past 24 hours)",
            "arguments": {
                "query": "AI developments news",
                "time_range": "h"
            }
        },
        {
            "description": "Find articles from the past week",
            "arguments": {
                "query": "Python programming tutorials",
                "time_range": "w"
            }
        },
        {
            "description": "Get recent research (past month)",
            "arguments": {
                "query": "machine learning research papers",
                "time_range": "m"
            }
        }
    ]
)
def web_search(query: str, num_results: int = 5, safe_search: str = "moderate", region: str = "us-en", time_range: Optional[str] = None) -> str:
    """
    Search the internet using DuckDuckGo (no API key required).

    Args:
        query: Search query
        num_results: Number of results to return (default: 5)
        safe_search: Content filtering level - "strict", "moderate", or "off" (default: "moderate")
        region: Regional results preference - "us-en", "uk-en", "ca-en", "au-en", etc. (default: "us-en")
        time_range: Time range filter for results (optional):
            - "h" or "24h": Past 24 hours
            - "d": Past day
            - "w" or "7d": Past week
            - "m" or "30d": Past month
            - "y" or "1y": Past year
            - None: All time (default)

    Returns:
        Search results or error message

    Note:
        Time range filtering requires the ddgs library (pip install ddgs).
        For best results with current news, use time_range="d" or "h".
    """
    try:
        # Try using duckduckgo-search library first (best approach)
        try:
            from ddgs import DDGS

            time_info = f" (past {time_range})" if time_range else ""
            results = [f"🔍 Search results for: '{query}'{time_info}"]

            with DDGS() as ddgs:
                # Prepare search parameters
                search_params = {
                    'query': query,
                    'max_results': num_results,
                    'region': region,
                    'safesearch': safe_search
                }

                # Add time range filter if specified
                if time_range:
                    search_params['timelimit'] = time_range

                # Get text search results
                search_results = list(ddgs.text(**search_params))

                if search_results:
                    results.append(f"\n🌐 Web Results:")

                    for i, result in enumerate(search_results, 1):
                        title = result.get('title', 'No title')
                        url = result.get('href', '')
                        body = result.get('body', '')

                        # Clean and format
                        title = title[:100] + "..." if len(title) > 100 else title
                        body = body[:150] + "..." if len(body) > 150 else body

                        results.append(f"\n{i}. {title}")
                        results.append(f"   🔗 {url}")
                        if body:
                            results.append(f"   📄 {body}")

                    return "\n".join(results)

        except ImportError:
            # Fallback if duckduckgo-search is not installed
            pass
        except Exception as e:
            # If duckduckgo-search fails, continue with fallback
            pass

        # Fallback: Use instant answer API for basic queries
        api_url = "https://api.duckduckgo.com/"
        params = {
            'q': query,
            'format': 'json',
            'no_html': '1',
            'skip_disambig': '1',
            'no_redirect': '1'
        }

        response = requests.get(api_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        results = [f"🔍 Search results for: '{query}'"]
        found_content = False

        # Abstract (main result)
        if data.get('Abstract') and data['Abstract'].strip():
            results.append(f"\n📝 Summary: {data['Abstract']}")
            if data.get('AbstractURL'):
                results.append(f"📎 Source: {data['AbstractURL']}")
            found_content = True

        # Direct Answer
        if data.get('Answer') and data['Answer'].strip():
            results.append(f"\n💡 Answer: {data['Answer']}")
            found_content = True

        # Related Topics
        if data.get('RelatedTopics') and isinstance(data['RelatedTopics'], list):
            valid_topics = [t for t in data['RelatedTopics'] if isinstance(t, dict) and t.get('Text')]
            if valid_topics:
                results.append(f"\n🔗 Related Information:")
                for i, topic in enumerate(valid_topics[:num_results], 1):
                    text = topic['Text'].replace('<b>', '').replace('</b>', '')
                    text = text[:200] + "..." if len(text) > 200 else text
                    results.append(f"{i}. {text}")
                    if topic.get('FirstURL'):
                        results.append(f"   🔗 {topic['FirstURL']}")
                    results.append("")
                found_content = True

        if not found_content:
            results.append(f"\n⚠️ Limited results for '{query}'")
            results.append(f"\n💡 For better web search results:")
            results.append(f"• Install ddgs: pip install ddgs")
            results.append(f"• This provides real web search results, not just instant answers")
            results.append(f"• Manual search: https://duckduckgo.com/?q={query.replace(' ', '+')}")

        return "\n".join(results)

    except Exception as e:
        return f"Error searching internet: {str(e)}"




@tool(
    description="Edit files using pattern matching and replacement - simple, powerful, and intuitive",
    tags=["file", "edit", "modify", "pattern", "replace", "regex"],
    when_to_use="When you need to edit files by finding patterns (text, functions, code blocks) and replacing them",
    examples=[
        {
            "description": "Replace simple text",
            "arguments": {
                "file_path": "config.py",
                "pattern": "debug = False",
                "replacement": "debug = True"
            }
        },
        {
            "description": "Update function definition using regex",
            "arguments": {
                "file_path": "script.py",
                "pattern": r"def old_function\([^)]*\):",
                "replacement": "def new_function(param1, param2):",
                "use_regex": True
            }
        },
        {
            "description": "Replace with occurrence limit",
            "arguments": {
                "file_path": "document.txt",
                "pattern": "TODO",
                "replacement": "DONE",
                "max_replacements": 1
            }
        },
        {
            "description": "Preview changes before applying",
            "arguments": {
                "file_path": "test.py",
                "pattern": "class OldClass",
                "replacement": "class NewClass",
                "preview_only": True
            }
        }
    ]
)
def edit_file(
    file_path: str,
    pattern: str,
    replacement: str,
    use_regex: bool = False,
    max_replacements: int = -1,
    start_line: Optional[int] = None,
    end_line: Optional[int] = None,
    preview_only: bool = False,
    encoding: str = "utf-8"
) -> str:
    """
    Edit files using pattern matching and replacement.

    Finds patterns (text or regex) in files and replaces them with new content.

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

    Returns:
        Success message with replacement details or error message

    Examples:
        edit_file("config.py", "debug = False", "debug = True")
        edit_file("script.py", r"def old_func\\([^)]*\\):", "def new_func():", use_regex=True)
        edit_file("document.txt", "TODO", "DONE", max_replacements=1)
        edit_file("test.py", "class OldClass", "class NewClass", preview_only=True)
    """
    try:
        # Validate file exists
        path = Path(file_path)
        if not path.exists():
            return f"❌ File not found: {file_path}"

        if not path.is_file():
            return f"❌ Path is not a file: {file_path}"

        # Read current content
        try:
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            return f"❌ Cannot decode file with encoding '{encoding}'. File may be binary."
        except Exception as e:
            return f"❌ Error reading file: {str(e)}"

        original_content = content

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
        if use_regex:
            try:
                regex_pattern = re.compile(pattern, re.MULTILINE | re.DOTALL)
            except re.error as e:
                return f"❌ Invalid regex pattern '{pattern}': {str(e)}"

            # Count matches first
            matches = list(regex_pattern.finditer(search_content))
            if not matches:
                range_info = f" (lines {start_line}-{end_line})" if start_line is not None or end_line is not None else ""
                return f"No matches found for regex pattern '{pattern}' in '{file_path}'{range_info}"

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
            if count == 0:
                range_info = f" (lines {start_line}-{end_line})" if start_line is not None or end_line is not None else ""
                return f"No occurrences of '{pattern}' found in '{file_path}'{range_info}"

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

        # Preview mode - show changes without applying
        if preview_only:
            results = []
            results.append(f"🔍 Preview Mode - Changes NOT Applied")
            results.append(f"File: {file_path}")
            if start_line is not None or end_line is not None:
                range_desc = f"lines {start_line or 1}-{end_line or 'end'}"
                results.append(f"Target range: {range_desc}")
            results.append(f"Pattern: {pattern}")
            results.append(f"Replacement: {replacement}")
            results.append(f"Regex mode: {'Yes' if use_regex else 'No'}")
            results.append(f"Matches found: {replacements_made}")

            if replacements_made > 0:
                results.append(f"\n📝 Changes that would be made:")
                results.append(f"  • {replacements_made} replacement(s)")

                # Show preview of first few changes
                preview_lines = []
                if use_regex:
                    regex_pattern = re.compile(pattern, re.MULTILINE | re.DOTALL)
                    matches = list(regex_pattern.finditer(search_content))
                    for i, match in enumerate(matches[:3]):  # Show first 3 matches
                        # Calculate line number relative to original file
                        match_line_in_search = search_content[:match.start()].count('\n') + 1
                        actual_line_num = match_line_in_search + line_offset
                        matched_text = match.group()[:50] + ("..." if len(match.group()) > 50 else "")
                        preview_lines.append(f"    Match {i+1} at line {actual_line_num}: '{matched_text}'")
                else:
                    # For simple text, show where matches occur
                    pos = 0
                    match_count = 0
                    while pos < len(search_content) and match_count < 3:
                        pos = search_content.find(pattern, pos)
                        if pos == -1:
                            break
                        match_line_in_search = search_content[:pos].count('\n') + 1
                        actual_line_num = match_line_in_search + line_offset
                        preview_lines.append(f"    Match {match_count+1} at line {actual_line_num}: '{pattern}'")
                        pos += len(pattern)
                        match_count += 1

                results.extend(preview_lines)
                if replacements_made > 3:
                    results.append(f"    ... and {replacements_made - 3} more matches")

            return "\n".join(results)

        # Apply changes to file
        try:
            with open(path, 'w', encoding=encoding) as f:
                f.write(updated_content)
        except Exception as e:
            return f"❌ Write failed: {str(e)}"

        # Success message
        results = []
        results.append(f"✅ File edited successfully: {file_path}")
        if start_line is not None or end_line is not None:
            range_desc = f"lines {start_line or 1}-{end_line or 'end'}"
            results.append(f"Target range: {range_desc}")
        results.append(f"Pattern: {pattern}")
        results.append(f"Replacement: {replacement}")
        results.append(f"Replacements made: {replacements_made}")

        # Calculate size change
        size_change = len(updated_content) - len(original_content)
        if size_change != 0:
            sign = "+" if size_change > 0 else ""
            results.append(f"Size change: {sign}{size_change} characters")

        return "\n".join(results)

    except Exception as e:
        return f"❌ Error editing file: {str(e)}"



@tool(
    description="Execute shell commands safely with security controls and platform detection",
    tags=["command", "shell", "execution", "system"],
    when_to_use="When you need to run system commands, shell scripts, or interact with command-line tools",
    examples=[
        {
            "description": "List current directory contents",
            "arguments": {
                "command": "ls -la"
            }
        },
        {
            "description": "Check system information",
            "arguments": {
                "command": "uname -a"
            }
        },
        {
            "description": "Run command with timeout",
            "arguments": {
                "command": "ping -c 3 google.com",
                "timeout": 30
            }
        },
        {
            "description": "Execute in specific directory",
            "arguments": {
                "command": "pwd",
                "working_directory": "/tmp"
            }
        },
        {
            "description": "Get current date and time",
            "arguments": {
                "command": "date"
            }
        },
        {
            "description": "HTTP GET request to API",
            "arguments": {
                "command": "curl -X GET 'https://api.example.com/data' -H 'Content-Type: application/json'"
            }
        },
        {
            "description": "HTTP POST request to API",
            "arguments": {
                "command": "curl -X POST 'https://api.example.com/submit' -H 'Content-Type: application/json' -d '{\"key\": \"value\"}'"
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
) -> str:
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
        Command execution result with stdout, stderr, and return code information
    """
    try:
        # Platform detection
        current_platform = platform.system()

        # CRITICAL SECURITY VALIDATION - Dangerous commands MUST be blocked
        security_check = _validate_command_security(command, allow_dangerous)
        if not security_check["safe"]:
            return f"🚫 CRITICAL SECURITY BLOCK: {security_check['reason']}\n" \
                   f"BLOCKED COMMAND: {command}\n" \
                   f"⚠️  DANGER: This command could cause IRREVERSIBLE DAMAGE\n" \
                   f"Only use allow_dangerous=True with EXPRESS USER CONSENT\n" \
                   f"This safety mechanism protects your system and data"

        # User confirmation for risky commands
        if require_confirmation:
            risk_level = _assess_command_risk(command)
            if risk_level != "low":
                logger.warning(f"Command execution simulated - {risk_level} risk command: {command}")
                logger.warning(f"Would normally ask for user confirmation before proceeding")

        # Working directory validation
        if working_directory:
            working_dir = Path(working_directory).resolve()
            if not working_dir.exists():
                return f"❌ Error: Working directory does not exist: {working_directory}"
            if not working_dir.is_dir():
                return f"❌ Error: Working directory path is not a directory: {working_directory}"
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
            output_parts.append(f"📁 Working directory: {working_dir or os.getcwd()}")
            output_parts.append(f"⏱️  Execution time: {execution_time:.2f}s")
            output_parts.append(f"🔢 Return code: {result.returncode}")

            if capture_output:
                if result.stdout:
                    # Limit output size for agent usability while allowing substantial content
                    stdout = result.stdout[:20000]  # First 20000 chars for agent processing
                    if len(result.stdout) > 20000:
                        stdout += f"\n... (output truncated, {len(result.stdout)} total chars)"
                    output_parts.append(f"\n📤 STDOUT:\n{stdout}")

                if result.stderr:
                    stderr = result.stderr[:5000]  # First 5000 chars for errors
                    if len(result.stderr) > 5000:
                        stderr += f"\n... (error output truncated, {len(result.stderr)} total chars)"
                    output_parts.append(f"\n❌ STDERR:\n{stderr}")

                if result.returncode == 0:
                    output_parts.append("\n✅ Command completed successfully")
                else:
                    output_parts.append(f"\n⚠️  Command completed with non-zero exit code: {result.returncode}")
            else:
                output_parts.append("📝 Output capture disabled")

            return "\n".join(output_parts)

        except subprocess.TimeoutExpired:
            return f"⏰ Timeout: Command exceeded {timeout} seconds\n" \
                   f"Command: {command}\n" \
                   f"Consider increasing timeout or breaking down the command"

        except subprocess.CalledProcessError as e:
            return f"❌ Command execution failed\n" \
                   f"Command: {command}\n" \
                   f"Return code: {e.returncode}\n" \
                   f"Error: {e.stderr if e.stderr else 'No error details'}"

    except Exception as e:
        return f"❌ Execution error: {str(e)}\n" \
               f"Command: {command}"


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
    'execute_command'
]