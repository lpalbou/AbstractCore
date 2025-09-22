"""
Common tools for AbstractLLM - adapted from original implementation.
"""

import os
import json
import subprocess
import platform
from pathlib import Path
from typing import List, Optional, Dict, Any, Union
import fnmatch


def list_files(directory: str = ".", pattern: str = "*", recursive: bool = False,
               include_hidden: bool = False, limit: Optional[int] = 50) -> str:
    """
    List files in a directory with pattern matching.

    Args:
        directory: Directory to list files from
        pattern: Glob pattern(s) to match (use | for multiple)
        recursive: Search recursively
        include_hidden: Include hidden files
        limit: Maximum files to return

    Returns:
        Formatted file listing or error message
    """
    try:
        dir_path = Path(directory)

        if not dir_path.exists():
            return f"Error: Directory '{directory}' does not exist"

        if not dir_path.is_dir():
            return f"Error: '{directory}' is not a directory"

        # Split patterns
        patterns = [p.strip() for p in pattern.split('|')]

        # Collect files
        all_files = []
        if recursive:
            for root, dirs, files in os.walk(dir_path):
                for f in files:
                    file_path = Path(root) / f
                    all_files.append(file_path)
        else:
            all_files = list(dir_path.iterdir())

        # Filter by pattern (case-insensitive)
        matched = []
        for file_path in all_files:
            if file_path.is_file():
                name = file_path.name

                # Skip hidden files if not requested
                if not include_hidden and name.startswith('.'):
                    continue

                # Check patterns
                for pat in patterns:
                    if fnmatch.fnmatch(name.lower(), pat.lower()):
                        matched.append(str(file_path))
                        break

        if not matched:
            return f"No files found matching '{pattern}' in '{directory}'"

        # Sort by modification time
        matched.sort(key=lambda f: Path(f).stat().st_mtime if Path(f).exists() else 0,
                    reverse=True)

        # Apply limit
        total = len(matched)
        if limit and len(matched) > limit:
            matched = matched[:limit]
            header = f"Files in {directory} (showing {limit} of {total}):\n"
        else:
            header = f"Files in {directory} ({total} files):\n"

        # Format output
        result = [header]
        for f in matched:
            rel_path = Path(f).relative_to(dir_path) if dir_path != Path('.') else Path(f)
            result.append(f"  {rel_path}")

        return "\n".join(result)

    except Exception as e:
        return f"Error listing files: {str(e)}"


def read_file(file_path: str, encoding: str = "utf-8") -> str:
    """
    Read contents of a file.

    Args:
        file_path: Path to file
        encoding: File encoding

    Returns:
        File contents or error message
    """
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File '{file_path}' not found"
    except Exception as e:
        return f"Error reading file: {str(e)}"


def write_file(file_path: str, content: str, encoding: str = "utf-8") -> str:
    """
    Write content to a file.

    Args:
        file_path: Path to file
        content: Content to write
        encoding: File encoding

    Returns:
        Success message or error
    """
    try:
        with open(file_path, 'w', encoding=encoding) as f:
            f.write(content)
        return f"Successfully wrote to '{file_path}'"
    except Exception as e:
        return f"Error writing file: {str(e)}"


def run_command(command: str, shell: bool = True, timeout: Optional[int] = 30) -> str:
    """
    Execute a shell command.

    Args:
        command: Command to execute
        shell: Use shell execution
        timeout: Command timeout in seconds

    Returns:
        Command output or error
    """
    try:
        result = subprocess.run(
            command,
            shell=shell,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]: {result.stderr}"

        return output if output else "Command executed successfully (no output)"

    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout} seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"


def get_system_info() -> str:
    """
    Get system information.

    Returns:
        System information string
    """
    info = []
    info.append(f"Platform: {platform.platform()}")
    info.append(f"Python: {platform.python_version()}")
    info.append(f"Machine: {platform.machine()}")
    info.append(f"Processor: {platform.processor()}")

    try:
        import psutil
        info.append(f"CPU Count: {psutil.cpu_count()}")
        info.append(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    except ImportError:
        pass

    return "\n".join(info)


# Tool definitions for LLM providers
COMMON_TOOLS = [
    {
        "name": "list_files",
        "description": "List files in a directory with pattern matching",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "Directory path to list files from",
                    "default": "."
                },
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern(s) to match (use | for multiple)",
                    "default": "*"
                },
                "recursive": {
                    "type": "boolean",
                    "description": "Search recursively in subdirectories",
                    "default": False
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files",
                    "default": False
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of files to return",
                    "default": 50
                }
            },
            "required": []
        }
    },
    {
        "name": "read_file",
        "description": "Read contents of a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to read"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding",
                    "default": "utf-8"
                }
            },
            "required": ["file_path"]
        }
    },
    {
        "name": "write_file",
        "description": "Write content to a file",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file to write"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding",
                    "default": "utf-8"
                }
            },
            "required": ["file_path", "content"]
        }
    },
    {
        "name": "run_command",
        "description": "Execute a shell command",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 30
                }
            },
            "required": ["command"]
        }
    },
    {
        "name": "get_system_info",
        "description": "Get system information",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]


# Tool executor
def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """
    Execute a tool by name with arguments.

    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments

    Returns:
        Tool execution result
    """
    tools = {
        "list_files": list_files,
        "read_file": read_file,
        "write_file": write_file,
        "run_command": run_command,
        "get_system_info": get_system_info
    }

    if tool_name not in tools:
        return f"Error: Unknown tool '{tool_name}'"

    try:
        tool_func = tools[tool_name]
        return tool_func(**arguments)
    except TypeError as e:
        return f"Error: Invalid arguments for {tool_name}: {str(e)}"
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"

