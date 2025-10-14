"""
Version management for AbstractCore.

This module provides a single source of truth for the package version,
reading it dynamically from pyproject.toml to avoid version synchronization issues.
"""

import os
import tomllib
from pathlib import Path


def get_version() -> str:
    """
    Get the package version from pyproject.toml.
    
    Returns:
        str: The package version
        
    Raises:
        FileNotFoundError: If pyproject.toml is not found
        KeyError: If version is not found in pyproject.toml
    """
    # Get the project root directory (where pyproject.toml is located)
    # When imported as a module, we need to go up from abstractllm/utils/ to the project root
    current_dir = Path(__file__).parent  # abstractllm/utils/
    project_root = current_dir.parent.parent  # Go up two levels: utils -> abstractllm -> project_root
    pyproject_path = project_root / "pyproject.toml"
    
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")
    
    # Read and parse pyproject.toml
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    
    try:
        return data["project"]["version"]
    except KeyError:
        raise KeyError("Version not found in pyproject.toml [project] section")


# Set the version as a module-level constant
__version__ = get_version()
