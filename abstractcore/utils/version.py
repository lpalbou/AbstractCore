"""
Version management for AbstractCore.

This module provides the package version as a static constant that serves as the
single source of truth. Packaging reads the version from this module via
`[tool.setuptools.dynamic]` in `pyproject.toml`.

This approach ensures reliable version access in all deployment scenarios,
including when the package is installed from PyPI where pyproject.toml is not available.
"""

# Package version - update this when releasing new versions
__version__ = "2.11.9"
