"""
Version management for AbstractCore.

This module provides the package version as a static constant that serves as the
single source of truth for the Python code. The version is also maintained in
pyproject.toml for packaging, requiring manual synchronization during releases.

This approach ensures reliable version access in all deployment scenarios,
including when the package is installed from PyPI where pyproject.toml is not available.
"""

# Package version - update this when releasing new versions
# This must be manually synchronized with the version in pyproject.toml
__version__ = "2.3.8"
