"""
Utility functions for AbstractCore.
"""

from .file_filters import (
    extensions_for_family,
    file_matches_filters,
    guess_file_family,
    normalize_extensions,
)
from .workspace_paths import (
    WorkspacePathError,
    WorkspacePathResolution,
    build_workspace_mounts,
    is_under_path,
    resolve_no_strict,
    resolve_workspace_path,
    slug_workspace_mount_name,
)
from .structured_logging import configure_logging, get_logger, capture_session
from .version import __version__
from .token_utils import (
    TokenUtils,
    count_tokens,
    estimate_tokens,
    count_tokens_precise,
    TokenCountMethod,
    ContentType
)
from .message_preprocessor import MessagePreprocessor, parse_files, has_files
from .trace_export import export_traces, summarize_traces
from .data_registry import (
    DATA_HOME_KINDS,
    DataHome,
    DataRegistryError,
    data_home_size,
    get_data_home,
    list_data_homes,
    purge_data_home,
    register_core_data_homes,
    register_data_home,
    registry_path,
    unregister_data_home,
)

__all__ = [
    'configure_logging',
    'get_logger',
    'capture_session',
    '__version__',
    'WorkspacePathError',
    'WorkspacePathResolution',
    'build_workspace_mounts',
    'extensions_for_family',
    'file_matches_filters',
    'guess_file_family',
    'is_under_path',
    'normalize_extensions',
    'resolve_no_strict',
    'resolve_workspace_path',
    'slug_workspace_mount_name',
    'TokenUtils',
    'count_tokens',
    'estimate_tokens',
    'count_tokens_precise',
    'TokenCountMethod',
    'ContentType',
    'MessagePreprocessor',
    'parse_files',
    'has_files',
    'export_traces',
    'summarize_traces',
    'DATA_HOME_KINDS',
    'DataHome',
    'DataRegistryError',
    'data_home_size',
    'get_data_home',
    'list_data_homes',
    'purge_data_home',
    'register_core_data_homes',
    'register_data_home',
    'registry_path',
    'unregister_data_home',
]
