"""
AbstractCore - Core abstractions and interfaces

This module provides the fundamental building blocks for AbstractCore:
- Factory functions for creating LLM providers
- Session management for conversation tracking
- Type definitions and interfaces
- Provider abstractions
"""

from .factory import create_llm
from .session import BasicSession
from .cached_session import CachedSession
from .types import GenerateResponse, Message
from .multimodal_generation import (
    GeneratedItem,
    GeneratedResource,
    GenerationIssue,
    MultimodalGenerateResponse,
)
from .output_specs import (
    GenerationOutputSpec,
    is_output_request,
    normalize_output_spec,
    normalize_output_specs,
    output_has_generated_media,
    output_requires_non_chat_dispatch,
    strip_runtime_output_metadata,
)
from .generate_contract import (
    GenerateRequest,
    ResolvedGenerateRoute,
    ResolvedGenerateRouteEntry,
    normalize_generate_request,
    resolve_capability_default_route,
    resolve_generate_route,
)
from .enums import ModelParameter, ModelCapability, MessageRole
from .interface import AbstractCoreInterface
from .bloc_kv import (
    BlocDeleteResult,
    BlocKVArtifactManifest,
    BlocKVArtifactInUseError,
    BlocKVCompileResult,
    BlocKVDeleteResult,
    BlocKVLoadResult,
    compile_bloc_kv_artifact,
    delete_bloc,
    delete_bloc_kv_artifact,
    ensure_bloc_kv_artifact,
    find_bloc_kv_live_bindings,
    list_bloc_kv_artifacts,
    load_bloc_kv_artifact,
    prune_bloc_kv_artifacts,
    read_bloc_kv_manifest,
)

__all__ = [
    'create_llm',
    'BasicSession',
    'CachedSession',
    'GenerateResponse',
    'GenerationOutputSpec',
    'GeneratedItem',
    'GeneratedResource',
    'GenerateRequest',
    'GenerationIssue',
    'MultimodalGenerateResponse',
    'is_output_request',
    'normalize_output_spec',
    'normalize_output_specs',
    'normalize_generate_request',
    'output_has_generated_media',
    'output_requires_non_chat_dispatch',
    'ResolvedGenerateRoute',
    'ResolvedGenerateRouteEntry',
    'resolve_capability_default_route',
    'resolve_generate_route',
    'strip_runtime_output_metadata',
    'Message',
    'ModelParameter',
    'ModelCapability',
    'MessageRole',
    'AbstractCoreInterface',
    'BlocDeleteResult',
    'BlocKVArtifactManifest',
    'BlocKVArtifactInUseError',
    'BlocKVCompileResult',
    'BlocKVDeleteResult',
    'BlocKVLoadResult',
    'compile_bloc_kv_artifact',
    'delete_bloc',
    'delete_bloc_kv_artifact',
    'ensure_bloc_kv_artifact',
    'find_bloc_kv_live_bindings',
    'list_bloc_kv_artifacts',
    'load_bloc_kv_artifact',
    'prune_bloc_kv_artifacts',
    'read_bloc_kv_manifest',
]
