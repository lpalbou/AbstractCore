"""
Tool call tag integration for generation methods.

This module provides clean integration points for tool call tag rewriting
in both generate() and session.generate() methods.
"""

from typing import Optional, Union, Iterator, Dict, Any
from .tag_rewriter import ToolCallTagRewriter, ToolCallTags, create_tag_rewriter
from ..core.types import GenerateResponse


def apply_tool_call_tag_rewriting(
    response: Union[GenerateResponse, Iterator[GenerateResponse]],
    tool_call_tags: Optional[Union[str, ToolCallTags, ToolCallTagRewriter]] = None
) -> Union[GenerateResponse, Iterator[GenerateResponse]]:
    """
    Apply tool call tag rewriting to a response.
    
    Args:
        response: Response to rewrite (single or streaming)
        tool_call_tags: Tag configuration (string, ToolCallTags, or ToolCallTagRewriter)
        
    Returns:
        Response with rewritten tool call tags
    """
    if not tool_call_tags:
        return response
    
    # Convert string to rewriter
    if isinstance(tool_call_tags, str):
        rewriter = create_tag_rewriter(tool_call_tags)
    elif isinstance(tool_call_tags, ToolCallTags):
        rewriter = ToolCallTagRewriter(tool_call_tags)
    elif isinstance(tool_call_tags, ToolCallTagRewriter):
        rewriter = tool_call_tags
    else:
        raise ValueError(f"Invalid tool_call_tags type: {type(tool_call_tags)}")
    
    # Handle streaming vs non-streaming
    if isinstance(response, Iterator):
        return _rewrite_streaming_response(response, rewriter)
    else:
        return _rewrite_single_response(response, rewriter)


def _rewrite_single_response(response: GenerateResponse, rewriter: ToolCallTagRewriter) -> GenerateResponse:
    """Rewrite tool call tags in a single response."""
    if not hasattr(response, 'content') or not response.content:
        return response
    
    # Rewrite the content
    rewritten_content = rewriter.rewrite_text(response.content)
    
    # Create new response with rewritten content
    new_response = GenerateResponse(
        content=rewritten_content,
        model=getattr(response, 'model', None),
        usage=getattr(response, 'usage', None),
        metadata=getattr(response, 'metadata', None),
        tool_calls=getattr(response, 'tool_calls', None),
        finish_reason=getattr(response, 'finish_reason', None),
        raw_response=getattr(response, 'raw_response', None)
    )
    
    return new_response


def _rewrite_streaming_response(
    response_iterator: Iterator[GenerateResponse], 
    rewriter: ToolCallTagRewriter
) -> Iterator[GenerateResponse]:
    """Rewrite tool call tags in a streaming response."""
    buffer = ""
    
    for chunk in response_iterator:
        if not hasattr(chunk, 'content') or not chunk.content:
            yield chunk
            continue
        
        # Rewrite the chunk with buffer for handling split tool calls
        rewritten_content, buffer = rewriter.rewrite_streaming_chunk(chunk.content, buffer)
        
        # Create new chunk with rewritten content
        new_chunk = GenerateResponse(
            content=rewritten_content,
            model=getattr(chunk, 'model', None),
            usage=getattr(chunk, 'usage', None),
            metadata=getattr(chunk, 'metadata', None),
            tool_calls=getattr(chunk, 'tool_calls', None),
            finish_reason=getattr(chunk, 'finish_reason', None),
            raw_response=getattr(chunk, 'raw_response', None)
        )
        
        yield new_chunk
    
    # Handle any remaining buffer
    if buffer:
        final_chunk = GenerateResponse(
            content=buffer,
            model=getattr(chunk, 'model', None) if 'chunk' in locals() else None,
            usage=getattr(chunk, 'usage', None) if 'chunk' in locals() else None,
            metadata=getattr(chunk, 'metadata', None) if 'chunk' in locals() else None,
            tool_calls=getattr(chunk, 'tool_calls', None) if 'chunk' in locals() else None,
            finish_reason=getattr(chunk, 'finish_reason', None) if 'chunk' in locals() else None,
            raw_response=getattr(chunk, 'raw_response', None) if 'chunk' in locals() else None
        )
        yield final_chunk


def get_tool_call_tags_from_kwargs(kwargs: Dict[str, Any]) -> Optional[Union[str, ToolCallTags, ToolCallTagRewriter]]:
    """
    Extract tool_call_tags from kwargs and remove it.
    
    Args:
        kwargs: Keyword arguments
        
    Returns:
        Tool call tags configuration or None
    """
    return kwargs.pop('tool_call_tags', None)