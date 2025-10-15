"""
Self-correction utilities for fixing malformed LLM outputs

Provides clean, efficient methods to attempt fixing common LLM output issues
before giving up on parsing.
"""

import json
import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def fix_json(text: str) -> Optional[str]:
    """
    Attempt to fix malformed JSON using multiple strategies

    Args:
        text: The potentially malformed JSON text

    Returns:
        str: Fixed JSON text, or None if all strategies failed
    """
    if not text or not text.strip():
        return None

    # Strategy 1: Extract JSON from text with extra content
    extracted = _extract_json_from_text(text)
    if extracted and _is_valid_json(extracted):
        logger.info("JSON self-fix: Extracted JSON from surrounding text")
        return extracted

    # Strategy 2: Fix common formatting issues
    fixed = _fix_common_issues(text)
    if fixed and _is_valid_json(fixed):
        logger.info("JSON self-fix: Fixed common formatting issues")
        return fixed

    # Strategy 3: Repair truncated JSON
    repaired = _repair_truncated(text)
    if repaired and _is_valid_json(repaired):
        logger.info("JSON self-fix: Repaired truncated JSON")
        return repaired

    # Strategy 4: Create minimal structure from extractable content
    minimal = _create_minimal_structure(text)
    if minimal and _is_valid_json(minimal):
        logger.info("JSON self-fix: Created minimal structure from content")
        return minimal

    return None


def _extract_json_from_text(text: str) -> Optional[str]:
    """Extract JSON from text that may contain other content"""
    if not text or not text.strip():
        return None

    # Strategy 1: Simple brace counting from first {
    # This handles cases where JSON is embedded in explanatory text
    first_brace = text.find('{')
    if first_brace != -1:
        candidate = _extract_json_by_brace_counting(text[first_brace:])
        if candidate and _is_valid_json(candidate):
            return candidate

    # Strategy 2: Regex patterns (fallback)
    # Improved patterns that handle nested braces better
    patterns = [
        r'\{.*?"@context".*?"@graph".*?\}',  # JSON-LD (simplified)
        r'\{.*?"@context".*?\}',             # JSON with @context
        r'\{.*?"@graph".*?\}',               # JSON with @graph
        r'\{.*?\}',                          # Any JSON object
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            candidate = _extract_json_by_brace_counting(match)
            if candidate and _is_valid_json(candidate):
                return candidate

    return None


def _extract_json_by_brace_counting(text: str) -> Optional[str]:
    """Extract complete JSON using brace counting"""
    if not text or not text.strip():
        return None

    # Find first opening brace
    start = text.find('{')
    if start == -1:
        return None

    # Count braces to find matching closing brace
    brace_count = 0
    for i, char in enumerate(text[start:], start):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                return text[start:i+1]

    # If we get here, braces weren't balanced
    return None


def _fix_common_issues(text: str) -> Optional[str]:
    """Fix common JSON formatting problems"""
    fixed = text.strip()

    # Extract JSON bounds
    if '{' in fixed and '}' in fixed:
        start = fixed.find('{')
        end = fixed.rfind('}')
        fixed = fixed[start:end+1]

    # Apply common fixes (conservative approach)
    fixes = [
        (r',(\s*[}\]])', r'\1'),                              # Remove trailing commas
        (r"'([^']*)'", r'"\1"'),                            # Single to double quotes (simple cases)
    ]

    for pattern, replacement in fixes:
        fixed = re.sub(pattern, replacement, fixed)

    return fixed if fixed != text else None


def _repair_truncated(text: str) -> Optional[str]:
    """Repair truncated JSON by adding missing closing braces"""
    if not text.strip().startswith('{'):
        return None

    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')

    if open_braces <= close_braces:
        return None

    repaired = text.strip()
    repaired += ']' * (open_brackets - close_brackets)
    repaired += '}' * (open_braces - close_braces)

    return repaired


def _create_minimal_structure(text: str) -> Optional[str]:
    """Create minimal JSON-LD structure from extractable content"""
    # Extract potential entities
    entity_patterns = [
        r'\b[A-Z][a-zA-Z]+\b',  # Capitalized words
        r'\b\w+\.com\b',        # Websites
        r'\b[A-Z]{2,}\b',       # Acronyms
    ]

    found_entities = set()
    for pattern in entity_patterns:
        matches = re.findall(pattern, text)
        found_entities.update(matches[:5])  # Limit to 5

    if not found_entities:
        return None

    graph = []
    for entity in list(found_entities):
        graph.append({
            "@id": f"e:{entity.lower()}",
            "@type": "s:Thing",
            "s:name": entity,
            "confidence": 0.5
        })

    minimal_json = {
        "@context": {
            "s": "https://schema.org/",
            "e": "http://example.org/entity/",
            "confidence": "http://example.org/confidence"
        },
        "@graph": graph
    }

    return json.dumps(minimal_json)


def _is_valid_json(text: str) -> bool:
    """Check if text is valid JSON"""
    if not text or not text.strip():
        return False

    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False