"""Utilities package for morag-graph."""

from .id_generation import (
    IDCollisionDetector,
    IDCollisionError,
    IDValidationError,
    IDValidator,
    UnifiedIDGenerator,
)
from .retry_utils import (
    is_retryable_error,
    retry_on_api_errors,
    retry_with_exponential_backoff,
)

# LLM response parser moved to agents.base.response_parser
# Kept here for backward compatibility only
try:
    from .llm_response_parser import (
        LLMResponseParseError,
        clean_json_response,
        extract_json_from_text,
        parse_json_response,
    )
except ImportError:
    # If the local parser is removed, provide fallback imports
    try:
        from agents.base import LLMResponseParseError, LLMResponseParser

        parse_json_response = LLMResponseParser.parse_json_response
        extract_json_from_text = LLMResponseParser.extract_json_from_text
        clean_json_response = LLMResponseParser.clean_response
    except ImportError:
        # No parser available
        parse_json_response = None
        extract_json_from_text = None
        clean_json_response = None
        LLMResponseParseError = Exception

__all__ = [
    "UnifiedIDGenerator",
    "IDValidator",
    "IDCollisionDetector",
    "IDValidationError",
    "IDCollisionError",
    "retry_with_exponential_backoff",
    "retry_on_api_errors",
    "is_retryable_error",
    "parse_json_response",
    "extract_json_from_text",
    "clean_json_response",
    "LLMResponseParseError",
]
