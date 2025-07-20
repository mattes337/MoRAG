"""Enhanced LLM response handling with robust error recovery and retry logic."""

import asyncio
import json
import logging
import structlog
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Generic
from functools import wraps

from .json_parser import parse_llm_response_with_retry, JSONParsingError

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class LLMResponseError(Exception):
    """Custom exception for LLM response handling errors."""
    pass


class StructuredResponseHandler(Generic[T]):
    """Handler for structured LLM responses with validation and error recovery."""
    
    def __init__(
        self,
        response_type: str = "JSON",
        max_parse_attempts: int = 3,
        max_llm_retries: int = 2,
        fallback_factory: Optional[Callable[[], T]] = None
    ):
        """Initialize the response handler.
        
        Args:
            response_type: Type of response expected (JSON, etc.)
            max_parse_attempts: Maximum attempts to parse a single response
            max_llm_retries: Maximum attempts to regenerate response from LLM
            fallback_factory: Function to create fallback response
        """
        self.response_type = response_type
        self.max_parse_attempts = max_parse_attempts
        self.max_llm_retries = max_llm_retries
        self.fallback_factory = fallback_factory
        self.logger = logger.bind(component="llm_response_handler")
    
    async def handle_response(
        self,
        llm_response: str,
        context: str = "LLM response",
        validator: Optional[Callable[[Any], bool]] = None,
        llm_regenerate_func: Optional[Callable[[], str]] = None
    ) -> T:
        """Handle LLM response with comprehensive error recovery.
        
        Args:
            llm_response: Raw LLM response string
            context: Context description for logging
            validator: Optional function to validate parsed response
            llm_regenerate_func: Optional async function to regenerate response
            
        Returns:
            Parsed and validated response
            
        Raises:
            LLMResponseError: If all recovery attempts fail
        """
        last_error = None
        
        # Try parsing the current response
        for parse_attempt in range(self.max_parse_attempts):
            try:
                parsed = await self._parse_response(llm_response, context, parse_attempt)
                
                # Validate if validator provided
                if validator and not validator(parsed):
                    raise LLMResponseError(f"Response validation failed: {context}")
                
                self.logger.info(
                    "Successfully parsed LLM response",
                    context=context,
                    parse_attempt=parse_attempt + 1
                )
                return parsed
                
            except (JSONParsingError, LLMResponseError) as e:
                last_error = e
                self.logger.warning(
                    "LLM response parsing failed",
                    context=context,
                    parse_attempt=parse_attempt + 1,
                    max_attempts=self.max_parse_attempts,
                    error=str(e)
                )
        
        # If we have an LLM regeneration function, try regenerating the response
        if llm_regenerate_func:
            for retry_attempt in range(self.max_llm_retries):
                try:
                    self.logger.info(
                        "Regenerating LLM response",
                        context=context,
                        retry_attempt=retry_attempt + 1,
                        max_retries=self.max_llm_retries
                    )
                    
                    new_response = await llm_regenerate_func()
                    parsed = await self._parse_response(new_response, context, 0)
                    
                    if validator and not validator(parsed):
                        raise LLMResponseError(f"Regenerated response validation failed: {context}")
                    
                    self.logger.info(
                        "Successfully parsed regenerated LLM response",
                        context=context,
                        retry_attempt=retry_attempt + 1
                    )
                    return parsed
                    
                except Exception as e:
                    last_error = e
                    self.logger.warning(
                        "LLM response regeneration failed",
                        context=context,
                        retry_attempt=retry_attempt + 1,
                        max_retries=self.max_llm_retries,
                        error=str(e)
                    )
        
        # Try fallback if available
        if self.fallback_factory:
            try:
                fallback_response = self.fallback_factory()
                self.logger.warning(
                    "Using fallback response",
                    context=context,
                    error=str(last_error)
                )
                return fallback_response
            except Exception as e:
                self.logger.error(
                    "Fallback response creation failed",
                    context=context,
                    error=str(e)
                )
        
        # All attempts failed
        self.logger.error(
            "All LLM response handling attempts failed",
            context=context,
            error=str(last_error)
        )
        raise LLMResponseError(f"Failed to handle LLM response for {context}: {str(last_error)}")
    
    async def _parse_response(self, response: str, context: str, attempt: int) -> Any:
        """Parse response with attempt-specific preprocessing."""
        try:
            # Apply attempt-specific preprocessing
            preprocessed = self._preprocess_response(response, attempt)
            
            # Parse using enhanced JSON parser
            parsed = parse_llm_response_with_retry(
                preprocessed,
                fallback_value=None,
                max_attempts=1,  # We handle retries at higher level
                context=context
            )
            
            return parsed
            
        except Exception as e:
            raise LLMResponseError(f"Response parsing failed: {str(e)}")
    
    def _preprocess_response(self, response: str, attempt: int) -> str:
        """Preprocess response based on attempt number."""
        if attempt == 0:
            # First attempt: minimal preprocessing
            return response.strip()
        elif attempt == 1:
            # Second attempt: normalize whitespace and quotes
            import re
            response = re.sub(r'\s+', ' ', response.strip())
            response = response.replace('"', '"').replace('"', '"')
            return response
        else:
            # Third attempt: aggressive cleaning
            import re
            response = re.sub(r'[^\x20-\x7E\n\r\t]', '', response)
            response = re.sub(r'\n+', '\n', response.strip())
            return response


def with_structured_response_handling(
    response_type: str = "JSON",
    max_parse_attempts: int = 3,
    max_llm_retries: int = 2,
    fallback_factory: Optional[Callable[[], Any]] = None
):
    """Decorator for functions that handle LLM responses.
    
    Args:
        response_type: Type of response expected
        max_parse_attempts: Maximum attempts to parse a single response
        max_llm_retries: Maximum attempts to regenerate response
        fallback_factory: Function to create fallback response
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            handler = StructuredResponseHandler(
                response_type=response_type,
                max_parse_attempts=max_parse_attempts,
                max_llm_retries=max_llm_retries,
                fallback_factory=fallback_factory
            )
            
            try:
                return await func(*args, **kwargs)
            except (JSONParsingError, LLMResponseError) as e:
                # If the function failed due to parsing, try to handle it
                logger.warning(
                    "Function failed due to LLM response parsing error",
                    function=func.__name__,
                    error=str(e)
                )
                
                if fallback_factory:
                    return fallback_factory()
                raise
        
        return wrapper
    return decorator


# Convenience functions for common use cases
async def parse_json_response_safely(
    response: str,
    context: str = "JSON response",
    fallback: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Safely parse JSON response with fallback."""
    handler = StructuredResponseHandler(
        response_type="JSON",
        fallback_factory=lambda: fallback or {}
    )
    
    return await handler.handle_response(response, context)


async def parse_list_response_safely(
    response: str,
    context: str = "List response",
    fallback: Optional[List[Any]] = None
) -> List[Any]:
    """Safely parse list response with fallback."""
    handler = StructuredResponseHandler(
        response_type="JSON",
        fallback_factory=lambda: fallback or []
    )
    
    return await handler.handle_response(response, context)
