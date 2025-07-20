"""Comprehensive error handling for LLM interactions with automatic retry and fallback logic."""

import asyncio
import json
import logging
import structlog
from typing import Any, Dict, List, Optional, Union, Callable, TypeVar, Awaitable
from functools import wraps

from .json_parser import parse_llm_response_with_retry, JSONParsingError
from .llm_response_handler import LLMResponseError

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class LLMErrorHandler:
    """Comprehensive error handler for LLM interactions."""
    
    def __init__(
        self,
        max_llm_retries: int = 2,
        max_parse_retries: int = 3,
        retry_delay: float = 1.0,
        exponential_backoff: bool = True
    ):
        """Initialize the error handler.
        
        Args:
            max_llm_retries: Maximum number of LLM call retries
            max_parse_retries: Maximum number of parsing retries per response
            retry_delay: Base delay between retries in seconds
            exponential_backoff: Whether to use exponential backoff
        """
        self.max_llm_retries = max_llm_retries
        self.max_parse_retries = max_parse_retries
        self.retry_delay = retry_delay
        self.exponential_backoff = exponential_backoff
        self.logger = logger.bind(component="llm_error_handler")
    
    async def execute_with_retry(
        self,
        llm_func: Callable[[], Awaitable[str]],
        parser_func: Callable[[str], T],
        fallback_value: Optional[T] = None,
        context: str = "LLM operation",
        validator: Optional[Callable[[T], bool]] = None
    ) -> T:
        """Execute LLM function with comprehensive error handling and retry logic.
        
        Args:
            llm_func: Async function that calls LLM and returns raw response
            parser_func: Function that parses the LLM response
            fallback_value: Value to return if all attempts fail
            context: Context description for logging
            validator: Optional function to validate parsed result
            
        Returns:
            Parsed and validated result
            
        Raises:
            LLMResponseError: If all attempts fail and no fallback provided
        """
        last_error = None
        
        for llm_attempt in range(self.max_llm_retries + 1):
            try:
                # Call LLM
                self.logger.debug(
                    "Executing LLM call",
                    context=context,
                    llm_attempt=llm_attempt + 1,
                    max_attempts=self.max_llm_retries + 1
                )
                
                raw_response = await llm_func()
                
                # Try parsing with retries
                for parse_attempt in range(self.max_parse_retries):
                    try:
                        parsed_result = parser_func(raw_response)
                        
                        # Validate if validator provided
                        if validator and not validator(parsed_result):
                            raise LLMResponseError(f"Validation failed for {context}")
                        
                        self.logger.info(
                            "LLM operation successful",
                            context=context,
                            llm_attempt=llm_attempt + 1,
                            parse_attempt=parse_attempt + 1
                        )
                        return parsed_result
                        
                    except (JSONParsingError, json.JSONDecodeError, ValueError) as e:
                        last_error = e
                        self.logger.warning(
                            "Parsing attempt failed",
                            context=context,
                            llm_attempt=llm_attempt + 1,
                            parse_attempt=parse_attempt + 1,
                            max_parse_attempts=self.max_parse_retries,
                            error=str(e)
                        )
                        
                        if parse_attempt < self.max_parse_retries - 1:
                            # Preprocess response for next parsing attempt
                            raw_response = self._preprocess_for_retry(raw_response, parse_attempt)
                
                # All parsing attempts failed for this LLM response
                self.logger.warning(
                    "All parsing attempts failed for LLM response",
                    context=context,
                    llm_attempt=llm_attempt + 1,
                    error=str(last_error)
                )
                
            except Exception as e:
                last_error = e
                self.logger.warning(
                    "LLM call failed",
                    context=context,
                    llm_attempt=llm_attempt + 1,
                    max_attempts=self.max_llm_retries + 1,
                    error=str(e)
                )
            
            # Wait before next LLM retry (except on last attempt)
            if llm_attempt < self.max_llm_retries:
                delay = self._calculate_delay(llm_attempt)
                self.logger.debug(
                    "Waiting before LLM retry",
                    context=context,
                    delay=delay,
                    attempt=llm_attempt + 1
                )
                await asyncio.sleep(delay)
        
        # All attempts failed
        self.logger.error(
            "All LLM operation attempts failed",
            context=context,
            error=str(last_error)
        )
        
        if fallback_value is not None:
            self.logger.warning(
                "Using fallback value",
                context=context
            )
            return fallback_value
        
        raise LLMResponseError(f"LLM operation failed for {context}: {str(last_error)}")
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.exponential_backoff:
            return self.retry_delay * (2 ** attempt)
        return self.retry_delay
    
    def _preprocess_for_retry(self, response: str, attempt: int) -> str:
        """Preprocess response for retry attempt."""
        if attempt == 0:
            # First retry: normalize whitespace and quotes
            import re
            response = re.sub(r'\s+', ' ', response.strip())
            response = response.replace('"', '"').replace('"', '"')
            response = response.replace(''', "'").replace(''', "'")
        elif attempt == 1:
            # Second retry: remove non-printable characters
            import re
            response = re.sub(r'[^\x20-\x7E\n\r\t]', '', response)
            response = re.sub(r'\n+', '\n', response.strip())
        
        return response


def with_llm_error_handling(
    max_llm_retries: int = 2,
    max_parse_retries: int = 3,
    retry_delay: float = 1.0,
    exponential_backoff: bool = True,
    fallback_value: Any = None,
    context: Optional[str] = None
):
    """Decorator for LLM functions that provides comprehensive error handling.
    
    Args:
        max_llm_retries: Maximum number of LLM call retries
        max_parse_retries: Maximum number of parsing retries per response
        retry_delay: Base delay between retries in seconds
        exponential_backoff: Whether to use exponential backoff
        fallback_value: Value to return if all attempts fail
        context: Context description for logging
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            handler = LLMErrorHandler(
                max_llm_retries=max_llm_retries,
                max_parse_retries=max_parse_retries,
                retry_delay=retry_delay,
                exponential_backoff=exponential_backoff
            )
            
            func_context = context or func.__name__
            
            try:
                return await func(*args, **kwargs)
            except (JSONParsingError, LLMResponseError, json.JSONDecodeError) as e:
                logger.warning(
                    "Function failed with parsing/LLM error",
                    function=func.__name__,
                    context=func_context,
                    error=str(e)
                )
                
                if fallback_value is not None:
                    logger.info(
                        "Using decorator fallback value",
                        function=func.__name__,
                        context=func_context
                    )
                    return fallback_value
                
                raise LLMResponseError(f"Function {func.__name__} failed: {str(e)}")
        
        return wrapper
    return decorator


# Convenience function for simple JSON parsing with retries
def parse_json_with_retries(
    response: str,
    fallback_value: Optional[Dict[str, Any]] = None,
    context: str = "JSON parsing"
) -> Dict[str, Any]:
    """Parse JSON with comprehensive error handling and retries.
    
    Args:
        response: Raw JSON response string
        fallback_value: Fallback value if parsing fails
        context: Context for logging
        
    Returns:
        Parsed JSON data or fallback value
    """
    try:
        return parse_llm_response_with_retry(
            response,
            fallback_value=fallback_value,
            context=context
        )
    except JSONParsingError as e:
        logger.error(
            "JSON parsing failed with retries",
            context=context,
            error=str(e)
        )
        
        if fallback_value is not None:
            return fallback_value
        
        raise


# Global error handler instance
_global_handler = LLMErrorHandler()


async def execute_llm_with_retry(
    llm_func: Callable[[], Awaitable[str]],
    parser_func: Callable[[str], T],
    fallback_value: Optional[T] = None,
    context: str = "LLM operation"
) -> T:
    """Execute LLM function with global error handler.
    
    Args:
        llm_func: Async function that calls LLM
        parser_func: Function that parses the response
        fallback_value: Fallback value if all attempts fail
        context: Context for logging
        
    Returns:
        Parsed result or fallback value
    """
    return await _global_handler.execute_with_retry(
        llm_func=llm_func,
        parser_func=parser_func,
        fallback_value=fallback_value,
        context=context
    )
