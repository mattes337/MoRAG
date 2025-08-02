"""Retry utilities for graph extraction operations."""

import asyncio
import random
import structlog
from typing import Callable, Any, Optional
from functools import wraps

logger = structlog.get_logger(__name__)


def is_retryable_error(error: Exception) -> bool:
    """Check if an error is retryable (rate limits, overload, etc.)."""
    error_str = str(error).lower()
    
    # Check for common retryable error patterns
    retryable_patterns = [
        "503",
        "429",
        "overload",
        "rate limit",
        "quota",
        "too many requests",
        "service unavailable",
        "temporarily unavailable",
        "server error",
        "resource_exhausted",
        "resource exhausted",
        "unavailable"
    ]
    
    return any(pattern in error_str for pattern in retryable_patterns)


async def retry_with_exponential_backoff(
    func: Callable,
    max_retries: int = 20,
    base_delay: float = 1.0,
    max_delay: float = 300.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    operation_name: str = "operation"
) -> Any:
    """
    Retry a function with exponential backoff for retryable errors.
    
    Args:
        func: Function to retry (can be sync or async)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter to delays
        operation_name: Name of the operation for logging
        
    Returns:
        Result of the function call
        
    Raises:
        The last exception if all retries fail
    """
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            # Call function (handle both sync and async)
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()
                
        except Exception as e:
            last_error = e
            
            # Check if this is a retryable error
            if not is_retryable_error(e):
                logger.error(
                    f"{operation_name} failed with non-retryable error",
                    attempt=attempt,
                    error=str(e),
                    error_type=type(e).__name__
                )
                raise e
            
            # If we've exhausted retries, raise the last error
            if attempt >= max_retries:
                logger.error(
                    f"{operation_name} failed after {max_retries} attempts",
                    error=str(e),
                    error_type=type(e).__name__
                )
                break
            
            # Calculate delay with exponential backoff
            delay = min(
                base_delay * (exponential_base ** (attempt - 1)),
                max_delay
            )
            
            # Add jitter to prevent thundering herd
            if jitter:
                delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
            
            logger.warning(
                f"{operation_name} failed with retryable error, retrying with exponential backoff",
                attempt=attempt,
                max_retries=max_retries,
                delay=delay,
                error=str(e),
                error_type=type(e).__name__
            )
            
            await asyncio.sleep(delay)
    
    # If we get here, all retries failed
    raise last_error


def retry_on_api_errors(
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    max_delay: Optional[float] = None,
    operation_name: str = "API operation"
):
    """
    Decorator for retrying functions on API errors with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts (uses config default if None)
        base_delay: Base delay in seconds (uses config default if None)
        max_delay: Maximum delay in seconds (uses config default if None)
        operation_name: Name of the operation for logging
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get retry configuration from settings
            from morag_core.config import settings
            
            effective_max_retries = max_retries if max_retries is not None else settings.entity_extraction_max_retries
            effective_base_delay = base_delay if base_delay is not None else settings.entity_extraction_retry_base_delay
            effective_max_delay = max_delay if max_delay is not None else settings.entity_extraction_retry_max_delay
            
            return await retry_with_exponential_backoff(
                lambda: func(*args, **kwargs),
                max_retries=effective_max_retries,
                base_delay=effective_base_delay,
                max_delay=effective_max_delay,
                jitter=settings.retry_jitter,
                operation_name=operation_name
            )
        return wrapper
    return decorator
