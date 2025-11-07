"""Enhanced retry utilities specifically for handling quota exhaustion."""

import asyncio
import random
import re
import time
from functools import wraps
from typing import Callable, Any, Optional, Dict, Union
import structlog

logger = structlog.get_logger(__name__)


class QuotaExhaustionError(Exception):
    """Exception raised when API quota is exhausted."""

    def __init__(self, message: str, retry_after: Optional[int] = None, quota_type: Optional[str] = None):
        super().__init__(message)
        self.retry_after = retry_after
        self.quota_type = quota_type


def parse_quota_error(error: Exception) -> Optional[Dict[str, Any]]:
    """Parse quota error to extract retry information.

    Args:
        error: Exception to parse

    Returns:
        Dictionary with quota information or None if not a quota error
    """
    error_str = str(error)

    # Check if this is a quota/rate limit error
    quota_patterns = [
        "429",
        "resource_exhausted",
        "resource exhausted",
        "quota",
        "rate limit",
        "too many requests"
    ]

    if not any(pattern in error_str.lower() for pattern in quota_patterns):
        return None

    quota_info = {
        "is_quota_error": True,
        "retry_after": None,
        "quota_type": "unknown",
        "quota_limit": None
    }

    # Extract retry delay from error message
    retry_patterns = [
        r"retry.*?(\d+)\s*s",  # "retry in 20s"
        r"retrydelay.*?(\d+)",  # "retryDelay: 20s"
        r"wait.*?(\d+)\s*s",   # "wait 20s"
        r"(\d+)\s*second",     # "20 seconds"
    ]

    for pattern in retry_patterns:
        match = re.search(pattern, error_str.lower())
        if match:
            quota_info["retry_after"] = int(match.group(1))
            break

    # Extract quota type
    if "free" in error_str.lower() or "freetier" in error_str.lower():
        quota_info["quota_type"] = "free_tier"
        # Free tier typically resets every minute
        if quota_info["retry_after"] is None:
            quota_info["retry_after"] = 60
    elif "per_minute" in error_str.lower():
        quota_info["quota_type"] = "per_minute"
        if quota_info["retry_after"] is None:
            quota_info["retry_after"] = 60
    elif "per_hour" in error_str.lower():
        quota_info["quota_type"] = "per_hour"
        if quota_info["retry_after"] is None:
            quota_info["retry_after"] = 3600

    # Default retry after if not found
    if quota_info["retry_after"] is None:
        quota_info["retry_after"] = 60  # Default to 1 minute

    return quota_info


async def retry_with_quota_handling(
    func: Callable,
    max_retries: int = 10,
    base_delay: float = 1.0,
    max_delay: float = 300.0,
    quota_wait_multiplier: float = 1.2,
    max_quota_wait: float = 600.0,
    operation_name: str = "operation"
) -> Any:
    """
    Retry a function with intelligent quota exhaustion handling.

    Args:
        func: Function to retry (can be sync or async)
        max_retries: Maximum number of retry attempts
        base_delay: Base delay for non-quota errors
        max_delay: Maximum delay for exponential backoff
        quota_wait_multiplier: Multiplier for quota wait times
        max_quota_wait: Maximum time to wait for quota reset
        operation_name: Name of the operation for logging

    Returns:
        Result of the function call

    Raises:
        The last exception if all retries fail
    """
    last_error = None
    quota_wait_count = 0

    for attempt in range(1, max_retries + 1):
        try:
            # Call function (handle both sync and async)
            if asyncio.iscoroutinefunction(func):
                return await func()
            else:
                return func()

        except Exception as e:
            last_error = e

            # Parse quota error information
            quota_info = parse_quota_error(e)

            if quota_info:
                # This is a quota error - handle specially
                retry_after = quota_info["retry_after"]
                quota_type = quota_info["quota_type"]

                # Apply multiplier to wait time (but cap it)
                actual_wait = min(retry_after * quota_wait_multiplier, max_quota_wait)
                quota_wait_count += 1

                logger.warning(
                    f"{operation_name} hit quota limit, waiting for quota reset",
                    attempt=attempt,
                    quota_type=quota_type,
                    retry_after=retry_after,
                    actual_wait=actual_wait,
                    quota_wait_count=quota_wait_count,
                    error=str(e)
                )

                # Wait for quota to reset
                await asyncio.sleep(actual_wait)

                # Continue to next attempt without counting this as a "normal" retry
                continue

            # Check if this is a retryable non-quota error
            error_str = str(e).lower()
            is_retryable = any(pattern in error_str for pattern in [
                "503", "overload", "service unavailable", "temporarily unavailable", "server error"
            ])

            if not is_retryable:
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
                    error_type=type(e).__name__,
                    quota_waits=quota_wait_count
                )
                break

            # Calculate delay with exponential backoff for non-quota errors
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)

            # Add jitter to prevent thundering herd
            delay *= (0.5 + random.random() * 0.5)

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


async def never_fail_extraction(
    extraction_func: Callable,
    fallback_strategies: Optional[list] = None,
    operation_name: str = "extraction"
) -> Any:
    """
    Wrapper that ensures extraction never completely fails.

    This function tries the main extraction, then fallback strategies,
    and finally returns an empty result rather than failing.

    Args:
        extraction_func: Main extraction function to try
        fallback_strategies: List of fallback functions to try
        operation_name: Name for logging

    Returns:
        Extraction result or empty result if all strategies fail
    """
    # Try main extraction with quota handling
    try:
        return await retry_with_quota_handling(
            extraction_func,
            max_retries=15,  # More retries for quota issues
            operation_name=f"{operation_name} (main)"
        )
    except Exception as main_error:
        logger.warning(
            f"Main {operation_name} failed, trying fallback strategies",
            error=str(main_error),
            error_type=type(main_error).__name__
        )

    # Try fallback strategies
    if fallback_strategies:
        for i, strategy in enumerate(fallback_strategies):
            try:
                logger.info(
                    f"Trying {operation_name} fallback strategy {i+1}/{len(fallback_strategies)}",
                    strategy=getattr(strategy, '__name__', str(strategy))
                )

                return await retry_with_quota_handling(
                    strategy,
                    max_retries=10,  # Fewer retries for fallbacks
                    operation_name=f"{operation_name} (fallback {i+1})"
                )

            except Exception as fallback_error:
                logger.warning(
                    f"{operation_name} fallback strategy {i+1} failed",
                    strategy=getattr(strategy, '__name__', str(strategy)),
                    error=str(fallback_error),
                    error_type=type(fallback_error).__name__
                )
                continue

    # If everything fails, return empty result
    logger.error(
        f"All {operation_name} strategies failed, returning empty result",
        operation_name=operation_name
    )

    # Return appropriate empty result based on operation type
    if "entity" in operation_name.lower():
        return EmptyExtractionResult(extraction_type="entities")
    elif "relation" in operation_name.lower():
        return EmptyExtractionResult(extraction_type="relations")
    else:
        return EmptyExtractionResult(extraction_type="unknown")


class EmptyExtractionResult:
    """Empty result for when all extraction strategies fail."""

    def __init__(self, extraction_type: str = "unknown"):
        self.extraction_type = extraction_type
        self.extractions = []
        self.success = False
        self.error_message = f"All {extraction_type} extraction strategies failed"

    def __bool__(self):
        return False

    def __len__(self):
        return 0


def quota_aware_retry(
    max_retries: Optional[int] = None,
    base_delay: Optional[float] = None,
    quota_wait_multiplier: float = 1.2,
    operation_name: str = "operation"
):
    """Decorator for quota-aware retry logic."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            from morag_core.config import settings

            effective_max_retries = max_retries or getattr(settings, 'entity_extraction_max_retries', 15)
            effective_base_delay = base_delay or getattr(settings, 'entity_extraction_retry_base_delay', 2.0)

            return await retry_with_quota_handling(
                lambda: func(*args, **kwargs),
                max_retries=effective_max_retries,
                base_delay=effective_base_delay,
                quota_wait_multiplier=quota_wait_multiplier,
                operation_name=operation_name
            )
        return wrapper
    return decorator
