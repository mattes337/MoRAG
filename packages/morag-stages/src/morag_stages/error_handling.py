"""Error handling utilities for MoRAG Stages."""

from functools import wraps
from typing import Callable, TypeVar, Any
import structlog

from .exceptions import StageValidationError, StageExecutionError

# Type variables for proper type hinting
F = TypeVar('F', bound=Callable[..., Any])

logger = structlog.get_logger(__name__)


def stage_error_handler(operation_name: str) -> Callable[[F], F]:
    """Decorator for consistent error handling in stage operations.

    This decorator ensures that:
    1. StageValidationError and StageExecutionError are propagated as-is
    2. Other exceptions are wrapped in StageExecutionError with context
    3. All errors are logged with structured logging

    Args:
        operation_name: Name of the operation being decorated (for logging)

    Returns:
        Decorated function with consistent error handling

    Example:
        @stage_error_handler("markdown_conversion")
        async def execute(self, input_files, context, output_dir=None):
            # Implementation here
            pass
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(self, *args, **kwargs):
            try:
                return await func(self, *args, **kwargs)
            except StageValidationError as e:
                # Validation errors should propagate with logging
                logger.error(
                    f"{operation_name} validation failed",
                    operation=operation_name,
                    stage_type=getattr(self, 'stage_type', 'unknown'),
                    error=str(e),
                    error_type=type(e).__name__,
                    invalid_files=getattr(e, 'invalid_files', []),
                    details=getattr(e, 'details', {})
                )
                raise
            except StageExecutionError as e:
                # Execution errors should propagate with logging
                logger.error(
                    f"{operation_name} execution failed",
                    operation=operation_name,
                    stage_type=getattr(self, 'stage_type', 'unknown'),
                    error=str(e),
                    error_type=type(e).__name__,
                    original_error=str(getattr(e, 'original_error', 'None')),
                    details=getattr(e, 'details', {})
                )
                raise
            except Exception as e:
                # Wrap generic exceptions with context
                logger.error(
                    f"{operation_name} failed with unexpected error",
                    operation=operation_name,
                    stage_type=getattr(self, 'stage_type', 'unknown'),
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True  # Include full traceback in logs
                )
                raise StageExecutionError(
                    f"{operation_name} failed: {str(e)}",
                    stage_type=getattr(self, 'stage_type', None),
                    original_error=e,
                    details={
                        'operation': operation_name,
                        'original_error_type': type(e).__name__
                    }
                ) from e

        @wraps(func)
        def sync_wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except StageValidationError as e:
                # Validation errors should propagate with logging
                logger.error(
                    f"{operation_name} validation failed",
                    operation=operation_name,
                    stage_type=getattr(self, 'stage_type', 'unknown'),
                    error=str(e),
                    error_type=type(e).__name__,
                    invalid_files=getattr(e, 'invalid_files', []),
                    details=getattr(e, 'details', {})
                )
                raise
            except StageExecutionError as e:
                # Execution errors should propagate with logging
                logger.error(
                    f"{operation_name} execution failed",
                    operation=operation_name,
                    stage_type=getattr(self, 'stage_type', 'unknown'),
                    error=str(e),
                    error_type=type(e).__name__,
                    original_error=str(getattr(e, 'original_error', 'None')),
                    details=getattr(e, 'details', {})
                )
                raise
            except Exception as e:
                # Wrap generic exceptions with context
                logger.error(
                    f"{operation_name} failed with unexpected error",
                    operation=operation_name,
                    stage_type=getattr(self, 'stage_type', 'unknown'),
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True  # Include full traceback in logs
                )
                raise StageExecutionError(
                    f"{operation_name} failed: {str(e)}",
                    stage_type=getattr(self, 'stage_type', None),
                    original_error=e,
                    details={
                        'operation': operation_name,
                        'original_error_type': type(e).__name__
                    }
                ) from e

        # Return appropriate wrapper based on whether the function is async
        import inspect
        if inspect.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        else:
            return sync_wrapper  # type: ignore

    return decorator


def validation_error_handler(operation_name: str) -> Callable[[F], F]:
    """Decorator specifically for validation operations.

    This is a specialized decorator for validation functions that should
    raise StageValidationError on failure.

    Args:
        operation_name: Name of the validation operation

    Returns:
        Decorated function that wraps exceptions in StageValidationError
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except StageValidationError:
                # Already a validation error, just re-raise
                raise
            except Exception as e:
                logger.error(
                    f"{operation_name} validation failed",
                    operation=operation_name,
                    stage_type=getattr(self, 'stage_type', 'unknown'),
                    error=str(e),
                    error_type=type(e).__name__,
                    exc_info=True
                )
                raise StageValidationError(
                    f"{operation_name} validation failed: {str(e)}",
                    stage_type=getattr(self, 'stage_type', None),
                    details={
                        'operation': operation_name,
                        'original_error_type': type(e).__name__
                    }
                ) from e

        return wrapper  # type: ignore
    return decorator


def standalone_validation_handler(operation_name: str) -> Callable[[F], F]:
    """Decorator for standalone validation functions that don't have 'self'.

    This is for functions that are not methods and should handle validation
    errors gracefully by logging and returning False instead of raising.

    Args:
        operation_name: Name of the validation operation

    Returns:
        Decorated function that handles validation errors gracefully
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.warning(
                    f"{operation_name} validation encountered error",
                    operation=operation_name,
                    error=str(e),
                    error_type=type(e).__name__,
                    # Don't include full traceback for validation warnings
                )
                # Return False for validation failures instead of raising
                return False

        return wrapper  # type: ignore
    return decorator
