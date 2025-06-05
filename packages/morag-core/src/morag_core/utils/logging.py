"""Logging utilities for MoRAG."""

import logging
import sys
from typing import Any, Dict, Optional, Union

import structlog
from structlog.types import Processor

from ..config import settings


def configure_logging(log_level: Optional[str] = None) -> None:
    """Configure structured logging for the application.
    
    Args:
        log_level: Optional log level to override settings
    """
    level = log_level or settings.log_level
    
    # Convert string level to numeric level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=numeric_level,
        stream=sys.stdout,
    )
    
    # Configure structlog
    processors: list[Processor] = [
        # Add timestamps and log level
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        
        # Add extra context from threadlocal storage
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        
        # Format the output
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add JSON formatting for production
    if settings.environment == "production":
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Pretty printing for development
        processors.append(
            structlog.dev.ConsoleRenderer(colors=True, exception_formatter=structlog.dev.plain_traceback)
        )
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Structured logger
    """
    return structlog.get_logger(name)


def add_request_context(logger: structlog.BoundLogger, request_id: str, **kwargs) -> structlog.BoundLogger:
    """Add request context to logger.
    
    Args:
        logger: Logger to bind context to
        request_id: Request ID
        **kwargs: Additional context
        
    Returns:
        Logger with bound context
    """
    context = {"request_id": request_id, **kwargs}
    return logger.bind(**context)


def log_exception(
    logger: structlog.BoundLogger,
    exc: Exception,
    message: str = "An error occurred",
    level: str = "error",
    **kwargs
) -> None:
    """Log exception with context.
    
    Args:
        logger: Logger to use
        exc: Exception to log
        message: Message to log
        level: Log level
        **kwargs: Additional context
    """
    log_method = getattr(logger, level.lower())
    log_method(
        message,
        error=str(exc),
        error_type=exc.__class__.__name__,
        **kwargs,
        exc_info=exc,
    )


# Alias for backward compatibility
setup_logging = configure_logging