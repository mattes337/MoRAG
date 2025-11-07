"""Exceptions for MoRAG AI agents."""

from typing import Optional, Any


class AgentError(Exception):
    """Base exception for agent-related errors."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.details = details


class ProviderError(AgentError):
    """Exception for provider-related errors."""
    pass


class ValidationError(AgentError):
    """Exception for validation errors."""
    pass


class RetryExhaustedError(AgentError):
    """Exception raised when all retry attempts are exhausted."""
    pass


class CircuitBreakerOpenError(AgentError):
    """Exception raised when circuit breaker is open."""
    pass


class TimeoutError(AgentError):
    """Exception for timeout errors."""
    pass


class RateLimitError(AgentError):
    """Exception for rate limit errors."""
    pass


class QuotaExceededError(AgentError):
    """Exception for quota exceeded errors."""
    pass


class ContentPolicyError(AgentError):
    """Exception for content policy violations."""
    pass


class ExternalServiceError(AgentError):
    """Exception for external service errors."""
    pass
