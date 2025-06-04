"""Custom exceptions for MoRAG."""
from typing import Optional

class MoRAGException(Exception):
    """Base exception for MoRAG application."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_type: str = "internal_error"
    ):
        self.message = message
        self.status_code = status_code
        self.error_type = error_type
        super().__init__(self.message)

class ValidationError(MoRAGException):
    """Raised when input validation fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=400, error_type="validation_error")

class ProcessingError(MoRAGException):
    """Raised when content processing fails."""

    def __init__(self, message: str):
        super().__init__(message, status_code=422, error_type="processing_error")

class StorageError(MoRAGException):
    """Raised when storage operations fail."""

    def __init__(self, message: str):
        super().__init__(message, status_code=500, error_type="storage_error")

class ExternalServiceError(MoRAGException):
    """Raised when external service calls fail."""

    def __init__(self, message: str, service: str):
        self.service = service
        super().__init__(
            f"{service} error: {message}",
            status_code=502,
            error_type="external_service_error"
        )

class AuthenticationError(MoRAGException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication required"):
        super().__init__(message, status_code=401, error_type="authentication_error")

class RateLimitError(MoRAGException):
    """Raised when rate limits are exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429, error_type="rate_limit_error")

class CircuitBreakerOpenError(MoRAGException):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str = "Circuit breaker is open"):
        super().__init__(message, status_code=503, error_type="circuit_breaker_open")

class TimeoutError(MoRAGException):
    """Raised when operations timeout."""

    def __init__(self, message: str = "Operation timed out"):
        super().__init__(message, status_code=408, error_type="timeout_error")

class QuotaExceededError(MoRAGException):
    """Raised when API quotas are exceeded."""

    def __init__(self, message: str = "API quota exceeded"):
        super().__init__(message, status_code=402, error_type="quota_exceeded_error")

class ContentPolicyError(MoRAGException):
    """Raised when content violates AI service policies."""

    def __init__(self, message: str = "Content policy violation"):
        super().__init__(message, status_code=400, error_type="content_policy_error")

# Legacy aliases for backward compatibility
MoragException = MoRAGException
ConfigurationError = MoRAGException
EmbeddingError = MoRAGException
TaskError = MoRAGException
