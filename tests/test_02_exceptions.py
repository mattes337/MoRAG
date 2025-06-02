import pytest
from morag.core.exceptions import (
    MoRAGException, ValidationError, ProcessingError,
    StorageError, ExternalServiceError, AuthenticationError, RateLimitError
)

class TestExceptions:
    """Test custom exception classes."""

    def test_morag_exception_base(self):
        """Test base MoRAGException."""
        exc = MoRAGException("Test message", status_code=500, error_type="test_error")

        assert str(exc) == "Test message"
        assert exc.message == "Test message"
        assert exc.status_code == 500
        assert exc.error_type == "test_error"

    def test_validation_error(self):
        """Test ValidationError."""
        exc = ValidationError("Invalid input")

        assert exc.status_code == 400
        assert exc.error_type == "validation_error"
        assert exc.message == "Invalid input"

    def test_processing_error(self):
        """Test ProcessingError."""
        exc = ProcessingError("Processing failed")

        assert exc.status_code == 422
        assert exc.error_type == "processing_error"
        assert exc.message == "Processing failed"

    def test_storage_error(self):
        """Test StorageError."""
        exc = StorageError("Storage operation failed")

        assert exc.status_code == 500
        assert exc.error_type == "storage_error"
        assert exc.message == "Storage operation failed"

    def test_external_service_error(self):
        """Test ExternalServiceError."""
        exc = ExternalServiceError("API call failed", "gemini")

        assert exc.status_code == 502
        assert exc.error_type == "external_service_error"
        assert "gemini error: API call failed" in exc.message
        assert exc.service == "gemini"

    def test_authentication_error(self):
        """Test AuthenticationError."""
        exc = AuthenticationError("Invalid token")

        assert exc.status_code == 401
        assert exc.error_type == "authentication_error"
        assert exc.message == "Invalid token"

    def test_authentication_error_default(self):
        """Test AuthenticationError with default message."""
        exc = AuthenticationError()

        assert exc.status_code == 401
        assert exc.error_type == "authentication_error"
        assert exc.message == "Authentication required"

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        exc = RateLimitError("Too many requests")

        assert exc.status_code == 429
        assert exc.error_type == "rate_limit_error"
        assert exc.message == "Too many requests"

    def test_rate_limit_error_default(self):
        """Test RateLimitError with default message."""
        exc = RateLimitError()

        assert exc.status_code == 429
        assert exc.error_type == "rate_limit_error"
        assert exc.message == "Rate limit exceeded"

    def test_exception_inheritance(self):
        """Test that all custom exceptions inherit from MoRAGException."""
        assert issubclass(ValidationError, MoRAGException)
        assert issubclass(ProcessingError, MoRAGException)
        assert issubclass(StorageError, MoRAGException)
        assert issubclass(ExternalServiceError, MoRAGException)
        assert issubclass(AuthenticationError, MoRAGException)
        assert issubclass(RateLimitError, MoRAGException)
