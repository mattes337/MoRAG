"""Unit tests for AI error handling and resilience framework."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.morag.core.resilience import (
    AIServiceResilience, RetryConfig, CircuitBreakerConfig,
    HealthMonitor, CircuitBreaker, ErrorType, CircuitBreakerState
)
from src.morag.core.ai_error_handlers import (
    GeminiErrorHandler, WhisperErrorHandler, VisionErrorHandler,
    UniversalAIErrorHandler, execute_with_ai_resilience
)
from morag_core.exceptions import (
    CircuitBreakerOpenError, RateLimitError, QuotaExceededError,
    TimeoutError, ContentPolicyError, ExternalServiceError
)


class TestErrorClassification:
    """Test error classification functionality."""
    
    def test_classify_rate_limit_errors(self):
        """Test classification of rate limit errors."""
        handler = GeminiErrorHandler()
        
        # Test various rate limit error patterns
        rate_limit_errors = [
            Exception("429 Too Many Requests"),
            Exception("Rate limit exceeded"),
            Exception("too many requests"),
            RateLimitError("API rate limit")
        ]
        
        for error in rate_limit_errors:
            error_type = handler._classify_error(error)
            assert error_type == ErrorType.RATE_LIMIT
    
    def test_classify_quota_errors(self):
        """Test classification of quota exceeded errors."""
        handler = GeminiErrorHandler()
        
        quota_errors = [
            Exception("quota exceeded"),
            Exception("402 Payment Required"),
            Exception("billing limit reached"),
            QuotaExceededError("Usage limit exceeded")
        ]
        
        for error in quota_errors:
            error_type = handler._classify_error(error)
            assert error_type == ErrorType.QUOTA_EXCEEDED
    
    def test_classify_authentication_errors(self):
        """Test classification of authentication errors."""
        handler = GeminiErrorHandler()
        
        auth_errors = [
            Exception("401 Unauthorized"),
            Exception("403 Forbidden"),
            Exception("invalid api key"),
            Exception("unauthorized access")
        ]
        
        for error in auth_errors:
            error_type = handler._classify_error(error)
            assert error_type == ErrorType.AUTHENTICATION
    
    def test_classify_timeout_errors(self):
        """Test classification of timeout errors."""
        handler = GeminiErrorHandler()
        
        timeout_errors = [
            Exception("timeout occurred"),
            Exception("deadline exceeded"),
            asyncio.TimeoutError("Operation timed out"),
            TimeoutError("Request timeout")
        ]
        
        for error in timeout_errors:
            error_type = handler._classify_error(error)
            assert error_type == ErrorType.TIMEOUT
    
    def test_classify_content_policy_errors(self):
        """Test classification of content policy errors."""
        handler = GeminiErrorHandler()
        
        policy_errors = [
            Exception("safety filter triggered"),
            Exception("content policy violation"),
            Exception("harmful content detected"),
            ContentPolicyError("Blocked content")
        ]
        
        for error in policy_errors:
            error_type = handler._classify_error(error)
            assert error_type == ErrorType.CONTENT_POLICY


class TestRetryLogic:
    """Test retry logic and exponential backoff."""
    
    def test_should_retry_logic(self):
        """Test retry decision logic."""
        config = RetryConfig(max_retries=3)
        handler = AIServiceResilience("test", config)
        
        # Should retry rate limits
        assert handler._should_retry(ErrorType.RATE_LIMIT, 0) == True
        assert handler._should_retry(ErrorType.RATE_LIMIT, 2) == True
        assert handler._should_retry(ErrorType.RATE_LIMIT, 3) == False  # Max retries
        
        # Should not retry authentication errors
        assert handler._should_retry(ErrorType.AUTHENTICATION, 0) == False
        
        # Should not retry content policy errors
        assert handler._should_retry(ErrorType.CONTENT_POLICY, 0) == False
        
        # Should retry quota only once
        assert handler._should_retry(ErrorType.QUOTA_EXCEEDED, 0) == True
        assert handler._should_retry(ErrorType.QUOTA_EXCEEDED, 1) == False
    
    def test_exponential_backoff_calculation(self):
        """Test exponential backoff delay calculation."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=60.0,
            jitter=False
        )
        handler = AIServiceResilience("test", config)
        
        # Test exponential progression
        delay_0 = handler._calculate_delay(0, ErrorType.NETWORK)
        delay_1 = handler._calculate_delay(1, ErrorType.NETWORK)
        delay_2 = handler._calculate_delay(2, ErrorType.NETWORK)
        
        assert delay_0 == 1.0
        assert delay_1 == 2.0
        assert delay_2 == 4.0
        
        # Test max delay cap
        delay_large = handler._calculate_delay(10, ErrorType.NETWORK)
        assert delay_large == 60.0
        
        # Test rate limit special handling
        rate_limit_delay = handler._calculate_delay(0, ErrorType.RATE_LIMIT)
        assert rate_limit_delay == 2.0  # 2x base delay for rate limits
    
    def test_jitter_application(self):
        """Test jitter is applied correctly."""
        config = RetryConfig(base_delay=1.0, jitter=True)
        handler = AIServiceResilience("test", config)
        
        delays = [handler._calculate_delay(0, ErrorType.NETWORK) for _ in range(10)]
        
        # All delays should be different due to jitter
        assert len(set(delays)) > 1
        
        # All delays should be between 0.5 and 1.0 (base_delay * jitter range)
        for delay in delays:
            assert 0.5 <= delay <= 1.0


class TestCircuitBreaker:
    """Test circuit breaker functionality."""
    
    def test_circuit_breaker_states(self):
        """Test circuit breaker state transitions."""
        config = CircuitBreakerConfig(failure_threshold=3, recovery_timeout=1.0)
        cb = CircuitBreaker("test", config)
        
        # Initial state should be closed
        assert cb.state == CircuitBreakerState.CLOSED
        assert cb.is_open() == False
        
        # Record failures to trigger opening
        for _ in range(3):
            cb.record_failure()
        
        # Should be open now
        assert cb.state == CircuitBreakerState.OPEN
        assert cb.is_open() == True
    
    def test_circuit_breaker_recovery(self):
        """Test circuit breaker recovery mechanism."""
        config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=0.1, half_open_max_calls=2)
        cb = CircuitBreaker("test", config)

        # Trigger opening
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN

        # Wait for recovery timeout
        import time
        time.sleep(0.2)

        # Should transition to half-open
        assert cb.is_open() == False
        assert cb.state == CircuitBreakerState.HALF_OPEN

        # Need multiple successes to close (based on half_open_max_calls)
        cb.record_success()
        assert cb.state == CircuitBreakerState.HALF_OPEN  # Still half-open
        cb.record_success()
        assert cb.state == CircuitBreakerState.CLOSED  # Now closed
    
    def test_half_open_failure_reopens(self):
        """Test that failure in half-open state reopens circuit."""
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)
        cb = CircuitBreaker("test", config)
        
        # Open circuit
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN
        
        # Wait and transition to half-open
        import time
        time.sleep(0.2)
        cb.is_open()  # This triggers transition to half-open
        assert cb.state == CircuitBreakerState.HALF_OPEN
        
        # Failure should reopen
        cb.record_failure()
        assert cb.state == CircuitBreakerState.OPEN


class TestHealthMonitor:
    """Test health monitoring functionality."""
    
    def test_health_metrics_calculation(self):
        """Test health metrics calculation."""
        monitor = HealthMonitor("test", window_size=10)
        
        # Record some operations
        for _ in range(5):
            monitor.record_attempt()
            monitor.record_success(1.0)
        
        for _ in range(2):
            monitor.record_attempt()
            monitor.record_failure(ErrorType.RATE_LIMIT)
        
        metrics = monitor.get_health_metrics()
        
        assert metrics["service_name"] == "test"
        assert metrics["total_attempts"] == 7
        assert metrics["total_successes"] == 5
        assert metrics["total_failures"] == 2
        assert metrics["avg_response_time"] == 1.0
        assert ErrorType.RATE_LIMIT in metrics["error_distribution"]
        assert metrics["error_distribution"][ErrorType.RATE_LIMIT] == 2
    
    def test_health_status_calculation(self):
        """Test health status determination."""
        monitor = HealthMonitor("test")
        
        # Test healthy status (95%+ success rate)
        assert monitor._calculate_health_status(0.95) == "healthy"
        assert monitor._calculate_health_status(1.0) == "healthy"
        
        # Test degraded status (80-95% success rate)
        assert monitor._calculate_health_status(0.8) == "degraded"
        assert monitor._calculate_health_status(0.9) == "degraded"
        
        # Test unhealthy status (<80% success rate)
        assert monitor._calculate_health_status(0.7) == "unhealthy"
        assert monitor._calculate_health_status(0.0) == "unhealthy"


@pytest.mark.asyncio
class TestAIServiceResilience:
    """Test the main AI service resilience framework."""
    
    async def test_successful_operation(self):
        """Test successful operation execution."""
        handler = AIServiceResilience("test")
        
        async def mock_operation():
            return "success"
        
        result = await handler.execute_with_resilience(mock_operation)
        assert result == "success"
        
        # Check health metrics
        health = handler.get_health_status()
        assert health["total_attempts"] == 1
        assert health["total_successes"] == 1
        assert health["total_failures"] == 0
    
    async def test_retry_on_transient_failure(self):
        """Test retry behavior on transient failures."""
        config = RetryConfig(max_retries=2, base_delay=0.01)  # Fast retries for testing
        handler = AIServiceResilience("test", config)
        
        call_count = 0
        
        async def mock_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("503 Service Unavailable")
            return "success"
        
        result = await handler.execute_with_resilience(mock_operation)
        assert result == "success"
        assert call_count == 3  # Initial + 2 retries
    
    async def test_no_retry_on_auth_error(self):
        """Test that authentication errors are not retried."""
        handler = AIServiceResilience("test")
        
        call_count = 0
        
        async def mock_operation():
            nonlocal call_count
            call_count += 1
            raise Exception("401 Unauthorized")
        
        with pytest.raises(Exception, match="401 Unauthorized"):
            await handler.execute_with_resilience(mock_operation)
        
        assert call_count == 1  # No retries
    
    async def test_fallback_execution(self):
        """Test fallback execution when primary fails."""
        handler = AIServiceResilience("test")
        
        async def failing_operation():
            raise Exception("401 Unauthorized")
        
        async def fallback_operation():
            return "fallback_result"
        
        result = await handler.execute_with_resilience(
            failing_operation,
            fallback=fallback_operation
        )
        assert result == "fallback_result"
    
    async def test_circuit_breaker_prevents_execution(self):
        """Test that open circuit breaker prevents execution."""
        config = RetryConfig(max_retries=1)
        cb_config = CircuitBreakerConfig(failure_threshold=1)
        handler = AIServiceResilience("test", config, cb_config)
        
        # Trigger circuit breaker
        async def failing_operation():
            raise Exception("500 Internal Server Error")
        
        with pytest.raises(Exception):
            await handler.execute_with_resilience(failing_operation)
        
        # Circuit should be open now
        assert handler.circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Next call should be blocked
        with pytest.raises(CircuitBreakerOpenError):
            await handler.execute_with_resilience(failing_operation)


class TestProviderSpecificHandlers:
    """Test provider-specific error handlers."""
    
    def test_gemini_error_classification(self):
        """Test Gemini-specific error patterns."""
        handler = GeminiErrorHandler()
        
        # Gemini-specific patterns
        assert handler._classify_error(Exception("resource_exhausted")) == ErrorType.QUOTA_EXCEEDED
        assert handler._classify_error(Exception("invalid_api_key")) == ErrorType.AUTHENTICATION
        assert handler._classify_error(Exception("safety filter")) == ErrorType.CONTENT_POLICY
        assert handler._classify_error(Exception("model_overloaded")) == ErrorType.SERVICE_UNAVAILABLE
        assert handler._classify_error(Exception("deadline_exceeded")) == ErrorType.TIMEOUT
    
    def test_whisper_error_classification(self):
        """Test Whisper-specific error patterns."""
        handler = WhisperErrorHandler()
        
        # Whisper-specific patterns
        assert handler._classify_error(Exception("model not found")) == ErrorType.SERVICE_UNAVAILABLE
        assert handler._classify_error(Exception("audio file corrupt")) == ErrorType.UNKNOWN
        assert handler._classify_error(Exception("out of memory")) == ErrorType.SERVICE_UNAVAILABLE
        assert handler._classify_error(Exception("cuda error")) == ErrorType.SERVICE_UNAVAILABLE
    
    def test_vision_error_classification(self):
        """Test Vision-specific error patterns."""
        handler = VisionErrorHandler()
        
        # Vision-specific patterns
        assert handler._classify_error(Exception("image format unsupported")) == ErrorType.UNKNOWN
        assert handler._classify_error(Exception("image too large")) == ErrorType.UNKNOWN
        assert handler._classify_error(Exception("image corrupted")) == ErrorType.UNKNOWN
        assert handler._classify_error(Exception("vision model unavailable")) == ErrorType.SERVICE_UNAVAILABLE


class TestUniversalHandler:
    """Test the universal AI error handler."""

    def test_get_appropriate_handler(self):
        """Test getting appropriate handler for different services."""
        universal = UniversalAIErrorHandler()

        gemini_handler = universal.get_handler("gemini")
        assert isinstance(gemini_handler, GeminiErrorHandler)

        whisper_handler = universal.get_handler("whisper")
        assert isinstance(whisper_handler, WhisperErrorHandler)

        vision_handler = universal.get_handler("vision")
        assert isinstance(vision_handler, VisionErrorHandler)

        # Unknown service should get generic handler
        generic_handler = universal.get_handler("unknown_service")
        assert isinstance(generic_handler, AIServiceResilience)
        assert generic_handler.service_name == "unknown_service"

    @pytest.mark.asyncio
    async def test_execute_with_resilience(self):
        """Test universal handler execution."""
        universal = UniversalAIErrorHandler()

        async def mock_operation():
            return "test_result"

        result = await universal.execute_with_resilience(
            "gemini",
            mock_operation
        )
        assert result == "test_result"

    def test_health_status_collection(self):
        """Test health status collection for all services."""
        universal = UniversalAIErrorHandler()

        # Initialize some handlers
        universal.get_handler("gemini")
        universal.get_handler("whisper")

        all_health = universal.get_all_health_status()
        assert "gemini" in all_health
        assert "whisper" in all_health

        # Test specific service health
        gemini_health = universal.get_service_health("gemini")
        assert "service_name" in gemini_health
        assert gemini_health["service_name"] == "gemini"
