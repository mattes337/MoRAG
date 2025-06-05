"""AI service resilience framework with retry, circuit breaker, and health monitoring."""

import asyncio
import logging
import time
import random
from typing import Optional, Callable, Any, Dict, List
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from datetime import datetime, timedelta
import structlog

from morag_core.exceptions import (
    CircuitBreakerOpenError, RateLimitError, QuotaExceededError,
    TimeoutError, ContentPolicyError, ExternalServiceError, AuthenticationError
)

logger = structlog.get_logger()


class ErrorType(Enum):
    """Classification of AI service errors."""
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    QUOTA_EXCEEDED = "quota_exceeded"
    AUTHENTICATION = "authentication"
    TIMEOUT = "timeout"
    CONTENT_POLICY = "content_policy"
    NETWORK = "network"
    UNKNOWN = "unknown"


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_errors: List[ErrorType] = field(default_factory=lambda: [
        ErrorType.RATE_LIMIT,
        ErrorType.SERVICE_UNAVAILABLE,
        ErrorType.NETWORK,
        ErrorType.TIMEOUT,
        ErrorType.QUOTA_EXCEEDED
    ])


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    half_open_max_calls: int = 3


class HealthMonitor:
    """Monitor health metrics for AI services."""
    
    def __init__(self, service_name: str, window_size: int = 100):
        self.service_name = service_name
        self.window_size = window_size
        self.attempts = deque(maxlen=window_size)
        self.successes = deque(maxlen=window_size)
        self.failures = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        self.error_counts = {}
        self.logger = structlog.get_logger().bind(service=service_name, component="health_monitor")
    
    def record_attempt(self):
        """Record an attempt."""
        self.attempts.append(datetime.now())
    
    def record_success(self, response_time: float = None):
        """Record a successful operation."""
        self.successes.append(datetime.now())
        if response_time:
            self.response_times.append(response_time)
    
    def record_failure(self, error_type: ErrorType):
        """Record a failed operation."""
        self.failures.append(datetime.now())
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_health_metrics(self) -> Dict[str, Any]:
        """Get current health metrics."""
        now = datetime.now()
        recent_window = now - timedelta(minutes=5)
        
        recent_attempts = sum(1 for t in self.attempts if t > recent_window)
        recent_successes = sum(1 for t in self.successes if t > recent_window)
        recent_failures = sum(1 for t in self.failures if t > recent_window)
        
        success_rate = (recent_successes / recent_attempts) if recent_attempts > 0 else 0
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        return {
            "service_name": self.service_name,
            "success_rate": success_rate,
            "recent_attempts": recent_attempts,
            "recent_successes": recent_successes,
            "recent_failures": recent_failures,
            "avg_response_time": avg_response_time,
            "error_distribution": dict(self.error_counts),
            "health_status": self._calculate_health_status(success_rate),
            "total_attempts": len(self.attempts),
            "total_successes": len(self.successes),
            "total_failures": len(self.failures)
        }
    
    def _calculate_health_status(self, success_rate: float) -> str:
        """Calculate health status based on success rate."""
        if success_rate >= 0.95:
            return "healthy"
        elif success_rate >= 0.8:
            return "degraded"
        else:
            return "unhealthy"


class CircuitBreaker:
    """Circuit breaker implementation for AI services."""
    
    def __init__(
        self, 
        service_name: str,
        config: CircuitBreakerConfig = None
    ):
        self.service_name = service_name
        self.config = config or CircuitBreakerConfig()
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.logger = structlog.get_logger().bind(service=service_name, component="circuit_breaker")
    
    def is_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                self.logger.info("Circuit breaker transitioning to half-open")
                return False
            return True
        return False
    
    def record_success(self):
        """Record a successful operation."""
        self.failure_count = 0

        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.half_open_max_calls:
                self.state = CircuitBreakerState.CLOSED
                self.logger.info("Circuit breaker closed after successful recovery")
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset success count when already closed
            self.success_count = 0
    
    def record_failure(self):
        """Record a failed operation."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning("Circuit breaker opened from half-open state")
        elif self.failure_count >= self.config.failure_threshold:
            if self.state != CircuitBreakerState.OPEN:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(
                    "Circuit breaker opened",
                    failure_count=self.failure_count,
                    threshold=self.config.failure_threshold
                )
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt reset."""
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.config.recovery_timeout
        )
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state."""
        return {
            "service_name": self.service_name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "last_failure_time": self.last_failure_time,
            "time_until_reset": max(0, self.config.recovery_timeout - (time.time() - (self.last_failure_time or 0)))
        }


class AIServiceResilience:
    """Main resilience framework for AI services."""
    
    def __init__(
        self, 
        service_name: str, 
        retry_config: RetryConfig = None,
        circuit_breaker_config: CircuitBreakerConfig = None
    ):
        self.service_name = service_name
        self.retry_config = retry_config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(service_name, circuit_breaker_config)
        self.health_monitor = HealthMonitor(service_name)
        self.logger = structlog.get_logger().bind(service=service_name, component="resilience")
    
    async def execute_with_resilience(
        self, 
        operation: Callable,
        *args,
        fallback: Optional[Callable] = None,
        timeout: float = 30.0,
        **kwargs
    ) -> Any:
        """Execute operation with full resilience patterns."""
        
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            self.logger.warning("Circuit breaker is open, attempting fallback")
            if fallback:
                return await self._execute_fallback(fallback, *args, **kwargs)
            raise CircuitBreakerOpenError(f"Circuit breaker open for {self.service_name}")
        
        # Execute with retry logic
        return await self._execute_with_retry(operation, fallback, timeout, *args, **kwargs)

    async def _execute_with_retry(
        self,
        operation: Callable,
        fallback: Optional[Callable],
        timeout: float,
        *args,
        **kwargs
    ) -> Any:
        """Execute operation with retry logic."""
        last_exception = None
        start_time = time.time()

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                # Record attempt
                self.health_monitor.record_attempt()

                # Execute operation with timeout
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=timeout
                )

                # Record success
                response_time = time.time() - start_time
                self.health_monitor.record_success(response_time)
                self.circuit_breaker.record_success()

                self.logger.debug(
                    "Operation completed successfully",
                    attempt=attempt + 1,
                    response_time=response_time
                )

                return result

            except Exception as e:
                last_exception = e
                error_type = self._classify_error(e)

                # Record failure
                self.health_monitor.record_failure(error_type)
                self.circuit_breaker.record_failure()

                # Log error details
                self.logger.error(
                    "Operation attempt failed",
                    attempt=attempt + 1,
                    max_retries=self.retry_config.max_retries,
                    error_type=error_type.value,
                    error=str(e),
                    error_class=type(e).__name__
                )

                # Check if we should retry
                if not self._should_retry(error_type, attempt):
                    self.logger.info(
                        "Not retrying due to error type or max attempts reached",
                        error_type=error_type.value,
                        attempt=attempt + 1
                    )
                    break

                # Calculate delay and wait
                if attempt < self.retry_config.max_retries:
                    delay = self._calculate_delay(attempt, error_type)
                    self.logger.info(
                        "Retrying operation",
                        delay=delay,
                        next_attempt=attempt + 2
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted, try fallback
        if fallback:
            self.logger.info("Executing fallback after retry exhaustion")
            return await self._execute_fallback(fallback, *args, **kwargs)

        # No fallback available, raise last exception
        self.logger.error(
            "Operation failed after all retries, no fallback available",
            final_error=str(last_exception)
        )
        raise last_exception

    async def _execute_fallback(self, fallback: Callable, *args, **kwargs) -> Any:
        """Execute fallback operation."""
        try:
            self.logger.info("Executing fallback operation")
            result = await fallback(*args, **kwargs)
            self.logger.info("Fallback operation completed successfully")
            return result
        except Exception as e:
            self.logger.error("Fallback operation failed", error=str(e))
            raise

    def _classify_error(self, exception: Exception) -> ErrorType:
        """Classify error type for appropriate handling."""
        error_message = str(exception).lower()
        exception_type = type(exception).__name__.lower()

        # Check exception types first
        if isinstance(exception, RateLimitError):
            return ErrorType.RATE_LIMIT
        elif isinstance(exception, QuotaExceededError):
            return ErrorType.QUOTA_EXCEEDED
        elif isinstance(exception, AuthenticationError):
            return ErrorType.AUTHENTICATION
        elif isinstance(exception, TimeoutError) or isinstance(exception, asyncio.TimeoutError):
            return ErrorType.TIMEOUT
        elif isinstance(exception, ContentPolicyError):
            return ErrorType.CONTENT_POLICY

        # Check error messages for patterns
        if any(pattern in error_message for pattern in ["429", "rate limit", "too many requests"]):
            return ErrorType.RATE_LIMIT
        elif any(pattern in error_message for pattern in ["503", "service unavailable", "temporarily unavailable"]):
            return ErrorType.SERVICE_UNAVAILABLE
        elif any(pattern in error_message for pattern in ["quota", "402", "billing", "usage limit"]):
            return ErrorType.QUOTA_EXCEEDED
        elif any(pattern in error_message for pattern in ["401", "403", "unauthorized", "forbidden", "api key"]):
            return ErrorType.AUTHENTICATION
        elif any(pattern in error_message for pattern in ["timeout", "timed out", "deadline"]):
            return ErrorType.TIMEOUT
        elif any(pattern in error_message for pattern in ["safety", "content policy", "harmful", "blocked"]):
            return ErrorType.CONTENT_POLICY
        elif any(pattern in error_message for pattern in ["connection", "network", "dns", "resolve"]):
            return ErrorType.NETWORK
        else:
            return ErrorType.UNKNOWN

    def _should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        """Determine if operation should be retried."""
        if attempt >= self.retry_config.max_retries:
            return False

        # Don't retry certain error types
        if error_type in [ErrorType.AUTHENTICATION, ErrorType.CONTENT_POLICY]:
            return False

        # Don't retry quota exceeded after first attempt
        if error_type == ErrorType.QUOTA_EXCEEDED and attempt > 0:
            return False

        # Check if error type is in retry list
        return error_type in self.retry_config.retry_on_errors

    def _calculate_delay(self, attempt: int, error_type: ErrorType) -> float:
        """Calculate delay for next retry attempt."""
        # Special handling for rate limits
        if error_type == ErrorType.RATE_LIMIT:
            base_delay = self.retry_config.base_delay * 2
        else:
            base_delay = self.retry_config.base_delay

        # Exponential backoff
        delay = base_delay * (self.retry_config.exponential_base ** attempt)
        delay = min(delay, self.retry_config.max_delay)

        # Add jitter to prevent thundering herd
        if self.retry_config.jitter:
            delay *= (0.5 + random.random() * 0.5)

        return delay

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        health_metrics = self.health_monitor.get_health_metrics()
        circuit_breaker_state = self.circuit_breaker.get_state()

        return {
            **health_metrics,
            "circuit_breaker": circuit_breaker_state,
            "retry_config": {
                "max_retries": self.retry_config.max_retries,
                "base_delay": self.retry_config.base_delay,
                "max_delay": self.retry_config.max_delay
            }
        }
