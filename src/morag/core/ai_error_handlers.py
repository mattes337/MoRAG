"""Provider-specific AI error handlers with custom retry and fallback logic."""

import asyncio
from typing import Optional, Dict, Any, List
import structlog

from morag.core.resilience import (
    AIServiceResilience, RetryConfig, CircuitBreakerConfig, ErrorType
)
from morag.core.exceptions import (
    RateLimitError, QuotaExceededError, AuthenticationError, 
    ExternalServiceError, ContentPolicyError
)

logger = structlog.get_logger()


class GeminiErrorHandler(AIServiceResilience):
    """Gemini-specific error handler with optimized retry strategies."""
    
    def __init__(self):
        retry_config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=120.0,  # Longer max delay for Gemini
            exponential_base=2.0,
            jitter=True,
            retry_on_errors=[
                ErrorType.RATE_LIMIT, 
                ErrorType.SERVICE_UNAVAILABLE, 
                ErrorType.NETWORK,
                ErrorType.TIMEOUT
            ]
        )
        
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=120.0,  # 2 minutes for Gemini
            half_open_max_calls=2
        )
        
        super().__init__("gemini", retry_config, circuit_breaker_config)
    
    def _classify_error(self, exception: Exception) -> ErrorType:
        """Gemini-specific error classification."""
        error_message = str(exception).lower()
        
        # Gemini-specific error patterns
        if "resource_exhausted" in error_message or "quota exceeded" in error_message:
            return ErrorType.QUOTA_EXCEEDED
        elif "invalid_api_key" in error_message or "api_key_invalid" in error_message:
            return ErrorType.AUTHENTICATION
        elif "safety" in error_message or "harm" in error_message:
            return ErrorType.CONTENT_POLICY
        elif "model_overloaded" in error_message or "overloaded" in error_message:
            return ErrorType.SERVICE_UNAVAILABLE
        elif "deadline_exceeded" in error_message:
            return ErrorType.TIMEOUT
        
        # Fall back to base classification
        return super()._classify_error(exception)
    
    def _calculate_delay(self, attempt: int, error_type: ErrorType) -> float:
        """Gemini-specific delay calculation."""
        # Longer delays for Gemini rate limits
        if error_type == ErrorType.RATE_LIMIT:
            base_delay = self.retry_config.base_delay * 3  # 3x longer for rate limits
        elif error_type == ErrorType.QUOTA_EXCEEDED:
            base_delay = self.retry_config.base_delay * 5  # 5x longer for quota
        else:
            base_delay = self.retry_config.base_delay
        
        # Exponential backoff with Gemini-specific scaling
        delay = base_delay * (self.retry_config.exponential_base ** attempt)
        delay = min(delay, self.retry_config.max_delay)
        
        # Add jitter
        if self.retry_config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)
        
        return delay


class WhisperErrorHandler(AIServiceResilience):
    """Whisper-specific error handler for audio processing."""
    
    def __init__(self):
        retry_config = RetryConfig(
            max_retries=2,  # Fewer retries for audio processing
            base_delay=2.0,  # Longer base delay
            max_delay=60.0,
            exponential_base=2.0,
            jitter=True,
            retry_on_errors=[
                ErrorType.NETWORK,
                ErrorType.TIMEOUT,
                ErrorType.SERVICE_UNAVAILABLE
            ]
        )
        
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=3,  # Lower threshold for audio
            recovery_timeout=60.0,
            half_open_max_calls=1
        )
        
        super().__init__("whisper", retry_config, circuit_breaker_config)
    
    def _classify_error(self, exception: Exception) -> ErrorType:
        """Whisper-specific error classification."""
        error_message = str(exception).lower()
        
        # Whisper-specific patterns
        if "model not found" in error_message or "model loading" in error_message:
            return ErrorType.SERVICE_UNAVAILABLE
        elif "audio file" in error_message and "corrupt" in error_message:
            return ErrorType.UNKNOWN  # Don't retry corrupted files
        elif "memory" in error_message or "out of memory" in error_message:
            return ErrorType.SERVICE_UNAVAILABLE
        elif any(gpu_keyword in error_message for gpu_keyword in [
            "cuda", "gpu", "device", "nvidia", "cudnn", "cublas", "curand",
            "device-side assert", "kernel launch", "gpu memory"
        ]):
            return ErrorType.SERVICE_UNAVAILABLE
        
        return super()._classify_error(exception)
    
    def _should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        """Whisper-specific retry logic."""
        # Don't retry unknown errors (likely file corruption)
        if error_type == ErrorType.UNKNOWN:
            return False
        
        return super()._should_retry(error_type, attempt)


class VisionErrorHandler(AIServiceResilience):
    """Vision service error handler for image processing."""
    
    def __init__(self):
        retry_config = RetryConfig(
            max_retries=3,
            base_delay=1.5,
            max_delay=90.0,
            exponential_base=2.0,
            jitter=True,
            retry_on_errors=[
                ErrorType.RATE_LIMIT,
                ErrorType.SERVICE_UNAVAILABLE,
                ErrorType.NETWORK,
                ErrorType.TIMEOUT
            ]
        )
        
        circuit_breaker_config = CircuitBreakerConfig(
            failure_threshold=4,
            recovery_timeout=90.0,
            half_open_max_calls=2
        )
        
        super().__init__("vision", retry_config, circuit_breaker_config)
    
    def _classify_error(self, exception: Exception) -> ErrorType:
        """Vision-specific error classification."""
        error_message = str(exception).lower()
        
        # Vision-specific patterns
        if "image format" in error_message or "unsupported format" in error_message:
            return ErrorType.UNKNOWN  # Don't retry format errors
        elif "image too large" in error_message or "file size" in error_message:
            return ErrorType.UNKNOWN  # Don't retry size errors
        elif "image corrupted" in error_message or "cannot decode" in error_message:
            return ErrorType.UNKNOWN  # Don't retry corrupted images
        elif "vision model" in error_message and "unavailable" in error_message:
            return ErrorType.SERVICE_UNAVAILABLE
        elif any(gpu_keyword in error_message for gpu_keyword in [
            "cuda", "gpu", "device", "nvidia", "cudnn", "cublas", "curand",
            "device-side assert", "kernel launch", "gpu memory"
        ]):
            return ErrorType.SERVICE_UNAVAILABLE
        elif "memory" in error_message or "out of memory" in error_message:
            return ErrorType.SERVICE_UNAVAILABLE
        
        return super()._classify_error(exception)


class UniversalAIErrorHandler:
    """Universal error handler that manages multiple AI service handlers."""
    
    def __init__(self):
        self.handlers = {
            "gemini": GeminiErrorHandler(),
            "whisper": WhisperErrorHandler(),
            "vision": VisionErrorHandler()
        }
        self.logger = structlog.get_logger().bind(component="universal_ai_handler")
    
    def get_handler(self, service_name: str) -> AIServiceResilience:
        """Get appropriate error handler for service."""
        handler = self.handlers.get(service_name.lower())
        if not handler:
            # Create generic handler for unknown services
            self.logger.warning(
                "No specific handler found, using generic handler",
                service=service_name
            )
            handler = AIServiceResilience(service_name)
            self.handlers[service_name.lower()] = handler
        
        return handler
    
    async def execute_with_resilience(
        self,
        service_name: str,
        operation,
        *args,
        fallback=None,
        timeout=30.0,
        **kwargs
    ):
        """Execute operation with appropriate service handler."""
        handler = self.get_handler(service_name)
        return await handler.execute_with_resilience(
            operation, *args, fallback=fallback, timeout=timeout, **kwargs
        )
    
    def get_all_health_status(self) -> Dict[str, Any]:
        """Get health status for all services."""
        return {
            service: handler.get_health_status()
            for service, handler in self.handlers.items()
        }
    
    def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get health status for specific service."""
        handler = self.handlers.get(service_name.lower())
        if handler:
            return handler.get_health_status()
        return {"error": f"No handler found for service: {service_name}"}


# Global instance for easy access
universal_ai_handler = UniversalAIErrorHandler()


# Convenience functions for common operations
async def execute_with_ai_resilience(
    service_name: str,
    operation,
    *args,
    fallback=None,
    timeout=30.0,
    **kwargs
):
    """Convenience function to execute AI operations with resilience."""
    return await universal_ai_handler.execute_with_resilience(
        service_name, operation, *args, fallback=fallback, timeout=timeout, **kwargs
    )


def get_ai_service_health(service_name: str = None) -> Dict[str, Any]:
    """Get health status for AI services."""
    if service_name:
        return universal_ai_handler.get_service_health(service_name)
    return universal_ai_handler.get_all_health_status()
