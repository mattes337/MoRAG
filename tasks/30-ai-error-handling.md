# Task 30: Robust AI/LLM Error Handling and Resilience

## Objective
Implement comprehensive error handling and resilience patterns for AI/LLM services, including exponential backoff, circuit breakers, graceful degradation, and comprehensive logging for debugging API issues.

## Research Phase

### Common AI/LLM API Errors
1. **Rate Limiting (429)**: Too many requests per time window
2. **Service Unavailable (503)**: Temporary service outages
3. **Quota Exceeded (402/429)**: Usage limits reached
4. **Authentication Errors (401/403)**: Invalid or expired API keys
5. **Timeout Errors**: Requests taking too long
6. **Model Overload**: Model temporarily unavailable
7. **Content Policy Violations**: Input/output filtered by safety systems
8. **Network Errors**: Connection issues, DNS failures

### Resilience Patterns
1. **Exponential Backoff**: Gradually increase retry delays
2. **Circuit Breaker**: Stop requests when service is failing
3. **Bulkhead**: Isolate failures to prevent cascade
4. **Timeout**: Prevent hanging requests
5. **Fallback**: Use alternative services or cached responses
6. **Health Checks**: Monitor service availability

### Error Handling Libraries
1. **tenacity** - Retry library with various strategies
2. **circuitbreaker** - Circuit breaker pattern implementation
3. **aiohttp-retry** - Async HTTP retry mechanisms
4. **backoff** - Exponential backoff decorators

## Implementation Strategy

### Phase 1: Core Resilience Framework
```python
import asyncio
import logging
from typing import Optional, Callable, Any, Dict
from dataclasses import dataclass
from enum import Enum
import time
import random

class ErrorType(Enum):
    RATE_LIMIT = "rate_limit"
    SERVICE_UNAVAILABLE = "service_unavailable"
    QUOTA_EXCEEDED = "quota_exceeded"
    AUTHENTICATION = "authentication"
    TIMEOUT = "timeout"
    CONTENT_POLICY = "content_policy"
    NETWORK = "network"
    UNKNOWN = "unknown"

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True
    retry_on_errors: list = None

class AIServiceResilience:
    def __init__(self, service_name: str, config: RetryConfig = None):
        self.service_name = service_name
        self.config = config or RetryConfig()
        self.circuit_breaker = CircuitBreaker(service_name)
        self.health_monitor = HealthMonitor(service_name)
        self.logger = logging.getLogger(f"ai_resilience.{service_name}")
    
    async def execute_with_resilience(
        self, 
        operation: Callable,
        *args,
        fallback: Optional[Callable] = None,
        **kwargs
    ) -> Any:
        """Execute operation with full resilience patterns"""
        
        # Check circuit breaker
        if self.circuit_breaker.is_open():
            self.logger.warning(f"Circuit breaker open for {self.service_name}")
            if fallback:
                return await self._execute_fallback(fallback, *args, **kwargs)
            raise CircuitBreakerOpenError(f"Circuit breaker open for {self.service_name}")
        
        # Execute with retry logic
        return await self._execute_with_retry(operation, fallback, *args, **kwargs)
    
    async def _execute_with_retry(
        self, 
        operation: Callable, 
        fallback: Optional[Callable],
        *args, 
        **kwargs
    ) -> Any:
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Record attempt
                self.health_monitor.record_attempt()
                
                # Execute operation with timeout
                result = await asyncio.wait_for(
                    operation(*args, **kwargs),
                    timeout=kwargs.get('timeout', 30.0)
                )
                
                # Record success
                self.health_monitor.record_success()
                self.circuit_breaker.record_success()
                
                return result
                
            except Exception as e:
                last_exception = e
                error_type = self._classify_error(e)
                
                # Record failure
                self.health_monitor.record_failure(error_type)
                self.circuit_breaker.record_failure()
                
                # Log error details
                self.logger.error(
                    f"Attempt {attempt + 1} failed for {self.service_name}: {error_type.value}",
                    extra={
                        "error_type": error_type.value,
                        "attempt": attempt + 1,
                        "max_retries": self.config.max_retries,
                        "exception": str(e)
                    }
                )
                
                # Check if we should retry
                if not self._should_retry(error_type, attempt):
                    break
                
                # Calculate delay and wait
                if attempt < self.config.max_retries:
                    delay = self._calculate_delay(attempt, error_type)
                    self.logger.info(f"Retrying in {delay:.2f} seconds...")
                    await asyncio.sleep(delay)
        
        # All retries exhausted, try fallback
        if fallback:
            self.logger.info(f"Executing fallback for {self.service_name}")
            return await self._execute_fallback(fallback, *args, **kwargs)
        
        # No fallback available, raise last exception
        raise last_exception
```

### Phase 2: Circuit Breaker Implementation
```python
class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(
        self, 
        service_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: type = Exception
    ):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self.logger = logging.getLogger(f"circuit_breaker.{service_name}")
    
    def is_open(self) -> bool:
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info(f"Circuit breaker half-open for {self.service_name}")
                return False
            return True
        return False
    
    def record_success(self):
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            self.logger.info(f"Circuit breaker closed for {self.service_name}")
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitBreakerState.OPEN:
                self.state = CircuitBreakerState.OPEN
                self.logger.warning(
                    f"Circuit breaker opened for {self.service_name} "
                    f"after {self.failure_count} failures"
                )
    
    def _should_attempt_reset(self) -> bool:
        return (
            self.last_failure_time and
            time.time() - self.last_failure_time >= self.recovery_timeout
        )
```

### Phase 3: Health Monitoring and Metrics
```python
from collections import deque
from datetime import datetime, timedelta

class HealthMonitor:
    def __init__(self, service_name: str, window_size: int = 100):
        self.service_name = service_name
        self.window_size = window_size
        self.attempts = deque(maxlen=window_size)
        self.successes = deque(maxlen=window_size)
        self.failures = deque(maxlen=window_size)
        self.response_times = deque(maxlen=window_size)
        self.error_counts = {}
        self.logger = logging.getLogger(f"health_monitor.{service_name}")
    
    def record_attempt(self):
        self.attempts.append(datetime.now())
    
    def record_success(self, response_time: float = None):
        self.successes.append(datetime.now())
        if response_time:
            self.response_times.append(response_time)
    
    def record_failure(self, error_type: ErrorType):
        self.failures.append(datetime.now())
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
    
    def get_health_metrics(self) -> Dict[str, Any]:
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
            "health_status": self._calculate_health_status(success_rate)
        }
    
    def _calculate_health_status(self, success_rate: float) -> str:
        if success_rate >= 0.95:
            return "healthy"
        elif success_rate >= 0.8:
            return "degraded"
        else:
            return "unhealthy"
```

### Phase 4: Provider-Specific Error Handling
```python
class GeminiErrorHandler(AIServiceResilience):
    def __init__(self):
        super().__init__("gemini", RetryConfig(
            max_retries=3,
            base_delay=1.0,
            retry_on_errors=[ErrorType.RATE_LIMIT, ErrorType.SERVICE_UNAVAILABLE]
        ))
    
    def _classify_error(self, exception: Exception) -> ErrorType:
        error_message = str(exception).lower()
        
        if "429" in error_message or "rate limit" in error_message:
            return ErrorType.RATE_LIMIT
        elif "503" in error_message or "service unavailable" in error_message:
            return ErrorType.SERVICE_UNAVAILABLE
        elif "quota" in error_message or "402" in error_message:
            return ErrorType.QUOTA_EXCEEDED
        elif "401" in error_message or "403" in error_message:
            return ErrorType.AUTHENTICATION
        elif "timeout" in error_message:
            return ErrorType.TIMEOUT
        elif "safety" in error_message or "content policy" in error_message:
            return ErrorType.CONTENT_POLICY
        else:
            return ErrorType.UNKNOWN
    
    def _should_retry(self, error_type: ErrorType, attempt: int) -> bool:
        if attempt >= self.config.max_retries:
            return False
        
        # Don't retry authentication or content policy errors
        if error_type in [ErrorType.AUTHENTICATION, ErrorType.CONTENT_POLICY]:
            return False
        
        # Don't retry quota exceeded after first attempt
        if error_type == ErrorType.QUOTA_EXCEEDED and attempt > 0:
            return False
        
        return True
    
    def _calculate_delay(self, attempt: int, error_type: ErrorType) -> float:
        # Special handling for rate limits
        if error_type == ErrorType.RATE_LIMIT:
            # Longer delays for rate limits
            base_delay = self.config.base_delay * 2
        else:
            base_delay = self.config.base_delay
        
        # Exponential backoff
        delay = base_delay * (self.config.exponential_base ** attempt)
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            delay *= (0.5 + random.random() * 0.5)
        
        return delay
```

## Integration with MoRAG System

### Enhanced AI Service Layer
```python
# Update services/gemini_service.py
class GeminiService:
    def __init__(self):
        self.client = genai.GenerativeModel('gemini-1.5-pro')
        self.error_handler = GeminiErrorHandler()
        self.fallback_service = OpenAIService()  # Fallback to OpenAI
        self.logger = logging.getLogger("gemini_service")
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        return await self.error_handler.execute_with_resilience(
            self._generate_text_internal,
            prompt,
            fallback=self._fallback_generate_text,
            **kwargs
        )
    
    async def _generate_text_internal(self, prompt: str, **kwargs) -> str:
        response = await self.client.generate_content_async(prompt)
        return response.text
    
    async def _fallback_generate_text(self, prompt: str, **kwargs) -> str:
        self.logger.info("Using OpenAI fallback for text generation")
        return await self.fallback_service.generate_text(prompt, **kwargs)
    
    async def embed_text(self, text: str) -> List[float]:
        return await self.error_handler.execute_with_resilience(
            self._embed_text_internal,
            text,
            fallback=self._fallback_embed_text
        )
    
    async def get_health_status(self) -> Dict[str, Any]:
        return self.error_handler.health_monitor.get_health_metrics()
```

### Comprehensive Logging Configuration
```python
# Enhanced logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(extra)s'
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "service": "%(name)s", "level": "%(levelname)s", "message": "%(message)s", "extra": %(extra)s}'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
            'level': 'INFO'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/ai_services.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'formatter': 'json',
            'level': 'DEBUG'
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': 'logs/ai_errors.log',
            'maxBytes': 10485760,
            'backupCount': 10,
            'formatter': 'json',
            'level': 'ERROR'
        }
    },
    'loggers': {
        'ai_resilience': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': False
        },
        'circuit_breaker': {
            'handlers': ['console', 'file', 'error_file'],
            'level': 'INFO',
            'propagate': False
        },
        'health_monitor': {
            'handlers': ['file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}
```

### Database Schema for Error Tracking
```sql
-- AI service health and error tracking
CREATE TABLE ai_service_health (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(100),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    success_rate FLOAT,
    avg_response_time FLOAT,
    total_requests INTEGER,
    successful_requests INTEGER,
    failed_requests INTEGER,
    health_status VARCHAR(20)
);

CREATE TABLE ai_service_errors (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(100),
    error_type VARCHAR(50),
    error_message TEXT,
    stack_trace TEXT,
    request_context JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE
);

CREATE TABLE circuit_breaker_events (
    id SERIAL PRIMARY KEY,
    service_name VARCHAR(100),
    event_type VARCHAR(20), -- opened, closed, half_open
    failure_count INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_ai_health_service_time ON ai_service_health(service_name, timestamp);
CREATE INDEX idx_ai_errors_service_time ON ai_service_errors(service_name, timestamp);
CREATE INDEX idx_circuit_events_service ON circuit_breaker_events(service_name, timestamp);
```

## Testing Requirements

### Unit Tests
- [ ] Test retry logic with different error types
- [ ] Test circuit breaker state transitions
- [ ] Test exponential backoff calculations
- [ ] Test health monitoring metrics
- [ ] Test fallback mechanisms

### Integration Tests
- [ ] Test with actual AI service failures
- [ ] Test circuit breaker behavior under load
- [ ] Test fallback service switching
- [ ] Test error classification accuracy
- [ ] Test logging and monitoring

### Chaos Engineering Tests
- [ ] Simulate AI service outages
- [ ] Test rate limiting scenarios
- [ ] Test network failures and timeouts
- [ ] Test quota exhaustion scenarios
- [ ] Test concurrent failure handling

## Implementation Steps

### Step 1: Core Resilience Framework (2-3 days)
- [ ] Implement retry mechanisms with exponential backoff
- [ ] Create error classification system
- [ ] Add comprehensive logging
- [ ] Basic health monitoring

### Step 2: Circuit Breaker Pattern (1-2 days)
- [ ] Implement circuit breaker logic
- [ ] Add state management and transitions
- [ ] Integrate with retry mechanisms
- [ ] Add monitoring and alerting

### Step 3: Provider-Specific Handlers (2-3 days)
- [ ] Create Gemini-specific error handler
- [ ] Add OpenAI error handler
- [ ] Implement fallback mechanisms
- [ ] Add provider health checks

### Step 4: Monitoring and Observability (1-2 days)
- [ ] Enhanced logging configuration
- [ ] Health metrics collection
- [ ] Error tracking and analysis
- [ ] Dashboard integration

### Step 5: Integration and Testing (2-3 days)
- [ ] Integrate with existing MoRAG services
- [ ] Update database schema
- [ ] Comprehensive testing
- [ ] Performance validation

## Success Criteria
- [ ] >99% uptime for AI operations with resilience patterns
- [ ] Automatic recovery from transient failures
- [ ] Graceful degradation when primary services fail
- [ ] Comprehensive error logging and monitoring
- [ ] Circuit breaker prevents cascade failures
- [ ] Fallback mechanisms work correctly
- [ ] Performance impact <5% overhead

## Dependencies
- Enhanced logging framework
- Health monitoring infrastructure
- Fallback AI service providers
- Updated database schema
- Monitoring and alerting systems

## Risks and Mitigation
- **Risk**: Resilience patterns adding too much complexity
  - **Mitigation**: Simple, well-tested patterns, comprehensive documentation
- **Risk**: Fallback services having different capabilities
  - **Mitigation**: Capability mapping, graceful degradation strategies
- **Risk**: Circuit breaker being too sensitive or not sensitive enough
  - **Mitigation**: Configurable thresholds, monitoring and tuning
