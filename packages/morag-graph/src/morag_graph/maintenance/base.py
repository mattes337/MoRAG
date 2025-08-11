"""Base classes and utilities for maintenance jobs.

Provides standardized error handling, configuration validation, and common patterns
for all maintenance jobs in the MoRAG system.
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Generic, Callable
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class MaintenanceJobError(Exception):
    """Base exception for maintenance jobs."""
    pass


class PartialFailureError(MaintenanceJobError):
    """Raised when job partially fails but can continue."""
    
    def __init__(self, message: str, successful_count: int, failed_count: int, failures: List[Exception]):
        super().__init__(message)
        self.successful_count = successful_count
        self.failed_count = failed_count
        self.failures = failures


class CriticalMaintenanceError(MaintenanceJobError):
    """Raised when job encounters critical failure and must stop."""
    pass


class CircuitBreakerError(MaintenanceJobError):
    """Raised when circuit breaker trips due to repeated failures."""
    pass


class ConfigurationError(MaintenanceJobError):
    """Raised when configuration is invalid."""
    pass


@dataclass
class BatchResult(Generic[T]):
    """Result of a batch operation."""
    successful: List[T] = field(default_factory=list)
    failed: List[Tuple[Any, Exception]] = field(default_factory=list)
    
    @property
    def success_count(self) -> int:
        return len(self.successful)
    
    @property
    def failure_count(self) -> int:
        return len(self.failed)
    
    @property
    def total_count(self) -> int:
        return self.success_count + self.failure_count
    
    @property
    def success_rate(self) -> float:
        if self.total_count == 0:
            return 1.0
        return self.success_count / self.total_count


@dataclass
class CircuitBreakerState:
    """State for circuit breaker pattern."""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    failure_threshold: int = 5
    timeout_seconds: int = 60


class CircuitBreaker:
    """Circuit breaker for LLM and external service calls."""
    
    def __init__(self, failure_threshold: int = 5, timeout_seconds: int = 60):
        self.state = CircuitBreakerState(
            failure_threshold=failure_threshold,
            timeout_seconds=timeout_seconds
        )
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state.state == "OPEN":
            if self._should_attempt_reset():
                self.state.state = "HALF_OPEN"
                logger.info("Circuit breaker attempting reset")
            else:
                raise CircuitBreakerError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            if self.state.state == "HALF_OPEN":
                self._reset()
            return result
        except Exception as e:
            self._record_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        if self.state.last_failure_time is None:
            return True
        
        time_since_failure = datetime.utcnow() - self.state.last_failure_time
        return time_since_failure.total_seconds() >= self.state.timeout_seconds
    
    def _record_failure(self):
        """Record a failure and potentially open the circuit."""
        self.state.failure_count += 1
        self.state.last_failure_time = datetime.utcnow()
        
        if self.state.failure_count >= self.state.failure_threshold:
            self.state.state = "OPEN"
            logger.warning("Circuit breaker opened", 
                         failure_count=self.state.failure_count)
    
    def _reset(self):
        """Reset the circuit breaker to closed state."""
        self.state.failure_count = 0
        self.state.last_failure_time = None
        self.state.state = "CLOSED"
        logger.info("Circuit breaker reset to CLOSED")


class MaintenanceJobBase(ABC):
    """Base class for all maintenance jobs."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=config.get('circuit_breaker_threshold', 5),
            timeout_seconds=config.get('circuit_breaker_timeout', 60)
        )
        self.job_tag = config.get('job_tag', self._generate_job_tag())
        self.dry_run = config.get('dry_run', True)
        self.batch_size = config.get('batch_size', 100)
    
    def _generate_job_tag(self) -> str:
        """Generate a unique job tag for this run."""
        timestamp = int(time.time())
        job_name = self.__class__.__name__.lower()
        return f"{job_name}_{timestamp}"
    
    @abstractmethod
    async def run(self) -> Dict[str, Any]:
        """Run the maintenance job."""
        pass
    
    @abstractmethod
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of errors."""
        pass
    
    async def safe_execute_batch(
        self, 
        batch_items: List[Any], 
        operation: Callable,
        critical_failure_threshold: float = 0.5
    ) -> BatchResult:
        """Execute batch with error handling and partial failure recovery."""
        successful = []
        failed = []
        
        for item in batch_items:
            try:
                result = await operation(item)
                successful.append(result)
            except CriticalMaintenanceError:
                # Critical errors should stop the entire job
                raise
            except Exception as e:
                logger.error("Batch item failed", 
                           item=str(item)[:100], 
                           error=str(e))
                failed.append((item, e))
        
        result = BatchResult(successful=successful, failed=failed)
        
        # Check if failure rate is too high
        if result.failure_count > 0 and result.success_rate < critical_failure_threshold:
            raise PartialFailureError(
                f"Batch failure rate too high: {result.success_rate:.2%}",
                result.success_count,
                result.failure_count,
                [e for _, e in failed]
            )
        
        return result
    
    async def safe_llm_call(self, llm_func: Callable, *args, **kwargs):
        """Execute LLM call with circuit breaker protection."""
        return await self.circuit_breaker.call(llm_func, *args, **kwargs)
    
    def log_job_start(self):
        """Log job start with configuration."""
        logger.info("Maintenance job starting",
                   job_name=self.__class__.__name__,
                   job_tag=self.job_tag,
                   dry_run=self.dry_run,
                   config=self.config)
    
    def log_job_complete(self, result: Dict[str, Any]):
        """Log job completion with results."""
        logger.info("Maintenance job completed",
                   job_name=self.__class__.__name__,
                   job_tag=self.job_tag,
                   result=result)


def validate_positive_int(value: Any, name: str, min_value: int = 1, max_value: Optional[int] = None) -> List[str]:
    """Validate that a value is a positive integer within bounds."""
    errors = []
    
    if not isinstance(value, int):
        errors.append(f"{name} must be an integer")
        return errors
    
    if value < min_value:
        errors.append(f"{name} must be >= {min_value}")
    
    if max_value is not None and value > max_value:
        errors.append(f"{name} must be <= {max_value}")
    
    return errors


def validate_float_range(value: Any, name: str, min_value: float = 0.0, max_value: float = 1.0) -> List[str]:
    """Validate that a value is a float within the specified range."""
    errors = []
    
    if not isinstance(value, (int, float)):
        errors.append(f"{name} must be a number")
        return errors
    
    if value < min_value or value > max_value:
        errors.append(f"{name} must be between {min_value} and {max_value}")
    
    return errors


def validate_string_choice(value: Any, name: str, choices: List[str]) -> List[str]:
    """Validate that a value is one of the allowed string choices."""
    errors = []
    
    if not isinstance(value, str):
        errors.append(f"{name} must be a string")
        return errors
    
    if value not in choices:
        errors.append(f"{name} must be one of: {', '.join(choices)}")
    
    return errors
