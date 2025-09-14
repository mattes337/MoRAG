"""Base interfaces for services."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from ..exceptions import ExternalServiceError


class ServiceStatus(str, Enum):
    """Service status enum."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class ServiceConfig:
    """Base configuration for services."""
    enabled: bool = True
    timeout: float = 300.0
    retry_attempts: int = 3
    retry_delay: float = 1.0
    max_concurrent_tasks: int = 5
    custom_options: Dict[str, Any] = field(default_factory=dict)


class BaseService(ABC):
    """Base class for services."""
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the service.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the service and release resources."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check service health.
        
        Returns:
            Dictionary with health status information
        """
        pass


class CircuitBreaker:
    """Circuit breaker pattern implementation for service resilience."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_timeout: float = 30.0
    ):
        """Initialize circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Time in seconds before resetting circuit
            half_open_timeout: Time in seconds before trying half-open state
        """
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        self.failures = 0
        self.state = "closed"  # closed, open, half-open
        self.last_failure_time = 0.0
        self.last_success_time = 0.0
    
    def record_success(self) -> None:
        """Record successful operation."""
        self.failures = 0
        self.state = "closed"
    
    def record_failure(self) -> None:
        """Record failed operation."""
        import time
        self.failures += 1
        self.last_failure_time = time.time()
        
        if self.failures >= self.failure_threshold:
            self.state = "open"
    
    def is_closed(self) -> bool:
        """Check if circuit is closed."""
        import time
        current_time = time.time()
        
        if self.state == "open":
            # Check if reset timeout has passed
            if current_time - self.last_failure_time > self.reset_timeout:
                self.state = "half-open"
                return True
            return False
        elif self.state == "half-open":
            # Allow occasional requests in half-open state
            if current_time - self.last_success_time > self.half_open_timeout:
                return True
            return False
        
        return True