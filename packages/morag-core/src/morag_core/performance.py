"""Performance monitoring and optimization utilities for MoRAG."""

import time
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self) -> float:
        """Mark operation as finished and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        return self.duration


class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.active_operations: Dict[str, PerformanceMetrics] = {}
    
    def start_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None) -> PerformanceMetrics:
        """Start tracking an operation."""
        metric = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time(),
            metadata=metadata or {}
        )
        self.active_operations[operation_name] = metric
        return metric
    
    def finish_operation(self, operation_name: str) -> Optional[float]:
        """Finish tracking an operation."""
        if operation_name in self.active_operations:
            metric = self.active_operations.pop(operation_name)
            duration = metric.finish()
            self.metrics.append(metric)
            
            logger.info(
                "Operation completed",
                operation=operation_name,
                duration=duration,
                metadata=metric.metadata
            )
            return duration
        return None
    
    @asynccontextmanager
    async def track_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for tracking operations."""
        metric = self.start_operation(operation_name, metadata)
        try:
            yield metric
        finally:
            self.finish_operation(operation_name)
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        if not self.metrics:
            return {"total_operations": 0}
        
        operations_by_name = {}
        for metric in self.metrics:
            if metric.operation_name not in operations_by_name:
                operations_by_name[metric.operation_name] = []
            operations_by_name[metric.operation_name].append(metric.duration)
        
        summary = {"total_operations": len(self.metrics)}
        for op_name, durations in operations_by_name.items():
            summary[op_name] = {
                "count": len(durations),
                "total_time": sum(durations),
                "avg_time": sum(durations) / len(durations),
                "min_time": min(durations),
                "max_time": max(durations)
            }
        
        return summary


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


class PerformanceOptimizer:
    """Optimize performance based on metrics and configuration."""
    
    @staticmethod
    def optimize_batch_size(
        current_batch_size: int,
        processing_time: float,
        target_time: float = 2.0,
        min_batch_size: int = 10,
        max_batch_size: int = 200
    ) -> int:
        """Optimize batch size based on processing time."""
        if processing_time > target_time * 1.5:
            # Too slow, reduce batch size
            new_size = max(min_batch_size, int(current_batch_size * 0.8))
        elif processing_time < target_time * 0.5:
            # Too fast, increase batch size
            new_size = min(max_batch_size, int(current_batch_size * 1.2))
        else:
            # Good performance, keep current size
            new_size = current_batch_size
        
        if new_size != current_batch_size:
            logger.info(
                "Optimizing batch size",
                old_size=current_batch_size,
                new_size=new_size,
                processing_time=processing_time,
                target_time=target_time
            )
        
        return new_size
    
    @staticmethod
    def calculate_optimal_delay(
        batch_size: int,
        rate_limit_per_minute: int,
        safety_factor: float = 0.8
    ) -> float:
        """Calculate optimal delay between batches."""
        # Calculate requests per second with safety factor
        max_rps = (rate_limit_per_minute / 60.0) * safety_factor
        
        # Calculate delay needed between batches
        delay = max(0.01, 1.0 / max_rps)
        
        logger.debug(
            "Calculated optimal delay",
            batch_size=batch_size,
            rate_limit_per_minute=rate_limit_per_minute,
            delay=delay
        )
        
        return delay


async def optimize_async_processing(
    items: List[Any],
    process_func: Callable,
    batch_size: int = 50,
    max_concurrent: int = 5,
    delay_between_batches: float = 0.05
) -> List[Any]:
    """Optimize async processing with batching and concurrency control."""
    
    async def process_batch(batch: List[Any]) -> List[Any]:
        """Process a single batch."""
        tasks = [process_func(item) for item in batch]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    # Split items into batches
    batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
    
    # Process batches with concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)
    results = []
    
    async def process_batch_with_semaphore(batch: List[Any]) -> List[Any]:
        async with semaphore:
            return await process_batch(batch)
    
    # Process all batches
    for i, batch in enumerate(batches):
        batch_results = await process_batch_with_semaphore(batch)
        results.extend(batch_results)
        
        # Add delay between batches (except for the last one)
        if i < len(batches) - 1:
            await asyncio.sleep(delay_between_batches)
    
    return results


def profile_function(func: Callable) -> Callable:
    """Decorator to profile function execution time."""
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            logger.debug(
                "Function profiled",
                function=func.__name__,
                duration=duration,
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
    
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            logger.debug(
                "Async function profiled",
                function=func.__name__,
                duration=duration,
                args_count=len(args),
                kwargs_count=len(kwargs)
            )
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else wrapper
