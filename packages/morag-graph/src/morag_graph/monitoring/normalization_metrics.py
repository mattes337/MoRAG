"""Monitoring and metrics for entity normalization."""

import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class NormalizationMetrics:
    """Metrics for entity normalization operations."""
    
    # Basic counters
    entities_processed: int = 0
    entities_normalized: int = 0
    merge_candidates_found: int = 0
    merges_applied: int = 0
    
    # Rule usage tracking
    normalization_rules_used: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Confidence tracking
    confidence_scores: List[float] = field(default_factory=list)
    
    # Performance tracking
    processing_times: List[float] = field(default_factory=list)
    error_count: int = 0
    
    # Method tracking
    normalization_methods: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Language tracking
    languages_processed: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Entity type tracking
    entity_types_processed: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    @property
    def average_confidence(self) -> float:
        """Calculate average confidence score."""
        if not self.confidence_scores:
            return 0.0
        return sum(self.confidence_scores) / len(self.confidence_scores)
    
    @property
    def average_processing_time(self) -> float:
        """Calculate average processing time in seconds."""
        if not self.processing_times:
            return 0.0
        return sum(self.processing_times) / len(self.processing_times)
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate as percentage."""
        if self.entities_processed == 0:
            return 0.0
        return (self.error_count / self.entities_processed) * 100
    
    @property
    def normalization_rate(self) -> float:
        """Calculate normalization rate as percentage."""
        if self.entities_processed == 0:
            return 0.0
        return (self.entities_normalized / self.entities_processed) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'entities_processed': self.entities_processed,
            'entities_normalized': self.entities_normalized,
            'merge_candidates_found': self.merge_candidates_found,
            'merges_applied': self.merges_applied,
            'normalization_rules_used': dict(self.normalization_rules_used),
            'average_confidence': self.average_confidence,
            'average_processing_time': self.average_processing_time,
            'error_rate': self.error_rate,
            'normalization_rate': self.normalization_rate,
            'normalization_methods': dict(self.normalization_methods),
            'languages_processed': dict(self.languages_processed),
            'entity_types_processed': dict(self.entity_types_processed),
            'total_processing_time': sum(self.processing_times),
            'error_count': self.error_count
        }


class NormalizationMonitor:
    """Monitor for tracking entity normalization metrics."""
    
    def __init__(self, enabled: bool = True):
        """Initialize normalization monitor.
        
        Args:
            enabled: Whether monitoring is enabled
        """
        self.enabled = enabled
        self.metrics = NormalizationMetrics()
        self.session_start_time = time.time()
        
        if self.enabled:
            logger.info("Normalization monitoring enabled")
    
    def record_entity_processed(self, entity_name: str, entity_type: Optional[str] = None, language: Optional[str] = None):
        """Record that an entity was processed.
        
        Args:
            entity_name: Name of the entity
            entity_type: Type of the entity
            language: Language of the entity
        """
        if not self.enabled:
            return
        
        self.metrics.entities_processed += 1
        
        if entity_type:
            self.metrics.entity_types_processed[entity_type] += 1
        
        if language:
            self.metrics.languages_processed[language] += 1
    
    def record_entity_normalized(self, original: str, normalized: str, confidence: float, method: str, rule_applied: Optional[str] = None):
        """Record that an entity was normalized.
        
        Args:
            original: Original entity name
            normalized: Normalized entity name
            confidence: Confidence score
            method: Normalization method used
            rule_applied: Specific rule applied (if any)
        """
        if not self.enabled:
            return
        
        self.metrics.entities_normalized += 1
        self.metrics.confidence_scores.append(confidence)
        self.metrics.normalization_methods[method] += 1
        
        if rule_applied:
            self.metrics.normalization_rules_used[rule_applied] += 1
        
        logger.debug(
            "Entity normalized",
            original=original,
            normalized=normalized,
            confidence=confidence,
            method=method,
            rule_applied=rule_applied
        )
    
    def record_merge_candidates(self, count: int):
        """Record merge candidates found.
        
        Args:
            count: Number of merge candidates found
        """
        if not self.enabled:
            return
        
        self.metrics.merge_candidates_found += count
        
        logger.debug("Merge candidates found", count=count)
    
    def record_merges_applied(self, count: int):
        """Record merges applied.
        
        Args:
            count: Number of merges applied
        """
        if not self.enabled:
            return
        
        self.metrics.merges_applied += count
        
        logger.info("Merges applied", count=count)
    
    def record_processing_time(self, duration: float):
        """Record processing time.
        
        Args:
            duration: Processing duration in seconds
        """
        if not self.enabled:
            return
        
        self.metrics.processing_times.append(duration)
    
    def record_error(self, error: Exception, context: Optional[str] = None):
        """Record an error.
        
        Args:
            error: The error that occurred
            context: Optional context information
        """
        if not self.enabled:
            return
        
        self.metrics.error_count += 1
        
        logger.error(
            "Normalization error recorded",
            error=str(error),
            error_type=type(error).__name__,
            context=context
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.
        
        Returns:
            Dictionary of current metrics
        """
        metrics_dict = self.metrics.to_dict()
        
        # Add session information
        session_duration = time.time() - self.session_start_time
        metrics_dict['session_duration'] = session_duration
        metrics_dict['entities_per_second'] = (
            self.metrics.entities_processed / session_duration 
            if session_duration > 0 else 0
        )
        
        return metrics_dict
    
    def log_summary(self):
        """Log a summary of current metrics."""
        if not self.enabled:
            return
        
        metrics = self.get_metrics()
        
        logger.info(
            "Normalization metrics summary",
            entities_processed=metrics['entities_processed'],
            entities_normalized=metrics['entities_normalized'],
            merge_candidates_found=metrics['merge_candidates_found'],
            merges_applied=metrics['merges_applied'],
            average_confidence=f"{metrics['average_confidence']:.3f}",
            normalization_rate=f"{metrics['normalization_rate']:.1f}%",
            error_rate=f"{metrics['error_rate']:.1f}%",
            session_duration=f"{metrics['session_duration']:.1f}s",
            entities_per_second=f"{metrics['entities_per_second']:.1f}"
        )
    
    def reset_metrics(self):
        """Reset all metrics."""
        if not self.enabled:
            return
        
        self.metrics = NormalizationMetrics()
        self.session_start_time = time.time()
        
        logger.info("Normalization metrics reset")


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, monitor: NormalizationMonitor, operation_name: str = "operation"):
        """Initialize performance timer.
        
        Args:
            monitor: Normalization monitor to record timing
            operation_name: Name of the operation being timed
        """
        self.monitor = monitor
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and record duration."""
        if self.start_time is not None:
            duration = time.time() - self.start_time
            self.monitor.record_processing_time(duration)
            
            logger.debug(
                "Operation completed",
                operation=self.operation_name,
                duration=f"{duration:.3f}s"
            )


class MetricsAggregator:
    """Aggregator for combining metrics from multiple sources."""
    
    def __init__(self):
        """Initialize metrics aggregator."""
        self.aggregated_metrics = NormalizationMetrics()
    
    def add_metrics(self, metrics: NormalizationMetrics):
        """Add metrics to the aggregation.
        
        Args:
            metrics: Metrics to add
        """
        # Add counters
        self.aggregated_metrics.entities_processed += metrics.entities_processed
        self.aggregated_metrics.entities_normalized += metrics.entities_normalized
        self.aggregated_metrics.merge_candidates_found += metrics.merge_candidates_found
        self.aggregated_metrics.merges_applied += metrics.merges_applied
        self.aggregated_metrics.error_count += metrics.error_count
        
        # Combine rule usage
        for rule, count in metrics.normalization_rules_used.items():
            self.aggregated_metrics.normalization_rules_used[rule] += count
        
        # Combine method usage
        for method, count in metrics.normalization_methods.items():
            self.aggregated_metrics.normalization_methods[method] += count
        
        # Combine language tracking
        for language, count in metrics.languages_processed.items():
            self.aggregated_metrics.languages_processed[language] += count
        
        # Combine entity type tracking
        for entity_type, count in metrics.entity_types_processed.items():
            self.aggregated_metrics.entity_types_processed[entity_type] += count
        
        # Extend lists
        self.aggregated_metrics.confidence_scores.extend(metrics.confidence_scores)
        self.aggregated_metrics.processing_times.extend(metrics.processing_times)
    
    def get_aggregated_metrics(self) -> Dict[str, Any]:
        """Get aggregated metrics.
        
        Returns:
            Dictionary of aggregated metrics
        """
        return self.aggregated_metrics.to_dict()
    
    def reset(self):
        """Reset aggregated metrics."""
        self.aggregated_metrics = NormalizationMetrics()


# Global monitor instance
_global_monitor: Optional[NormalizationMonitor] = None


def get_global_monitor() -> NormalizationMonitor:
    """Get the global normalization monitor.
    
    Returns:
        Global NormalizationMonitor instance
    """
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = NormalizationMonitor()
    
    return _global_monitor


def enable_monitoring():
    """Enable global monitoring."""
    monitor = get_global_monitor()
    monitor.enabled = True
    logger.info("Global normalization monitoring enabled")


def disable_monitoring():
    """Disable global monitoring."""
    monitor = get_global_monitor()
    monitor.enabled = False
    logger.info("Global normalization monitoring disabled")


def get_global_metrics() -> Dict[str, Any]:
    """Get global metrics.
    
    Returns:
        Dictionary of global metrics
    """
    return get_global_monitor().get_metrics()


def log_global_summary():
    """Log global metrics summary."""
    get_global_monitor().log_summary()


def reset_global_metrics():
    """Reset global metrics."""
    get_global_monitor().reset_metrics()
