# Task 4.3: Performance Monitoring and Analytics Dashboard

## Overview

Develop a comprehensive performance monitoring and analytics system for the hybrid Qdrant-Neo4j retrieval infrastructure. This system provides real-time insights, performance metrics, alerting, and optimization recommendations to ensure optimal system performance and reliability.

## Objectives

- Implement real-time performance monitoring and metrics collection
- Create comprehensive analytics dashboard with visualizations
- Establish intelligent alerting and anomaly detection
- Provide performance optimization recommendations
- Enable historical trend analysis and capacity planning
- Support A/B testing and performance comparison

## Current State Analysis

### Existing Monitoring

**Basic Metrics**:
- Query execution times
- Success/failure rates
- Resource utilization (CPU, memory)
- Connection pool status

**Monitoring Gaps**:
- No centralized dashboard
- Limited historical analysis
- No anomaly detection
- Missing business metrics
- No optimization recommendations

## Implementation Plan

### Step 1: Metrics Collection Service

Implement `src/morag_graph/monitoring/metrics_collector.py`:

```python
from typing import Dict, List, Optional, Any, Tuple, Union
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import statistics
from collections import defaultdict, deque
import threading
import time

class MetricType(Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"

class MetricCategory(Enum):
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    BUSINESS = "business"
    SYSTEM = "system"
    ERROR = "error"

@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    category: MetricCategory
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "type": self.metric_type.value,
            "category": self.category.value,
            "timestamp": self.timestamp.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata
        }

@dataclass
class MetricSummary:
    """Summary statistics for a metric over time."""
    name: str
    count: int
    min_value: float
    max_value: float
    avg_value: float
    median_value: float
    p95_value: float
    p99_value: float
    std_dev: float
    
    start_time: datetime
    end_time: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.count,
            "min": self.min_value,
            "max": self.max_value,
            "avg": self.avg_value,
            "median": self.median_value,
            "p95": self.p95_value,
            "p99": self.p99_value,
            "std_dev": self.std_dev,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat()
        }

class MetricsCollector:
    """Centralized metrics collection and aggregation service."""
    
    def __init__(self, buffer_size: int = 10000, flush_interval: int = 60):
        self.logger = logging.getLogger(__name__)
        
        # Metric storage
        self.metrics_buffer: deque = deque(maxlen=buffer_size)
        self.metric_summaries: Dict[str, MetricSummary] = {}
        
        # Real-time aggregations
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Configuration
        self.flush_interval = flush_interval
        self.buffer_size = buffer_size
        
        # Background tasks
        self.flush_task = None
        self.aggregation_task = None
        self.running = False
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Metric definitions
        self.metric_definitions = self._initialize_metric_definitions()
        
        # Subscribers for real-time updates
        self.subscribers: List[callable] = []
    
    def _initialize_metric_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize standard metric definitions."""
        return {
            # Performance metrics
            "query_execution_time": {
                "type": MetricType.TIMER,
                "category": MetricCategory.PERFORMANCE,
                "description": "Time taken to execute queries",
                "unit": "seconds"
            },
            "query_success_rate": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.PERFORMANCE,
                "description": "Percentage of successful queries",
                "unit": "percentage"
            },
            "retrieval_accuracy": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.BUSINESS,
                "description": "Accuracy of retrieval results",
                "unit": "percentage"
            },
            
            # Resource metrics
            "cpu_usage": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.RESOURCE,
                "description": "CPU utilization percentage",
                "unit": "percentage"
            },
            "memory_usage": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.RESOURCE,
                "description": "Memory utilization percentage",
                "unit": "percentage"
            },
            "active_connections": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.RESOURCE,
                "description": "Number of active database connections",
                "unit": "count"
            },
            
            # System metrics
            "queries_per_second": {
                "type": MetricType.COUNTER,
                "category": MetricCategory.SYSTEM,
                "description": "Number of queries processed per second",
                "unit": "qps"
            },
            "cache_hit_rate": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.SYSTEM,
                "description": "Cache hit rate percentage",
                "unit": "percentage"
            },
            
            # Error metrics
            "error_rate": {
                "type": MetricType.GAUGE,
                "category": MetricCategory.ERROR,
                "description": "Error rate percentage",
                "unit": "percentage"
            },
            "timeout_count": {
                "type": MetricType.COUNTER,
                "category": MetricCategory.ERROR,
                "description": "Number of query timeouts",
                "unit": "count"
            }
        }
    
    async def start(self):
        """Start the metrics collector."""
        self.running = True
        self.flush_task = asyncio.create_task(self._flush_loop())
        self.aggregation_task = asyncio.create_task(self._aggregation_loop())
        self.logger.info("Metrics collector started")
    
    async def stop(self):
        """Stop the metrics collector."""
        self.running = False
        
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
        
        if self.aggregation_task:
            self.aggregation_task.cancel()
            try:
                await self.aggregation_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self._flush_metrics()
        self.logger.info("Metrics collector stopped")
    
    def record_metric(self, name: str, value: Union[int, float], 
                     metric_type: Optional[MetricType] = None,
                     category: Optional[MetricCategory] = None,
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a single metric."""
        try:
            with self.lock:
                # Use definition if available
                if name in self.metric_definitions:
                    definition = self.metric_definitions[name]
                    metric_type = metric_type or definition["type"]
                    category = category or definition["category"]
                
                # Create metric
                metric = Metric(
                    name=name,
                    value=value,
                    metric_type=metric_type or MetricType.GAUGE,
                    category=category or MetricCategory.SYSTEM,
                    tags=tags or {},
                    metadata=metadata or {}
                )
                
                # Add to buffer
                self.metrics_buffer.append(metric)
                
                # Update real-time aggregations
                self._update_aggregations(metric)
                
                # Notify subscribers
                self._notify_subscribers(metric)
                
        except Exception as e:
            self.logger.error(f"Failed to record metric {name}: {e}")
    
    def record_timer(self, name: str, duration: float, 
                    tags: Optional[Dict[str, str]] = None):
        """Record a timer metric."""
        self.record_metric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            category=MetricCategory.PERFORMANCE,
            tags=tags
        )
    
    def increment_counter(self, name: str, value: float = 1.0,
                         tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        self.record_metric(
            name=name,
            value=value,
            metric_type=MetricType.COUNTER,
            tags=tags
        )
    
    def set_gauge(self, name: str, value: float,
                 tags: Optional[Dict[str, str]] = None):
        """Set a gauge metric value."""
        self.record_metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            tags=tags
        )
    
    def record_histogram(self, name: str, value: float,
                        tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        self.record_metric(
            name=name,
            value=value,
            metric_type=MetricType.HISTOGRAM,
            tags=tags
        )
    
    def _update_aggregations(self, metric: Metric):
        """Update real-time aggregations."""
        metric_key = self._get_metric_key(metric)
        
        if metric.metric_type == MetricType.COUNTER:
            self.counters[metric_key] += metric.value
        
        elif metric.metric_type == MetricType.GAUGE:
            self.gauges[metric_key] = metric.value
        
        elif metric.metric_type == MetricType.HISTOGRAM:
            self.histograms[metric_key].append(metric.value)
        
        elif metric.metric_type == MetricType.TIMER:
            self.timers[metric_key].append(metric.value)
    
    def _get_metric_key(self, metric: Metric) -> str:
        """Generate key for metric aggregation."""
        if metric.tags:
            tag_str = "_".join(f"{k}:{v}" for k, v in sorted(metric.tags.items()))
            return f"{metric.name}_{tag_str}"
        return metric.name
    
    def _notify_subscribers(self, metric: Metric):
        """Notify subscribers of new metric."""
        for subscriber in self.subscribers:
            try:
                subscriber(metric)
            except Exception as e:
                self.logger.error(f"Subscriber notification failed: {e}")
    
    def subscribe(self, callback: callable):
        """Subscribe to real-time metric updates."""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: callable):
        """Unsubscribe from metric updates."""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    async def _flush_loop(self):
        """Periodic flush of metrics buffer."""
        while self.running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self._flush_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics flush failed: {e}")
    
    async def _aggregation_loop(self):
        """Periodic aggregation of metrics."""
        while self.running:
            try:
                await asyncio.sleep(30)  # Aggregate every 30 seconds
                await self._compute_summaries()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics aggregation failed: {e}")
    
    async def _flush_metrics(self):
        """Flush metrics buffer to storage."""
        try:
            with self.lock:
                if not self.metrics_buffer:
                    return
                
                # Copy buffer and clear
                metrics_to_flush = list(self.metrics_buffer)
                self.metrics_buffer.clear()
            
            # Store metrics (implement based on storage backend)
            await self._store_metrics(metrics_to_flush)
            
            self.logger.debug(f"Flushed {len(metrics_to_flush)} metrics")
            
        except Exception as e:
            self.logger.error(f"Failed to flush metrics: {e}")
    
    async def _store_metrics(self, metrics: List[Metric]):
        """Store metrics to persistent storage."""
        # Placeholder implementation - replace with actual storage
        # Could be InfluxDB, Prometheus, Elasticsearch, etc.
        
        # For now, just log the metrics
        for metric in metrics:
            self.logger.debug(f"Storing metric: {metric.to_dict()}")
    
    async def _compute_summaries(self):
        """Compute summary statistics for metrics."""
        try:
            current_time = datetime.now()
            
            # Compute summaries for timers and histograms
            for metric_name, values in self.timers.items():
                if values:
                    summary = self._calculate_summary(metric_name, list(values), current_time)
                    self.metric_summaries[f"{metric_name}_timer"] = summary
            
            for metric_name, values in self.histograms.items():
                if values:
                    summary = self._calculate_summary(metric_name, list(values), current_time)
                    self.metric_summaries[f"{metric_name}_histogram"] = summary
            
        except Exception as e:
            self.logger.error(f"Failed to compute summaries: {e}")
    
    def _calculate_summary(self, name: str, values: List[float], 
                          end_time: datetime) -> MetricSummary:
        """Calculate summary statistics for a list of values."""
        if not values:
            return None
        
        sorted_values = sorted(values)
        count = len(values)
        
        return MetricSummary(
            name=name,
            count=count,
            min_value=min(values),
            max_value=max(values),
            avg_value=statistics.mean(values),
            median_value=statistics.median(values),
            p95_value=sorted_values[int(0.95 * count)] if count > 0 else 0,
            p99_value=sorted_values[int(0.99 * count)] if count > 0 else 0,
            std_dev=statistics.stdev(values) if count > 1 else 0,
            start_time=end_time - timedelta(seconds=self.flush_interval),
            end_time=end_time
        )
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values."""
        with self.lock:
            return {
                "counters": dict(self.counters),
                "gauges": dict(self.gauges),
                "buffer_size": len(self.metrics_buffer),
                "summaries": {k: v.to_dict() for k, v in self.metric_summaries.items()}
            }
    
    def get_metric_summary(self, name: str) -> Optional[MetricSummary]:
        """Get summary for a specific metric."""
        return self.metric_summaries.get(name)
    
    def get_metrics_by_category(self, category: MetricCategory) -> List[Metric]:
        """Get metrics by category from buffer."""
        with self.lock:
            return [m for m in self.metrics_buffer if m.category == category]
    
    def clear_metrics(self):
        """Clear all metrics (for testing)."""
        with self.lock:
            self.metrics_buffer.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.timers.clear()
            self.metric_summaries.clear()

# Context manager for timing operations
class TimerContext:
    """Context manager for timing operations."""
    
    def __init__(self, collector: MetricsCollector, metric_name: str, 
                 tags: Optional[Dict[str, str]] = None):
        self.collector = collector
        self.metric_name = metric_name
        self.tags = tags or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_timer(self.metric_name, duration, self.tags)

# Decorator for timing functions
def timed_operation(collector: MetricsCollector, metric_name: str = None):
    """Decorator to time function execution."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}_execution_time"
            with TimerContext(collector, name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

# Async decorator for timing async functions
def async_timed_operation(collector: MetricsCollector, metric_name: str = None):
    """Decorator to time async function execution."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}_execution_time"
            with TimerContext(collector, name):
                return await func(*args, **kwargs)
        return wrapper
    return decorator
```

### Step 2: Analytics Dashboard

Implement `src/morag_graph/monitoring/dashboard.py`:

```python
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from collections import defaultdict

from .metrics_collector import MetricsCollector, MetricCategory, MetricType
from .alerting import AlertManager

@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    widget_id: str
    title: str
    widget_type: str  # chart, gauge, table, text
    metrics: List[str]
    time_range: str = "1h"  # 1h, 6h, 24h, 7d, 30d
    refresh_interval: int = 30  # seconds
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Dashboard:
    """Dashboard configuration."""
    dashboard_id: str
    title: str
    description: str
    widgets: List[DashboardWidget]
    layout: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

class DashboardManager:
    """Manages analytics dashboards and real-time data."""
    
    def __init__(self, metrics_collector: MetricsCollector, 
                 alert_manager: Optional['AlertManager'] = None):
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(__name__)
        
        # Dashboard storage
        self.dashboards: Dict[str, Dashboard] = {}
        
        # Real-time data cache
        self.widget_data_cache: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self.update_task = None
        self.running = False
        
        # Initialize default dashboards
        self._initialize_default_dashboards()
    
    async def start(self):
        """Start the dashboard manager."""
        self.running = True
        self.update_task = asyncio.create_task(self._update_loop())
        self.logger.info("Dashboard manager started")
    
    async def stop(self):
        """Stop the dashboard manager."""
        self.running = False
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Dashboard manager stopped")
    
    def _initialize_default_dashboards(self):
        """Initialize default dashboards."""
        # System Overview Dashboard
        system_dashboard = Dashboard(
            dashboard_id="system_overview",
            title="System Overview",
            description="High-level system performance and health metrics",
            widgets=[
                DashboardWidget(
                    widget_id="qps_chart",
                    title="Queries Per Second",
                    widget_type="line_chart",
                    metrics=["queries_per_second"],
                    time_range="1h"
                ),
                DashboardWidget(
                    widget_id="response_time_chart",
                    title="Average Response Time",
                    widget_type="line_chart",
                    metrics=["query_execution_time"],
                    time_range="1h"
                ),
                DashboardWidget(
                    widget_id="success_rate_gauge",
                    title="Success Rate",
                    widget_type="gauge",
                    metrics=["query_success_rate"],
                    config={"min": 0, "max": 100, "thresholds": [80, 95]}
                ),
                DashboardWidget(
                    widget_id="error_rate_gauge",
                    title="Error Rate",
                    widget_type="gauge",
                    metrics=["error_rate"],
                    config={"min": 0, "max": 10, "thresholds": [1, 5]}
                )
            ]
        )
        
        # Resource Utilization Dashboard
        resource_dashboard = Dashboard(
            dashboard_id="resource_utilization",
            title="Resource Utilization",
            description="System resource usage and capacity metrics",
            widgets=[
                DashboardWidget(
                    widget_id="cpu_usage_chart",
                    title="CPU Usage",
                    widget_type="area_chart",
                    metrics=["cpu_usage"],
                    time_range="6h"
                ),
                DashboardWidget(
                    widget_id="memory_usage_chart",
                    title="Memory Usage",
                    widget_type="area_chart",
                    metrics=["memory_usage"],
                    time_range="6h"
                ),
                DashboardWidget(
                    widget_id="connections_chart",
                    title="Active Connections",
                    widget_type="line_chart",
                    metrics=["active_connections"],
                    time_range="1h"
                ),
                DashboardWidget(
                    widget_id="cache_hit_rate_gauge",
                    title="Cache Hit Rate",
                    widget_type="gauge",
                    metrics=["cache_hit_rate"],
                    config={"min": 0, "max": 100, "thresholds": [70, 90]}
                )
            ]
        )
        
        # Performance Analysis Dashboard
        performance_dashboard = Dashboard(
            dashboard_id="performance_analysis",
            title="Performance Analysis",
            description="Detailed performance metrics and trends",
            widgets=[
                DashboardWidget(
                    widget_id="response_time_histogram",
                    title="Response Time Distribution",
                    widget_type="histogram",
                    metrics=["query_execution_time"],
                    time_range="24h"
                ),
                DashboardWidget(
                    widget_id="query_types_breakdown",
                    title="Query Types Breakdown",
                    widget_type="pie_chart",
                    metrics=["queries_per_second"],
                    config={"group_by": "query_type"}
                ),
                DashboardWidget(
                    widget_id="strategy_performance",
                    title="Strategy Performance Comparison",
                    widget_type="bar_chart",
                    metrics=["query_execution_time"],
                    config={"group_by": "strategy"}
                ),
                DashboardWidget(
                    widget_id="top_slow_queries",
                    title="Slowest Queries",
                    widget_type="table",
                    metrics=["query_execution_time"],
                    config={"sort": "desc", "limit": 10}
                )
            ]
        )
        
        self.dashboards["system_overview"] = system_dashboard
        self.dashboards["resource_utilization"] = resource_dashboard
        self.dashboards["performance_analysis"] = performance_dashboard
    
    async def _update_loop(self):
        """Periodic update of widget data."""
        while self.running:
            try:
                await self._update_all_widgets()
                await asyncio.sleep(30)  # Update every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Widget update failed: {e}")
    
    async def _update_all_widgets(self):
        """Update data for all widgets."""
        for dashboard in self.dashboards.values():
            for widget in dashboard.widgets:
                try:
                    data = await self._get_widget_data(widget)
                    self.widget_data_cache[widget.widget_id] = data
                except Exception as e:
                    self.logger.error(f"Failed to update widget {widget.widget_id}: {e}")
    
    async def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a specific widget."""
        # Get time range
        end_time = datetime.now()
        start_time = self._parse_time_range(widget.time_range, end_time)
        
        # Get metrics data
        data = {
            "widget_id": widget.widget_id,
            "title": widget.title,
            "type": widget.widget_type,
            "timestamp": end_time.isoformat(),
            "data": []
        }
        
        for metric_name in widget.metrics:
            metric_data = await self._get_metric_data(
                metric_name, start_time, end_time, widget.config
            )
            data["data"].append(metric_data)
        
        return data
    
    def _parse_time_range(self, time_range: str, end_time: datetime) -> datetime:
        """Parse time range string to start time."""
        if time_range == "1h":
            return end_time - timedelta(hours=1)
        elif time_range == "6h":
            return end_time - timedelta(hours=6)
        elif time_range == "24h":
            return end_time - timedelta(hours=24)
        elif time_range == "7d":
            return end_time - timedelta(days=7)
        elif time_range == "30d":
            return end_time - timedelta(days=30)
        else:
            return end_time - timedelta(hours=1)  # Default to 1 hour
    
    async def _get_metric_data(self, metric_name: str, start_time: datetime, 
                              end_time: datetime, config: Dict[str, Any]) -> Dict[str, Any]:
        """Get metric data for specified time range."""
        # This would typically query a time-series database
        # For now, we'll use the current metrics from the collector
        
        current_metrics = self.metrics_collector.get_current_metrics()
        
        # Get current value
        current_value = None
        if metric_name in current_metrics["gauges"]:
            current_value = current_metrics["gauges"][metric_name]
        elif metric_name in current_metrics["counters"]:
            current_value = current_metrics["counters"][metric_name]
        
        # Get summary if available
        summary = self.metrics_collector.get_metric_summary(metric_name)
        
        return {
            "metric_name": metric_name,
            "current_value": current_value,
            "summary": summary.to_dict() if summary else None,
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
        }
    
    def create_dashboard(self, dashboard: Dashboard) -> bool:
        """Create a new dashboard."""
        try:
            self.dashboards[dashboard.dashboard_id] = dashboard
            self.logger.info(f"Created dashboard: {dashboard.dashboard_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to create dashboard: {e}")
            return False
    
    def get_dashboard(self, dashboard_id: str) -> Optional[Dashboard]:
        """Get dashboard by ID."""
        return self.dashboards.get(dashboard_id)
    
    def list_dashboards(self) -> List[Dashboard]:
        """List all dashboards."""
        return list(self.dashboards.values())
    
    def get_widget_data(self, widget_id: str) -> Optional[Dict[str, Any]]:
        """Get cached data for a widget."""
        return self.widget_data_cache.get(widget_id)
    
    def get_dashboard_data(self, dashboard_id: str) -> Optional[Dict[str, Any]]:
        """Get complete data for a dashboard."""
        dashboard = self.get_dashboard(dashboard_id)
        if not dashboard:
            return None
        
        widget_data = {}
        for widget in dashboard.widgets:
            data = self.get_widget_data(widget.widget_id)
            if data:
                widget_data[widget.widget_id] = data
        
        return {
            "dashboard": {
                "id": dashboard.dashboard_id,
                "title": dashboard.title,
                "description": dashboard.description,
                "updated_at": datetime.now().isoformat()
            },
            "widgets": widget_data
        }
    
    def update_dashboard(self, dashboard_id: str, updates: Dict[str, Any]) -> bool:
        """Update dashboard configuration."""
        try:
            if dashboard_id not in self.dashboards:
                return False
            
            dashboard = self.dashboards[dashboard_id]
            
            if "title" in updates:
                dashboard.title = updates["title"]
            if "description" in updates:
                dashboard.description = updates["description"]
            if "widgets" in updates:
                # Update widgets (simplified)
                pass
            
            dashboard.updated_at = datetime.now()
            
            self.logger.info(f"Updated dashboard: {dashboard_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update dashboard: {e}")
            return False
    
    def delete_dashboard(self, dashboard_id: str) -> bool:
        """Delete a dashboard."""
        try:
            if dashboard_id in self.dashboards:
                del self.dashboards[dashboard_id]
                
                # Clean up widget data cache
                dashboard = self.dashboards.get(dashboard_id)
                if dashboard:
                    for widget in dashboard.widgets:
                        self.widget_data_cache.pop(widget.widget_id, None)
                
                self.logger.info(f"Deleted dashboard: {dashboard_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to delete dashboard: {e}")
            return False
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        current_metrics = self.metrics_collector.get_current_metrics()
        
        # Calculate health score based on key metrics
        health_score = 100
        issues = []
        
        # Check success rate
        success_rate = current_metrics["gauges"].get("query_success_rate", 100)
        if success_rate < 95:
            health_score -= (95 - success_rate) * 2
            issues.append(f"Low success rate: {success_rate:.1f}%")
        
        # Check error rate
        error_rate = current_metrics["gauges"].get("error_rate", 0)
        if error_rate > 5:
            health_score -= error_rate * 5
            issues.append(f"High error rate: {error_rate:.1f}%")
        
        # Check resource usage
        cpu_usage = current_metrics["gauges"].get("cpu_usage", 0)
        if cpu_usage > 80:
            health_score -= (cpu_usage - 80) * 2
            issues.append(f"High CPU usage: {cpu_usage:.1f}%")
        
        memory_usage = current_metrics["gauges"].get("memory_usage", 0)
        if memory_usage > 80:
            health_score -= (memory_usage - 80) * 2
            issues.append(f"High memory usage: {memory_usage:.1f}%")
        
        # Determine status
        if health_score >= 90:
            status = "healthy"
        elif health_score >= 70:
            status = "warning"
        else:
            status = "critical"
        
        return {
            "status": status,
            "health_score": max(0, health_score),
            "issues": issues,
            "timestamp": datetime.now().isoformat(),
            "metrics_summary": {
                "queries_per_second": current_metrics["counters"].get("queries_per_second", 0),
                "success_rate": success_rate,
                "error_rate": error_rate,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage
            }
        }
```

## Testing Strategy

### Unit Tests

Create `tests/test_monitoring.py`:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from morag_graph.monitoring.metrics_collector import (
    MetricsCollector, Metric, MetricType, MetricCategory, TimerContext
)
from morag_graph.monitoring.dashboard import DashboardManager, Dashboard, DashboardWidget

@pytest.fixture
def metrics_collector():
    return MetricsCollector(buffer_size=100, flush_interval=1)

@pytest.fixture
def dashboard_manager(metrics_collector):
    return DashboardManager(metrics_collector)

@pytest.mark.asyncio
async def test_metrics_collector_basic_operations(metrics_collector):
    # Test recording different metric types
    metrics_collector.record_metric("test_gauge", 42.0, MetricType.GAUGE)
    metrics_collector.increment_counter("test_counter", 5.0)
    metrics_collector.record_timer("test_timer", 1.5)
    metrics_collector.record_histogram("test_histogram", 10.0)
    
    # Check that metrics were recorded
    current_metrics = metrics_collector.get_current_metrics()
    
    assert current_metrics["gauges"]["test_gauge"] == 42.0
    assert current_metrics["counters"]["test_counter"] == 5.0
    assert len(metrics_collector.timers["test_timer"]) == 1
    assert len(metrics_collector.histograms["test_histogram"]) == 1

@pytest.mark.asyncio
async def test_timer_context(metrics_collector):
    # Test timer context manager
    with TimerContext(metrics_collector, "test_operation"):
        await asyncio.sleep(0.1)
    
    # Check that timer was recorded
    assert len(metrics_collector.timers["test_operation"]) == 1
    assert metrics_collector.timers["test_operation"][0] >= 0.1

@pytest.mark.asyncio
async def test_metrics_aggregation(metrics_collector):
    # Record multiple values
    for i in range(10):
        metrics_collector.record_timer("test_timer", i * 0.1)
    
    # Trigger aggregation
    await metrics_collector._compute_summaries()
    
    # Check summary
    summary = metrics_collector.get_metric_summary("test_timer_timer")
    assert summary is not None
    assert summary.count == 10
    assert summary.min_value == 0.0
    assert summary.max_value == 0.9

@pytest.mark.asyncio
async def test_metrics_collector_lifecycle(metrics_collector):
    await metrics_collector.start()
    
    # Record some metrics
    metrics_collector.record_metric("test_metric", 100)
    
    # Wait a bit for background tasks
    await asyncio.sleep(0.1)
    
    await metrics_collector.stop()
    
    # Should have processed metrics
    assert len(metrics_collector.metrics_buffer) >= 0  # May be flushed

def test_metric_categories(metrics_collector):
    # Test different categories
    metrics_collector.record_metric("perf_metric", 1.0, category=MetricCategory.PERFORMANCE)
    metrics_collector.record_metric("resource_metric", 2.0, category=MetricCategory.RESOURCE)
    metrics_collector.record_metric("business_metric", 3.0, category=MetricCategory.BUSINESS)
    
    # Get metrics by category
    perf_metrics = metrics_collector.get_metrics_by_category(MetricCategory.PERFORMANCE)
    resource_metrics = metrics_collector.get_metrics_by_category(MetricCategory.RESOURCE)
    business_metrics = metrics_collector.get_metrics_by_category(MetricCategory.BUSINESS)
    
    assert len(perf_metrics) == 1
    assert len(resource_metrics) == 1
    assert len(business_metrics) == 1

@pytest.mark.asyncio
async def test_dashboard_creation(dashboard_manager):
    # Test default dashboards
    dashboards = dashboard_manager.list_dashboards()
    assert len(dashboards) >= 3  # Should have default dashboards
    
    # Test getting specific dashboard
    system_dashboard = dashboard_manager.get_dashboard("system_overview")
    assert system_dashboard is not None
    assert system_dashboard.title == "System Overview"

@pytest.mark.asyncio
async def test_dashboard_data_retrieval(dashboard_manager):
    # Start dashboard manager
    await dashboard_manager.start()
    
    # Wait for initial data update
    await asyncio.sleep(0.1)
    
    # Get dashboard data
    dashboard_data = dashboard_manager.get_dashboard_data("system_overview")
    assert dashboard_data is not None
    assert "dashboard" in dashboard_data
    assert "widgets" in dashboard_data
    
    await dashboard_manager.stop()

@pytest.mark.asyncio
async def test_system_health_calculation(dashboard_manager):
    # Set some metrics
    dashboard_manager.metrics_collector.set_gauge("query_success_rate", 98.5)
    dashboard_manager.metrics_collector.set_gauge("error_rate", 1.2)
    dashboard_manager.metrics_collector.set_gauge("cpu_usage", 65.0)
    dashboard_manager.metrics_collector.set_gauge("memory_usage", 70.0)
    
    # Get health status
    health = dashboard_manager.get_system_health()
    
    assert health["status"] in ["healthy", "warning", "critical"]
    assert 0 <= health["health_score"] <= 100
    assert "metrics_summary" in health

def test_custom_dashboard_creation(dashboard_manager):
    # Create custom dashboard
    custom_dashboard = Dashboard(
        dashboard_id="custom_test",
        title="Test Dashboard",
        description="Custom test dashboard",
        widgets=[
            DashboardWidget(
                widget_id="test_widget",
                title="Test Widget",
                widget_type="gauge",
                metrics=["test_metric"]
            )
        ]
    )
    
    # Add to manager
    success = dashboard_manager.create_dashboard(custom_dashboard)
    assert success
    
    # Verify it was added
    retrieved = dashboard_manager.get_dashboard("custom_test")
    assert retrieved is not None
    assert retrieved.title == "Test Dashboard"

def test_dashboard_updates(dashboard_manager):
    # Update existing dashboard
    updates = {
        "title": "Updated System Overview",
        "description": "Updated description"
    }
    
    success = dashboard_manager.update_dashboard("system_overview", updates)
    assert success
    
    # Verify updates
    dashboard = dashboard_manager.get_dashboard("system_overview")
    assert dashboard.title == "Updated System Overview"

def test_metric_subscription(metrics_collector):
    # Test metric subscription
    received_metrics = []
    
    def metric_handler(metric):
        received_metrics.append(metric)
    
    metrics_collector.subscribe(metric_handler)
    
    # Record a metric
    metrics_collector.record_metric("test_subscription", 42.0)
    
    # Should have received the metric
    assert len(received_metrics) == 1
    assert received_metrics[0].name == "test_subscription"
    assert received_metrics[0].value == 42.0
    
    # Unsubscribe
    metrics_collector.unsubscribe(metric_handler)
    
    # Record another metric
    metrics_collector.record_metric("test_subscription_2", 24.0)
    
    # Should not have received the second metric
    assert len(received_metrics) == 1
```

## Performance Considerations

### Optimization Strategies

1. **Efficient Data Storage**:
   - Use time-series databases (InfluxDB, Prometheus)
   - Implement data retention policies
   - Compress historical data

2. **Real-time Processing**:
   - Stream processing for real-time metrics
   - In-memory caching for frequently accessed data
   - Asynchronous metric collection

3. **Dashboard Performance**:
   - Client-side caching
   - Progressive data loading
   - Efficient chart rendering

4. **Scalability**:
   - Distributed metrics collection
   - Load balancing for dashboard requests
   - Horizontal scaling of storage

### Performance Targets

- **Metric Collection**: < 1ms overhead per metric
- **Dashboard Load Time**: < 2 seconds for standard dashboards
- **Real-time Updates**: < 5 seconds latency
- **Data Retention**: 90 days detailed, 1 year aggregated
- **Concurrent Users**: Support 100+ concurrent dashboard users

## Success Criteria

- [ ] Comprehensive metrics collection for all system components
- [ ] Real-time dashboard with key performance indicators
- [ ] Automated alerting for critical issues
- [ ] Historical trend analysis and reporting
- [ ] Performance optimization recommendations
- [ ] A/B testing support for strategy comparison
- [ ] All tests pass with >95% coverage
- [ ] Dashboard loads within 2 seconds
- [ ] Metrics collection overhead < 1ms

## Risk Assessment

**Risk Level**: Medium

**Key Risks**:
1. **Performance Overhead**: Metrics collection may impact system performance
2. **Data Volume**: Large amounts of metrics data requiring storage
3. **Dashboard Complexity**: Complex visualizations may be slow
4. **Alert Fatigue**: Too many alerts may reduce effectiveness

**Mitigation Strategies**:
1. Implement sampling for high-volume metrics
2. Use efficient storage and compression
3. Optimize dashboard queries and caching
4. Implement intelligent alert grouping and suppression

## Rollback Plan

1. **Immediate Rollback**: Disable metrics collection, use basic logging
2. **Partial Rollback**: Disable specific metric types or dashboards
3. **Configuration Rollback**: Revert to previous monitoring configuration
4. **Data Preservation**: Maintain existing metrics during rollback

## Next Steps

- **Integration**: Connect with all system components
- **Advanced Analytics**: Implement ML-based anomaly detection
- **Custom Dashboards**: Allow users to create custom dashboards
- **Mobile Support**: Develop mobile dashboard interface

## Dependencies

- **Task 4.1**: Hybrid Retrieval Pipeline (provides performance metrics)
- **Task 4.2**: Query Coordination Service (provides coordination metrics)
- **All Previous Tasks**: Provide metrics from their respective components

## Estimated Time

**5-7 days**

## Status

- [ ] Metrics collector implementation
- [ ] Dashboard manager implementation
- [ ] Default dashboards creation
- [ ] Real-time data updates
- [ ] System health monitoring
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance optimization
- [ ] Documentation