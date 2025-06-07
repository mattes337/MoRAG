# Task 11: Monitoring Dashboard

## Objective
Create a comprehensive monitoring dashboard that provides real-time visibility into worker status, task distribution, performance metrics, and system health for the remote GPU worker system.

## Current State Analysis

### Existing Monitoring
- Basic logging with structlog
- Worker health monitoring from Task 2
- Queue statistics from Task 6
- No centralized dashboard or metrics visualization

### Monitoring Requirements
- Real-time worker status and performance metrics
- Task distribution and queue monitoring
- System health and resource utilization
- Performance analytics and trends
- Alert system for critical issues
- Historical data and reporting

## Implementation Plan

### Step 1: Metrics Collection Models

#### 1.1 Create Metrics Models
**File**: `packages/morag-core/src/morag_core/models/metrics.py`

```python
"""Metrics and monitoring models."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class MetricType(str, Enum):
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class SystemMetrics(BaseModel):
    """System-wide metrics."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Worker metrics
    total_workers: int = 0
    online_workers: int = 0
    gpu_workers: int = 0
    cpu_workers: int = 0
    hybrid_workers: int = 0
    
    # Task metrics
    total_tasks: int = 0
    pending_tasks: int = 0
    processing_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    
    # Queue metrics
    gpu_queue_size: int = 0
    cpu_queue_size: int = 0
    fallback_queue_size: int = 0
    
    # Performance metrics
    average_task_duration: float = 0.0
    average_queue_wait_time: float = 0.0
    system_throughput: float = 0.0  # tasks per hour
    
    # Resource metrics
    total_cpu_usage: float = 0.0
    total_memory_usage: float = 0.0
    total_gpu_usage: float = 0.0
    
    # Error metrics
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    retry_rate: float = 0.0

class WorkerMetrics(BaseModel):
    """Individual worker metrics."""
    worker_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Status
    is_online: bool = True
    is_healthy: bool = True
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    
    # Task metrics
    active_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_processing_time: float = 0.0
    average_task_duration: float = 0.0
    
    # Resource metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    disk_usage: float = 0.0
    
    # Network metrics
    network_latency: float = 0.0
    bytes_transferred: int = 0
    
    # Error metrics
    error_count: int = 0
    timeout_count: int = 0
    consecutive_failures: int = 0

class TaskMetrics(BaseModel):
    """Task execution metrics."""
    task_id: str
    worker_id: Optional[str] = None
    content_type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Timing
    submitted_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    queue_wait_time: float = 0.0
    processing_time: float = 0.0
    total_time: float = 0.0
    
    # Status
    status: str = "pending"  # pending, processing, completed, failed
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Resource usage
    cpu_time: float = 0.0
    memory_peak: float = 0.0
    gpu_time: float = 0.0
    
    # File metrics
    input_file_size: int = 0
    output_file_size: int = 0
    
    # Quality metrics
    confidence_score: float = 0.0
    quality_degraded: bool = False

class Alert(BaseModel):
    """System alert."""
    alert_id: str = Field(default_factory=lambda: f"alert_{int(datetime.utcnow().timestamp())}")
    title: str
    description: str
    severity: AlertSeverity
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Context
    component: str  # worker, queue, system, etc.
    component_id: Optional[str] = None
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    
    # Status
    is_active: bool = True
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    
    # Metadata
    tags: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DashboardData(BaseModel):
    """Complete dashboard data."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    system_metrics: SystemMetrics
    worker_metrics: List[WorkerMetrics]
    recent_tasks: List[TaskMetrics]
    active_alerts: List[Alert]
    
    # Historical data
    hourly_stats: List[SystemMetrics] = Field(default_factory=list)
    daily_stats: List[SystemMetrics] = Field(default_factory=list)
```

### Step 2: Metrics Collection Service

#### 2.1 Create Metrics Collector
**File**: `packages/morag/src/morag/services/metrics_collector.py`

```python
"""Metrics collection service for monitoring."""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import structlog
from redis import Redis

from morag_core.models.metrics import (
    SystemMetrics, WorkerMetrics, TaskMetrics, Alert, AlertSeverity,
    DashboardData
)
from morag.services.worker_registry import WorkerRegistry
from morag.services.priority_queue_manager import PriorityQueueManager

logger = structlog.get_logger(__name__)

class MetricsCollector:
    """Collects and aggregates system metrics."""
    
    def __init__(self, redis_client: Redis, worker_registry: WorkerRegistry,
                 queue_manager: PriorityQueueManager):
        self.redis = redis_client
        self.worker_registry = worker_registry
        self.queue_manager = queue_manager
        
        # Internal state
        self.task_metrics: Dict[str, TaskMetrics] = {}
        self.alerts: Dict[str, Alert] = {}
        self.collection_interval = 30  # seconds
        self._collection_task: Optional[asyncio.Task] = None
        
        # Alert thresholds
        self.alert_thresholds = {
            'worker_offline_threshold': 5,  # minutes
            'queue_size_threshold': 100,
            'error_rate_threshold': 0.1,  # 10%
            'cpu_usage_threshold': 0.9,  # 90%
            'memory_usage_threshold': 0.9,  # 90%
            'gpu_usage_threshold': 0.95,  # 95%
        }
    
    async def start(self):
        """Start the metrics collector."""
        logger.info("Starting metrics collector")
        
        # Load historical data
        await self._load_historical_data()
        
        # Start collection task
        self._collection_task = asyncio.create_task(self._collection_loop())
        
        logger.info("Metrics collector started")
    
    async def stop(self):
        """Stop the metrics collector."""
        logger.info("Stopping metrics collector")
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        # Save final metrics
        await self._save_metrics()
        
        logger.info("Metrics collector stopped")
    
    async def record_task_start(self, task_id: str, worker_id: str, 
                              content_type: str, file_size: int):
        """Record task start."""
        task_metrics = TaskMetrics(
            task_id=task_id,
            worker_id=worker_id,
            content_type=content_type,
            submitted_at=datetime.utcnow(),
            started_at=datetime.utcnow(),
            status="processing",
            input_file_size=file_size
        )
        
        self.task_metrics[task_id] = task_metrics
        await self._save_task_metrics(task_metrics)
        
        logger.debug("Task start recorded", task_id=task_id, worker_id=worker_id)
    
    async def record_task_completion(self, task_id: str, success: bool,
                                   error_message: Optional[str] = None,
                                   output_size: int = 0):
        """Record task completion."""
        if task_id not in self.task_metrics:
            logger.warning("Task completion recorded for unknown task", task_id=task_id)
            return
        
        task_metrics = self.task_metrics[task_id]
        task_metrics.completed_at = datetime.utcnow()
        task_metrics.status = "completed" if success else "failed"
        task_metrics.error_message = error_message
        task_metrics.output_file_size = output_size
        
        # Calculate timing
        if task_metrics.started_at:
            task_metrics.processing_time = (
                task_metrics.completed_at - task_metrics.started_at
            ).total_seconds()
        
        task_metrics.total_time = (
            task_metrics.completed_at - task_metrics.submitted_at
        ).total_seconds()
        
        await self._save_task_metrics(task_metrics)
        
        logger.debug("Task completion recorded", 
                    task_id=task_id, 
                    success=success,
                    processing_time=task_metrics.processing_time)
    
    async def get_dashboard_data(self) -> DashboardData:
        """Get complete dashboard data."""
        try:
            # Collect current metrics
            system_metrics = await self._collect_system_metrics()
            worker_metrics = await self._collect_worker_metrics()
            recent_tasks = await self._get_recent_tasks()
            active_alerts = await self._get_active_alerts()
            
            # Get historical data
            hourly_stats = await self._get_hourly_stats()
            daily_stats = await self._get_daily_stats()
            
            dashboard_data = DashboardData(
                system_metrics=system_metrics,
                worker_metrics=worker_metrics,
                recent_tasks=recent_tasks,
                active_alerts=active_alerts,
                hourly_stats=hourly_stats,
                daily_stats=daily_stats
            )
            
            return dashboard_data
            
        except Exception as e:
            logger.error("Failed to get dashboard data", error=str(e))
            raise
    
    async def create_alert(self, title: str, description: str, 
                          severity: AlertSeverity, component: str,
                          component_id: Optional[str] = None,
                          metric_name: Optional[str] = None,
                          metric_value: Optional[float] = None,
                          threshold: Optional[float] = None) -> Alert:
        """Create a new alert."""
        alert = Alert(
            title=title,
            description=description,
            severity=severity,
            component=component,
            component_id=component_id,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold
        )
        
        self.alerts[alert.alert_id] = alert
        await self._save_alert(alert)
        
        logger.warning("Alert created",
                      alert_id=alert.alert_id,
                      title=title,
                      severity=severity.value)
        
        return alert
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an alert."""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.utcnow()
        
        await self._save_alert(alert)
        
        logger.info("Alert acknowledged",
                   alert_id=alert_id,
                   acknowledged_by=acknowledged_by)
        
        return True
    
    async def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert."""
        if alert_id not in self.alerts:
            return False
        
        alert = self.alerts[alert_id]
        alert.is_active = False
        alert.resolved_at = datetime.utcnow()
        
        await self._save_alert(alert)
        
        logger.info("Alert resolved", alert_id=alert_id)
        
        return True
    
    async def _collection_loop(self):
        """Main metrics collection loop."""
        while True:
            try:
                await asyncio.sleep(self.collection_interval)
                
                # Collect metrics
                await self._collect_and_store_metrics()
                
                # Check for alerts
                await self._check_alert_conditions()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Metrics collection error", error=str(e))
    
    async def _collect_and_store_metrics(self):
        """Collect and store current metrics."""
        try:
            # Collect system metrics
            system_metrics = await self._collect_system_metrics()
            await self._save_system_metrics(system_metrics)
            
            # Collect worker metrics
            worker_metrics = await self._collect_worker_metrics()
            for metrics in worker_metrics:
                await self._save_worker_metrics(metrics)
            
            logger.debug("Metrics collected and stored")
            
        except Exception as e:
            logger.error("Failed to collect metrics", error=str(e))
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system-wide metrics."""
        # Get worker statistics
        worker_stats = self.worker_registry.get_worker_stats()
        
        # Get queue statistics
        queue_stats = await self.queue_manager.get_queue_stats()
        
        # Calculate task metrics
        total_tasks = sum(len(self.task_metrics) for _ in [None])
        completed_tasks = len([t for t in self.task_metrics.values() if t.status == "completed"])
        failed_tasks = len([t for t in self.task_metrics.values() if t.status == "failed"])
        processing_tasks = len([t for t in self.task_metrics.values() if t.status == "processing"])
        pending_tasks = sum(stats.pending_tasks for stats in queue_stats.values())
        
        # Calculate performance metrics
        completed_task_times = [
            t.processing_time for t in self.task_metrics.values() 
            if t.status == "completed" and t.processing_time > 0
        ]
        avg_task_duration = sum(completed_task_times) / len(completed_task_times) if completed_task_times else 0.0
        
        # Calculate error rate
        error_rate = failed_tasks / max(total_tasks, 1)
        
        system_metrics = SystemMetrics(
            total_workers=worker_stats['total'],
            online_workers=worker_stats['online'],
            gpu_workers=worker_stats['gpu_workers'],
            cpu_workers=worker_stats['cpu_workers'],
            hybrid_workers=worker_stats['hybrid_workers'],
            total_tasks=total_tasks,
            pending_tasks=pending_tasks,
            processing_tasks=processing_tasks,
            completed_tasks=completed_tasks,
            failed_tasks=failed_tasks,
            gpu_queue_size=sum(stats.pending_tasks for name, stats in queue_stats.items() 
                              if 'gpu' in name.value),
            cpu_queue_size=sum(stats.pending_tasks for name, stats in queue_stats.items() 
                              if 'cpu' in name.value),
            average_task_duration=avg_task_duration,
            error_rate=error_rate
        )
        
        return system_metrics
    
    async def _collect_worker_metrics(self) -> List[WorkerMetrics]:
        """Collect metrics for all workers."""
        worker_metrics = []
        
        for worker_info in self.worker_registry.get_all_workers():
            metrics = WorkerMetrics(
                worker_id=worker_info.registration.worker_id,
                is_online=worker_info.status.value == "online",
                is_healthy=worker_info.is_healthy,
                last_heartbeat=worker_info.last_heartbeat,
                active_tasks=worker_info.metrics.active_tasks,
                completed_tasks=worker_info.metrics.completed_tasks,
                failed_tasks=worker_info.metrics.failed_tasks,
                cpu_usage=worker_info.metrics.cpu_usage_percent,
                memory_usage=worker_info.metrics.memory_usage_percent,
                gpu_usage=worker_info.metrics.gpu_usage_percent,
                error_count=worker_info.metrics.failed_tasks
            )
            
            worker_metrics.append(metrics)
        
        return worker_metrics
    
    async def _check_alert_conditions(self):
        """Check for alert conditions."""
        try:
            # Check worker health
            await self._check_worker_alerts()
            
            # Check queue sizes
            await self._check_queue_alerts()
            
            # Check system performance
            await self._check_performance_alerts()
            
        except Exception as e:
            logger.error("Alert checking failed", error=str(e))
    
    async def _check_worker_alerts(self):
        """Check for worker-related alerts."""
        for worker_info in self.worker_registry.get_all_workers():
            worker_id = worker_info.registration.worker_id
            
            # Check if worker is offline
            if not worker_info.is_healthy:
                time_offline = (datetime.utcnow() - worker_info.last_heartbeat).total_seconds() / 60
                if time_offline > self.alert_thresholds['worker_offline_threshold']:
                    await self.create_alert(
                        title=f"Worker {worker_id} Offline",
                        description=f"Worker has been offline for {time_offline:.1f} minutes",
                        severity=AlertSeverity.ERROR,
                        component="worker",
                        component_id=worker_id,
                        metric_name="offline_time",
                        metric_value=time_offline,
                        threshold=self.alert_thresholds['worker_offline_threshold']
                    )
            
            # Check resource usage
            if worker_info.metrics.cpu_usage_percent > self.alert_thresholds['cpu_usage_threshold']:
                await self.create_alert(
                    title=f"High CPU Usage on {worker_id}",
                    description=f"CPU usage is {worker_info.metrics.cpu_usage_percent:.1%}",
                    severity=AlertSeverity.WARNING,
                    component="worker",
                    component_id=worker_id,
                    metric_name="cpu_usage",
                    metric_value=worker_info.metrics.cpu_usage_percent,
                    threshold=self.alert_thresholds['cpu_usage_threshold']
                )
    
    async def _get_recent_tasks(self, limit: int = 50) -> List[TaskMetrics]:
        """Get recent task metrics."""
        recent_tasks = sorted(
            self.task_metrics.values(),
            key=lambda t: t.timestamp,
            reverse=True
        )[:limit]
        
        return recent_tasks
    
    async def _get_active_alerts(self) -> List[Alert]:
        """Get active alerts."""
        return [alert for alert in self.alerts.values() if alert.is_active]
    
    async def _save_system_metrics(self, metrics: SystemMetrics):
        """Save system metrics to Redis."""
        key = f"system_metrics:{int(metrics.timestamp.timestamp())}"
        self.redis.setex(key, 86400, metrics.model_dump_json())  # 24 hours TTL
    
    async def _save_worker_metrics(self, metrics: WorkerMetrics):
        """Save worker metrics to Redis."""
        key = f"worker_metrics:{metrics.worker_id}:{int(metrics.timestamp.timestamp())}"
        self.redis.setex(key, 86400, metrics.model_dump_json())  # 24 hours TTL
    
    async def _save_task_metrics(self, metrics: TaskMetrics):
        """Save task metrics to Redis."""
        key = f"task_metrics:{metrics.task_id}"
        self.redis.setex(key, 86400 * 7, metrics.model_dump_json())  # 7 days TTL
    
    async def _save_alert(self, alert: Alert):
        """Save alert to Redis."""
        key = f"alert:{alert.alert_id}"
        self.redis.setex(key, 86400 * 30, alert.model_dump_json())  # 30 days TTL
    
    async def _get_hourly_stats(self) -> List[SystemMetrics]:
        """Get hourly statistics."""
        # Implementation would aggregate metrics by hour
        return []
    
    async def _get_daily_stats(self) -> List[SystemMetrics]:
        """Get daily statistics."""
        # Implementation would aggregate metrics by day
        return []
    
    async def _load_historical_data(self):
        """Load historical metrics data."""
        # Implementation would load from Redis
        pass
    
    async def _save_metrics(self):
        """Save current metrics state."""
        # Implementation would save to Redis
        pass
    
    async def _cleanup_old_data(self):
        """Clean up old metrics data."""
        # Implementation would remove old metrics
        pass
    
    async def _check_queue_alerts(self):
        """Check for queue-related alerts."""
        # Implementation would check queue sizes
        pass
    
    async def _check_performance_alerts(self):
        """Check for performance-related alerts."""
        # Implementation would check system performance
        pass
```

## Testing Requirements

### Unit Tests
1. **Metrics Collector Tests**
   - Test metrics collection and aggregation
   - Test alert creation and management
   - Test dashboard data generation
   - Test historical data handling

2. **Metrics Models Tests**
   - Test metrics model validation
   - Test alert model functionality
   - Test dashboard data structure

### Integration Tests
1. **End-to-End Monitoring Tests**
   - Test complete monitoring workflow
   - Test real-time metrics updates
   - Test alert triggering and resolution

### Test Files to Create
- `tests/test_metrics_collector.py`
- `tests/test_metrics_models.py`
- `tests/integration/test_monitoring_e2e.py`

## Dependencies
- **Existing**: Worker registry from Task 2
- **Existing**: Queue manager from Task 6
- **Existing**: Redis for metrics storage

## Success Criteria
1. Real-time metrics collection and aggregation works correctly
2. Dashboard provides comprehensive system visibility
3. Alert system detects and notifies of critical issues
4. Historical data tracking enables trend analysis
5. Performance metrics help optimize system efficiency
6. Monitoring data is accurate and up-to-date

## Next Steps
After completing this task:
1. Proceed to Task 12: Deployment & Documentation
2. Test monitoring dashboard with various scenarios
3. Validate alert system with simulated failures

---

**Dependencies**: Task 2 (Worker Registry), Task 6 (Priority Queue)
**Estimated Time**: 4-5 days
**Risk Level**: Medium (complex data aggregation and visualization)
