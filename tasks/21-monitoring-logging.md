# Task 21: Monitoring and Logging (Observability)

## Overview
Implement comprehensive monitoring, logging, and observability for the MoRAG pipeline to ensure production readiness and operational visibility.

## Prerequisites
- All core tasks completed (01-20)
- Basic understanding of observability principles
- Familiarity with Prometheus metrics and structured logging

## Dependencies
- Task 02: API Framework (logging foundation)
- Task 04: Task Queue Setup (worker monitoring)
- Task 18: Status Tracking (existing health checks)

## Implementation Steps

### 1. Enhanced Logging Configuration
Update `src/morag/core/config.py`:
```python
# Add to Settings class
class Settings(BaseSettings):
    # ... existing settings ...
    
    # Enhanced Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"  # json or console
    log_file: str = "./logs/morag.log"
    log_max_size: str = "100MB"
    log_backup_count: int = 5
    log_rotation: str = "daily"  # daily, weekly, size
    
    # Monitoring Configuration
    metrics_enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    
    # Performance Monitoring
    enable_profiling: bool = False
    slow_query_threshold: float = 1.0  # seconds
    memory_threshold: int = 80  # percentage
    cpu_threshold: int = 80  # percentage
    
    # Alerting Configuration
    webhook_alerts_enabled: bool = False
    alert_webhook_url: str = ""
    alert_email_enabled: bool = False
    alert_email_smtp_host: str = ""
    alert_email_smtp_port: int = 587
    alert_email_from: str = ""
    alert_email_to: List[str] = []
```

### 2. Structured Logging Service
Create `src/morag/services/logging_service.py`:
```python
import logging
import logging.handlers
import structlog
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from morag.core.config import settings

class LoggingService:
    """Enhanced logging service with rotation and structured output."""
    
    def __init__(self):
        self.setup_logging()
        self.logger = structlog.get_logger()
    
    def setup_logging(self):
        """Configure structured logging with file rotation."""
        # Create logs directory
        log_dir = Path(settings.log_file).parent
        log_dir.mkdir(exist_ok=True)
        
        # Configure file handler with rotation
        if settings.log_rotation == "size":
            file_handler = logging.handlers.RotatingFileHandler(
                settings.log_file,
                maxBytes=self._parse_size(settings.log_max_size),
                backupCount=settings.log_backup_count
            )
        else:
            file_handler = logging.handlers.TimedRotatingFileHandler(
                settings.log_file,
                when=settings.log_rotation,
                backupCount=settings.log_backup_count
            )
        
        # Configure console handler
        console_handler = logging.StreamHandler()
        
        # Set up structlog
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        if settings.log_format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, settings.log_level.upper()))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '100MB' to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def log_request(self, method: str, url: str, status_code: int, 
                   duration: float, client_ip: str = None, user_id: str = None):
        """Log HTTP request with structured data."""
        self.logger.info(
            "HTTP request",
            method=method,
            url=url,
            status_code=status_code,
            duration_ms=round(duration * 1000, 2),
            client_ip=client_ip,
            user_id=user_id,
            event_type="http_request"
        )
    
    def log_task_start(self, task_id: str, task_type: str, **kwargs):
        """Log task start with context."""
        self.logger.info(
            "Task started",
            task_id=task_id,
            task_type=task_type,
            event_type="task_start",
            **kwargs
        )
    
    def log_task_complete(self, task_id: str, task_type: str, 
                         duration: float, success: bool, **kwargs):
        """Log task completion with metrics."""
        self.logger.info(
            "Task completed",
            task_id=task_id,
            task_type=task_type,
            duration_seconds=round(duration, 2),
            success=success,
            event_type="task_complete",
            **kwargs
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with full context."""
        self.logger.error(
            "Error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            event_type="error",
            **(context or {})
        )
    
    def log_performance_metric(self, metric_name: str, value: float, 
                              unit: str = "", tags: Dict[str, str] = None):
        """Log performance metrics."""
        self.logger.info(
            "Performance metric",
            metric_name=metric_name,
            value=value,
            unit=unit,
            tags=tags or {},
            event_type="performance_metric"
        )

# Global logging service instance
logging_service = LoggingService()
```

### 3. Metrics Collection Service
Create `src/morag/services/metrics_service.py`:
```python
import time
import psutil
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import structlog

from morag.core.config import settings
from morag.services.storage import qdrant_service
from morag.services.task_manager import task_manager

logger = structlog.get_logger()

@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_total_mb: float
    disk_percent: float
    disk_used_gb: float
    disk_total_gb: float
    network_bytes_sent: int
    network_bytes_recv: int

@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: datetime
    active_tasks: int
    completed_tasks_1h: int
    failed_tasks_1h: int
    queue_lengths: Dict[str, int]
    avg_task_duration: float
    documents_processed: int
    storage_size_mb: float
    api_requests_1h: int
    api_errors_1h: int

class MetricsCollector:
    """Collect and store system and application metrics."""
    
    def __init__(self):
        self.metrics_history: List[Dict[str, Any]] = []
        self.max_history_size = 1000
        self.collection_interval = 60  # seconds
        self._running = False
        
    async def start_collection(self):
        """Start periodic metrics collection."""
        self._running = True
        while self._running:
            try:
                await self.collect_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.error("Metrics collection failed", error=str(e))
                await asyncio.sleep(self.collection_interval)
    
    def stop_collection(self):
        """Stop metrics collection."""
        self._running = False
    
    async def collect_metrics(self):
        """Collect all metrics."""
        timestamp = datetime.utcnow()
        
        # Collect system metrics
        system_metrics = self._collect_system_metrics(timestamp)
        
        # Collect application metrics
        app_metrics = await self._collect_application_metrics(timestamp)
        
        # Store metrics
        metrics_data = {
            'timestamp': timestamp.isoformat(),
            'system': asdict(system_metrics),
            'application': asdict(app_metrics)
        }
        
        self.metrics_history.append(metrics_data)
        
        # Trim history if too large
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
        
        # Log metrics
        logger.info("Metrics collected", **metrics_data)
        
        return metrics_data
    
    def _collect_system_metrics(self, timestamp: datetime) -> SystemMetrics:
        """Collect system resource metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_total_mb = memory.total / (1024 * 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_used_gb = disk.used / (1024 * 1024 * 1024)
        disk_total_gb = disk.total / (1024 * 1024 * 1024)
        
        # Network usage
        network = psutil.net_io_counters()
        
        return SystemMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            disk_percent=disk_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv
        )
    
    async def _collect_application_metrics(self, timestamp: datetime) -> ApplicationMetrics:
        """Collect application-specific metrics."""
        # Task queue metrics
        queue_stats = task_manager.get_queue_stats()
        active_tasks = queue_stats.get('active_tasks', 0)
        queue_lengths = queue_stats.get('queues', {})
        
        # Storage metrics
        storage_size_mb = 0.0
        try:
            if qdrant_service.client:
                collection_info = await qdrant_service.get_collection_info()
                storage_size_mb = collection_info.get('vectors_count', 0) * 0.001  # Estimate
        except Exception:
            pass
        
        return ApplicationMetrics(
            timestamp=timestamp,
            active_tasks=active_tasks,
            completed_tasks_1h=0,  # TODO: Implement task history tracking
            failed_tasks_1h=0,     # TODO: Implement task history tracking
            queue_lengths=queue_lengths,
            avg_task_duration=0.0,  # TODO: Implement duration tracking
            documents_processed=0,  # TODO: Implement document counter
            storage_size_mb=storage_size_mb,
            api_requests_1h=0,     # TODO: Implement request counter
            api_errors_1h=0        # TODO: Implement error counter
        )
    
    def get_recent_metrics(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get metrics from the last N hours."""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) > cutoff
        ]
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get the most recent metrics."""
        return self.metrics_history[-1] if self.metrics_history else {}

# Global metrics collector
metrics_collector = MetricsCollector()
```

### 4. Performance Monitoring Middleware
Create `src/morag/middleware/monitoring.py`:
```python
import time
import asyncio
from typing import Callable
from fastapi import Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
import structlog

from morag.core.config import settings
from morag.services.logging_service import logging_service

logger = structlog.get_logger()

class PerformanceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring API performance and logging requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        
        # Extract request info
        method = request.method
        url = str(request.url)
        client_ip = request.client.host if request.client else "unknown"
        
        # Process request
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Log request
            logging_service.log_request(
                method=method,
                url=url,
                status_code=response.status_code,
                duration=duration,
                client_ip=client_ip
            )
            
            # Check for slow requests
            if duration > settings.slow_query_threshold:
                logger.warning(
                    "Slow request detected",
                    method=method,
                    url=url,
                    duration=duration,
                    threshold=settings.slow_query_threshold
                )
            
            # Add performance headers
            response.headers["X-Process-Time"] = str(round(duration, 4))
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Log error
            logging_service.log_error(e, {
                'method': method,
                'url': url,
                'duration': duration,
                'client_ip': client_ip
            })
            
            raise

class ResourceMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring system resources during requests."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check system resources before processing
        import psutil
        
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Log resource warnings
        if cpu_percent > settings.cpu_threshold:
            logger.warning(
                "High CPU usage detected",
                cpu_percent=cpu_percent,
                threshold=settings.cpu_threshold,
                url=str(request.url)
            )
        
        if memory_percent > settings.memory_threshold:
            logger.warning(
                "High memory usage detected",
                memory_percent=memory_percent,
                threshold=settings.memory_threshold,
                url=str(request.url)
            )
        
        response = await call_next(request)
        return response
```

## Testing Instructions

### 1. Unit Tests
Create `tests/test_monitoring.py`:
```python
import pytest
import asyncio
from unittest.mock import Mock, patch
from datetime import datetime

from morag.services.logging_service import LoggingService
from morag.services.metrics_service import MetricsCollector, SystemMetrics

class TestLoggingService:
    def test_logging_service_initialization(self):
        """Test logging service initializes correctly."""
        service = LoggingService()
        assert service.logger is not None
    
    def test_log_request(self):
        """Test HTTP request logging."""
        service = LoggingService()
        # Should not raise exception
        service.log_request("GET", "/test", 200, 0.5, "127.0.0.1")
    
    def test_log_task_lifecycle(self):
        """Test task logging."""
        service = LoggingService()
        service.log_task_start("task-123", "document_processing")
        service.log_task_complete("task-123", "document_processing", 2.5, True)

class TestMetricsCollector:
    @pytest.fixture
    def collector(self):
        return MetricsCollector()
    
    def test_system_metrics_collection(self, collector):
        """Test system metrics collection."""
        timestamp = datetime.utcnow()
        metrics = collector._collect_system_metrics(timestamp)
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.timestamp == timestamp
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
    
    @pytest.mark.asyncio
    async def test_application_metrics_collection(self, collector):
        """Test application metrics collection."""
        timestamp = datetime.utcnow()
        metrics = await collector._collect_application_metrics(timestamp)
        
        assert metrics.timestamp == timestamp
        assert metrics.active_tasks >= 0
    
    def test_metrics_history_management(self, collector):
        """Test metrics history size management."""
        collector.max_history_size = 2
        
        # Add metrics beyond limit
        for i in range(5):
            collector.metrics_history.append({'test': i})
        
        # Should trim to max size
        assert len(collector.metrics_history) <= collector.max_history_size
```

## Success Criteria
- [ ] Enhanced logging configuration implemented
- [ ] Structured logging service working
- [ ] Metrics collection service operational
- [ ] Performance monitoring middleware active
- [ ] Log rotation and file management working
- [ ] System resource monitoring functional
- [ ] Application metrics being collected
- [ ] All tests passing (>95% coverage)
- [ ] Logs are properly structured and searchable
- [ ] Metrics provide actionable insights

## Next Steps
- Integrate with external monitoring systems (Prometheus, Grafana)
- Add custom business metrics
- Implement alerting rules
- Set up log aggregation (ELK stack)
- Add distributed tracing
- Configure monitoring dashboards
