import time
import psutil
import asyncio
from typing import Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import structlog

from morag.core.config import settings

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
        # Import here to avoid circular imports
        try:
            from morag.services.task_manager import task_manager
            from morag.services.storage import qdrant_service
            
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
                
        except ImportError:
            # Services not available yet
            active_tasks = 0
            queue_lengths = {}
            storage_size_mb = 0.0
        
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
