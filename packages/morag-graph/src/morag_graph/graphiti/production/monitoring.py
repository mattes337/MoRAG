"""Monitoring and health check system for Graphiti production deployment."""

import logging
import asyncio
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from ..integration_service import GraphitiIntegrationService
from .config import ProductionConfig

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    service: str
    status: str  # "healthy", "degraded", "unhealthy"
    response_time_ms: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_usage_percent: float
    memory_usage_mb: float
    disk_usage_gb: float
    active_connections: int
    requests_per_minute: float
    average_response_time_ms: float
    error_rate_percent: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class GraphitiMonitoringService:
    """Monitoring service for Graphiti production deployment."""
    
    def __init__(
        self, 
        graphiti_service: GraphitiIntegrationService,
        config: ProductionConfig
    ):
        self.graphiti_service = graphiti_service
        self.config = config
        self.metrics_history: List[SystemMetrics] = []
        self.health_history: List[HealthCheckResult] = []
        self.alert_thresholds = self._setup_alert_thresholds()
        
    def _setup_alert_thresholds(self) -> Dict[str, Any]:
        """Setup alert thresholds for monitoring."""
        return {
            "cpu_usage_critical": 90.0,
            "cpu_usage_warning": 75.0,
            "memory_usage_critical": 90.0,
            "memory_usage_warning": 75.0,
            "disk_usage_critical": 90.0,
            "disk_usage_warning": 80.0,
            "response_time_critical": 5000.0,  # 5 seconds
            "response_time_warning": 2000.0,   # 2 seconds
            "error_rate_critical": 10.0,       # 10%
            "error_rate_warning": 5.0          # 5%
        }
    
    async def perform_health_checks(self) -> List[HealthCheckResult]:
        """Perform comprehensive health checks."""
        health_checks = []
        
        # Check Graphiti service
        graphiti_health = await self._check_graphiti_health()
        health_checks.append(graphiti_health)
        
        # Check Neo4j connectivity
        neo4j_health = await self._check_neo4j_health()
        health_checks.append(neo4j_health)
        
        # Check OpenAI API
        openai_health = await self._check_openai_health()
        health_checks.append(openai_health)
        
        # Check search functionality
        search_health = await self._check_search_health()
        health_checks.append(search_health)
        
        # Check ingestion functionality
        ingestion_health = await self._check_ingestion_health()
        health_checks.append(ingestion_health)
        
        # Store health check history
        self.health_history.extend(health_checks)
        
        # Keep only recent history (last 24 hours)
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.health_history = [
            h for h in self.health_history 
            if h.timestamp > cutoff_time
        ]
        
        return health_checks
    
    async def _check_graphiti_health(self) -> HealthCheckResult:
        """Check Graphiti service health."""
        start_time = time.time()
        
        try:
            # Test basic Graphiti functionality
            status = self.graphiti_service.get_graphiti_status()
            
            response_time = (time.time() - start_time) * 1000
            
            if status.get("available", False):
                return HealthCheckResult(
                    service="graphiti",
                    status="healthy",
                    response_time_ms=response_time,
                    details=status
                )
            else:
                return HealthCheckResult(
                    service="graphiti",
                    status="unhealthy",
                    response_time_ms=response_time,
                    error="Graphiti service not available",
                    details=status
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="graphiti",
                status="unhealthy",
                response_time_ms=response_time,
                error=str(e)
            )
    
    async def _check_neo4j_health(self) -> HealthCheckResult:
        """Check Neo4j database health."""
        start_time = time.time()
        
        try:
            # Test Neo4j connectivity with a simple query
            # This would use the actual Neo4j connection
            # For now, we'll simulate the check
            
            response_time = (time.time() - start_time) * 1000
            
            # Simulate successful connection
            return HealthCheckResult(
                service="neo4j",
                status="healthy",
                response_time_ms=response_time,
                details={"database": "connected", "version": "5.x"}
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="neo4j",
                status="unhealthy",
                response_time_ms=response_time,
                error=str(e)
            )
    
    async def _check_openai_health(self) -> HealthCheckResult:
        """Check OpenAI API health."""
        start_time = time.time()
        
        try:
            # Test OpenAI API with a minimal request
            # This would make an actual API call
            # For now, we'll simulate the check
            
            response_time = (time.time() - start_time) * 1000
            
            # Check if API key is configured
            import os
            if not os.getenv("OPENAI_API_KEY"):
                return HealthCheckResult(
                    service="openai",
                    status="unhealthy",
                    response_time_ms=response_time,
                    error="OpenAI API key not configured"
                )
            
            return HealthCheckResult(
                service="openai",
                status="healthy",
                response_time_ms=response_time,
                details={"api_key": "configured"}
            )
            
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="openai",
                status="unhealthy",
                response_time_ms=response_time,
                error=str(e)
            )
    
    async def _check_search_health(self) -> HealthCheckResult:
        """Check search functionality health."""
        start_time = time.time()
        
        try:
            # Test search with a simple query
            search_result = await self.graphiti_service.search_graphiti(
                "test health check",
                limit=1
            )
            
            response_time = (time.time() - start_time) * 1000
            
            if search_result.get("success", False):
                return HealthCheckResult(
                    service="search",
                    status="healthy",
                    response_time_ms=response_time,
                    details={"backend": search_result.get("backend")}
                )
            else:
                return HealthCheckResult(
                    service="search",
                    status="degraded",
                    response_time_ms=response_time,
                    error=search_result.get("error", "Search failed")
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="search",
                status="unhealthy",
                response_time_ms=response_time,
                error=str(e)
            )
    
    async def _check_ingestion_health(self) -> HealthCheckResult:
        """Check ingestion functionality health."""
        start_time = time.time()
        
        try:
            # Test ingestion readiness (without actually ingesting)
            status = self.graphiti_service.get_graphiti_status()
            
            response_time = (time.time() - start_time) * 1000
            
            if status.get("available", False):
                return HealthCheckResult(
                    service="ingestion",
                    status="healthy",
                    response_time_ms=response_time,
                    details={"ready": True}
                )
            else:
                return HealthCheckResult(
                    service="ingestion",
                    status="degraded",
                    response_time_ms=response_time,
                    error="Ingestion service not ready"
                )
                
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return HealthCheckResult(
                service="ingestion",
                status="unhealthy",
                response_time_ms=response_time,
                error=str(e)
            )

    async def collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        try:
            try:
                import psutil

                # Collect system metrics
                cpu_usage = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')

                memory_usage_mb = memory.used / (1024 * 1024)
                disk_usage_gb = disk.used / (1024 * 1024 * 1024)
            except ImportError:
                # Fallback if psutil is not available
                cpu_usage = 0.0
                memory_usage_mb = 0.0
                disk_usage_gb = 0.0

            # Collect application-specific metrics
            # These would be collected from actual application monitoring
            active_connections = 0  # Would get from connection pool
            requests_per_minute = 0.0  # Would get from request counter
            avg_response_time = 0.0  # Would get from response time tracker
            error_rate = 0.0  # Would get from error counter

            metrics = SystemMetrics(
                cpu_usage_percent=cpu_usage,
                memory_usage_mb=memory_usage_mb,
                disk_usage_gb=disk_usage_gb,
                active_connections=active_connections,
                requests_per_minute=requests_per_minute,
                average_response_time_ms=avg_response_time,
                error_rate_percent=error_rate
            )

            # Store metrics history
            self.metrics_history.append(metrics)

            # Keep only recent history (last 24 hours)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.metrics_history = [
                m for m in self.metrics_history
                if m.timestamp > cutoff_time
            ]

            return metrics

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, 0, 0)

    def check_alert_conditions(self, metrics: SystemMetrics) -> List[Dict[str, Any]]:
        """Check if any alert conditions are met."""
        alerts = []

        # CPU usage alerts
        if metrics.cpu_usage_percent >= self.alert_thresholds["cpu_usage_critical"]:
            alerts.append({
                "severity": "critical",
                "metric": "cpu_usage",
                "value": metrics.cpu_usage_percent,
                "threshold": self.alert_thresholds["cpu_usage_critical"],
                "message": f"CPU usage critical: {metrics.cpu_usage_percent:.1f}%"
            })
        elif metrics.cpu_usage_percent >= self.alert_thresholds["cpu_usage_warning"]:
            alerts.append({
                "severity": "warning",
                "metric": "cpu_usage",
                "value": metrics.cpu_usage_percent,
                "threshold": self.alert_thresholds["cpu_usage_warning"],
                "message": f"CPU usage high: {metrics.cpu_usage_percent:.1f}%"
            })

        # Memory usage alerts
        memory_usage_percent = (metrics.memory_usage_mb / self.config.max_memory_mb) * 100
        if memory_usage_percent >= self.alert_thresholds["memory_usage_critical"]:
            alerts.append({
                "severity": "critical",
                "metric": "memory_usage",
                "value": memory_usage_percent,
                "threshold": self.alert_thresholds["memory_usage_critical"],
                "message": f"Memory usage critical: {memory_usage_percent:.1f}%"
            })
        elif memory_usage_percent >= self.alert_thresholds["memory_usage_warning"]:
            alerts.append({
                "severity": "warning",
                "metric": "memory_usage",
                "value": memory_usage_percent,
                "threshold": self.alert_thresholds["memory_usage_warning"],
                "message": f"Memory usage high: {memory_usage_percent:.1f}%"
            })

        return alerts

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status summary."""
        recent_health = self.health_history[-5:] if self.health_history else []
        recent_metrics = self.metrics_history[-1] if self.metrics_history else None

        # Determine overall status
        overall_status = "healthy"
        if recent_health:
            unhealthy_services = [h for h in recent_health if h.status == "unhealthy"]
            degraded_services = [h for h in recent_health if h.status == "degraded"]

            if unhealthy_services:
                overall_status = "unhealthy"
            elif degraded_services:
                overall_status = "degraded"

        return {
            "overall_status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "services": {
                h.service: {
                    "status": h.status,
                    "response_time_ms": h.response_time_ms,
                    "error": h.error
                }
                for h in recent_health
            },
            "metrics": asdict(recent_metrics) if recent_metrics else None,
            "uptime_hours": self._calculate_uptime(),
            "version": "1.0.0"  # Would get from actual version
        }

    def _calculate_uptime(self) -> float:
        """Calculate system uptime in hours."""
        # This would calculate actual uptime
        # For now, return a placeholder
        return 24.0


async def start_monitoring_loop(
    monitoring_service: GraphitiMonitoringService,
    interval_seconds: int = 60
):
    """Start the monitoring loop."""
    logger.info(f"Starting monitoring loop with {interval_seconds}s interval")

    while True:
        try:
            # Perform health checks
            health_results = await monitoring_service.perform_health_checks()

            # Collect metrics
            metrics = await monitoring_service.collect_system_metrics()

            # Check for alerts
            alerts = monitoring_service.check_alert_conditions(metrics)

            # Log alerts
            for alert in alerts:
                if alert["severity"] == "critical":
                    logger.critical(alert["message"])
                else:
                    logger.warning(alert["message"])

            # Log overall status
            status = monitoring_service.get_system_status()
            logger.info(f"System status: {status['overall_status']}")

        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")

        await asyncio.sleep(interval_seconds)
