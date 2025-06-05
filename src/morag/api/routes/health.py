from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import structlog
import redis

from morag_core.config import settings
from morag_services.storage import qdrant_service
from src.morag.services.task_manager import task_manager
from morag_services.embedding import gemini_service
from src.morag.services.metrics_service import metrics_collector

logger = structlog.get_logger()
router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    version: str
    services: Dict[str, str]

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Basic health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        services={}
    )

@router.get("/ready")
async def readiness_check():
    """Readiness check with service dependencies."""
    services = {}
    
    # Check Redis connection
    try:
        r = redis.from_url(settings.redis_url)
        r.ping()
        services["redis"] = "healthy"

        # Also check Celery
        stats = task_manager.get_queue_stats()
        if 'error' not in stats:
            services["celery"] = "healthy"
        else:
            services["celery"] = "unhealthy"

    except Exception as e:
        logger.error("Redis/Celery health check failed", error=str(e))
        services["redis"] = "unhealthy"
        services["celery"] = "unhealthy"
    
    # Check Qdrant connection
    try:
        if qdrant_service.client:
            await qdrant_service.get_collection_info()
            services["qdrant"] = "healthy"
        else:
            services["qdrant"] = "not_connected"
    except Exception as e:
        logger.error("Qdrant health check failed", error=str(e))
        services["qdrant"] = "unhealthy"
    
    # Check Gemini API
    try:
        gemini_health = await gemini_service.health_check()
        services["gemini"] = gemini_health["status"]
    except Exception as e:
        logger.error("Gemini health check failed", error=str(e))
        services["gemini"] = "unhealthy"
    
    all_healthy = all(status == "healthy" for status in services.values())
    
    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        version="0.1.0",
        services=services
    )

@router.get("/metrics")
async def get_metrics():
    """Get current system and application metrics."""
    if not settings.metrics_enabled:
        return {"error": "Metrics collection is disabled"}

    current_metrics = metrics_collector.get_current_metrics()
    if not current_metrics:
        # Collect metrics if none available
        current_metrics = await metrics_collector.collect_metrics()

    return current_metrics

@router.get("/metrics/history")
async def get_metrics_history(hours: int = 1):
    """Get metrics history for the specified number of hours."""
    if not settings.metrics_enabled:
        return {"error": "Metrics collection is disabled"}

    return {
        "metrics": metrics_collector.get_recent_metrics(hours),
        "hours": hours,
        "count": len(metrics_collector.get_recent_metrics(hours))
    }
