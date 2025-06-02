from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict, Any
import asyncio
import structlog
import redis

from morag.core.config import settings
from morag.services.storage import qdrant_service
from morag.services.task_manager import task_manager
from morag.services.embedding import gemini_service

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
