"""Health check endpoints for AI services and system components."""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import structlog

from src.morag.core.ai_error_handlers import get_ai_service_health
from morag_services.embedding import gemini_service
from morag_audio.services import whisper_service

logger = structlog.get_logger()

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "service": "morag",
        "version": "1.0.0"
    }


@router.get("/ai-services")
async def ai_services_health() -> Dict[str, Any]:
    """Get health status for all AI services."""
    try:
        health_data = {
            "overall_status": "healthy",
            "services": {}
        }
        
        # Get AI resilience health for all services
        ai_health = get_ai_service_health()
        health_data["ai_resilience"] = ai_health
        
        # Get Gemini service health
        try:
            gemini_health = await gemini_service.health_check()
            health_data["services"]["gemini"] = gemini_health
        except Exception as e:
            logger.error("Failed to get Gemini health", error=str(e))
            health_data["services"]["gemini"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Get Whisper service health (basic check)
        try:
            whisper_models = whisper_service.get_available_models()
            health_data["services"]["whisper"] = {
                "status": "healthy",
                "available_models": whisper_models,
                "loaded_models": len(whisper_service._models),
                "resilience_health": get_ai_service_health("whisper")
            }
        except Exception as e:
            logger.error("Failed to get Whisper health", error=str(e))
            health_data["services"]["whisper"] = {
                "status": "unhealthy",
                "error": str(e),
                "resilience_health": get_ai_service_health("whisper")
            }
        
        # Determine overall status
        service_statuses = [
            service.get("status", "unknown") 
            for service in health_data["services"].values()
        ]
        
        if any(status == "unhealthy" for status in service_statuses):
            health_data["overall_status"] = "degraded"
        elif any(status == "unknown" for status in service_statuses):
            health_data["overall_status"] = "unknown"
        
        return health_data
        
    except Exception as e:
        logger.error("Failed to get AI services health", error=str(e))
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/ai-services/{service_name}")
async def ai_service_health(service_name: str) -> Dict[str, Any]:
    """Get health status for specific AI service."""
    try:
        if service_name.lower() == "gemini":
            health = await gemini_service.health_check()
            return health
        elif service_name.lower() == "whisper":
            return {
                "status": "healthy",
                "available_models": whisper_service.get_available_models(),
                "loaded_models": len(whisper_service._models),
                "resilience_health": get_ai_service_health("whisper")
            }
        else:
            # Get resilience health for any service
            resilience_health = get_ai_service_health(service_name)
            if "error" in resilience_health:
                raise HTTPException(status_code=404, detail=f"Service not found: {service_name}")
            return {
                "service": service_name,
                "resilience_health": resilience_health
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get service health", service=service_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/circuit-breakers")
async def circuit_breakers_status() -> Dict[str, Any]:
    """Get status of all circuit breakers."""
    try:
        ai_health = get_ai_service_health()
        circuit_breakers = {}
        
        for service_name, health_data in ai_health.items():
            if isinstance(health_data, dict) and "circuit_breaker" in health_data:
                circuit_breakers[service_name] = health_data["circuit_breaker"]
        
        return {
            "circuit_breakers": circuit_breakers,
            "total_services": len(circuit_breakers)
        }
        
    except Exception as e:
        logger.error("Failed to get circuit breaker status", error=str(e))
        raise HTTPException(status_code=500, detail=f"Circuit breaker status check failed: {str(e)}")


@router.get("/metrics")
async def ai_metrics() -> Dict[str, Any]:
    """Get comprehensive AI service metrics."""
    try:
        ai_health = get_ai_service_health()
        metrics = {
            "timestamp": None,
            "services": {},
            "summary": {
                "total_services": 0,
                "healthy_services": 0,
                "degraded_services": 0,
                "unhealthy_services": 0,
                "total_requests": 0,
                "total_successes": 0,
                "total_failures": 0,
                "overall_success_rate": 0.0
            }
        }
        
        total_requests = 0
        total_successes = 0
        total_failures = 0
        
        for service_name, health_data in ai_health.items():
            if isinstance(health_data, dict):
                metrics["services"][service_name] = health_data
                
                # Aggregate metrics
                if "recent_attempts" in health_data:
                    total_requests += health_data.get("recent_attempts", 0)
                if "recent_successes" in health_data:
                    total_successes += health_data.get("recent_successes", 0)
                if "recent_failures" in health_data:
                    total_failures += health_data.get("recent_failures", 0)
                
                # Count service health status
                health_status = health_data.get("health_status", "unknown")
                if health_status == "healthy":
                    metrics["summary"]["healthy_services"] += 1
                elif health_status == "degraded":
                    metrics["summary"]["degraded_services"] += 1
                elif health_status == "unhealthy":
                    metrics["summary"]["unhealthy_services"] += 1
                
                metrics["summary"]["total_services"] += 1
        
        # Calculate overall success rate
        metrics["summary"]["total_requests"] = total_requests
        metrics["summary"]["total_successes"] = total_successes
        metrics["summary"]["total_failures"] = total_failures
        
        if total_requests > 0:
            metrics["summary"]["overall_success_rate"] = total_successes / total_requests
        
        return metrics
        
    except Exception as e:
        logger.error("Failed to get AI metrics", error=str(e))
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {str(e)}")


@router.post("/circuit-breakers/{service_name}/reset")
async def reset_circuit_breaker(service_name: str) -> Dict[str, Any]:
    """Reset circuit breaker for specific service."""
    try:
        # This would require access to the actual circuit breaker instance
        # For now, return a placeholder response
        return {
            "service": service_name,
            "action": "reset_requested",
            "message": "Circuit breaker reset functionality not yet implemented",
            "status": "pending"
        }
        
    except Exception as e:
        logger.error("Failed to reset circuit breaker", service=service_name, error=str(e))
        raise HTTPException(status_code=500, detail=f"Circuit breaker reset failed: {str(e)}")
