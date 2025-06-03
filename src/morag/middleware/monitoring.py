import time
import asyncio
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
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
