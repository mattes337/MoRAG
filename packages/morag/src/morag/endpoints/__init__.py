"""API endpoints for MoRAG."""

from .remote_jobs import router as remote_jobs_router

__all__ = [
    "remote_jobs_router",
]
