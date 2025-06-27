"""API endpoints for MoRAG."""

from .remote_jobs import router as remote_jobs_router
from .enhanced_query import router as enhanced_query_router
# Legacy router temporarily disabled
# from .legacy import router as legacy_router

__all__ = [
    "remote_jobs_router",
    "enhanced_query_router",
    # "legacy_router",
]
