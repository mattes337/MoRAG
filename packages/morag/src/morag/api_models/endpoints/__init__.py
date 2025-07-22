"""API endpoint modules."""

from .processing import processing_router
from .ingestion import ingestion_router
from .search import search_router
from .tasks import tasks_router
from .admin import admin_router

__all__ = [
    "processing_router",
    "ingestion_router", 
    "search_router",
    "tasks_router",
    "admin_router"
]
