"""Document management package for MoRAG core."""

from .models import (
    DocumentCreate,
    DocumentUpdate,
    DocumentResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
    DocumentStatsResponse,
    DocumentBatchOperation,
    DocumentBatchResponse,
    DocumentStatus,
    DocumentType,
)
from .service import DocumentService
from .lifecycle import DocumentLifecycleManager

__all__ = [
    # Models
    "DocumentCreate",
    "DocumentUpdate",
    "DocumentResponse",
    "DocumentSearchRequest",
    "DocumentSearchResponse",
    "DocumentStatsResponse",
    "DocumentBatchOperation",
    "DocumentBatchResponse",
    "DocumentStatus",
    "DocumentType",
    # Services
    "DocumentService",
    "DocumentLifecycleManager",
]
