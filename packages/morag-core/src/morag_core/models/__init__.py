"""Data models for MoRAG."""

from .api import (
    BatchIngestionResponse,
    ErrorResponse,
    IngestionResponse,
    SourceType,
    TaskStatusResponse,
)
from .config import (
    AudioProcessingConfig,
    DocumentProcessingConfig,
    EmbeddingConfig,
    ImageProcessingConfig,
    ProcessingConfig,
    ProcessingResult,
    VideoProcessingConfig,
    WebProcessingConfig,
)
from .document import Document, DocumentChunk, DocumentMetadata, DocumentType
from .embedding import BatchEmbeddingResult, EmbeddingResult, SummaryResult
from .remote_job import RemoteJob

__all__ = [
    # API models
    "SourceType",
    "ErrorResponse",
    "TaskStatusResponse",
    "IngestionResponse",
    "BatchIngestionResponse",
    # Document models
    "DocumentType",
    "DocumentMetadata",
    "DocumentChunk",
    "Document",
    # Embedding models
    "EmbeddingResult",
    "BatchEmbeddingResult",
    "SummaryResult",
    # Configuration models
    "ProcessingConfig",
    "DocumentProcessingConfig",
    "AudioProcessingConfig",
    "VideoProcessingConfig",
    "ImageProcessingConfig",
    "WebProcessingConfig",
    "EmbeddingConfig",
    "ProcessingResult",
    # Remote job models
    "RemoteJob",
]
