"""Data models for MoRAG."""

from .api import (
    SourceType,
    ErrorResponse,
    TaskStatusResponse,
    IngestionResponse,
    BatchIngestionResponse,
)
from .document import (
    DocumentType,
    DocumentMetadata,
    DocumentChunk,
    Document,
)
from .embedding import (
    EmbeddingResult,
    BatchEmbeddingResult,
    SummaryResult,
)
from .config import (
    ProcessingConfig,
    DocumentProcessingConfig,
    AudioProcessingConfig,
    VideoProcessingConfig,
    ImageProcessingConfig,
    WebProcessingConfig,
    EmbeddingConfig,
    ProcessingResult,
)
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