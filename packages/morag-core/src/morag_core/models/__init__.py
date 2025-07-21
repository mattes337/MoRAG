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
# Import database config from parent config module for backward compatibility
from ..config import DatabaseConfig, DatabaseType, DatabaseServerConfig
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
    # Database configuration models (backward compatibility)
    "DatabaseConfig",
    "DatabaseType",
    "DatabaseServerConfig",
    # Remote job models
    "RemoteJob",
]