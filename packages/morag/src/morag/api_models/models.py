"""Pydantic models for MoRAG API."""

from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field


class ProcessURLRequest(BaseModel):
    """Request model for processing URLs."""
    url: str
    content_type: Optional[str] = None
    options: Optional[Dict[str, Any]] = None


class ProcessBatchRequest(BaseModel):
    """Request model for batch processing."""
    items: List[Dict[str, Any]]
    options: Optional[Dict[str, Any]] = None


class SearchRequest(BaseModel):
    """Request model for search operations."""
    query: str
    limit: int = 10
    database_config: Optional[Dict[str, Any]] = None
    retrieval_config: Optional[Dict[str, Any]] = None


class ProcessingResultResponse(BaseModel):
    """Response model for processing results."""
    success: bool
    content: str
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    warnings: Optional[List[str]] = None
    error_message: Optional[str] = None
    thumbnails: Optional[List[str]] = None


class MarkdownConversionRequest(BaseModel):
    """Request model for markdown conversion."""
    include_metadata: bool = Field(default=True, description="Include file metadata in response")
    language: Optional[str] = Field(default=None, description="Language hint for processing")


class MarkdownConversionResponse(BaseModel):
    """Response model for markdown conversion."""
    success: bool
    markdown: str = Field(description="Converted markdown content")
    metadata: Dict[str, Any] = Field(description="File metadata and conversion info")
    processing_time_ms: float = Field(description="Processing time in milliseconds")
    error_message: Optional[str] = None


class ProcessIngestRequest(BaseModel):
    """Request model for processing with ingestion and webhooks."""
    document_id: Optional[str] = Field(default=None, description="Optional document ID for deduplication")
    webhook_url: str = Field(description="URL for webhook progress notifications")
    webhook_auth_token: Optional[str] = Field(default=None, description="Optional bearer token for webhook authentication")
    collection_name: Optional[str] = Field(default=None, description="Target collection name")
    language: Optional[str] = Field(default=None, description="Language hint for processing")
    chunking_strategy: Optional[str] = Field(default=None, description="Chunking strategy to use")
    chunk_size: Optional[int] = Field(default=None, description="Chunk size for text splitting")
    chunk_overlap: Optional[int] = Field(default=None, description="Overlap between chunks")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")


class ProcessIngestResponse(BaseModel):
    """Response model for processing with ingestion."""
    success: bool
    task_id: str = Field(description="Background task ID for tracking progress")
    document_id: Optional[str] = Field(description="Document ID (user-provided or generated)")
    estimated_time_seconds: int = Field(description="Estimated processing time in seconds")
    status_url: str = Field(description="URL to check task status")
    message: str = Field(description="Human-readable status message")


class WebhookProgressNotification(BaseModel):
    """Model for webhook progress notifications."""
    task_id: str = Field(description="Background task ID")
    document_id: Optional[str] = Field(description="Document ID if provided")
    step: str = Field(description="Current processing step")
    status: str = Field(description="Step status: started|completed|failed")
    progress_percent: float = Field(description="Overall progress percentage (0-100)")
    timestamp: str = Field(description="ISO8601 timestamp")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Step-specific data")
    error_message: Optional[str] = Field(default=None, description="Error message if status is failed")


class IngestFileRequest(BaseModel):
    """Request model for file ingestion."""
    source_type: Optional[str] = None  # Auto-detect if not provided
    document_id: Optional[str] = None  # Custom document identifier
    collection_name: Optional[str] = None  # Target collection
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    chunking_strategy: Optional[str] = None  # Chunking strategy
    chunk_size: Optional[int] = None  # Chunk size
    chunk_overlap: Optional[int] = None  # Chunk overlap
    database_config: Optional[Dict[str, Any]] = None  # Database configuration
    webhook_url: Optional[str] = None  # Webhook for completion notification


class IngestURLRequest(BaseModel):
    """Request model for URL ingestion."""
    source_type: Optional[str] = None  # Auto-detect if not provided
    url: str
    document_id: Optional[str] = None  # Custom document identifier
    collection_name: Optional[str] = None  # Target collection
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    database_config: Optional[Dict[str, Any]] = None  # Database configuration
    webhook_url: Optional[str] = None  # Webhook for completion notification


class IngestBatchRequest(BaseModel):
    """Request model for batch ingestion."""
    items: List[Dict[str, Any]]
    webhook_url: Optional[str] = None
    database_config: Optional[Dict[str, Any]] = None  # Database configuration


class IngestRemoteFileRequest(BaseModel):
    """Request model for remote file ingestion."""
    file_path: str  # UNC path or HTTP/HTTPS URL
    source_type: Optional[str] = None  # Auto-detect if not provided
    document_id: Optional[str] = None  # Custom document identifier
    collection_name: Optional[str] = None  # Target collection
    metadata: Optional[Dict[str, Any]] = None  # Additional metadata
    chunking_strategy: Optional[str] = None  # Chunking strategy
    chunk_size: Optional[int] = None  # Chunk size
    chunk_overlap: Optional[int] = None  # Chunk overlap
    database_config: Optional[Dict[str, Any]] = None  # Database configuration
    webhook_url: Optional[str] = None  # Webhook for completion notification


class ProcessRemoteFileRequest(BaseModel):
    """Request model for remote file processing."""
    file_path: str  # UNC path or HTTP/HTTPS URL
    content_type: Optional[str] = None  # Auto-detect if not provided
    options: Optional[Dict[str, Any]] = None


class IngestResponse(BaseModel):
    """Response model for ingestion operations."""
    task_id: str
    status: str = "pending"
    message: Optional[str] = None


class BatchIngestResponse(BaseModel):
    """Response model for batch ingestion operations."""
    batch_id: str
    task_ids: List[str]
    status: str = "pending"


class TaskStatus(BaseModel):
    """Model for task status information."""
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    progress: Optional[float] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


# ============================================================================
# UNIFIED API MODELS - New consolidated models for unified processing endpoint
# ============================================================================

class DatabaseConfig(BaseModel):
    """Database configuration for ingestion."""
    neo4j_uri: Optional[str] = None
    neo4j_user: Optional[str] = None
    neo4j_password: Optional[str] = None
    qdrant_url: Optional[str] = None
    qdrant_api_key: Optional[str] = None
    collection_name: Optional[str] = None


class ProcessingOptions(BaseModel):
    """Processing configuration options."""
    language: Optional[str] = Field(default=None, description="Language hint for processing")
    chunking_strategy: Optional[str] = Field(default=None, description="Chunking strategy to use")
    chunk_size: Optional[int] = Field(default=None, description="Chunk size for text splitting")
    chunk_overlap: Optional[int] = Field(default=None, description="Overlap between chunks")
    include_thumbnails: bool = Field(default=False, description="Include thumbnails for video content")
    include_metadata: bool = Field(default=True, description="Include metadata in response")


class WebhookConfig(BaseModel):
    """Webhook configuration for async processing."""
    url: str = Field(description="Webhook URL for progress notifications")
    auth_token: Optional[str] = Field(default=None, description="Optional bearer token for webhook authentication")


class UnifiedProcessRequest(BaseModel):
    """Unified request model for all processing operations."""

    # Processing mode - determines the type of processing
    mode: Literal["convert", "process", "ingest"] = Field(
        description="Processing mode: 'convert' for markdown only, 'process' for full processing, 'ingest' for processing + storage"
    )

    # Source type - determines input method
    source_type: Literal["file", "url", "batch"] = Field(
        description="Source type: 'file' for uploads, 'url' for web content, 'batch' for multiple items"
    )

    # Input data (one of these will be used based on source_type)
    url: Optional[str] = Field(default=None, description="URL to process (for source_type='url')")
    items: Optional[List[Dict[str, Any]]] = Field(default=None, description="Batch items to process (for source_type='batch')")

    # Content type hint
    content_type: Optional[str] = Field(default=None, description="Content type hint for processing")

    # Document identification and deduplication
    document_id: Optional[str] = Field(default=None, description="Optional document ID for deduplication")
    replace_existing: bool = Field(default=False, description="Replace existing document if document_id matches")

    # Processing configuration
    processing_options: Optional[ProcessingOptions] = Field(default=None, description="Processing configuration options")

    # Database configuration (for mode='ingest')
    database_config: Optional[DatabaseConfig] = Field(default=None, description="Database configuration for ingestion")

    # Webhook configuration (for async processing)
    webhook_config: Optional[WebhookConfig] = Field(default=None, description="Webhook configuration for progress notifications")

    # Additional metadata
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata to store with document")


class UnifiedProcessResponse(BaseModel):
    """Unified response model for all processing operations."""

    success: bool = Field(description="Whether the operation was successful")
    mode: str = Field(description="Processing mode that was used")

    # Immediate response data (for mode='convert' and mode='process')
    content: Optional[str] = Field(default=None, description="Processed content (markdown, full text, etc.)")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Processing metadata")
    processing_time_ms: Optional[float] = Field(default=None, description="Processing time in milliseconds")

    # Async response data (for mode='ingest')
    task_id: Optional[str] = Field(default=None, description="Background task ID for tracking progress")
    estimated_time_seconds: Optional[int] = Field(default=None, description="Estimated processing time in seconds")
    status_url: Optional[str] = Field(default=None, description="URL to check processing status")

    # Document identification
    document_id: Optional[str] = Field(default=None, description="Document ID (user-provided or generated)")

    # Error handling
    error_message: Optional[str] = Field(default=None, description="Error message if operation failed")
    warnings: Optional[List[str]] = Field(default=None, description="Warning messages")

    # Additional response data
    thumbnails: Optional[List[str]] = Field(default=None, description="Base64 encoded thumbnails for video content")
    message: Optional[str] = Field(default=None, description="Human-readable status message")
