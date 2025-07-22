"""Pydantic models for MoRAG API."""

from typing import Dict, Any, List, Optional
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
