"""Document management models."""

from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ARCHIVED = "ARCHIVED"


class DocumentType(str, Enum):
    PDF = "PDF"
    TEXT = "TEXT"
    MARKDOWN = "MARKDOWN"
    HTML = "HTML"
    DOCX = "DOCX"
    AUDIO = "AUDIO"
    VIDEO = "VIDEO"
    IMAGE = "IMAGE"
    URL = "URL"
    YOUTUBE = "YOUTUBE"
    WEB_PAGE = "WEB_PAGE"
    UNKNOWN = "UNKNOWN"


class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=500)
    source_path: Optional[str] = None
    source_url: Optional[HttpUrl] = None
    content_type: Optional[DocumentType] = None
    metadata: Optional[Dict[str, Any]] = None
    database_id: Optional[str] = None  # Target database
    collection_name: Optional[str] = None  # Target collection
    
    # Processing options
    use_docling: Optional[bool] = False
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    chunking_strategy: Optional[str] = None
    replace_existing: Optional[bool] = False


class DocumentUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=500)
    status: Optional[DocumentStatus] = None
    metadata: Optional[Dict[str, Any]] = None
    content_checksum: Optional[str] = None
    processing_error: Optional[str] = None


class DocumentResponse(BaseModel):
    id: str
    title: str
    source_path: Optional[str]
    source_url: Optional[str]
    content_type: DocumentType
    status: DocumentStatus
    content_checksum: Optional[str]
    metadata: Dict[str, Any]
    processing_error: Optional[str]
    user_id: str
    database_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    processed_at: Optional[datetime]
    
    # Vector database information
    vector_point_ids: Optional[List[str]] = None
    collection_name: Optional[str] = None
    chunk_count: Optional[int] = None

    class Config:
        from_attributes = True


class DocumentSearchRequest(BaseModel):
    query: Optional[str] = None
    status: Optional[DocumentStatus] = None
    content_type: Optional[DocumentType] = None
    database_id: Optional[str] = None
    user_id: Optional[str] = None  # Admin only
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: str = Field(default="created_at")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")


class DocumentSearchResponse(BaseModel):
    documents: List[DocumentResponse]
    total_count: int
    limit: int
    offset: int
    has_more: bool


class DocumentStatsResponse(BaseModel):
    total_documents: int
    by_status: Dict[str, int]
    by_content_type: Dict[str, int]
    by_database: Dict[str, int]
    total_chunks: int
    total_size_bytes: int
    processing_errors: int


class DocumentBatchOperation(BaseModel):
    document_ids: List[str]
    operation: str = Field(..., pattern="^(delete|archive|reprocess)$")
    options: Optional[Dict[str, Any]] = None


class DocumentBatchResponse(BaseModel):
    success_count: int
    failure_count: int
    errors: List[Dict[str, str]]
    operation: str
