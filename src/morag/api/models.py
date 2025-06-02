from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List, Union
from enum import Enum
import mimetypes

class SourceType(str, Enum):
    """Supported source types for ingestion."""
    DOCUMENT = "document"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    WEB = "web"
    YOUTUBE = "youtube"

class IngestionRequest(BaseModel):
    """Base request for content ingestion."""
    source_type: SourceType
    webhook_url: Optional[str] = Field(None, description="URL to notify when processing completes")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        use_enum_values = True

class FileIngestionRequest(IngestionRequest):
    """Request for file-based ingestion (documents, audio, video, images)."""
    # File will be uploaded via multipart form data
    use_docling: Optional[bool] = Field(False, description="Use docling for PDF parsing")
    
class URLIngestionRequest(IngestionRequest):
    """Request for URL-based ingestion (web, youtube)."""
    url: str = Field(..., description="URL to process")
    
    @validator('url')
    def validate_url(cls, v):
        if not v.startswith(('http://', 'https://')):
            raise ValueError('URL must start with http:// or https://')
        return v

class IngestionResponse(BaseModel):
    """Response for ingestion request."""
    task_id: str = Field(..., description="Unique task identifier")
    status: str = Field(..., description="Initial task status")
    message: str = Field(..., description="Human-readable message")
    estimated_time: Optional[int] = Field(None, description="Estimated processing time in seconds")

class TaskStatusResponse(BaseModel):
    """Response for task status check."""
    task_id: str
    status: str
    progress: Optional[float] = Field(None, ge=0.0, le=1.0)
    message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    estimated_time_remaining: Optional[int] = None

class BatchIngestionRequest(BaseModel):
    """Request for batch ingestion of multiple items."""
    items: List[Union[FileIngestionRequest, URLIngestionRequest]]
    webhook_url: Optional[str] = None
    
    @validator('items')
    def validate_items(cls, v):
        if len(v) == 0:
            raise ValueError('At least one item is required')
        if len(v) > 50:  # Reasonable batch limit
            raise ValueError('Maximum 50 items per batch')
        return v

class BatchIngestionResponse(BaseModel):
    """Response for batch ingestion."""
    batch_id: str
    task_ids: List[str]
    total_items: int
    message: str

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
