from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class MetadataType(str, Enum):
    """Types of metadata."""
    FILE = "file"
    CONTENT = "content"
    PROCESSING = "processing"
    USER = "user"
    SYSTEM = "system"

class FileMetadata(BaseModel):
    """File-specific metadata."""
    original_filename: str
    file_size: int
    mime_type: str
    file_extension: str
    file_hash: Optional[str] = None
    upload_timestamp: datetime = Field(default_factory=datetime.utcnow)
    file_path: Optional[str] = None

class ContentMetadata(BaseModel):
    """Content-specific metadata extracted from the file."""
    # Document metadata
    title: Optional[str] = None
    author: Optional[str] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    language: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    
    # Audio/Video metadata
    duration: Optional[float] = None
    bitrate: Optional[int] = None
    sample_rate: Optional[int] = None
    channels: Optional[int] = None
    codec: Optional[str] = None
    
    # Image metadata
    width: Optional[int] = None
    height: Optional[int] = None
    color_mode: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    
    # Web content metadata
    url: Optional[str] = None
    domain: Optional[str] = None
    last_modified: Optional[datetime] = None
    content_type: Optional[str] = None

class ProcessingMetadata(BaseModel):
    """Processing-specific metadata."""
    processing_start: datetime = Field(default_factory=datetime.utcnow)
    processing_end: Optional[datetime] = None
    processing_duration: Optional[float] = None
    processor_version: str
    processing_steps: List[str] = Field(default_factory=list)
    chunk_count: Optional[int] = None
    embedding_model: Optional[str] = None
    summary_generated: bool = False
    ocr_performed: bool = False
    transcription_performed: bool = False
    
class UserMetadata(BaseModel):
    """User-provided metadata."""
    tags: List[str] = Field(default_factory=list)
    categories: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None
    priority: Optional[int] = Field(None, ge=1, le=5)

class SystemMetadata(BaseModel):
    """System-generated metadata."""
    ingestion_id: str
    task_id: Optional[str] = None
    collection_name: str
    storage_backend: str = "qdrant"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    version: int = 1

class ComprehensiveMetadata(BaseModel):
    """Complete metadata container."""
    file: Optional[FileMetadata] = None
    content: Optional[ContentMetadata] = None
    processing: Optional[ProcessingMetadata] = None
    user: Optional[UserMetadata] = None
    system: SystemMetadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return self.model_dump(exclude_none=True)
    
    def get_searchable_text(self) -> str:
        """Extract searchable text from metadata."""
        searchable_parts = []
        
        if self.file:
            searchable_parts.append(self.file.original_filename)
        
        if self.content:
            if self.content.title:
                searchable_parts.append(self.content.title)
            if self.content.author:
                searchable_parts.append(self.content.author)
        
        if self.user:
            searchable_parts.extend(self.user.tags)
            searchable_parts.extend(self.user.categories)
            if self.user.notes:
                searchable_parts.append(self.user.notes)
        
        return " ".join(filter(None, searchable_parts))
