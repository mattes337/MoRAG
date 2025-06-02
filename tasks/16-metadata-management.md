# Task 16: Metadata Management

## Overview
Implement comprehensive metadata extraction, association, and management for all content types processed by MoRAG. This includes file metadata, content metadata, processing metadata, and custom user metadata.

## Prerequisites
- Task 01: Project Setup completed
- Task 03: Database Setup completed
- Task 05: Document Parser completed
- Task 08: Audio Processing completed
- Task 10: Image Processing completed
- Task 15: Vector Storage completed

## Dependencies
- Task 01: Project Setup
- Task 03: Database Setup
- Task 05: Document Parser
- Task 08: Audio Processing
- Task 10: Image Processing
- Task 15: Vector Storage

## Implementation Steps

### 1. Metadata Schema Definition
Create `src/morag/models/metadata.py`:
```python
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
```

### 2. Metadata Extraction Service
Create `src/morag/services/metadata_service.py`:
```python
from typing import Dict, Any, Optional, Union
from pathlib import Path
import hashlib
import mimetypes
import structlog
from datetime import datetime

from morag.models.metadata import (
    ComprehensiveMetadata, FileMetadata, ContentMetadata, 
    ProcessingMetadata, UserMetadata, SystemMetadata
)
from morag.core.exceptions import ValidationError

logger = structlog.get_logger()

class MetadataService:
    """Service for extracting and managing metadata."""
    
    def __init__(self):
        self.processor_version = "1.0.0"
    
    async def extract_file_metadata(
        self, 
        file_path: Union[str, Path],
        original_filename: Optional[str] = None
    ) -> FileMetadata:
        """Extract metadata from file system."""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        # Get file stats
        stat = file_path.stat()
        file_size = stat.st_size
        
        # Detect MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if not mime_type:
            mime_type = "application/octet-stream"
        
        # Generate file hash
        file_hash = await self._calculate_file_hash(file_path)
        
        return FileMetadata(
            original_filename=original_filename or file_path.name,
            file_size=file_size,
            mime_type=mime_type,
            file_extension=file_path.suffix.lower().lstrip('.'),
            file_hash=file_hash,
            file_path=str(file_path),
            upload_timestamp=datetime.fromtimestamp(stat.st_ctime)
        )
    
    async def extract_content_metadata(
        self,
        file_path: Union[str, Path],
        content_type: str,
        extracted_data: Dict[str, Any] = None
    ) -> ContentMetadata:
        """Extract content-specific metadata."""
        metadata = ContentMetadata()
        
        if extracted_data:
            # Map extracted data to content metadata
            metadata.title = extracted_data.get('title')
            metadata.author = extracted_data.get('author')
            metadata.language = extracted_data.get('language')
            metadata.word_count = extracted_data.get('word_count')
            metadata.duration = extracted_data.get('duration')
            metadata.width = extracted_data.get('width')
            metadata.height = extracted_data.get('height')
            
            # Handle dates
            if 'creation_date' in extracted_data:
                metadata.creation_date = self._parse_date(extracted_data['creation_date'])
            if 'modification_date' in extracted_data:
                metadata.modification_date = self._parse_date(extracted_data['modification_date'])
        
        return metadata
    
    def create_processing_metadata(
        self,
        processing_steps: List[str],
        **kwargs
    ) -> ProcessingMetadata:
        """Create processing metadata."""
        return ProcessingMetadata(
            processor_version=self.processor_version,
            processing_steps=processing_steps,
            **kwargs
        )
    
    def create_user_metadata(
        self,
        user_data: Dict[str, Any]
    ) -> UserMetadata:
        """Create user metadata from provided data."""
        return UserMetadata(
            tags=user_data.get('tags', []),
            categories=user_data.get('categories', []),
            custom_fields=user_data.get('custom_fields', {}),
            notes=user_data.get('notes'),
            priority=user_data.get('priority')
        )
    
    def create_system_metadata(
        self,
        ingestion_id: str,
        collection_name: str,
        task_id: Optional[str] = None
    ) -> SystemMetadata:
        """Create system metadata."""
        return SystemMetadata(
            ingestion_id=ingestion_id,
            task_id=task_id,
            collection_name=collection_name
        )
    
    async def create_comprehensive_metadata(
        self,
        file_path: Union[str, Path],
        ingestion_id: str,
        collection_name: str,
        content_type: str,
        user_data: Dict[str, Any] = None,
        extracted_data: Dict[str, Any] = None,
        processing_steps: List[str] = None,
        task_id: Optional[str] = None,
        original_filename: Optional[str] = None
    ) -> ComprehensiveMetadata:
        """Create comprehensive metadata for content."""
        
        # Extract file metadata
        file_metadata = await self.extract_file_metadata(file_path, original_filename)
        
        # Extract content metadata
        content_metadata = await self.extract_content_metadata(
            file_path, content_type, extracted_data
        )
        
        # Create processing metadata
        processing_metadata = self.create_processing_metadata(
            processing_steps or [],
            **{k: v for k, v in (extracted_data or {}).items() 
               if k in ['chunk_count', 'embedding_model', 'summary_generated', 
                       'ocr_performed', 'transcription_performed']}
        )
        
        # Create user metadata
        user_metadata = self.create_user_metadata(user_data or {})
        
        # Create system metadata
        system_metadata = self.create_system_metadata(
            ingestion_id, collection_name, task_id
        )
        
        return ComprehensiveMetadata(
            file=file_metadata,
            content=content_metadata,
            processing=processing_metadata,
            user=user_metadata,
            system=system_metadata
        )
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _parse_date(self, date_value: Any) -> Optional[datetime]:
        """Parse date from various formats."""
        if isinstance(date_value, datetime):
            return date_value
        elif isinstance(date_value, str):
            try:
                return datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            except ValueError:
                logger.warning("Failed to parse date", date_value=date_value)
                return None
        return None

# Global instance
metadata_service = MetadataService()
```

## Success Criteria
- [ ] Metadata models defined for all content types
- [ ] Metadata extraction service implemented
- [ ] File metadata extraction works
- [ ] Content metadata extraction works
- [ ] Processing metadata tracking works
- [ ] User metadata handling works
- [ ] System metadata generation works
- [ ] Comprehensive metadata creation works
- [ ] Integration with existing processors works
- [ ] Tests pass with >95% coverage

## Next Steps
- Task 17: Ingestion API (uses metadata management)
- Task 18: Status Tracking (enhanced with metadata)
- Update existing processors to use metadata service

## Notes
- Metadata is stored alongside content in Qdrant
- Searchable metadata improves content discovery
- Extensible design allows for new metadata types
- Comprehensive metadata enables advanced filtering and search
