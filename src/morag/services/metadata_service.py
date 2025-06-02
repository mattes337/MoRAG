from typing import Dict, Any, Optional, Union, List
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
            metadata.bitrate = extracted_data.get('bitrate')
            metadata.sample_rate = extracted_data.get('sample_rate')
            metadata.channels = extracted_data.get('channels')
            metadata.codec = extracted_data.get('codec')
            metadata.color_mode = extracted_data.get('color_mode')
            metadata.camera_make = extracted_data.get('camera_make')
            metadata.camera_model = extracted_data.get('camera_model')
            metadata.url = extracted_data.get('url')
            metadata.domain = extracted_data.get('domain')
            metadata.content_type = extracted_data.get('content_type')
            metadata.page_count = extracted_data.get('page_count')
            
            # Handle dates
            if 'creation_date' in extracted_data:
                metadata.creation_date = self._parse_date(extracted_data['creation_date'])
            if 'modification_date' in extracted_data:
                metadata.modification_date = self._parse_date(extracted_data['modification_date'])
            if 'last_modified' in extracted_data:
                metadata.last_modified = self._parse_date(extracted_data['last_modified'])
        
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
