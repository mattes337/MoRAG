"""Document lifecycle management."""

from typing import Optional, Dict, Any
from datetime import datetime, timezone
import structlog

from .service import DocumentService
from .models import DocumentCreate, DocumentUpdate, DocumentStatus, DocumentType
from morag_core.exceptions import ValidationError, NotFoundError

logger = structlog.get_logger(__name__)


class DocumentLifecycleManager:
    """Manages document lifecycle from creation to completion."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.document_service = DocumentService(database_url)
    
    def start_document_processing(
        self, 
        title: str,
        source_path: Optional[str] = None,
        source_url: Optional[str] = None,
        content_type: Optional[DocumentType] = None,
        user_id: str = None,
        metadata: Optional[Dict[str, Any]] = None,
        **processing_options
    ) -> str:
        """Start document processing lifecycle."""
        try:
            # Validate input
            if not source_path and not source_url:
                raise ValidationError("Either source_path or source_url must be provided")
            
            # Auto-detect content type if not provided
            if not content_type:
                content_type = self._detect_content_type(source_path, source_url)
            
            # Create document record
            document_data = DocumentCreate(
                title=title,
                source_path=source_path,
                source_url=source_url,
                content_type=content_type,
                metadata=metadata or {},
                **processing_options
            )
            
            document = self.document_service.create_document(document_data, user_id)
            
            logger.info("Document processing started",
                       document_id=document.id,
                       title=title,
                       content_type=content_type.value if content_type else None,
                       user_id=user_id)
            
            return document.id
            
        except Exception as e:
            logger.error("Failed to start document processing", 
                        title=title, error=str(e))
            raise
    
    def update_processing_status(
        self,
        document_id: str,
        status: DocumentStatus,
        metadata: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """Update document processing status."""
        try:
            update_data = DocumentUpdate(
                status=status,
                metadata=metadata,
                processing_error=error_message
            )
            
            self.document_service.update_document(document_id, update_data, user_id)
            
            logger.info("Document status updated",
                       document_id=document_id,
                       status=status.value,
                       has_error=bool(error_message))
            
            return True
            
        except Exception as e:
            logger.error("Failed to update document status",
                        document_id=document_id,
                        status=status.value if status else None,
                        error=str(e))
            return False
    
    def mark_processing_started(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """Mark document as processing started."""
        return self.update_processing_status(
            document_id, 
            DocumentStatus.PROCESSING,
            user_id=user_id
        )
    
    def mark_processing_completed(
        self,
        document_id: str,
        vector_point_ids: Optional[list] = None,
        collection_name: Optional[str] = None,
        chunk_count: Optional[int] = None,
        processing_metadata: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """Mark document as processing completed."""
        metadata = processing_metadata or {}
        
        if vector_point_ids:
            metadata['vector_point_ids'] = vector_point_ids
        if collection_name:
            metadata['collection_name'] = collection_name
        if chunk_count:
            metadata['chunk_count'] = chunk_count
        
        return self.update_processing_status(
            document_id,
            DocumentStatus.COMPLETED,
            metadata=metadata,
            user_id=user_id
        )
    
    def mark_processing_failed(
        self,
        document_id: str,
        error_message: str,
        user_id: Optional[str] = None
    ) -> bool:
        """Mark document as processing failed."""
        return self.update_processing_status(
            document_id,
            DocumentStatus.FAILED,
            error_message=error_message,
            user_id=user_id
        )
    
    def archive_document(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """Archive a document."""
        return self.update_processing_status(
            document_id,
            DocumentStatus.ARCHIVED,
            user_id=user_id
        )
    
    def reprocess_document(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """Reset document for reprocessing."""
        return self.update_processing_status(
            document_id,
            DocumentStatus.PENDING,
            error_message=None,  # Clear previous error
            user_id=user_id
        )
    
    def get_processing_status(self, document_id: str, user_id: Optional[str] = None) -> Optional[DocumentStatus]:
        """Get current processing status of a document."""
        document = self.document_service.get_document(document_id, user_id)
        if document:
            return DocumentStatus(document.status)
        return None
    
    def cleanup_failed_documents(self, older_than_hours: int = 24, user_id: Optional[str] = None) -> int:
        """Clean up failed documents older than specified hours."""
        try:
            from datetime import timedelta
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
            
            # Search for failed documents
            from .models import DocumentSearchRequest
            search_request = DocumentSearchRequest(
                status=DocumentStatus.FAILED,
                created_before=cutoff_time,
                limit=1000  # Process in batches
            )
            
            search_result = self.document_service.search_documents(search_request, user_id)
            
            # Delete failed documents
            deleted_count = 0
            for document in search_result.documents:
                if self.document_service.delete_document(document.id, user_id):
                    deleted_count += 1
            
            logger.info("Cleaned up failed documents",
                       deleted_count=deleted_count,
                       cutoff_hours=older_than_hours)
            
            return deleted_count
            
        except Exception as e:
            logger.error("Failed to cleanup failed documents", error=str(e))
            return 0
    
    def _detect_content_type(self, source_path: Optional[str], source_url: Optional[str]) -> DocumentType:
        """Auto-detect content type from source."""
        if source_url:
            url_str = str(source_url).lower()
            if 'youtube.com' in url_str or 'youtu.be' in url_str:
                return DocumentType.YOUTUBE
            elif any(ext in url_str for ext in ['.pdf']):
                return DocumentType.PDF
            elif any(ext in url_str for ext in ['.mp4', '.avi', '.mov', '.mkv']):
                return DocumentType.VIDEO
            elif any(ext in url_str for ext in ['.mp3', '.wav', '.m4a', '.flac']):
                return DocumentType.AUDIO
            else:
                return DocumentType.WEB_PAGE
        
        if source_path:
            from pathlib import Path
            path = Path(source_path)
            suffix = path.suffix.lower()
            
            if suffix == '.pdf':
                return DocumentType.PDF
            elif suffix in ['.txt']:
                return DocumentType.TEXT
            elif suffix in ['.md', '.markdown']:
                return DocumentType.MARKDOWN
            elif suffix in ['.html', '.htm']:
                return DocumentType.HTML
            elif suffix in ['.docx', '.doc']:
                return DocumentType.DOCX
            elif suffix in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                return DocumentType.VIDEO
            elif suffix in ['.mp3', '.wav', '.m4a', '.flac', '.ogg']:
                return DocumentType.AUDIO
            elif suffix in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']:
                return DocumentType.IMAGE
        
        return DocumentType.UNKNOWN
