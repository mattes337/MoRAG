"""Document management service."""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
from datetime import datetime, timezone
import structlog
import hashlib
import json

from morag_core.database import Document, get_database_manager, get_session_context
from morag_core.database.models import DocumentState as DBDocumentState
from .models import (
    DocumentCreate, DocumentUpdate, DocumentResponse, DocumentSearchRequest,
    DocumentSearchResponse, DocumentStatsResponse, DocumentBatchOperation, DocumentBatchResponse,
    DocumentStatus, DocumentType
)
from morag_core.exceptions import NotFoundError, ValidationError, DatabaseError

logger = structlog.get_logger(__name__)


# Mapping between new DocumentStatus and existing DocumentState
STATUS_MAPPING = {
    DocumentStatus.PENDING: DBDocumentState.PENDING,
    DocumentStatus.PROCESSING: DBDocumentState.INGESTING,
    DocumentStatus.COMPLETED: DBDocumentState.INGESTED,
    DocumentStatus.FAILED: DBDocumentState.DEPRECATED,  # Use DEPRECATED for failed
    DocumentStatus.ARCHIVED: DBDocumentState.DELETED,   # Use DELETED for archived
}

# Reverse mapping
STATE_TO_STATUS = {v: k for k, v in STATUS_MAPPING.items()}


class DocumentService:
    """Document management service."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.db_manager = get_database_manager(database_url)
    
    def create_document(self, document_data: DocumentCreate, user_id: str) -> DocumentResponse:
        """Create a new document record."""
        try:
            with get_session_context(self.db_manager) as session:
                # Generate content checksum if source content is available
                content_checksum = None
                if document_data.source_path:
                    try:
                        from pathlib import Path
                        if Path(document_data.source_path).exists():
                            with open(document_data.source_path, 'rb') as f:
                                content_checksum = hashlib.sha256(f.read()).hexdigest()
                    except Exception as e:
                        logger.warning("Failed to generate content checksum", 
                                     source_path=document_data.source_path, error=str(e))
                
                # Create document using existing schema
                document = Document(
                    name=document_data.title,  # Use 'name' field from existing schema
                    type=document_data.content_type.value if document_data.content_type else "UNKNOWN",  # Use 'type' string field
                    state=DBDocumentState.PENDING,  # Use 'state' field with DocumentState enum
                    user_id=user_id,
                    database_id=document_data.database_id
                )
                
                session.add(document)
                session.flush()  # Get document ID
                
                logger.info("Document created",
                           document_id=document.id,
                           title=document.name,  # Use 'name' field
                           user_id=user_id)
                
                return self._document_to_response(document)
                
        except Exception as e:
            logger.error("Failed to create document", error=str(e))
            raise DatabaseError(f"Failed to create document: {str(e)}")
    
    def get_document(self, document_id: str, user_id: Optional[str] = None) -> Optional[DocumentResponse]:
        """Get document by ID."""
        try:
            with get_session_context(self.db_manager) as session:
                query = session.query(Document).filter_by(id=document_id)
                
                # Add user filter if provided (for non-admin users)
                if user_id:
                    query = query.filter_by(user_id=user_id)
                
                document = query.first()
                if document:
                    return self._document_to_response(document)
                return None
                
        except Exception as e:
            logger.error("Failed to get document", document_id=document_id, error=str(e))
            return None
    
    def update_document(self, document_id: str, document_data: DocumentUpdate, user_id: Optional[str] = None) -> DocumentResponse:
        """Update document information."""
        try:
            with get_session_context(self.db_manager) as session:
                query = session.query(Document).filter_by(id=document_id)
                
                # Add user filter if provided (for non-admin users)
                if user_id:
                    query = query.filter_by(user_id=user_id)
                
                document = query.first()
                if not document:
                    raise NotFoundError(f"Document {document_id} not found")
                
                # Update fields using existing schema
                if document_data.title is not None:
                    document.name = document_data.title  # Use 'name' field
                if document_data.status is not None:
                    document.state = STATUS_MAPPING.get(document_data.status, DBDocumentState.PENDING)
                # Note: Other fields like metadata, content_checksum, processing_error
                # are not in the existing schema, so we skip them for now
                
                logger.info("Document updated", document_id=document_id)
                return self._document_to_response(document)
                
        except NotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update document", document_id=document_id, error=str(e))
            raise DatabaseError(f"Failed to update document: {str(e)}")
    
    def delete_document(self, document_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a document."""
        try:
            with get_session_context(self.db_manager) as session:
                query = session.query(Document).filter_by(id=document_id)
                
                # Add user filter if provided (for non-admin users)
                if user_id:
                    query = query.filter_by(user_id=user_id)
                
                document = query.first()
                if not document:
                    return False
                
                session.delete(document)
                logger.info("Document deleted", document_id=document_id)
                return True
                
        except Exception as e:
            logger.error("Failed to delete document", document_id=document_id, error=str(e))
            return False
    
    def search_documents(self, search_request: DocumentSearchRequest, user_id: Optional[str] = None) -> DocumentSearchResponse:
        """Search documents with filters."""
        try:
            with get_session_context(self.db_manager) as session:
                query = session.query(Document)
                
                # Add user filter if provided (for non-admin users)
                if user_id:
                    query = query.filter_by(user_id=user_id)
                elif search_request.user_id:
                    # Admin can search by specific user
                    query = query.filter_by(user_id=search_request.user_id)
                
                # Apply filters using existing schema
                if search_request.query:
                    # Search in name field (existing schema)
                    search_term = f"%{search_request.query}%"
                    query = query.filter(Document.name.ilike(search_term))

                if search_request.status:
                    db_state = STATUS_MAPPING.get(search_request.status, DBDocumentState.PENDING)
                    query = query.filter_by(state=db_state)

                if search_request.content_type:
                    query = query.filter_by(type=search_request.content_type.value)
                
                if search_request.database_id:
                    query = query.filter_by(database_id=search_request.database_id)
                
                if search_request.created_after:
                    query = query.filter(Document.created_at >= search_request.created_after)
                
                if search_request.created_before:
                    query = query.filter(Document.created_at <= search_request.created_before)
                
                # Get total count before pagination
                total_count = query.count()
                
                # Apply sorting
                sort_column = getattr(Document, search_request.sort_by, Document.created_at)
                if search_request.sort_order == "desc":
                    query = query.order_by(desc(sort_column))
                else:
                    query = query.order_by(asc(sort_column))
                
                # Apply pagination
                documents = query.offset(search_request.offset).limit(search_request.limit).all()
                
                return DocumentSearchResponse(
                    documents=[self._document_to_response(doc) for doc in documents],
                    total_count=total_count,
                    limit=search_request.limit,
                    offset=search_request.offset,
                    has_more=search_request.offset + len(documents) < total_count
                )
                
        except Exception as e:
            logger.error("Failed to search documents", error=str(e))
            raise DatabaseError(f"Failed to search documents: {str(e)}")
    
    def get_document_stats(self, user_id: Optional[str] = None) -> DocumentStatsResponse:
        """Get document statistics."""
        try:
            with get_session_context(self.db_manager) as session:
                query = session.query(Document)
                
                # Add user filter if provided
                if user_id:
                    query = query.filter_by(user_id=user_id)
                
                documents = query.all()
                
                # Calculate statistics
                total_documents = len(documents)
                by_status = {}
                by_content_type = {}
                by_database = {}
                total_chunks = 0
                processing_errors = 0
                
                for doc in documents:
                    # Status stats using existing schema
                    status_key = doc.state.value
                    by_status[status_key] = by_status.get(status_key, 0) + 1

                    # Content type stats using existing schema
                    type_key = doc.type
                    by_content_type[type_key] = by_content_type.get(type_key, 0) + 1

                    # Database stats
                    db_key = doc.database_id or "default"
                    by_database[db_key] = by_database.get(db_key, 0) + 1

                    # Chunk count from existing chunks field
                    total_chunks += doc.chunks

                    # Error count (not available in existing schema)
                    # processing_errors would need to be tracked separately
                
                return DocumentStatsResponse(
                    total_documents=total_documents,
                    by_status=by_status,
                    by_content_type=by_content_type,
                    by_database=by_database,
                    total_chunks=total_chunks,
                    total_size_bytes=0,  # Would need file size tracking
                    processing_errors=processing_errors
                )
                
        except Exception as e:
            logger.error("Failed to get document stats", error=str(e))
            raise DatabaseError(f"Failed to get document stats: {str(e)}")
    
    def batch_operation(self, operation: DocumentBatchOperation, user_id: Optional[str] = None) -> DocumentBatchResponse:
        """Perform batch operations on documents."""
        try:
            with get_session_context(self.db_manager) as session:
                success_count = 0
                failure_count = 0
                errors = []
                
                for document_id in operation.document_ids:
                    try:
                        query = session.query(Document).filter_by(id=document_id)
                        
                        # Add user filter if provided
                        if user_id:
                            query = query.filter_by(user_id=user_id)
                        
                        document = query.first()
                        if not document:
                            errors.append({
                                "document_id": document_id,
                                "error": "Document not found"
                            })
                            failure_count += 1
                            continue
                        
                        if operation.operation == "delete":
                            session.delete(document)
                        elif operation.operation == "archive":
                            document.state = STATUS_MAPPING[DocumentStatus.ARCHIVED]
                        elif operation.operation == "reprocess":
                            document.state = STATUS_MAPPING[DocumentStatus.PENDING]
                        
                        success_count += 1
                        
                    except Exception as e:
                        errors.append({
                            "document_id": document_id,
                            "error": str(e)
                        })
                        failure_count += 1
                
                logger.info("Batch operation completed",
                           operation=operation.operation,
                           success_count=success_count,
                           failure_count=failure_count)
                
                return DocumentBatchResponse(
                    success_count=success_count,
                    failure_count=failure_count,
                    errors=errors,
                    operation=operation.operation
                )
                
        except Exception as e:
            logger.error("Failed to perform batch operation", error=str(e))
            raise DatabaseError(f"Failed to perform batch operation: {str(e)}")
    
    def _document_to_response(self, document: Document) -> DocumentResponse:
        """Convert Document model to DocumentResponse."""
        # Map existing schema fields to new response format
        # Use chunks field for chunk_count, quality for processing info

        return DocumentResponse(
            id=document.id,
            title=document.name,  # Use 'name' field from existing schema
            source_path=None,  # Not in existing schema
            source_url=None,   # Not in existing schema
            content_type=DocumentType(document.type) if document.type in [t.value for t in DocumentType] else DocumentType.UNKNOWN,
            status=STATE_TO_STATUS.get(document.state, DocumentStatus.PENDING),
            content_checksum=None,  # Not in existing schema
            metadata={},  # Not in existing schema
            processing_error=None,  # Not in existing schema
            user_id=document.user_id,
            database_id=document.database_id,
            created_at=document.created_at,
            updated_at=document.updated_at,
            processed_at=None,  # Not in existing schema
            vector_point_ids=None,  # Would need to be stored separately
            collection_name=None,   # Would need to be stored separately
            chunk_count=document.chunks  # Use existing chunks field
        )
