# Task 04: Document Lifecycle Management

## 📋 Task Overview

**Objective**: Implement comprehensive document lifecycle management including document tracking, state management, version control, and metadata management integrated with the existing MoRAG processing pipeline.

**Priority**: High - Core functionality for document management
**Estimated Time**: 1-2 weeks
**Dependencies**: Task 03 (Authentication Middleware)

## 🎯 Goals

1. Integrate document tracking with existing processing pipeline
2. Implement document state management (PENDING, INGESTING, INGESTED, DEPRECATED, DELETED)
3. Add document version control and replacement functionality
4. Create document metadata management and search
5. Implement document ownership and access control
6. Add document statistics and analytics
7. Create document management API endpoints

## 📊 Current State Analysis

### Existing Document Model (from database/DATABASE.md)
- **Fields**: ID, name, type, state, version, chunks, quality, upload_date, user_id, database_id
- **States**: PENDING, INGESTING, INGESTED, DEPRECATED, DELETED
- **Relationships**: User (owner), Database (storage location), Jobs (processing history)
- **Complete Schema**: See `database/DATABASE.md` for full entity definitions and business logic

### Current MoRAG Document Processing
- **Processing**: File upload → Processing → Vector storage
- **Tracking**: Basic task tracking via Celery
- **Metadata**: Limited metadata extraction
- **Storage**: Qdrant vector database only

## 🔧 Implementation Plan

### Step 1: Create Document Service Layer

**Files to Create**:
```
packages/morag-core/src/morag_core/
├── documents/
│   ├── __init__.py
│   ├── models.py          # Pydantic models for documents
│   ├── service.py         # Document service logic
│   ├── lifecycle.py       # Document lifecycle management
│   └── metadata.py        # Metadata extraction and management
```

**Implementation Details**:

1. **Document Models**:
```python
# packages/morag-core/src/morag_core/documents/models.py
"""Document management models."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class DocumentState(str, Enum):
    PENDING = "PENDING"
    INGESTING = "INGESTING"
    INGESTED = "INGESTED"
    DEPRECATED = "DEPRECATED"
    DELETED = "DELETED"

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    YOUTUBE = "youtube"
    WEB = "web"
    UNKNOWN = "unknown"

class DocumentCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    type: DocumentType
    database_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    source_path: Optional[str] = None
    source_url: Optional[str] = None

class DocumentUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    state: Optional[DocumentState] = None
    metadata: Optional[Dict[str, Any]] = None
    quality: Optional[float] = Field(None, ge=0.0, le=1.0)

class DocumentResponse(BaseModel):
    id: str
    name: str
    type: DocumentType
    state: DocumentState
    version: int
    chunks: int
    quality: float
    upload_date: datetime
    created_at: datetime
    updated_at: datetime
    user_id: str
    database_id: Optional[str]
    metadata: Optional[Dict[str, Any]]

class DocumentSearchRequest(BaseModel):
    query: Optional[str] = None
    document_type: Optional[DocumentType] = None
    state: Optional[DocumentState] = None
    database_id: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=50, le=1000)
    offset: int = Field(default=0, ge=0)

class DocumentStatistics(BaseModel):
    total_documents: int
    documents_by_type: Dict[str, int]
    documents_by_state: Dict[str, int]
    total_chunks: int
    average_quality: float
    storage_size_mb: float
```

2. **Document Service**:
```python
# packages/morag-core/src/morag_core/documents/service.py
"""Document management service."""

from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
import structlog
from datetime import datetime

from morag_core.database import (
    Document, get_database_manager, DocumentState as DBDocumentState
)
from .models import (
    DocumentCreate, DocumentUpdate, DocumentResponse, 
    DocumentSearchRequest, DocumentStatistics, DocumentState, DocumentType
)
from morag_core.exceptions import NotFoundError, ValidationError, ConflictError

logger = structlog.get_logger(__name__)

class DocumentService:
    """Document management service."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
    
    def create_document(self, user_id: str, document_data: DocumentCreate) -> DocumentResponse:
        """Create a new document record."""
        with self.db_manager.get_session() as session:
            # Check for duplicate names in same database
            existing = session.query(Document).filter(
                and_(
                    Document.user_id == user_id,
                    Document.name == document_data.name,
                    Document.database_id == document_data.database_id,
                    Document.state != DBDocumentState.DELETED
                )
            ).first()
            
            if existing:
                raise ConflictError(f"Document '{document_data.name}' already exists in this database")
            
            # Create document
            document = Document(
                name=document_data.name,
                type=document_data.type.value,
                state=DBDocumentState.PENDING,
                user_id=user_id,
                database_id=document_data.database_id,
                metadata=document_data.metadata or {}
            )
            
            session.add(document)
            session.flush()
            
            logger.info("Document created", 
                       document_id=document.id, 
                       name=document.name,
                       user_id=user_id)
            
            return self._document_to_response(document)
    
    def get_document(self, document_id: str, user_id: str) -> Optional[DocumentResponse]:
        """Get document by ID with user ownership check."""
        with self.db_manager.get_session() as session:
            document = session.query(Document).filter(
                and_(
                    Document.id == document_id,
                    Document.user_id == user_id,
                    Document.state != DBDocumentState.DELETED
                )
            ).first()
            
            if document:
                return self._document_to_response(document)
            return None
    
    def update_document(self, document_id: str, user_id: str, 
                       document_data: DocumentUpdate) -> DocumentResponse:
        """Update document information."""
        with self.db_manager.get_session() as session:
            document = session.query(Document).filter(
                and_(
                    Document.id == document_id,
                    Document.user_id == user_id,
                    Document.state != DBDocumentState.DELETED
                )
            ).first()
            
            if not document:
                raise NotFoundError(f"Document {document_id} not found")
            
            # Update fields
            if document_data.name is not None:
                # Check for name conflicts
                existing = session.query(Document).filter(
                    and_(
                        Document.user_id == user_id,
                        Document.name == document_data.name,
                        Document.database_id == document.database_id,
                        Document.id != document_id,
                        Document.state != DBDocumentState.DELETED
                    )
                ).first()
                
                if existing:
                    raise ConflictError(f"Document name '{document_data.name}' already exists")
                
                document.name = document_data.name
            
            if document_data.state is not None:
                document.state = DBDocumentState(document_data.state.value)
            
            if document_data.metadata is not None:
                document.metadata = document_data.metadata
            
            if document_data.quality is not None:
                document.quality = document_data.quality
            
            logger.info("Document updated", 
                       document_id=document_id,
                       user_id=user_id)
            
            return self._document_to_response(document)
    
    def search_documents(self, user_id: str, 
                        search_request: DocumentSearchRequest) -> Tuple[List[DocumentResponse], int]:
        """Search user's documents with filters."""
        with self.db_manager.get_session() as session:
            query = session.query(Document).filter(
                and_(
                    Document.user_id == user_id,
                    Document.state != DBDocumentState.DELETED
                )
            )
            
            # Apply filters
            if search_request.query:
                query = query.filter(
                    or_(
                        Document.name.ilike(f"%{search_request.query}%"),
                        Document.metadata.op('->>')('title').ilike(f"%{search_request.query}%")
                    )
                )
            
            if search_request.document_type:
                query = query.filter(Document.type == search_request.document_type.value)
            
            if search_request.state:
                query = query.filter(Document.state == DBDocumentState(search_request.state.value))
            
            if search_request.database_id:
                query = query.filter(Document.database_id == search_request.database_id)
            
            if search_request.date_from:
                query = query.filter(Document.upload_date >= search_request.date_from)
            
            if search_request.date_to:
                query = query.filter(Document.upload_date <= search_request.date_to)
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination and ordering
            documents = query.order_by(desc(Document.upload_date))\
                           .offset(search_request.offset)\
                           .limit(search_request.limit)\
                           .all()
            
            return [self._document_to_response(doc) for doc in documents], total_count
    
    def delete_document(self, document_id: str, user_id: str, soft_delete: bool = True) -> bool:
        """Delete document (soft delete by default)."""
        with self.db_manager.get_session() as session:
            document = session.query(Document).filter(
                and_(
                    Document.id == document_id,
                    Document.user_id == user_id
                )
            ).first()
            
            if not document:
                return False
            
            if soft_delete:
                document.state = DBDocumentState.DELETED
                logger.info("Document soft deleted", 
                           document_id=document_id,
                           user_id=user_id)
            else:
                session.delete(document)
                logger.info("Document hard deleted", 
                           document_id=document_id,
                           user_id=user_id)
            
            return True
    
    def get_document_statistics(self, user_id: str, database_id: Optional[str] = None) -> DocumentStatistics:
        """Get document statistics for user."""
        with self.db_manager.get_session() as session:
            query = session.query(Document).filter(
                and_(
                    Document.user_id == user_id,
                    Document.state != DBDocumentState.DELETED
                )
            )
            
            if database_id:
                query = query.filter(Document.database_id == database_id)
            
            documents = query.all()
            
            # Calculate statistics
            total_documents = len(documents)
            documents_by_type = {}
            documents_by_state = {}
            total_chunks = 0
            total_quality = 0.0
            
            for doc in documents:
                # Count by type
                doc_type = doc.type
                documents_by_type[doc_type] = documents_by_type.get(doc_type, 0) + 1
                
                # Count by state
                doc_state = doc.state.value
                documents_by_state[doc_state] = documents_by_state.get(doc_state, 0) + 1
                
                # Sum chunks and quality
                total_chunks += doc.chunks
                total_quality += doc.quality
            
            average_quality = total_quality / total_documents if total_documents > 0 else 0.0
            
            return DocumentStatistics(
                total_documents=total_documents,
                documents_by_type=documents_by_type,
                documents_by_state=documents_by_state,
                total_chunks=total_chunks,
                average_quality=average_quality,
                storage_size_mb=0.0  # TODO: Calculate from vector storage
            )
    
    def _document_to_response(self, document: Document) -> DocumentResponse:
        """Convert Document model to DocumentResponse."""
        return DocumentResponse(
            id=document.id,
            name=document.name,
            type=DocumentType(document.type),
            state=DocumentState(document.state.value),
            version=document.version,
            chunks=document.chunks,
            quality=document.quality,
            upload_date=document.upload_date,
            created_at=document.created_at,
            updated_at=document.updated_at,
            user_id=document.user_id,
            database_id=document.database_id,
            metadata=document.metadata
        )
```

### Step 2: Create Document Lifecycle Manager

**File to Create**: `packages/morag-core/src/morag_core/documents/lifecycle.py`

```python
"""Document lifecycle management."""

import structlog
from typing import Optional, Dict, Any
from datetime import datetime

from morag_core.database import (
    Document, Job, get_database_manager, 
    DocumentState as DBDocumentState, JobStatus
)
from .service import DocumentService

logger = structlog.get_logger(__name__)

class DocumentLifecycleManager:
    """Manage document lifecycle states and transitions."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
        self.document_service = DocumentService()
    
    def start_ingestion(self, document_id: str, job_id: str) -> bool:
        """Mark document as starting ingestion process."""
        with self.db_manager.get_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if not document:
                logger.error("Document not found for ingestion start", document_id=document_id)
                return False
            
            # Update document state
            document.state = DBDocumentState.INGESTING
            
            # Create job record
            job = Job(
                id=job_id,
                document_id=document_id,
                user_id=document.user_id,
                document_name=document.name,
                document_type=document.type,
                status=JobStatus.PROCESSING,
                percentage=0
            )
            session.add(job)
            
            logger.info("Document ingestion started", 
                       document_id=document_id, 
                       job_id=job_id)
            return True
    
    def complete_ingestion(self, document_id: str, job_id: str, 
                          chunks_count: int, quality_score: float,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Mark document ingestion as completed."""
        with self.db_manager.get_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            job = session.query(Job).filter_by(id=job_id).first()
            
            if not document or not job:
                logger.error("Document or job not found for ingestion completion", 
                           document_id=document_id, job_id=job_id)
                return False
            
            # Update document
            document.state = DBDocumentState.INGESTED
            document.chunks = chunks_count
            document.quality = quality_score
            if metadata:
                document.metadata.update(metadata)
            
            # Update job
            job.status = JobStatus.FINISHED
            job.percentage = 100
            job.end_date = datetime.utcnow()
            job.summary = f"Successfully processed {chunks_count} chunks with quality score {quality_score:.2f}"
            
            logger.info("Document ingestion completed", 
                       document_id=document_id, 
                       job_id=job_id,
                       chunks=chunks_count,
                       quality=quality_score)
            return True
    
    def fail_ingestion(self, document_id: str, job_id: str, error_message: str) -> bool:
        """Mark document ingestion as failed."""
        with self.db_manager.get_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            job = session.query(Job).filter_by(id=job_id).first()
            
            if not document or not job:
                logger.error("Document or job not found for ingestion failure", 
                           document_id=document_id, job_id=job_id)
                return False
            
            # Keep document in PENDING state for retry
            document.state = DBDocumentState.PENDING
            
            # Update job
            job.status = JobStatus.FAILED
            job.end_date = datetime.utcnow()
            job.summary = f"Ingestion failed: {error_message}"
            
            logger.error("Document ingestion failed", 
                        document_id=document_id, 
                        job_id=job_id,
                        error=error_message)
            return True
    
    def deprecate_document(self, document_id: str, reason: str = "") -> bool:
        """Mark document as deprecated."""
        with self.db_manager.get_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if not document:
                return False
            
            document.state = DBDocumentState.DEPRECATED
            if reason:
                if not document.metadata:
                    document.metadata = {}
                document.metadata['deprecation_reason'] = reason
                document.metadata['deprecated_at'] = datetime.utcnow().isoformat()
            
            logger.info("Document deprecated", 
                       document_id=document_id, 
                       reason=reason)
            return True
    
    def restore_document(self, document_id: str) -> bool:
        """Restore deprecated document."""
        with self.db_manager.get_session() as session:
            document = session.query(Document).filter_by(id=document_id).first()
            if not document:
                return False
            
            if document.state == DBDocumentState.DEPRECATED:
                document.state = DBDocumentState.INGESTED
                if document.metadata and 'deprecation_reason' in document.metadata:
                    del document.metadata['deprecation_reason']
                    del document.metadata['deprecated_at']
                
                logger.info("Document restored", document_id=document_id)
                return True
            
            return False
```

## 🧪 Testing Requirements

### Unit Tests
```python
# tests/test_document_lifecycle.py
import pytest
from morag_core.documents import DocumentService, DocumentLifecycleManager
from morag_core.documents.models import DocumentCreate, DocumentType, DocumentState

def test_document_creation():
    """Test document creation."""
    service = DocumentService()
    document_data = DocumentCreate(
        name="Test Document",
        type=DocumentType.PDF,
        metadata={"source": "test"}
    )
    
    document = service.create_document("user123", document_data)
    assert document.name == "Test Document"
    assert document.type == DocumentType.PDF
    assert document.state == DocumentState.PENDING

def test_document_lifecycle():
    """Test complete document lifecycle."""
    service = DocumentService()
    lifecycle = DocumentLifecycleManager()
    
    # Create document
    document_data = DocumentCreate(name="Lifecycle Test", type=DocumentType.PDF)
    document = service.create_document("user123", document_data)
    
    # Start ingestion
    assert lifecycle.start_ingestion(document.id, "job123")
    updated_doc = service.get_document(document.id, "user123")
    assert updated_doc.state == DocumentState.INGESTING
    
    # Complete ingestion
    assert lifecycle.complete_ingestion(document.id, "job123", 10, 0.85)
    final_doc = service.get_document(document.id, "user123")
    assert final_doc.state == DocumentState.INGESTED
    assert final_doc.chunks == 10
    assert final_doc.quality == 0.85
```

## 📋 Acceptance Criteria

- [ ] Document service with CRUD operations implemented
- [ ] Document lifecycle management working
- [ ] Document state transitions properly managed
- [ ] Document search and filtering functional
- [ ] Document statistics and analytics available
- [ ] Integration with existing processing pipeline
- [ ] User ownership and access control enforced
- [ ] Comprehensive unit tests passing
- [ ] API endpoints for document management created
- [ ] Error handling for document operations

## 🔄 Next Steps

After completing this task:
1. Proceed to [Task 05: Job Tracking Integration](./task-05-job-tracking-integration.md)
2. Integrate document tracking with existing ingestion tasks
3. Add document management UI components
4. Test document lifecycle with real processing jobs

## 📝 Notes

- Ensure proper integration with existing Celery task system
- Add comprehensive logging for document state changes
- Consider implementing document versioning for updates
- Add document backup and recovery procedures
- Implement document access permissions for shared documents
