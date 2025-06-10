# Task 1: API Extensions for Remote Job Management

## Overview

Extend the MoRAG REST API with new endpoints for remote job management. These endpoints will handle the creation, polling, status checking, and result submission for remote conversion jobs.

## Objectives

1. Add remote job management endpoints to the FastAPI server
2. Implement request/response models for remote job operations
3. Add database models for remote job tracking
4. Integrate with existing authentication and validation systems
5. Ensure proper error handling and logging

## Technical Requirements

### 1. Database Models

Create new SQLAlchemy models for remote job tracking:

**File**: `packages/morag-core/src/morag_core/models/remote_job.py`

```python
from sqlalchemy import Column, String, DateTime, Integer, Text, Boolean
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

class RemoteJob(Base):
    __tablename__ = 'remote_jobs'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    ingestion_task_id = Column(String(255), nullable=False, index=True)
    source_file_path = Column(Text, nullable=False)
    content_type = Column(String(50), nullable=False, index=True)
    task_options = Column(JSONB, nullable=False, default={})
    status = Column(String(20), nullable=False, default='pending', index=True)
    worker_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    result_data = Column(JSONB, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    max_retries = Column(Integer, nullable=False, default=3)
    timeout_at = Column(DateTime, nullable=True)
    
    def to_dict(self):
        return {
            'job_id': str(self.id),
            'ingestion_task_id': self.ingestion_task_id,
            'source_file_path': self.source_file_path,
            'content_type': self.content_type,
            'task_options': self.task_options,
            'status': self.status,
            'worker_id': self.worker_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }
```

### 2. Pydantic Request/Response Models

**File**: `packages/morag/src/morag/models/remote_job_api.py`

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class CreateRemoteJobRequest(BaseModel):
    source_file_path: str = Field(..., description="Path to source file for processing")
    content_type: str = Field(..., description="Content type (audio, video, etc.)")
    task_options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")
    ingestion_task_id: str = Field(..., description="Associated ingestion task ID")

class CreateRemoteJobResponse(BaseModel):
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    created_at: datetime = Field(..., description="Job creation timestamp")

class PollJobsRequest(BaseModel):
    worker_id: str = Field(..., description="Unique worker identifier")
    content_types: List[str] = Field(..., description="Supported content types")
    max_jobs: int = Field(default=1, description="Maximum jobs to return")

class PollJobsResponse(BaseModel):
    job_id: Optional[str] = Field(None, description="Job ID if available")
    source_file_url: Optional[str] = Field(None, description="Download URL for source file")
    content_type: Optional[str] = Field(None, description="Content type")
    task_options: Optional[Dict[str, Any]] = Field(None, description="Processing options")

class SubmitResultRequest(BaseModel):
    success: bool = Field(..., description="Whether processing succeeded")
    content: Optional[str] = Field(None, description="Processed content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Processing metadata")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")

class SubmitResultResponse(BaseModel):
    status: str = Field(..., description="Updated job status")
    ingestion_continued: bool = Field(..., description="Whether ingestion pipeline continued")

class JobStatusResponse(BaseModel):
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current job status")
    worker_id: Optional[str] = Field(None, description="Assigned worker ID")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Processing start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(..., description="Number of retry attempts")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
```

### 3. Database Service Layer

**File**: `packages/morag/src/morag/services/remote_job_service.py`

```python
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import uuid

from morag_core.models.remote_job import RemoteJob
from morag.models.remote_job_api import CreateRemoteJobRequest, SubmitResultRequest

class RemoteJobService:
    def __init__(self, db_session: Session):
        self.db = db_session
    
    def create_job(self, request: CreateRemoteJobRequest) -> RemoteJob:
        """Create a new remote job."""
        job = RemoteJob(
            ingestion_task_id=request.ingestion_task_id,
            source_file_path=request.source_file_path,
            content_type=request.content_type,
            task_options=request.task_options,
            timeout_at=datetime.utcnow() + timedelta(hours=1)  # Default 1 hour timeout
        )
        
        self.db.add(job)
        self.db.commit()
        self.db.refresh(job)
        return job
    
    def poll_available_jobs(self, worker_id: str, content_types: List[str], max_jobs: int = 1) -> List[RemoteJob]:
        """Poll for available jobs matching worker capabilities."""
        jobs = self.db.query(RemoteJob).filter(
            and_(
                RemoteJob.status == 'pending',
                RemoteJob.content_type.in_(content_types),
                or_(
                    RemoteJob.timeout_at.is_(None),
                    RemoteJob.timeout_at > datetime.utcnow()
                )
            )
        ).order_by(RemoteJob.created_at).limit(max_jobs).all()
        
        # Claim jobs for this worker
        for job in jobs:
            job.status = 'processing'
            job.worker_id = worker_id
            job.started_at = datetime.utcnow()
        
        self.db.commit()
        return jobs
    
    def submit_result(self, job_id: str, result: SubmitResultRequest) -> Optional[RemoteJob]:
        """Submit processing result for a job."""
        job = self.db.query(RemoteJob).filter(RemoteJob.id == job_id).first()
        if not job:
            return None
        
        job.completed_at = datetime.utcnow()
        if result.success:
            job.status = 'completed'
            job.result_data = {
                'content': result.content,
                'metadata': result.metadata or {},
                'processing_time': result.processing_time
            }
        else:
            job.status = 'failed'
            job.error_message = result.error_message
            job.retry_count += 1
        
        self.db.commit()
        self.db.refresh(job)
        return job
    
    def get_job_status(self, job_id: str) -> Optional[RemoteJob]:
        """Get current status of a job."""
        return self.db.query(RemoteJob).filter(RemoteJob.id == job_id).first()
    
    def cleanup_expired_jobs(self) -> int:
        """Mark expired jobs as failed."""
        expired_jobs = self.db.query(RemoteJob).filter(
            and_(
                RemoteJob.status.in_(['pending', 'processing']),
                RemoteJob.timeout_at < datetime.utcnow()
            )
        ).all()
        
        for job in expired_jobs:
            job.status = 'timeout'
            job.error_message = 'Job exceeded maximum processing time'
        
        self.db.commit()
        return len(expired_jobs)
```

### 4. API Endpoints

**File**: `packages/morag/src/morag/api/remote_jobs.py`

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
import structlog

from morag.models.remote_job_api import (
    CreateRemoteJobRequest, CreateRemoteJobResponse,
    PollJobsRequest, PollJobsResponse,
    SubmitResultRequest, SubmitResultResponse,
    JobStatusResponse
)
from morag.services.remote_job_service import RemoteJobService
from morag.database import get_db_session

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/remote-jobs", tags=["Remote Jobs"])

@router.post("/", response_model=CreateRemoteJobResponse)
async def create_remote_job(
    request: CreateRemoteJobRequest,
    db: Session = Depends(get_db_session)
):
    """Create a new remote conversion job."""
    try:
        service = RemoteJobService(db)
        job = service.create_job(request)
        
        logger.info("Remote job created",
                   job_id=str(job.id),
                   content_type=job.content_type,
                   ingestion_task_id=job.ingestion_task_id)
        
        return CreateRemoteJobResponse(
            job_id=str(job.id),
            status=job.status,
            created_at=job.created_at
        )
    except Exception as e:
        logger.error("Failed to create remote job", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create remote job: {str(e)}"
        )

@router.get("/poll", response_model=PollJobsResponse)
async def poll_for_jobs(
    worker_id: str,
    content_types: str,  # Comma-separated list
    max_jobs: int = 1,
    db: Session = Depends(get_db_session)
):
    """Poll for available remote jobs."""
    try:
        content_type_list = [ct.strip() for ct in content_types.split(',')]
        service = RemoteJobService(db)
        jobs = service.poll_available_jobs(worker_id, content_type_list, max_jobs)
        
        if not jobs:
            return PollJobsResponse()
        
        job = jobs[0]  # Return first job for now
        
        # Generate secure download URL for source file
        source_file_url = f"/api/v1/remote-jobs/{job.id}/download"
        
        logger.info("Job assigned to worker",
                   job_id=str(job.id),
                   worker_id=worker_id,
                   content_type=job.content_type)
        
        return PollJobsResponse(
            job_id=str(job.id),
            source_file_url=source_file_url,
            content_type=job.content_type,
            task_options=job.task_options
        )
    except Exception as e:
        logger.error("Failed to poll for jobs", worker_id=worker_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to poll for jobs: {str(e)}"
        )

@router.put("/{job_id}/result", response_model=SubmitResultResponse)
async def submit_job_result(
    job_id: str,
    result: SubmitResultRequest,
    db: Session = Depends(get_db_session)
):
    """Submit processing result for a remote job."""
    try:
        service = RemoteJobService(db)
        job = service.submit_result(job_id, result)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        logger.info("Job result submitted",
                   job_id=job_id,
                   status=job.status,
                   success=result.success)
        
        # TODO: Continue ingestion pipeline if successful
        ingestion_continued = False
        if result.success:
            # Trigger continuation of ingestion task
            ingestion_continued = True
        
        return SubmitResultResponse(
            status=job.status,
            ingestion_continued=ingestion_continued
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to submit job result", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit job result: {str(e)}"
        )

@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db_session)
):
    """Get current status of a remote job."""
    try:
        service = RemoteJobService(db)
        job = service.get_job_status(job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )
        
        # Estimate completion time based on content type and processing time
        estimated_completion = None
        if job.status == 'processing' and job.started_at:
            if job.content_type == 'audio':
                estimated_duration = timedelta(minutes=5)  # Estimate 5 minutes for audio
            elif job.content_type == 'video':
                estimated_duration = timedelta(minutes=15)  # Estimate 15 minutes for video
            else:
                estimated_duration = timedelta(minutes=10)  # Default estimate
            
            estimated_completion = job.started_at + estimated_duration
        
        return JobStatusResponse(
            job_id=str(job.id),
            status=job.status,
            worker_id=job.worker_id,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            error_message=job.error_message,
            retry_count=job.retry_count,
            estimated_completion=estimated_completion
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get job status", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get job status: {str(e)}"
        )
```

## Implementation Steps

1. **Create Database Models** (Day 1)
   - Add `RemoteJob` model to morag-core
   - Create database migration script
   - Add model to exports

2. **Create API Models** (Day 1)
   - Add Pydantic request/response models
   - Include proper validation and documentation
   - Add to model exports

3. **Implement Service Layer** (Day 2)
   - Create `RemoteJobService` class
   - Implement all CRUD operations
   - Add proper error handling and logging

4. **Create API Endpoints** (Day 2-3)
   - Implement all remote job endpoints
   - Add proper authentication and validation
   - Include comprehensive error handling

5. **Integration with Main Server** (Day 3)
   - Add router to main FastAPI app
   - Configure database connection
   - Add to API documentation

6. **Testing** (Day 4)
   - Unit tests for service layer
   - Integration tests for API endpoints
   - Error scenario testing

## Testing Requirements

### Unit Tests

**File**: `tests/test_remote_job_service.py`

```python
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from morag.services.remote_job_service import RemoteJobService
from morag.models.remote_job_api import CreateRemoteJobRequest, SubmitResultRequest

class TestRemoteJobService:
    def test_create_job(self):
        # Test job creation
        pass
    
    def test_poll_available_jobs(self):
        # Test job polling
        pass
    
    def test_submit_result_success(self):
        # Test successful result submission
        pass
    
    def test_submit_result_failure(self):
        # Test failed result submission
        pass
    
    def test_cleanup_expired_jobs(self):
        # Test job timeout cleanup
        pass
```

### Integration Tests

**File**: `tests/test_remote_job_api.py`

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from morag.server import create_app

class TestRemoteJobAPI:
    def test_create_remote_job(self):
        # Test job creation endpoint
        pass
    
    def test_poll_for_jobs(self):
        # Test job polling endpoint
        pass
    
    def test_submit_job_result(self):
        # Test result submission endpoint
        pass
    
    def test_get_job_status(self):
        # Test status endpoint
        pass
```

## Success Criteria

1. All API endpoints return proper responses with correct status codes
2. Database operations handle concurrent access correctly
3. Error handling provides meaningful messages to clients
4. Authentication and validation work as expected
5. All tests pass with >95% coverage

## Dependencies

- SQLAlchemy for database operations
- FastAPI for API endpoints
- Pydantic for request/response validation
- PostgreSQL for job storage
- Existing MoRAG authentication system

## Next Steps

After completing this task:
1. Proceed to [Task 2: Database Schema](./02-database-schema.md)
2. Set up database migrations
3. Test API endpoints with mock data
4. Begin integration with existing ingestion system
