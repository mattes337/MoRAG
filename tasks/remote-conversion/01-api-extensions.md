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

### 1. Data Models

Create data models for remote job tracking using file-based storage:

**File**: `packages/morag-core/src/morag_core/models/remote_job.py`

```python
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from datetime import datetime
import uuid
import json

@dataclass
class RemoteJob:
    """Remote conversion job data model."""

    id: str
    ingestion_task_id: str
    source_file_path: str
    content_type: str
    task_options: Dict[str, Any]
    status: str = 'pending'
    worker_id: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_at: Optional[datetime] = None

    def __post_init__(self):
        """Initialize default values after creation."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()

    @classmethod
    def create_new(cls, ingestion_task_id: str, source_file_path: str,
                   content_type: str, task_options: Dict[str, Any]) -> 'RemoteJob':
        """Create a new remote job with generated ID."""
        return cls(
            id=str(uuid.uuid4()),
            ingestion_task_id=ingestion_task_id,
            source_file_path=source_file_path,
            content_type=content_type,
            task_options=task_options,
            created_at=datetime.utcnow()
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for field in ['created_at', 'started_at', 'completed_at', 'timeout_at']:
            if data[field] is not None:
                data[field] = data[field].isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RemoteJob':
        """Create instance from dictionary (JSON deserialization)."""
        # Convert ISO strings back to datetime objects
        for field in ['created_at', 'started_at', 'completed_at', 'timeout_at']:
            if data.get(field) is not None:
                data[field] = datetime.fromisoformat(data[field])
        return cls(**data)

    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.retry_count < self.max_retries and self.status in ['failed', 'timeout']

    @property
    def is_expired(self) -> bool:
        """Check if job has expired."""
        if not self.timeout_at:
            return False
        return datetime.utcnow() > self.timeout_at

    @property
    def processing_duration(self) -> float:
        """Get processing duration in seconds."""
        if not self.started_at:
            return 0.0

        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.started_at).total_seconds()
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

### 3. Repository Layer

**File**: `packages/morag/src/morag/repositories/remote_job_repository.py`

```python
import os
import json
import glob
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import structlog

from morag_core.models.remote_job import RemoteJob

logger = structlog.get_logger(__name__)

class RemoteJobRepository:
    """File-based repository for remote job storage."""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir or os.getenv('MORAG_REMOTE_JOBS_DATA_DIR', '/app/data/remote_jobs'))
        self._ensure_directories()

    def _ensure_directories(self):
        """Ensure all required directories exist."""
        status_dirs = ['pending', 'processing', 'completed', 'failed', 'timeout', 'cancelled']
        for status in status_dirs:
            (self.data_dir / status).mkdir(parents=True, exist_ok=True)

    def _get_job_file_path(self, job_id: str, status: str) -> Path:
        """Get the file path for a job based on its status."""
        return self.data_dir / status / f"{job_id}.json"

    def _find_job_file(self, job_id: str) -> Optional[Path]:
        """Find the job file across all status directories."""
        for status_dir in self.data_dir.iterdir():
            if status_dir.is_dir():
                job_file = status_dir / f"{job_id}.json"
                if job_file.exists():
                    return job_file
        return None

    def create_job(self, ingestion_task_id: str, source_file_path: str,
                   content_type: str, task_options: Dict[str, Any]) -> RemoteJob:
        """Create a new remote job."""
        try:
            job = RemoteJob.create_new(
                ingestion_task_id=ingestion_task_id,
                source_file_path=source_file_path,
                content_type=content_type,
                task_options=task_options
            )

            # Set timeout based on content type
            timeout_hours = {'audio': 0.5, 'video': 1, 'document': 0.25}.get(content_type, 1)
            job.timeout_at = datetime.utcnow() + timedelta(hours=timeout_hours)

            # Save to pending directory
            job_file = self._get_job_file_path(job.id, 'pending')
            with open(job_file, 'w') as f:
                json.dump(job.to_dict(), f, indent=2)

            logger.info("Remote job created", job_id=job.id, content_type=content_type)
            return job

        except Exception as e:
            logger.error("Failed to create remote job", error=str(e))
            raise

    def get_job(self, job_id: str) -> Optional[RemoteJob]:
        """Get a job by ID."""
        try:
            job_file = self._find_job_file(job_id)
            if not job_file:
                return None

            with open(job_file, 'r') as f:
                data = json.load(f)

            return RemoteJob.from_dict(data)

        except Exception as e:
            logger.error("Failed to get job", job_id=job_id, error=str(e))
            return None

    def update_job(self, job: RemoteJob) -> bool:
        """Update a job (may involve moving between status directories)."""
        try:
            # Find current job file
            old_file = self._find_job_file(job.id)
            if not old_file:
                logger.error("Job file not found for update", job_id=job.id)
                return False

            # Determine new file path based on status
            new_file = self._get_job_file_path(job.id, job.status)

            # Save updated job data
            with open(new_file, 'w') as f:
                json.dump(job.to_dict(), f, indent=2)

            # Remove old file if it's in a different directory
            if old_file != new_file:
                old_file.unlink()

            logger.debug("Job updated", job_id=job.id, status=job.status)
            return True

        except Exception as e:
            logger.error("Failed to update job", job_id=job.id, error=str(e))
            return False

    def poll_available_jobs(self, worker_id: str, content_types: List[str], max_jobs: int = 1) -> List[RemoteJob]:
        """Poll for available jobs matching worker capabilities."""
        try:
            available_jobs = []
            pending_dir = self.data_dir / 'pending'

            # Get all pending job files
            job_files = list(pending_dir.glob('*.json'))
            job_files.sort(key=lambda f: f.stat().st_mtime)  # Sort by creation time

            for job_file in job_files:
                if len(available_jobs) >= max_jobs:
                    break

                try:
                    with open(job_file, 'r') as f:
                        data = json.load(f)

                    job = RemoteJob.from_dict(data)

                    # Check if job matches criteria
                    if (job.content_type in content_types and
                        not job.is_expired):

                        # Claim the job
                        job.status = 'processing'
                        job.worker_id = worker_id
                        job.started_at = datetime.utcnow()

                        # Update the job (moves to processing directory)
                        if self.update_job(job):
                            available_jobs.append(job)

                except Exception as e:
                    logger.error("Failed to process job file", file=str(job_file), error=str(e))
                    continue

            logger.info("Jobs polled", worker_id=worker_id, count=len(available_jobs))
            return available_jobs

        except Exception as e:
            logger.error("Failed to poll jobs", worker_id=worker_id, error=str(e))
            return []

    def submit_result(self, job_id: str, success: bool, content: str = None,
                     metadata: Dict[str, Any] = None, error_message: str = None,
                     processing_time: float = None) -> Optional[RemoteJob]:
        """Submit processing result for a job."""
        try:
            job = self.get_job(job_id)
            if not job:
                logger.error("Job not found for result submission", job_id=job_id)
                return None

            job.completed_at = datetime.utcnow()

            if success:
                job.status = 'completed'
                job.result_data = {
                    'content': content or '',
                    'metadata': metadata or {},
                    'processing_time': processing_time or 0.0
                }
            else:
                job.status = 'failed'
                job.error_message = error_message
                job.retry_count += 1

            # Update the job (moves to appropriate status directory)
            if self.update_job(job):
                logger.info("Job result submitted", job_id=job_id, success=success)
                return job
            else:
                logger.error("Failed to update job after result submission", job_id=job_id)
                return None

        except Exception as e:
            logger.error("Failed to submit job result", job_id=job_id, error=str(e))
            return None

    def find_jobs_by_status(self, status: str) -> List[RemoteJob]:
        """Find all jobs with a specific status."""
        try:
            jobs = []
            status_dir = self.data_dir / status

            if not status_dir.exists():
                return jobs

            for job_file in status_dir.glob('*.json'):
                try:
                    with open(job_file, 'r') as f:
                        data = json.load(f)
                    jobs.append(RemoteJob.from_dict(data))
                except Exception as e:
                    logger.error("Failed to load job file", file=str(job_file), error=str(e))

            return jobs

        except Exception as e:
            logger.error("Failed to find jobs by status", status=status, error=str(e))
            return []

    def find_jobs_by_content_type(self, content_type: str) -> List[RemoteJob]:
        """Find all jobs with a specific content type."""
        try:
            jobs = []

            # Search across all status directories
            for status_dir in self.data_dir.iterdir():
                if not status_dir.is_dir():
                    continue

                for job_file in status_dir.glob('*.json'):
                    try:
                        with open(job_file, 'r') as f:
                            data = json.load(f)

                        if data.get('content_type') == content_type:
                            jobs.append(RemoteJob.from_dict(data))

                    except Exception as e:
                        logger.error("Failed to load job file", file=str(job_file), error=str(e))

            return jobs

        except Exception as e:
            logger.error("Failed to find jobs by content type", content_type=content_type, error=str(e))
            return []

    def cleanup_expired_jobs(self) -> int:
        """Mark expired jobs as timed out."""
        try:
            expired_count = 0
            now = datetime.utcnow()

            # Check pending and processing directories for expired jobs
            for status in ['pending', 'processing']:
                status_dir = self.data_dir / status
                if not status_dir.exists():
                    continue

                for job_file in status_dir.glob('*.json'):
                    try:
                        with open(job_file, 'r') as f:
                            data = json.load(f)

                        job = RemoteJob.from_dict(data)

                        if job.is_expired:
                            job.status = 'timeout'
                            job.error_message = 'Job exceeded maximum processing time'
                            job.completed_at = now

                            if self.update_job(job):
                                expired_count += 1

                    except Exception as e:
                        logger.error("Failed to process expired job", file=str(job_file), error=str(e))

            if expired_count > 0:
                logger.info("Expired jobs cleaned up", count=expired_count)

            return expired_count

        except Exception as e:
            logger.error("Failed to cleanup expired jobs", error=str(e))
            return 0

    def cleanup_old_jobs(self, days_old: int = 7) -> int:
        """Remove old completed/failed jobs."""
        try:
            cleaned_count = 0
            cutoff_time = datetime.utcnow() - timedelta(days=days_old)

            # Clean up completed, failed, timeout, and cancelled jobs
            for status in ['completed', 'failed', 'timeout', 'cancelled']:
                status_dir = self.data_dir / status
                if not status_dir.exists():
                    continue

                for job_file in status_dir.glob('*.json'):
                    try:
                        # Check file modification time
                        if datetime.fromtimestamp(job_file.stat().st_mtime) < cutoff_time:
                            job_file.unlink()
                            cleaned_count += 1

                    except Exception as e:
                        logger.error("Failed to clean up job file", file=str(job_file), error=str(e))

            if cleaned_count > 0:
                logger.info("Old jobs cleaned up", count=cleaned_count, days_old=days_old)

            return cleaned_count

        except Exception as e:
            logger.error("Failed to cleanup old jobs", error=str(e))
            return 0

    def delete_job(self, job_id: str) -> bool:
        """Delete a job completely."""
        try:
            job_file = self._find_job_file(job_id)
            if job_file:
                job_file.unlink()
                logger.info("Job deleted", job_id=job_id)
                return True
            else:
                logger.warning("Job file not found for deletion", job_id=job_id)
                return False

        except Exception as e:
            logger.error("Failed to delete job", job_id=job_id, error=str(e))
            return False
```

### 4. Service Layer

**File**: `packages/morag/src/morag/services/remote_job_service.py`

```python
import structlog
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

from morag_core.models.remote_job import RemoteJob
from morag.repositories.remote_job_repository import RemoteJobRepository
from morag.models.remote_job_api import CreateRemoteJobRequest, SubmitResultRequest

logger = structlog.get_logger(__name__)

class RemoteJobService:
    """Service layer for remote job operations."""

    def __init__(self, repository: RemoteJobRepository = None):
        self.repository = repository or RemoteJobRepository()

    def create_job(self, request: CreateRemoteJobRequest) -> RemoteJob:
        """Create a new remote job."""
        return self.repository.create_job(
            ingestion_task_id=request.ingestion_task_id,
            source_file_path=request.source_file_path,
            content_type=request.content_type,
            task_options=request.task_options
        )

    def poll_available_jobs(self, worker_id: str, content_types: List[str], max_jobs: int = 1) -> List[RemoteJob]:
        """Poll for available jobs matching worker capabilities."""
        return self.repository.poll_available_jobs(worker_id, content_types, max_jobs)

    def submit_result(self, job_id: str, result: SubmitResultRequest) -> Optional[RemoteJob]:
        """Submit processing result for a job."""
        return self.repository.submit_result(
            job_id=job_id,
            success=result.success,
            content=result.content,
            metadata=result.metadata,
            error_message=result.error_message,
            processing_time=result.processing_time
        )

    def get_job_status(self, job_id: str) -> Optional[RemoteJob]:
        """Get current status of a job."""
        return self.repository.get_job(job_id)

    def cleanup_expired_jobs(self) -> int:
        """Clean up expired jobs."""
        return self.repository.cleanup_expired_jobs()

    def cleanup_old_jobs(self, days_old: int = 7) -> int:
        """Clean up old completed jobs."""
        return self.repository.cleanup_old_jobs(days_old)
```

### 5. API Endpoints

**File**: `packages/morag/src/morag/api/remote_jobs.py`

```python
from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
import structlog

from morag.models.remote_job_api import (
    CreateRemoteJobRequest, CreateRemoteJobResponse,
    PollJobsRequest, PollJobsResponse,
    SubmitResultRequest, SubmitResultResponse,
    JobStatusResponse
)
from morag.services.remote_job_service import RemoteJobService

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/remote-jobs", tags=["Remote Jobs"])

def get_remote_job_service() -> RemoteJobService:
    """Dependency to get remote job service."""
    return RemoteJobService()

@router.post("/", response_model=CreateRemoteJobResponse)
async def create_remote_job(
    request: CreateRemoteJobRequest,
    service: RemoteJobService = Depends(get_remote_job_service)
):
    """Create a new remote conversion job."""
    try:
        job = service.create_job(request)

        logger.info("Remote job created",
                   job_id=job.id,
                   content_type=job.content_type,
                   ingestion_task_id=job.ingestion_task_id)

        return CreateRemoteJobResponse(
            job_id=job.id,
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
    service: RemoteJobService = Depends(get_remote_job_service)
):
    """Poll for available remote jobs."""
    try:
        content_type_list = [ct.strip() for ct in content_types.split(',')]
        jobs = service.poll_available_jobs(worker_id, content_type_list, max_jobs)

        if not jobs:
            return PollJobsResponse()

        job = jobs[0]  # Return first job for now

        # Generate secure download URL for source file
        source_file_url = f"/api/v1/remote-jobs/{job.id}/download"

        logger.info("Job assigned to worker",
                   job_id=job.id,
                   worker_id=worker_id,
                   content_type=job.content_type)

        return PollJobsResponse(
            job_id=job.id,
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
