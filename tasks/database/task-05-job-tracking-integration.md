# Task 05: Job Tracking Integration

## 📋 Task Overview

**Objective**: Integrate comprehensive job tracking and monitoring with the existing Celery task system, providing real-time status updates, progress tracking, and detailed job history management.

**Priority**: High - Essential for monitoring and debugging
**Estimated Time**: 1-2 weeks
**Dependencies**: Task 04 (Document Lifecycle Management)

## 🎯 Goals

1. Integrate database job tracking with Celery tasks
2. Implement real-time job status updates and progress tracking
3. Add detailed job history and logging
4. Create job management API endpoints
5. Implement job retry and failure handling
6. Add job performance metrics and analytics
7. Create job monitoring dashboard data

## 📊 Current State Analysis

### Existing Job Model
- **Fields**: ID, document_name, document_type, start_date, end_date, status, percentage, summary, document_id, user_id
- **Status**: PENDING, WAITING_FOR_REMOTE_WORKER, PROCESSING, FINISHED, FAILED, CANCELLED
- **Relationships**: Document (target), User (owner)

### Current MoRAG Job Tracking
- **System**: Celery task tracking only
- **Status**: Basic task states in Redis
- **Monitoring**: Limited to Celery flower
- **History**: No persistent job history

## 🔧 Implementation Plan

### Step 1: Create Job Service Layer

**Files to Create**:
```
packages/morag-core/src/morag_core/
├── jobs/
│   ├── __init__.py
│   ├── models.py          # Pydantic models for jobs
│   ├── service.py         # Job service logic
│   ├── tracker.py         # Job tracking integration
│   └── analytics.py       # Job analytics and metrics
```

**Implementation Details**:

1. **Job Models**:
```python
# packages/morag-core/src/morag_core/jobs/models.py
"""Job tracking and management models."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class JobStatus(str, Enum):
    PENDING = "PENDING"
    WAITING_FOR_REMOTE_WORKER = "WAITING_FOR_REMOTE_WORKER"
    PROCESSING = "PROCESSING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

class JobPriority(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    URGENT = "URGENT"

class JobCreate(BaseModel):
    document_id: str
    document_name: str
    document_type: str
    priority: JobPriority = JobPriority.NORMAL
    metadata: Optional[Dict[str, Any]] = None
    estimated_duration: Optional[int] = None  # seconds

class JobUpdate(BaseModel):
    status: Optional[JobStatus] = None
    percentage: Optional[int] = Field(None, ge=0, le=100)
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None

class JobResponse(BaseModel):
    id: str
    document_id: str
    document_name: str
    document_type: str
    user_id: str
    status: JobStatus
    priority: JobPriority
    percentage: int
    summary: str
    start_date: datetime
    end_date: Optional[datetime]
    duration_seconds: Optional[int]
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]]
    error_details: Optional[str]

class JobSearchRequest(BaseModel):
    status: Optional[JobStatus] = None
    document_type: Optional[str] = None
    priority: Optional[JobPriority] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = Field(default=50, le=1000)
    offset: int = Field(default=0, ge=0)

class JobStatistics(BaseModel):
    total_jobs: int
    jobs_by_status: Dict[str, int]
    jobs_by_type: Dict[str, int]
    average_duration_seconds: float
    success_rate: float
    total_processing_time: int
    active_jobs: int

class JobProgressUpdate(BaseModel):
    job_id: str
    percentage: int
    message: str
    timestamp: datetime
    details: Optional[Dict[str, Any]] = None
```

2. **Job Service**:
```python
# packages/morag-core/src/morag_core/jobs/service.py
"""Job management service."""

from typing import Optional, List, Dict, Any, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, func
import structlog
from datetime import datetime, timedelta

from morag_core.database import (
    Job, get_database_manager, JobStatus as DBJobStatus
)
from .models import (
    JobCreate, JobUpdate, JobResponse, JobSearchRequest, 
    JobStatistics, JobStatus, JobPriority
)
from morag_core.exceptions import NotFoundError, ValidationError

logger = structlog.get_logger(__name__)

class JobService:
    """Job management service."""
    
    def __init__(self):
        self.db_manager = get_database_manager()
    
    def create_job(self, user_id: str, job_data: JobCreate, celery_task_id: str) -> JobResponse:
        """Create a new job record."""
        with self.db_manager.get_session() as session:
            job = Job(
                id=celery_task_id,  # Use Celery task ID as job ID
                document_id=job_data.document_id,
                document_name=job_data.document_name,
                document_type=job_data.document_type,
                user_id=user_id,
                status=DBJobStatus.PENDING,
                percentage=0,
                summary="Job created",
                metadata=job_data.metadata or {}
            )
            
            # Add priority and estimated duration to metadata
            job.metadata.update({
                'priority': job_data.priority.value,
                'estimated_duration': job_data.estimated_duration
            })
            
            session.add(job)
            session.flush()
            
            logger.info("Job created", 
                       job_id=job.id, 
                       document_id=job_data.document_id,
                       user_id=user_id)
            
            return self._job_to_response(job)
    
    def get_job(self, job_id: str, user_id: str) -> Optional[JobResponse]:
        """Get job by ID with user ownership check."""
        with self.db_manager.get_session() as session:
            job = session.query(Job).filter(
                and_(
                    Job.id == job_id,
                    Job.user_id == user_id
                )
            ).first()
            
            if job:
                return self._job_to_response(job)
            return None
    
    def update_job(self, job_id: str, job_data: JobUpdate) -> Optional[JobResponse]:
        """Update job status and progress."""
        with self.db_manager.get_session() as session:
            job = session.query(Job).filter_by(id=job_id).first()
            
            if not job:
                logger.warning("Job not found for update", job_id=job_id)
                return None
            
            # Update fields
            if job_data.status is not None:
                old_status = job.status.value
                job.status = DBJobStatus(job_data.status.value)
                
                # Set end date for terminal states
                if job_data.status in [JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    job.end_date = datetime.utcnow()
                
                logger.info("Job status updated", 
                           job_id=job_id,
                           old_status=old_status,
                           new_status=job_data.status.value)
            
            if job_data.percentage is not None:
                job.percentage = job_data.percentage
            
            if job_data.summary is not None:
                job.summary = job_data.summary
            
            if job_data.metadata is not None:
                if job.metadata:
                    job.metadata.update(job_data.metadata)
                else:
                    job.metadata = job_data.metadata
            
            if job_data.error_details is not None:
                if not job.metadata:
                    job.metadata = {}
                job.metadata['error_details'] = job_data.error_details
            
            return self._job_to_response(job)
    
    def search_jobs(self, user_id: str, search_request: JobSearchRequest) -> Tuple[List[JobResponse], int]:
        """Search user's jobs with filters."""
        with self.db_manager.get_session() as session:
            query = session.query(Job).filter(Job.user_id == user_id)
            
            # Apply filters
            if search_request.status:
                query = query.filter(Job.status == DBJobStatus(search_request.status.value))
            
            if search_request.document_type:
                query = query.filter(Job.document_type == search_request.document_type)
            
            if search_request.priority:
                query = query.filter(
                    Job.metadata.op('->>')('priority') == search_request.priority.value
                )
            
            if search_request.date_from:
                query = query.filter(Job.start_date >= search_request.date_from)
            
            if search_request.date_to:
                query = query.filter(Job.start_date <= search_request.date_to)
            
            # Get total count
            total_count = query.count()
            
            # Apply pagination and ordering
            jobs = query.order_by(desc(Job.start_date))\
                       .offset(search_request.offset)\
                       .limit(search_request.limit)\
                       .all()
            
            return [self._job_to_response(job) for job in jobs], total_count
    
    def cancel_job(self, job_id: str, user_id: str) -> bool:
        """Cancel a job."""
        with self.db_manager.get_session() as session:
            job = session.query(Job).filter(
                and_(
                    Job.id == job_id,
                    Job.user_id == user_id,
                    Job.status.in_([DBJobStatus.PENDING, DBJobStatus.PROCESSING])
                )
            ).first()
            
            if not job:
                return False
            
            job.status = DBJobStatus.CANCELLED
            job.end_date = datetime.utcnow()
            job.summary = "Job cancelled by user"
            
            logger.info("Job cancelled", job_id=job_id, user_id=user_id)
            return True
    
    def get_active_jobs(self, user_id: str) -> List[JobResponse]:
        """Get all active jobs for user."""
        with self.db_manager.get_session() as session:
            jobs = session.query(Job).filter(
                and_(
                    Job.user_id == user_id,
                    Job.status.in_([
                        DBJobStatus.PENDING, 
                        DBJobStatus.WAITING_FOR_REMOTE_WORKER,
                        DBJobStatus.PROCESSING
                    ])
                )
            ).order_by(desc(Job.start_date)).all()
            
            return [self._job_to_response(job) for job in jobs]
    
    def get_job_statistics(self, user_id: str, days: int = 30) -> JobStatistics:
        """Get job statistics for user."""
        with self.db_manager.get_session() as session:
            # Date filter
            date_from = datetime.utcnow() - timedelta(days=days)
            
            jobs = session.query(Job).filter(
                and_(
                    Job.user_id == user_id,
                    Job.start_date >= date_from
                )
            ).all()
            
            # Calculate statistics
            total_jobs = len(jobs)
            jobs_by_status = {}
            jobs_by_type = {}
            total_duration = 0
            successful_jobs = 0
            active_jobs = 0
            
            for job in jobs:
                # Count by status
                status = job.status.value
                jobs_by_status[status] = jobs_by_status.get(status, 0) + 1
                
                # Count by type
                job_type = job.document_type
                jobs_by_type[job_type] = jobs_by_type.get(job_type, 0) + 1
                
                # Calculate duration
                if job.end_date:
                    duration = (job.end_date - job.start_date).total_seconds()
                    total_duration += duration
                
                # Count successful jobs
                if job.status == DBJobStatus.FINISHED:
                    successful_jobs += 1
                
                # Count active jobs
                if job.status in [DBJobStatus.PENDING, DBJobStatus.PROCESSING]:
                    active_jobs += 1
            
            # Calculate averages
            average_duration = total_duration / total_jobs if total_jobs > 0 else 0.0
            success_rate = successful_jobs / total_jobs if total_jobs > 0 else 0.0
            
            return JobStatistics(
                total_jobs=total_jobs,
                jobs_by_status=jobs_by_status,
                jobs_by_type=jobs_by_type,
                average_duration_seconds=average_duration,
                success_rate=success_rate,
                total_processing_time=int(total_duration),
                active_jobs=active_jobs
            )
    
    def cleanup_old_jobs(self, days: int = 90) -> int:
        """Clean up old completed jobs."""
        with self.db_manager.get_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            deleted_count = session.query(Job).filter(
                and_(
                    Job.end_date < cutoff_date,
                    Job.status.in_([
                        DBJobStatus.FINISHED, 
                        DBJobStatus.FAILED, 
                        DBJobStatus.CANCELLED
                    ])
                )
            ).delete()
            
            logger.info("Old jobs cleaned up", count=deleted_count, days=days)
            return deleted_count
    
    def _job_to_response(self, job: Job) -> JobResponse:
        """Convert Job model to JobResponse."""
        duration_seconds = None
        if job.end_date:
            duration_seconds = int((job.end_date - job.start_date).total_seconds())
        
        priority = JobPriority.NORMAL
        if job.metadata and 'priority' in job.metadata:
            try:
                priority = JobPriority(job.metadata['priority'])
            except ValueError:
                pass
        
        error_details = None
        if job.metadata and 'error_details' in job.metadata:
            error_details = job.metadata['error_details']
        
        return JobResponse(
            id=job.id,
            document_id=job.document_id,
            document_name=job.document_name,
            document_type=job.document_type,
            user_id=job.user_id,
            status=JobStatus(job.status.value),
            priority=priority,
            percentage=job.percentage,
            summary=job.summary,
            start_date=job.start_date,
            end_date=job.end_date,
            duration_seconds=duration_seconds,
            created_at=job.created_at,
            updated_at=job.updated_at,
            metadata=job.metadata,
            error_details=error_details
        )
```

### Step 2: Create Job Tracker Integration

**File to Create**: `packages/morag-core/src/morag_core/jobs/tracker.py`

```python
"""Job tracking integration with Celery."""

import structlog
from typing import Optional, Dict, Any
from celery import Task
from datetime import datetime

from .service import JobService
from .models import JobUpdate, JobStatus

logger = structlog.get_logger(__name__)

class JobTracker:
    """Integration between Celery tasks and database job tracking."""
    
    def __init__(self):
        self.job_service = JobService()
    
    def update_job_progress(self, task_id: str, percentage: int, 
                           message: str, details: Optional[Dict[str, Any]] = None):
        """Update job progress from Celery task."""
        try:
            job_update = JobUpdate(
                status=JobStatus.PROCESSING,
                percentage=percentage,
                summary=message,
                metadata=details or {}
            )
            
            updated_job = self.job_service.update_job(task_id, job_update)
            if updated_job:
                logger.info("Job progress updated", 
                           task_id=task_id, 
                           percentage=percentage,
                           message=message)
            else:
                logger.warning("Job not found for progress update", task_id=task_id)
                
        except Exception as e:
            logger.error("Failed to update job progress", 
                        task_id=task_id, 
                        error=str(e))
    
    def mark_job_started(self, task_id: str):
        """Mark job as started."""
        self.update_job_progress(task_id, 0, "Job started")
    
    def mark_job_completed(self, task_id: str, summary: str, 
                          metadata: Optional[Dict[str, Any]] = None):
        """Mark job as completed."""
        try:
            job_update = JobUpdate(
                status=JobStatus.FINISHED,
                percentage=100,
                summary=summary,
                metadata=metadata or {}
            )
            
            updated_job = self.job_service.update_job(task_id, job_update)
            if updated_job:
                logger.info("Job completed", task_id=task_id, summary=summary)
            
        except Exception as e:
            logger.error("Failed to mark job as completed", 
                        task_id=task_id, 
                        error=str(e))
    
    def mark_job_failed(self, task_id: str, error_message: str, 
                       error_details: Optional[str] = None):
        """Mark job as failed."""
        try:
            job_update = JobUpdate(
                status=JobStatus.FAILED,
                summary=f"Job failed: {error_message}",
                error_details=error_details
            )
            
            updated_job = self.job_service.update_job(task_id, job_update)
            if updated_job:
                logger.error("Job failed", task_id=task_id, error=error_message)
            
        except Exception as e:
            logger.error("Failed to mark job as failed", 
                        task_id=task_id, 
                        error=str(e))

# Global job tracker instance
job_tracker = JobTracker()

class TrackedTask(Task):
    """Celery task base class with automatic job tracking."""
    
    def on_success(self, retval, task_id, args, kwargs):
        """Called on task success."""
        summary = f"Task completed successfully"
        if hasattr(retval, 'summary'):
            summary = retval.summary
        
        metadata = {}
        if hasattr(retval, 'metadata'):
            metadata = retval.metadata
        
        job_tracker.mark_job_completed(task_id, summary, metadata)
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Called on task failure."""
        error_message = str(exc)
        error_details = str(einfo)
        
        job_tracker.mark_job_failed(task_id, error_message, error_details)
    
    def on_retry(self, exc, task_id, args, kwargs, einfo):
        """Called on task retry."""
        job_tracker.update_job_progress(
            task_id, 0, f"Task retrying: {str(exc)}"
        )
    
    def update_progress(self, percentage: int, message: str, 
                       details: Optional[Dict[str, Any]] = None):
        """Update task progress."""
        job_tracker.update_job_progress(
            self.request.id, percentage, message, details
        )
```

## 🧪 Testing Requirements

### Unit Tests
```python
# tests/test_job_tracking.py
import pytest
from morag_core.jobs import JobService, JobTracker
from morag_core.jobs.models import JobCreate, JobStatus, JobPriority

def test_job_creation():
    """Test job creation."""
    service = JobService()
    job_data = JobCreate(
        document_id="doc123",
        document_name="Test Document",
        document_type="pdf",
        priority=JobPriority.HIGH
    )
    
    job = service.create_job("user123", job_data, "task123")
    assert job.document_id == "doc123"
    assert job.status == JobStatus.PENDING
    assert job.percentage == 0

def test_job_progress_tracking():
    """Test job progress updates."""
    service = JobService()
    tracker = JobTracker()
    
    # Create job
    job_data = JobCreate(
        document_id="doc123",
        document_name="Progress Test",
        document_type="pdf"
    )
    job = service.create_job("user123", job_data, "task123")
    
    # Update progress
    tracker.update_job_progress("task123", 50, "Processing...")
    updated_job = service.get_job("task123", "user123")
    assert updated_job.percentage == 50
    assert updated_job.status == JobStatus.PROCESSING
    
    # Complete job
    tracker.mark_job_completed("task123", "Job completed successfully")
    final_job = service.get_job("task123", "user123")
    assert final_job.status == JobStatus.FINISHED
    assert final_job.percentage == 100
```

## 📋 Acceptance Criteria

- [ ] Job service with CRUD operations implemented
- [ ] Job tracking integration with Celery tasks
- [ ] Real-time job progress updates working
- [ ] Job search and filtering functional
- [ ] Job statistics and analytics available
- [ ] Job cancellation functionality
- [ ] Error handling and retry logic
- [ ] Job cleanup for old records
- [ ] Comprehensive unit tests passing
- [ ] API endpoints for job management created

## 🔄 Next Steps

After completing this task:
1. Proceed to [Task 06: Database Server Management](./task-06-database-server-management.md)
2. Integrate job tracking with existing ingestion tasks
3. Add real-time job monitoring dashboard
4. Test job tracking with various processing scenarios

## 📝 Notes

- Ensure proper integration with existing Celery configuration
- Add comprehensive logging for job state changes
- Consider implementing job queues for different priorities
- Add job performance monitoring and alerting
- Implement job retry policies and failure recovery
