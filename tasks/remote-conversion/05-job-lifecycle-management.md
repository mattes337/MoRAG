# Task 5: Job Lifecycle Management

## Overview

Implement complete job lifecycle management for remote conversion jobs, including status transitions, timeout handling, cleanup of completed jobs, and integration with the existing ingestion task completion flow.

## Objectives

1. Implement comprehensive job status management
2. Add timeout and heartbeat monitoring for remote jobs
3. Create cleanup services for completed and failed jobs
4. Integrate remote job completion with ingestion pipeline
5. Add monitoring and alerting for job lifecycle events
6. Implement retry and recovery mechanisms

## Technical Requirements

### 1. Job Lifecycle State Machine

**File**: `packages/morag/src/morag/services/job_lifecycle_manager.py`

```python
import structlog
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from enum import Enum
import time

from morag_core.models.remote_job import RemoteJob
from morag.storage.storage_manager import storage_manager

logger = structlog.get_logger(__name__)

class JobStatus(Enum):
    """Remote job status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class JobLifecycleManager:
    """Manages the complete lifecycle of remote conversion jobs."""

    def __init__(self):
        self.heartbeat_timeout = timedelta(minutes=5)  # Worker must send heartbeat every 5 minutes
        self.job_timeouts = {
            'audio': timedelta(minutes=30),
            'video': timedelta(hours=1),
            'document': timedelta(minutes=10),
            'image': timedelta(minutes=5),
            'web': timedelta(minutes=10),
            'youtube': timedelta(minutes=30)
        }

    def transition_job_status(self, job_id: str, new_status: JobStatus,
                             error_message: str = None, result_data: Dict[str, Any] = None) -> bool:
        """Transition job to new status with validation."""
        try:
            # Load current job data
            job_data = storage_manager.remote_jobs.load_job(job_id)
            if not job_data:
                logger.error("Job not found for status transition", job_id=job_id)
                return False

            job = RemoteJob.from_dict(job_data)
            old_status = job.status

            # Validate status transition
            if not self._is_valid_transition(old_status, new_status.value):
                logger.error("Invalid status transition",
                           job_id=job_id,
                           old_status=old_status,
                           new_status=new_status.value)
                return False

            # Update job status
            job.status = new_status.value

            # Update timestamps based on status
            now = datetime.utcnow()
            if new_status == JobStatus.PROCESSING:
                job.started_at = now
                job.timeout_at = now + self.job_timeouts.get(job.content_type, timedelta(hours=1))
            elif new_status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.TIMEOUT, JobStatus.CANCELLED]:
                job.completed_at = now

            # Update error message and result data
            if error_message:
                job.error_message = error_message
            if result_data:
                job.result_data = result_data

            # Save updated job
            if not storage_manager.remote_jobs.save_job(job.to_dict()):
                logger.error("Failed to save job after status transition", job_id=job_id)
                return False

            logger.info("Job status transitioned",
                       job_id=job_id,
                       old_status=old_status,
                       new_status=new_status.value)

            # Trigger post-transition actions
            self._handle_status_transition(job, old_status, new_status.value)

            return True

        except Exception as e:
            logger.error("Failed to transition job status",
                        job_id=job_id,
                        error=str(e))
            return False

    def _is_valid_transition(self, old_status: str, new_status: str) -> bool:
        """Validate if status transition is allowed."""
        valid_transitions = {
            'pending': ['processing', 'cancelled', 'timeout'],
            'processing': ['completed', 'failed', 'timeout', 'cancelled'],
            'failed': ['retrying', 'cancelled'],
            'timeout': ['retrying', 'cancelled'],
            'retrying': ['processing', 'failed', 'cancelled'],
            'completed': ['cancelled'],  # Allow cancellation of completed jobs for cleanup
            'cancelled': []  # Terminal state
        }

        return new_status in valid_transitions.get(old_status, [])

    def _handle_status_transition(self, job: RemoteJob, old_status: str, new_status: str):
        """Handle post-transition actions."""
        try:
            if new_status == 'completed':
                # Continue ingestion pipeline
                self._continue_ingestion_pipeline(job)
            elif new_status in ['failed', 'timeout'] and job.can_retry():
                # Schedule retry
                self._schedule_retry(job)
            elif new_status in ['failed', 'timeout', 'cancelled']:
                # Handle final failure
                self._handle_job_failure(job)

        except Exception as e:
            logger.error("Error in post-transition handling",
                        job_id=str(job.id),
                        error=str(e))

    def _continue_ingestion_pipeline(self, job: RemoteJob):
        """Continue the ingestion pipeline after successful remote processing."""
        try:
            from morag.ingest_tasks import continue_ingestion_after_remote_processing

            # Extract result data
            result_data = job.result_data or {}

            # Continue ingestion task
            continue_ingestion_after_remote_processing.delay(
                job.ingestion_task_id,
                {
                    'success': True,
                    'content': result_data.get('content', ''),
                    'metadata': result_data.get('metadata', {}),
                    'processing_time': result_data.get('processing_time', 0.0)
                },
                job.task_options
            )

            logger.info("Ingestion pipeline continuation scheduled",
                       job_id=str(job.id),
                       ingestion_task_id=job.ingestion_task_id)

        except Exception as e:
            logger.error("Failed to continue ingestion pipeline",
                        job_id=str(job.id),
                        error=str(e))

    def _schedule_retry(self, job: RemoteJob):
        """Schedule job retry with exponential backoff."""
        try:
            # Update retry count and reset status
            job.retry_count += 1
            job.status = 'pending'
            job.worker_id = None
            job.started_at = None
            job.completed_at = None

            # Calculate retry delay (exponential backoff)
            delay_minutes = min(2 ** job.retry_count, 60)  # Max 1 hour delay
            job.timeout_at = datetime.utcnow() + timedelta(minutes=delay_minutes)

            # Save updated job
            if storage_manager.remote_jobs.save_job(job.to_dict()):
                logger.info("Job retry scheduled",
                           job_id=job.id,
                           retry_count=job.retry_count,
                           delay_minutes=delay_minutes)
            else:
                logger.error("Failed to save job for retry", job_id=job.id)

        except Exception as e:
            logger.error("Failed to schedule job retry",
                        job_id=job.id,
                        error=str(e))

    def _handle_job_failure(self, job: RemoteJob):
        """Handle final job failure."""
        try:
            from morag.ingest_tasks import handle_remote_job_failure

            # Notify ingestion task of failure
            handle_remote_job_failure.delay(
                job.ingestion_task_id,
                {
                    'error_message': job.error_message or 'Remote processing failed',
                    'retry_count': job.retry_count,
                    'job_id': str(job.id)
                }
            )

            logger.info("Job failure handled",
                       job_id=str(job.id),
                       ingestion_task_id=job.ingestion_task_id)

        except Exception as e:
            logger.error("Failed to handle job failure",
                        job_id=str(job.id),
                        error=str(e))

    def check_expired_jobs(self) -> int:
        """Check for and handle expired jobs."""
        try:
            # Get expired jobs from storage
            expired_jobs_data = storage_manager.remote_jobs.get_expired_jobs()

            expired_count = 0
            for job_data in expired_jobs_data:
                job = RemoteJob.from_dict(job_data)
                if self.transition_job_status(job.id, JobStatus.TIMEOUT):
                    expired_count += 1

            if expired_count > 0:
                logger.info("Expired jobs handled", count=expired_count)

            return expired_count

        except Exception as e:
            logger.error("Failed to check expired jobs", error=str(e))
            return 0

    def cleanup_old_jobs(self, days_old: int = 7) -> int:
        """Clean up old completed/failed jobs."""
        try:
            # Get old jobs from storage
            old_jobs_data = storage_manager.remote_jobs.get_old_jobs(days_old)

            cleaned_count = 0
            for job_data in old_jobs_data:
                try:
                    job = RemoteJob.from_dict(job_data)

                    # Clean up any associated files
                    self._cleanup_job_files(job)

                    # Delete job from storage
                    if storage_manager.remote_jobs.delete_job(job.id):
                        cleaned_count += 1

                except Exception as e:
                    logger.error("Failed to clean up job",
                               job_id=job_data.get('id'),
                               error=str(e))

            if cleaned_count > 0:
                logger.info("Old jobs cleaned up", count=cleaned_count)

            return cleaned_count

        except Exception as e:
            logger.error("Failed to cleanup old jobs", error=str(e))
            return 0

    def _cleanup_job_files(self, job: RemoteJob):
        """Clean up files associated with a job."""
        try:
            import os
            from pathlib import Path

            # Clean up temporary files for this job
            temp_dir = Path(f"/tmp/morag_remote_{job.id}")
            if temp_dir.exists():
                import shutil
                shutil.rmtree(temp_dir)
                logger.debug("Cleaned up job temp directory",
                           job_id=str(job.id),
                           temp_dir=str(temp_dir))

        except Exception as e:
            logger.warning("Failed to cleanup job files",
                          job_id=str(job.id),
                          error=str(e))

    def get_job_statistics(self) -> Dict[str, Any]:
        """Get statistics about job processing."""
        try:
            # Get basic statistics from storage
            storage_stats = storage_manager.remote_jobs.get_statistics()

            # Calculate additional statistics
            stats = {
                'status_counts': storage_stats.get('status_counts', {}),
                'content_type_counts': storage_stats.get('content_type_counts', {}),
                'total_jobs': storage_stats.get('total_jobs', 0),
                'storage_size_mb': storage_stats.get('storage_size_mb', 0),
                'avg_processing_times': {},
                'recent_jobs_24h': 0,
                'timestamp': datetime.utcnow().isoformat()
            }

            # Calculate average processing times by content type
            recent_cutoff = datetime.utcnow() - timedelta(hours=24)

            for content_type in storage_stats.get('content_type_counts', {}):
                jobs = storage_manager.remote_jobs.find_jobs_by_content_type(content_type)

                # Filter completed jobs and calculate average processing time
                completed_jobs = [
                    job for job in jobs
                    if job.get('status') == 'completed' and
                       job.get('started_at') and job.get('completed_at')
                ]

                if completed_jobs:
                    processing_times = []
                    for job in completed_jobs:
                        try:
                            started = datetime.fromisoformat(job['started_at'])
                            completed = datetime.fromisoformat(job['completed_at'])
                            processing_times.append((completed - started).total_seconds())
                        except Exception:
                            continue

                    if processing_times:
                        stats['avg_processing_times'][content_type] = sum(processing_times) / len(processing_times)

                # Count recent jobs
                recent_jobs = [
                    job for job in jobs
                    if job.get('created_at') and
                       datetime.fromisoformat(job['created_at']) >= recent_cutoff
                ]
                stats['recent_jobs_24h'] += len(recent_jobs)

            return stats

        except Exception as e:
            logger.error("Failed to get job statistics", error=str(e))
            return {}
```

### 2. Background Job Monitor Service

**File**: `packages/morag/src/morag/services/job_monitor.py`

```python
import asyncio
import structlog
from typing import Dict, Any
import os
from datetime import datetime, timedelta

from morag.services.job_lifecycle_manager import JobLifecycleManager

logger = structlog.get_logger(__name__)

class JobMonitorService:
    """Background service for monitoring remote job lifecycle."""

    def __init__(self):
        self.lifecycle_manager = JobLifecycleManager()
        self.running = False
        self.check_interval = int(os.getenv('MORAG_JOB_MONITOR_INTERVAL', '60'))  # 1 minute
        self.cleanup_interval = int(os.getenv('MORAG_JOB_CLEANUP_INTERVAL', '3600'))  # 1 hour
        self.last_cleanup = datetime.utcnow()

    async def start(self):
        """Start the job monitor service."""
        self.running = True
        logger.info("Starting job monitor service",
                   check_interval=self.check_interval,
                   cleanup_interval=self.cleanup_interval)

        while self.running:
            try:
                await self._monitor_cycle()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error("Error in job monitor cycle", error=str(e))
                await asyncio.sleep(self.check_interval)

    def stop(self):
        """Stop the job monitor service."""
        self.running = False
        logger.info("Stopping job monitor service")

    async def _monitor_cycle(self):
        """Execute one monitoring cycle."""
        try:
            # Check for expired jobs
            expired_count = self.lifecycle_manager.check_expired_jobs()
            if expired_count > 0:
                logger.info("Handled expired jobs", count=expired_count)

            # Periodic cleanup
            now = datetime.utcnow()
            if (now - self.last_cleanup).total_seconds() >= self.cleanup_interval:
                cleanup_count = self.lifecycle_manager.cleanup_old_jobs()
                if cleanup_count > 0:
                    logger.info("Cleaned up old jobs", count=cleanup_count)
                self.last_cleanup = now

            # Log statistics periodically
            if now.minute % 10 == 0:  # Every 10 minutes
                stats = self.lifecycle_manager.get_job_statistics()
                if stats:
                    logger.info("Job statistics", **stats)

        except Exception as e:
            logger.error("Error in monitor cycle", error=str(e))
```

### 3. Enhanced Ingestion Task Completion

**File**: `packages/morag/src/morag/ingest_tasks_completion.py`

```python
import structlog
from typing import Dict, Any
from celery import current_task
import asyncio

from morag.worker import celery_app
from morag_core.models import ProcessingResult

logger = structlog.get_logger(__name__)

@celery_app.task(bind=True)
def continue_ingestion_after_remote_processing(self, ingestion_task_id: str,
                                             processing_result: Dict[str, Any],
                                             task_options: Dict[str, Any]):
    """Continue ingestion pipeline after remote processing completion."""

    async def _continue():
        try:
            logger.info("Continuing ingestion after remote processing",
                       ingestion_task_id=ingestion_task_id,
                       task_id=self.request.id)

            self.update_state(state='PROGRESS', meta={'stage': 'remote_processing_completed', 'progress': 0.6})

            # Convert processing result back to ProcessingResult object
            result = ProcessingResult(
                success=processing_result['success'],
                text_content=processing_result.get('content', ''),
                metadata=processing_result.get('metadata', {}),
                processing_time=processing_result.get('processing_time', 0.0),
                error_message=processing_result.get('error_message')
            )

            # Continue with vector storage (reuse existing logic)
            from morag.ingest_tasks_enhanced import _complete_ingestion
            return await _complete_ingestion(self, result, task_options)

        except Exception as e:
            logger.error("Failed to continue ingestion after remote processing",
                        ingestion_task_id=ingestion_task_id,
                        error=str(e))
            self.update_state(state='FAILURE', meta={'error': str(e)})
            raise

    return asyncio.run(_continue())

@celery_app.task(bind=True)
def handle_remote_job_failure(self, ingestion_task_id: str, failure_info: Dict[str, Any]):
    """Handle remote job failure and update ingestion task."""

    try:
        logger.error("Remote job failed for ingestion task",
                    ingestion_task_id=ingestion_task_id,
                    failure_info=failure_info)

        # Update the original ingestion task with failure information
        error_message = failure_info.get('error_message', 'Remote processing failed')
        retry_count = failure_info.get('retry_count', 0)
        job_id = failure_info.get('job_id')

        # Check if we should attempt local fallback
        # This would depend on the original task options

        self.update_state(
            state='FAILURE',
            meta={
                'error': error_message,
                'remote_job_id': job_id,
                'retry_count': retry_count,
                'stage': 'remote_processing_failed'
            }
        )

        # Send webhook notification if configured
        # This would be handled by existing webhook logic

    except Exception as e:
        logger.error("Failed to handle remote job failure",
                    ingestion_task_id=ingestion_task_id,
                    error=str(e))
        raise
```

### 4. Job Lifecycle API Endpoints

**File**: `packages/morag/src/morag/api/job_lifecycle.py`

```python
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Dict, Any
import structlog

from morag.services.job_lifecycle_manager import JobLifecycleManager, JobStatus
from morag.database import get_db_session

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/api/v1/remote-jobs", tags=["Job Lifecycle"])

@router.post("/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    db: Session = Depends(get_db_session)
):
    """Cancel a remote job."""
    try:
        lifecycle_manager = JobLifecycleManager()
        success = lifecycle_manager.transition_job_status(
            db, job_id, JobStatus.CANCELLED
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found or cannot be cancelled"
            )

        logger.info("Job cancelled", job_id=job_id)
        return {"status": "cancelled", "job_id": job_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel job", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel job: {str(e)}"
        )

@router.post("/{job_id}/retry")
async def retry_job(
    job_id: str,
    db: Session = Depends(get_db_session)
):
    """Retry a failed remote job."""
    try:
        lifecycle_manager = JobLifecycleManager()
        success = lifecycle_manager.transition_job_status(
            db, job_id, JobStatus.RETRYING
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found or cannot be retried"
            )

        logger.info("Job retry initiated", job_id=job_id)
        return {"status": "retrying", "job_id": job_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to retry job", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retry job: {str(e)}"
        )

@router.get("/statistics")
async def get_job_statistics():
    """Get job processing statistics."""
    try:
        lifecycle_manager = JobLifecycleManager()
        stats = lifecycle_manager.get_job_statistics()
        return stats

    except Exception as e:
        logger.error("Failed to get job statistics", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )

@router.post("/cleanup")
async def cleanup_old_jobs(
    days_old: int = 7,
    db: Session = Depends(get_db_session)
):
    """Manually trigger cleanup of old jobs."""
    try:
        lifecycle_manager = JobLifecycleManager()
        cleaned_count = lifecycle_manager.cleanup_old_jobs(days_old)

        logger.info("Manual job cleanup completed", count=cleaned_count)
        return {"cleaned_jobs": cleaned_count, "days_old": days_old}

    except Exception as e:
        logger.error("Failed to cleanup old jobs", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup jobs: {str(e)}"
        )
```

## Implementation Steps

1. **Create Job Lifecycle Manager** (Day 1)
   - Implement status transition logic
   - Add validation and error handling
   - Create retry and cleanup mechanisms

2. **Add Background Monitor Service** (Day 1-2)
   - Implement job monitoring loop
   - Add timeout detection
   - Create periodic cleanup

3. **Enhance Ingestion Task Completion** (Day 2)
   - Add remote processing continuation
   - Handle remote job failures
   - Integrate with existing pipeline

4. **Create Lifecycle API Endpoints** (Day 2-3)
   - Add job management endpoints
   - Implement statistics and monitoring
   - Add manual cleanup capabilities

5. **Integration and Testing** (Day 3-4)
   - Test complete job lifecycle
   - Validate error handling
   - Performance testing

## Testing Requirements

### Unit Tests

**File**: `tests/test_job_lifecycle.py`

```python
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from morag.services.job_lifecycle_manager import JobLifecycleManager, JobStatus

class TestJobLifecycleManager:
    def test_status_transitions(self):
        # Test valid and invalid status transitions
        pass

    def test_job_timeout_handling(self):
        # Test timeout detection and handling
        pass

    def test_retry_mechanism(self):
        # Test job retry with exponential backoff
        pass

    def test_cleanup_old_jobs(self):
        # Test cleanup of old completed jobs
        pass

    def test_ingestion_continuation(self):
        # Test continuation of ingestion pipeline
        pass
```

## Success Criteria

1. Job status transitions work correctly with validation
2. Timeout detection and handling work reliably
3. Retry mechanisms function with proper backoff
4. Cleanup services maintain database hygiene
5. Integration with ingestion pipeline is seamless

## Dependencies

- Remote job database schema (Task 2)
- Remote job API endpoints (Task 1)
- Enhanced ingestion tasks (Task 3)
- Existing Celery task system

## Next Steps

After completing this task:
1. Proceed to [Task 6: Testing and Validation](./06-testing-and-validation.md)
2. Set up monitoring and alerting
3. Test complete system integration
4. Prepare for production deployment
