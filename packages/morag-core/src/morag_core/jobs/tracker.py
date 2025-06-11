"""Job tracking and progress management."""

from typing import Optional, Dict, Any, Callable
from datetime import datetime, timezone
import structlog

from .service import JobService
from .models import JobCreate, JobUpdate, JobProgressUpdate, JobStatus
from morag_core.exceptions import NotFoundError

logger = structlog.get_logger(__name__)


class JobTracker:
    """Tracks job progress and manages job lifecycle."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.job_service = JobService(database_url)
        self._progress_callbacks: Dict[str, Callable] = {}
    
    def start_job(
        self,
        document_name: str,
        document_type: str,
        document_id: str,
        user_id: str,
        summary: Optional[str] = None
    ) -> str:
        """Start tracking a new job."""
        try:
            job_data = JobCreate(
                document_name=document_name,
                document_type=document_type,
                document_id=document_id,
                summary=summary or f"Processing {document_type} document: {document_name}"
            )
            
            job = self.job_service.create_job(job_data, user_id)
            
            logger.info("Job tracking started",
                       job_id=job.id,
                       document_name=document_name,
                       document_type=document_type,
                       user_id=user_id)
            
            return job.id
            
        except Exception as e:
            logger.error("Failed to start job tracking", 
                        document_name=document_name, error=str(e))
            raise
    
    def update_progress(
        self,
        job_id: str,
        percentage: int,
        status: Optional[JobStatus] = None,
        summary: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """Update job progress."""
        try:
            progress_update = JobProgressUpdate(
                job_id=job_id,
                percentage=percentage,
                status=status,
                summary=summary
            )
            
            self.job_service.update_job_progress(progress_update, user_id)
            
            # Call progress callback if registered
            if job_id in self._progress_callbacks:
                try:
                    self._progress_callbacks[job_id](job_id, percentage, status, summary)
                except Exception as e:
                    logger.warning("Progress callback failed", job_id=job_id, error=str(e))
            
            logger.debug("Job progress updated",
                        job_id=job_id,
                        percentage=percentage,
                        status=status.value if status else None)
            
            return True
            
        except Exception as e:
            logger.error("Failed to update job progress",
                        job_id=job_id,
                        percentage=percentage,
                        error=str(e))
            return False
    
    def mark_processing(self, job_id: str, user_id: Optional[str] = None) -> bool:
        """Mark job as processing."""
        return self.update_progress(job_id, 10, JobStatus.PROCESSING, "Processing started", user_id)
    
    def mark_completed(
        self,
        job_id: str,
        summary: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """Mark job as completed."""
        completion_summary = summary or "Processing completed successfully"
        return self.update_progress(job_id, 100, JobStatus.FINISHED, completion_summary, user_id)
    
    def mark_failed(
        self,
        job_id: str,
        error_message: str,
        user_id: Optional[str] = None
    ) -> bool:
        """Mark job as failed."""
        failure_summary = f"Processing failed: {error_message}"
        return self.update_progress(job_id, 0, JobStatus.FAILED, failure_summary, user_id)
    
    def mark_cancelled(self, job_id: str, user_id: Optional[str] = None) -> bool:
        """Mark job as cancelled."""
        return self.update_progress(job_id, 0, JobStatus.CANCELLED, "Job cancelled by user", user_id)
    
    def mark_waiting_for_remote(self, job_id: str, user_id: Optional[str] = None) -> bool:
        """Mark job as waiting for remote worker."""
        return self.update_progress(
            job_id, 5, JobStatus.WAITING_FOR_REMOTE_WORKER, 
            "Waiting for remote worker", user_id
        )
    
    def get_job_status(self, job_id: str, user_id: Optional[str] = None) -> Optional[JobStatus]:
        """Get current job status."""
        job = self.job_service.get_job(job_id, user_id)
        if job:
            return job.status
        return None
    
    def get_job_progress(self, job_id: str, user_id: Optional[str] = None) -> Optional[int]:
        """Get current job progress percentage."""
        job = self.job_service.get_job(job_id, user_id)
        if job:
            return job.percentage
        return None
    
    def register_progress_callback(self, job_id: str, callback: Callable) -> None:
        """Register a callback function for progress updates."""
        self._progress_callbacks[job_id] = callback
        logger.debug("Progress callback registered", job_id=job_id)
    
    def unregister_progress_callback(self, job_id: str) -> None:
        """Unregister progress callback."""
        if job_id in self._progress_callbacks:
            del self._progress_callbacks[job_id]
            logger.debug("Progress callback unregistered", job_id=job_id)
    
    def cleanup_completed_jobs(self, older_than_hours: int = 24, user_id: Optional[str] = None) -> int:
        """Clean up completed jobs older than specified hours."""
        try:
            from datetime import timedelta
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
            
            # Search for completed jobs
            from .models import JobSearchRequest
            search_request = JobSearchRequest(
                status=JobStatus.FINISHED,
                created_before=cutoff_time,
                limit=1000  # Process in batches
            )
            
            search_result = self.job_service.search_jobs(search_request, user_id)
            
            # Delete completed jobs
            deleted_count = 0
            for job in search_result.jobs:
                if self.job_service.delete_job(job.id, user_id):
                    deleted_count += 1
            
            logger.info("Cleaned up completed jobs",
                       deleted_count=deleted_count,
                       cutoff_hours=older_than_hours)
            
            return deleted_count
            
        except Exception as e:
            logger.error("Failed to cleanup completed jobs", error=str(e))
            return 0
    
    def cancel_job(self, job_id: str, user_id: Optional[str] = None) -> bool:
        """Cancel a job."""
        try:
            success = self.job_service.cancel_job(job_id, user_id)
            if success:
                # Unregister callback if exists
                self.unregister_progress_callback(job_id)
            return success
        except Exception as e:
            logger.error("Failed to cancel job", job_id=job_id, error=str(e))
            return False
    
    def get_active_jobs_count(self, user_id: Optional[str] = None) -> int:
        """Get count of active jobs."""
        try:
            active_jobs = self.job_service.get_active_jobs(user_id)
            return len(active_jobs)
        except Exception as e:
            logger.error("Failed to get active jobs count", error=str(e))
            return 0
    
    def create_job_context(self, job_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Create job context for task processing."""
        job = self.job_service.get_job(job_id, user_id)
        if not job:
            return {}
        
        return {
            "job_id": job.id,
            "document_id": job.document_id,
            "document_name": job.document_name,
            "document_type": job.document_type,
            "user_id": job.user_id,
            "start_date": job.start_date.isoformat(),
            "status": job.status.value
        }
