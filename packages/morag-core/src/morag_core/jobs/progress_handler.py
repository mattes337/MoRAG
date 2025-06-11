"""Progress event handler for updating job entities based on progress events."""

from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import structlog

from .progress_parser import ProgressEventParser, ProgressEvent
from .tracker import JobTracker
from .models import JobStatus
from morag_core.exceptions import NotFoundError

logger = structlog.get_logger(__name__)


class ProgressHandler:
    """Handles progress events and updates job entities accordingly."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.parser = ProgressEventParser()
        self.job_tracker = JobTracker(database_url)
        self._job_mappings: Dict[str, str] = {}  # Maps worker/task IDs to job IDs
    
    def register_job_mapping(self, worker_id: str, job_id: str):
        """Register a mapping between worker/task ID and job ID."""
        self._job_mappings[worker_id] = job_id
        logger.debug("Job mapping registered", worker_id=worker_id, job_id=job_id)
    
    def unregister_job_mapping(self, worker_id: str):
        """Remove a job mapping."""
        if worker_id in self._job_mappings:
            del self._job_mappings[worker_id]
            logger.debug("Job mapping removed", worker_id=worker_id)
    
    def get_job_id_for_worker(self, worker_id: str) -> Optional[str]:
        """Get the job ID associated with a worker/task ID."""
        return self._job_mappings.get(worker_id)
    
    def process_log_line(self, log_line: str, job_id: Optional[str] = None, worker_id: Optional[str] = None) -> bool:
        """Process a single log line and update job progress if applicable."""
        try:
            # Parse the log line for progress information
            progress_event = self.parser.parse_json_log(log_line)
            if not progress_event:
                return False
            
            # Determine the job ID to update
            target_job_id = job_id
            if not target_job_id and worker_id:
                target_job_id = self.get_job_id_for_worker(worker_id)
            
            if not target_job_id:
                logger.debug("No job ID found for progress event", 
                           worker_id=worker_id, 
                           message=progress_event.message[:50])
                return False
            
            # Update job progress
            return self.update_job_progress(target_job_id, progress_event)
            
        except Exception as e:
            logger.warning("Failed to process log line for progress", 
                         error=str(e), 
                         log_line=log_line[:100])
            return False
    
    def process_log_lines(self, log_lines: List[str], job_id: Optional[str] = None, worker_id: Optional[str] = None) -> int:
        """Process multiple log lines and return the number of progress updates made."""
        updates_made = 0
        for line in log_lines:
            if self.process_log_line(line, job_id, worker_id):
                updates_made += 1
        return updates_made
    
    def update_job_progress(self, job_id: str, progress_event: ProgressEvent, user_id: Optional[str] = None) -> bool:
        """Update job progress based on a progress event."""
        try:
            # Determine job status based on percentage
            status = None
            if progress_event.percentage == 0:
                status = JobStatus.PENDING
            elif 0 < progress_event.percentage < 100:
                status = JobStatus.PROCESSING
            elif progress_event.percentage == 100:
                status = JobStatus.FINISHED

            # Update job progress
            result = self.job_tracker.update_progress(
                job_id=job_id,
                percentage=progress_event.percentage,
                status=status,
                summary=progress_event.message,
                user_id=user_id
            )

            # Ensure we return a boolean
            success = bool(result) if result is not None else True

            if success:
                logger.debug("Job progress updated from event",
                           job_id=job_id,
                           percentage=progress_event.percentage,
                           message=progress_event.message[:50])

            return success

        except Exception as e:
            logger.error("Failed to update job progress from event",
                        job_id=job_id,
                        percentage=progress_event.percentage,
                        error=str(e))
            return False
    
    def process_remote_worker_progress(self, worker_id: str, percentage: int, message: str) -> bool:
        """Process progress update from a remote worker."""
        try:
            # Create a progress event
            progress_event = ProgressEvent(
                percentage=percentage,
                message=message,
                timestamp=datetime.now(timezone.utc),
                logger_name="remote_worker",
                level="info"
            )
            
            # Find the associated job ID
            job_id = self.get_job_id_for_worker(worker_id)
            if not job_id:
                logger.warning("No job mapping found for remote worker", worker_id=worker_id)
                return False
            
            # Update job progress
            return self.update_job_progress(job_id, progress_event)
            
        except Exception as e:
            logger.error("Failed to process remote worker progress",
                        worker_id=worker_id,
                        percentage=percentage,
                        error=str(e))
            return False
    
    def process_celery_task_progress(self, task_id: str, percentage: int, message: str, user_id: Optional[str] = None) -> bool:
        """Process progress update from a Celery task."""
        try:
            # Create a progress event
            progress_event = ProgressEvent(
                percentage=percentage,
                message=message,
                timestamp=datetime.now(timezone.utc),
                logger_name="celery_task",
                level="info"
            )
            
            # For Celery tasks, the task_id is often the job_id
            job_id = self.get_job_id_for_worker(task_id) or task_id
            
            # Update job progress
            return self.update_job_progress(job_id, progress_event, user_id)
            
        except Exception as e:
            logger.error("Failed to process Celery task progress",
                        task_id=task_id,
                        percentage=percentage,
                        error=str(e))
            return False
    
    def extract_latest_progress_from_logs(self, log_lines: List[str]) -> Optional[ProgressEvent]:
        """Extract the latest progress event from a list of log lines."""
        return self.parser.get_latest_progress(log_lines)
    
    def sync_job_with_logs(self, job_id: str, log_lines: List[str], user_id: Optional[str] = None) -> bool:
        """Synchronize job progress with the latest information from log lines."""
        try:
            latest_progress = self.extract_latest_progress_from_logs(log_lines)
            if latest_progress:
                return self.update_job_progress(job_id, latest_progress, user_id)
            return False
        except Exception as e:
            logger.error("Failed to sync job with logs", job_id=job_id, error=str(e))
            return False
    
    def handle_job_completion(self, job_id: str, success: bool, final_message: str, user_id: Optional[str] = None) -> bool:
        """Handle job completion with final status update."""
        try:
            if success:
                result = self.job_tracker.mark_completed(job_id, final_message, user_id)
            else:
                result = self.job_tracker.mark_failed(job_id, final_message, user_id)

            # Ensure we return a boolean
            return bool(result) if result is not None else True

        except Exception as e:
            logger.error("Failed to handle job completion",
                        job_id=job_id,
                        success=success,
                        error=str(e))
            return False
    
    def handle_job_error(self, job_id: str, error_message: str, user_id: Optional[str] = None) -> bool:
        """Handle job error with failure status update."""
        return self.handle_job_completion(job_id, False, f"Job failed: {error_message}", user_id)
