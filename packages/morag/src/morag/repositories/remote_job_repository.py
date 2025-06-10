"""File-based repository for remote job storage."""

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
        try:
            # First ensure the base data directory exists
            self.data_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Remote jobs data directory ensured", path=str(self.data_dir))

            # Create status subdirectories
            status_dirs = ['pending', 'processing', 'completed', 'failed', 'timeout', 'cancelled']
            for status in status_dirs:
                status_path = self.data_dir / status
                status_path.mkdir(parents=True, exist_ok=True)
                logger.debug("Status directory ensured", status=status, path=str(status_path))

        except PermissionError as e:
            logger.error("Permission denied creating remote jobs directories",
                        path=str(self.data_dir), error=str(e))
            raise RuntimeError(f"Cannot create remote jobs directories at {self.data_dir}: {e}")
        except Exception as e:
            logger.error("Failed to create remote jobs directories",
                        path=str(self.data_dir), error=str(e))
            raise

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
