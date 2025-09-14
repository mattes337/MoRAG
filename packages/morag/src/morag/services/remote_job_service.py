"""Service layer for remote job operations."""

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
