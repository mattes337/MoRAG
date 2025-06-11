"""Job management service."""

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from datetime import datetime, timezone
import structlog

from morag_core.database import Job, get_database_manager, get_session_context
from morag_core.database.models import JobStatus as DBJobStatus
from .models import (
    JobCreate, JobUpdate, JobResponse, JobSearchRequest,
    JobSearchResponse, JobStatsResponse, JobProgressUpdate,
    JobStatus
)
from morag_core.exceptions import NotFoundError, ValidationError, DatabaseError

logger = structlog.get_logger(__name__)


# Mapping between new JobStatus and existing JobStatus (they should be the same)
STATUS_MAPPING = {
    JobStatus.PENDING: DBJobStatus.PENDING,
    JobStatus.WAITING_FOR_REMOTE_WORKER: DBJobStatus.WAITING_FOR_REMOTE_WORKER,
    JobStatus.PROCESSING: DBJobStatus.PROCESSING,
    JobStatus.FINISHED: DBJobStatus.FINISHED,
    JobStatus.FAILED: DBJobStatus.FAILED,
    JobStatus.CANCELLED: DBJobStatus.CANCELLED,
}

# Reverse mapping
DB_STATUS_TO_STATUS = {v: k for k, v in STATUS_MAPPING.items()}


class JobService:
    """Job management service."""
    
    def __init__(self, database_url: Optional[str] = None):
        self.db_manager = get_database_manager(database_url)
    
    def create_job(self, job_data: JobCreate, user_id: str) -> JobResponse:
        """Create a new job record."""
        try:
            with get_session_context(self.db_manager) as session:
                job = Job(
                    document_name=job_data.document_name,
                    document_type=job_data.document_type,
                    status=DBJobStatus.PENDING,
                    percentage=0,
                    summary=job_data.summary or "",
                    document_id=job_data.document_id,
                    user_id=user_id
                )
                
                session.add(job)
                session.flush()  # Get job ID
                
                logger.info("Job created", 
                           job_id=job.id, 
                           document_name=job.document_name,
                           document_type=job.document_type,
                           user_id=user_id)
                
                return self._job_to_response(job)
                
        except Exception as e:
            logger.error("Failed to create job", error=str(e))
            raise DatabaseError(f"Failed to create job: {str(e)}")
    
    def get_job(self, job_id: str, user_id: Optional[str] = None) -> Optional[JobResponse]:
        """Get job by ID."""
        try:
            with get_session_context(self.db_manager) as session:
                query = session.query(Job).filter_by(id=job_id)
                
                # Add user filter if provided (for non-admin users)
                if user_id:
                    query = query.filter_by(user_id=user_id)
                
                job = query.first()
                if job:
                    return self._job_to_response(job)
                return None
                
        except Exception as e:
            logger.error("Failed to get job", job_id=job_id, error=str(e))
            return None
    
    def update_job(self, job_id: str, job_data: JobUpdate, user_id: Optional[str] = None) -> JobResponse:
        """Update job information."""
        try:
            with get_session_context(self.db_manager) as session:
                query = session.query(Job).filter_by(id=job_id)
                
                # Add user filter if provided (for non-admin users)
                if user_id:
                    query = query.filter_by(user_id=user_id)
                
                job = query.first()
                if not job:
                    raise NotFoundError(f"Job {job_id} not found")
                
                # Update fields
                if job_data.status is not None:
                    job.status = STATUS_MAPPING.get(job_data.status, DBJobStatus.PENDING)
                    # Set end_date when job finishes
                    if job_data.status in [JobStatus.FINISHED, JobStatus.FAILED, JobStatus.CANCELLED]:
                        job.end_date = datetime.now(timezone.utc)
                
                if job_data.percentage is not None:
                    job.percentage = job_data.percentage
                
                if job_data.summary is not None:
                    job.summary = job_data.summary
                
                if job_data.end_date is not None:
                    job.end_date = job_data.end_date
                
                logger.info("Job updated", job_id=job_id, status=job.status.value if job_data.status else None)
                return self._job_to_response(job)
                
        except NotFoundError:
            raise
        except Exception as e:
            logger.error("Failed to update job", job_id=job_id, error=str(e))
            raise DatabaseError(f"Failed to update job: {str(e)}")
    
    def update_job_progress(self, progress_update: JobProgressUpdate, user_id: Optional[str] = None) -> JobResponse:
        """Update job progress."""
        job_update = JobUpdate(
            percentage=progress_update.percentage,
            status=progress_update.status,
            summary=progress_update.summary
        )
        return self.update_job(progress_update.job_id, job_update, user_id)
    
    def delete_job(self, job_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a job."""
        try:
            with get_session_context(self.db_manager) as session:
                query = session.query(Job).filter_by(id=job_id)
                
                # Add user filter if provided (for non-admin users)
                if user_id:
                    query = query.filter_by(user_id=user_id)
                
                job = query.first()
                if not job:
                    return False
                
                session.delete(job)
                logger.info("Job deleted", job_id=job_id)
                return True
                
        except Exception as e:
            logger.error("Failed to delete job", job_id=job_id, error=str(e))
            return False
    
    def search_jobs(self, search_request: JobSearchRequest, user_id: Optional[str] = None) -> JobSearchResponse:
        """Search jobs with filters."""
        try:
            with get_session_context(self.db_manager) as session:
                query = session.query(Job)
                
                # Add user filter if provided (for non-admin users)
                if user_id:
                    query = query.filter_by(user_id=user_id)
                elif search_request.user_id:
                    # Admin can search by specific user
                    query = query.filter_by(user_id=search_request.user_id)
                
                # Apply filters
                if search_request.status:
                    db_status = STATUS_MAPPING.get(search_request.status, DBJobStatus.PENDING)
                    query = query.filter_by(status=db_status)
                
                if search_request.document_type:
                    query = query.filter_by(document_type=search_request.document_type)
                
                if search_request.document_id:
                    query = query.filter_by(document_id=search_request.document_id)
                
                if search_request.created_after:
                    query = query.filter(Job.created_at >= search_request.created_after)
                
                if search_request.created_before:
                    query = query.filter(Job.created_at <= search_request.created_before)
                
                # Get total count before pagination
                total_count = query.count()
                
                # Apply sorting
                sort_column = getattr(Job, search_request.sort_by, Job.created_at)
                if search_request.sort_order == "desc":
                    query = query.order_by(desc(sort_column))
                else:
                    query = query.order_by(asc(sort_column))
                
                # Apply pagination
                jobs = query.offset(search_request.offset).limit(search_request.limit).all()
                
                return JobSearchResponse(
                    jobs=[self._job_to_response(job) for job in jobs],
                    total_count=total_count,
                    limit=search_request.limit,
                    offset=search_request.offset,
                    has_more=search_request.offset + len(jobs) < total_count
                )
                
        except Exception as e:
            logger.error("Failed to search jobs", error=str(e))
            raise DatabaseError(f"Failed to search jobs: {str(e)}")
    
    def get_job_stats(self, user_id: Optional[str] = None) -> JobStatsResponse:
        """Get job statistics."""
        try:
            with get_session_context(self.db_manager) as session:
                query = session.query(Job)
                
                # Add user filter if provided
                if user_id:
                    query = query.filter_by(user_id=user_id)
                
                jobs = query.all()
                
                # Calculate statistics
                total_jobs = len(jobs)
                by_status = {}
                by_document_type = {}
                processing_times = []
                active_jobs = 0
                failed_jobs = 0
                finished_jobs = 0
                
                for job in jobs:
                    # Status stats
                    status_key = job.status.value
                    by_status[status_key] = by_status.get(status_key, 0) + 1
                    
                    # Document type stats
                    type_key = job.document_type
                    by_document_type[type_key] = by_document_type.get(type_key, 0) + 1
                    
                    # Processing time calculation
                    if job.end_date and job.start_date:
                        processing_time = (job.end_date - job.start_date).total_seconds() / 60  # minutes
                        processing_times.append(processing_time)
                    
                    # Count active, failed, and finished jobs
                    if job.status in [DBJobStatus.PENDING, DBJobStatus.PROCESSING, DBJobStatus.WAITING_FOR_REMOTE_WORKER]:
                        active_jobs += 1
                    elif job.status == DBJobStatus.FAILED:
                        failed_jobs += 1
                    elif job.status == DBJobStatus.FINISHED:
                        finished_jobs += 1
                
                # Calculate averages
                avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0.0
                success_rate = (finished_jobs / total_jobs * 100) if total_jobs > 0 else 0.0
                
                return JobStatsResponse(
                    total_jobs=total_jobs,
                    by_status=by_status,
                    by_document_type=by_document_type,
                    average_processing_time_minutes=avg_processing_time,
                    success_rate=success_rate,
                    active_jobs=active_jobs,
                    failed_jobs=failed_jobs
                )
                
        except Exception as e:
            logger.error("Failed to get job stats", error=str(e))
            raise DatabaseError(f"Failed to get job stats: {str(e)}")
    
    def get_active_jobs(self, user_id: Optional[str] = None) -> List[JobResponse]:
        """Get all active jobs."""
        search_request = JobSearchRequest(
            status=JobStatus.PROCESSING,
            limit=1000  # Get all active jobs
        )
        
        # Also get pending and waiting jobs
        active_statuses = [JobStatus.PENDING, JobStatus.PROCESSING, JobStatus.WAITING_FOR_REMOTE_WORKER]
        all_active_jobs = []
        
        for status in active_statuses:
            search_request.status = status
            result = self.search_jobs(search_request, user_id)
            all_active_jobs.extend(result.jobs)
        
        return all_active_jobs
    
    def cancel_job(self, job_id: str, user_id: Optional[str] = None) -> bool:
        """Cancel a job."""
        try:
            job_update = JobUpdate(
                status=JobStatus.CANCELLED,
                end_date=datetime.now(timezone.utc)
            )
            self.update_job(job_id, job_update, user_id)
            logger.info("Job cancelled", job_id=job_id)
            return True
        except Exception as e:
            logger.error("Failed to cancel job", job_id=job_id, error=str(e))
            return False
    
    def _job_to_response(self, job: Job) -> JobResponse:
        """Convert Job model to JobResponse."""
        return JobResponse(
            id=job.id,
            document_name=job.document_name,
            document_type=job.document_type,
            start_date=job.start_date,
            end_date=job.end_date,
            status=DB_STATUS_TO_STATUS.get(job.status, JobStatus.PENDING),
            percentage=job.percentage,
            summary=job.summary,
            document_id=job.document_id,
            user_id=job.user_id,
            created_at=job.created_at,
            updated_at=job.updated_at
        )
