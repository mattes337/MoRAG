"""Remote job API endpoints."""

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from typing import List
import structlog
from datetime import timedelta
from pathlib import Path

from morag.models.remote_job_api import (
    CreateRemoteJobRequest, CreateRemoteJobResponse,
    PollJobsRequest, PollJobsResponse,
    SubmitResultRequest, SubmitResultResponse,
    JobStatusResponse
)
from morag.services.remote_job_service import RemoteJobService
import asyncio

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
    service: RemoteJobService = Depends(get_remote_job_service)
):
    """Submit processing result for a remote job."""
    try:
        job = service.submit_result(job_id, result)

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )



        # Continue ingestion pipeline if successful
        ingestion_continued = False
        if result.success and result.content:
            try:
                # Import here to avoid circular imports
                from morag.ingest_tasks import continue_ingestion_after_remote_processing

                # Continue the ingestion pipeline
                ingestion_continued = await continue_ingestion_after_remote_processing(
                    job_id,
                    result.content,
                    result.metadata or {},
                    result.processing_time or 0.0
                )

            except Exception as e:
                logger.error("Failed to continue ingestion pipeline",
                           job_id=job_id,
                           error=str(e))
                ingestion_continued = False

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
    service: RemoteJobService = Depends(get_remote_job_service)
):
    """Get current status of a remote job."""
    try:
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


@router.get("/{job_id}/download")
async def download_job_file(
    job_id: str,
    service: RemoteJobService = Depends(get_remote_job_service)
):
    """Download source file for a remote job."""
    try:
        job = service.get_job_status(job_id)

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job {job_id} not found"
            )

        # Check if job is assigned to a worker (in processing state)
        if job.status != 'processing':
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Job {job_id} is not in processing state"
            )

        # Check if source file exists
        source_file = Path(job.source_file_path)
        if not source_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Source file not found: {job.source_file_path}"
            )



        return FileResponse(
            path=str(source_file),
            filename=source_file.name,
            media_type='application/octet-stream'
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to download job file", job_id=job_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download job file: {str(e)}"
        )
