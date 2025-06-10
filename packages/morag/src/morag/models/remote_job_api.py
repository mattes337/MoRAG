"""Pydantic models for remote job API endpoints."""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class CreateRemoteJobRequest(BaseModel):
    """Request model for creating a remote job."""
    source_file_path: str = Field(..., description="Path to source file for processing")
    content_type: str = Field(..., description="Content type (audio, video, etc.)")
    task_options: Dict[str, Any] = Field(default_factory=dict, description="Processing options")
    ingestion_task_id: str = Field(..., description="Associated ingestion task ID")


class CreateRemoteJobResponse(BaseModel):
    """Response model for creating a remote job."""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status")
    created_at: datetime = Field(..., description="Job creation timestamp")


class PollJobsRequest(BaseModel):
    """Request model for polling jobs."""
    worker_id: str = Field(..., description="Unique worker identifier")
    content_types: List[str] = Field(..., description="Supported content types")
    max_jobs: int = Field(default=1, description="Maximum jobs to return")


class PollJobsResponse(BaseModel):
    """Response model for polling jobs."""
    job_id: Optional[str] = Field(None, description="Job ID if available")
    source_file_url: Optional[str] = Field(None, description="Download URL for source file")
    content_type: Optional[str] = Field(None, description="Content type")
    task_options: Optional[Dict[str, Any]] = Field(None, description="Processing options")


class SubmitResultRequest(BaseModel):
    """Request model for submitting job results."""
    success: bool = Field(..., description="Whether processing succeeded")
    content: Optional[str] = Field(None, description="Processed content")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Processing metadata")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")


class SubmitResultResponse(BaseModel):
    """Response model for submitting job results."""
    status: str = Field(..., description="Updated job status")
    ingestion_continued: bool = Field(..., description="Whether ingestion pipeline continued")


class JobStatusResponse(BaseModel):
    """Response model for job status."""
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current job status")
    worker_id: Optional[str] = Field(None, description="Assigned worker ID")
    created_at: datetime = Field(..., description="Job creation timestamp")
    started_at: Optional[datetime] = Field(None, description="Processing start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Processing completion timestamp")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    retry_count: int = Field(..., description="Number of retry attempts")
    estimated_completion: Optional[datetime] = Field(None, description="Estimated completion time")
