"""Job tracking and management models."""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "PENDING"
    WAITING_FOR_REMOTE_WORKER = "WAITING_FOR_REMOTE_WORKER"
    PROCESSING = "PROCESSING"
    FINISHED = "FINISHED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class JobCreate(BaseModel):
    document_name: str = Field(..., min_length=1, max_length=255)
    document_type: str = Field(..., min_length=1, max_length=50)
    document_id: str
    summary: Optional[str] = ""


class JobUpdate(BaseModel):
    status: Optional[JobStatus] = None
    percentage: Optional[int] = Field(None, ge=0, le=100)
    summary: Optional[str] = None
    end_date: Optional[datetime] = None


class JobResponse(BaseModel):
    id: str
    document_name: str
    document_type: str
    start_date: datetime
    end_date: Optional[datetime]
    status: JobStatus
    percentage: int
    summary: str
    document_id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class JobSearchRequest(BaseModel):
    status: Optional[JobStatus] = None
    document_type: Optional[str] = None
    document_id: Optional[str] = None
    user_id: Optional[str] = None  # Admin only
    created_after: Optional[datetime] = None
    created_before: Optional[datetime] = None
    limit: int = Field(default=50, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)
    sort_by: str = Field(default="created_at")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")


class JobSearchResponse(BaseModel):
    jobs: List[JobResponse]
    total_count: int
    limit: int
    offset: int
    has_more: bool


class JobStatsResponse(BaseModel):
    total_jobs: int
    by_status: Dict[str, int]
    by_document_type: Dict[str, int]
    average_processing_time_minutes: float
    success_rate: float
    active_jobs: int
    failed_jobs: int


class JobProgressUpdate(BaseModel):
    job_id: str
    percentage: int = Field(..., ge=0, le=100)
    status: Optional[JobStatus] = None
    summary: Optional[str] = None
