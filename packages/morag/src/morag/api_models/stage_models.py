"""Pydantic models for stage-based API endpoints."""

from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, validator


class StageTypeEnum(str, Enum):
    """Enumeration of available stage types using canonical names."""
    MARKDOWN_CONVERSION = "markdown-conversion"
    MARKDOWN_OPTIMIZER = "markdown-optimizer"
    CHUNKER = "chunker"
    FACT_GENERATOR = "fact-generator"
    INGESTOR = "ingestor"


class StageStatusEnum(str, Enum):
    """Enumeration of stage execution statuses."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WebhookConfig(BaseModel):
    """Configuration for webhook notifications."""
    url: str = Field(..., description="Webhook URL to send notifications")
    auth_token: Optional[str] = Field(None, description="Optional authentication token")
    headers: Optional[Dict[str, str]] = Field(None, description="Optional custom headers")
    retry_count: int = Field(3, description="Number of retry attempts")
    timeout: int = Field(30, description="Timeout in seconds")


class StageExecutionRequest(BaseModel):
    """Request model for executing a single stage."""
    input_files: Optional[List[str]] = Field(None, description="List of input file paths (for non-initial stages)")
    config: Optional[Dict[str, Any]] = Field(None, description="Stage-specific configuration")
    output_dir: str = Field("./output", description="Output directory for stage results")
    webhook_config: Optional[WebhookConfig] = Field(None, description="Webhook notification configuration")
    skip_if_exists: bool = Field(True, description="Skip execution if output files already exist")


class StageFileMetadata(BaseModel):
    """Metadata for stage output files."""
    filename: str = Field(..., description="Name of the output file")
    file_path: str = Field(..., description="Full path to the output file")
    file_size: int = Field(..., description="File size in bytes")
    created_at: datetime = Field(..., description="File creation timestamp")
    stage_type: StageTypeEnum = Field(..., description="Stage that created this file")
    content_type: Optional[str] = Field(None, description="MIME type of the file")
    checksum: Optional[str] = Field(None, description="File checksum for integrity verification")


class StageExecutionMetadata(BaseModel):
    """Metadata about stage execution."""
    execution_time: float = Field(..., description="Execution time in seconds")
    start_time: datetime = Field(..., description="Stage start timestamp")
    end_time: datetime = Field(..., description="Stage end timestamp")
    input_files: List[str] = Field(..., description="List of input files processed")
    config_used: Dict[str, Any] = Field(..., description="Configuration used for execution")
    warnings: List[str] = Field(default_factory=list, description="Any warnings during execution")


class StageExecutionResponse(BaseModel):
    """Response model for stage execution."""
    success: bool = Field(..., description="Whether the stage executed successfully")
    stage_type: StageTypeEnum = Field(..., description="Type of stage executed")
    status: StageStatusEnum = Field(..., description="Final status of the stage")
    output_files: List[StageFileMetadata] = Field(..., description="List of output files created")
    metadata: StageExecutionMetadata = Field(..., description="Execution metadata")
    error_message: Optional[str] = Field(None, description="Error message if execution failed")
    webhook_sent: bool = Field(False, description="Whether webhook notification was sent")


class StageChainRequest(BaseModel):
    """Request model for executing a chain of stages."""
    stages: List[StageTypeEnum] = Field(..., description="List of stages to execute in order")
    input_files: Optional[List[str]] = Field(None, description="Initial input files (for first stage)")
    global_config: Optional[Dict[str, Any]] = Field(None, description="Global configuration for all stages")
    stage_configs: Optional[Dict[StageTypeEnum, Dict[str, Any]]] = Field(None, description="Stage-specific configurations")
    output_dir: str = Field("./output", description="Base output directory")
    webhook_config: Optional[WebhookConfig] = Field(None, description="Webhook notification configuration")
    stop_on_failure: bool = Field(True, description="Stop chain execution if any stage fails")
    skip_existing: bool = Field(True, description="Skip stages if their outputs already exist")


class StageChainResponse(BaseModel):
    """Response model for stage chain execution."""
    success: bool = Field(..., description="Whether the entire chain executed successfully")
    stages_executed: List[StageExecutionResponse] = Field(..., description="Results for each stage executed")
    total_execution_time: float = Field(..., description="Total execution time for the chain")
    failed_stage: Optional[StageTypeEnum] = Field(None, description="Stage that caused failure (if any)")
    final_output_files: List[StageFileMetadata] = Field(..., description="Final output files from the last stage")


class StageStatusRequest(BaseModel):
    """Request model for checking stage status."""
    job_id: Optional[str] = Field(None, description="Job ID for background execution")
    output_dir: Optional[str] = Field(None, description="Output directory to check for files")


class StageStatusResponse(BaseModel):
    """Response model for stage status."""
    job_id: Optional[str] = Field(None, description="Job ID if this was a background execution")
    stages_completed: List[StageTypeEnum] = Field(..., description="List of stages that have completed")
    current_stage: Optional[StageTypeEnum] = Field(None, description="Currently executing stage")
    available_files: List[StageFileMetadata] = Field(..., description="Available output files")
    overall_status: StageStatusEnum = Field(..., description="Overall execution status")
    progress_percentage: float = Field(..., description="Progress percentage (0-100)")


class FileDownloadRequest(BaseModel):
    """Request model for file download."""
    file_path: str = Field(..., description="Path to the file to download")
    inline: bool = Field(False, description="Whether to display inline or as attachment")


class FileDownloadResponse(BaseModel):
    """Response model for file download (metadata only)."""
    filename: str = Field(..., description="Name of the downloaded file")
    content_type: str = Field(..., description="MIME type of the file")
    file_size: int = Field(..., description="File size in bytes")
    download_url: str = Field(..., description="URL to download the file")


class FileListRequest(BaseModel):
    """Request model for listing files."""
    output_dir: str = Field("./output", description="Directory to list files from")
    stage_type: Optional[StageTypeEnum] = Field(None, description="Filter by stage type")
    file_extension: Optional[str] = Field(None, description="Filter by file extension")


class FileListResponse(BaseModel):
    """Response model for file listing."""
    files: List[StageFileMetadata] = Field(..., description="List of available files")
    total_count: int = Field(..., description="Total number of files")
    total_size: int = Field(..., description="Total size of all files in bytes")


class StageInfoResponse(BaseModel):
    """Response model for stage information."""
    stage_type: StageTypeEnum = Field(..., description="Stage type")
    description: str = Field(..., description="Stage description")
    input_formats: List[str] = Field(..., description="Supported input file formats")
    output_formats: List[str] = Field(..., description="Output file formats produced")
    required_config: List[str] = Field(..., description="Required configuration parameters")
    optional_config: List[str] = Field(..., description="Optional configuration parameters")
    dependencies: List[StageTypeEnum] = Field(..., description="Stages that must run before this one")


class StageListResponse(BaseModel):
    """Response model for listing all available stages."""
    stages: List[StageInfoResponse] = Field(..., description="List of available stages")
    total_count: int = Field(..., description="Total number of available stages")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


class HealthCheckResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Health check timestamp")
    version: str = Field(..., description="API version")
    stages_available: List[StageTypeEnum] = Field(..., description="Available stages")
    services_status: Dict[str, str] = Field(..., description="Status of underlying services")
