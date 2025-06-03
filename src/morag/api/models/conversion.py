"""Pydantic models for conversion API endpoints."""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from enum import Enum


class ChunkingStrategyEnum(str, Enum):
    """Supported chunking strategies."""
    PAGE = "page"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    SEMANTIC = "semantic"


class ConversionRequest(BaseModel):
    """Request model for document conversion."""
    
    chunking_strategy: ChunkingStrategyEnum = Field(
        default=ChunkingStrategyEnum.PAGE,
        description="Strategy for chunking the converted content"
    )
    preserve_formatting: bool = Field(
        default=True,
        description="Whether to preserve original formatting"
    )
    extract_images: bool = Field(
        default=True,
        description="Whether to extract and process images"
    )
    include_metadata: bool = Field(
        default=True,
        description="Whether to include document metadata"
    )
    min_quality_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum quality threshold for conversion"
    )
    enable_fallback: bool = Field(
        default=True,
        description="Whether to enable fallback converters"
    )
    generate_embeddings: bool = Field(
        default=True,
        description="Whether to generate embeddings for chunks"
    )
    format_options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Format-specific conversion options"
    )


class QualityScoreModel(BaseModel):
    """Quality score model."""
    
    overall_score: float = Field(ge=0.0, le=1.0)
    completeness_score: float = Field(ge=0.0, le=1.0)
    readability_score: float = Field(ge=0.0, le=1.0)
    structure_score: float = Field(ge=0.0, le=1.0)
    metadata_preservation: float = Field(ge=0.0, le=1.0)


class ConversionResponse(BaseModel):
    """Response model for document conversion."""
    
    success: bool = Field(description="Whether conversion was successful")
    content: str = Field(description="Converted markdown content")
    metadata: Dict[str, Any] = Field(description="Document metadata")
    quality_score: Optional[QualityScoreModel] = Field(
        description="Quality assessment of conversion"
    )
    processing_time: float = Field(description="Processing time in seconds")
    chunks_count: int = Field(description="Number of chunks generated")
    embeddings_count: int = Field(description="Number of embeddings generated")
    warnings: List[str] = Field(default_factory=list, description="Conversion warnings")
    error_message: Optional[str] = Field(description="Error message if conversion failed")
    converter_used: Optional[str] = Field(description="Name of converter used")
    fallback_used: bool = Field(default=False, description="Whether fallback converter was used")
    original_format: Optional[str] = Field(description="Detected original format")
    word_count: int = Field(description="Word count of converted content")


class ConversionStatus(str, Enum):
    """Conversion status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BatchConversionRequest(BaseModel):
    """Request model for batch document conversion."""
    
    conversion_options: ConversionRequest = Field(
        default_factory=ConversionRequest,
        description="Conversion options to apply to all files"
    )
    max_concurrent: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum number of concurrent conversions"
    )


class BatchConversionResponse(BaseModel):
    """Response model for batch document conversion."""
    
    batch_id: str = Field(description="Unique batch identifier")
    total_files: int = Field(description="Total number of files in batch")
    successful_conversions: int = Field(description="Number of successful conversions")
    failed_conversions: int = Field(description="Number of failed conversions")
    total_processing_time: float = Field(description="Total processing time in seconds")
    average_quality_score: float = Field(description="Average quality score across successful conversions")
    results: List[ConversionResponse] = Field(description="Individual conversion results")


class SupportedFormatsResponse(BaseModel):
    """Response model for supported formats."""
    
    supported_formats: List[str] = Field(description="List of supported document formats")
    converter_info: Dict[str, Dict[str, Any]] = Field(
        description="Information about registered converters"
    )


class ConverterInfoResponse(BaseModel):
    """Response model for converter information."""
    
    name: str = Field(description="Converter name")
    supported_formats: List[str] = Field(description="Formats supported by this converter")
    is_primary: bool = Field(description="Whether this is a primary converter")
    capabilities: List[str] = Field(description="Converter capabilities")
    configuration: Dict[str, Any] = Field(description="Converter configuration options")


class ConversionStatistics(BaseModel):
    """Model for conversion statistics."""
    
    total_conversions: int = Field(description="Total number of conversions")
    successful_conversions: int = Field(description="Number of successful conversions")
    failed_conversions: int = Field(description="Number of failed conversions")
    success_rate: float = Field(description="Success rate as percentage")
    average_processing_time: float = Field(description="Average processing time in seconds")
    average_quality_score: float = Field(description="Average quality score")
    conversions_by_format: Dict[str, Dict[str, Any]] = Field(
        description="Statistics broken down by format"
    )


class ConversionHealthResponse(BaseModel):
    """Response model for conversion service health check."""
    
    status: str = Field(description="Service status (healthy/unhealthy)")
    supported_formats_count: int = Field(description="Number of supported formats")
    total_conversions: int = Field(description="Total conversions processed")
    success_rate: float = Field(description="Success rate as percentage")
    error: Optional[str] = Field(description="Error message if unhealthy")


class ConversionConfigRequest(BaseModel):
    """Request model for updating conversion configuration."""
    
    default_options: Optional[Dict[str, Any]] = Field(
        description="Default conversion options"
    )
    format_specific: Optional[Dict[str, Dict[str, Any]]] = Field(
        description="Format-specific options"
    )
    quality_settings: Optional[Dict[str, Any]] = Field(
        description="Quality assessment settings"
    )
    performance_settings: Optional[Dict[str, Any]] = Field(
        description="Performance settings"
    )


class ConversionConfigResponse(BaseModel):
    """Response model for conversion configuration."""
    
    default_options: Dict[str, Any] = Field(description="Default conversion options")
    format_specific: Dict[str, Dict[str, Any]] = Field(description="Format-specific options")
    quality_settings: Dict[str, Any] = Field(description="Quality assessment settings")
    performance_settings: Dict[str, Any] = Field(description="Performance settings")


class ConversionJobRequest(BaseModel):
    """Request model for asynchronous conversion job."""
    
    file_url: Optional[str] = Field(description="URL to download file from")
    conversion_options: ConversionRequest = Field(
        default_factory=ConversionRequest,
        description="Conversion options"
    )
    callback_url: Optional[str] = Field(
        description="URL to POST results to when complete"
    )
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Job priority (1=highest, 10=lowest)"
    )


class ConversionJobResponse(BaseModel):
    """Response model for conversion job status."""
    
    job_id: str = Field(description="Unique job identifier")
    status: ConversionStatus = Field(description="Current job status")
    created_at: str = Field(description="Job creation timestamp")
    started_at: Optional[str] = Field(description="Job start timestamp")
    completed_at: Optional[str] = Field(description="Job completion timestamp")
    progress: float = Field(
        ge=0.0,
        le=100.0,
        description="Job progress percentage"
    )
    result: Optional[ConversionResponse] = Field(
        description="Conversion result if completed"
    )
    error_message: Optional[str] = Field(
        description="Error message if job failed"
    )


# Validation functions
def validate_quality_threshold(cls, v):
    """Validate quality threshold is between 0 and 1."""
    if not 0.0 <= v <= 1.0:
        raise ValueError('Quality threshold must be between 0.0 and 1.0')
    return v


def validate_format_options(cls, v):
    """Validate format options structure."""
    if v is not None and not isinstance(v, dict):
        raise ValueError('Format options must be a dictionary')
    return v


# Add validators to models
ConversionRequest.validator('min_quality_threshold')(validate_quality_threshold)
ConversionRequest.validator('format_options')(validate_format_options)
