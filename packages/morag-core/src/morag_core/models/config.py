"""Configuration models for MoRAG Core."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ProcessingConfig(BaseModel):
    """Base processing configuration."""
    enabled: bool = Field(default=True, description="Whether processing is enabled")
    timeout_seconds: Optional[int] = Field(default=None, description="Processing timeout")
    max_retries: int = Field(default=3, description="Maximum retry attempts")


class DocumentProcessingConfig(ProcessingConfig):
    """Configuration for document processing."""
    extract_images: bool = Field(default=False, description="Extract images from documents")
    preserve_formatting: bool = Field(default=True, description="Preserve document formatting")
    chunk_size: int = Field(default=4000, description="Default chunk size")


class AudioProcessingConfig(ProcessingConfig):
    """Configuration for audio processing."""
    transcription_model: str = Field(default="whisper-large", description="Transcription model")
    speaker_diarization: bool = Field(default=True, description="Enable speaker diarization")
    include_timestamps: bool = Field(default=True, description="Include timestamps")


class VideoProcessingConfig(ProcessingConfig):
    """Configuration for video processing."""
    extract_audio: bool = Field(default=True, description="Extract audio for transcription")
    generate_thumbnails: bool = Field(default=False, description="Generate video thumbnails")
    thumbnail_interval: int = Field(default=60, description="Thumbnail interval in seconds")


class ImageProcessingConfig(ProcessingConfig):
    """Configuration for image processing."""
    extract_text: bool = Field(default=True, description="Extract text from images")
    generate_descriptions: bool = Field(default=False, description="Generate image descriptions")


class WebProcessingConfig(ProcessingConfig):
    """Configuration for web content processing."""
    follow_links: bool = Field(default=False, description="Follow links on pages")
    max_depth: int = Field(default=1, description="Maximum crawl depth")
    respect_robots: bool = Field(default=True, description="Respect robots.txt")


class EmbeddingConfig(BaseModel):
    """Configuration for embedding generation."""
    model: str = Field(default="text-embedding-004", description="Embedding model")
    batch_size: int = Field(default=100, description="Batch size for embedding generation")
    dimensions: Optional[int] = Field(default=None, description="Embedding dimensions")


class ProcessingResult(BaseModel):
    """Result of content processing."""
    success: bool = Field(description="Whether processing was successful")
    content: Optional[str] = Field(default=None, description="Processed content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    processing_time: float = Field(default=0.0, description="Processing time in seconds")
