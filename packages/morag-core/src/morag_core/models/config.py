"""Configuration models for MoRAG."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ProcessingConfig:
    """Base configuration for processing operations."""
    
    # General processing options
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    timeout: float = 300.0  # 5 minutes
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Output options
    output_format: str = "text"
    preserve_formatting: bool = True
    extract_metadata: bool = True
    
    # Quality options
    quality_threshold: float = 0.8
    min_confidence: float = 0.5
    
    # Custom options
    custom_options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.
        
        Returns:
            Dictionary representation
        """
        return {
            "max_file_size": self.max_file_size,
            "timeout": self.timeout,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            "output_format": self.output_format,
            "preserve_formatting": self.preserve_formatting,
            "extract_metadata": self.extract_metadata,
            "quality_threshold": self.quality_threshold,
            "min_confidence": self.min_confidence,
            "custom_options": self.custom_options,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessingConfig":
        """Create from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            ProcessingConfig instance
        """
        return cls(**data)


@dataclass
class DocumentProcessingConfig(ProcessingConfig):
    """Configuration for document processing."""
    
    # Document-specific options
    extract_images: bool = False
    extract_tables: bool = True
    ocr_enabled: bool = True
    page_chunking: bool = True
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # Language detection
    detect_language: bool = True
    target_language: Optional[str] = None
    
    # PDF-specific options
    pdf_password: Optional[str] = None
    pdf_extract_annotations: bool = False


@dataclass
class AudioProcessingConfig(ProcessingConfig):
    """Configuration for audio processing."""
    
    # Audio-specific options
    model_name: str = "base"
    language: Optional[str] = None
    task: str = "transcribe"  # transcribe or translate
    
    # Quality options
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0
    
    # Diarization options
    enable_diarization: bool = True
    min_speakers: int = 1
    max_speakers: int = 10
    
    # Output options
    include_timestamps: bool = True
    word_timestamps: bool = False
    highlight_words: bool = False


@dataclass
class VideoProcessingConfig(ProcessingConfig):
    """Configuration for video processing."""
    
    # Video-specific options
    extract_audio: bool = True
    extract_frames: bool = False
    frame_interval: float = 1.0  # seconds
    
    # Audio processing (inherits from AudioProcessingConfig)
    audio_config: Optional[AudioProcessingConfig] = None
    
    # Video analysis
    scene_detection: bool = False
    object_detection: bool = False
    face_detection: bool = False
    
    # Output options
    video_format: str = "mp4"
    audio_format: str = "wav"
    frame_format: str = "jpg"


@dataclass
class ImageProcessingConfig(ProcessingConfig):
    """Configuration for image processing."""
    
    # OCR options
    ocr_engine: str = "tesseract"  # tesseract, easyocr, paddleocr
    ocr_languages: list = field(default_factory=lambda: ["en"])
    
    # Image preprocessing
    enhance_image: bool = True
    denoise: bool = True
    deskew: bool = True
    
    # Detection options
    detect_text_regions: bool = True
    detect_tables: bool = True
    detect_figures: bool = False
    
    # AI vision options
    use_ai_vision: bool = False
    ai_model: str = "gpt-4-vision"
    describe_image: bool = True


@dataclass
class WebProcessingConfig(ProcessingConfig):
    """Configuration for web processing."""
    
    # Crawling options
    max_depth: int = 1
    max_pages: int = 10
    follow_links: bool = False
    respect_robots_txt: bool = True
    
    # Content extraction
    extract_main_content: bool = True
    remove_navigation: bool = True
    remove_ads: bool = True
    remove_comments: bool = True
    
    # Browser options
    use_browser: bool = False
    wait_for_js: bool = False
    screenshot: bool = False
    
    # Rate limiting
    delay_between_requests: float = 1.0
    max_concurrent_requests: int = 5


@dataclass
class EmbeddingConfig(ProcessingConfig):
    """Configuration for embedding generation."""
    
    # Model options
    model_name: str = "text-embedding-004"
    provider: str = "gemini"  # gemini, openai, huggingface
    
    # Processing options
    batch_size: int = 50
    max_tokens: int = 8192
    normalize: bool = True
    
    # Chunking options
    chunk_text: bool = True
    chunk_size: int = 512
    chunk_overlap: int = 50
    
    # Custom options
    api_key: Optional[str] = None
    api_base: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of a processing operation."""

    # Basic result info
    success: bool
    task_id: str
    source_type: str

    # Content
    content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Processing details
    processing_time: float = 0.0
    model_used: Optional[str] = None
    quality_score: Optional[float] = None

    # Error information
    error_message: Optional[str] = None
    error_type: Optional[str] = None

    # Additional data
    chunks: List[Dict[str, Any]] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)
    summary: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "success": self.success,
            "task_id": self.task_id,
            "source_type": self.source_type,
            "content": self.content,
            "metadata": self.metadata,
            "processing_time": self.processing_time,
            "model_used": self.model_used,
            "quality_score": self.quality_score,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "chunks": self.chunks,
            "embeddings": self.embeddings,
            "summary": self.summary,
        }
