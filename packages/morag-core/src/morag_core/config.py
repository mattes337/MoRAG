"""Configuration management for MoRAG."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator, Field
from typing import List, Optional, Union
import os
import json
import structlog

logger = structlog.get_logger(__name__)

def detect_device() -> str:
    """Detect the best available device (CPU/GPU) with fallback to CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            logger.info("GPU (CUDA) detected and available", device=device)
            return device
    except ImportError:
        logger.debug("PyTorch not available, using CPU")
    except Exception as e:
        logger.warning("GPU detection failed, falling back to CPU", error=str(e))

    logger.info("Using CPU device", device="cpu")
    return "cpu"

def get_safe_device(preferred_device: Optional[str] = None) -> str:
    """Get a safe device with automatic fallback to CPU if GPU is not available."""
    if preferred_device == "cpu":
        return "cpu"

    if preferred_device == "cuda" or preferred_device == "gpu":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                logger.warning("CUDA requested but not available, falling back to CPU")
                return "cpu"
        except ImportError:
            logger.warning("PyTorch not available for CUDA, falling back to CPU")
            return "cpu"
        except Exception as e:
            logger.warning("GPU check failed, falling back to CPU", error=str(e))
            return "cpu"

    # Auto-detect if no preference specified
    return detect_device()


def validate_chunk_size(chunk_size: int, content: str = "") -> tuple[bool, str]:
    """Validate chunk size against content and model limits."""

    # Estimate token count (rough approximation: 1 token ≈ 4 characters)
    if content:
        estimated_tokens = len(content) // 4
        if estimated_tokens > 8000:  # Conservative token limit
            return False, f"Chunk too large: ~{estimated_tokens} tokens (max: 8000)"

    if chunk_size < 500:
        return False, "Chunk size too small: minimum 500 characters recommended"

    if chunk_size > 16000:
        return False, "Chunk size too large: maximum 16000 characters recommended"

    return True, "Chunk size valid"


class Settings(BaseSettings):
    """Core settings for MoRAG."""
    # API Configuration
    api_host: str = Field(default="0.0.0.0", alias="MORAG_API_HOST")
    api_port: int = Field(default=8000, alias="MORAG_API_PORT")
    api_workers: int = Field(default=4, alias="MORAG_API_WORKERS")
    allowed_origins: List[str] = Field(default=["*"], alias="MORAG_ALLOWED_ORIGINS")

    # Gemini API Configuration
    gemini_api_key: Optional[str] = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-2.0-flash", alias="MORAG_GEMINI_MODEL")
    gemini_generation_model: str = Field(default="gemini-2.0-flash", alias="MORAG_GEMINI_GENERATION_MODEL")
    gemini_embedding_model: str = Field(default="text-embedding-004", alias="MORAG_GEMINI_EMBEDDING_MODEL")
    gemini_vision_model: str = Field(default="gemini-1.5-flash", alias="MORAG_GEMINI_VISION_MODEL")

    # Embedding Configuration
    embedding_batch_size: int = Field(default=100, alias="MORAG_EMBEDDING_BATCH_SIZE")
    enable_batch_embedding: bool = Field(default=True, alias="MORAG_ENABLE_BATCH_EMBEDDING")
    embedding_delay_between_batches: float = Field(default=0.05, alias="MORAG_EMBEDDING_DELAY_BETWEEN_BATCHES")
    rate_limit_per_minute: int = Field(default=200, alias="MORAG_RATE_LIMIT_PER_MINUTE")
    enable_performance_monitoring: bool = Field(default=True, alias="MORAG_ENABLE_PERFORMANCE_MONITORING")

    # LLM Batch Configuration
    llm_batch_size: int = Field(default=10, alias="MORAG_LLM_BATCH_SIZE", ge=1, le=50, description="Number of prompts to batch together for LLM calls")
    enable_llm_batching: bool = Field(default=True, alias="MORAG_ENABLE_LLM_BATCHING", description="Enable batching of LLM requests to reduce API calls")
    llm_batch_delay: float = Field(default=1.0, alias="MORAG_LLM_BATCH_DELAY", ge=0.0, le=10.0, description="Delay between LLM batch requests in seconds")
    llm_max_batch_tokens: int = Field(default=800000, alias="MORAG_LLM_MAX_BATCH_TOKENS", ge=100000, le=1000000, description="Maximum tokens per batch request (considering Gemini's 1M context limit)")
    llm_batch_timeout: int = Field(default=120, alias="MORAG_LLM_BATCH_TIMEOUT", ge=30, le=300, description="Timeout for batch LLM requests in seconds")

    # Redis Configuration
    redis_url: str = Field(default="redis://localhost:6379/0", alias="MORAG_REDIS_URL")

    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection_name: str = Field(default="morag_documents", alias="QDRANT_COLLECTION_NAME", description="Qdrant collection name")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")

    # Performance Monitoring
    slow_query_threshold: float = Field(default=5.0, alias="MORAG_SLOW_QUERY_THRESHOLD")  # seconds
    cpu_threshold: float = Field(default=80.0, alias="MORAG_CPU_THRESHOLD")  # percentage
    memory_threshold: float = Field(default=80.0, alias="MORAG_MEMORY_THRESHOLD")  # percentage
    metrics_enabled: bool = Field(default=True, alias="MORAG_METRICS_ENABLED")

    # Retry Configuration
    retry_indefinitely: bool = Field(default=True, alias="MORAG_RETRY_INDEFINITELY")  # Enable indefinite retries for rate limits
    retry_base_delay: float = Field(default=1.0, alias="MORAG_RETRY_BASE_DELAY")  # Base delay in seconds
    retry_max_delay: float = Field(default=300.0, alias="MORAG_RETRY_MAX_DELAY")  # Maximum delay in seconds (5 minutes)
    retry_exponential_base: float = Field(default=2.0, alias="MORAG_RETRY_EXPONENTIAL_BASE")  # Exponential backoff multiplier
    retry_jitter: bool = Field(default=True, alias="MORAG_RETRY_JITTER")  # Add random jitter to delays

    # Entity Extraction Retry Configuration
    entity_extraction_max_retries: int = Field(default=20, alias="MORAG_ENTITY_EXTRACTION_MAX_RETRIES")  # Max retries for entity extraction
    entity_extraction_retry_base_delay: float = Field(default=1.0, alias="MORAG_ENTITY_EXTRACTION_RETRY_BASE_DELAY")  # Base delay for entity extraction retries
    entity_extraction_retry_max_delay: float = Field(default=300.0, alias="MORAG_ENTITY_EXTRACTION_RETRY_MAX_DELAY")  # Max delay for entity extraction retries

    # Celery Task Configuration
    celery_task_soft_time_limit: int = Field(default=120 * 60, alias="MORAG_CELERY_TASK_SOFT_TIME_LIMIT")  # 2 hours (7200 seconds) - soft limit
    celery_task_time_limit: int = Field(default=150 * 60, alias="MORAG_CELERY_TASK_TIME_LIMIT")  # 2.5 hours (9000 seconds) - hard limit
    celery_worker_prefetch_multiplier: int = Field(default=1, alias="MORAG_CELERY_WORKER_PREFETCH_MULTIPLIER")  # Tasks per worker process
    celery_worker_max_tasks_per_child: int = Field(default=1000, alias="MORAG_CELERY_WORKER_MAX_TASKS_PER_CHILD")  # Max tasks before worker restart

    # File Storage
    upload_dir: str = Field(default="./uploads", alias="MORAG_UPLOAD_DIR")
    temp_dir: str = Field(default="./temp", alias="MORAG_TEMP_DIR")
    max_file_size: str = Field(default="100MB", alias="MORAG_MAX_FILE_SIZE")

    # Upload Size Limits (configurable via environment)
    max_upload_size_bytes: int = Field(default=5 * 1024 * 1024 * 1024, alias="MORAG_MAX_UPLOAD_SIZE_BYTES")  # 5GB default

    # File Size Limits (in bytes) - for specific content types
    max_document_size: int = Field(default=100 * 1024 * 1024, alias="MORAG_MAX_DOCUMENT_SIZE")  # 100MB
    max_audio_size: int = Field(default=2 * 1024 * 1024 * 1024, alias="MORAG_MAX_AUDIO_SIZE")  # 2GB
    max_video_size: int = Field(default=5 * 1024 * 1024 * 1024, alias="MORAG_MAX_VIDEO_SIZE")  # 5GB
    max_image_size: int = Field(default=50 * 1024 * 1024, alias="MORAG_MAX_IMAGE_SIZE")  # 50MB

    # Enhanced Logging Configuration
    log_level: str = Field(default="INFO", alias="MORAG_LOG_LEVEL")
    log_format: str = Field(default="json", alias="MORAG_LOG_FORMAT")  # json or console
    log_file: str = Field(default="./logs/morag.log", alias="MORAG_LOG_FILE")
    log_max_size: str = Field(default="100MB", alias="MORAG_LOG_MAX_SIZE")
    log_backup_count: int = Field(default=5, alias="MORAG_LOG_BACKUP_COUNT")
    log_rotation: str = Field(default="daily", alias="MORAG_LOG_ROTATION")  # daily, weekly, size

    # Environment settings
    environment: str = Field(default="development", alias="MORAG_ENVIRONMENT")  # development, testing, production
    debug: bool = Field(default=True, alias="MORAG_DEBUG")

    # Document Processing Configuration
    default_chunk_size: int = Field(
        default=4000,
        alias="MORAG_DEFAULT_CHUNK_SIZE",
        ge=500,  # Minimum 500 characters
        le=16000,  # Maximum 16000 characters (safe for most models)
        description="Default chunk size for document processing"
    )

    default_chunk_overlap: int = Field(
        default=200,
        alias="MORAG_DEFAULT_CHUNK_OVERLAP",
        ge=0,
        le=1000,
        description="Default overlap between chunks"
    )

    # Token limit validation
    max_tokens_per_chunk: int = Field(
        default=8000,
        alias="MORAG_MAX_TOKENS_PER_CHUNK",
        description="Maximum tokens per chunk for embedding models"
    )

    # Markitdown Configuration
    markitdown_enabled: bool = Field(
        default=True,
        alias="MORAG_MARKITDOWN_ENABLED",
        description="Enable markitdown for document conversion"
    )
    markitdown_use_azure_doc_intel: bool = Field(
        default=False,
        alias="MORAG_MARKITDOWN_USE_AZURE_DOC_INTEL",
        description="Use Azure Document Intelligence with markitdown"
    )
    markitdown_azure_endpoint: Optional[str] = Field(
        default=None,
        alias="MORAG_MARKITDOWN_AZURE_ENDPOINT",
        description="Azure Document Intelligence endpoint URL"
    )
    markitdown_use_llm_image_description: bool = Field(
        default=False,
        alias="MORAG_MARKITDOWN_USE_LLM_IMAGE_DESCRIPTION",
        description="Use LLM for image descriptions in markitdown"
    )
    markitdown_llm_model: str = Field(
        default="gpt-4o",
        alias="MORAG_MARKITDOWN_LLM_MODEL",
        description="LLM model for image descriptions"
    )
    markitdown_enable_plugins: bool = Field(
        default=False,
        alias="MORAG_MARKITDOWN_ENABLE_PLUGINS",
        description="Enable markitdown plugins"
    )

    # Page-based chunking configuration
    default_chunking_strategy: str = Field(
        default="page",
        alias="MORAG_DEFAULT_CHUNKING_STRATEGY",
        description="Default chunking strategy for document processing"
    )

    enable_page_based_chunking: bool = Field(
        default=True,
        alias="MORAG_ENABLE_PAGE_BASED_CHUNKING",
        description="Enable page-based chunking for documents"
    )

    max_page_chunk_size: int = Field(
        default=8000,
        alias="MORAG_MAX_PAGE_CHUNK_SIZE",
        ge=1000,  # Minimum 1000 characters
        le=32000,  # Maximum 32000 characters
        description="Maximum size for page-based chunks"
    )



    openie_enable_entity_linking: bool = Field(
        default=True,
        alias="MORAG_OPENIE_ENABLE_ENTITY_LINKING",
        description="Enable entity linking between OpenIE and spaCy entities"
    )

    openie_enable_predicate_normalization: bool = Field(
        default=True,
        alias="MORAG_OPENIE_ENABLE_PREDICATE_NORMALIZATION",
        description="Enable predicate normalization for consistent relationships"
    )

    openie_batch_size: int = Field(
        default=100,
        alias="MORAG_OPENIE_BATCH_SIZE",
        ge=1,
        le=1000,
        description="Batch size for OpenIE processing"
    )

    openie_timeout_seconds: int = Field(
        default=30,
        alias="MORAG_OPENIE_TIMEOUT_SECONDS",
        ge=5,
        le=300,
        description="Timeout for OpenIE processing in seconds"
    )

    @field_validator('allowed_origins', mode='before')
    @classmethod
    def parse_allowed_origins(cls, v):
        """Parse allowed_origins from string or list."""
        if isinstance(v, str):
            try:
                # Try to parse as JSON
                return json.loads(v)
            except json.JSONDecodeError:
                # If not JSON, treat as single origin
                return [v] if v else ["*"]
        return v or ["*"]

    @field_validator('qdrant_collection_name')
    @classmethod
    def validate_collection_name(cls, v):
        """Validate that collection name is provided."""
        if not v:
            logger.warning("QDRANT_COLLECTION_NAME not set, using default 'morag_documents'")
            return "morag_documents"
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    def get_max_upload_size_bytes(self) -> int:
        """Get the maximum upload size in bytes.

        This method first checks MORAG_MAX_UPLOAD_SIZE_BYTES, then falls back to
        parsing MORAG_MAX_FILE_SIZE if the bytes setting is not configured.

        Returns:
            Maximum upload size in bytes
        """
        # If max_upload_size_bytes is explicitly set and not the default, use it
        if hasattr(self, '_max_upload_size_bytes_set'):
            return self.max_upload_size_bytes

        # Check if MORAG_MAX_UPLOAD_SIZE_BYTES was set via environment
        import os
        if os.getenv("MORAG_MAX_UPLOAD_SIZE_BYTES"):
            return self.max_upload_size_bytes

        # Fall back to parsing max_file_size string
        try:
            # Simple size parsing to avoid circular imports
            size_str = self.max_file_size.upper()
            if size_str.endswith('MB'):
                return int(float(size_str[:-2]) * 1024 * 1024)
            elif size_str.endswith('GB'):
                return int(float(size_str[:-2]) * 1024 * 1024 * 1024)
            elif size_str.endswith('KB'):
                return int(float(size_str[:-2]) * 1024)
            else:
                return int(size_str)
        except Exception:
            # If parsing fails, return the default bytes value
            return self.max_upload_size_bytes

# Global settings instance - lazy loaded
_settings_instance = None

def get_settings() -> Settings:
    """Get the global settings instance, creating it if necessary."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance


def log_configuration_debug(settings: Settings) -> None:
    """Log current configuration values for debugging.

    Args:
        settings: Settings instance to log
    """
    logger.info("=== MoRAG Document Processing Configuration ===")
    logger.info("Chunking Configuration:",
                default_chunk_size=settings.default_chunk_size,
                max_page_chunk_size=settings.max_page_chunk_size,
                enable_page_based_chunking=settings.enable_page_based_chunking,
                default_chunking_strategy=settings.default_chunking_strategy,
                default_chunk_overlap=settings.default_chunk_overlap,
                max_tokens_per_chunk=settings.max_tokens_per_chunk)
    logger.info("Environment Variables Check:",
                MORAG_DEFAULT_CHUNK_SIZE=os.getenv("MORAG_DEFAULT_CHUNK_SIZE", "NOT SET"),
                MORAG_MAX_PAGE_CHUNK_SIZE=os.getenv("MORAG_MAX_PAGE_CHUNK_SIZE", "NOT SET"),
                MORAG_ENABLE_PAGE_BASED_CHUNKING=os.getenv("MORAG_ENABLE_PAGE_BASED_CHUNKING", "NOT SET"),
                MORAG_DEFAULT_CHUNKING_STRATEGY=os.getenv("MORAG_DEFAULT_CHUNKING_STRATEGY", "NOT SET"))
    logger.info("=== End Configuration Debug ===")


def validate_configuration_and_log() -> Settings:
    """Validate configuration and log debug information.

    Returns:
        Validated settings instance
    """
    settings = get_settings()
    log_configuration_debug(settings)
    return settings


def reset_settings() -> None:
    """Reset the global settings instance to force reload from environment."""
    global _settings_instance
    _settings_instance = None

# For backward compatibility, provide a property-like access
class SettingsProxy:
    """Proxy object that provides lazy access to settings."""

    def __getattr__(self, name):
        return getattr(get_settings(), name)

    def __setattr__(self, name, value):
        return setattr(get_settings(), name, value)

    def __dir__(self):
        return dir(get_settings())

# Global settings proxy for backward compatibility
settings = SettingsProxy()
