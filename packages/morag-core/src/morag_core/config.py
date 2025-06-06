"""Configuration management for MoRAG."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
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

class Settings(BaseSettings):
    """Core settings for MoRAG."""
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4
    allowed_origins: List[str] = ["*"]

    # Gemini API Configuration
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash"
    gemini_generation_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "text-embedding-004"
    gemini_vision_model: str = "gemini-1.5-flash"

    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "morag_documents"
    qdrant_api_key: Optional[str] = None

    # Performance Monitoring
    slow_query_threshold: float = 5.0  # seconds
    cpu_threshold: float = 80.0  # percentage
    memory_threshold: float = 80.0  # percentage
    metrics_enabled: bool = True

    # Retry Configuration
    retry_indefinitely: bool = True  # Enable indefinite retries for rate limits
    retry_base_delay: float = 1.0  # Base delay in seconds
    retry_max_delay: float = 300.0  # Maximum delay in seconds (5 minutes)
    retry_exponential_base: float = 2.0  # Exponential backoff multiplier
    retry_jitter: bool = True  # Add random jitter to delays

    # Celery Task Configuration
    celery_task_soft_time_limit: int = 120 * 60  # 2 hours (7200 seconds) - soft limit
    celery_task_time_limit: int = 150 * 60  # 2.5 hours (9000 seconds) - hard limit
    celery_worker_prefetch_multiplier: int = 1  # Tasks per worker process
    celery_worker_max_tasks_per_child: int = 1000  # Max tasks before worker restart

    # File Storage
    upload_dir: str = "./uploads"
    temp_dir: str = "./temp"
    max_file_size: str = "100MB"

    # File Size Limits (in bytes)
    max_document_size: int = 100 * 1024 * 1024  # 100MB
    max_audio_size: int = 2 * 1024 * 1024 * 1024  # 2GB
    max_video_size: int = 5 * 1024 * 1024 * 1024  # 5GB
    max_image_size: int = 50 * 1024 * 1024  # 50MB
    
    # Enhanced Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"  # json or console
    log_file: str = "./logs/morag.log"
    log_max_size: str = "100MB"
    log_backup_count: int = 5
    log_rotation: str = "daily"  # daily, weekly, size

    # Environment settings
    environment: str = "development"  # development, testing, production
    debug: bool = True

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

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MORAG_",
        case_sensitive=False,
        extra="ignore"
    )

# Global settings instance
settings = Settings()