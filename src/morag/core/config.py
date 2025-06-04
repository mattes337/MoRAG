from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os
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
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 4

    # Gemini API
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-pro"
    gemini_embedding_model: str = "text-embedding-004"
    
    # Qdrant Configuration
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "morag_documents"
    qdrant_api_key: Optional[str] = None
    
    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    
    # Task Queue
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    celery_worker_concurrency: int = 4
    
    # File Storage
    upload_dir: str = "./uploads"
    temp_dir: str = "./temp"
    max_file_size: str = "100MB"

    # File Size Limits (in bytes)
    max_document_size: int = 100 * 1024 * 1024  # 100MB
    max_audio_size: int = 2 * 1024 * 1024 * 1024  # 2GB - increased for large audio files
    max_video_size: int = 5 * 1024 * 1024 * 1024  # 5GB
    max_image_size: int = 50 * 1024 * 1024  # 50MB
    
    # Enhanced Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"  # json or console
    log_file: str = "./logs/morag.log"
    log_max_size: str = "100MB"
    log_backup_count: int = 5
    log_rotation: str = "daily"  # daily, weekly, size

    # Monitoring Configuration
    metrics_enabled: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"

    # Performance Monitoring
    enable_profiling: bool = False
    slow_query_threshold: float = 1.0  # seconds
    memory_threshold: int = 80  # percentage
    cpu_threshold: int = 80  # percentage

    # Alerting Configuration
    webhook_alerts_enabled: bool = False
    alert_webhook_url: str = ""
    alert_email_enabled: bool = False
    alert_email_smtp_host: str = ""
    alert_email_smtp_port: int = 587
    alert_email_from: str = ""
    alert_email_to: List[str] = []
    
    # Security
    api_key_header: str = "X-API-Key"
    allowed_origins: List[str] = ["http://localhost:3000"]
    
    # Processing Limits
    max_chunk_size: int = 1000
    max_concurrent_tasks: int = 10

    # Document Processing Configuration
    default_chunking_strategy: str = "page"  # page, semantic, sentence, paragraph, simple
    enable_page_based_chunking: bool = True
    max_page_chunk_size: int = 8000  # Larger size for page-based chunks

    # Device Configuration
    # Automatically detect best available device (cpu/cuda) with CPU fallback
    preferred_device: str = "auto"  # auto, cpu, cuda
    force_cpu: bool = False  # Force CPU usage even if GPU is available

    # Audio Processing Configuration
    # Speaker Diarization
    enable_speaker_diarization: bool = True
    speaker_diarization_model: str = "pyannote/speaker-diarization-3.1"
    min_speakers: int = 1
    max_speakers: int = 10
    speaker_embedding_model: str = "pyannote/embedding"
    huggingface_token: Optional[str] = None

    # Topic Segmentation
    enable_topic_segmentation: bool = True
    topic_similarity_threshold: float = 0.7
    min_topic_sentences: int = 3
    max_topics: int = 10
    topic_embedding_model: str = "all-MiniLM-L6-v2"
    use_llm_topic_summarization: bool = True

    # Audio Quality and Processing
    audio_quality_threshold: float = 0.6
    enable_audio_enhancement: bool = False
    audio_chunk_overlap: float = 0.1  # 10% overlap between chunks

    # Whisper Configuration for Better Quality
    whisper_model_size: str = "large-v3"  # Use large-v3 for best quality
    whisper_beam_size: int = 5  # Increased beam size for better accuracy
    whisper_best_of: int = 5  # Multiple candidates for better results
    whisper_temperature: float = 0.0  # Deterministic output
    whisper_compression_ratio_threshold: float = 2.4
    whisper_log_prob_threshold: float = -1.0
    whisper_no_speech_threshold: float = 0.6

    # Webhook Configuration
    webhook_timeout: int = 30
    webhook_max_retries: int = 3
    webhook_retry_delay: int = 5

    # AI Error Handling Configuration
    ai_retry_max_attempts: int = 3
    ai_retry_base_delay: float = 1.0
    ai_retry_max_delay: float = 60.0
    ai_retry_exponential_base: float = 2.0
    ai_retry_jitter: bool = True

    # Circuit Breaker Configuration
    ai_circuit_breaker_failure_threshold: int = 5
    ai_circuit_breaker_recovery_timeout: float = 60.0
    ai_circuit_breaker_half_open_max_calls: int = 3

    # Service-specific timeouts
    gemini_timeout: float = 30.0
    whisper_timeout: float = 300.0  # 5 minutes for audio processing
    vision_timeout: float = 60.0

    # Health monitoring
    ai_health_window_size: int = 100
    ai_health_check_interval: int = 60  # seconds

    # Web Scraping Configuration
    enable_dynamic_web_scraping: bool = True
    web_scraping_fallback_only: bool = False
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False
    )

    def get_device(self) -> str:
        """Get the configured device with automatic fallback to CPU."""
        if self.force_cpu:
            return "cpu"

        if self.preferred_device == "auto":
            return detect_device()
        else:
            return get_safe_device(self.preferred_device)

    def validate_gemini_config(self) -> None:
        """Validate Gemini API configuration."""
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        if not self.gemini_api_key.startswith("AI"):
            raise ValueError("Invalid Gemini API key format - should start with 'AI'")

settings = Settings()
