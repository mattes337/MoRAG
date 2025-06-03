from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Optional
import os

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
    
    # File Storage
    upload_dir: str = "./uploads"
    temp_dir: str = "./temp"
    max_file_size: str = "100MB"
    
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

    # Webhook Configuration
    webhook_timeout: int = 30
    webhook_max_retries: int = 3
    webhook_retry_delay: int = 5
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False
    )

    def validate_gemini_config(self) -> None:
        """Validate Gemini API configuration."""
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")

        if not self.gemini_api_key.startswith("AI"):
            raise ValueError("Invalid Gemini API key format - should start with 'AI'")

settings = Settings()
