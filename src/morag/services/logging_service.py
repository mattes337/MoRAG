import logging
import logging.handlers
import structlog
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from morag_core.config import settings

class LoggingService:
    """Enhanced logging service with rotation and structured output."""
    
    def __init__(self):
        self.setup_logging()
        self.logger = structlog.get_logger()
    
    def setup_logging(self):
        """Configure structured logging with file rotation."""
        # Create logs directory
        log_dir = Path(settings.log_file).parent
        log_dir.mkdir(exist_ok=True)
        
        # Configure file handler with rotation
        if settings.log_rotation == "size":
            file_handler = logging.handlers.RotatingFileHandler(
                settings.log_file,
                maxBytes=self._parse_size(settings.log_max_size),
                backupCount=settings.log_backup_count
            )
        else:
            # Map rotation settings to valid TimedRotatingFileHandler values
            rotation_map = {
                "daily": "D",
                "weekly": "W0",
                "hourly": "H"
            }
            when = rotation_map.get(settings.log_rotation.lower(), "D")
            file_handler = logging.handlers.TimedRotatingFileHandler(
                settings.log_file,
                when=when,
                backupCount=settings.log_backup_count
            )
        
        # Configure console handler
        console_handler = logging.StreamHandler()
        
        # Set up structlog
        processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        
        if settings.log_format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        structlog.configure(
            processors=processors,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, settings.log_level.upper()))
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string like '100MB' to bytes."""
        size_str = size_str.upper()
        if size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)
    
    def log_request(self, method: str, url: str, status_code: int, 
                   duration: float, client_ip: str = None, user_id: str = None):
        """Log HTTP request with structured data."""
        self.logger.info(
            "HTTP request",
            method=method,
            url=url,
            status_code=status_code,
            duration_ms=round(duration * 1000, 2),
            client_ip=client_ip,
            user_id=user_id,
            event_type="http_request"
        )
    
    def log_task_start(self, task_id: str, task_type: str, **kwargs):
        """Log task start with context."""
        self.logger.info(
            "Task started",
            task_id=task_id,
            task_type=task_type,
            event_type="task_start",
            **kwargs
        )
    
    def log_task_complete(self, task_id: str, task_type: str, 
                         duration: float, success: bool, **kwargs):
        """Log task completion with metrics."""
        self.logger.info(
            "Task completed",
            task_id=task_id,
            task_type=task_type,
            duration_seconds=round(duration, 2),
            success=success,
            event_type="task_complete",
            **kwargs
        )
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log error with full context."""
        self.logger.error(
            "Error occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            event_type="error",
            **(context or {})
        )
    
    def log_performance_metric(self, metric_name: str, value: float, 
                              unit: str = "", tags: Dict[str, str] = None):
        """Log performance metrics."""
        self.logger.info(
            "Performance metric",
            metric_name=metric_name,
            value=value,
            unit=unit,
            tags=tags or {},
            event_type="performance_metric"
        )

# Global logging service instance
logging_service = LoggingService()
