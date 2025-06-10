#!/usr/bin/env python3
"""Configuration management for MoRAG Remote Converter."""

import os
import yaml
from typing import Dict, Any
import structlog

logger = structlog.get_logger(__name__)


class RemoteConverterConfig:
    """Configuration management for remote converter."""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file or "remote_converter_config.yaml"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file and environment variables."""
        config = {}
        
        # Load from config file if it exists
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    if file_config:
                        config.update(file_config)
                logger.info("Loaded configuration from file", file=self.config_file)
            except Exception as e:
                logger.warning("Failed to load config file", file=self.config_file, error=str(e))
        
        # Override with environment variables
        env_config = {
            'worker_id': os.getenv('MORAG_WORKER_ID', f'remote-worker-{os.getpid()}'),
            'api_base_url': os.getenv('MORAG_API_BASE_URL', 'http://localhost:8000'),
            'api_key': os.getenv('MORAG_API_KEY'),
            'content_types': os.getenv('MORAG_WORKER_CONTENT_TYPES', 'audio,video').split(','),
            'poll_interval': int(os.getenv('MORAG_WORKER_POLL_INTERVAL', '10')),
            'max_concurrent_jobs': int(os.getenv('MORAG_WORKER_MAX_CONCURRENT_JOBS', '2')),
            'log_level': os.getenv('MORAG_LOG_LEVEL', 'INFO'),
            'temp_dir': os.getenv('MORAG_TEMP_DIR', '/tmp/morag_remote')
        }
        
        # Remove None values
        env_config = {k: v for k, v in env_config.items() if v is not None}
        config.update(env_config)
        
        return config
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration."""
        return self.config.copy()
    
    def validate_config(self) -> bool:
        """Validate the configuration."""
        required_fields = ['worker_id', 'api_base_url', 'content_types']
        
        for field in required_fields:
            if field not in self.config or not self.config[field]:
                logger.error("Missing required configuration field", field=field)
                return False
        
        # Validate content types
        valid_content_types = ['audio', 'video', 'document', 'image', 'web', 'youtube']
        for content_type in self.config['content_types']:
            if content_type not in valid_content_types:
                logger.error("Invalid content type", content_type=content_type, valid_types=valid_content_types)
                return False
        
        # Validate numeric values
        try:
            if self.config['poll_interval'] <= 0:
                logger.error("Poll interval must be positive", poll_interval=self.config['poll_interval'])
                return False
            
            if self.config['max_concurrent_jobs'] <= 0:
                logger.error("Max concurrent jobs must be positive", max_concurrent_jobs=self.config['max_concurrent_jobs'])
                return False
        except (KeyError, TypeError, ValueError) as e:
            logger.error("Invalid numeric configuration", error=str(e))
            return False
        
        return True
    
    def create_sample_config(self, file_path: str = None):
        """Create a sample configuration file."""
        if not file_path:
            file_path = "remote_converter_config.yaml.example"
        
        sample_config = {
            'worker_id': 'gpu-worker-01',
            'api_base_url': 'https://api.morag.com',
            'api_key': 'your-api-key-here',
            'content_types': ['audio', 'video'],
            'poll_interval': 10,
            'max_concurrent_jobs': 2,
            'log_level': 'INFO',
            'temp_dir': '/tmp/morag_remote'
        }
        
        try:
            with open(file_path, 'w') as f:
                yaml.dump(sample_config, f, default_flow_style=False, indent=2)
            
            logger.info("Sample configuration created", file=file_path)
            return True
        except Exception as e:
            logger.error("Failed to create sample configuration", file=file_path, error=str(e))
            return False
    
    def print_config(self):
        """Print current configuration (hiding sensitive data)."""
        config_copy = self.config.copy()
        
        # Hide sensitive information
        if 'api_key' in config_copy and config_copy['api_key']:
            config_copy['api_key'] = '***HIDDEN***'
        
        print("Current Configuration:")
        print("=" * 40)
        for key, value in config_copy.items():
            print(f"{key}: {value}")
        print("=" * 40)


def setup_logging(log_level: str = 'INFO'):
    """Set up structured logging."""
    import logging
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
