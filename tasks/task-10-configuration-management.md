# Task 10: Configuration Management

## Objective
Implement comprehensive configuration management for both server and remote workers, including environment-specific configs, secure credential management, and dynamic configuration updates.

## Current State Analysis

### Existing Configuration
- Basic environment variable configuration in core settings
- Hardcoded configuration values in various components
- No centralized configuration management
- No secure credential storage

### Configuration Requirements
- Centralized configuration management
- Environment-specific configurations (dev, staging, prod)
- Secure credential storage and rotation
- Dynamic configuration updates without restarts
- Configuration validation and defaults
- Remote worker configuration deployment

## Implementation Plan

### Step 1: Configuration Models

#### 1.1 Create Configuration Models
**File**: `packages/morag-core/src/morag_core/models/config.py`

```python
"""Configuration models for MoRAG system."""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field, validator
import os

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class DatabaseConfig(BaseModel):
    """Database configuration."""
    redis_url: str = "redis://localhost:6379/0"
    redis_max_connections: int = 20
    redis_retry_on_timeout: bool = True
    
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_api_key: Optional[str] = None
    qdrant_collection_name: str = "morag_documents"
    qdrant_timeout: int = 30

class APIConfig(BaseModel):
    """API server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    access_log: bool = True
    
    # CORS settings
    allowed_origins: List[str] = Field(default_factory=lambda: ["*"])
    allowed_methods: List[str] = Field(default_factory=lambda: ["*"])
    allowed_headers: List[str] = Field(default_factory=lambda: ["*"])
    
    # Request limits
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    request_timeout: int = 300  # 5 minutes

class SecurityConfig(BaseModel):
    """Security configuration."""
    jwt_secret: str = Field(default="change-this-secret-in-production")
    jwt_expiration_hours: int = 24
    api_key_length: int = 32
    
    # Rate limiting
    rate_limit_requests_per_minute: int = 60
    rate_limit_requests_per_hour: int = 1000
    rate_limit_burst: int = 10
    
    # File transfer encryption
    transfer_encryption_key: str = Field(default="change-this-key-in-production")
    
    # SSL/TLS
    ssl_enabled: bool = False
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None

class ProcessingConfig(BaseModel):
    """Processing configuration."""
    # Celery settings
    celery_task_soft_time_limit: int = 7200  # 2 hours
    celery_task_time_limit: int = 9000  # 2.5 hours
    celery_worker_prefetch_multiplier: int = 1
    celery_worker_max_tasks_per_child: int = 1000
    
    # File size limits
    max_document_size: int = 100 * 1024 * 1024  # 100MB
    max_audio_size: int = 2 * 1024 * 1024 * 1024  # 2GB
    max_video_size: int = 5 * 1024 * 1024 * 1024  # 5GB
    max_image_size: int = 50 * 1024 * 1024  # 50MB
    
    # Processing options
    enable_gpu: bool = True
    gpu_memory_limit_gb: Optional[float] = None
    temp_dir: str = "/tmp/morag"
    cleanup_interval_hours: int = 24

class AIConfig(BaseModel):
    """AI service configuration."""
    # Gemini API
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash"
    gemini_embedding_model: str = "text-embedding-004"
    gemini_vision_model: str = "gemini-1.5-flash"
    
    # Embedding settings
    embedding_batch_size: int = 10
    enable_batch_embedding: bool = True
    
    # Retry settings
    retry_indefinitely: bool = True
    retry_base_delay: float = 1.0
    retry_max_delay: float = 300.0
    retry_exponential_base: float = 2.0
    retry_jitter: bool = True

class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration."""
    log_level: LogLevel = LogLevel.INFO
    log_format: str = "json"
    log_file: Optional[str] = None
    
    # Metrics
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    # Health checks
    health_check_interval: int = 60
    health_check_timeout: int = 30

class WorkerConfig(BaseModel):
    """Worker-specific configuration."""
    worker_type: str = "cpu"  # cpu, gpu, hybrid
    max_concurrent_tasks: int = 2
    heartbeat_interval: int = 30
    
    # Queue configuration
    queues: List[str] = Field(default_factory=lambda: ["cpu_standard"])
    queue_priorities: Dict[str, int] = Field(default_factory=dict)
    
    # Processing capabilities
    supported_content_types: List[str] = Field(default_factory=list)
    
    # Audio processing
    whisper_model_size: str = "base"
    enable_diarization: bool = True
    enable_topic_segmentation: bool = True
    
    # Video processing
    enable_gpu_acceleration: bool = True
    thumbnail_quality: int = 85

class MoRAGConfig(BaseModel):
    """Complete MoRAG system configuration."""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    worker: WorkerConfig = Field(default_factory=WorkerConfig)
    
    # Custom settings
    custom: Dict[str, Any] = Field(default_factory=dict)
    
    # Metadata
    config_version: str = "1.0"
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('environment', pre=True)
    def validate_environment(cls, v):
        if isinstance(v, str):
            return Environment(v.lower())
        return v
    
    @validator('security')
    def validate_security_production(cls, v, values):
        env = values.get('environment')
        if env == Environment.PRODUCTION:
            if v.jwt_secret == "change-this-secret-in-production":
                raise ValueError("JWT secret must be changed in production")
            if v.transfer_encryption_key == "change-this-key-in-production":
                raise ValueError("Transfer encryption key must be changed in production")
        return v
    
    class Config:
        env_prefix = "MORAG_"
        env_nested_delimiter = "__"
        case_sensitive = False
```

### Step 2: Configuration Manager

#### 2.1 Create Configuration Manager
**File**: `packages/morag/src/morag/services/config_manager.py`

```python
"""Configuration manager for MoRAG system."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import structlog
from redis import Redis

from morag_core.models.config import MoRAGConfig, Environment

logger = structlog.get_logger(__name__)

class ConfigurationManager:
    """Manages configuration for MoRAG system."""
    
    def __init__(self, redis_client: Optional[Redis] = None):
        self.redis = redis_client
        self.config: Optional[MoRAGConfig] = None
        self.config_watchers: List[callable] = []
        
    async def load_config(self, config_path: Optional[str] = None,
                         environment: Optional[str] = None) -> MoRAGConfig:
        """Load configuration from various sources."""
        try:
            logger.info("Loading configuration", config_path=config_path, environment=environment)
            
            # Start with default configuration
            config_data = {}
            
            # Load from file if specified
            if config_path:
                config_data.update(await self._load_from_file(config_path))
            
            # Load environment-specific config
            if environment:
                env_config = await self._load_environment_config(environment)
                config_data.update(env_config)
            
            # Override with environment variables
            env_overrides = self._load_from_environment()
            config_data.update(env_overrides)
            
            # Load from Redis if available
            if self.redis:
                redis_config = await self._load_from_redis()
                if redis_config:
                    config_data.update(redis_config)
            
            # Create configuration object
            self.config = MoRAGConfig(**config_data)
            
            # Validate configuration
            await self._validate_config()
            
            # Save to Redis for sharing
            if self.redis:
                await self._save_to_redis()
            
            logger.info("Configuration loaded successfully",
                       environment=self.config.environment.value,
                       version=self.config.config_version)
            
            return self.config
            
        except Exception as e:
            logger.error("Failed to load configuration", error=str(e))
            raise
    
    async def update_config(self, updates: Dict[str, Any], 
                           persist: bool = True) -> MoRAGConfig:
        """Update configuration with new values."""
        try:
            if not self.config:
                raise RuntimeError("Configuration not loaded")
            
            # Apply updates
            config_dict = self.config.model_dump()
            config_dict.update(updates)
            
            # Create new configuration
            new_config = MoRAGConfig(**config_dict)
            
            # Validate new configuration
            old_config = self.config
            self.config = new_config
            
            try:
                await self._validate_config()
            except Exception as e:
                # Rollback on validation failure
                self.config = old_config
                raise
            
            # Persist if requested
            if persist and self.redis:
                await self._save_to_redis()
            
            # Notify watchers
            await self._notify_config_change(old_config, new_config)
            
            logger.info("Configuration updated", updates=list(updates.keys()))
            
            return self.config
            
        except Exception as e:
            logger.error("Failed to update configuration", error=str(e))
            raise
    
    def watch_config_changes(self, callback: callable):
        """Register a callback for configuration changes."""
        self.config_watchers.append(callback)
        logger.debug("Configuration watcher registered")
    
    async def get_worker_config(self, worker_id: str) -> Dict[str, Any]:
        """Get configuration specific to a worker."""
        if not self.config:
            raise RuntimeError("Configuration not loaded")
        
        # Base worker configuration
        worker_config = self.config.worker.model_dump()
        
        # Add worker-specific overrides from Redis
        if self.redis:
            worker_key = f"worker_config:{worker_id}"
            worker_overrides = self.redis.get(worker_key)
            if worker_overrides:
                overrides = json.loads(worker_overrides)
                worker_config.update(overrides)
        
        return worker_config
    
    async def set_worker_config(self, worker_id: str, config: Dict[str, Any]):
        """Set worker-specific configuration."""
        if not self.redis:
            raise RuntimeError("Redis not available for worker config")
        
        worker_key = f"worker_config:{worker_id}"
        self.redis.set(worker_key, json.dumps(config))
        
        logger.info("Worker configuration updated", worker_id=worker_id)
    
    async def _load_from_file(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        config_file = Path(config_path)
        
        if not config_file.exists():
            logger.warning("Configuration file not found", path=config_path)
            return {}
        
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() in ['.yml', '.yaml']:
                    return yaml.safe_load(f) or {}
                elif config_file.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    logger.warning("Unsupported config file format", path=config_path)
                    return {}
                    
        except Exception as e:
            logger.error("Failed to load config file", path=config_path, error=str(e))
            return {}
    
    async def _load_environment_config(self, environment: str) -> Dict[str, Any]:
        """Load environment-specific configuration."""
        env_file = f"config/{environment}.yml"
        
        if Path(env_file).exists():
            return await self._load_from_file(env_file)
        
        return {}
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Map environment variables to config structure
        env_mappings = {
            'MORAG_ENVIRONMENT': 'environment',
            'MORAG_DEBUG': 'debug',
            'MORAG_API_HOST': 'api.host',
            'MORAG_API_PORT': 'api.port',
            'MORAG_REDIS_URL': 'database.redis_url',
            'MORAG_QDRANT_HOST': 'database.qdrant_host',
            'MORAG_QDRANT_PORT': 'database.qdrant_port',
            'MORAG_JWT_SECRET': 'security.jwt_secret',
            'MORAG_GEMINI_API_KEY': 'ai.gemini_api_key',
            'MORAG_LOG_LEVEL': 'monitoring.log_level',
            'MORAG_WORKER_TYPE': 'worker.worker_type',
            'MORAG_MAX_CONCURRENT_TASKS': 'worker.max_concurrent_tasks',
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Convert value to appropriate type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '').isdigit():
                    value = float(value)
                
                # Set nested configuration
                self._set_nested_config(config, config_path, value)
        
        return config
    
    def _set_nested_config(self, config: Dict, path: str, value: Any):
        """Set a nested configuration value."""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    async def _load_from_redis(self) -> Optional[Dict[str, Any]]:
        """Load configuration from Redis."""
        try:
            config_data = self.redis.get("morag_config")
            if config_data:
                return json.loads(config_data)
            return None
            
        except Exception as e:
            logger.error("Failed to load config from Redis", error=str(e))
            return None
    
    async def _save_to_redis(self):
        """Save configuration to Redis."""
        try:
            if self.config:
                config_data = self.config.model_dump_json()
                self.redis.set("morag_config", config_data)
                logger.debug("Configuration saved to Redis")
                
        except Exception as e:
            logger.error("Failed to save config to Redis", error=str(e))
    
    async def _validate_config(self):
        """Validate the current configuration."""
        if not self.config:
            raise ValueError("No configuration to validate")
        
        # Environment-specific validations
        if self.config.environment == Environment.PRODUCTION:
            # Production-specific validations
            if self.config.debug:
                logger.warning("Debug mode enabled in production")
            
            if not self.config.ai.gemini_api_key:
                raise ValueError("Gemini API key required in production")
            
            if self.config.security.ssl_enabled and not self.config.security.ssl_cert_path:
                raise ValueError("SSL certificate path required when SSL is enabled")
        
        # General validations
        if self.config.api.port < 1 or self.config.api.port > 65535:
            raise ValueError("Invalid API port number")
        
        if self.config.processing.max_concurrent_tasks < 1:
            raise ValueError("Max concurrent tasks must be at least 1")
        
        logger.debug("Configuration validation passed")
    
    async def _notify_config_change(self, old_config: MoRAGConfig, new_config: MoRAGConfig):
        """Notify watchers of configuration changes."""
        for callback in self.config_watchers:
            try:
                await callback(old_config, new_config)
            except Exception as e:
                logger.error("Config change callback failed", error=str(e))

# Global configuration manager instance
config_manager = ConfigurationManager()

async def get_config() -> MoRAGConfig:
    """Get the current configuration."""
    if not config_manager.config:
        await config_manager.load_config()
    return config_manager.config

async def update_config(updates: Dict[str, Any]) -> MoRAGConfig:
    """Update the current configuration."""
    return await config_manager.update_config(updates)
```

### Step 3: Configuration CLI

#### 3.1 Create Configuration CLI
**File**: `packages/morag/src/morag/cli/config.py`

```python
"""Configuration management CLI."""

import asyncio
import json
import yaml
from pathlib import Path
from typing import Optional
import click
import structlog

from morag.services.config_manager import ConfigurationManager
from morag_core.models.config import Environment

logger = structlog.get_logger(__name__)

@click.group()
def config():
    """Configuration management commands."""
    pass

@config.command()
@click.option('--environment', '-e', type=click.Choice(['development', 'staging', 'production']),
              help='Environment to generate config for')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-f', type=click.Choice(['yaml', 'json']), default='yaml',
              help='Output format')
def generate(environment: Optional[str], output: Optional[str], format: str):
    """Generate a configuration file template."""
    async def _generate():
        config_manager = ConfigurationManager()
        
        # Load default configuration
        config = await config_manager.load_config(environment=environment)
        
        # Convert to dict
        config_dict = config.model_dump()
        
        # Format output
        if format == 'yaml':
            content = yaml.dump(config_dict, default_flow_style=False, indent=2)
        else:
            content = json.dumps(config_dict, indent=2)
        
        # Write to file or stdout
        if output:
            Path(output).write_text(content)
            click.echo(f"Configuration written to {output}")
        else:
            click.echo(content)
    
    asyncio.run(_generate())

@config.command()
@click.option('--config-file', '-c', type=click.Path(exists=True),
              help='Configuration file to validate')
@click.option('--environment', '-e', type=click.Choice(['development', 'staging', 'production']),
              help='Environment to validate for')
def validate(config_file: Optional[str], environment: Optional[str]):
    """Validate a configuration file."""
    async def _validate():
        try:
            config_manager = ConfigurationManager()
            config = await config_manager.load_config(
                config_path=config_file,
                environment=environment
            )
            
            click.echo("✅ Configuration is valid")
            click.echo(f"Environment: {config.environment.value}")
            click.echo(f"Version: {config.config_version}")
            
        except Exception as e:
            click.echo(f"❌ Configuration validation failed: {str(e)}")
            raise click.Abort()
    
    asyncio.run(_validate())

@config.command()
@click.option('--key', '-k', required=True, help='Configuration key to get')
@click.option('--config-file', '-c', type=click.Path(exists=True),
              help='Configuration file to read from')
def get(key: str, config_file: Optional[str]):
    """Get a configuration value."""
    async def _get():
        config_manager = ConfigurationManager()
        config = await config_manager.load_config(config_path=config_file)
        
        # Navigate to the key
        value = config.model_dump()
        for part in key.split('.'):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                click.echo(f"Key '{key}' not found")
                raise click.Abort()
        
        click.echo(json.dumps(value, indent=2))
    
    asyncio.run(_get())

@config.command()
@click.option('--worker-id', '-w', required=True, help='Worker ID')
@click.option('--config', '-c', required=True, help='Configuration JSON')
def set_worker(worker_id: str, config: str):
    """Set worker-specific configuration."""
    async def _set_worker():
        try:
            config_data = json.loads(config)
            
            config_manager = ConfigurationManager()
            await config_manager.load_config()
            await config_manager.set_worker_config(worker_id, config_data)
            
            click.echo(f"✅ Worker configuration updated for {worker_id}")
            
        except json.JSONDecodeError:
            click.echo("❌ Invalid JSON configuration")
            raise click.Abort()
        except Exception as e:
            click.echo(f"❌ Failed to update worker config: {str(e)}")
            raise click.Abort()
    
    asyncio.run(_set_worker())

@config.command()
@click.option('--worker-id', '-w', required=True, help='Worker ID')
def get_worker(worker_id: str):
    """Get worker-specific configuration."""
    async def _get_worker():
        try:
            config_manager = ConfigurationManager()
            await config_manager.load_config()
            worker_config = await config_manager.get_worker_config(worker_id)
            
            click.echo(json.dumps(worker_config, indent=2))
            
        except Exception as e:
            click.echo(f"❌ Failed to get worker config: {str(e)}")
            raise click.Abort()
    
    asyncio.run(_get_worker())

if __name__ == '__main__':
    config()
```

## Testing Requirements

### Unit Tests
1. **Configuration Manager Tests**
   - Test configuration loading from various sources
   - Test configuration validation
   - Test configuration updates
   - Test worker-specific configuration

2. **Configuration Model Tests**
   - Test configuration validation rules
   - Test environment-specific validation
   - Test nested configuration handling

### Integration Tests
1. **End-to-End Configuration Tests**
   - Test complete configuration workflow
   - Test configuration changes propagation
   - Test worker configuration deployment

### Test Files to Create
- `tests/test_config_manager.py`
- `tests/test_config_models.py`
- `tests/integration/test_config_e2e.py`

## Dependencies
- **New**: `PyYAML` for YAML configuration files
- **New**: `click` for CLI interface
- **Existing**: Redis for configuration storage

## Success Criteria
1. Configuration can be loaded from multiple sources with proper precedence
2. Environment-specific configurations work correctly
3. Configuration validation prevents invalid settings
4. Worker-specific configurations can be managed remotely
5. Configuration changes can be applied without system restart
6. CLI tools provide easy configuration management

## Next Steps
After completing this task:
1. Proceed to Task 11: Monitoring Dashboard
2. Test configuration management with different environments
3. Validate configuration deployment to remote workers

---

**Dependencies**: Task 9 (Authentication & Security)
**Estimated Time**: 3-4 days
**Risk Level**: Medium (configuration complexity)
