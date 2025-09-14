"""Unified configuration system for MoRAG."""

from .unified import (
    ConfigMixin,
    LLMConfig,
    MarkdownOptimizerConfig,
    FactGeneratorConfig,
    ChunkerConfig,
    IngestorConfig,
)

from .manager import (
    ConfigurationManager,
    ConfigurationSource,
    EnvironmentSource,
    FileSource,
    DefaultsSource,
    ConfigOverride,
    get_config_manager,
    get_package_config,
    get_config_value,
    override_config,
    clear_config_overrides,
)

# Import settings from the main config module
try:
    from ..config import settings, Settings, get_settings, validate_configuration_and_log, validate_chunk_size, detect_device, get_safe_device
except ImportError:
    from typing import Optional
    # Create a minimal settings fallback that handles missing attributes
    class MockSettings:
        def __init__(self):
            self._defaults = {
                'log_level': "INFO",
                'temp_dir': "./temp",
                'api_host': "0.0.0.0",
                'api_port': 8000,
                'gemini_api_key': None,
                'gemini_model': "gemini-1.5-flash",
                'gemini_generation_model': "gemini-1.5-flash",
                'gemini_embedding_model': "text-embedding-004",
                'qdrant_host': "localhost",
                'qdrant_port': 6333,
                'qdrant_collection_name': "morag_documents",
                'qdrant_api_key': None,
                'neo4j_uri': "bolt://localhost:7687",
                'neo4j_user': "neo4j",
                'neo4j_password': "password",
                'upload_dir': "./uploads",
                'max_file_size': "100MB",
                'max_upload_size_bytes': 104857600,
                'embedding_batch_size': 100,
                'max_concurrent_tasks': 4,
                'redis_url': "redis://localhost:6379/0",
            }

        def __getattr__(self, name):
            # Return default value or None for any missing attribute
            return self._defaults.get(name, None)

        def get_max_upload_size_bytes(self) -> int:
            """Get the maximum upload size in bytes."""
            return self._defaults.get('max_upload_size_bytes', 104857600)  # 100MB default

    def validate_chunk_size(size: int) -> int:
        """Validate chunk size."""
        if size < 100:
            return 100
        if size > 10000:
            return 10000
        return size

    def detect_device() -> str:
        """Mock detect_device function."""
        return "cpu"

    def get_safe_device(preferred_device: Optional[str] = None) -> str:
        """Mock get_safe_device function."""
        return "cpu"

    settings = MockSettings()
    Settings = type(settings)
    get_settings = lambda: settings
    validate_configuration_and_log = lambda: settings

__all__ = [
    'ConfigMixin',
    'LLMConfig',
    'MarkdownOptimizerConfig',
    'FactGeneratorConfig',
    'ChunkerConfig',
    'IngestorConfig',
    'ConfigurationManager',
    'ConfigurationSource',
    'EnvironmentSource',
    'FileSource',
    'DefaultsSource',
    'ConfigOverride',
    'get_config_manager',
    'get_package_config',
    'get_config_value',
    'override_config',
    'clear_config_overrides',
    'settings',
    'Settings',
    'get_settings',
    'validate_configuration_and_log',
    'validate_chunk_size',
    'detect_device',
    'get_safe_device',
]
