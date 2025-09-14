"""Centralized configuration management for MoRAG packages.

This module provides a unified configuration management system that:
1. Supports multiple configuration sources (environment, files, defaults)
2. Provides caching for improved performance  
3. Supports package-specific configurations
4. Allows testing overrides
5. Reduces configuration coupling across packages
"""

import os
import yaml
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pathlib import Path
from functools import lru_cache
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T')


class ConfigurationSource(ABC):
    """Abstract base class for configuration sources."""
    
    @abstractmethod
    def load(self, package: str) -> Dict[str, Any]:
        """Load configuration for a specific package."""
        pass
    
    @abstractmethod 
    def priority(self) -> int:
        """Return priority (higher numbers override lower numbers)."""
        pass


class EnvironmentSource(ConfigurationSource):
    """Load configuration from environment variables."""
    
    def __init__(self, env_prefix: str = "MORAG_"):
        self.env_prefix = env_prefix
        
    def load(self, package: str) -> Dict[str, Any]:
        """Load environment variables for package."""
        config = {}
        package_name = package.upper().replace('-', '_')
        package_prefix = f"{self.env_prefix}{package_name}_"

        # Load package-specific variables
        for key, value in os.environ.items():
            if key.startswith(package_prefix):
                # Convert MORAG_AUDIO_MAX_DURATION to max_duration
                config_key = key[len(package_prefix):].lower()
                config[config_key] = self._convert_value(value)

        # Load global MORAG_ variables as fallbacks
        global_prefix = self.env_prefix
        for key, value in os.environ.items():
            if key.startswith(global_prefix) and not key.startswith(package_prefix):
                # Convert MORAG_CHUNK_SIZE to chunk_size
                config_key = key[len(global_prefix):].lower()
                if config_key not in config:  # Don't override package-specific
                    config[config_key] = self._convert_value(value)

        return config
    
    def priority(self) -> int:
        return 200  # High priority
        
    def _convert_value(self, value: str) -> Any:
        """Convert string environment value to appropriate type."""
        # Handle boolean values
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
            
        # Handle numeric values
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            pass
            
        # Handle JSON values
        if value.startswith('{') or value.startswith('['):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
                
        # Return as string
        return value


class FileSource(ConfigurationSource):
    """Load configuration from YAML/JSON files."""
    
    def __init__(self, config_path: Union[str, Path]):
        self.config_path = Path(config_path)
        
    def load(self, package: str) -> Dict[str, Any]:
        """Load configuration from file for package."""
        if not self.config_path.exists():
            return {}
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                if self.config_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f) or {}
                elif self.config_path.suffix.lower() == '.json':
                    data = json.load(f)
                else:
                    logger.warning(f"Unsupported config file format: {self.config_path}")
                    return {}
                    
            # Extract package-specific config
            package_config = data.get('packages', {}).get(package, {})
            global_config = data.get('global', {})
            
            # Merge global and package-specific (package overrides global)
            merged_config = {**global_config, **package_config}
            return merged_config
            
        except Exception as e:
            logger.warning(f"Failed to load config from {self.config_path}: {e}")
            return {}
            
    def priority(self) -> int:
        return 100  # Medium priority


class DefaultsSource(ConfigurationSource):
    """Provide default configuration values."""
    
    def __init__(self):
        self.defaults = {
            'morag-core': {
                'log_level': 'INFO',
                'temp_dir': './temp',
                'chunk_size': 4000,
                'chunk_overlap': 200,
            },
            'morag-audio': {
                'model': 'whisper-1',
                'max_duration': 600,
                'enable_diarization': False,
                'chunk_length': 30,
            },
            'morag-document': {
                'max_file_size': '100MB',
                'enable_ocr': True,
                'preserve_formatting': True,
            },
            'morag-services': {
                'max_workers': 4,
                'timeout': 30,
                'retry_attempts': 3,
            },
            'morag-graph': {
                'max_entities': 100,
                'max_relations': 50,
                'confidence_threshold': 0.7,
            },
            'morag-embedding': {
                'batch_size': 100,
                'model': 'text-embedding-004',
                'enable_caching': True,
            }
        }
        
    def load(self, package: str) -> Dict[str, Any]:
        """Load default configuration for package."""
        return self.defaults.get(package, {})
        
    def priority(self) -> int:
        return 0  # Lowest priority


class ConfigurationManager:
    """Centralized configuration management with multiple sources and caching."""
    
    def __init__(self, env_prefix: str = "MORAG_", config_file: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            env_prefix: Prefix for environment variables
            config_file: Optional path to configuration file
        """
        self.env_prefix = env_prefix
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._overrides: Dict[str, Dict[str, Any]] = {}
        
        # Initialize configuration sources (in priority order)
        self._sources: List[ConfigurationSource] = [
            DefaultsSource(),
            EnvironmentSource(env_prefix),
        ]
        
        # Add file source if specified
        if config_file:
            file_path = Path(config_file)
            if file_path.exists():
                self._sources.append(FileSource(file_path))
            else:
                # Try common locations
                for common_path in ['config.yaml', 'config.yml', 'morag.yaml']:
                    if Path(common_path).exists():
                        self._sources.append(FileSource(common_path))
                        break
                        
        # Sort sources by priority
        self._sources.sort(key=lambda s: s.priority())
        
    def get_package_config(self, package: str) -> Dict[str, Any]:
        """Get configuration for specific package with caching.
        
        Args:
            package: Package name (e.g., 'morag-audio', 'morag-core')
            
        Returns:
            Configuration dictionary for the package
        """
        if package not in self._cache:
            self._cache[package] = self._load_config(package)
            
        # Apply any active overrides
        if package in self._overrides:
            config = self._cache[package].copy()
            config.update(self._overrides[package])
            return config
            
        return self._cache[package].copy()
        
    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration that applies to all packages."""
        return self.get_package_config('global')
        
    def get_config_value(self, package: str, key: str, default: Any = None) -> Any:
        """Get a specific configuration value.
        
        Args:
            package: Package name
            key: Configuration key (supports dot notation like 'llm.model')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        config = self.get_package_config(package)
        
        # Handle dot notation (e.g., 'llm.model')
        keys = key.split('.')
        current = config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
                
        return current
        
    def set_config_value(self, package: str, key: str, value: Any) -> None:
        """Set a configuration value (creates override).
        
        Args:
            package: Package name
            key: Configuration key (supports dot notation)
            value: Value to set
        """
        if package not in self._overrides:
            self._overrides[package] = {}
            
        # Handle dot notation
        keys = key.split('.')
        current = self._overrides[package]
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
            
        current[keys[-1]] = value
        
        # Clear cache to force reload
        if package in self._cache:
            del self._cache[package]
            
    def override(self, package: str, overrides: Dict[str, Any]) -> None:
        """Override configuration for testing or runtime changes.
        
        Args:
            package: Package name
            overrides: Dictionary of configuration overrides
        """
        if package not in self._overrides:
            self._overrides[package] = {}
            
        self._overrides[package].update(overrides)
        
        # Clear cache to force reload
        if package in self._cache:
            del self._cache[package]
            
    def clear_overrides(self, package: Optional[str] = None) -> None:
        """Clear configuration overrides.
        
        Args:
            package: Package name to clear overrides for, or None for all packages
        """
        if package:
            self._overrides.pop(package, None)
            if package in self._cache:
                del self._cache[package]
        else:
            self._overrides.clear()
            self._cache.clear()
            
    def reload_config(self, package: Optional[str] = None) -> None:
        """Reload configuration from sources.
        
        Args:
            package: Package name to reload, or None for all packages
        """
        if package:
            if package in self._cache:
                del self._cache[package]
        else:
            self._cache.clear()
            
    def get_available_packages(self) -> List[str]:
        """Get list of packages with available configuration."""
        packages = set()
        
        # Get packages from sources
        for source in self._sources:
            if hasattr(source, 'defaults'):
                packages.update(source.defaults.keys())
                
        # Add packages from environment variables
        for key in os.environ:
            if key.startswith(self.env_prefix):
                parts = key[len(self.env_prefix):].split('_')
                if len(parts) > 1:
                    package = f"morag-{parts[0].lower()}"
                    packages.add(package)
                    
        return sorted(list(packages))
        
    def validate_config(self, package: str, schema: Optional[Dict[str, Any]] = None) -> List[str]:
        """Validate package configuration against schema.
        
        Args:
            package: Package name
            schema: Optional validation schema
            
        Returns:
            List of validation errors (empty if valid)
        """
        config = self.get_package_config(package)
        errors = []
        
        # Basic validation without schema
        if not config:
            errors.append(f"No configuration found for package '{package}'")
            return errors
            
        # If schema provided, validate against it
        if schema:
            errors.extend(self._validate_against_schema(config, schema))
            
        return errors
        
    def _load_config(self, package: str) -> Dict[str, Any]:
        """Load configuration from all sources and merge."""
        merged_config = {}
        
        # Load from all sources in priority order
        for source in self._sources:
            try:
                source_config = source.load(package)
                if source_config:
                    # Merge with lower priority configs
                    merged_config = self._deep_merge(merged_config, source_config)
                    logger.debug(f"Loaded config from {source.__class__.__name__} for {package}")
            except Exception as e:
                logger.warning(f"Failed to load config from {source.__class__.__name__} for {package}: {e}")
                
        return merged_config
        
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def _validate_against_schema(self, config: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate configuration against schema."""
        errors = []
        
        # Simple schema validation
        required = schema.get('required', [])
        for field in required:
            if field not in config:
                errors.append(f"Required field '{field}' missing")
                
        properties = schema.get('properties', {})
        for field, value in config.items():
            if field in properties:
                field_schema = properties[field]
                field_type = field_schema.get('type')
                
                if field_type and not self._check_type(value, field_type):
                    errors.append(f"Field '{field}' has invalid type (expected {field_type})")
                    
        return errors
        
    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            'string': str,
            'integer': int,
            'number': (int, float),
            'boolean': bool,
            'array': list,
            'object': dict,
        }
        
        expected = type_map.get(expected_type)
        if expected:
            return isinstance(value, expected)
        return True


# Global configuration manager instance
@lru_cache(maxsize=1)
def get_config_manager() -> ConfigurationManager:
    """Get the global configuration manager instance."""
    return ConfigurationManager()


# Convenience functions for common operations
def get_package_config(package: str) -> Dict[str, Any]:
    """Get configuration for a package."""
    return get_config_manager().get_package_config(package)


def get_config_value(package: str, key: str, default: Any = None) -> Any:
    """Get a specific configuration value."""
    return get_config_manager().get_config_value(package, key, default)


def override_config(package: str, overrides: Dict[str, Any]) -> None:
    """Override configuration for testing."""
    get_config_manager().override(package, overrides)


def clear_config_overrides(package: Optional[str] = None) -> None:
    """Clear configuration overrides."""
    get_config_manager().clear_overrides(package)


# Configuration context manager for testing
class ConfigOverride:
    """Context manager for temporary configuration overrides."""
    
    def __init__(self, package: str, overrides: Dict[str, Any]):
        self.package = package
        self.overrides = overrides
        self.manager = get_config_manager()
        self.original_overrides = None
        
    def __enter__(self):
        # Save current overrides
        self.original_overrides = self.manager._overrides.get(self.package, {}).copy()
        # Apply new overrides
        self.manager.override(self.package, self.overrides)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original overrides
        if self.original_overrides:
            self.manager._overrides[self.package] = self.original_overrides
        else:
            self.manager._overrides.pop(self.package, None)
            
        # Clear cache
        if self.package in self.manager._cache:
            del self.manager._cache[self.package]