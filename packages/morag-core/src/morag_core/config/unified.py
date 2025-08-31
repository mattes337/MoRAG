"""Unified configuration system for MoRAG with environment variables, CLI, and REST overrides."""

import os
from typing import Any, Dict, Optional, Type, TypeVar, Union, get_type_hints
from pydantic import BaseModel, Field
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseModel)


class ConfigMixin:
    """Mixin class that provides unified configuration loading with fallbacks."""
    
    @classmethod
    def from_env_and_overrides(
        cls: Type[T], 
        overrides: Optional[Dict[str, Any]] = None,
        prefix: Optional[str] = None
    ) -> T:
        """Create configuration from environment variables with optional overrides.
        
        Args:
            overrides: Dictionary of override values (from CLI/REST)
            prefix: Environment variable prefix (auto-detected if None)
            
        Returns:
            Configuration instance with environment variables and overrides applied
        """
        if prefix is None:
            prefix = getattr(cls, '_env_prefix', 'MORAG_')
        
        # Get field information from the model
        field_info = cls.model_fields if hasattr(cls, 'model_fields') else {}
        type_hints = get_type_hints(cls)
        
        # Load from environment variables
        env_config = {}
        for field_name, field in field_info.items():
            env_var = f"{prefix}{field_name.upper()}"
            env_value = os.environ.get(env_var)

            if env_value is not None:
                try:
                    # Get field type from annotation or default
                    field_type = type_hints.get(field_name, str)

                    # Handle Optional types
                    if hasattr(field_type, '__origin__') and field_type.__origin__ is Union:
                        # Extract the non-None type from Optional[T]
                        args = field_type.__args__
                        field_type = next((arg for arg in args if arg is not type(None)), str)

                    converted_value = cls._convert_env_value(env_value, field_type)
                    env_config[field_name] = converted_value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {env_value}", error=str(e))
        
        # Apply overrides
        if overrides:
            env_config.update(overrides)
        
        return cls(**env_config)
    
    @staticmethod
    def _convert_env_value(value: str, target_type: Type) -> Any:
        """Convert environment variable string to target type."""
        if target_type == bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        elif target_type == int:
            return int(value)
        elif target_type == float:
            return float(value)
        elif target_type == str:
            return value
        else:
            # For complex types, try to parse as string
            return value


class LLMConfig(BaseModel, ConfigMixin):
    """Unified LLM configuration with environment variable support."""
    
    _env_prefix = "MORAG_LLM_"
    
    # Provider and model with fallbacks
    provider: str = Field(default="gemini", description="LLM provider")
    model: str = Field(default="gemini-1.5-flash", description="LLM model")
    api_key: Optional[str] = Field(default=None, description="API key")
    
    # Generation parameters
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: int = Field(default=4000, ge=1, description="Maximum tokens")
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")
    
    # Retry configuration
    max_retries: int = Field(default=5, ge=0, description="Maximum retry attempts")
    base_delay: float = Field(default=2.0, ge=0.0, description="Base retry delay")
    max_delay: float = Field(default=120.0, ge=0.0, description="Maximum retry delay")
    
    @classmethod
    def from_env_and_overrides(
        cls,
        overrides: Optional[Dict[str, Any]] = None,
        prefix: Optional[str] = None
    ) -> 'LLMConfig':
        """Create LLM config with proper fallbacks to MORAG_GEMINI_MODEL."""
        # Load base configuration from environment variables
        config_dict = {}

        # Load from environment with fallbacks
        config_dict['provider'] = os.environ.get("MORAG_LLM_PROVIDER", "gemini")

        # Model fallback chain: MORAG_LLM_MODEL -> MORAG_GEMINI_MODEL -> default
        config_dict['model'] = (
            os.environ.get("MORAG_LLM_MODEL") or
            os.environ.get("MORAG_GEMINI_MODEL") or
            "gemini-1.5-flash"
        )

        # API key fallback chain
        config_dict['api_key'] = (
            os.environ.get("MORAG_LLM_API_KEY") or
            os.environ.get("GEMINI_API_KEY") or
            os.environ.get("GOOGLE_API_KEY")
        )

        # Load other environment variables with defaults
        config_dict['temperature'] = float(os.environ.get("MORAG_LLM_TEMPERATURE", "0.1"))
        config_dict['max_tokens'] = int(os.environ.get("MORAG_LLM_MAX_TOKENS", "4000"))
        config_dict['timeout'] = int(os.environ.get("MORAG_LLM_TIMEOUT", "30"))
        config_dict['max_retries'] = int(os.environ.get("MORAG_LLM_MAX_RETRIES", "5"))
        config_dict['base_delay'] = float(os.environ.get("MORAG_LLM_BASE_DELAY", "2.0"))
        config_dict['max_delay'] = float(os.environ.get("MORAG_LLM_MAX_DELAY", "120.0"))

        # Apply overrides
        if overrides:
            config_dict.update(overrides)

        return cls(**config_dict)


class MarkdownOptimizerConfig(BaseModel, ConfigMixin):
    """Configuration for markdown optimizer stage."""

    enabled: bool = Field(default=True, description="Enable markdown optimization")

    # LLM configuration (inherits from global LLM config)
    model: Optional[str] = Field(default=None, description="LLM model override")
    provider: Optional[str] = Field(default=None, description="LLM provider override")
    temperature: Optional[float] = Field(default=None, description="LLM temperature override")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens override")
    max_retries: Optional[int] = Field(default=None, description="Max retries override")

    # Text splitting configuration
    max_chunk_size: int = Field(default=50000, ge=1000, description="Maximum characters per chunk")
    enable_splitting: bool = Field(default=True, description="Enable text splitting for large files")

    # Optimization settings
    fix_transcription_errors: bool = Field(default=True, description="Fix transcription errors")
    improve_structure: bool = Field(default=True, description="Improve document structure")
    preserve_timestamps: bool = Field(default=True, description="Preserve timestamp information")
    preserve_metadata: bool = Field(default=True, description="Preserve metadata headers")

    @classmethod
    def from_env_and_overrides(
        cls,
        overrides: Optional[Dict[str, Any]] = None,
        prefix: Optional[str] = None
    ) -> 'MarkdownOptimizerConfig':
        """Create configuration from environment variables with overrides."""
        if prefix is None:
            prefix = "MORAG_MARKDOWN_OPTIMIZER_"



        # Load from environment variables
        env_config = {}

        # Define field mappings with types
        field_mappings = {
            'enabled': ('ENABLED', bool),
            'model': ('MODEL', str),
            'provider': ('PROVIDER', str),
            'temperature': ('TEMPERATURE', float),
            'max_tokens': ('MAX_TOKENS', int),
            'max_retries': ('MAX_RETRIES', int),
            'max_chunk_size': ('MAX_CHUNK_SIZE', int),
            'enable_splitting': ('ENABLE_SPLITTING', bool),
            'fix_transcription_errors': ('FIX_TRANSCRIPTION_ERRORS', bool),
            'improve_structure': ('IMPROVE_STRUCTURE', bool),
            'preserve_timestamps': ('PRESERVE_TIMESTAMPS', bool),
            'preserve_metadata': ('PRESERVE_METADATA', bool),
        }

        for field_name, (env_suffix, field_type) in field_mappings.items():
            env_var = f"{prefix}{env_suffix}"
            env_value = os.environ.get(env_var)

            if env_value is not None:
                try:
                    converted_value = cls._convert_env_value(env_value, field_type)
                    env_config[field_name] = converted_value

                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {env_value}", error=str(e))

        # Apply overrides
        if overrides:
            env_config.update(overrides)

        return cls(**env_config)
    
    def get_llm_config(self, base_llm_config: Optional[LLMConfig] = None) -> LLMConfig:
        """Get LLM configuration with stage-specific overrides."""
        if base_llm_config is None:
            base_llm_config = LLMConfig.from_env_and_overrides()
        
        # Create overrides dict with non-None values
        overrides = {}
        if self.model is not None:
            overrides['model'] = self.model
        if self.provider is not None:
            overrides['provider'] = self.provider
        if self.temperature is not None:
            overrides['temperature'] = self.temperature
        if self.max_tokens is not None:
            overrides['max_tokens'] = self.max_tokens
        if self.max_retries is not None:
            overrides['max_retries'] = self.max_retries
        
        # Apply overrides to base config
        config_dict = base_llm_config.model_dump()
        config_dict.update(overrides)
        
        return LLMConfig(**config_dict)


class FactGeneratorConfig(BaseModel, ConfigMixin):
    """Configuration for fact generator stage."""

    enabled: bool = Field(default=True, description="Enable fact generation")

    # LLM configuration overrides
    model: Optional[str] = Field(default=None, description="LLM model override")
    provider: Optional[str] = Field(default=None, description="LLM provider override")

    # Quality validation settings
    min_confidence: float = Field(default=0.5, description="Minimum confidence threshold for facts")
    allow_vague_language: bool = Field(default=False, description="Allow facts with vague language (typically, usually, etc.)")
    require_entities: bool = Field(default=True, description="Require primary entities in structured metadata")
    min_fact_length: int = Field(default=20, description="Minimum fact text length")
    strict_validation: bool = Field(default=True, description="Enable strict quality validation")

    temperature: Optional[float] = Field(default=None, description="LLM temperature override")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens override")
    
    # Extraction settings
    extract_entities: bool = Field(default=True, description="Extract entities")
    extract_relations: bool = Field(default=True, description="Extract relations")
    extract_keywords: bool = Field(default=True, description="Extract keywords")
    domain: str = Field(default="general", description="Domain for extraction")
    
    # Quality settings
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_entities_per_chunk: int = Field(default=20, ge=1, description="Maximum entities per chunk")
    max_relations_per_chunk: int = Field(default=15, ge=1, description="Maximum relations per chunk")
    
    @classmethod
    def from_env_and_overrides(
        cls,
        overrides: Optional[Dict[str, Any]] = None,
        prefix: Optional[str] = None
    ) -> 'FactGeneratorConfig':
        """Create configuration from environment variables with overrides."""
        if prefix is None:
            prefix = "MORAG_FACT_GENERATOR_"

        # Load from environment variables
        env_config = {}

        # Define field mappings with types
        field_mappings = {
            'enabled': ('ENABLED', bool),
            'model': ('MODEL', str),
            'provider': ('PROVIDER', str),
            'temperature': ('TEMPERATURE', float),
            'max_tokens': ('MAX_TOKENS', int),
            'extract_entities': ('EXTRACT_ENTITIES', bool),
            'extract_relations': ('EXTRACT_RELATIONS', bool),
            'extract_keywords': ('EXTRACT_KEYWORDS', bool),
            'domain': ('DOMAIN', str),
            'min_confidence': ('MIN_CONFIDENCE', float),
            'max_entities_per_chunk': ('MAX_ENTITIES_PER_CHUNK', int),
            'max_relations_per_chunk': ('MAX_RELATIONS_PER_CHUNK', int),
        }

        for field_name, (env_suffix, field_type) in field_mappings.items():
            env_var = f"{prefix}{env_suffix}"
            env_value = os.environ.get(env_var)

            if env_value is not None:
                try:
                    converted_value = cls._convert_env_value(env_value, field_type)
                    env_config[field_name] = converted_value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {env_value}", error=str(e))

        # Apply overrides
        if overrides:
            env_config.update(overrides)

        return cls(**env_config)

    def get_llm_config(self, base_llm_config: Optional[LLMConfig] = None) -> LLMConfig:
        """Get LLM configuration with stage-specific overrides."""
        if base_llm_config is None:
            base_llm_config = LLMConfig.from_env_and_overrides()

        # Create overrides dict with non-None values
        overrides = {}
        if self.model is not None:
            overrides['model'] = self.model
        if self.provider is not None:
            overrides['provider'] = self.provider
        if self.temperature is not None:
            overrides['temperature'] = self.temperature
        if self.max_tokens is not None:
            overrides['max_tokens'] = self.max_tokens

        # Apply overrides to base config
        config_dict = base_llm_config.model_dump()
        config_dict.update(overrides)

        return LLMConfig(**config_dict)


class ChunkerConfig(BaseModel, ConfigMixin):
    """Configuration for chunker stage."""

    enabled: bool = Field(default=True, description="Enable chunking")

    # Chunking strategy
    chunk_strategy: str = Field(default="semantic", description="Chunking strategy")
    chunk_size: int = Field(default=4000, ge=100, description="Target chunk size")
    overlap: int = Field(default=200, ge=0, description="Chunk overlap")

    # Processing options
    generate_summary: bool = Field(default=True, description="Generate chunk summaries")
    extract_metadata: bool = Field(default=True, description="Extract metadata")

    @classmethod
    def from_env_and_overrides(
        cls,
        overrides: Optional[Dict[str, Any]] = None,
        prefix: Optional[str] = None
    ) -> 'ChunkerConfig':
        """Create configuration from environment variables with overrides."""
        if prefix is None:
            prefix = "MORAG_CHUNKER_"

        # Load from environment variables
        env_config = {}

        # Define field mappings with types
        field_mappings = {
            'enabled': ('ENABLED', bool),
            'chunk_strategy': ('CHUNK_STRATEGY', str),
            'chunk_size': ('CHUNK_SIZE', int),
            'overlap': ('OVERLAP', int),
            'generate_summary': ('GENERATE_SUMMARY', bool),
            'extract_metadata': ('EXTRACT_METADATA', bool),
        }

        for field_name, (env_suffix, field_type) in field_mappings.items():
            env_var = f"{prefix}{env_suffix}"
            env_value = os.environ.get(env_var)

            if env_value is not None:
                try:
                    converted_value = cls._convert_env_value(env_value, field_type)
                    env_config[field_name] = converted_value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {env_value}", error=str(e))

        # Apply overrides
        if overrides:
            env_config.update(overrides)

        return cls(**env_config)


class IngestorConfig(BaseModel, ConfigMixin):
    """Configuration for ingestor stage."""

    enabled: bool = Field(default=True, description="Enable ingestion")

    # Database configuration
    databases: list[str] = Field(default=["qdrant"], description="Target databases")
    batch_size: int = Field(default=50, ge=1, description="Ingestion batch size")

    # Processing options
    generate_embeddings: bool = Field(default=True, description="Generate embeddings")
    validate_data: bool = Field(default=True, description="Validate data before ingestion")

    @classmethod
    def from_env_and_overrides(
        cls,
        overrides: Optional[Dict[str, Any]] = None,
        prefix: Optional[str] = None
    ) -> 'IngestorConfig':
        """Create configuration from environment variables with overrides."""
        if prefix is None:
            prefix = "MORAG_INGESTOR_"

        # Load from environment variables
        env_config = {}

        # Define field mappings with types
        field_mappings = {
            'enabled': ('ENABLED', bool),
            'batch_size': ('BATCH_SIZE', int),
            'generate_embeddings': ('GENERATE_EMBEDDINGS', bool),
            'validate_data': ('VALIDATE_DATA', bool),
        }

        for field_name, (env_suffix, field_type) in field_mappings.items():
            env_var = f"{prefix}{env_suffix}"
            env_value = os.environ.get(env_var)

            if env_value is not None:
                try:
                    converted_value = cls._convert_env_value(env_value, field_type)
                    env_config[field_name] = converted_value
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid value for {env_var}: {env_value}", error=str(e))

        # Handle databases list separately
        databases_env = os.environ.get(f"{prefix}DATABASES")
        if databases_env:
            try:
                # Parse as JSON array or comma-separated list
                if databases_env.startswith('['):
                    import json
                    env_config['databases'] = json.loads(databases_env)
                else:
                    env_config['databases'] = [db.strip() for db in databases_env.split(',')]
            except (ValueError, TypeError) as e:
                logger.warning(f"Invalid value for {prefix}DATABASES: {databases_env}", error=str(e))

        # Apply overrides
        if overrides:
            env_config.update(overrides)

        return cls(**env_config)
