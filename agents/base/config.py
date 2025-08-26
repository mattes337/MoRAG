"""Configuration classes for MoRAG agents."""

import os
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class ModelProvider(str, Enum):
    """Supported LLM providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class ConfidenceLevel(str, Enum):
    """Confidence levels for agent outputs."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ModelConfig(BaseModel):
    """Configuration for the underlying LLM model."""
    
    provider: ModelProvider = Field(default=ModelProvider.GEMINI, description="LLM provider")
    model: str = Field(default="gemini-1.5-flash", description="Model identifier")
    api_key: Optional[str] = Field(default=None, description="API key for the provider")
    base_url: Optional[str] = Field(default=None, description="Custom base URL for API")
    
    # Generation parameters
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: int = Field(default=4000, ge=1, description="Maximum tokens in response")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: Optional[int] = Field(default=None, ge=1, description="Top-k sampling")
    
    @validator('api_key', pre=True, always=True)
    def get_api_key(cls, v, values):
        """Get API key from environment if not provided."""
        if v is not None:
            return v
        
        provider = values.get('provider', ModelProvider.GEMINI)
        if provider == ModelProvider.GEMINI:
            return os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        elif provider == ModelProvider.OPENAI:
            return os.getenv('OPENAI_API_KEY')
        elif provider == ModelProvider.ANTHROPIC:
            return os.getenv('ANTHROPIC_API_KEY')
        
        return None


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""
    
    max_retries: int = Field(default=3, ge=0, description="Maximum retry attempts")
    base_delay: float = Field(default=1.0, ge=0.0, description="Base delay between retries")
    max_delay: float = Field(default=60.0, ge=0.0, description="Maximum delay between retries")
    exponential_base: float = Field(default=2.0, ge=1.0, description="Exponential backoff base")
    jitter: bool = Field(default=True, description="Add random jitter to delays")


class PromptConfig(BaseModel):
    """Configuration for prompt generation and behavior."""
    
    # Prompt behavior
    include_examples: bool = Field(default=True, description="Include few-shot examples")
    include_context: bool = Field(default=True, description="Include contextual information")
    include_instructions: bool = Field(default=True, description="Include detailed instructions")
    
    # Output format
    output_format: str = Field(default="json", description="Expected output format")
    strict_json: bool = Field(default=True, description="Enforce strict JSON output")
    include_confidence: bool = Field(default=True, description="Include confidence scores")
    
    # Domain and language
    domain: str = Field(default="general", description="Domain context for prompts")
    language: str = Field(default="en", description="Target language")
    
    # Customization
    custom_instructions: Optional[str] = Field(default=None, description="Custom instructions to append")
    custom_examples: Optional[List[Dict[str, Any]]] = Field(default=None, description="Custom examples")
    
    # Quality control
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")
    max_output_length: Optional[int] = Field(default=None, ge=1, description="Maximum output length")


class AgentConfig(BaseModel):
    """Main configuration class for agents."""
    
    # Agent identity
    name: str = Field(..., description="Agent name")
    version: str = Field(default="1.0.0", description="Agent version")
    description: Optional[str] = Field(default=None, description="Agent description")
    
    # Model configuration
    model: ModelConfig = Field(default_factory=ModelConfig, description="Model configuration")
    
    # Retry configuration
    retry: RetryConfig = Field(default_factory=RetryConfig, description="Retry configuration")
    
    # Prompt configuration
    prompt: PromptConfig = Field(default_factory=PromptConfig, description="Prompt configuration")
    
    # Execution configuration
    timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")
    enable_caching: bool = Field(default=True, description="Enable response caching")
    cache_ttl: int = Field(default=3600, ge=0, description="Cache TTL in seconds")
    
    # Logging and monitoring
    log_level: str = Field(default="INFO", description="Logging level")
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_tracing: bool = Field(default=False, description="Enable request tracing")
    
    # Agent-specific configuration
    agent_config: Dict[str, Any] = Field(default_factory=dict, description="Agent-specific configuration")
    
    class Config:
        """Pydantic configuration."""
        use_enum_values = True
        validate_assignment = True
        extra = "allow"  # Allow additional fields for agent-specific config
    
    def get_agent_config(self, key: str, default: Any = None) -> Any:
        """Get agent-specific configuration value."""
        return self.agent_config.get(key, default)
    
    def set_agent_config(self, key: str, value: Any) -> None:
        """Set agent-specific configuration value."""
        self.agent_config[key] = value
    
    def update_from_env(self, prefix: str = "MORAG_AGENT_") -> None:
        """Update configuration from environment variables."""
        for key, value in os.environ.items():
            if key.startswith(prefix):
                config_key = key[len(prefix):].lower()
                if hasattr(self, config_key):
                    # Try to convert to appropriate type
                    try:
                        if isinstance(getattr(self, config_key), bool):
                            setattr(self, config_key, value.lower() in ('true', '1', 'yes', 'on'))
                        elif isinstance(getattr(self, config_key), int):
                            setattr(self, config_key, int(value))
                        elif isinstance(getattr(self, config_key), float):
                            setattr(self, config_key, float(value))
                        else:
                            setattr(self, config_key, value)
                    except (ValueError, TypeError):
                        # If conversion fails, store as string
                        setattr(self, config_key, value)
