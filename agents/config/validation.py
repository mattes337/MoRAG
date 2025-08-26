"""Configuration validation for MoRAG agents."""

from typing import Dict, List, Any, Optional, Set
from pydantic import BaseModel, Field, validator
import structlog

from ..base.config import AgentConfig, PromptConfig, ModelConfig, RetryConfig

logger = structlog.get_logger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


class ConfigValidationResult(BaseModel):
    """Result of configuration validation."""
    
    is_valid: bool = Field(..., description="Whether configuration is valid")
    errors: List[str] = Field(default_factory=list, description="Validation errors")
    warnings: List[str] = Field(default_factory=list, description="Validation warnings")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class ConfigValidator:
    """Validates agent configurations for consistency and best practices."""
    
    def __init__(self):
        """Initialize the configuration validator."""
        self.required_agent_names = {
            "fact_extraction", "entity_extraction", "relation_extraction", 
            "keyword_extraction", "query_analysis", "content_analysis",
            "sentiment_analysis", "topic_analysis", "summarization",
            "path_selection", "reasoning", "response_generation",
            "decision_making", "context_analysis", "explanation",
            "synthesis", "chunking", "classification", "validation", "filtering"
        }
        
        self.valid_output_formats = {"json", "text", "structured"}
        self.valid_domains = {"general", "medical", "legal", "technical", "business", "academic"}
        self.valid_providers = {"gemini", "openai", "anthropic"}
    
    def validate_config(self, config: AgentConfig) -> ConfigValidationResult:
        """Validate a single agent configuration.
        
        Args:
            config: Agent configuration to validate
            
        Returns:
            Validation result with errors and warnings
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Validate basic configuration
        errors.extend(self._validate_basic_config(config))
        
        # Validate prompt configuration
        prompt_errors, prompt_warnings = self._validate_prompt_config(config.prompt)
        errors.extend(prompt_errors)
        warnings.extend(prompt_warnings)
        
        # Validate model configuration
        model_errors, model_warnings = self._validate_model_config(config.model)
        errors.extend(model_errors)
        warnings.extend(model_warnings)
        
        # Validate retry configuration
        retry_errors = self._validate_retry_config(config.retry)
        errors.extend(retry_errors)
        
        # Validate agent-specific configuration
        agent_warnings, agent_suggestions = self._validate_agent_config(config)
        warnings.extend(agent_warnings)
        suggestions.extend(agent_suggestions)
        
        return ConfigValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    def _validate_basic_config(self, config: AgentConfig) -> List[str]:
        """Validate basic configuration fields."""
        errors = []
        
        if not config.name:
            errors.append("Agent name is required")
        elif config.name not in self.required_agent_names:
            errors.append(f"Unknown agent name: {config.name}")
        
        if not config.description:
            errors.append("Agent description is required")
        elif len(config.description) < 10:
            errors.append("Agent description should be at least 10 characters")
        
        if config.timeout <= 0:
            errors.append("Timeout must be positive")
        elif config.timeout > 300:
            errors.append("Timeout should not exceed 300 seconds")
        
        if config.cache_ttl < 0:
            errors.append("Cache TTL cannot be negative")
        
        return errors
    
    def _validate_prompt_config(self, prompt: PromptConfig) -> tuple[List[str], List[str]]:
        """Validate prompt configuration."""
        errors = []
        warnings = []
        
        if prompt.output_format not in self.valid_output_formats:
            errors.append(f"Invalid output format: {prompt.output_format}")
        
        if prompt.domain not in self.valid_domains:
            warnings.append(f"Non-standard domain: {prompt.domain}")
        
        if prompt.min_confidence < 0 or prompt.min_confidence > 1:
            errors.append("Min confidence must be between 0 and 1")
        
        if prompt.max_output_length and prompt.max_output_length < 100:
            warnings.append("Max output length seems very small")
        
        return errors, warnings
    
    def _validate_model_config(self, model: ModelConfig) -> tuple[List[str], List[str]]:
        """Validate model configuration."""
        errors = []
        warnings = []
        
        if not model.api_key:
            warnings.append("API key not set (required for actual usage)")
        
        if model.temperature < 0 or model.temperature > 2:
            errors.append("Temperature must be between 0 and 2")
        
        if model.max_tokens <= 0:
            errors.append("Max tokens must be positive")
        elif model.max_tokens > 32000:
            warnings.append("Very high max tokens setting")
        
        if model.top_p is not None and (model.top_p < 0 or model.top_p > 1):
            errors.append("Top-p must be between 0 and 1")

        if model.top_k is not None and model.top_k < 0:
            errors.append("Top-k cannot be negative")
        
        return errors, warnings
    
    def _validate_retry_config(self, retry: RetryConfig) -> List[str]:
        """Validate retry configuration."""
        errors = []
        
        if retry.max_retries < 0:
            errors.append("Max retries cannot be negative")
        elif retry.max_retries > 10:
            errors.append("Max retries should not exceed 10")
        
        if retry.base_delay <= 0:
            errors.append("Base delay must be positive")
        
        if retry.max_delay <= retry.base_delay:
            errors.append("Max delay must be greater than base delay")
        
        if retry.exponential_base <= 1:
            errors.append("Exponential base must be greater than 1")
        
        return errors
    
    def _validate_agent_config(self, config: AgentConfig) -> tuple[List[str], List[str]]:
        """Validate agent-specific configuration."""
        warnings = []
        suggestions = []
        
        agent_config = config.agent_config or {}
        
        # Check for common configuration patterns
        if config.name == "entity_extraction":
            if "entity_types" not in agent_config:
                warnings.append("Entity extraction should specify entity types")
            elif not isinstance(agent_config["entity_types"], list):
                warnings.append("Entity types should be a list")
        
        if config.name == "chunking":
            if "max_chunk_size" not in agent_config:
                warnings.append("Chunking agent should specify max chunk size")
            elif agent_config.get("max_chunk_size", 0) < 100:
                warnings.append("Max chunk size seems very small")
        
        if config.name == "fact_extraction":
            if "max_facts" not in agent_config:
                suggestions.append("Consider setting max_facts limit")
        
        # Check for unused or deprecated configuration keys
        common_keys = {
            "entity_types", "max_chunk_size", "min_chunk_size", "overlap",
            "max_facts", "include_offsets", "normalize_entities", "min_entity_length"
        }
        
        for key in agent_config.keys():
            if key not in common_keys:
                suggestions.append(f"Uncommon configuration key: {key}")
        
        return warnings, suggestions
    
    def validate_all_configs(self, configs: Dict[str, AgentConfig]) -> Dict[str, ConfigValidationResult]:
        """Validate multiple agent configurations.
        
        Args:
            configs: Dictionary of agent configurations
            
        Returns:
            Dictionary of validation results
        """
        results = {}
        
        for agent_name, config in configs.items():
            results[agent_name] = self.validate_config(config)
        
        return results
    
    def get_validation_summary(self, results: Dict[str, ConfigValidationResult]) -> Dict[str, Any]:
        """Get summary of validation results.
        
        Args:
            results: Dictionary of validation results
            
        Returns:
            Summary statistics
        """
        total_configs = len(results)
        valid_configs = sum(1 for r in results.values() if r.is_valid)
        total_errors = sum(len(r.errors) for r in results.values())
        total_warnings = sum(len(r.warnings) for r in results.values())
        total_suggestions = sum(len(r.suggestions) for r in results.values())
        
        return {
            "total_configs": total_configs,
            "valid_configs": valid_configs,
            "invalid_configs": total_configs - valid_configs,
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "total_suggestions": total_suggestions,
            "validation_rate": valid_configs / total_configs if total_configs > 0 else 0
        }


# Global validator instance
_config_validator = None


def get_config_validator() -> ConfigValidator:
    """Get the global configuration validator instance.
    
    Returns:
        ConfigValidator instance
    """
    global _config_validator
    if _config_validator is None:
        _config_validator = ConfigValidator()
    return _config_validator


def validate_config(config: AgentConfig) -> ConfigValidationResult:
    """Validate an agent configuration.
    
    Args:
        config: Agent configuration to validate
        
    Returns:
        Validation result
    """
    return get_config_validator().validate_config(config)


def validate_all_configs(configs: Dict[str, AgentConfig]) -> Dict[str, ConfigValidationResult]:
    """Validate multiple agent configurations.
    
    Args:
        configs: Dictionary of agent configurations
        
    Returns:
        Dictionary of validation results
    """
    return get_config_validator().validate_all_configs(configs)
