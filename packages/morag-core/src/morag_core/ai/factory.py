"""Factory patterns for creating MoRAG AI agents."""

from typing import Type, TypeVar, Optional, Dict, Any
from .base_agent import MoRAGBaseAgent, AgentConfig
from .providers import GeminiProvider, ProviderConfig, ProviderFactory
import structlog

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=MoRAGBaseAgent)


class AgentFactory:
    """Factory for creating MoRAG AI agents."""
    
    def __init__(self, default_config: Optional[AgentConfig] = None):
        """Initialize the agent factory.
        
        Args:
            default_config: Default configuration for agents
        """
        self.default_config = default_config or AgentConfig()
        self.logger = logger.bind(component="agent_factory")
    
    def create_agent(
        self,
        agent_class: Type[T],
        config: Optional[AgentConfig] = None,
        provider: Optional[GeminiProvider] = None,
        **kwargs
    ) -> T:
        """Create an agent instance.
        
        Args:
            agent_class: The agent class to instantiate
            config: Agent configuration (uses default if not provided)
            provider: Provider instance (creates default if not provided)
            **kwargs: Additional arguments passed to the agent constructor
            
        Returns:
            Agent instance
        """
        # Use provided config or default
        agent_config = config or self.default_config
        
        # Create provider if not provided
        if provider is None:
            provider_config = agent_config.provider_config
            provider = ProviderFactory.create_provider("gemini", provider_config)
        
        self.logger.info(
            "Creating agent",
            agent_class=agent_class.__name__,
            model=agent_config.model,
            provider_available=provider.is_available()
        )
        
        try:
            return agent_class(config=agent_config, provider=provider, **kwargs)
        except Exception as e:
            self.logger.error(
                "Failed to create agent",
                agent_class=agent_class.__name__,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    def create_agent_with_config(
        self,
        agent_class: Type[T],
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        api_key: Optional[str] = None,
        **kwargs
    ) -> T:
        """Create an agent with inline configuration.
        
        Args:
            agent_class: The agent class to instantiate
            model: Model identifier
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            temperature: Model temperature
            max_tokens: Maximum tokens in response
            api_key: API key for the provider
            **kwargs: Additional arguments passed to the agent constructor
            
        Returns:
            Agent instance
        """
        # Build configuration from parameters
        config_dict = {}
        if model is not None:
            config_dict["model"] = model
        if timeout is not None:
            config_dict["timeout"] = timeout
        if max_retries is not None:
            config_dict["max_retries"] = max_retries
        if temperature is not None:
            config_dict["temperature"] = temperature
        if max_tokens is not None:
            config_dict["max_tokens"] = max_tokens
        
        # Build provider configuration
        provider_config_dict = {}
        if api_key is not None:
            provider_config_dict["api_key"] = api_key
        
        # Create configurations
        provider_config = ProviderConfig(**provider_config_dict) if provider_config_dict else None
        if provider_config:
            config_dict["provider_config"] = provider_config
        
        agent_config = AgentConfig(**config_dict)
        
        return self.create_agent(agent_class, config=agent_config, **kwargs)
    
    def get_agent_info(self, agent: MoRAGBaseAgent) -> Dict[str, Any]:
        """Get information about an agent.
        
        Args:
            agent: The agent instance
            
        Returns:
            Dictionary with agent information
        """
        return {
            "class": agent.__class__.__name__,
            "config": agent.config.model_dump(exclude={"provider_config"}),
            "provider": agent.provider.get_provider_info(),
            "result_type": agent.get_result_type().__name__,
            "system_prompt_preview": agent.get_system_prompt()[:100] + "..." 
                if len(agent.get_system_prompt()) > 100 else agent.get_system_prompt()
        }


# Global factory instance
default_factory = AgentFactory()


def create_agent(
    agent_class: Type[T],
    config: Optional[AgentConfig] = None,
    provider: Optional[GeminiProvider] = None,
    **kwargs
) -> T:
    """Convenience function to create an agent using the default factory.
    
    Args:
        agent_class: The agent class to instantiate
        config: Agent configuration
        provider: Provider instance
        **kwargs: Additional arguments passed to the agent constructor
        
    Returns:
        Agent instance
    """
    return default_factory.create_agent(agent_class, config, provider, **kwargs)


def create_agent_with_config(
    agent_class: Type[T],
    model: Optional[str] = None,
    timeout: Optional[int] = None,
    max_retries: Optional[int] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> T:
    """Convenience function to create an agent with inline configuration.
    
    Args:
        agent_class: The agent class to instantiate
        model: Model identifier
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts
        temperature: Model temperature
        max_tokens: Maximum tokens in response
        api_key: API key for the provider
        **kwargs: Additional arguments passed to the agent constructor
        
    Returns:
        Agent instance
    """
    return default_factory.create_agent_with_config(
        agent_class, model, timeout, max_retries, temperature, max_tokens, api_key, **kwargs
    )
