"""Agent factory for creating and managing agent instances."""

from typing import Type, Dict, Any, Optional, TypeVar, Generic
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig
from ..base.exceptions import ConfigurationError
from ..config.manager import AgentConfigManager

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseAgent)


class AgentFactory:
    """Factory for creating agent instances."""
    
    def __init__(self, config_manager: Optional[AgentConfigManager] = None):
        """Initialize the agent factory.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config_manager = config_manager or AgentConfigManager()
        self.agent_classes: Dict[str, Type[BaseAgent]] = {}
        self.logger = logger.bind(component="agent_factory")
    
    def register_agent_class(self, agent_name: str, agent_class: Type[BaseAgent]) -> None:
        """Register an agent class.
        
        Args:
            agent_name: Name to register the agent under
            agent_class: Agent class to register
        """
        self.agent_classes[agent_name] = agent_class
        self.logger.info(f"Registered agent class: {agent_name}")
    
    def create_agent(
        self,
        agent_name: str,
        config: Optional[AgentConfig] = None,
        **config_overrides
    ) -> BaseAgent:
        """Create an agent instance.
        
        Args:
            agent_name: Name of the agent to create
            config: Optional configuration to use
            **config_overrides: Configuration overrides
            
        Returns:
            Agent instance
            
        Raises:
            ConfigurationError: If agent class not found or creation fails
        """
        if agent_name not in self.agent_classes:
            raise ConfigurationError(f"Unknown agent: {agent_name}")
        
        agent_class = self.agent_classes[agent_name]
        
        # Get configuration
        if config is None:
            config = self.config_manager.get_config(agent_name)
        
        # Apply overrides
        if config_overrides:
            config = config.copy(deep=True)
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    config.set_agent_config(key, value)
        
        try:
            agent = agent_class(config)
            self.logger.info(f"Created agent: {agent_name}")
            return agent
            
        except Exception as e:
            raise ConfigurationError(f"Failed to create agent {agent_name}: {e}") from e
    
    def create_agent_with_config(
        self,
        agent_class: Type[T],
        config: Optional[AgentConfig] = None,
        **config_overrides
    ) -> T:
        """Create an agent instance with a specific class and config.
        
        Args:
            agent_class: Agent class to instantiate
            config: Optional configuration
            **config_overrides: Configuration overrides
            
        Returns:
            Agent instance
        """
        # Use class name as agent name if not registered
        agent_name = agent_class.__name__
        
        if config is None:
            try:
                config = self.config_manager.get_config(agent_name)
            except ConfigurationError:
                # Create default config if none exists
                config = AgentConfig(name=agent_name)
        
        # Apply overrides
        if config_overrides:
            config = config.copy(deep=True)
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    config.set_agent_config(key, value)
        
        try:
            return agent_class(config)
        except Exception as e:
            raise ConfigurationError(f"Failed to create agent {agent_name}: {e}") from e
    
    def list_available_agents(self) -> Dict[str, str]:
        """List all available agent classes.
        
        Returns:
            Dictionary mapping agent names to class names
        """
        return {
            name: cls.__name__
            for name, cls in self.agent_classes.items()
        }
    
    def get_agent_class(self, agent_name: str) -> Type[BaseAgent]:
        """Get an agent class by name.
        
        Args:
            agent_name: Name of the agent
            
        Returns:
            Agent class
            
        Raises:
            ConfigurationError: If agent not found
        """
        if agent_name not in self.agent_classes:
            raise ConfigurationError(f"Unknown agent: {agent_name}")
        
        return self.agent_classes[agent_name]
    
    def validate_agent_config(self, agent_name: str, config: AgentConfig) -> bool:
        """Validate configuration for an agent.
        
        Args:
            agent_name: Name of the agent
            config: Configuration to validate
            
        Returns:
            True if valid
            
        Raises:
            ConfigurationError: If validation fails
        """
        # Basic validation
        self.config_manager.validate_config(config)
        
        # Agent-specific validation
        if agent_name in self.agent_classes:
            agent_class = self.agent_classes[agent_name]
            
            # Check if agent class has custom validation
            if hasattr(agent_class, 'validate_config'):
                return agent_class.validate_config(config)
        
        return True
