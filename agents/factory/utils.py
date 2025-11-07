"""Utility functions for agent creation and management."""

from typing import Type, Optional, TypeVar
import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig
from .registry import AgentRegistry

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseAgent)

# Global registry instance
_registry = None


def get_registry() -> AgentRegistry:
    """Get the global agent registry instance."""
    global _registry
    if _registry is None:
        _registry = AgentRegistry()
    return _registry


def create_agent(
    agent_name: str,
    config: Optional[AgentConfig] = None,
    model_override: Optional[str] = None,
    **config_overrides
) -> BaseAgent:
    """Create an agent instance.

    Args:
        agent_name: Name of the agent to create
        config: Optional configuration
        model_override: Optional model override for this specific agent
        **config_overrides: Configuration overrides

    Returns:
        Agent instance
    """
    registry = get_registry()
    return registry.create_agent(agent_name, config, model_override=model_override, **config_overrides)


def get_agent(
    agent_name: str,
    config: Optional[AgentConfig] = None,
    model_override: Optional[str] = None,
    **config_overrides
) -> BaseAgent:
    """Get an agent instance (cached if available).

    Args:
        agent_name: Name of the agent
        config: Optional configuration
        model_override: Optional model override for this specific agent
        **config_overrides: Configuration overrides

    Returns:
        Agent instance
    """
    registry = get_registry()
    return registry.get_agent(agent_name, config, model_override=model_override, **config_overrides)


def register_agent(agent_name: str, agent_class: Type[BaseAgent]) -> None:
    """Register an agent class.

    Args:
        agent_name: Name to register under
        agent_class: Agent class to register
    """
    registry = get_registry()
    registry.register_agent_class(agent_name, agent_class)


def list_agents() -> dict:
    """List all available agents.

    Returns:
        Dictionary of available agents
    """
    registry = get_registry()
    return registry.list_available_agents()


def clear_agent_cache(agent_name: Optional[str] = None) -> None:
    """Clear agent cache.

    Args:
        agent_name: Optional specific agent to clear
    """
    registry = get_registry()
    registry.clear_cache(agent_name)


def create_agent_with_config(
    agent_class: Type[T],
    config: Optional[AgentConfig] = None,
    **config_overrides
) -> T:
    """Create an agent with a specific class and configuration.

    Args:
        agent_class: Agent class to instantiate
        config: Optional configuration
        **config_overrides: Configuration overrides

    Returns:
        Agent instance
    """
    registry = get_registry()
    return registry.factory.create_agent_with_config(agent_class, config, **config_overrides)
