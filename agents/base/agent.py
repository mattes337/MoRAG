"""Base agent class for MoRAG agents.

This module provides backward compatibility by importing from the new location.
The actual implementation is now in morag_core.ai.base_agent.
"""

# Import from the new location for backward compatibility
try:
    from morag_core.ai.base_agent import AgentConfig
    from morag_core.ai.base_agent import MoRAGBaseAgent as BaseAgent
except ImportError:
    # If morag_core is not available, provide a minimal stub
    from abc import ABC, abstractmethod
    from typing import Any, Generic, Optional, Type, TypeVar

    from pydantic import BaseModel

    T = TypeVar("T", bound=BaseModel)

    class BaseAgent(Generic[T], ABC):
        """Minimal stub for BaseAgent when morag_core is not available.

        This is a generic class that can be subscripted with a Pydantic model type.
        """

        def __init__(self, config: Optional[Any] = None):
            """Initialize base agent."""
            self.config = config

        @abstractmethod
        def get_result_type(self) -> Type[T]:
            """Return the result type for this agent."""
            pass

        @abstractmethod
        def get_system_prompt(self) -> str:
            """Return the system prompt for this agent."""
            pass

        def run_sync(self, user_prompt: str, **kwargs) -> T:
            """Synchronous run method."""
            raise NotImplementedError(
                "BaseAgent stub - install morag_core for full functionality"
            )

    # Minimal AgentConfig stub
    class AgentConfig(BaseModel):
        """Minimal stub for AgentConfig."""

        model: str = "gemini-1.5-flash"
        temperature: float = 0.1
        max_retries: int = 3


__all__ = ["BaseAgent", "AgentConfig"]
