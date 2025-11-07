"""Agent registry for managing agent instances."""

import threading
import weakref
from typing import Any, Dict, Optional, Type

import structlog

from ..base.agent import BaseAgent
from ..base.config import AgentConfig
from ..base.exceptions import ConfigurationError
from .factory import AgentFactory

logger = structlog.get_logger(__name__)


class AgentRegistry:
    """Registry for managing agent instances."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the registry."""
        if hasattr(self, "_initialized"):
            return

        self._initialized = True
        self.factory = AgentFactory()
        self.instances: Dict[str, weakref.ref] = {}
        self.configs: Dict[str, AgentConfig] = {}
        self.logger = logger.bind(component="agent_registry")

        # Auto-register known agent classes
        self._auto_register_agents()

    def _auto_register_agents(self) -> None:
        """Auto-register known agent classes."""
        try:
            # Import and register extraction agents
            from ..extraction import (
                EntityExtractionAgent,
                FactExtractionAgent,
                KeywordExtractionAgent,
                RelationExtractionAgent,
            )

            self.factory.register_agent_class("fact_extraction", FactExtractionAgent)
            self.factory.register_agent_class(
                "entity_extraction", EntityExtractionAgent
            )
            self.factory.register_agent_class(
                "relation_extraction", RelationExtractionAgent
            )
            self.factory.register_agent_class(
                "keyword_extraction", KeywordExtractionAgent
            )

        except ImportError:
            self.logger.warning("Could not import extraction agents")

        try:
            # Import and register analysis agents
            from ..analysis import (
                ContentAnalysisAgent,
                QueryAnalysisAgent,
                SentimentAnalysisAgent,
                TopicAnalysisAgent,
            )

            self.factory.register_agent_class("query_analysis", QueryAnalysisAgent)
            self.factory.register_agent_class("content_analysis", ContentAnalysisAgent)
            self.factory.register_agent_class(
                "sentiment_analysis", SentimentAnalysisAgent
            )
            self.factory.register_agent_class("topic_analysis", TopicAnalysisAgent)

        except ImportError:
            self.logger.warning("Could not import analysis agents")

        try:
            # Import and register reasoning agents
            from ..reasoning import (
                ContextAnalysisAgent,
                DecisionMakingAgent,
                PathSelectionAgent,
                ReasoningAgent,
            )

            self.factory.register_agent_class("path_selection", PathSelectionAgent)
            self.factory.register_agent_class("reasoning", ReasoningAgent)
            self.factory.register_agent_class("decision_making", DecisionMakingAgent)
            self.factory.register_agent_class("context_analysis", ContextAnalysisAgent)

        except ImportError:
            self.logger.warning("Could not import reasoning agents")

        try:
            # Import and register generation agents
            from ..generation import (
                ExplanationAgent,
                ResponseGenerationAgent,
                SummarizationAgent,
                SynthesisAgent,
            )

            self.factory.register_agent_class("summarization", SummarizationAgent)
            self.factory.register_agent_class(
                "response_generation", ResponseGenerationAgent
            )
            self.factory.register_agent_class("explanation", ExplanationAgent)
            self.factory.register_agent_class("synthesis", SynthesisAgent)

        except ImportError:
            self.logger.warning("Could not import generation agents")

        try:
            # Import and register processing agents
            from ..processing import (
                ChunkingAgent,
                ClassificationAgent,
                FilteringAgent,
                ValidationAgent,
            )

            self.factory.register_agent_class("chunking", ChunkingAgent)
            self.factory.register_agent_class("classification", ClassificationAgent)
            self.factory.register_agent_class("validation", ValidationAgent)
            self.factory.register_agent_class("filtering", FilteringAgent)

        except ImportError:
            self.logger.warning("Could not import processing agents")

    def get_agent(
        self,
        agent_name: str,
        config: Optional[AgentConfig] = None,
        model_override: Optional[str] = None,
        **config_overrides,
    ) -> BaseAgent:
        """Get an agent instance, creating it if necessary.

        Args:
            agent_name: Name of the agent
            config: Optional configuration
            model_override: Optional model override for this specific agent
            **config_overrides: Configuration overrides

        Returns:
            Agent instance
        """
        # Check if we have a cached instance and no overrides
        if agent_name in self.instances and not model_override and not config_overrides:
            agent_ref = self.instances[agent_name]
            agent = agent_ref()
            if agent is not None:
                return agent
            else:
                # Clean up dead reference
                del self.instances[agent_name]

        # Create new instance with overrides
        agent = self.factory.create_agent(
            agent_name, config, model_override=model_override, **config_overrides
        )

        # Store weak reference
        self.instances[agent_name] = weakref.ref(agent)
        self.configs[agent_name] = agent.get_config()

        self.logger.info(f"Created and cached agent: {agent_name}")
        return agent

    def create_agent(
        self,
        agent_name: str,
        config: Optional[AgentConfig] = None,
        model_override: Optional[str] = None,
        **config_overrides,
    ) -> BaseAgent:
        """Create a new agent instance (not cached).

        Args:
            agent_name: Name of the agent
            config: Optional configuration
            model_override: Optional model override for this specific agent
            **config_overrides: Configuration overrides

        Returns:
            New agent instance
        """
        return self.factory.create_agent(
            agent_name, config, model_override=model_override, **config_overrides
        )

    def register_agent_class(
        self, agent_name: str, agent_class: Type[BaseAgent]
    ) -> None:
        """Register an agent class.

        Args:
            agent_name: Name to register under
            agent_class: Agent class
        """
        self.factory.register_agent_class(agent_name, agent_class)

    def list_available_agents(self) -> Dict[str, str]:
        """List all available agents.

        Returns:
            Dictionary mapping agent names to class names
        """
        return self.factory.list_available_agents()

    def clear_cache(self, agent_name: Optional[str] = None) -> None:
        """Clear cached agent instances.

        Args:
            agent_name: Optional specific agent to clear, clears all if None
        """
        if agent_name:
            if agent_name in self.instances:
                del self.instances[agent_name]
                self.logger.info(f"Cleared cache for agent: {agent_name}")
        else:
            self.instances.clear()
            self.logger.info("Cleared all agent cache")

    def get_cached_agents(self) -> Dict[str, str]:
        """Get list of currently cached agents.

        Returns:
            Dictionary mapping agent names to their status
        """
        cached = {}
        for name, ref in self.instances.items():
            agent = ref()
            if agent is not None:
                cached[name] = "active"
            else:
                cached[name] = "dead_reference"

        return cached
