"""Base agent classes for MoRAG AI agents using PydanticAI."""

import asyncio
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Dict, Any, Type, Union
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import structlog

from .exceptions import AgentError, ValidationError, RetryExhaustedError
from .providers import GeminiProvider, ProviderConfig, OutlinesProvider

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseModel)


class AgentConfig(BaseModel):
    """Configuration for AI agents."""

    model: str = Field(default="google-gla:gemini-1.5-flash", description="Model identifier")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Model temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens in response")
    provider_config: Optional[ProviderConfig] = Field(default=None, description="Provider-specific configuration")

    # Outlines configuration
    outlines_provider: str = Field(
        default="gemini",
        description="Provider for Outlines integration (gemini, openai)"
    )


class MoRAGBaseAgent(Generic[T], ABC):
    """Base class for all MoRAG AI agents with built-in Outlines support."""

    def __init__(
        self,
        config: Optional[AgentConfig] = None,
        provider: Optional[GeminiProvider] = None
    ):
        """Initialize the base agent.

        Args:
            config: Agent configuration
            provider: Provider instance (optional, will create default if not provided)
        """
        self.config = config or AgentConfig()
        self.provider = provider or GeminiProvider(self.config.provider_config)
        self._agent: Optional[Agent] = None
        self.logger = logger.bind(agent_class=self.__class__.__name__)

        # Initialize Outlines provider (required for all agents)
        try:
            self.outlines_provider = OutlinesProvider(
                config=self.config.provider_config,
                provider=self.config.outlines_provider,
                model=self._extract_model_name(self.config.model)
            )
            self.logger.info(
                "Outlines provider initialized",
                provider=self.config.outlines_provider,
                model=self.config.model
            )
        except Exception as e:
            self.logger.error(
                "Failed to initialize Outlines provider",
                error=str(e)
            )
            raise AgentError(f"Outlines provider initialization failed: {e}") from e

    @property
    def agent(self) -> Agent:
        """Get the PydanticAI agent instance, creating it if necessary."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent

    def _create_agent(self) -> Agent:
        """Create the PydanticAI agent instance."""
        try:
            # Set the API key in environment for PydanticAI if available
            import os
            if self.provider.api_key and not os.getenv("GOOGLE_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = self.provider.api_key

            # Create agent with proper parameter names for current PydanticAI version
            return Agent(
                self.config.model,
                result_type=self.get_result_type(),
                system_prompt=self.get_system_prompt(),
                deps_type=self.get_deps_type(),
            )
        except Exception as e:
            raise AgentError(f"Failed to create agent: {e}") from e

    @abstractmethod
    def get_result_type(self) -> Type[T]:
        """Return the Pydantic model for structured output.

        Returns:
            The Pydantic model class for this agent's output
        """
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this agent.

        Returns:
            The system prompt string
        """
        pass

    def get_deps_type(self) -> Optional[Type]:
        """Return the dependencies type for this agent.

        Returns:
            The dependencies type or None if no dependencies are needed
        """
        return None

    def _extract_model_name(self, model: str) -> str:
        """Extract the base model name from the full model identifier.

        Args:
            model: The full model identifier (e.g., "google-gla:gemini-1.5-flash")

        Returns:
            The base model name (e.g., "gemini-1.5-flash")
        """
        if ":" in model:
            return model.split(":", 1)[1]
        return model

    async def run(
        self,
        user_prompt: str,
        deps: Optional[Any] = None,
        **kwargs
    ) -> T:
        """Run the agent with structured generation using Outlines.

        Args:
            user_prompt: The user prompt to process
            deps: Optional dependencies for the agent
            **kwargs: Additional arguments passed to the generator

        Returns:
            The structured result from the agent

        Raises:
            AgentError: If the agent execution fails
            ValidationError: If the result validation fails
        """
        try:
            self.logger.info(
                "Running structured generation with Outlines",
                user_prompt=user_prompt[:100] + "..." if len(user_prompt) > 100 else user_prompt
            )

            # Get the Outlines generator for our result type
            generator = self.outlines_provider.get_generator(self.get_result_type())

            # Generate structured output
            result_str = generator(user_prompt, **kwargs)

            # Parse the JSON result into our Pydantic model
            result = self.get_result_type().model_validate_json(result_str)

            self.logger.info("Structured generation successful")
            return result

        except Exception as e:
            self.logger.error(
                "Structured generation failed",
                error=str(e)
            )
            raise AgentError(f"Structured generation failed: {e}") from e

    def run_sync(
        self,
        user_prompt: str,
        deps: Optional[Any] = None,
        **kwargs
    ) -> T:
        """Synchronous version of run().

        Args:
            user_prompt: The user prompt to process
            deps: Optional dependencies for the agent
            **kwargs: Additional arguments passed to the agent

        Returns:
            The structured result from the agent
        """
        return asyncio.run(self.run(user_prompt, deps, **kwargs))

    def _validate_result(self, result: Any) -> T:
        """Validate the agent result.

        Args:
            result: The raw result from the agent

        Returns:
            The validated result

        Raises:
            ValidationError: If validation fails
        """
        try:
            if isinstance(result, self.get_result_type()):
                return result
            elif isinstance(result, dict):
                return self.get_result_type()(**result)
            else:
                return self.get_result_type().model_validate(result)
        except Exception as e:
            raise ValidationError(f"Result validation failed: {e}") from e

    def add_tool(self, func):
        """Add a tool function to the agent.

        Args:
            func: The tool function to add
        """
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent.tool(func)

    def add_system_prompt_part(self, func):
        """Add a system prompt part to the agent.

        Args:
            func: The system prompt function to add
        """
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent.system_prompt(func)

    def is_outlines_available(self) -> bool:
        """Check if Outlines is available and configured.

        Returns:
            True if Outlines is available, False otherwise
        """
        return (
            self.outlines_provider is not None and
            self.outlines_provider.is_available()
        )

    def get_generation_info(self) -> Dict[str, Any]:
        """Get information about the generation configuration.

        Returns:
            Dictionary with generation configuration information
        """
        return {
            "structured_generation_enabled": True,  # Always enabled now
            "outlines_available": self.is_outlines_available(),
            "outlines_provider": self.config.outlines_provider,
            "model": self.config.model
        }
