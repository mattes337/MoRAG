"""Base agent classes for MoRAG AI agents using PydanticAI."""

import asyncio
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional, Dict, Any, Type, Union
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
import structlog

from .exceptions import AgentError, ValidationError, RetryExhaustedError
from .providers import GeminiProvider, ProviderConfig

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


class MoRAGBaseAgent(Generic[T], ABC):
    """Base class for all MoRAG AI agents using PydanticAI."""
    
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
        
    @property
    def agent(self) -> Agent:
        """Get the PydanticAI agent instance, creating it if necessary."""
        if self._agent is None:
            self._agent = self._create_agent()
        return self._agent
    
    def _create_agent(self) -> Agent:
        """Create the PydanticAI agent instance."""
        try:
            return Agent(
                model=self.config.model,
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
    
    async def run(
        self,
        user_prompt: str,
        deps: Optional[Any] = None,
        **kwargs
    ) -> T:
        """Run the agent with the given prompt.
        
        Args:
            user_prompt: The user prompt to process
            deps: Optional dependencies for the agent
            **kwargs: Additional arguments passed to the agent
            
        Returns:
            The structured result from the agent
            
        Raises:
            AgentError: If the agent execution fails
            ValidationError: If the result validation fails
            RetryExhaustedError: If all retry attempts are exhausted
        """
        for attempt in range(self.config.max_retries + 1):
            try:
                self.logger.info(
                    "Running agent",
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries + 1,
                    user_prompt=user_prompt[:100] + "..." if len(user_prompt) > 100 else user_prompt
                )
                
                # Run the agent with timeout
                result = await asyncio.wait_for(
                    self.agent.run(user_prompt, deps=deps, **kwargs),
                    timeout=self.config.timeout
                )
                
                # Validate the result
                validated_result = self._validate_result(result.data)
                
                self.logger.info("Agent execution successful", attempt=attempt + 1)
                return validated_result
                
            except asyncio.TimeoutError as e:
                self.logger.warning(
                    "Agent execution timeout",
                    attempt=attempt + 1,
                    timeout=self.config.timeout
                )
                if attempt == self.config.max_retries:
                    raise AgentError(f"Agent execution timed out after {self.config.timeout}s") from e
                    
            except ValidationError:
                # Don't retry validation errors
                raise
                
            except Exception as e:
                self.logger.warning(
                    "Agent execution failed",
                    attempt=attempt + 1,
                    error=str(e),
                    error_type=type(e).__name__
                )
                if attempt == self.config.max_retries:
                    raise AgentError(f"Agent execution failed after {attempt + 1} attempts: {e}") from e
                
                # Wait before retry
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise RetryExhaustedError(f"All {self.config.max_retries + 1} attempts exhausted")
    
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
