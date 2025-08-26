"""Base agent class for MoRAG agents."""

import asyncio
import time
import hashlib
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, TypeVar, Generic, List, Union
from pydantic import BaseModel
import structlog

try:
    from jinja2 import Template, Environment, BaseLoader
except ImportError:
    # Fallback if jinja2 not available
    Template = None
    Environment = None
    BaseLoader = None

from .config import AgentConfig, ModelProvider
from .template import PromptTemplate, ConfigurablePromptTemplate, GlobalPromptLoader
from .response_parser import LLMResponseParser
from .exceptions import (
    AgentError,
    ConfigurationError,
    ValidationError,
    RetryExhaustedError,
    ModelError,
    TimeoutError,
)

logger = structlog.get_logger(__name__)

T = TypeVar('T', bound=BaseModel)


class BaseAgent(Generic[T], ABC):
    """Base class for all MoRAG agents."""
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """Initialize the agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config or self._get_default_config()
        self.logger = logger.bind(
            agent=self.__class__.__name__,
            version=self.config.version
        )
        self._client = None
        self._cache = {} if self.config.enable_caching else None
        
        # Validate configuration
        self._validate_config()
        
        # Initialize prompt template
        self._template = self._create_template()
        
        self.logger.info("Agent initialized", config=self.config.dict(exclude={'model': {'api_key'}}))
    
    @abstractmethod
    def _get_default_config(self) -> AgentConfig:
        """Get default configuration for this agent.
        
        Returns:
            Default agent configuration
        """
        pass
    
    def _create_template(self) -> PromptTemplate:
        """Create the prompt template for this agent.

        Returns:
            Prompt template instance
        """
        # Load prompts from global YAML file
        prompt_loader = GlobalPromptLoader()
        prompts = prompt_loader.get_prompts(self.config.name)

        # Create template with loaded prompts
        return ConfigurablePromptTemplate(
            self.config.prompt,
            prompts["system_prompt"],
            prompts["user_prompt"]
        )
    
    @abstractmethod
    def get_result_type(self) -> Type[T]:
        """Get the expected result type for this agent.
        
        Returns:
            Pydantic model class for the result
        """
        pass
    
    def _validate_config(self) -> None:
        """Validate the agent configuration.
        
        Raises:
            ConfigurationError: If configuration is invalid
        """
        if not self.config.model.api_key:
            raise ConfigurationError(f"API key required for provider {self.config.model.provider}")
        
        if self.config.timeout <= 0:
            raise ConfigurationError("Timeout must be positive")
        
        if self.config.retry.max_retries < 0:
            raise ConfigurationError("Max retries cannot be negative")
    
    async def _get_client(self):
        """Get or create the LLM client."""
        if self._client is None:
            self._client = await self._create_client()
        return self._client
    
    async def _create_client(self):
        """Create the appropriate LLM client based on configuration."""
        if self.config.model.provider == ModelProvider.GEMINI:
            return await self._create_gemini_client()
        elif self.config.model.provider == ModelProvider.OPENAI:
            return await self._create_openai_client()
        else:
            raise ConfigurationError(f"Unsupported provider: {self.config.model.provider}")
    
    async def _create_gemini_client(self):
        """Create Gemini client."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.config.model.api_key)
            return genai.GenerativeModel(self.config.model.model)
        except ImportError:
            raise ConfigurationError("google-generativeai package required for Gemini provider")
    
    async def _create_openai_client(self):
        """Create OpenAI client."""
        try:
            import openai
            return openai.AsyncOpenAI(
                api_key=self.config.model.api_key,
                base_url=self.config.model.base_url
            )
        except ImportError:
            raise ConfigurationError("openai package required for OpenAI provider")
    
    def _generate_cache_key(self, prompt: str, **kwargs) -> str:
        """Generate cache key for the request.
        
        Args:
            prompt: The prompt string
            **kwargs: Additional parameters
            
        Returns:
            Cache key string
        """
        # Create a hash of the prompt and relevant config
        content = {
            'prompt': prompt,
            'model': self.config.model.model,
            'temperature': self.config.model.temperature,
            'max_tokens': self.config.model.max_tokens,
            **kwargs
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.md5(content_str.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[T]:
        """Get cached result if available and not expired.
        
        Args:
            cache_key: Cache key
            
        Returns:
            Cached result or None
        """
        if not self._cache or cache_key not in self._cache:
            return None
        
        cached_item = self._cache[cache_key]
        if time.time() - cached_item['timestamp'] > self.config.cache_ttl:
            del self._cache[cache_key]
            return None
        
        return cached_item['result']
    
    def _cache_result(self, cache_key: str, result: T) -> None:
        """Cache the result.
        
        Args:
            cache_key: Cache key
            result: Result to cache
        """
        if self._cache is not None:
            self._cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
    
    async def _call_model(self, prompt: Dict[str, str]) -> str:
        """Call the underlying model with retry logic.
        
        Args:
            prompt: Dictionary with 'system' and 'user' prompts
            
        Returns:
            Model response
            
        Raises:
            ModelError: If model call fails
            TimeoutError: If call times out
        """
        client = await self._get_client()
        
        for attempt in range(self.config.retry.max_retries + 1):
            try:
                if self.config.model.provider == ModelProvider.GEMINI:
                    return await self._call_gemini(client, prompt)
                elif self.config.model.provider == ModelProvider.OPENAI:
                    return await self._call_openai(client, prompt)
                else:
                    raise ModelError(f"Unsupported provider: {self.config.model.provider}")
                    
            except asyncio.TimeoutError:
                if attempt == self.config.retry.max_retries:
                    raise TimeoutError(f"Model call timed out after {self.config.timeout}s")
                await self._wait_for_retry(attempt)
                
            except Exception as e:
                if attempt == self.config.retry.max_retries:
                    raise ModelError(f"Model call failed: {e}") from e
                await self._wait_for_retry(attempt)
        
        raise RetryExhaustedError(f"All {self.config.retry.max_retries + 1} attempts exhausted")
    
    async def _call_gemini(self, client, prompt: Dict[str, str]) -> str:
        """Call Gemini model."""
        full_prompt = f"{prompt['system']}\n\n{prompt['user']}"
        
        response = await asyncio.wait_for(
            client.generate_content_async(
                full_prompt,
                generation_config={
                    'temperature': self.config.model.temperature,
                    'max_output_tokens': self.config.model.max_tokens,
                    'top_p': self.config.model.top_p,
                    'top_k': self.config.model.top_k,
                }
            ),
            timeout=self.config.timeout
        )
        
        return response.text
    
    async def _call_openai(self, client, prompt: Dict[str, str]) -> str:
        """Call OpenAI model."""
        messages = [
            {"role": "system", "content": prompt['system']},
            {"role": "user", "content": prompt['user']}
        ]
        
        response = await asyncio.wait_for(
            client.chat.completions.create(
                model=self.config.model.model,
                messages=messages,
                temperature=self.config.model.temperature,
                max_tokens=self.config.model.max_tokens,
                top_p=self.config.model.top_p,
            ),
            timeout=self.config.timeout
        )
        
        return response.choices[0].message.content
    
    async def _wait_for_retry(self, attempt: int) -> None:
        """Wait before retry with exponential backoff.
        
        Args:
            attempt: Current attempt number (0-based)
        """
        delay = min(
            self.config.retry.base_delay * (self.config.retry.exponential_base ** attempt),
            self.config.retry.max_delay
        )
        
        if self.config.retry.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        await asyncio.sleep(delay)
    
    def _validate_result(self, raw_result: str) -> T:
        """Validate and parse the model result using robust parsing.

        Args:
            raw_result: Raw result from the model

        Returns:
            Validated and parsed result

        Raises:
            ValidationError: If validation fails
        """
        context = f"{self.config.name}_validation"
        strict_json = self.config.prompt.output_format == "json"

        return LLMResponseParser.parse_and_validate(
            response=raw_result,
            result_type=self.get_result_type(),
            context=context,
            strict_json=strict_json
        )

    def parse_json_response(self, response: str, fallback_value: Optional[Any] = None) -> Any:
        """Parse JSON from LLM response with robust error handling.

        Args:
            response: Raw LLM response text
            fallback_value: Value to return if parsing fails

        Returns:
            Parsed JSON data or fallback value
        """
        context = f"{self.config.name}_json_parsing"
        return LLMResponseParser.parse_json_response(
            response=response,
            fallback_value=fallback_value,
            context=context
        )

    def extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract the first valid JSON object from text.

        Args:
            text: Text that may contain JSON

        Returns:
            First valid JSON object found, or None
        """
        return LLMResponseParser.extract_json_from_text(text)

    def clean_response(self, response: str) -> str:
        """Clean LLM response for better parsing.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned response text
        """
        return LLMResponseParser.clean_response(response)

    async def execute(self, user_input: str, **kwargs) -> T:
        """Execute the agent with the given input.

        Args:
            user_input: User input/query
            **kwargs: Additional parameters for prompt generation

        Returns:
            Validated result from the agent

        Raises:
            AgentError: If execution fails
        """
        start_time = time.time()

        try:
            # Generate cache key if caching is enabled
            cache_key = None
            if self.config.enable_caching:
                cache_key = self._generate_cache_key(user_input, **kwargs)
                cached_result = self._get_cached_result(cache_key)
                if cached_result:
                    self.logger.info("Returning cached result", cache_key=cache_key)
                    return cached_result

            # Generate prompts
            prompts = self._template.generate_full_prompt(user_input, **kwargs)

            # Call the model
            raw_result = await self._call_model(prompts)

            # Validate and parse result
            result = self._validate_result(raw_result)

            # Cache result if enabled
            if cache_key:
                self._cache_result(cache_key, result)

            # Log execution metrics
            execution_time = time.time() - start_time
            self.logger.info(
                "Agent execution completed",
                execution_time=execution_time,
                input_length=len(user_input),
                cached=cache_key is not None and cache_key in (self._cache or {})
            )

            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(
                "Agent execution failed",
                error=str(e),
                error_type=type(e).__name__,
                execution_time=execution_time
            )
            raise

    def execute_sync(self, user_input: str, **kwargs) -> T:
        """Synchronous version of execute.

        Args:
            user_input: User input/query
            **kwargs: Additional parameters

        Returns:
            Validated result from the agent
        """
        return asyncio.run(self.execute(user_input, **kwargs))

    async def batch_execute(self, inputs: List[str], **kwargs) -> List[T]:
        """Execute the agent on multiple inputs.

        Args:
            inputs: List of user inputs
            **kwargs: Additional parameters for all executions

        Returns:
            List of results
        """
        if not inputs:
            return []

        self.logger.info(f"Starting batch execution for {len(inputs)} inputs")

        # Limit concurrency to avoid overwhelming the API
        semaphore = asyncio.Semaphore(3)

        async def execute_single(input_text: str) -> T:
            async with semaphore:
                return await self.execute(input_text, **kwargs)

        results = await asyncio.gather(
            *[execute_single(input_text) for input_text in inputs],
            return_exceptions=True
        )

        # Handle exceptions in results
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.warning(f"Input {i} failed", error=str(result))
                # Create a default result or re-raise based on configuration
                raise result
            else:
                valid_results.append(result)

        self.logger.info(f"Batch execution completed for {len(valid_results)} inputs")
        return valid_results

    def get_config(self) -> AgentConfig:
        """Get the agent configuration.

        Returns:
            Agent configuration
        """
        return self.config

    def update_config(self, **kwargs) -> None:
        """Update agent configuration.

        Args:
            **kwargs: Configuration updates
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.set_agent_config(key, value)

        # Re-validate configuration
        self._validate_config()

        # Recreate template if prompt config changed
        if any(key.startswith('prompt') for key in kwargs.keys()):
            self._template = self._create_template()

    def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = {
            'agent_name': self.__class__.__name__,
            'version': self.config.version,
            'cache_enabled': self.config.enable_caching,
            'cache_size': len(self._cache) if self._cache else 0,
        }

        return metrics
