"""LLM client interface for reasoning components."""

import asyncio
import logging
import os
from typing import Dict, List, Optional

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration for LLM client."""

    provider: str = "gemini"  # openai, gemini, anthropic, etc.
    model: str = "gemini-1.5-flash"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 30

    # Retry configuration
    max_retries: int = 8  # Increased for better handling of overload
    base_delay: float = 2.0  # Increased base delay
    max_delay: float = 120.0  # Increased max delay
    exponential_base: float = 2.0
    jitter: bool = True

    # Batch configuration
    batch_size: int = 10  # Number of prompts to batch together
    enable_batching: bool = True  # Enable batch processing
    batch_delay: float = 1.0  # Delay between batch requests
    max_batch_tokens: int = (
        800000  # Max tokens per batch (considering 1M context limit)
    )
    batch_timeout: int = 120  # Timeout for batch requests


class LLMClient:
    """Unified LLM client interface for reasoning components.

    This class provides a simplified interface for LLM interactions,
    wrapping the existing BaseExtractor functionality from morag-graph.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the LLM client.

        Args:
            config: LLM configuration. If None, uses environment variables.
        """
        if config is None:
            config = LLMConfig(
                provider=os.getenv("MORAG_LLM_PROVIDER", "gemini"),
                model=os.getenv("MORAG_GEMINI_MODEL", "gemini-1.5-flash"),
                api_key=os.getenv("GEMINI_API_KEY"),
                temperature=float(os.getenv("MORAG_LLM_TEMPERATURE", "0.1")),
                max_tokens=int(os.getenv("MORAG_LLM_MAX_TOKENS", "2000")),
                max_retries=int(os.getenv("MORAG_LLM_MAX_RETRIES", "8")),
                base_delay=float(os.getenv("MORAG_LLM_BASE_DELAY", "2.0")),
                max_delay=float(os.getenv("MORAG_LLM_MAX_DELAY", "120.0")),
                batch_size=int(os.getenv("MORAG_LLM_BATCH_SIZE", "10")),
                enable_batching=os.getenv("MORAG_ENABLE_LLM_BATCHING", "true").lower()
                == "true",
                batch_delay=float(os.getenv("MORAG_LLM_BATCH_DELAY", "1.0")),
                max_batch_tokens=int(os.getenv("MORAG_LLM_MAX_BATCH_TOKENS", "800000")),
                batch_timeout=int(os.getenv("MORAG_LLM_BATCH_TIMEOUT", "120")),
            )

        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout)
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()

    def get_model(self) -> str:
        """Get the model name for PydanticAI compatibility.

        Returns:
            Model name in PydanticAI format
        """
        # PydanticAI expects provider:model format for Gemini models
        if self.config.provider == "gemini":
            if self.config.model.startswith("google-gla:"):
                return self.config.model
            else:
                return f"google-gla:{self.config.model}"
        elif self.config.provider == "openai":
            return f"openai:{self.config.model}"
        else:
            # For other providers, return as-is and let PydanticAI handle it
            return self.config.model

    async def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate text from a prompt with quota-aware retry.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Temperature for generation (overrides config)
            **kwargs: Additional generation parameters

        Returns:
            Generated text

        Raises:
            Exception: If generation fails after all retries
        """
        try:
            from morag_graph.utils.quota_retry import retry_with_quota_handling

            async def generate_with_retry():
                messages = [{"role": "user", "content": prompt}]
                return await self.generate_from_messages(
                    messages, max_tokens=max_tokens, temperature=temperature, **kwargs
                )

            return await retry_with_quota_handling(
                generate_with_retry, max_retries=15, operation_name="text generation"
            )

        except ImportError:
            # Fallback to original behavior if quota_retry is not available
            messages = [{"role": "user", "content": prompt}]
            return await self.generate_from_messages(
                messages, max_tokens=max_tokens, temperature=temperature, **kwargs
            )

    async def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """Generate text from a list of messages.

        Args:
            messages: List of messages with 'role' and 'content' keys
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Temperature for generation (overrides config)
            **kwargs: Additional generation parameters

        Returns:
            Generated text

        Raises:
            Exception: If generation fails
        """
        # Use provided values or fall back to config
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        if self.config.provider == "openai":
            return await self._call_openai(messages, max_tokens, temperature)
        elif self.config.provider == "gemini":
            return await self._call_gemini(messages, max_tokens, temperature)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    async def _call_openai(
        self, messages: List[Dict[str, str]], max_tokens: int, temperature: float
    ) -> str:
        """Call OpenAI API with exponential backoff retry logic."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        base_url = self.config.base_url or "https://api.openai.com/v1"
        url = f"{base_url}/chat/completions"

        last_error = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await self.client.post(url, headers=headers, json=payload)
                response.raise_for_status()

                data = response.json()

                if attempt > 1:
                    self.logger.info(f"OpenAI API call succeeded on attempt {attempt}")

                return data["choices"][0]["message"]["content"]

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if this is a retryable error
                is_retryable = (
                    "503" in error_str
                    or "overload" in error_str
                    or "rate limit" in error_str
                    or "quota" in error_str
                    or "too many requests" in error_str
                    or "service unavailable" in error_str
                    or "temporarily unavailable" in error_str
                    or "server error" in error_str
                    or (
                        hasattr(e, "response")
                        and hasattr(e.response, "status_code")
                        and e.response.status_code in [503, 429, 500, 502, 504]
                    )
                )

                if attempt < self.config.max_retries and is_retryable:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"OpenAI API call failed with retryable error (attempt {attempt}/{self.config.max_retries}), "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                elif not is_retryable:
                    self.logger.error(
                        f"OpenAI API call failed with non-retryable error: {str(e)}"
                    )
                    break
                else:
                    break

        self.logger.error(
            f"OpenAI API call failed after {self.config.max_retries} attempts"
        )
        raise last_error

    async def _call_gemini(
        self, messages: List[Dict[str, str]], max_tokens: int, temperature: float
    ) -> str:
        """Call Google Gemini API with exponential backoff retry logic."""
        # Convert messages to Gemini format
        gemini_contents = []
        for message in messages:
            role = "user" if message["role"] in ["user", "system"] else "model"
            gemini_contents.append(
                {"role": role, "parts": [{"text": message["content"]}]}
            )

        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        base_url = (
            self.config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        )
        url = f"{base_url}/models/{self.config.model}:generateContent?key={self.config.api_key}"

        headers = {
            "Content-Type": "application/json",
        }

        last_error = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await self.client.post(url, headers=headers, json=payload)
                response.raise_for_status()

                data = response.json()

                if "candidates" not in data or not data["candidates"]:
                    raise ValueError("No candidates in Gemini response")

                candidate = data["candidates"][0]

                if "content" not in candidate or "parts" not in candidate["content"]:
                    raise ValueError("Invalid Gemini response structure")

                if attempt > 1:
                    self.logger.info(f"Gemini API call succeeded on attempt {attempt}")

                return candidate["content"]["parts"][0]["text"]

            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if this is a retryable error
                is_retryable = (
                    "503" in error_str
                    or "overload" in error_str
                    or "rate limit" in error_str
                    or "quota" in error_str
                    or "too many requests" in error_str
                    or "service unavailable" in error_str
                    or "temporarily unavailable" in error_str
                    or "server error" in error_str
                    or (
                        hasattr(e, "response")
                        and hasattr(e.response, "status_code")
                        and e.response.status_code in [503, 429, 500, 502, 504]
                    )
                )

                if attempt < self.config.max_retries and is_retryable:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"Gemini API call failed with retryable error (attempt {attempt}/{self.config.max_retries}), "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                elif not is_retryable:
                    self.logger.error(
                        f"Gemini API call failed with non-retryable error: {str(e)}"
                    )
                    break
                else:
                    break

        self.logger.error(
            f"Gemini API call failed after {self.config.max_retries} attempts"
        )
        raise last_error

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter."""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** (attempt - 1)),
            self.config.max_delay,
        )

        if self.config.jitter:
            import random

            delay *= 0.5 + random.random() * 0.5  # Add 0-50% jitter

        return delay

    async def _process_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Process a batch of prompts in a single API call.

        Args:
            prompts: List of prompts to process in this batch
            max_tokens: Maximum tokens per response
            temperature: Temperature for generation

        Returns:
            List of responses for the batch
        """
        # Estimate token count for the batch
        estimated_tokens = self._estimate_batch_tokens(prompts, max_tokens)

        if estimated_tokens > self.config.max_batch_tokens:
            logger.warning(
                f"Batch token estimate ({estimated_tokens}) exceeds limit "
                f"({self.config.max_batch_tokens}), splitting batch"
            )
            # Split the batch and process recursively
            mid = len(prompts) // 2
            first_half = await self._process_batch(
                prompts[:mid], max_tokens, temperature
            )
            second_half = await self._process_batch(
                prompts[mid:], max_tokens, temperature
            )
            return first_half + second_half

        if self.config.provider == "gemini":
            return await self._process_batch_gemini(prompts, max_tokens, temperature)
        else:
            # For other providers, fall back to individual calls
            logger.warning(
                f"Batch processing not implemented for provider {self.config.provider}, "
                "falling back to individual calls"
            )
            results = []
            for prompt in prompts:
                response = await self.generate_text(prompt, max_tokens, temperature)
                results.append(response)
            return results

    def _estimate_batch_tokens(
        self, prompts: List[str], max_tokens: Optional[int] = None
    ) -> int:
        """Estimate total tokens for a batch of prompts.

        Args:
            prompts: List of prompts
            max_tokens: Maximum tokens per response

        Returns:
            Estimated total token count
        """
        # Rough estimation: 4 characters per token
        input_tokens = sum(len(prompt) // 4 for prompt in prompts)
        output_tokens = len(prompts) * (max_tokens or self.config.max_tokens)

        # Add some overhead for formatting and system messages
        overhead = len(prompts) * 100  # 100 tokens overhead per prompt

        return input_tokens + output_tokens + overhead

    async def _process_batch_gemini(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> List[str]:
        """Process a batch of prompts using Gemini's large context window.

        This method combines multiple prompts into a single request using Gemini's
        1M token context window, with clear delimiters and instructions for
        processing multiple tasks.

        Args:
            prompts: List of prompts to process
            max_tokens: Maximum tokens per response
            temperature: Temperature for generation

        Returns:
            List of responses, one for each prompt
        """
        # Use provided values or fall back to config
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature or self.config.temperature

        # Create a combined prompt with clear delimiters
        delimiter = "=" * 50
        combined_prompt = self._create_batch_prompt(prompts, delimiter)

        # Convert to message format
        messages = [{"role": "user", "content": combined_prompt}]

        # Call Gemini with the combined prompt
        try:
            response = await self._call_gemini(
                messages,
                max_tokens * len(prompts),  # Increase max tokens for batch
                temperature,
            )

            # Parse the response to extract individual answers
            return self._parse_batch_response(response, prompts, delimiter)

        except Exception as e:
            logger.error(f"Batch Gemini call failed: {str(e)}")
            raise

    def _create_batch_prompt(self, prompts: List[str], delimiter: str) -> str:
        """Create a combined prompt for batch processing.

        Args:
            prompts: List of individual prompts
            delimiter: Delimiter to separate prompts and responses

        Returns:
            Combined prompt string
        """
        batch_prompt = f"""I will provide you with {len(prompts)} separate tasks to complete.
Please process each task independently and provide your response for each one.

IMPORTANT INSTRUCTIONS:
1. Process each task completely and independently
2. Provide a clear, complete response for each task
3. Separate your responses using the delimiter: {delimiter}
4. Maintain the same order as the input tasks
5. Do not reference other tasks in your responses
6. Each response should be self-contained and complete

Here are the tasks:

"""

        for i, prompt in enumerate(prompts, 1):
            batch_prompt += f"TASK {i}:\n{prompt}\n\n{delimiter}\n\n"

        batch_prompt += f"""Please provide your responses in the following format:

RESPONSE 1:
[Your complete response to task 1]

{delimiter}

RESPONSE 2:
[Your complete response to task 2]

{delimiter}

... and so on for all {len(prompts)} tasks.
"""

        return batch_prompt

    def _parse_batch_response(
        self, response: str, original_prompts: List[str], delimiter: str
    ) -> List[str]:
        """Parse a batch response into individual responses.

        Args:
            response: The combined response from the LLM
            original_prompts: Original prompts for fallback
            delimiter: Delimiter used to separate responses

        Returns:
            List of individual responses
        """
        try:
            # Split the response by delimiter
            parts = response.split(delimiter)

            # Extract responses (skip the first part which is usually instructions)
            responses = []
            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # Look for "RESPONSE X:" pattern and extract content after it
                if "RESPONSE" in part.upper():
                    lines = part.split("\n")
                    response_content = []
                    found_response_marker = False

                    for line in lines:
                        if "RESPONSE" in line.upper() and ":" in line:
                            found_response_marker = True
                            continue
                        elif found_response_marker:
                            response_content.append(line)

                    if response_content:
                        responses.append("\n".join(response_content).strip())
                elif part and not any(
                    keyword in part.upper()
                    for keyword in ["TASK", "PLEASE", "IMPORTANT"]
                ):
                    # If it doesn't look like instructions, treat it as a response
                    responses.append(part)

            # Ensure we have the right number of responses
            if len(responses) == len(original_prompts):
                return responses
            elif len(responses) > len(original_prompts):
                # Too many responses, take the first N
                logger.warning(
                    f"Got {len(responses)} responses for {len(original_prompts)} prompts, "
                    "taking first {len(original_prompts)}"
                )
                return responses[: len(original_prompts)]
            else:
                # Too few responses, pad with error messages
                logger.warning(
                    f"Got {len(responses)} responses for {len(original_prompts)} prompts, "
                    "padding with error messages"
                )
                while len(responses) < len(original_prompts):
                    responses.append("Error: No response generated for this prompt")
                return responses

        except Exception as e:
            logger.error(f"Failed to parse batch response: {str(e)}")
            # Return error messages for all prompts
            return [f"Error parsing batch response: {str(e)}"] * len(original_prompts)

    async def generate_batch_with_messages(
        self,
        message_lists: List[List[Dict[str, str]]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Generate responses for multiple message conversations using batch processing.

        Args:
            message_lists: List of message lists (each containing role/content dicts)
            max_tokens: Maximum tokens per response
            temperature: Temperature for generation
            batch_size: Override default batch size

        Returns:
            List of generated responses in the same order as message_lists
        """
        # Convert message lists to simple prompts for batch processing
        prompts = []
        for messages in message_lists:
            # Combine messages into a single prompt
            prompt_parts = []
            for message in messages:
                role = message.get("role", "user")
                content = message.get("content", "")
                if role == "system":
                    prompt_parts.append(f"System: {content}")
                elif role == "user":
                    prompt_parts.append(f"User: {content}")
                elif role == "assistant":
                    prompt_parts.append(f"Assistant: {content}")
                else:
                    prompt_parts.append(content)

            prompts.append("\n".join(prompt_parts))

        # Use the regular batch processing
        return await self.generate_batch(prompts, max_tokens, temperature, batch_size)

    async def generate_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        batch_size: Optional[int] = None,
    ) -> List[str]:
        """Generate responses for multiple prompts using batch processing.

        Args:
            prompts: List of prompts to process
            max_tokens: Maximum tokens per response
            temperature: Temperature for generation
            batch_size: Override default batch size

        Returns:
            List of generated responses in the same order as prompts

        Raises:
            Exception: If batch generation fails
        """
        if not prompts:
            return []

        if not self.config.enable_batching:
            # Fall back to individual calls if batching is disabled
            logger.info("LLM batching disabled, processing prompts individually")
            results = []
            for prompt in prompts:
                response = await self.generate(prompt, max_tokens, temperature)
                results.append(response)
            return results

        # Use provided batch size or config default
        effective_batch_size = batch_size or self.config.batch_size

        logger.info(
            f"Processing {len(prompts)} prompts in batches of {effective_batch_size}"
        )

        all_results = []

        # Process prompts in batches with quota-aware retry
        for i in range(0, len(prompts), effective_batch_size):
            batch_prompts = prompts[i : i + effective_batch_size]
            batch_num = i // effective_batch_size + 1

            try:
                # Use quota-aware retry for batch processing
                from morag_graph.utils.quota_retry import retry_with_quota_handling

                async def process_this_batch():
                    return await self._process_batch(
                        batch_prompts, max_tokens, temperature
                    )

                batch_results = await retry_with_quota_handling(
                    process_this_batch,
                    max_retries=15,  # More retries for quota issues
                    operation_name=f"batch processing (batch {batch_num})",
                )

                all_results.extend(batch_results)

                logger.debug(
                    f"Processed batch {batch_num}, "
                    f"completed {len(all_results)}/{len(prompts)} prompts"
                )

                # Add delay between batches if not the last batch
                if i + effective_batch_size < len(prompts):
                    await asyncio.sleep(self.config.batch_delay)

            except Exception as e:
                logger.error(
                    f"Batch processing failed for batch {batch_num} after retries: {str(e)}"
                )

                # Fall back to individual processing with quota handling for this batch
                for j, prompt in enumerate(batch_prompts):
                    try:

                        async def process_individual():
                            return await self.generate(prompt, max_tokens, temperature)

                        response = await retry_with_quota_handling(
                            process_individual,
                            max_retries=10,
                            operation_name=f"individual prompt (batch {batch_num}, item {j+1})",
                        )
                        all_results.append(response)

                    except Exception as individual_error:
                        logger.error(
                            f"Individual prompt processing failed after retries: {str(individual_error)}"
                        )
                        all_results.append(f"Error: {str(individual_error)}")

        return all_results
