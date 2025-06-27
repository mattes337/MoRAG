"""Base classes for extraction functionality."""

import json
import logging
import asyncio
import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class LLMConfig(BaseModel):
    """Configuration for LLM-based extraction."""

    provider: str = "gemini"  # openai, gemini, anthropic, etc.
    model: str = "gemini-1.5-flash"  # gemini-1.5-flash, gemini-1.5-pro, gpt-3.5-turbo, etc.
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 30

    # Retry configuration
    max_retries: int = 5
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: bool = True


class BaseExtractor(ABC):
    """Base class for LLM-based extraction.
    
    This class provides common functionality for entity and relation extraction
    using Large Language Models. It handles API communication, prompt formatting,
    and response parsing.
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize the extractor.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.client.aclose()

    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter.

        Args:
            attempt: Current attempt number (1-based)

        Returns:
            Delay in seconds
        """
        delay = self.config.base_delay * (self.config.exponential_base ** (attempt - 1))
        delay = min(delay, self.config.max_delay)

        if self.config.jitter:
            # Add random jitter (±25% of delay)
            jitter = delay * 0.25 * (2 * random.random() - 1)
            delay += jitter

        return max(0, delay)

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if an error is retryable.

        Args:
            error: Exception to check

        Returns:
            True if error is retryable
        """
        error_str = str(error).lower()

        # Retryable HTTP status codes and error messages
        retryable_conditions = [
            "503",  # Service Unavailable
            "502",  # Bad Gateway
            "504",  # Gateway Timeout
            "429",  # Too Many Requests
            "500",  # Internal Server Error (sometimes retryable)
            "overloaded",
            "rate limit",
            "quota",
            "timeout",
            "connection",
            "network",
            "unavailable"
        ]

        return any(condition in error_str for condition in retryable_conditions)
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for the extraction task.
        
        Returns:
            System prompt string
        """
        pass
    
    @abstractmethod
    def get_user_prompt(self, text: str, **kwargs) -> str:
        """Get the user prompt for the extraction task.
        
        Args:
            text: Text to extract from
            **kwargs: Additional arguments
            
        Returns:
            User prompt string
        """
        pass
    
    @abstractmethod
    def parse_response(self, response: str) -> Any:
        """Parse the LLM response.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed extraction result
        """
        pass
    
    async def call_llm(self, messages: List[Dict[str, str]]) -> str:
        """Call the LLM API.
        
        Args:
            messages: List of messages for the LLM
            
        Returns:
            LLM response text
            
        Raises:
            Exception: If API call fails
        """
        if self.config.provider == "openai":
            return await self._call_openai(messages)
        elif self.config.provider == "gemini":
            return await self._call_gemini(messages)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")
    
    async def _call_openai(self, messages: List[Dict[str, str]]) -> str:
        """Call OpenAI API with exponential backoff retry logic.

        Args:
            messages: List of messages for the LLM

        Returns:
            LLM response text
        """
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        base_url = self.config.base_url or "https://api.openai.com/v1"
        url = f"{base_url}/chat/completions"

        last_error = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await self.client.post(url, headers=headers, json=payload)
                response.raise_for_status()

                data = response.json()

                # Success! Return the result
                if attempt > 1:
                    logger.info(f"OpenAI API call succeeded on attempt {attempt}")

                return data["choices"][0]["message"]["content"]

            except Exception as e:
                last_error = e

                # Log the error
                if isinstance(e, httpx.HTTPStatusError):
                    logger.warning(f"HTTP error calling OpenAI API (attempt {attempt}/{self.config.max_retries}): {e}")
                else:
                    logger.warning(f"Error calling OpenAI API (attempt {attempt}/{self.config.max_retries}): {e}")

                # Check if this is the last attempt or if error is not retryable
                if attempt >= self.config.max_retries or not self._is_retryable_error(e):
                    break

                # Calculate delay and wait before retry
                delay = self._calculate_delay(attempt)
                logger.info(f"Retrying OpenAI API call in {delay:.2f} seconds (attempt {attempt + 1}/{self.config.max_retries})")
                await asyncio.sleep(delay)

        # All retries exhausted, raise the last error
        logger.error(f"OpenAI API call failed after {self.config.max_retries} attempts")
        raise last_error
    
    async def _call_gemini(self, messages: List[Dict[str, str]]) -> str:
        """Call Google Gemini API with exponential backoff retry logic.

        Args:
            messages: List of messages for the LLM

        Returns:
            LLM response text
        """
        # Convert messages to Gemini format
        gemini_contents = []

        for message in messages:
            role = message["role"]
            content = message["content"]

            if role == "system":
                # Gemini doesn't have system role, prepend to first user message
                if gemini_contents and gemini_contents[-1]["role"] == "user":
                    gemini_contents[-1]["parts"][0]["text"] = content + "\n\n" + gemini_contents[-1]["parts"][0]["text"]
                else:
                    gemini_contents.append({
                        "role": "user",
                        "parts": [{"text": content}]
                    })
            elif role == "user":
                gemini_contents.append({
                    "role": "user",
                    "parts": [{"text": content}]
                })
            elif role == "assistant":
                gemini_contents.append({
                    "role": "model",
                    "parts": [{"text": content}]
                })

        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": self.config.temperature,
                "maxOutputTokens": self.config.max_tokens,
            }
        }

        base_url = self.config.base_url or "https://generativelanguage.googleapis.com/v1beta"
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

                # Success! Return the result
                if attempt > 1:
                    logger.info(f"Gemini API call succeeded on attempt {attempt}")

                return candidate["content"]["parts"][0]["text"]

            except Exception as e:
                last_error = e

                # Log the error with response details if available
                if isinstance(e, httpx.HTTPStatusError):
                    logger.warning(f"HTTP error calling Gemini API (attempt {attempt}/{self.config.max_retries}): {e}")
                    if hasattr(e.response, 'text'):
                        logger.warning(f"Response: {e.response.text}")
                else:
                    logger.warning(f"Error calling Gemini API (attempt {attempt}/{self.config.max_retries}): {e}")

                # Check if this is the last attempt or if error is not retryable
                if attempt >= self.config.max_retries or not self._is_retryable_error(e):
                    break

                # Calculate delay and wait before retry
                delay = self._calculate_delay(attempt)
                logger.info(f"Retrying Gemini API call in {delay:.2f} seconds (attempt {attempt + 1}/{self.config.max_retries})")
                await asyncio.sleep(delay)

        # All retries exhausted, raise the last error
        logger.error(f"Gemini API call failed after {self.config.max_retries} attempts")
        raise last_error
    
    async def extract(self, text: str, **kwargs) -> Any:
        """Extract entities or relations from text.
        
        Args:
            text: Text to extract from
            **kwargs: Additional arguments
            
        Returns:
            Extraction result
        """
        if not text or not text.strip():
            return []
        
        system_prompt = self.get_system_prompt()
        user_prompt = self.get_user_prompt(text, **kwargs)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        try:
            response = await self.call_llm(messages)
            return self.parse_response(response, text=text)
            
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            return []
    
    def parse_json_response(self, response: str) -> Union[Dict, List]:
        """Parse JSON response from LLM with improved error handling.
        
        Args:
            response: Raw LLM response
            
        Returns:
            Parsed JSON data
            
        Raises:
            ValueError: If response is not valid JSON
        """
        try:
            # Try to find JSON in the response
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            response = response.strip()
            
            # Try to parse as-is first
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parse failed: {e}")
                logger.debug(f"Original response: {response[:1000]}{'...' if len(response) > 1000 else ''}")
                
                # Try to fix common issues
                fixed_response = self._fix_common_json_issues(response)
                try:
                    return json.loads(fixed_response)
                except json.JSONDecodeError as e2:
                    logger.warning(f"Fixed JSON parse failed: {e2}")
                    logger.debug(f"Fixed response: {fixed_response[:1000]}{'...' if len(fixed_response) > 1000 else ''}")
                    
                    # Try to extract partial valid JSON
                    partial_result = self._extract_partial_json(response)
                    if partial_result:
                        logger.info(f"Extracted {len(partial_result)} valid objects from partial JSON")
                        return partial_result
                    
                    raise e2
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response: {response[:500]}{'...' if len(response) > 500 else ''}")
            raise ValueError(f"Invalid JSON response: {e}")
    
    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues with enhanced error recovery.
        
        Args:
            json_str: JSON string with potential issues
            
        Returns:
            Fixed JSON string
        """
        import re
        
        # Remove any trailing text after the JSON (common with LLM responses)
        # Find the last closing bracket/brace and truncate there
        last_bracket = max(json_str.rfind(']'), json_str.rfind('}'))
        if last_bracket != -1:
            json_str = json_str[:last_bracket + 1]
        
        # Fix trailing commas before closing brackets/braces
        json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
        
        # Fix unquoted property names (basic cases)
        # This regex looks for word characters followed by colon, not already quoted
        json_str = re.sub(r'(?<!["]) \b([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'"\1":', json_str)
        
        # Fix single quotes to double quotes (but be careful with apostrophes in values)
        # Only replace single quotes that are clearly property delimiters
        json_str = re.sub(r"(?<!\\)'([^']*)'(?=\s*:)", r'"\1"', json_str)  # Property names
        json_str = re.sub(r"(?<!\\)'([^']*)'(?=\s*[,}\]])", r'"\1"', json_str)  # String values
        
        # Fix missing commas between objects/arrays
        json_str = re.sub(r'}\s*{', r'},{', json_str)
        json_str = re.sub(r']\s*\[', r'],[', json_str)
        json_str = re.sub(r'}\s*\[', r'},[', json_str)
        json_str = re.sub(r']\s*{', r'],[', json_str)
        
        # Fix incomplete objects at the end (common when LLM hits token limit)
        # If we have an incomplete object, try to close it
        if json_str.strip().endswith(','):
            json_str = json_str.rstrip().rstrip(',')
        
        # Handle incomplete strings at the end
        if json_str.count('"') % 2 != 0:
            # Odd number of quotes, likely incomplete string
            json_str = json_str.rstrip() + '"'
        
        # Count brackets and braces to ensure they're balanced
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        
        # Add missing closing brackets/braces
        if open_brackets > close_brackets:
            json_str += ']' * (open_brackets - close_brackets)
        if open_braces > close_braces:
            json_str += '}' * (open_braces - close_braces)
        
        return json_str
    
    def _extract_partial_json(self, json_str: str) -> List[Dict]:
        """Extract valid JSON objects from a malformed JSON string.
        
        Args:
            json_str: Malformed JSON string
            
        Returns:
            List of valid JSON objects extracted from the string
        """
        import re
        
        valid_objects = []
        
        # Try to find complete JSON objects in the string
        # Look for patterns like {"key": "value", ...}
        object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        
        matches = re.finditer(object_pattern, json_str)
        
        for match in matches:
            obj_str = match.group()
            try:
                # Try to parse this individual object
                obj = json.loads(obj_str)
                valid_objects.append(obj)
            except json.JSONDecodeError:
                # Try to fix this individual object
                try:
                    fixed_obj = self._fix_common_json_issues(obj_str)
                    obj = json.loads(fixed_obj)
                    valid_objects.append(obj)
                except json.JSONDecodeError:
                    # Skip this object if it can't be fixed
                    continue
        
        return valid_objects