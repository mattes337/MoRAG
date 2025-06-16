"""Base extractor for LLM-based entity and relation extraction."""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union

import httpx
from pydantic import BaseModel, Field

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
        """Call OpenAI API.
        
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
        
        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling OpenAI API: {e}")
            raise
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            raise
    
    async def _call_gemini(self, messages: List[Dict[str, str]]) -> str:
        """Call Google Gemini API.
        
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
        
        try:
            response = await self.client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            data = response.json()
            
            if "candidates" not in data or not data["candidates"]:
                raise ValueError("No candidates in Gemini response")
            
            candidate = data["candidates"][0]
            
            if "content" not in candidate or "parts" not in candidate["content"]:
                raise ValueError("Invalid Gemini response structure")
            
            return candidate["content"]["parts"][0]["text"]
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling Gemini API: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error calling Gemini API: {e}")
            raise
    
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
            except json.JSONDecodeError:
                # If that fails, try to fix common issues
                response = self._fix_common_json_issues(response)
                return json.loads(response)
            
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