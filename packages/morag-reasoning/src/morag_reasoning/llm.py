"""LLM client interface for reasoning components."""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

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
        **kwargs
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (overrides config)
            temperature: Temperature for generation (overrides config)
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
            
        Raises:
            Exception: If generation fails
        """
        messages = [{"role": "user", "content": prompt}]
        return await self.generate_from_messages(
            messages, max_tokens=max_tokens, temperature=temperature, **kwargs
        )
    
    async def generate_from_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
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
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int, 
        temperature: float
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
                    "503" in error_str or
                    "overload" in error_str or
                    "rate limit" in error_str or
                    "quota" in error_str or
                    "too many requests" in error_str or
                    "service unavailable" in error_str or
                    "temporarily unavailable" in error_str or
                    "server error" in error_str or
                    (hasattr(e, 'response') and hasattr(e.response, 'status_code') and
                     e.response.status_code in [503, 429, 500, 502, 504])
                )

                if attempt < self.config.max_retries and is_retryable:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"OpenAI API call failed with retryable error (attempt {attempt}/{self.config.max_retries}), "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                elif not is_retryable:
                    self.logger.error(f"OpenAI API call failed with non-retryable error: {str(e)}")
                    break
                else:
                    break
        
        self.logger.error(f"OpenAI API call failed after {self.config.max_retries} attempts")
        raise last_error
    
    async def _call_gemini(
        self, 
        messages: List[Dict[str, str]], 
        max_tokens: int, 
        temperature: float
    ) -> str:
        """Call Google Gemini API with exponential backoff retry logic."""
        # Convert messages to Gemini format
        gemini_contents = []
        for message in messages:
            role = "user" if message["role"] in ["user", "system"] else "model"
            gemini_contents.append({
                "role": role,
                "parts": [{"text": message["content"]}]
            })
        
        payload = {
            "contents": gemini_contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
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
                
                if attempt > 1:
                    self.logger.info(f"Gemini API call succeeded on attempt {attempt}")
                
                return candidate["content"]["parts"][0]["text"]
                
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Check if this is a retryable error
                is_retryable = (
                    "503" in error_str or
                    "overload" in error_str or
                    "rate limit" in error_str or
                    "quota" in error_str or
                    "too many requests" in error_str or
                    "service unavailable" in error_str or
                    "temporarily unavailable" in error_str or
                    "server error" in error_str or
                    (hasattr(e, 'response') and hasattr(e.response, 'status_code') and
                     e.response.status_code in [503, 429, 500, 502, 504])
                )

                if attempt < self.config.max_retries and is_retryable:
                    delay = self._calculate_delay(attempt)
                    self.logger.warning(
                        f"Gemini API call failed with retryable error (attempt {attempt}/{self.config.max_retries}), "
                        f"retrying in {delay:.2f}s: {str(e)}"
                    )
                    await asyncio.sleep(delay)
                elif not is_retryable:
                    self.logger.error(f"Gemini API call failed with non-retryable error: {str(e)}")
                    break
                else:
                    break
        
        self.logger.error(f"Gemini API call failed after {self.config.max_retries} attempts")
        raise last_error
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter."""
        delay = min(
            self.config.base_delay * (self.config.exponential_base ** (attempt - 1)),
            self.config.max_delay
        )
        
        if self.config.jitter:
            import random
            delay *= (0.5 + random.random() * 0.5)  # Add 0-50% jitter
        
        return delay
