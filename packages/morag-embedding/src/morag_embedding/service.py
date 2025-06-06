"""Embedding service implementation using Google Gemini models."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)
import structlog

from morag_core.config import settings
from morag_core.exceptions import (
    ExternalServiceError,
    RateLimitError,
    TimeoutError,
    QuotaExceededError,
)
from morag_core.models.embedding import EmbeddingResult, BatchEmbeddingResult, SummaryResult
from morag_core.interfaces.service import BaseService, CircuitBreaker

logger = structlog.get_logger(__name__)


class GeminiEmbeddingService(BaseService):
    """Embedding service using Google Gemini models."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        generation_model: Optional[str] = None,
        max_retries: int = 3,
        retry_min_wait: float = 1.0,
        retry_max_wait: float = 10.0,
        rate_limit_per_minute: int = 100,
    ):
        """Initialize the Gemini embedding service.

        Args:
            api_key: Gemini API key (defaults to settings.gemini_api_key)
            embedding_model: Embedding model name (defaults to settings.gemini_embedding_model)
            generation_model: Generation model name (defaults to settings.gemini_generation_model)
            max_retries: Maximum number of retries for API calls
            retry_min_wait: Minimum wait time between retries in seconds
            retry_max_wait: Maximum wait time between retries in seconds
            rate_limit_per_minute: Maximum number of API calls per minute
        """
        self.api_key = api_key or settings.gemini_api_key
        self.embedding_model = embedding_model or settings.gemini_embedding_model
        self.generation_model = generation_model or settings.gemini_generation_model
        self.max_retries = max_retries
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        self.rate_limit_per_minute = rate_limit_per_minute
        self.rate_limit_tokens = rate_limit_per_minute
        self.rate_limit_last_refill = time.time()
        self.rate_limit_refill_rate = rate_limit_per_minute / 60.0  # tokens per second
        self.circuit_breaker = CircuitBreaker()
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the Gemini service.

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)
            
            # Test the API with a simple embedding request
            _ = genai.embed_content(model=self.embedding_model, content="Test")
            
            self._initialized = True
            logger.info(
                "Gemini embedding service initialized",
                embedding_model=self.embedding_model,
                generation_model=self.generation_model,
            )
            return True
        except Exception as e:
            logger.error(
                "Failed to initialize Gemini embedding service",
                error=str(e),
                error_type=e.__class__.__name__,
            )
            return False

    async def shutdown(self) -> None:
        """Shutdown the Gemini service."""
        # No specific shutdown needed for Gemini API
        self._initialized = False
        logger.info("Gemini embedding service shut down")

    async def health_check(self) -> Dict[str, Any]:
        """Check Gemini service health.

        Returns:
            Dictionary with health status information
        """
        status = "healthy" if self._initialized else "unhealthy"
        circuit_status = self.circuit_breaker.state
        
        return {
            "status": status,
            "embedding_model": self.embedding_model,
            "generation_model": self.generation_model,
            "circuit_breaker": circuit_status,
            "rate_limit": {
                "limit": self.rate_limit_per_minute,
                "remaining": self._get_rate_limit_remaining(),
            },
        }

    def _get_rate_limit_remaining(self) -> int:
        """Get remaining rate limit tokens.

        Returns:
            Number of remaining tokens
        """
        current_time = time.time()
        time_elapsed = current_time - self.rate_limit_last_refill
        new_tokens = time_elapsed * self.rate_limit_refill_rate
        
        self.rate_limit_tokens = min(
            self.rate_limit_per_minute,
            self.rate_limit_tokens + new_tokens
        )
        self.rate_limit_last_refill = current_time
        
        return int(self.rate_limit_tokens)

    def _consume_rate_limit_token(self) -> bool:
        """Consume a rate limit token.

        Returns:
            True if token was consumed, False if rate limited
        """
        remaining = self._get_rate_limit_remaining()
        if remaining <= 0:
            return False
        
        self.rate_limit_tokens -= 1
        return True

    @retry(
        retry=retry_if_exception_type((ExternalServiceError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding result

        Raises:
            RateLimitError: If rate limit is exceeded
            ExternalServiceError: If Gemini API call fails
        """
        if not self._initialized:
            await self.initialize()

        if not self.circuit_breaker.is_closed():
            raise ExternalServiceError("Circuit breaker is open")

        if not self._consume_rate_limit_token():
            raise RateLimitError("Rate limit exceeded for embedding generation")

        try:
            # Call Gemini API to generate embedding
            result = genai.embed_content(model=self.embedding_model, content=text)
            
            # Record success for circuit breaker
            self.circuit_breaker.record_success()
            
            # Create and return embedding result
            embedding_result = EmbeddingResult(
                text=text,
                embedding=result["embedding"],
                model=self.embedding_model,
            )
            
            return embedding_result
        except Exception as e:
            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()
            
            # Map exception to appropriate error type
            error_str = str(e)
            if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str or
                "rate limit" in error_str.lower() or "quota" in error_str.lower()):
                raise RateLimitError(f"Gemini API rate limit exceeded: {error_str}")
            elif "timeout" in error_str.lower():
                raise TimeoutError(f"Gemini API timeout: {error_str}")
            else:
                raise ExternalServiceError(f"Gemini API error: {error_str}")

    async def generate_batch_embeddings(self, texts: List[str]) -> BatchEmbeddingResult:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            Batch embedding result

        Raises:
            RateLimitError: If rate limit is exceeded
            ExternalServiceError: If Gemini API calls fail
        """
        if not texts:
            return BatchEmbeddingResult(texts=[], embeddings=[], model=self.embedding_model)

        # Process embeddings with concurrency control
        embeddings = []
        errors = []

        # Process in smaller batches to avoid overwhelming the API
        batch_size = 5  # Reduced batch size for better rate limiting
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            # Process batch sequentially with delays to avoid rate limits
            for text in batch:
                try:
                    result = await self.generate_embedding(text)
                    embeddings.append(result.embedding)
                except Exception as e:
                    errors.append(str(e))
                    # Add a placeholder embedding (zeros)
                    embeddings.append([0.0] * 768)  # Assuming 768-dimensional embeddings

                # Small delay between requests to avoid rate limits
                await asyncio.sleep(0.2)  # 200ms delay between requests

            # Longer delay between batches
            if i + batch_size < len(texts):
                await asyncio.sleep(1.0)  # 1 second delay between batches

        # If all requests failed, raise an exception
        if len(errors) == len(texts):
            raise ExternalServiceError(f"All embedding requests failed: {errors[0]}")

        # Create and return batch result
        return BatchEmbeddingResult(
            texts=texts,
            embeddings=embeddings,
            model=self.embedding_model,
            metadata={"errors": errors} if errors else {},
        )

    @retry(
        retry=retry_if_exception_type((ExternalServiceError, TimeoutError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def generate_summary(self, text: str, max_length: int = 200) -> SummaryResult:
        """Generate summary for text.

        Args:
            text: Text to summarize
            max_length: Maximum length of summary

        Returns:
            Summary result

        Raises:
            RateLimitError: If rate limit is exceeded
            ExternalServiceError: If Gemini API call fails
        """
        if not self._initialized:
            await self.initialize()

        if not self.circuit_breaker.is_closed():
            raise ExternalServiceError("Circuit breaker is open")

        if not self._consume_rate_limit_token():
            raise RateLimitError("Rate limit exceeded for summary generation")

        try:
            # Configure the model
            model = genai.GenerativeModel(self.generation_model)
            
            # Create prompt for summarization
            prompt = f"Summarize the following text in {max_length} characters or less:\n\n{text}"
            
            # Generate summary
            response = model.generate_content(prompt)
            summary = response.text
            
            # Ensure summary is within max length
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            # Record success for circuit breaker
            self.circuit_breaker.record_success()
            
            # Create and return summary result
            return SummaryResult(
                original_text=text,
                summary=summary,
                model=self.generation_model,
            )
        except Exception as e:
            # Record failure for circuit breaker
            self.circuit_breaker.record_failure()
            
            # Map exception to appropriate error type
            error_str = str(e)
            if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str or
                "rate limit" in error_str.lower() or "quota" in error_str.lower()):
                raise RateLimitError(f"Gemini API rate limit exceeded: {error_str}")
            elif "timeout" in error_str.lower():
                raise TimeoutError(f"Gemini API timeout: {error_str}")
            else:
                raise ExternalServiceError(f"Gemini API error: {error_str}")