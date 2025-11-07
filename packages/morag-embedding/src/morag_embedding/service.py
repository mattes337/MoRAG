"""Embedding service implementation using Google Gemini models."""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

import google.generativeai as genai
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    stop_never,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)
import structlog

# Import agents framework - required
try:
    from agents import get_agent
except ImportError:
    raise ImportError("Agents framework is required. Please install the agents package.")

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


def get_retry_decorator_for_rate_limits():
    """Get retry decorator based on configuration."""
    from morag_core.config import settings

    if settings.retry_indefinitely:
        # Indefinite retries for rate limits with exponential backoff
        return retry(
            retry=retry_if_exception_type(RateLimitError),
            stop=stop_never,  # Never stop retrying for rate limits
            wait=wait_exponential(
                multiplier=settings.retry_base_delay,
                max=settings.retry_max_delay,
                exp_base=settings.retry_exponential_base
            ),
            reraise=True,
        )
    else:
        # Limited retries (legacy behavior)
        return retry(
            retry=retry_if_exception_type((ExternalServiceError, TimeoutError, RateLimitError)),
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            reraise=True,
        )


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
        batch_size: Optional[int] = None,
        enable_batch_embedding: Optional[bool] = None,
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
            batch_size: Number of texts to process in a single batch API call
            enable_batch_embedding: Whether to use batch embedding API (recommended)
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

        # Batch embedding configuration
        self.batch_size = max(1, min(batch_size or settings.embedding_batch_size, 100))  # Clamp between 1 and 100
        self.enable_batch_embedding = enable_batch_embedding if enable_batch_embedding is not None else settings.enable_batch_embedding

    async def initialize(self) -> bool:
        """Initialize the Gemini service.

        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Configure the Gemini API
            genai.configure(api_key=self.api_key)

            # Test the API with a simple embedding request
            # Ensure model name has proper prefix for new SDK
            model_name = self.embedding_model
            if not model_name.startswith(('models/', 'tunedModels/')):
                model_name = f"models/{model_name}"
            _ = genai.embed_content(model=model_name, content="Test")

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
            raise ExternalServiceError("Circuit breaker is open", "gemini")

        if not self._consume_rate_limit_token():
            raise RateLimitError("Rate limit exceeded for embedding generation")

        # Apply dynamic retry decorator based on configuration
        retry_decorator = get_retry_decorator_for_rate_limits()

        @retry_decorator
        async def _generate_with_retry():
            try:
                # Call Gemini API to generate embedding
                # Ensure model name has proper prefix for new SDK
                model_name = self.embedding_model
                if not model_name.startswith(('models/', 'tunedModels/')):
                    model_name = f"models/{model_name}"
                result = genai.embed_content(model=model_name, content=text)

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
                    raise ExternalServiceError(f"Gemini API error: {error_str}", "gemini")

        return await _generate_with_retry()

    async def _generate_batch_embeddings_native(self, texts: List[str]) -> BatchEmbeddingResult:
        """Generate embeddings using Gemini's native batch API.

        Args:
            texts: List of texts to embed (should be <= batch_size)

        Returns:
            Batch embedding result

        Raises:
            RateLimitError: If rate limit is exceeded
            ExternalServiceError: If Gemini API call fails
        """
        if not texts:
            return BatchEmbeddingResult(texts=[], embeddings=[], model=self.embedding_model)

        if not self._initialized:
            await self.initialize()

        if not self.circuit_breaker.is_closed():
            raise ExternalServiceError("Circuit breaker is open", "gemini")

        if not self._consume_rate_limit_token():
            raise RateLimitError("Rate limit exceeded for batch embedding generation")

        # Apply dynamic retry decorator based on configuration
        retry_decorator = get_retry_decorator_for_rate_limits()

        @retry_decorator
        async def _generate_batch_with_retry():
            try:
                # Call Gemini API with multiple contents for batch embedding
                # Ensure model name has proper prefix for new SDK
                model_name = self.embedding_model
                if not model_name.startswith(('models/', 'tunedModels/')):
                    model_name = f"models/{model_name}"
                result = genai.embed_content(
                    model=model_name,
                    content=texts  # Pass list of texts directly
                )

                # Record success for circuit breaker
                self.circuit_breaker.record_success()

                # Extract embeddings from batch result
                embeddings = []
                if hasattr(result, 'embedding') and isinstance(result.embedding, list):
                    # Single embedding returned (shouldn't happen with multiple texts)
                    embeddings = [result.embedding] * len(texts)
                elif hasattr(result, 'embeddings') and isinstance(result.embeddings, list):
                    # Multiple embeddings returned
                    embeddings = result.embeddings
                elif isinstance(result, dict):
                    # Handle dict response format
                    if 'embeddings' in result:
                        embeddings = result['embeddings']
                    elif 'embedding' in result:
                        embeddings = [result['embedding']] * len(texts)
                else:
                    # Fallback: try to extract from result structure
                    embeddings = [getattr(result, 'embedding', [])] * len(texts)

                # Ensure we have the right number of embeddings
                if len(embeddings) != len(texts):
                    logger.warning(
                        "Batch embedding count mismatch",
                        expected=len(texts),
                        received=len(embeddings),
                        texts_sample=texts[:3] if len(texts) > 3 else texts
                    )
                    # Pad or truncate as needed
                    if len(embeddings) < len(texts):
                        # Pad with zeros if we got fewer embeddings
                        zero_embedding = [0.0] * (len(embeddings[0]) if embeddings else 768)
                        embeddings.extend([zero_embedding] * (len(texts) - len(embeddings)))
                    else:
                        # Truncate if we got more embeddings
                        embeddings = embeddings[:len(texts)]

                # Create and return batch result
                return BatchEmbeddingResult(
                    texts=texts,
                    embeddings=embeddings,
                    model=self.embedding_model,
                    metadata={"batch_size": len(texts), "method": "native_batch"}
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
                    raise ExternalServiceError(f"Gemini API error: {error_str}", "gemini")

        return await _generate_batch_with_retry()

    async def generate_batch_embeddings(self, texts: List[str]) -> BatchEmbeddingResult:
        """Generate embeddings for multiple texts using batch API when possible.

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

        # Use batch embedding if enabled and supported
        if self.enable_batch_embedding:
            try:
                return await self._generate_batch_embeddings_optimized(texts)
            except Exception as e:
                logger.warning(
                    "Batch embedding failed, falling back to sequential processing",
                    error=str(e),
                    error_type=e.__class__.__name__
                )
                # Fall back to sequential processing
                return await self._generate_batch_embeddings_sequential(texts)
        else:
            # Use sequential processing
            return await self._generate_batch_embeddings_sequential(texts)

    async def _generate_batch_embeddings_optimized(self, texts: List[str]) -> BatchEmbeddingResult:
        """Generate embeddings using optimized batch processing.

        Args:
            texts: List of texts to embed

        Returns:
            Batch embedding result
        """
        all_embeddings = []
        all_errors = []
        processed_texts = []

        # Process texts in batches using the configured batch size
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i + self.batch_size]

            try:
                # Use native batch API for this chunk
                batch_result = await self._generate_batch_embeddings_native(batch_texts)

                all_embeddings.extend(batch_result.embeddings)
                processed_texts.extend(batch_result.texts)

                # Add any errors from metadata
                if batch_result.metadata and "errors" in batch_result.metadata:
                    all_errors.extend(batch_result.metadata["errors"])

                logger.debug(
                    "Processed batch successfully",
                    batch_size=len(batch_texts),
                    total_processed=len(processed_texts),
                    total_texts=len(texts)
                )

                # Small delay between batches to be respectful to the API
                if i + self.batch_size < len(texts):
                    await asyncio.sleep(0.1)  # 100ms delay between batches

            except Exception as e:
                logger.error(
                    "Batch processing failed for chunk",
                    batch_start=i,
                    batch_size=len(batch_texts),
                    error=str(e)
                )
                # Add error for each text in the failed batch
                for text in batch_texts:
                    all_errors.append(str(e))
                    # Add placeholder embedding (zeros)
                    all_embeddings.append([0.0] * 768)
                    processed_texts.append(text)

        # If all requests failed, raise an exception
        if len(all_errors) == len(texts):
            raise ExternalServiceError(f"All batch embedding requests failed: {all_errors[0]}", "gemini")

        # Create and return batch result
        return BatchEmbeddingResult(
            texts=processed_texts,
            embeddings=all_embeddings,
            model=self.embedding_model,
            metadata={
                "errors": all_errors if all_errors else [],
                "method": "optimized_batch",
                "batch_size": self.batch_size,
                "total_batches": (len(texts) + self.batch_size - 1) // self.batch_size
            }
        )

    async def _generate_batch_embeddings_sequential(self, texts: List[str]) -> BatchEmbeddingResult:
        """Generate embeddings using sequential processing (fallback method).

        Args:
            texts: List of texts to embed

        Returns:
            Batch embedding result
        """
        embeddings = []
        errors = []

        # Process in smaller batches to avoid overwhelming the API
        fallback_batch_size = 5  # Smaller batch size for sequential processing
        for i in range(0, len(texts), fallback_batch_size):
            batch = texts[i:i+fallback_batch_size]

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
            if i + fallback_batch_size < len(texts):
                await asyncio.sleep(1.0)  # 1 second delay between batches

        # If all requests failed, raise an exception
        if len(errors) == len(texts):
            raise ExternalServiceError(f"All sequential embedding requests failed: {errors[0]}", "gemini")

        # Create and return batch result
        return BatchEmbeddingResult(
            texts=texts,
            embeddings=embeddings,
            model=self.embedding_model,
            metadata={
                "errors": errors if errors else [],
                "method": "sequential_fallback"
            }
        )

    async def generate_summary(self, text: str, max_length: int = 200, language: Optional[str] = None) -> SummaryResult:
        """Generate summary for text.

        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            language: Language code for the summary (e.g., 'en', 'de', 'fr')

        Returns:
            Summary result

        Raises:
            RateLimitError: If rate limit is exceeded
            ExternalServiceError: If Gemini API call fails
        """
        if not self._initialized:
            await self.initialize()

        if not self.circuit_breaker.is_closed():
            raise ExternalServiceError("Circuit breaker is open", "gemini")

        if not self._consume_rate_limit_token():
            raise RateLimitError("Rate limit exceeded for summary generation")

        # Apply dynamic retry decorator based on configuration
        retry_decorator = get_retry_decorator_for_rate_limits()

        @retry_decorator
        async def _generate_summary_with_retry():
            try:
                # Configure the model
                model = genai.GenerativeModel(self.generation_model)

                # Create prompt for summarization with strong language specification
                language_instruction = ""
                if language:
                    language_names = {
                        'en': 'English',
                        'de': 'German',
                        'fr': 'French',
                        'es': 'Spanish',
                        'it': 'Italian',
                        'pt': 'Portuguese',
                        'nl': 'Dutch',
                        'ru': 'Russian',
                        'zh': 'Chinese',
                        'ja': 'Japanese',
                        'ko': 'Korean'
                    }
                    language_name = language_names.get(language, language)
                    language_instruction = f"CRITICAL: You MUST write the summary ONLY in {language_name}. Do NOT use any other language. The summary language MUST match the specified language ({language_name}). "

                # Use summarization agent - required
                summary_agent = get_agent("summarization")
                summary_agent.update_config(
                    agent_config={
                        "max_summary_length": max_length,
                        "target_language": language
                    }
                )

                result = await summary_agent.execute(
                    text,
                    target_length=max_length,
                    language=language
                )
                summary_text = result.summary

                # Record success for circuit breaker
                self.circuit_breaker.record_success()

                # Create and return summary result
                return SummaryResult(
                    original_text=text,
                    summary=summary_text,
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
                    raise ExternalServiceError(f"Gemini API error: {error_str}", "gemini")

        return await _generate_summary_with_retry()
