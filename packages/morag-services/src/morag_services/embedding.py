"""Embedding services for MoRAG using various AI providers."""

import google.generativeai as genai
from google.generativeai import types
from typing import List, Dict, Any, Optional
import structlog
import asyncio
import time
from dataclasses import dataclass

from morag_core.interfaces.embedding import BaseEmbeddingService
from morag_core.exceptions import ExternalServiceError, RateLimitError

logger = structlog.get_logger(__name__)


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embedding: List[float]
    token_count: int
    model: str


@dataclass
class SummaryResult:
    """Result of text summarization."""
    summary: str
    token_count: int
    model: str


class GeminiEmbeddingService(BaseEmbeddingService):
    """Gemini-based embedding service."""

    def __init__(
        self,
        api_key: str,
        embedding_model: str = "text-embedding-004",
        generation_model: str = None
    ):
        """Initialize Gemini embedding service.

        Args:
            api_key: Gemini API key
            embedding_model: Model for embeddings
            generation_model: Model for text generation (if None, uses environment variable)
        """
        # Create a basic config for the parent class
        from morag_core.interfaces.embedding import EmbeddingConfig, EmbeddingProvider
        # Get batch size from environment or use default
        import os
        batch_size = int(os.getenv('MORAG_EMBEDDING_BATCH_SIZE', '50'))

        # Use environment variable for generation model if not provided
        if generation_model is None:
            generation_model = os.getenv('MORAG_GEMINI_MODEL', 'gemini-2.0-flash')

        config = EmbeddingConfig(
            provider=EmbeddingProvider.GEMINI,
            model_name=embedding_model,
            api_key=api_key,
            max_tokens=8192,
            batch_size=batch_size
        )
        super().__init__(config)

        self.api_key = api_key
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        self.max_retries = 3
        self.base_delay = 1.0

        # Configure Gemini API
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.client = True  # Use boolean to indicate client is configured
            logger.info("Gemini service initialized",
                       embedding_model=self.embedding_model,
                       generation_model=self.generation_model)
        else:
            self.client = None
            logger.warning("Gemini API key not found - service will not work")

    async def initialize(self) -> bool:
        """Initialize the embedding service.

        Returns:
            True if initialization was successful
        """
        return self.client is not None

    async def shutdown(self) -> None:
        """Shutdown the embedding service and release resources."""
        pass

    async def health_check(self) -> Dict[str, Any]:
        """Check service health.

        Returns:
            Dictionary with health status information
        """
        try:
            if not self.client:
                return {
                    "status": "unhealthy",
                    "error": "Client not initialized",
                    "embedding_model": self.embedding_model,
                    "generation_model": self.generation_model
                }

            # Test with a simple embedding request
            test_result = await self.generate_embedding("test", "retrieval_document")

            return {
                "status": "healthy",
                "embedding_model": self.embedding_model,
                "generation_model": self.generation_model,
                "embedding_dimension": len(test_result.embedding),
                "test_successful": True
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "embedding_model": self.embedding_model,
                "generation_model": self.generation_model
            }
    
    async def generate_embedding(
        self,
        text: str,
        model: Optional[str] = None,
        task_type: str = "retrieval_document",
        **kwargs
    ) -> List[float]:
        """Generate embedding for a single text."""
        if not self.client:
            raise ExternalServiceError("Gemini client not initialized", "gemini")

        try:
            # Use asyncio.to_thread for CPU-bound operations
            result = await asyncio.to_thread(
                self._generate_embedding_sync,
                text,
                task_type
            )

            logger.debug("Generated embedding", text_length=len(text), model=model or self.embedding_model)
            return result.embedding  # Return just the embedding vector

        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            raise ExternalServiceError(f"Embedding generation failed: {str(e)}", "gemini")

    async def generate_embedding_with_result(
        self,
        text: str,
        task_type: str = "retrieval_document"
    ) -> EmbeddingResult:
        """Generate embedding for a single text and return full result (backward compatibility)."""
        if not self.client:
            raise ExternalServiceError("Gemini client not initialized", "gemini")

        try:
            # Use asyncio.to_thread for CPU-bound operations
            result = await asyncio.to_thread(
                self._generate_embedding_sync,
                text,
                task_type
            )

            logger.debug("Generated embedding", text_length=len(text), model=self.embedding_model)
            return result

        except Exception as e:
            logger.error("Failed to generate embedding", error=str(e))
            raise ExternalServiceError(f"Embedding generation failed: {str(e)}", "gemini")

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs
    ) -> "EmbeddingResult":
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: Optional model override
            **kwargs: Additional options

        Returns:
            Embedding result with multiple embeddings
        """
        from morag_core.interfaces.embedding import EmbeddingResult as CoreEmbeddingResult

        # Use batch processing
        results = await self.generate_embeddings_batch(texts, **kwargs)

        # Convert to core format
        embeddings = [result.embedding for result in results]

        return CoreEmbeddingResult(
            embeddings=embeddings,
            model=model or self.embedding_model,
            usage={"token_count": sum(result.token_count for result in results)},
            metadata={"batch_size": len(texts)}
        )

    def get_embedding_dimension(self, model: Optional[str] = None) -> int:
        """Get embedding dimension for model.

        Args:
            model: Optional model override

        Returns:
            Embedding dimension
        """
        # text-embedding-004 has 768 dimensions
        if (model or self.embedding_model) == "text-embedding-004":
            return 768
        else:
            # Default for most Gemini embedding models
            return 768

    def get_supported_models(self) -> List[str]:
        """Get list of supported models.

        Returns:
            List of model names
        """
        return [
            "text-embedding-004",
            "text-embedding-preview-0409",
            "embedding-001"
        ]

    def get_max_tokens(self, model: Optional[str] = None) -> int:
        """Get maximum tokens for model.

        Args:
            model: Optional model override

        Returns:
            Maximum tokens
        """
        # Most Gemini embedding models support up to 2048 tokens
        return 2048

    def _generate_embedding_sync(self, text: str, task_type: str) -> EmbeddingResult:
        """Synchronous embedding generation with indefinite retry logic for rate limits."""
        from morag_core.config import settings

        # Get retry configuration
        retry_indefinitely = settings.retry_indefinitely
        base_delay = settings.retry_base_delay
        max_delay = settings.retry_max_delay
        exponential_base = settings.retry_exponential_base
        use_jitter = settings.retry_jitter

        # For non-rate-limit errors, still use limited retries
        max_retries_non_rate_limit = 3
        attempt = 0

        while True:
            try:
                # Ensure model name has proper prefix for new SDK
                model_name = self.embedding_model
                if not model_name.startswith(('models/', 'tunedModels/')):
                    model_name = f"models/{model_name}"
                response = genai.embed_content(
                    model=model_name,
                    content=text
                )

                return EmbeddingResult(
                    embedding=response['embedding'],
                    token_count=len(text.split()),  # Approximate token count
                    model=self.embedding_model
                )

            except Exception as e:
                error_str = str(e)
                attempt += 1

                # Check for rate limiting errors
                if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str or
                    "quota exceeded" in error_str.lower() or "rate limit" in error_str.lower()):

                    # For rate limit errors, retry indefinitely if configured
                    if retry_indefinitely:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)

                        # Add jitter if enabled
                        if use_jitter:
                            jitter = (time.time() % 1) * 0.1 * delay  # 10% jitter
                            delay += jitter

                        logger.warning(
                            "Rate limit hit, retrying indefinitely with exponential backoff",
                            attempt=attempt,
                            delay=delay,
                            max_delay=max_delay,
                            error=error_str
                        )
                        time.sleep(delay)
                        continue
                    else:
                        # Legacy behavior: limited retries
                        if attempt <= max_retries_non_rate_limit:
                            delay = base_delay * (exponential_base ** (attempt - 1))
                            if use_jitter:
                                delay += (time.time() % 1)
                            logger.warning(
                                "Rate limit hit, retrying with limited attempts",
                                attempt=attempt,
                                max_retries=max_retries_non_rate_limit,
                                delay=delay,
                                error=error_str
                            )
                            time.sleep(delay)
                            continue
                        else:
                            logger.error("Rate limit exceeded after all retries", error=error_str)
                            raise RateLimitError(f"Rate limit exceeded after {max_retries_non_rate_limit} retries: {error_str}")
                else:
                    # Non-rate-limit error, use limited retries
                    if attempt <= max_retries_non_rate_limit:
                        delay = base_delay * (exponential_base ** (attempt - 1))
                        if use_jitter:
                            delay += (time.time() % 1)
                        logger.warning(
                            "Non-rate-limit error, retrying",
                            attempt=attempt,
                            max_retries=max_retries_non_rate_limit,
                            delay=delay,
                            error=error_str
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error("Failed to generate embedding after retries", error=error_str)
                        raise ExternalServiceError(f"Embedding generation failed: {error_str}", "gemini")
    
    async def generate_embeddings_batch_native(
        self,
        texts: List[str],
        task_type: str = "retrieval_document"
    ) -> List[EmbeddingResult]:
        """Generate embeddings using Gemini's native batch API.

        Args:
            texts: List of texts to embed (should be <= batch_size)
            task_type: Type of embedding task

        Returns:
            List of embedding results
        """
        if not self.client:
            raise ExternalServiceError("Gemini client not initialized", "gemini")

        if not texts:
            return []

        try:
            logger.debug("Processing native batch embedding",
                        batch_size=len(texts),
                        task_type=task_type)

            # Use the new Google GenAI SDK batch embedding
            result = await asyncio.to_thread(
                self._generate_batch_embeddings_sync,
                texts,
                task_type
            )

            # Convert to EmbeddingResult objects
            results = []
            for i, embedding in enumerate(result):
                results.append(EmbeddingResult(
                    embedding=embedding,
                    token_count=len(texts[i].split()) if i < len(texts) else 0,
                    model=self.embedding_model
                ))

            logger.debug("Native batch embedding completed",
                        input_count=len(texts),
                        output_count=len(results))

            return results

        except Exception as e:
            logger.error("Native batch embedding failed", error=str(e))
            raise ExternalServiceError(f"Batch embedding failed: {str(e)}", "gemini")

    def _generate_batch_embeddings_sync(self, texts: List[str], task_type: str) -> List[List[float]]:
        """Synchronous batch embedding generation."""
        try:
            # Use the Google GenAI SDK for batch embedding
            # Ensure model name has proper prefix for new SDK
            model_name = self.embedding_model
            if not model_name.startswith(('models/', 'tunedModels/')):
                model_name = f"models/{model_name}"

            # Process texts individually since batch API might not be available
            embeddings = []
            for text in texts:
                response = genai.embed_content(
                    model=model_name,
                    content=text
                )
                embeddings.append(response['embedding'])
            return embeddings

        except Exception as e:
            logger.error("Synchronous batch embedding failed", error=str(e))
            raise

    async def generate_embeddings_batch(
        self,
        texts: List[str],
        task_type: str = "retrieval_document",
        batch_size: Optional[int] = None,
        delay_between_batches: float = 0.1,
        use_native_batch: bool = True
    ) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts with optimized batch processing."""
        if not texts:
            return []

        # Use config batch size if none provided
        if batch_size is None:
            batch_size = self.config.batch_size

        results = []

        # Use native batch API if enabled and available
        if use_native_batch:
            try:
                return await self._generate_embeddings_batch_optimized(texts, task_type, batch_size, delay_between_batches)
            except Exception as e:
                logger.warning("Native batch embedding failed, falling back to sequential", error=str(e))
                # Fall back to sequential processing
                use_native_batch = False

        # Sequential processing (fallback)
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]

            logger.debug("Processing embedding batch",
                        batch_num=i//batch_size + 1,
                        batch_size=len(batch),
                        total_texts=len(texts))

            # Process batch with small delays between requests
            batch_results = []
            for j, text in enumerate(batch):
                try:
                    result = await self.generate_embedding(text, task_type)
                    batch_results.append(result)

                    # Small delay between individual requests to avoid hitting rate limits
                    if j < len(batch) - 1:  # Don't delay after the last item
                        await asyncio.sleep(0.1)  # 100ms delay between requests

                except Exception as e:
                    logger.error("Failed to generate embedding in batch", error=str(e))
                    # Create a dummy result to maintain order
                    batch_results.append(EmbeddingResult(
                        embedding=[0.0] * 768,  # Default dimension for text-embedding-004
                        token_count=0,
                        model=self.embedding_model
                    ))

            results.extend(batch_results)

            # Delay between batches to respect rate limits
            if i + batch_size < len(texts):
                await asyncio.sleep(delay_between_batches)

        logger.info("Completed batch embedding generation",
                   total_texts=len(texts),
                   successful_embeddings=len([r for r in results if r.token_count > 0]))

        return results

    async def _generate_embeddings_batch_optimized(
        self,
        texts: List[str],
        task_type: str,
        batch_size: int,
        delay_between_batches: float
    ) -> List[EmbeddingResult]:
        """Generate embeddings using optimized batch processing."""
        all_results = []

        # Process texts in batches using native batch API
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            try:
                # Use native batch API for this chunk
                batch_results = await self.generate_embeddings_batch_native(batch_texts, task_type)
                all_results.extend(batch_results)

                logger.debug("Processed batch successfully",
                           batch_size=len(batch_texts),
                           total_processed=len(all_results),
                           total_texts=len(texts))

                # Small delay between batches
                if i + batch_size < len(texts):
                    await asyncio.sleep(delay_between_batches)

            except Exception as e:
                logger.error("Batch processing failed for chunk",
                           batch_start=i,
                           batch_size=len(batch_texts),
                           error=str(e))
                # Add error results for each text in the failed batch
                for text in batch_texts:
                    all_results.append(EmbeddingResult(
                        embedding=[0.0] * 768,
                        token_count=0,
                        model=self.embedding_model
                    ))

        return all_results
    
    async def generate_summary(
        self,
        text: str,
        max_length: int = 150,
        style: str = "concise"
    ) -> SummaryResult:
        """Generate a summary of the given text."""
        if not self.client:
            raise ExternalServiceError("Gemini client not initialized", "gemini")

        try:
            prompt = self._build_summary_prompt(text, max_length, style)

            logger.info("Generating summary with Gemini API",
                       text_length=len(text),
                       text_preview=text[:100] + "..." if len(text) > 100 else text,
                       max_length=max_length,
                       style=style,
                       model=self.generation_model)

            result = await asyncio.to_thread(
                self._generate_text_sync,
                prompt
            )

            logger.info("Gemini API response received",
                       original_length=len(text),
                       summary_length=len(result),
                       summary_preview=result[:100] + "..." if len(result) > 100 else result,
                       model=self.generation_model)

            return SummaryResult(
                summary=result,
                token_count=len(result.split()),
                model=self.generation_model
            )
            
        except Exception as e:
            logger.error("Failed to generate summary", error=str(e))
            raise ExternalServiceError(f"Summary generation failed: {str(e)}", "gemini")
    
    def _generate_text_sync(self, prompt: str) -> str:
        """Synchronous text generation with indefinite retry logic for rate limits."""
        from morag_core.config import settings

        # Get retry configuration
        retry_indefinitely = settings.retry_indefinitely
        base_delay = settings.retry_base_delay
        max_delay = settings.retry_max_delay
        exponential_base = settings.retry_exponential_base
        use_jitter = settings.retry_jitter

        # For non-rate-limit errors, still use limited retries
        max_retries_non_rate_limit = 3
        attempt = 0

        while True:
            try:
                model = genai.GenerativeModel(self.generation_model)
                response = model.generate_content(prompt)
                return response.text

            except Exception as e:
                error_str = str(e)
                attempt += 1

                # Check for rate limiting errors
                if ("429" in error_str or "RESOURCE_EXHAUSTED" in error_str or
                    "quota exceeded" in error_str.lower() or "rate limit" in error_str.lower()):

                    # For rate limit errors, retry indefinitely if configured
                    if retry_indefinitely:
                        # Calculate delay with exponential backoff
                        delay = min(base_delay * (exponential_base ** (attempt - 1)), max_delay)

                        # Add jitter if enabled
                        if use_jitter:
                            jitter = (time.time() % 1) * 0.1 * delay  # 10% jitter
                            delay += jitter

                        logger.warning(
                            "Rate limit hit in text generation, retrying indefinitely with exponential backoff",
                            attempt=attempt,
                            delay=delay,
                            max_delay=max_delay,
                            error=error_str
                        )
                        time.sleep(delay)
                        continue
                    else:
                        # Legacy behavior: limited retries
                        if attempt <= max_retries_non_rate_limit:
                            delay = base_delay * (exponential_base ** (attempt - 1))
                            if use_jitter:
                                delay += (time.time() % 1)
                            logger.warning(
                                "Rate limit hit in text generation, retrying with limited attempts",
                                attempt=attempt,
                                max_retries=max_retries_non_rate_limit,
                                delay=delay,
                                error=error_str
                            )
                            time.sleep(delay)
                            continue
                        else:
                            logger.error("Rate limit exceeded after all retries in text generation", error=error_str)
                            raise RateLimitError(f"Rate limit exceeded after {max_retries_non_rate_limit} retries: {error_str}")
                else:
                    # Non-rate-limit error, use limited retries
                    if attempt <= max_retries_non_rate_limit:
                        delay = base_delay * (exponential_base ** (attempt - 1))
                        if use_jitter:
                            delay += (time.time() % 1)
                        logger.warning(
                            "Non-rate-limit error in text generation, retrying",
                            attempt=attempt,
                            max_retries=max_retries_non_rate_limit,
                            delay=delay,
                            error=error_str
                        )
                        time.sleep(delay)
                        continue
                    else:
                        logger.error("Failed to generate text after retries", error=error_str)
                        raise ExternalServiceError(f"Text generation failed: {error_str}", "gemini")

    async def generate_text_from_prompt(self, prompt: str) -> str:
        """Generate text directly from a prompt."""
        if not self.client:
            raise ExternalServiceError("Gemini client not initialized", "gemini")

        try:
            logger.info("Generating text from direct prompt",
                       prompt_length=len(prompt),
                       prompt_preview=prompt[:200] + "..." if len(prompt) > 200 else prompt,
                       model=self.generation_model)

            result = await asyncio.to_thread(
                self._generate_text_sync,
                prompt
            )

            logger.info("Direct prompt response received",
                       response_length=len(result),
                       response_preview=result[:100] + "..." if len(result) > 100 else result,
                       model=self.generation_model)

            return result
            
        except Exception as e:
            logger.error("Failed to generate text from prompt", error=str(e))
            raise ExternalServiceError(f"Text generation failed: {str(e)}", "gemini")
    
    def _build_summary_prompt(self, text: str, max_length: int, style: str) -> str:
        """Build prompt for summary generation."""
        style_instructions = {
            "concise": "Create a concise, factual summary",
            "detailed": "Create a detailed summary that captures key points",
            "bullet": "Create a bullet-point summary of key information",
            "abstract": "Create an abstract-style summary suitable for academic content"
        }

        instruction = style_instructions.get(style, style_instructions["concise"])

        return f"""
{instruction} of the following text in approximately {max_length} words or less.
Focus on the main ideas, key facts, and important details.

Text to summarize:
{text}

Summary:
"""

    async def health_check(self) -> Dict[str, Any]:
        """Check if Gemini API is accessible."""
        try:
            if not self.client:
                return {
                    "status": "unhealthy",
                    "error": "Gemini client not initialized",
                    "embedding_model": self.embedding_model,
                    "generation_model": self.generation_model
                }

            # Test with a simple embedding request
            test_result = await self.generate_embedding("Health check test")

            return {
                "status": "healthy",
                "embedding_model": self.embedding_model,
                "generation_model": self.generation_model,
                "embedding_dimension": len(test_result.embedding)
            }

        except Exception as e:
            logger.error("Gemini health check failed", error=str(e))
            return {
                "status": "unhealthy",
                "error": str(e),
                "embedding_model": self.embedding_model,
                "generation_model": self.generation_model
            }


class EmbeddingServiceFactory:
    """Factory for creating embedding services."""
    
    @staticmethod
    def create_gemini_service(
        api_key: str,
        embedding_model: str = "text-embedding-004",
        generation_model: str = "gemini-2.0-flash-001"
    ) -> GeminiEmbeddingService:
        """Create a Gemini embedding service."""
        return GeminiEmbeddingService(api_key, embedding_model, generation_model)
    
    @staticmethod
    def create_service(
        provider: str,
        **kwargs
    ) -> BaseEmbeddingService:
        """Create an embedding service based on provider."""
        if provider.lower() == "gemini":
            return EmbeddingServiceFactory.create_gemini_service(**kwargs)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")


# Legacy compatibility
class GeminiService(GeminiEmbeddingService):
    """Legacy GeminiService for backward compatibility."""

    def __init__(self):
        # Initialize with default settings - these would come from config
        import os
        super().__init__(
            api_key="",  # Should be set from config
            embedding_model="text-embedding-004",
            generation_model=None  # Will use environment variable
        )
