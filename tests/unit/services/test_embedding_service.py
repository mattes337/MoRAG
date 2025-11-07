"""Unit tests for embedding service."""

import asyncio
from typing import List

import pytest
from morag_core.exceptions import (
    ExternalServiceError,
    QuotaExceededError,
    RateLimitError,
    TimeoutError,
)
from morag_core.models.embedding import BatchEmbeddingResult, EmbeddingResult


class MockEmbeddingService:
    """Mock embedding service for testing."""

    def __init__(self, api_key: str = "test-key"):
        self.api_key = api_key
        self._circuit_breaker_failures = 0
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_open = False
        self._health_status = "healthy"

    async def generate_embedding(
        self, text: str, task_type: str = "retrieval_document"
    ) -> EmbeddingResult:
        """Generate embedding for a single text."""
        if self._circuit_breaker_open:
            raise ExternalServiceError("Circuit breaker is open")

        if not text.strip():
            raise ValueError("Text cannot be empty")

        # Simulate different behaviors for testing
        if "error" in text.lower():
            raise ExternalServiceError("Simulated API error")

        if "rate_limit" in text.lower():
            raise RateLimitError("Rate limit exceeded")

        if "timeout" in text.lower():
            raise TimeoutError("Request timed out")

        if "quota" in text.lower():
            raise QuotaExceededError("Quota exceeded")

        # Generate deterministic mock embedding based on text
        embedding = [hash(text + str(i)) % 1000 / 1000.0 for i in range(384)]

        return EmbeddingResult(
            embedding=embedding,
            token_count=len(text.split()),
            model="mock-embedding-model",
        )

    async def generate_embeddings(
        self,
        texts: List[str],
        task_type: str = "retrieval_document",
        batch_size: int = 50,
    ) -> BatchEmbeddingResult:
        """Generate embeddings for multiple texts."""
        if not texts:
            return BatchEmbeddingResult(embeddings=[], total_tokens=0)

        # Process in batches
        all_embeddings = []
        total_tokens = 0

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = []

            for text in batch:
                try:
                    result = await self.generate_embedding(text, task_type)
                    batch_embeddings.append(result.embedding)
                    total_tokens += result.token_count
                except Exception as e:
                    # In real implementation, might handle errors per-text
                    raise e

            all_embeddings.extend(batch_embeddings)

            # Simulate some delay between batches
            if len(texts) > batch_size:
                await asyncio.sleep(0.01)

        return BatchEmbeddingResult(
            embeddings=all_embeddings,
            total_tokens=total_tokens,
            batch_count=len(range(0, len(texts), batch_size)),
        )

    async def health_check(self) -> dict:
        """Check service health."""
        return {
            "status": self._health_status,
            "embedding_model": "mock-embedding-model",
            "embedding_dimension": 384,
            "circuit_breaker_open": self._circuit_breaker_open,
            "failures": self._circuit_breaker_failures,
        }

    def _increment_failure(self):
        """Simulate circuit breaker failure tracking."""
        self._circuit_breaker_failures += 1
        if self._circuit_breaker_failures >= self._circuit_breaker_threshold:
            self._circuit_breaker_open = True

    def _reset_circuit_breaker(self):
        """Reset circuit breaker for testing."""
        self._circuit_breaker_failures = 0
        self._circuit_breaker_open = False


class TestEmbeddingService:
    """Test embedding service functionality."""

    @pytest.fixture
    def embedding_service(self):
        """Create embedding service for testing."""
        return MockEmbeddingService(api_key="test-key")

    @pytest.fixture
    def sample_texts(self):
        """Sample texts for testing."""
        return [
            "This is a sample text for embedding.",
            "Another piece of text to embed.",
            "Machine learning is fascinating.",
            "Natural language processing enables AI.",
        ]

    async def test_single_embedding_success(self, embedding_service):
        """Test successful single embedding generation."""
        text = "This is a test text"

        result = await embedding_service.generate_embedding(text)

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 384
        assert all(isinstance(val, float) for val in result.embedding)
        assert result.token_count == len(text.split())
        assert result.model == "mock-embedding-model"

    async def test_single_embedding_empty_text(self, embedding_service):
        """Test embedding generation with empty text."""
        with pytest.raises(ValueError, match="Text cannot be empty"):
            await embedding_service.generate_embedding("")

        with pytest.raises(ValueError, match="Text cannot be empty"):
            await embedding_service.generate_embedding("   ")

    async def test_single_embedding_error_handling(self, embedding_service):
        """Test error handling in single embedding generation."""
        # Test API error
        with pytest.raises(ExternalServiceError, match="Simulated API error"):
            await embedding_service.generate_embedding("This will cause an error")

        # Test rate limit error
        with pytest.raises(RateLimitError, match="Rate limit exceeded"):
            await embedding_service.generate_embedding("This will cause rate_limit")

        # Test timeout error
        with pytest.raises(TimeoutError, match="Request timed out"):
            await embedding_service.generate_embedding("This will cause timeout")

        # Test quota error
        with pytest.raises(QuotaExceededError, match="Quota exceeded"):
            await embedding_service.generate_embedding("This will cause quota")

    async def test_batch_embeddings_success(self, embedding_service, sample_texts):
        """Test successful batch embedding generation."""
        result = await embedding_service.generate_embeddings(sample_texts)

        assert isinstance(result, BatchEmbeddingResult)
        assert len(result.embeddings) == len(sample_texts)
        assert all(len(embedding) == 384 for embedding in result.embeddings)
        assert result.total_tokens > 0
        assert result.batch_count == 1  # Single batch for small list

    async def test_batch_embeddings_empty_list(self, embedding_service):
        """Test batch embedding with empty list."""
        result = await embedding_service.generate_embeddings([])

        assert isinstance(result, BatchEmbeddingResult)
        assert result.embeddings == []
        assert result.total_tokens == 0
        assert result.batch_count == 0

    async def test_batch_embeddings_with_batching(self, embedding_service):
        """Test batch embedding with custom batch size."""
        texts = [f"Text {i}" for i in range(10)]

        result = await embedding_service.generate_embeddings(texts, batch_size=3)

        assert len(result.embeddings) == 10
        assert result.batch_count == 4  # ceil(10/3) = 4 batches
        assert result.total_tokens == 20  # 10 texts * 2 tokens each

    async def test_batch_embeddings_error_propagation(self, embedding_service):
        """Test error propagation in batch processing."""
        texts = ["Good text", "This will cause an error", "Another good text"]

        with pytest.raises(ExternalServiceError, match="Simulated API error"):
            await embedding_service.generate_embeddings(texts)

    async def test_health_check_healthy(self, embedding_service):
        """Test health check when service is healthy."""
        health = await embedding_service.health_check()

        assert health["status"] == "healthy"
        assert health["embedding_model"] == "mock-embedding-model"
        assert health["embedding_dimension"] == 384
        assert health["circuit_breaker_open"] is False
        assert health["failures"] == 0

    async def test_circuit_breaker_behavior(self, embedding_service):
        """Test circuit breaker behavior."""
        # Simulate multiple failures
        for i in range(5):
            embedding_service._increment_failure()

        # Circuit breaker should be open now
        health = await embedding_service.health_check()
        assert health["circuit_breaker_open"] is True

        # Requests should fail with circuit breaker error
        with pytest.raises(ExternalServiceError, match="Circuit breaker is open"):
            await embedding_service.generate_embedding("test")

        # Reset circuit breaker
        embedding_service._reset_circuit_breaker()

        # Should work again
        result = await embedding_service.generate_embedding("test")
        assert isinstance(result, EmbeddingResult)

    @pytest.mark.parametrize(
        "task_type",
        ["retrieval_document", "retrieval_query", "classification", "clustering"],
    )
    async def test_different_task_types(self, embedding_service, task_type):
        """Test embedding generation with different task types."""
        text = "Test text for different task types"

        result = await embedding_service.generate_embedding(text, task_type=task_type)

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 384
        assert result.model == "mock-embedding-model"

    async def test_embedding_deterministic(self, embedding_service):
        """Test that embeddings are deterministic for same input."""
        text = "This should produce consistent embeddings"

        result1 = await embedding_service.generate_embedding(text)
        result2 = await embedding_service.generate_embedding(text)

        assert result1.embedding == result2.embedding
        assert result1.token_count == result2.token_count

    async def test_embedding_different_for_different_texts(self, embedding_service):
        """Test that different texts produce different embeddings."""
        text1 = "This is the first text"
        text2 = "This is the second text"

        result1 = await embedding_service.generate_embedding(text1)
        result2 = await embedding_service.generate_embedding(text2)

        assert result1.embedding != result2.embedding

    async def test_concurrent_embedding_requests(self, embedding_service):
        """Test handling concurrent embedding requests."""
        texts = [f"Concurrent text {i}" for i in range(5)]

        # Create concurrent tasks
        tasks = [embedding_service.generate_embedding(text) for text in texts]

        results = await asyncio.gather(*tasks)

        assert len(results) == 5
        assert all(isinstance(result, EmbeddingResult) for result in results)
        assert all(len(result.embedding) == 384 for result in results)

    async def test_large_text_handling(self, embedding_service):
        """Test handling of large text inputs."""
        # Create a large text (simulate long document)
        large_text = "Large text. " * 1000

        result = await embedding_service.generate_embedding(large_text)

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 384
        assert result.token_count == 2000  # 1000 * 2 words

    async def test_special_characters_handling(self, embedding_service):
        """Test handling of special characters and unicode."""
        special_texts = [
            "Text with émojis 🚀 and ünïcödé",
            "Text with symbols: @#$%^&*()",
            "Mixed languages: English и русский",
            "Numbers and symbols: 123 + 456 = 579",
        ]

        for text in special_texts:
            result = await embedding_service.generate_embedding(text)
            assert isinstance(result, EmbeddingResult)
            assert len(result.embedding) == 384

    async def test_batch_size_validation(self, embedding_service, sample_texts):
        """Test batch size parameter validation."""
        # Test with batch size of 1
        result = await embedding_service.generate_embeddings(sample_texts, batch_size=1)
        assert len(result.embeddings) == len(sample_texts)
        assert result.batch_count == len(sample_texts)

        # Test with large batch size
        result = await embedding_service.generate_embeddings(
            sample_texts, batch_size=100
        )
        assert len(result.embeddings) == len(sample_texts)
        assert result.batch_count == 1

    async def test_performance_monitoring(self, embedding_service, sample_texts):
        """Test that embeddings can be generated within reasonable time."""
        import time

        start_time = time.time()
        result = await embedding_service.generate_embeddings(sample_texts)
        end_time = time.time()

        duration = end_time - start_time

        # Should complete quickly for mock service
        assert duration < 1.0
        assert len(result.embeddings) == len(sample_texts)

        # Test throughput calculation
        throughput = len(sample_texts) / duration
        assert throughput > 10  # Should process at least 10 texts per second
