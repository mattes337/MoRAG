"""Integration tests for Gemini API service."""

import pytest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from morag.services.embedding import gemini_service, EmbeddingResult, SummaryResult
from morag.core.exceptions import ExternalServiceError, RateLimitError


class TestGeminiIntegration:
    """Test Gemini service integration."""

    @pytest.mark.asyncio
    async def test_embedding_generation(self):
        """Test single embedding generation."""
        with patch.object(gemini_service, 'client') as mock_client:
            # Mock the response
            mock_response = MagicMock()
            mock_response.embeddings = [MagicMock()]
            mock_response.embeddings[0].values = [0.1] * 768
            mock_client.models.embed_content.return_value = mock_response
            
            text = "This is a test document about machine learning and artificial intelligence."
            
            result = await gemini_service.generate_embedding(text)
            
            assert isinstance(result, EmbeddingResult)
            assert isinstance(result.embedding, list)
            assert len(result.embedding) == 768  # text-embedding-004 dimension
            assert result.model == "text-embedding-004"
            assert result.token_count > 0

    @pytest.mark.asyncio
    async def test_batch_embedding_generation(self):
        """Test batch embedding generation."""
        with patch.object(gemini_service, 'client') as mock_client:
            # Mock the response
            mock_response = MagicMock()
            mock_response.embeddings = [MagicMock()]
            mock_response.embeddings[0].values = [0.1] * 768
            mock_client.models.embed_content.return_value = mock_response
            
            texts = [
                "First document about AI",
                "Second document about machine learning",
                "Third document about data science"
            ]
            
            results = await gemini_service.generate_embeddings_batch(texts, batch_size=2)
            
            assert len(results) == len(texts)
            for result in results:
                assert len(result.embedding) == 768

    @pytest.mark.asyncio
    async def test_summary_generation(self):
        """Test text summarization."""
        with patch.object(gemini_service, 'client') as mock_client:
            # Mock the response
            mock_response = MagicMock()
            mock_response.text = "Machine learning enables systems to learn from data automatically."
            mock_client.models.generate_content.return_value = mock_response
            
            text = """
            Machine learning is a subset of artificial intelligence that focuses on algorithms
            that can learn from and make predictions or decisions based on data. It involves
            training models on large datasets to identify patterns and relationships.
            The goal is to create systems that can automatically improve their performance
            on a specific task through experience.
            """
            
            result = await gemini_service.generate_summary(text, max_length=50)
            
            assert isinstance(result, SummaryResult)
            assert isinstance(result.summary, str)
            assert len(result.summary) > 0
            assert result.model == "gemini-2.0-flash-001"

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test Gemini service health check when healthy."""
        with patch.object(gemini_service, 'client') as mock_client:
            # Mock the response
            mock_response = MagicMock()
            mock_response.embeddings = [MagicMock()]
            mock_response.embeddings[0].values = [0.1] * 768
            mock_client.models.embed_content.return_value = mock_response
            
            health = await gemini_service.health_check()
            
            assert health["status"] == "healthy"
            assert health["embedding_dimension"] == 768

    @pytest.mark.asyncio
    async def test_health_check_no_client(self):
        """Test health check when client is not initialized."""
        with patch.object(gemini_service, 'client', None):
            health = await gemini_service.health_check()
            
            assert health["status"] == "unhealthy"
            assert "not initialized" in health["error"]

    @pytest.mark.asyncio
    async def test_embedding_rate_limit_error(self):
        """Test rate limit handling in embedding generation."""
        with patch.object(gemini_service, 'client') as mock_client:
            mock_client.models.embed_content.side_effect = Exception("quota exceeded")
            
            text = "Test text"
            
            with pytest.raises(RateLimitError):
                await gemini_service.generate_embedding(text)

    @pytest.mark.asyncio
    async def test_embedding_external_service_error(self):
        """Test external service error handling."""
        with patch.object(gemini_service, 'client') as mock_client:
            mock_client.models.embed_content.side_effect = Exception("API error")
            
            text = "Test text"
            
            with pytest.raises(ExternalServiceError):
                await gemini_service.generate_embedding(text)

    @pytest.mark.asyncio
    async def test_summary_rate_limit_error(self):
        """Test rate limit handling in summary generation."""
        with patch.object(gemini_service, 'client') as mock_client:
            mock_client.models.generate_content.side_effect = Exception("rate limit exceeded")
            
            text = "Test text for summarization"
            
            with pytest.raises(RateLimitError):
                await gemini_service.generate_summary(text)

    @pytest.mark.asyncio
    async def test_batch_embedding_with_failures(self):
        """Test batch embedding with some failures."""
        with patch.object(gemini_service, 'generate_embedding') as mock_generate:
            # First call succeeds, second fails, third succeeds
            mock_generate.side_effect = [
                EmbeddingResult(embedding=[0.1] * 768, token_count=10, model="text-embedding-004"),
                Exception("API error"),
                EmbeddingResult(embedding=[0.2] * 768, token_count=15, model="text-embedding-004")
            ]
            
            texts = ["Text 1", "Text 2", "Text 3"]
            
            results = await gemini_service.generate_embeddings_batch(texts)
            
            assert len(results) == 3
            # First result should be successful
            assert results[0].token_count == 10
            # Second result should be fallback (zero embedding)
            assert results[1].token_count == 0
            assert all(x == 0.0 for x in results[1].embedding)
            # Third result should be successful
            assert results[2].token_count == 15

    @pytest.mark.asyncio
    async def test_no_client_initialization(self):
        """Test behavior when client is not initialized."""
        with patch.object(gemini_service, 'client', None):
            text = "Test text"
            
            with pytest.raises(ExternalServiceError, match="not initialized"):
                await gemini_service.generate_embedding(text)

    def test_summary_prompt_building(self):
        """Test summary prompt building with different styles."""
        text = "Sample text for testing"
        
        # Test concise style
        prompt = gemini_service._build_summary_prompt(text, 100, "concise")
        assert "concise" in prompt.lower()
        assert "100 words" in prompt
        assert text in prompt
        
        # Test detailed style
        prompt = gemini_service._build_summary_prompt(text, 200, "detailed")
        assert "detailed" in prompt.lower()
        assert "200 words" in prompt
        
        # Test bullet style
        prompt = gemini_service._build_summary_prompt(text, 150, "bullet")
        assert "bullet" in prompt.lower()
        
        # Test unknown style (should default to concise)
        prompt = gemini_service._build_summary_prompt(text, 100, "unknown")
        assert "concise" in prompt.lower()


class TestGeminiServiceConfiguration:
    """Test Gemini service configuration and initialization."""

    def test_service_initialization_with_api_key(self):
        """Test service initialization with API key."""
        with patch('morag.services.embedding.settings') as mock_settings:
            mock_settings.gemini_api_key = "AIzaSyTest123"

            with patch('google.genai.Client') as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client

                from morag.services.embedding import GeminiService
                service = GeminiService()

                assert service.client is not None
                assert service.embedding_model == "text-embedding-004"
                assert service.generation_model == "gemini-2.0-flash-001"

    def test_service_initialization_without_api_key(self):
        """Test service initialization without API key."""
        with patch('morag.services.embedding.settings') as mock_settings:
            mock_settings.gemini_api_key = None

            from morag.services.embedding import GeminiService
            service = GeminiService()

            assert service.client is None


class TestTextProcessingIntegration:
    """Test text processing utilities with Gemini service."""

    @pytest.mark.asyncio
    async def test_embedding_with_text_preparation(self):
        """Test embedding generation with text preparation."""
        from morag.utils.text_processing import prepare_text_for_embedding
        
        with patch.object(gemini_service, 'client') as mock_client:
            # Mock the response
            mock_response = MagicMock()
            mock_response.embeddings = [MagicMock()]
            mock_response.embeddings[0].values = [0.1] * 768
            mock_client.models.embed_content.return_value = mock_response
            
            # Test with messy text
            messy_text = "  This   is    a\n\n\ntest   document!!!   @#$%^&*()  "
            clean_text = prepare_text_for_embedding(messy_text)
            
            result = await gemini_service.generate_embedding(clean_text)
            
            assert isinstance(result, EmbeddingResult)
            assert len(result.embedding) == 768

    @pytest.mark.asyncio
    async def test_summary_with_text_preparation(self):
        """Test summary generation with text preparation."""
        from morag.utils.text_processing import prepare_text_for_summary
        
        with patch.object(gemini_service, 'client') as mock_client:
            # Mock the response
            mock_response = MagicMock()
            mock_response.text = "Clean summary of the prepared text."
            mock_client.models.generate_content.return_value = mock_response
            
            # Test with messy text
            messy_text = "This\n\n\nis    a    test\n\n\ndocument   with   lots   of   whitespace."
            clean_text = prepare_text_for_summary(messy_text)
            
            result = await gemini_service.generate_summary(clean_text)
            
            assert isinstance(result, SummaryResult)
            assert len(result.summary) > 0
