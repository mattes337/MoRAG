"""Unit tests for Gemini service."""

import pytest
from unittest.mock import patch, MagicMock
from morag_services.embedding import GeminiService, EmbeddingResult, SummaryResult
from morag_core.exceptions import ExternalServiceError, RateLimitError


class TestGeminiService:
    """Test GeminiService functionality."""

    @pytest.fixture
    def mock_settings(self):
        """Mock settings for testing."""
        with patch('morag.services.embedding.settings') as mock:
            mock.gemini_api_key = "AIzaSyTest123"
            yield mock

    @pytest.fixture
    def gemini_service(self, mock_settings):
        """Create GeminiService instance for testing."""
        with patch('google.genai.Client') as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            service = GeminiService()
            service.client = mock_client
            return service

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, gemini_service):
        """Test successful embedding generation."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock()]
        mock_response.embeddings[0].values = [0.1, 0.2, 0.3] * 256  # 768 dimensions
        gemini_service.client.models.embed_content.return_value = mock_response

        text = "Test document for embedding"

        result = await gemini_service.generate_embedding(text)

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 768
        assert result.model == "text-embedding-004"
        assert result.token_count > 0

        # Verify the API call
        gemini_service.client.models.embed_content.assert_called_once_with(
            model="text-embedding-004",
            contents=text
        )

    @pytest.mark.asyncio
    async def test_generate_embedding_no_client(self):
        """Test embedding generation without client."""
        service = GeminiService()
        service.client = None

        with pytest.raises(ExternalServiceError, match="not initialized"):
            await service.generate_embedding("test text")

    @pytest.mark.asyncio
    async def test_generate_embedding_rate_limit(self, gemini_service):
        """Test rate limit error handling."""
        gemini_service.client.models.embed_content.side_effect = Exception("quota exceeded")

        with pytest.raises(RateLimitError):
            await gemini_service.generate_embedding("test text")

    @pytest.mark.asyncio
    async def test_generate_embedding_api_error(self, gemini_service):
        """Test API error handling."""
        gemini_service.client.models.embed_content.side_effect = Exception("API error")

        with pytest.raises(ExternalServiceError):
            await gemini_service.generate_embedding("test text")

    @pytest.mark.asyncio
    async def test_generate_summary_success(self, gemini_service):
        """Test successful summary generation."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.text = "This is a concise summary of the input text."
        gemini_service.client.models.generate_content.return_value = mock_response

        text = "Long text that needs to be summarized for testing purposes."

        result = await gemini_service.generate_summary(text, max_length=50, style="concise")

        assert isinstance(result, SummaryResult)
        assert result.summary == "This is a concise summary of the input text."
        assert result.model == "gemini-2.0-flash-001"
        assert result.token_count > 0

    @pytest.mark.asyncio
    async def test_generate_summary_rate_limit(self, gemini_service):
        """Test rate limit error in summary generation."""
        gemini_service.client.models.generate_content.side_effect = Exception("rate limit")

        with pytest.raises(RateLimitError):
            await gemini_service.generate_summary("test text")

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_success(self, gemini_service):
        """Test successful batch embedding generation."""
        # Mock the response
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock()]
        mock_response.embeddings[0].values = [0.1] * 768
        gemini_service.client.models.embed_content.return_value = mock_response

        texts = ["Text 1", "Text 2", "Text 3"]

        results = await gemini_service.generate_embeddings_batch(texts, batch_size=2)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, EmbeddingResult)
            assert len(result.embedding) == 768

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_with_errors(self, gemini_service):
        """Test batch embedding with some errors."""
        # Mock generate_embedding to fail on second call
        with patch.object(gemini_service, 'generate_embedding') as mock_generate:
            mock_generate.side_effect = [
                EmbeddingResult(embedding=[0.1] * 768, token_count=10, model="test"),
                Exception("API error"),
                EmbeddingResult(embedding=[0.2] * 768, token_count=15, model="test")
            ]

            texts = ["Text 1", "Text 2", "Text 3"]

            results = await gemini_service.generate_embeddings_batch(texts)

            assert len(results) == 3
            # First and third should be successful
            assert results[0].token_count == 10
            assert results[2].token_count == 15
            # Second should be fallback
            assert results[1].token_count == 0
            assert all(x == 0.0 for x in results[1].embedding)

    @pytest.mark.asyncio
    async def test_health_check_healthy(self, gemini_service):
        """Test health check when service is healthy."""
        # Mock successful embedding generation
        with patch.object(gemini_service, 'generate_embedding') as mock_generate:
            mock_generate.return_value = EmbeddingResult(
                embedding=[0.1] * 768,
                token_count=3,
                model="text-embedding-004"
            )

            health = await gemini_service.health_check()

            assert health["status"] == "healthy"
            assert health["embedding_dimension"] == 768
            assert health["embedding_model"] == "text-embedding-004"

    @pytest.mark.asyncio
    async def test_health_check_no_client(self):
        """Test health check when client is not initialized."""
        service = GeminiService()
        service.client = None

        health = await service.health_check()

        assert health["status"] == "unhealthy"
        assert "not initialized" in health["error"]

    @pytest.mark.asyncio
    async def test_health_check_api_error(self, gemini_service):
        """Test health check when API call fails."""
        with patch.object(gemini_service, 'generate_embedding') as mock_generate:
            mock_generate.side_effect = Exception("API error")

            health = await gemini_service.health_check()

            assert health["status"] == "unhealthy"
            assert "API error" in health["error"]

    def test_build_summary_prompt_styles(self, gemini_service):
        """Test summary prompt building with different styles."""
        text = "Sample text for testing"

        # Test all supported styles
        styles = ["concise", "detailed", "bullet", "abstract"]

        for style in styles:
            prompt = gemini_service._build_summary_prompt(text, 100, style)
            assert style in prompt.lower()
            assert "100 words" in prompt
            assert text in prompt
            assert "Summary:" in prompt

    def test_build_summary_prompt_unknown_style(self, gemini_service):
        """Test summary prompt with unknown style."""
        text = "Sample text"

        prompt = gemini_service._build_summary_prompt(text, 100, "unknown_style")
        # Should default to concise
        assert "concise" in prompt.lower()

    def test_service_initialization_with_api_key(self):
        """Test service initialization with API key."""
        with patch('morag.services.embedding.settings') as mock_settings:
            mock_settings.gemini_api_key = "AIzaSyTest123"

            with patch('google.genai.Client') as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client

                service = GeminiService()

                assert service.client is not None
                assert service.api_key == "AIzaSyTest123"
                assert service.embedding_model == "text-embedding-004"
                assert service.generation_model == "gemini-2.0-flash-001"
                mock_client_class.assert_called_once_with(api_key="AIzaSyTest123")

    def test_service_initialization_without_api_key(self):
        """Test service initialization without API key."""
        with patch('morag.services.embedding.settings') as mock_settings:
            mock_settings.gemini_api_key = None

            service = GeminiService()

            assert service.client is None
            assert service.api_key is None

    def test_embedding_result_dataclass(self):
        """Test EmbeddingResult dataclass."""
        embedding = [0.1, 0.2, 0.3]
        result = EmbeddingResult(
            embedding=embedding,
            token_count=10,
            model="test-model"
        )

        assert result.embedding == embedding
        assert result.token_count == 10
        assert result.model == "test-model"

    def test_summary_result_dataclass(self):
        """Test SummaryResult dataclass."""
        summary = "Test summary"
        result = SummaryResult(
            summary=summary,
            token_count=5,
            model="test-model"
        )

        assert result.summary == summary
        assert result.token_count == 5
        assert result.model == "test-model"


class TestGeminiServiceEdgeCases:
    """Test edge cases for Gemini service."""

    @pytest.fixture
    def gemini_service(self):
        """Create GeminiService instance for testing."""
        with patch('morag.services.embedding.settings') as mock_settings:
            mock_settings.gemini_api_key = "AIzaSyTest123"

            with patch('google.genai.Client') as mock_client_class:
                mock_client = MagicMock()
                mock_client_class.return_value = mock_client

                service = GeminiService()
                service.client = mock_client
                return service

    @pytest.mark.asyncio
    async def test_empty_text_embedding(self, gemini_service):
        """Test embedding generation with empty text."""
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock()]
        mock_response.embeddings[0].values = [0.0] * 768
        gemini_service.client.models.embed_content.return_value = mock_response

        result = await gemini_service.generate_embedding("")

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 768

    @pytest.mark.asyncio
    async def test_very_long_text_embedding(self, gemini_service):
        """Test embedding generation with very long text."""
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock()]
        mock_response.embeddings[0].values = [0.1] * 768
        gemini_service.client.models.embed_content.return_value = mock_response

        long_text = "word " * 10000  # Very long text

        result = await gemini_service.generate_embedding(long_text)

        assert isinstance(result, EmbeddingResult)
        assert len(result.embedding) == 768

    @pytest.mark.asyncio
    async def test_batch_embedding_empty_list(self, gemini_service):
        """Test batch embedding with empty list."""
        results = await gemini_service.generate_embeddings_batch([])

        assert results == []

    @pytest.mark.asyncio
    async def test_batch_embedding_single_item(self, gemini_service):
        """Test batch embedding with single item."""
        mock_response = MagicMock()
        mock_response.embeddings = [MagicMock()]
        mock_response.embeddings[0].values = [0.1] * 768
        gemini_service.client.models.embed_content.return_value = mock_response

        results = await gemini_service.generate_embeddings_batch(["single text"])

        assert len(results) == 1
        assert isinstance(results[0], EmbeddingResult)
