"""Tests for LLM client."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
import httpx

from morag_reasoning.llm import LLMClient, LLMConfig


class TestLLMConfig:
    """Test LLM configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()
        assert config.provider == "gemini"
        assert config.model == "gemini-1.5-flash"
        assert config.temperature == 0.1
        assert config.max_tokens == 2000
        assert config.max_retries == 8
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.5,
            max_tokens=1000,
            max_retries=3
        )
        assert config.provider == "openai"
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 1000
        assert config.max_retries == 3


class TestLLMClient:
    """Test LLM client functionality."""
    
    def test_init_with_config(self, llm_config):
        """Test initialization with provided config."""
        client = LLMClient(llm_config)
        assert client.config == llm_config
        assert client.client is not None
    
    def test_init_without_config(self):
        """Test initialization without config (uses environment)."""
        with patch.dict('os.environ', {
            'MORAG_LLM_PROVIDER': 'openai',
            'MORAG_GEMINI_MODEL': 'gpt-3.5-turbo',
            'GEMINI_API_KEY': 'test-key'
        }):
            client = LLMClient()
            assert client.config.provider == 'openai'
            assert client.config.model == 'gpt-3.5-turbo'
            assert client.config.api_key == 'test-key'
    
    @pytest.mark.asyncio
    async def test_context_manager(self, llm_config):
        """Test async context manager functionality."""
        async with LLMClient(llm_config) as client:
            assert client is not None
            assert client.client is not None
    
    @pytest.mark.asyncio
    async def test_generate_simple_prompt(self, mock_llm_client):
        """Test generating text from a simple prompt."""
        prompt = "What is the capital of France?"
        result = await mock_llm_client.generate(prompt)
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_generate_with_parameters(self, mock_llm_client):
        """Test generating text with custom parameters."""
        prompt = "Tell me about AI"
        result = await mock_llm_client.generate(
            prompt, 
            max_tokens=500, 
            temperature=0.5
        )
        assert isinstance(result, str)
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_generate_from_messages(self, mock_llm_client):
        """Test generating text from message list."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"}
        ]
        result = await mock_llm_client.generate_from_messages(messages)
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_calculate_delay(self, llm_config):
        """Test delay calculation for exponential backoff."""
        # Disable jitter for predictable testing
        llm_config.jitter = False
        client = LLMClient(llm_config)

        # Test first attempt (should be base delay)
        delay1 = client._calculate_delay(1)
        assert delay1 == llm_config.base_delay

        # Test second attempt (should be exponential)
        delay2 = client._calculate_delay(2)
        assert delay2 == llm_config.base_delay * llm_config.exponential_base

        # Test max delay cap
        delay_large = client._calculate_delay(10)
        assert delay_large <= llm_config.max_delay

        # Test with jitter enabled
        llm_config.jitter = True
        client_with_jitter = LLMClient(llm_config)
        delay_jitter = client_with_jitter._calculate_delay(1)
        # Should be between 50% and 100% of base delay
        assert llm_config.base_delay * 0.5 <= delay_jitter <= llm_config.base_delay
    
    @pytest.mark.asyncio
    async def test_gemini_api_call_success(self, llm_config):
        """Test successful Gemini API call."""
        client = LLMClient(llm_config)

        # Mock successful response
        mock_response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Test response"}]
                    }
                }
            ]
        }

        # Create a proper mock response object
        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status.return_value = None

        with patch.object(client.client, 'post', return_value=mock_response_obj) as mock_post:
            result = await client._call_gemini(
                [{"role": "user", "content": "test"}],
                1000,
                0.1
            )
            assert result == "Test response"
    
    @pytest.mark.asyncio
    async def test_openai_api_call_success(self, llm_config):
        """Test successful OpenAI API call."""
        llm_config.provider = "openai"
        client = LLMClient(llm_config)

        # Mock successful response
        mock_response = {
            "choices": [
                {
                    "message": {
                        "content": "Test response"
                    }
                }
            ]
        }

        # Create a proper mock response object
        mock_response_obj = MagicMock()
        mock_response_obj.json.return_value = mock_response
        mock_response_obj.raise_for_status.return_value = None

        with patch.object(client.client, 'post', return_value=mock_response_obj) as mock_post:
            result = await client._call_openai(
                [{"role": "user", "content": "test"}],
                1000,
                0.1
            )
            assert result == "Test response"
    
    @pytest.mark.asyncio
    async def test_api_call_retry_on_failure(self, llm_config):
        """Test API call retry mechanism on failure."""
        llm_config.max_retries = 2
        client = LLMClient(llm_config)

        # Create mock response for successful call
        success_response = MagicMock()
        success_response.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "Success"}]}}
            ]
        }
        success_response.raise_for_status.return_value = None

        with patch.object(client.client, 'post') as mock_post:
            # First call fails, second succeeds
            mock_post.side_effect = [
                httpx.HTTPStatusError("Server error", request=None, response=None),
                success_response
            ]

            result = await client._call_gemini(
                [{"role": "user", "content": "test"}],
                1000,
                0.1
            )
            assert result == "Success"
            assert mock_post.call_count == 2
    
    @pytest.mark.asyncio
    async def test_unsupported_provider_error(self, llm_config):
        """Test error for unsupported provider."""
        llm_config.provider = "unsupported"
        client = LLMClient(llm_config)
        
        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            await client.generate("test prompt")
