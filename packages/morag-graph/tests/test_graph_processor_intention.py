"""Test intention generation functionality."""

import pytest
import asyncio
import os
from unittest.mock import AsyncMock, patch, MagicMock

# Create a simple test class to test intention generation logic
class MockGraphProcessor:
    """Mock GraphProcessor for testing intention generation."""

    def __init__(self, api_key="test-key", model="gemini-1.5-flash"):
        self._llm_config = MagicMock()
        self._llm_config.api_key = api_key
        self._llm_config.model = model

    async def generate_document_intention(self, content: str, max_length: int = 200):
        """Generate a concise intention summary for the document."""
        if not self._llm_config or not self._llm_config.api_key:
            return None

        try:
            # Import here to avoid circular dependencies
            import google.generativeai as genai

            # Configure the API
            genai.configure(api_key=self._llm_config.api_key)

            # Create the model
            model = genai.GenerativeModel(self._llm_config.model or "gemini-1.5-flash")

            # Create intention analysis prompt
            prompt = f"""
Analyze the following document and provide a concise intention summary that captures the document's primary purpose and domain.

The intention should be a single sentence that describes what the document aims to achieve or communicate.

Examples:
- For medical content: "Heal the pineal gland for spiritual enlightenment"
- For organizational documents: "Document explaining the structure of the organization/company"
- For technical guides: "Guide for implementing software architecture patterns"
- For educational content: "Teach fundamental concepts of machine learning"

Document content:
{content[:2000]}...

Provide only the intention summary (maximum {max_length} characters):
"""

            # Generate intention
            response = model.generate_content(prompt)
            intention = response.text.strip()

            # Ensure it's within max length
            if len(intention) > max_length:
                intention = intention[:max_length-3] + "..."

            return intention

        except Exception as e:
            return None


@pytest.fixture
def graph_processor():
    """Create mock graph processor for testing."""
    return MockGraphProcessor()


class TestIntentionGeneration:
    """Test intention generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_document_intention_medical(self, graph_processor):
        """Test intention generation for medical content."""
        content = """
        The pineal gland is a small endocrine gland in the brain that produces melatonin.
        Many spiritual practitioners believe that activating the pineal gland through
        meditation and specific practices can lead to enhanced spiritual awareness
        and enlightenment. This guide provides methods for healing and activating
        the pineal gland for spiritual development.
        """
        
        # Mock the Gemini API response
        mock_response = MagicMock()
        mock_response.text = "Heal the pineal gland for spiritual enlightenment"
        
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model_class:
            
            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_model_class.return_value = mock_model
            
            intention = await graph_processor.generate_document_intention(content)
            
            assert intention == "Heal the pineal gland for spiritual enlightenment"
            mock_model.generate_content.assert_called_once()
            
            # Verify the prompt includes examples and guidance
            call_args = mock_model.generate_content.call_args[0][0]
            assert "For medical content:" in call_args
            assert "For organizational documents:" in call_args
            assert "pineal gland" in call_args
    
    @pytest.mark.asyncio
    async def test_generate_document_intention_organizational(self, graph_processor):
        """Test intention generation for organizational content."""
        content = """
        TechCorp Organizational Structure
        
        Our company is structured with clear hierarchies and reporting lines.
        John Smith serves as CEO, overseeing all operations. The engineering
        division is led by Jane Doe as CTO, while the marketing department
        is headed by Bob Johnson. This document outlines the complete
        organizational structure and reporting relationships.
        """
        
        # Mock the Gemini API response
        mock_response = MagicMock()
        mock_response.text = "Document explaining the structure of the organization/company"
        
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model_class:
            
            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_model_class.return_value = mock_model
            
            intention = await graph_processor.generate_document_intention(content)
            
            assert intention == "Document explaining the structure of the organization/company"
    
    @pytest.mark.asyncio
    async def test_generate_document_intention_truncation(self, graph_processor):
        """Test intention generation with length truncation."""
        content = "Short content"
        
        # Mock a long response that needs truncation
        long_intention = "This is a very long intention summary that exceeds the maximum length limit and should be truncated to fit within the specified character count"
        mock_response = MagicMock()
        mock_response.text = long_intention
        
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model_class:
            
            mock_model = MagicMock()
            mock_model.generate_content.return_value = mock_response
            mock_model_class.return_value = mock_model
            
            intention = await graph_processor.generate_document_intention(content, max_length=50)
            
            assert len(intention) <= 50
            assert intention.endswith("...")
    
    @pytest.mark.asyncio
    async def test_generate_document_intention_api_error(self, graph_processor):
        """Test intention generation with API error."""
        content = "Test content"
        
        with patch('google.generativeai.configure'), \
             patch('google.generativeai.GenerativeModel') as mock_model_class:
            
            mock_model = MagicMock()
            mock_model.generate_content.side_effect = Exception("API Error")
            mock_model_class.return_value = mock_model
            
            intention = await graph_processor.generate_document_intention(content)
            
            assert intention is None
    
    @pytest.mark.asyncio
    async def test_generate_document_intention_no_api_key(self):
        """Test intention generation without API key."""
        processor = MockGraphProcessor(api_key=None)

        intention = await processor.generate_document_intention("Test content")

        assert intention is None
    



if __name__ == "__main__":
    pytest.main([__file__])
