"""Tests for LLM batch processing functionality."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from morag_reasoning.llm import LLMClient, LLMConfig
from morag_reasoning.batch_processor import (
    BatchProcessor,
    BatchItem,
    BatchResult,
    TextAnalysisBatchProcessor,
    DocumentChunkBatchProcessor,
    batch_llm_calls,
    batch_text_analysis,
    batch_document_chunks
)


class TestLLMClientBatching:
    """Test LLM client batch processing functionality."""

    def test_batch_config_initialization(self):
        """Test that batch configuration is properly initialized."""
        config = LLMConfig(
            provider="gemini",
            model="gemini-1.5-flash",
            api_key="test-key",
            batch_size=5,
            enable_batching=True,
            batch_delay=1.0,
            max_batch_tokens=500000
        )

        assert config.batch_size == 5
        assert config.enable_batching is True
        assert config.batch_delay == 1.0
        assert config.max_batch_tokens == 500000

    @pytest.mark.asyncio
    async def test_batch_processing_disabled(self):
        """Test that batch processing falls back to individual calls when disabled."""
        config = LLMConfig(
            provider="gemini",
            model="gemini-1.5-flash",
            api_key="test-key",
            enable_batching=False
        )

        client = LLMClient(config)

        # Mock the generate_text method
        client.generate_text = AsyncMock(side_effect=["Response 1", "Response 2", "Response 3"])

        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = await client.generate_batch(prompts)

        assert len(responses) == 3
        assert responses == ["Response 1", "Response 2", "Response 3"]
        assert client.generate_text.call_count == 3

    @pytest.mark.asyncio
    async def test_batch_prompt_creation(self):
        """Test that batch prompts are created correctly."""
        config = LLMConfig(
            provider="gemini",
            model="gemini-1.5-flash",
            api_key="test-key",
            enable_batching=True
        )

        client = LLMClient(config)

        prompts = ["What is AI?", "What is ML?"]
        delimiter = "=" * 50

        batch_prompt = client._create_batch_prompt(prompts, delimiter)

        assert "TASK 1:" in batch_prompt
        assert "TASK 2:" in batch_prompt
        assert "What is AI?" in batch_prompt
        assert "What is ML?" in batch_prompt
        assert delimiter in batch_prompt
        assert "RESPONSE 1:" in batch_prompt
        assert "RESPONSE 2:" in batch_prompt

    @pytest.mark.asyncio
    async def test_batch_response_parsing(self):
        """Test that batch responses are parsed correctly."""
        config = LLMConfig(
            provider="gemini",
            model="gemini-1.5-flash",
            api_key="test-key"
        )

        client = LLMClient(config)

        delimiter = "=" * 50
        response = f"""RESPONSE 1:
Artificial Intelligence is a field of computer science.

{delimiter}

RESPONSE 2:
Machine Learning is a subset of AI.

{delimiter}"""

        original_prompts = ["What is AI?", "What is ML?"]
        parsed_responses = client._parse_batch_response(response, original_prompts, delimiter)

        assert len(parsed_responses) == 2
        assert "Artificial Intelligence" in parsed_responses[0]
        assert "Machine Learning" in parsed_responses[1]

    def test_token_estimation(self):
        """Test token estimation for batch processing."""
        config = LLMConfig(
            provider="gemini",
            model="gemini-1.5-flash",
            api_key="test-key"
        )

        client = LLMClient(config)

        prompts = ["Short prompt", "This is a longer prompt with more words"]
        estimated_tokens = client._estimate_batch_tokens(prompts, max_tokens=100)

        # Should include input tokens, output tokens, and overhead
        assert estimated_tokens > 0
        assert estimated_tokens > len(prompts) * 100  # At least output tokens


class TestBatchProcessor:
    """Test abstract batch processor functionality."""

    class SimpleBatchProcessor(BatchProcessor[str, str]):
        """Simple test implementation of BatchProcessor."""

        def create_prompt(self, item: str) -> str:
            return f"Process: {item}"

        def parse_response(self, response: str, item: str) -> str:
            return f"Processed: {response}"

    @pytest.mark.asyncio
    async def test_batch_processor_basic_functionality(self):
        """Test basic batch processor functionality."""
        # Mock LLM client
        mock_client = MagicMock()
        mock_client.config = MagicMock()
        mock_client.config.batch_size = 2
        mock_client.generate_batch = AsyncMock(return_value=["Result 1", "Result 2"])

        processor = self.SimpleBatchProcessor(mock_client)

        items = ["Item 1", "Item 2"]
        results = await processor.process_batch(items)

        assert len(results) == 2
        assert all(result.success for result in results)
        assert results[0].result == "Processed: Result 1"
        assert results[1].result == "Processed: Result 2"

    @pytest.mark.asyncio
    async def test_batch_processor_error_handling(self):
        """Test batch processor error handling."""
        # Mock LLM client that fails
        mock_client = MagicMock()
        mock_client.config = MagicMock()
        mock_client.config.batch_size = 2
        mock_client.generate_batch = AsyncMock(side_effect=Exception("LLM failed"))

        processor = self.SimpleBatchProcessor(mock_client)

        items = ["Item 1", "Item 2"]
        results = await processor.process_batch(items)

        assert len(results) == 2
        assert all(not result.success for result in results)
        assert all("Batch processing failed" in result.error_message for result in results)


class TestTextAnalysisBatchProcessor:
    """Test text analysis batch processor."""

    @pytest.mark.asyncio
    async def test_entity_extraction_prompt_creation(self):
        """Test entity extraction prompt creation."""
        mock_client = MagicMock()
        mock_client.config = MagicMock()
        mock_client.config.batch_size = 2

        processor = TextAnalysisBatchProcessor(mock_client, "entity_extraction")

        text = "Apple Inc. is a technology company."
        prompt = processor.create_prompt(text)

        assert "Extract entities" in prompt
        assert "JSON" in prompt
        assert text in prompt
        assert "name" in prompt
        assert "type" in prompt
        assert "confidence" in prompt

    @pytest.mark.asyncio
    async def test_relation_extraction_prompt_creation(self):
        """Test relation extraction prompt creation."""
        mock_client = MagicMock()
        mock_client.config = MagicMock()
        mock_client.config.batch_size = 2

        processor = TextAnalysisBatchProcessor(mock_client, "relation_extraction")

        text = "Steve Jobs founded Apple Inc."
        prompt = processor.create_prompt(text)

        assert "Extract relations" in prompt
        assert "JSON" in prompt
        assert text in prompt
        assert "source" in prompt
        assert "target" in prompt
        assert "relation" in prompt

    def test_response_parsing_json(self):
        """Test parsing JSON responses."""
        mock_client = MagicMock()
        processor = TextAnalysisBatchProcessor(mock_client, "entity_extraction")

        response = '[{"name": "Apple Inc.", "type": "ORGANIZATION", "confidence": 0.9}]'
        text = "Apple Inc. is a company."

        result = processor.parse_response(response, text)

        assert "analysis" in result
        assert isinstance(result["analysis"], list)
        assert len(result["analysis"]) == 1
        assert result["analysis"][0]["name"] == "Apple Inc."

    def test_response_parsing_text(self):
        """Test parsing text responses."""
        mock_client = MagicMock()
        processor = TextAnalysisBatchProcessor(mock_client, "summarization")

        response = "This is a summary of the text."
        text = "Original text here."

        result = processor.parse_response(response, text)

        assert result["analysis"] == "This is a summary of the text."
        assert result["original_text"] == "Original text here."


class TestDocumentChunkBatchProcessor:
    """Test document chunk batch processor."""

    def test_extraction_prompt_creation(self):
        """Test extraction prompt creation for document chunks."""
        mock_client = MagicMock()
        processor = DocumentChunkBatchProcessor(mock_client, "extraction")

        chunk = {
            "id": "chunk_1",
            "text": "Apple Inc. was founded by Steve Jobs.",
            "document_id": "doc_1"
        }

        prompt = processor.create_prompt(chunk)

        assert "Extract entities and relations" in prompt
        assert "chunk_1" in prompt
        assert "doc_1" in prompt
        assert chunk["text"] in prompt
        assert "JSON" in prompt

    def test_extraction_response_parsing(self):
        """Test parsing extraction responses."""
        mock_client = MagicMock()
        processor = DocumentChunkBatchProcessor(mock_client, "extraction")

        response = '{"entities": [{"name": "Apple Inc.", "type": "ORGANIZATION"}], "relations": [{"source": "Steve Jobs", "target": "Apple Inc.", "relation": "FOUNDED"}]}'
        chunk = {"id": "chunk_1", "document_id": "doc_1"}

        result = processor.parse_response(response, chunk)

        assert result["chunk_id"] == "chunk_1"
        assert result["document_id"] == "doc_1"
        assert len(result["entities"]) == 1
        assert len(result["relations"]) == 1
        assert result["entities"][0]["name"] == "Apple Inc."


class TestConvenienceFunctions:
    """Test convenience functions for batch processing."""

    @pytest.mark.asyncio
    async def test_batch_llm_calls(self):
        """Test the batch_llm_calls convenience function."""
        mock_client = MagicMock()
        mock_client.config = MagicMock()
        mock_client.config.batch_size = 2
        mock_client.generate_batch = AsyncMock(return_value=["Response 1", "Response 2"])

        def prompt_creator(item: str) -> str:
            return f"Process: {item}"

        def response_parser(response: str, item: str) -> str:
            return f"Parsed: {response}"

        items = ["Item 1", "Item 2"]
        results = await batch_llm_calls(
            mock_client, items, prompt_creator, response_parser
        )

        assert len(results) == 2
        assert all(result.success for result in results)
        assert results[0].result == "Parsed: Response 1"
        assert results[1].result == "Parsed: Response 2"

    @pytest.mark.asyncio
    async def test_batch_text_analysis(self):
        """Test the batch_text_analysis convenience function."""
        mock_client = MagicMock()
        mock_client.config = MagicMock()
        mock_client.config.batch_size = 2
        mock_client.generate_batch = AsyncMock(return_value=['[{"name": "Test", "type": "ENTITY"}]'])

        texts = ["Test text"]
        results = await batch_text_analysis(mock_client, texts, "entity_extraction")

        assert len(results) == 1
        assert results[0].success
        assert "analysis" in results[0].result

    @pytest.mark.asyncio
    async def test_batch_document_chunks(self):
        """Test the batch_document_chunks convenience function."""
        mock_client = MagicMock()
        mock_client.config = MagicMock()
        mock_client.config.batch_size = 2
        mock_client.generate_batch = AsyncMock(return_value=['{"entities": [], "relations": []}'])

        chunks = [{"id": "chunk_1", "text": "Test text", "document_id": "doc_1"}]
        results = await batch_document_chunks(mock_client, chunks, "extraction")

        assert len(results) == 1
        assert results[0].success
        assert "entities" in results[0].result
        assert "relations" in results[0].result
