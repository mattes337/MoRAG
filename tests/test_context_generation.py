#!/usr/bin/env python3
"""
Unit tests for the context generation testing script.
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import sys

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
cli_path = project_root / 'cli'
sys.path.insert(0, str(cli_path))

# Mock the imports to avoid dependency issues in tests
with patch.dict('sys.modules', {
    'morag_graph': MagicMock(),
    'morag_graph.ai.entity_agent': MagicMock(),
    'morag_graph.utils.entity_normalizer': MagicMock(),
    'morag.database_factory': MagicMock(),
    'morag_reasoning': MagicMock(),
    'morag_core.ai': MagicMock(),
}):
    import test_context_generation as tcg
    ContextGenerationResult = tcg.ContextGenerationResult
    AgenticContextGenerator = tcg.AgenticContextGenerator


class TestContextGenerationResult:
    """Test the ContextGenerationResult class."""
    
    def test_initialization(self):
        """Test that ContextGenerationResult initializes correctly."""
        result = ContextGenerationResult()
        
        assert result.prompt == ""
        assert result.extracted_entities == []
        assert result.graph_entities == []
        assert result.graph_relations == []
        assert result.vector_chunks == []
        assert result.reasoning_paths == []
        assert result.context_score == 0.0
        assert result.final_response == ""
        assert result.processing_steps == []
        assert result.performance_metrics == {}
        assert result.error is None
        assert result.timestamp is not None


class TestAgenticContextGenerator:
    """Test the AgenticContextGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_neo4j = MagicMock()
        self.mock_qdrant = MagicMock()
        self.mock_llm = MagicMock()
        
        self.generator = AgenticContextGenerator(
            neo4j_storage=self.mock_neo4j,
            qdrant_storage=self.mock_qdrant,
            llm_client=self.mock_llm,
            verbose=False
        )
    
    def test_initialization(self):
        """Test that AgenticContextGenerator initializes correctly."""
        assert self.generator.neo4j_storage == self.mock_neo4j
        assert self.generator.qdrant_storage == self.mock_qdrant
        assert self.generator.llm_client == self.mock_llm
        assert self.generator.verbose is False
    
    def test_initialization_without_dependencies(self):
        """Test initialization without database dependencies."""
        generator = AgenticContextGenerator(verbose=True)
        
        assert generator.neo4j_storage is None
        assert generator.qdrant_storage is None
        assert generator.llm_client is None
        assert generator.verbose is True
    
    def test_log_method(self):
        """Test the logging method."""
        # Test with verbose=False (should not print)
        generator = AgenticContextGenerator(verbose=False)
        generator._log("Test message", "Test details")  # Should not raise error
        
        # Test with verbose=True
        generator = AgenticContextGenerator(verbose=True)
        with patch('builtins.print') as mock_print:
            generator._log("Test message", "Test details")
            assert mock_print.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_extract_entities_agentic_no_llm(self):
        """Test entity extraction without LLM client."""
        generator = AgenticContextGenerator(llm_client=None)
        
        entities = await generator._extract_entities_agentic("test prompt")
        
        assert entities == []
    
    @pytest.mark.asyncio
    async def test_extract_entities_agentic_with_llm(self):
        """Test entity extraction with LLM client."""
        # Mock LLM response
        mock_llm_response = '''
        [
            {"name": "nutrition", "type": "CONCEPT", "confidence": 0.9, "relevance": "Key topic"},
            {"name": "ADHD", "type": "MEDICAL_CONDITION", "confidence": 0.95, "relevance": "Target condition"}
        ]
        '''
        
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = mock_llm_response
        
        generator = AgenticContextGenerator(llm_client=mock_llm)
        
        entities = await generator._extract_entities_agentic("How does nutrition affect ADHD?")
        
        assert len(entities) == 2
        assert entities[0]["name"] == "nutrition"
        assert entities[0]["type"] == "CONCEPT"
        assert entities[1]["name"] == "ADHD"
        assert entities[1]["type"] == "MEDICAL_CONDITION"
    
    @pytest.mark.asyncio
    async def test_search_vector_documents_no_qdrant(self):
        """Test vector search without Qdrant storage."""
        generator = AgenticContextGenerator(qdrant_storage=None)
        
        chunks = await generator._search_vector_documents("test prompt")
        
        assert chunks == []
    
    @pytest.mark.asyncio
    async def test_search_vector_documents_with_qdrant(self):
        """Test vector search with Qdrant storage."""
        # Mock Qdrant response
        mock_results = [
            {
                "content": "Test document content",
                "metadata": {"source": "test.pdf"},
                "score": 0.85
            }
        ]
        
        mock_qdrant = AsyncMock()
        mock_qdrant.search_entities.return_value = mock_results
        
        generator = AgenticContextGenerator(qdrant_storage=mock_qdrant)
        
        chunks = await generator._search_vector_documents("test prompt")
        
        assert len(chunks) == 1
        assert chunks[0]["content"] == "Test document content"
        assert chunks[0]["score"] == 0.85
    
    @pytest.mark.asyncio
    async def test_score_context_no_llm(self):
        """Test context scoring without LLM client."""
        generator = AgenticContextGenerator(llm_client=None)
        
        result = ContextGenerationResult()
        result.extracted_entities = [{"name": "test"}] * 3
        result.graph_entities = [{"name": "test"}] * 5
        result.vector_chunks = [{"content": "test"}] * 2
        
        score = await generator._score_context("test prompt", result)
        
        assert 0.0 <= score <= 1.0
    
    @pytest.mark.asyncio
    async def test_score_context_with_llm(self):
        """Test context scoring with LLM client."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "0.75"
        
        generator = AgenticContextGenerator(llm_client=mock_llm)
        
        result = ContextGenerationResult()
        result.extracted_entities = [{"name": "test"}]
        
        score = await generator._score_context("test prompt", result)
        
        assert score == 0.75
    
    @pytest.mark.asyncio
    async def test_generate_final_response_no_llm(self):
        """Test final response generation without LLM client."""
        generator = AgenticContextGenerator(llm_client=None)
        
        result = ContextGenerationResult()
        response = await generator._generate_final_response("test prompt", result)
        
        assert "LLM client not available" in response
    
    @pytest.mark.asyncio
    async def test_generate_final_response_with_llm(self):
        """Test final response generation with LLM client."""
        mock_llm = AsyncMock()
        mock_llm.generate.return_value = "This is a test response."
        
        generator = AgenticContextGenerator(llm_client=mock_llm)
        
        result = ContextGenerationResult()
        result.context_score = 0.8
        result.extracted_entities = [{"name": "test", "type": "CONCEPT", "relevance": "test"}]
        
        response = await generator._generate_final_response("test prompt", result)
        
        assert response == "This is a test response."
        mock_llm.generate.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_context_full_flow(self):
        """Test the complete context generation flow."""
        # Mock all dependencies
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '[{"name": "test", "type": "CONCEPT", "confidence": 0.9, "relevance": "test"}]',  # Entity extraction
            "0.8",  # Context scoring
            "Final response based on context."  # Final response
        ]
        
        mock_qdrant = AsyncMock()
        mock_qdrant.search_entities.return_value = [
            {"content": "Test content", "metadata": {}, "score": 0.9}
        ]
        
        generator = AgenticContextGenerator(
            neo4j_storage=None,  # Skip Neo4j for this test
            qdrant_storage=mock_qdrant,
            llm_client=mock_llm,
            verbose=False
        )
        
        result = await generator.generate_context("test prompt")
        
        assert result.prompt == "test prompt"
        assert len(result.extracted_entities) == 1
        assert len(result.vector_chunks) == 1
        assert result.context_score == 0.8
        assert result.final_response == "Final response based on context."
        assert result.error is None
        assert len(result.processing_steps) > 0
    
    @pytest.mark.asyncio
    async def test_generate_context_with_error(self):
        """Test context generation with error handling."""
        # Mock LLM to raise an exception
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = Exception("Test error")
        
        generator = AgenticContextGenerator(llm_client=mock_llm)
        
        result = await generator.generate_context("test prompt")
        
        assert result.error is not None
        assert "Test error" in result.error


class TestIntegration:
    """Integration tests for the context generation system."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_mock_flow(self):
        """Test end-to-end flow with mocked dependencies."""
        # This test simulates the complete flow without real database connections
        
        # Mock LLM responses
        mock_llm = AsyncMock()
        mock_llm.generate.side_effect = [
            '[{"name": "AI", "type": "CONCEPT", "confidence": 0.9, "relevance": "Main topic"}]',
            "0.85",
            "AI has significant applications in healthcare through diagnostic tools and treatment optimization."
        ]
        
        # Mock vector search
        mock_qdrant = AsyncMock()
        mock_qdrant.search_entities.return_value = [
            {
                "content": "AI in healthcare improves diagnostic accuracy",
                "metadata": {"source": "medical_ai.pdf"},
                "score": 0.92
            }
        ]
        
        generator = AgenticContextGenerator(
            neo4j_storage=None,
            qdrant_storage=mock_qdrant,
            llm_client=mock_llm,
            verbose=False
        )
        
        result = await generator.generate_context("How is AI used in healthcare?")
        
        # Verify the complete flow worked
        assert result.error is None
        assert result.prompt == "How is AI used in healthcare?"
        assert len(result.extracted_entities) == 1
        assert result.extracted_entities[0]["name"] == "AI"
        assert len(result.vector_chunks) == 1
        assert result.context_score == 0.85
        assert "healthcare" in result.final_response.lower()
        assert len(result.processing_steps) >= 3  # Entity extraction, vector search, scoring
        assert "total_time_seconds" in result.performance_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
