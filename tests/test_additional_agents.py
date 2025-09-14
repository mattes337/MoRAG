"""Tests for additional PydanticAI agents (summarization, query analysis)."""

import pytest
import asyncio
import sys
import os

# Add the packages directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'morag-core', 'src'))

from morag_core.ai import (
    SummarizationAgent,
    QueryAnalysisAgent,
    SummaryResult,
    QueryAnalysisResult,
    ConfidenceLevel,
)


class TestSummarizationAgent:
    """Test the PydanticAI summarization agent."""
    
    def test_agent_creation(self):
        """Test creating summarization agent."""
        agent = SummarizationAgent(max_summary_length=500)
        
        assert agent.max_summary_length == 500
        assert agent.get_result_type() == SummaryResult
        assert "summarization agent" in agent.get_system_prompt().lower()
    
    def test_agent_system_prompt(self):
        """Test the system prompt contains required elements."""
        agent = SummarizationAgent()
        prompt = agent.get_system_prompt()
        
        # Check for key elements
        assert "summary" in prompt.lower()
        assert "key_points" in prompt.lower()
        assert "confidence" in prompt.lower()
        assert "compression_ratio" in prompt.lower()
        assert "concise" in prompt.lower()
        assert "comprehensive" in prompt.lower()
    
    def test_prompt_building(self):
        """Test summarization prompt building."""
        agent = SummarizationAgent()
        
        text = "This is a sample text for testing."
        prompt = agent._build_summarization_prompt(
            text=text,
            max_length=100,
            style="concise",
            context="Test context"
        )
        
        assert "concise" in prompt.lower()
        assert "100 characters" in prompt
        assert "Test context" in prompt
        assert text in prompt
    
    @pytest.mark.asyncio
    async def test_empty_text_handling(self):
        """Test handling of empty text."""
        agent = SummarizationAgent()
        
        # Empty text
        result = await agent.summarize_text("")
        assert result.summary == ""
        assert result.key_points == []
        assert result.word_count == 0
        assert result.compression_ratio == 0.0
        
        # Whitespace-only text
        result = await agent.summarize_text("   \n\t   ")
        assert result.summary == ""
        assert result.key_points == []
        assert result.word_count == 0
        assert result.compression_ratio == 0.0
    
    @pytest.mark.asyncio
    async def test_summarize_chunks_empty(self):
        """Test summarizing empty chunks."""
        agent = SummarizationAgent()
        
        # Empty chunks list
        result = await agent.summarize_chunks([])
        assert result.summary == ""
        assert result.key_points == []
        assert result.word_count == 0
        assert result.compression_ratio == 0.0
    
    def test_summarization_styles(self):
        """Test different summarization styles."""
        agent = SummarizationAgent()

        text = "Sample text"

        # Test different styles and their expected content
        style_tests = {
            "concise": "concise",
            "detailed": "comprehensive",
            "bullet": "bullet",
            "abstract": "abstract"
        }

        for style, expected_word in style_tests.items():
            prompt = agent._build_summarization_prompt(text, 100, style, None)
            assert expected_word in prompt.lower(), f"Expected '{expected_word}' in prompt for style '{style}'"
    
    @pytest.mark.asyncio
    async def test_interface_compatibility(self):
        """Test that the agent interface is compatible with expected usage."""
        agent = SummarizationAgent()
        
        # Test method signatures
        import inspect
        
        # Check summarize_text signature
        sig = inspect.signature(agent.summarize_text)
        assert 'text' in sig.parameters
        assert 'max_length' in sig.parameters
        assert 'style' in sig.parameters
        assert 'context' in sig.parameters
        
        # Check summarize_document signature
        sig = inspect.signature(agent.summarize_document)
        assert 'text' in sig.parameters
        assert 'title' in sig.parameters
        assert 'document_type' in sig.parameters
        assert 'max_length' in sig.parameters
        
        # Check summarize_chunks signature
        sig = inspect.signature(agent.summarize_chunks)
        assert 'chunks' in sig.parameters
        assert 'max_length' in sig.parameters
        assert 'preserve_structure' in sig.parameters


class TestQueryAnalysisAgent:
    """Test the PydanticAI query analysis agent."""
    
    def test_agent_creation(self):
        """Test creating query analysis agent."""
        agent = QueryAnalysisAgent()
        
        assert agent.get_result_type() == QueryAnalysisResult
        assert "query analysis agent" in agent.get_system_prompt().lower()
    
    def test_agent_system_prompt(self):
        """Test the system prompt contains required elements."""
        agent = QueryAnalysisAgent()
        prompt = agent.get_system_prompt()
        
        # Check for key elements
        assert "intent" in prompt.lower()
        assert "entities" in prompt.lower()
        assert "keywords" in prompt.lower()
        assert "query_type" in prompt.lower()
        assert "complexity" in prompt.lower()
        assert "confidence" in prompt.lower()
        
        # Check for intent categories
        assert "SEARCH" in prompt
        assert "QUESTION" in prompt
        assert "COMPARISON" in prompt
        assert "ANALYSIS" in prompt
        
        # Check for query types
        assert "FACTUAL" in prompt
        assert "ANALYTICAL" in prompt
        assert "PROCEDURAL" in prompt
        
        # Check for complexity levels
        assert "SIMPLE" in prompt
        assert "MEDIUM" in prompt
        assert "COMPLEX" in prompt
    
    def test_prompt_building(self):
        """Test query analysis prompt building."""
        agent = QueryAnalysisAgent()
        
        query = "What is the capital of France?"
        context = "Geography quiz"
        history = ["What is the largest country?", "Name three European capitals"]
        
        prompt = agent._build_analysis_prompt(query, context, history)
        
        assert query in prompt
        assert context in prompt
        assert "largest country" in prompt  # From history
        assert "European capitals" in prompt  # From history
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self):
        """Test handling of empty queries."""
        agent = QueryAnalysisAgent()
        
        # Empty query
        result = await agent.analyze_query("")
        assert result.intent == "unknown"
        assert result.entities == []
        assert result.keywords == []
        assert result.query_type == "unknown"
        assert result.complexity == "simple"
        assert result.confidence == ConfidenceLevel.LOW
        assert "error" in result.metadata
        
        # Whitespace-only query
        result = await agent.analyze_query("   \n\t   ")
        assert result.intent == "unknown"
        assert result.entities == []
        assert result.keywords == []
        assert result.query_type == "unknown"
        assert result.complexity == "simple"
        assert result.confidence == ConfidenceLevel.LOW
    
    @pytest.mark.asyncio
    async def test_batch_analysis_empty(self):
        """Test batch analysis with empty input."""
        agent = QueryAnalysisAgent()
        
        # Empty queries list
        results = await agent.analyze_batch_queries([])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_extract_search_terms_interface(self):
        """Test search terms extraction interface."""
        agent = QueryAnalysisAgent()
        
        # Test method signature
        import inspect
        sig = inspect.signature(agent.extract_search_terms)
        assert 'query' in sig.parameters
        assert 'expand_terms' in sig.parameters
        
        # Test with simple query (would normally call API)
        # Here we're just testing the interface exists
        assert callable(agent.extract_search_terms)
    
    @pytest.mark.asyncio
    async def test_interface_compatibility(self):
        """Test that the agent interface is compatible with expected usage."""
        agent = QueryAnalysisAgent()
        
        # Test method signatures
        import inspect
        
        # Check analyze_query signature
        sig = inspect.signature(agent.analyze_query)
        assert 'query' in sig.parameters
        assert 'context' in sig.parameters
        assert 'user_history' in sig.parameters
        
        # Check analyze_batch_queries signature
        sig = inspect.signature(agent.analyze_batch_queries)
        assert 'queries' in sig.parameters
        assert 'context' in sig.parameters
        
        # Check extract_search_terms signature
        sig = inspect.signature(agent.extract_search_terms)
        assert 'query' in sig.parameters
        assert 'expand_terms' in sig.parameters


class TestAgentIntegration:
    """Integration tests for the additional agents."""
    
    def test_agent_result_types(self):
        """Test that agents return correct result types."""
        summarization_agent = SummarizationAgent()
        query_agent = QueryAnalysisAgent()
        
        # Check result types
        assert summarization_agent.get_result_type() == SummaryResult
        assert query_agent.get_result_type() == QueryAnalysisResult
    
    def test_agent_system_prompts(self):
        """Test that all agents have proper system prompts."""
        agents = [
            SummarizationAgent(),
            QueryAnalysisAgent(),
        ]
        
        for agent in agents:
            prompt = agent.get_system_prompt()
            assert isinstance(prompt, str)
            assert len(prompt) > 100  # Should be substantial
            assert "agent" in prompt.lower()
    
    @pytest.mark.asyncio
    async def test_error_handling_consistency(self):
        """Test that agents handle errors consistently."""
        summarization_agent = SummarizationAgent()
        query_agent = QueryAnalysisAgent()
        
        # Test empty input handling
        summary_result = await summarization_agent.summarize_text("")
        query_result = await query_agent.analyze_query("")
        
        # Both should handle empty input gracefully
        assert isinstance(summary_result, SummaryResult)
        assert isinstance(query_result, QueryAnalysisResult)
        
        # Both should have appropriate confidence levels for empty input
        assert summary_result.confidence in [ConfidenceLevel.HIGH]  # Empty is handled well
        assert query_result.confidence == ConfidenceLevel.LOW  # Empty query is problematic
    
    def test_agent_configuration(self):
        """Test agent configuration options."""
        # Test summarization agent with custom max length
        agent1 = SummarizationAgent(max_summary_length=2000)
        assert agent1.max_summary_length == 2000
        
        # Test default configuration
        agent2 = SummarizationAgent()
        assert agent2.max_summary_length == 1000  # Default value
        
        # Test query analysis agent (no specific config for now)
        agent3 = QueryAnalysisAgent()
        assert hasattr(agent3, 'logger')
    
    def test_logging_setup(self):
        """Test that agents have proper logging setup."""
        agents = [
            SummarizationAgent(),
            QueryAnalysisAgent(),
        ]
        
        for agent in agents:
            assert hasattr(agent, 'logger')
            assert agent.logger is not None


if __name__ == "__main__":
    pytest.main([__file__])
