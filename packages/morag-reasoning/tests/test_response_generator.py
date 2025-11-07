"""Tests for response generation system."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from morag_reasoning.citation_manager import CitationFormat, CitedFact, SourceReference
from morag_reasoning.graph_fact_extractor import ExtractedFact, FactType
from morag_reasoning.llm import LLMClient
from morag_reasoning.response_generator import (
    GeneratedResponse,
    ResponseFormat,
    ResponseGenerator,
    ResponseOptions,
    ResponseStructure,
)


class TestResponseGenerator:
    """Test the response generator."""

    @pytest.fixture
    def mock_llm_client(self):
        """Mock LLM client for testing."""
        return MagicMock(spec=LLMClient)

    @pytest.fixture
    def sample_facts(self):
        """Sample cited facts for testing."""
        # Create sample extracted facts
        extracted_fact1 = ExtractedFact(
            fact_id="fact_001",
            content="Einstein developed the theory of relativity",
            fact_type=FactType.DIRECT,
            confidence=0.9,
            source_entities=["ent_einstein", "ent_relativity"],
            source_relations=["rel_developed"],
            source_documents=["doc_physics"],
            extraction_path=["ent_einstein", "ent_relativity"],
            context={"relation_type": "DEVELOPED"},
            metadata={"extraction_method": "direct_triplet"},
        )

        extracted_fact2 = ExtractedFact(
            fact_id="fact_002",
            content="The theory of relativity was published in 1905",
            fact_type=FactType.DIRECT,
            confidence=0.8,
            source_entities=["ent_relativity"],
            source_relations=["rel_published"],
            source_documents=["doc_physics"],
            extraction_path=["ent_relativity"],
            context={"year": "1905"},
            metadata={"extraction_method": "direct_triplet"},
        )

        # Create source references
        source1 = SourceReference(
            document_id="doc_physics",
            document_title="Physics Papers",
            chunk_id="chunk_001",
            page_number=42,
            timestamp="1905-06-30",
            confidence=0.9,
        )

        source2 = SourceReference(
            document_id="doc_physics",
            document_title="Physics Papers",
            chunk_id="chunk_002",
            page_number=43,
            timestamp="1905-06-30",
            confidence=0.8,
        )

        # Create cited facts
        cited_fact1 = CitedFact(
            fact=extracted_fact1,
            score=0.9,
            sources=[source1],
            citation_text="Physics Papers, p. 42",
            citation_format=CitationFormat.STRUCTURED,
            verification_status="verified",
            metadata={},
        )

        cited_fact2 = CitedFact(
            fact=extracted_fact2,
            score=0.8,
            sources=[source2],
            citation_text="Physics Papers, p. 43",
            citation_format=CitationFormat.STRUCTURED,
            verification_status="verified",
            metadata={},
        )

        return [cited_fact1, cited_fact2]

    @pytest.mark.asyncio
    async def test_generator_initialization(self, mock_llm_client):
        """Test response generator initialization."""
        config = {
            "llm_enabled": False,  # Disable LLM for testing
            "default_max_length": 1000,
            "min_facts_required": 1,
        }

        generator = ResponseGenerator(mock_llm_client, config)

        assert generator.default_max_length == 1000
        assert generator.min_facts_required == 1
        assert not generator.llm_enabled

    @pytest.mark.asyncio
    async def test_fallback_response_generation(self, mock_llm_client, sample_facts):
        """Test fallback response generation when LLM is disabled."""
        config = {"llm_enabled": False}
        generator = ResponseGenerator(mock_llm_client, config)

        options = ResponseOptions(format=ResponseFormat.DETAILED, max_length=500)

        response = await generator.generate_response(
            sample_facts, "What did Einstein develop?", options
        )

        assert isinstance(response, GeneratedResponse)
        assert response.content
        assert "Einstein" in response.content
        assert "relativity" in response.content
        assert response.confidence_score > 0
        assert response.word_count > 0
        assert len(response.facts_used) == 2

    @pytest.mark.asyncio
    async def test_different_response_formats(self, mock_llm_client, sample_facts):
        """Test different response formats."""
        config = {"llm_enabled": False}
        generator = ResponseGenerator(mock_llm_client, config)

        formats = [
            ResponseFormat.DETAILED,
            ResponseFormat.SUMMARY,
            ResponseFormat.BULLET_POINTS,
        ]

        for format_type in formats:
            options = ResponseOptions(format=format_type)

            response = await generator.generate_response(
                sample_facts, "What did Einstein develop?", options
            )

            assert isinstance(response, GeneratedResponse)
            assert response.content
            assert response.metadata["generation_method"] == "fallback"

    @pytest.mark.asyncio
    async def test_empty_facts_handling(self, mock_llm_client):
        """Test handling of empty facts list."""
        config = {"llm_enabled": False}
        generator = ResponseGenerator(mock_llm_client, config)

        response = await generator.generate_response(
            [], "Test query", ResponseOptions()
        )

        assert isinstance(response, GeneratedResponse)
        assert not response.content or "No facts available" in response.content
        assert response.confidence_score == 0.0
        assert len(response.facts_used) == 0

    @pytest.mark.asyncio
    async def test_insufficient_facts_handling(self, mock_llm_client):
        """Test handling when facts don't meet minimum requirements."""
        config = {
            "llm_enabled": False,
            "min_facts_required": 5,  # Require more facts than provided
        }
        generator = ResponseGenerator(mock_llm_client, config)

        # Create a single low-quality fact
        low_quality_fact = CitedFact(
            fact=ExtractedFact(
                fact_id="fact_low",
                content="Low quality fact",
                fact_type=FactType.DIRECT,
                confidence=0.1,
                source_entities=[],
                source_relations=[],
                source_documents=[],
                extraction_path=[],
                context={},
                metadata={},
            ),
            score=0.1,
            sources=[],
            citation_text="",
            citation_format=CitationFormat.STRUCTURED,
            verification_status="unverified",
            metadata={},
        )

        response = await generator.generate_response(
            [low_quality_fact], "Test query", ResponseOptions()
        )

        assert isinstance(response, GeneratedResponse)
        assert "Insufficient facts" in response.content
        assert response.confidence_score == 0.0

    @pytest.mark.asyncio
    async def test_fact_preparation_and_filtering(self, mock_llm_client, sample_facts):
        """Test fact preparation and filtering."""
        config = {
            "llm_enabled": False,
            "max_facts_to_use": 1,  # Limit to 1 fact
            "min_fact_score": 0.5,
        }
        generator = ResponseGenerator(mock_llm_client, config)

        # Add a low-score fact that should be filtered out
        low_score_fact = CitedFact(
            fact=ExtractedFact(
                fact_id="fact_low",
                content="Low score fact",
                fact_type=FactType.DIRECT,
                confidence=0.3,
                source_entities=[],
                source_relations=[],
                source_documents=[],
                extraction_path=[],
                context={},
                metadata={},
            ),
            score=0.3,  # Below threshold
            sources=[],
            citation_text="",
            citation_format=CitationFormat.STRUCTURED,
            verification_status="unverified",
            metadata={},
        )

        all_facts = sample_facts + [low_score_fact]

        response = await generator.generate_response(
            all_facts, "Test query", ResponseOptions()
        )

        # Should use only the high-quality facts, limited by max_facts_to_use
        assert len(response.facts_used) == 1
        assert response.facts_used[0] in ["fact_001", "fact_002"]  # High-quality facts

    @pytest.mark.asyncio
    async def test_response_options_handling(self, mock_llm_client, sample_facts):
        """Test different response options."""
        config = {"llm_enabled": False}
        generator = ResponseGenerator(mock_llm_client, config)

        # Test with custom options
        options = ResponseOptions(
            format=ResponseFormat.SUMMARY,
            structure=ResponseStructure.ANALYTICAL,
            max_length=200,
            include_reasoning=True,
            include_confidence=True,
            tone="casual",
            language="en",
        )

        response = await generator.generate_response(
            sample_facts, "What did Einstein develop?", options
        )

        assert isinstance(response, GeneratedResponse)
        assert response.content
        assert response.reasoning
        assert response.metadata["options_used"]["format"] == "summary"
        assert response.metadata["options_used"]["tone"] == "casual"

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_llm_client):
        """Test error handling in response generation."""
        config = {"llm_enabled": False}
        generator = ResponseGenerator(mock_llm_client, config)

        # Mock an error in the fallback generation
        with patch.object(
            generator, "_generate_fallback", side_effect=Exception("Test error")
        ):
            response = await generator.generate_response(
                [], "Test query", ResponseOptions()  # Empty facts will trigger fallback
            )

        assert isinstance(response, GeneratedResponse)
        assert "Error generating response" in response.content
        assert response.confidence_score == 0.0
        assert "error" in response.metadata

    @pytest.mark.asyncio
    async def test_response_metadata(self, mock_llm_client, sample_facts):
        """Test response metadata generation."""
        config = {"llm_enabled": False}
        generator = ResponseGenerator(mock_llm_client, config)

        response = await generator.generate_response(
            sample_facts, "What did Einstein develop?", ResponseOptions()
        )

        # Check metadata completeness
        assert "generation_method" in response.metadata
        assert "options_used" in response.metadata
        assert "num_facts_used" in response.metadata
        assert "query_length" in response.metadata
        assert "response_length" in response.metadata

        assert response.metadata["generation_method"] == "fallback"
        assert response.metadata["num_facts_used"] == 2
        assert response.generation_time > 0

    @pytest.mark.asyncio
    async def test_key_points_extraction(self, mock_llm_client, sample_facts):
        """Test key points extraction from facts."""
        config = {"llm_enabled": False}
        generator = ResponseGenerator(mock_llm_client, config)

        response = await generator.generate_response(
            sample_facts, "What did Einstein develop?", ResponseOptions()
        )

        # Should extract key points from the facts
        assert len(response.key_points) > 0
        assert any("Einstein" in point for point in response.key_points)
        assert any("relativity" in point for point in response.key_points)
