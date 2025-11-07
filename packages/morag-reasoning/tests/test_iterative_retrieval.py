"""Tests for iterative retrieval components."""

from unittest.mock import AsyncMock

import pytest
from morag_reasoning.iterative_retrieval import (
    ContextAnalysis,
    ContextGap,
    IterativeRetriever,
    RetrievalContext,
)


class TestContextGap:
    """Test ContextGap data class."""

    def test_context_gap_creation(self):
        """Test creating a context gap."""
        gap = ContextGap(
            gap_type="missing_entity",
            description="Need more information about Apple",
            entities_needed=["Apple Inc."],
            priority=0.8,
        )

        assert gap.gap_type == "missing_entity"
        assert gap.description == "Need more information about Apple"
        assert gap.entities_needed == ["Apple Inc."]
        assert gap.priority == 0.8
        assert gap.relations_needed == []  # Default empty list


class TestContextAnalysis:
    """Test ContextAnalysis data class."""

    def test_context_analysis_creation(self):
        """Test creating a context analysis."""
        gaps = [
            ContextGap(
                gap_type="missing_entity",
                description="Need entity info",
                entities_needed=["entity1"],
            )
        ]

        analysis = ContextAnalysis(
            is_sufficient=False,
            confidence=0.6,
            gaps=gaps,
            reasoning="Missing key information",
            suggested_queries=["What is entity1?"],
        )

        assert not analysis.is_sufficient
        assert analysis.confidence == 0.6
        assert len(analysis.gaps) == 1
        assert analysis.reasoning == "Missing key information"
        assert analysis.suggested_queries == ["What is entity1?"]


class TestRetrievalContext:
    """Test RetrievalContext data class."""

    def test_retrieval_context_creation(self):
        """Test creating a retrieval context."""
        context = RetrievalContext(
            entities={"Apple": {"type": "ORG"}},
            relations=[
                {"subject": "Apple", "predicate": "FOUNDED_BY", "object": "Steve Jobs"}
            ],
            documents=[{"id": "doc1", "content": "Apple is a company"}],
            metadata={"source": "test"},
        )

        assert len(context.entities) == 1
        assert len(context.relations) == 1
        assert len(context.documents) == 1
        assert context.metadata["source"] == "test"
        assert len(context.paths) == 0  # Default empty list

    def test_retrieval_context_defaults(self):
        """Test retrieval context with default values."""
        context = RetrievalContext()

        assert len(context.entities) == 0
        assert len(context.relations) == 0
        assert len(context.documents) == 0
        assert len(context.paths) == 0
        assert len(context.metadata) == 0


class TestIterativeRetriever:
    """Test iterative retriever functionality."""

    def test_init(self, mock_llm_client, mock_graph_engine, mock_vector_retriever):
        """Test retriever initialization."""
        retriever = IterativeRetriever(
            llm_client=mock_llm_client,
            graph_engine=mock_graph_engine,
            vector_retriever=mock_vector_retriever,
            max_iterations=3,
            sufficiency_threshold=0.7,
        )

        assert retriever.llm_client == mock_llm_client
        assert retriever.graph_engine == mock_graph_engine
        assert retriever.vector_retriever == mock_vector_retriever
        assert retriever.max_iterations == 3
        assert retriever.sufficiency_threshold == 0.7

    @pytest.mark.asyncio
    async def test_refine_context_sufficient_immediately(
        self, iterative_retriever, sample_retrieval_context
    ):
        """Test context refinement when context is sufficient immediately."""
        # Mock analysis to return sufficient context
        sufficient_analysis = ContextAnalysis(
            is_sufficient=True,
            confidence=0.9,
            gaps=[],
            reasoning="Context is sufficient",
        )

        iterative_retriever._analyze_context = AsyncMock(
            return_value=sufficient_analysis
        )

        result = await iterative_retriever.refine_context(
            "What is Apple?", sample_retrieval_context
        )

        assert result.metadata["final_analysis"] == sufficient_analysis
        assert result.metadata["iterations_used"] == 1
        # Should have called analysis twice (once during iteration, once for final analysis)
        assert iterative_retriever._analyze_context.call_count == 2

    @pytest.mark.asyncio
    async def test_refine_context_multiple_iterations(
        self, iterative_retriever, sample_retrieval_context
    ):
        """Test context refinement with multiple iterations."""
        # Mock analysis to return insufficient context first, then sufficient
        insufficient_analysis = ContextAnalysis(
            is_sufficient=False,
            confidence=0.5,
            gaps=[
                ContextGap(
                    gap_type="missing_entity",
                    description="Need more info",
                    entities_needed=["entity1"],
                )
            ],
            reasoning="Need more information",
        )

        sufficient_analysis = ContextAnalysis(
            is_sufficient=True, confidence=0.9, gaps=[], reasoning="Now sufficient"
        )

        iterative_retriever._analyze_context = AsyncMock(
            side_effect=[
                insufficient_analysis,
                sufficient_analysis,
                sufficient_analysis,
            ]  # Add extra for final analysis
        )
        iterative_retriever._retrieve_additional = AsyncMock(
            return_value=RetrievalContext(entities={"entity1": {"type": "TEST"}})
        )

        result = await iterative_retriever.refine_context(
            "What is Apple?", sample_retrieval_context
        )

        assert result.metadata["iterations_used"] == 2
        assert "entity1" in result.entities
        iterative_retriever._retrieve_additional.assert_called_once()

    @pytest.mark.asyncio
    async def test_refine_context_max_iterations(
        self, iterative_retriever, sample_retrieval_context
    ):
        """Test context refinement hitting max iterations."""
        # Mock analysis to always return insufficient context
        insufficient_analysis = ContextAnalysis(
            is_sufficient=False, confidence=0.5, gaps=[], reasoning="Never sufficient"
        )

        iterative_retriever._analyze_context = AsyncMock(
            return_value=insufficient_analysis
        )
        iterative_retriever._retrieve_additional = AsyncMock(
            return_value=RetrievalContext()
        )

        result = await iterative_retriever.refine_context(
            "What is Apple?", sample_retrieval_context
        )

        assert result.metadata["iterations_used"] == iterative_retriever.max_iterations
        assert (
            iterative_retriever._analyze_context.call_count
            == iterative_retriever.max_iterations + 1
        )  # +1 for final analysis

    @pytest.mark.asyncio
    async def test_analyze_context_success(
        self, iterative_retriever, sample_retrieval_context
    ):
        """Test successful context analysis."""
        query = "What is Apple?"

        result = await iterative_retriever._analyze_context(
            query, sample_retrieval_context
        )

        assert isinstance(result, ContextAnalysis)
        assert isinstance(result.is_sufficient, bool)
        assert 0 <= result.confidence <= 1
        assert isinstance(result.gaps, list)
        assert isinstance(result.reasoning, str)

    @pytest.mark.asyncio
    async def test_analyze_context_llm_error(
        self, iterative_retriever, sample_retrieval_context
    ):
        """Test context analysis with LLM error (fallback)."""
        # Mock LLM to raise an error
        iterative_retriever.llm_client.generate = AsyncMock(
            side_effect=Exception("LLM error")
        )

        result = await iterative_retriever._analyze_context(
            "test query", sample_retrieval_context
        )

        assert isinstance(result, ContextAnalysis)
        assert result.confidence == 0.5  # Fallback confidence
        assert "Fallback analysis" in result.reasoning

    def test_create_analysis_prompt(
        self, iterative_retriever, sample_retrieval_context
    ):
        """Test creation of context analysis prompt."""
        query = "What is Apple?"
        prompt = iterative_retriever._create_analysis_prompt(
            query, sample_retrieval_context
        )

        assert query in prompt
        assert "Apple" in prompt  # Entity should be mentioned
        assert "FOUNDED_BY" in prompt  # Relation should be mentioned
        assert (
            "Apple Inc. is a technology company" in prompt
        )  # Document content should be mentioned
        assert "JSON" in prompt
        assert "is_sufficient" in prompt
        assert "confidence" in prompt
        assert "gaps" in prompt

    def test_parse_context_analysis_success(self, iterative_retriever):
        """Test successful parsing of context analysis."""
        response = """
        {
          "is_sufficient": false,
          "confidence": 6.5,
          "reasoning": "Context provides basic information but lacks details.",
          "gaps": [
            {
              "gap_type": "missing_entity",
              "description": "Need more information about entity X",
              "entities_needed": ["entity_x"],
              "priority": 0.8
            }
          ],
          "suggested_queries": ["What is entity X?"]
        }
        """

        result = iterative_retriever._parse_context_analysis(response)

        assert isinstance(result, ContextAnalysis)
        assert not result.is_sufficient
        assert result.confidence == 0.65  # Normalized from 6.5/10
        assert len(result.gaps) == 1
        assert result.gaps[0].gap_type == "missing_entity"
        assert result.gaps[0].entities_needed == ["entity_x"]
        assert result.suggested_queries == ["What is entity X?"]

    def test_parse_context_analysis_error(self, iterative_retriever):
        """Test parsing context analysis with invalid JSON."""
        response = "Invalid JSON response"

        result = iterative_retriever._parse_context_analysis(response)

        assert isinstance(result, ContextAnalysis)
        assert not result.is_sufficient
        assert result.confidence == 0.3  # Error fallback confidence
        assert "Failed to parse" in result.reasoning

    @pytest.mark.asyncio
    async def test_retrieve_additional_missing_entity(self, iterative_retriever):
        """Test retrieving additional information for missing entity."""
        gaps = [
            ContextGap(
                gap_type="missing_entity",
                description="Need entity info",
                entities_needed=["test_entity"],
                priority=0.9,
            )
        ]

        # Mock graph engine to return entity details
        iterative_retriever.graph_engine.get_entity_details = AsyncMock(
            return_value={"type": "ORG", "name": "Test Entity"}
        )

        result = await iterative_retriever._retrieve_additional(
            "test query", gaps, RetrievalContext()
        )

        assert "test_entity" in result.entities
        assert result.entities["test_entity"]["type"] == "ORG"
        iterative_retriever.graph_engine.get_entity_details.assert_called_with(
            "test_entity"
        )

    @pytest.mark.asyncio
    async def test_retrieve_additional_insufficient_detail(self, iterative_retriever):
        """Test retrieving additional information for insufficient detail."""
        gaps = [
            ContextGap(
                gap_type="insufficient_detail",
                description="Need more details about topic",
                priority=0.8,
            )
        ]

        result = await iterative_retriever._retrieve_additional(
            "test query", gaps, RetrievalContext()
        )

        # Should have called vector retriever
        iterative_retriever.vector_retriever.search.assert_called()
        assert len(result.documents) > 0

    def test_merge_context(self, iterative_retriever):
        """Test merging two retrieval contexts."""
        current_context = RetrievalContext(
            entities={"entity1": {"type": "ORG"}},
            relations=[{"subject": "A", "predicate": "REL1", "object": "B"}],
            documents=[{"id": "doc1", "content": "Document 1"}],
        )

        additional_context = RetrievalContext(
            entities={"entity2": {"type": "PERSON"}},
            relations=[
                {"subject": "A", "predicate": "REL1", "object": "B"},  # Duplicate
                {"subject": "C", "predicate": "REL2", "object": "D"},  # New
            ],
            documents=[
                {"id": "doc1", "content": "Document 1"},  # Duplicate
                {"id": "doc2", "content": "Document 2"},  # New
            ],
        )

        result = iterative_retriever._merge_context(current_context, additional_context)

        # Should have both entities
        assert len(result.entities) == 2
        assert "entity1" in result.entities
        assert "entity2" in result.entities

        # Should deduplicate relations
        assert len(result.relations) == 2

        # Should deduplicate documents
        assert len(result.documents) == 2
