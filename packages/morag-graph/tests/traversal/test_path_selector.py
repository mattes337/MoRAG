"""Tests for LLM-guided path selection."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from morag_graph.traversal.path_selector import (
    LLMPathSelector,
    PathRelevanceScore,
    TraversalStrategy,
    QueryContext
)
from morag_graph.models import Entity, Relation
from morag_graph.operations.traversal import GraphPath


class TestLLMPathSelector:
    """Test the LLM path selector."""

    @pytest.fixture
    def sample_entities(self):
        """Sample entities for testing."""
        return [
            Entity(
                id="ent_einstein_001",
                name="Einstein",
                canonical_name="einstein",
                type="PERSON",
                description="Famous physicist"
            ),
            Entity(
                id="ent_relativity_001",
                name="Theory of Relativity",
                canonical_name="theory of relativity",
                type="CONCEPT",
                description="Physics theory"
            ),
            Entity(
                id="ent_princeton_001",
                name="Princeton University",
                canonical_name="princeton university",
                type="ORG",
                description="Educational institution"
            )
        ]

    @pytest.fixture
    def sample_relations(self):
        """Sample relations for testing."""
        return [
            Relation(
                id="rel_001",
                type="DEVELOPED",
                source_entity_id="ent_einstein_001",
                target_entity_id="ent_relativity_001",
                confidence=0.9
            ),
            Relation(
                id="rel_002",
                type="WORKED_AT",
                source_entity_id="ent_einstein_001",
                target_entity_id="ent_princeton_001",
                confidence=0.8
            )
        ]

    @pytest.fixture
    def sample_paths(self, sample_entities, sample_relations):
        """Sample graph paths for testing."""
        return [
            GraphPath(
                entities=[sample_entities[0], sample_entities[1]],
                relations=[sample_relations[0]]
            ),
            GraphPath(
                entities=[sample_entities[0], sample_entities[2]],
                relations=[sample_relations[1]]
            ),
            GraphPath(
                entities=[sample_entities[0], sample_entities[1], sample_entities[2]],
                relations=sample_relations
            )
        ]

    @pytest.fixture
    def query_context(self):
        """Sample query context."""
        return QueryContext(
            query="What did Einstein develop?",
            intent="factual",
            entities=["Einstein"],
            keywords=["Einstein", "develop"],
            complexity_score=0.3
        )

    @pytest.mark.asyncio
    async def test_path_selector_initialization(self):
        """Test path selector initialization."""
        config = {
            'llm_enabled': False,  # Disable LLM for testing
            'max_paths_to_evaluate': 10,
            'min_relevance_threshold': 0.2
        }

        selector = LLMPathSelector(config)

        assert selector.max_paths_to_evaluate == 10
        assert selector.min_relevance_threshold == 0.2
        assert not selector.llm_enabled

    @pytest.mark.asyncio
    async def test_heuristic_path_scoring(self, sample_entities, sample_paths, query_context):
        """Test heuristic path scoring when LLM is disabled."""
        config = {'llm_enabled': False}
        selector = LLMPathSelector(config)

        scored_paths = await selector.select_paths(
            query="What did Einstein develop?",
            starting_entities=[sample_entities[0]],
            available_paths=sample_paths,
            query_context=query_context,
            strategy=TraversalStrategy.RELEVANCE_GUIDED
        )

        assert len(scored_paths) > 0
        assert all(isinstance(p, PathRelevanceScore) for p in scored_paths)
        assert all(0.0 <= p.relevance_score <= 1.0 for p in scored_paths)
        assert all(p.confidence > 0 for p in scored_paths)

    @pytest.mark.asyncio
    async def test_path_filtering(self, sample_entities, sample_relations):
        """Test basic path filtering."""
        config = {'llm_enabled': False, 'max_path_length': 2}
        selector = LLMPathSelector(config)

        # Create paths with different lengths
        short_path = GraphPath(
            entities=[sample_entities[0], sample_entities[1]],
            relations=[sample_relations[0]]
        )

        long_path = GraphPath(
            entities=sample_entities * 3,  # Very long path
            relations=sample_relations * 3
        )

        # Create circular path (same start and end)
        circular_path = GraphPath(
            entities=[sample_entities[0], sample_entities[1], sample_entities[0]],
            relations=sample_relations
        )

        paths = [short_path, long_path, circular_path]
        filtered = selector._filter_paths_basic(paths)

        # Should keep only the short path
        assert len(filtered) == 1
        assert filtered[0] == short_path

    @pytest.mark.asyncio
    async def test_empty_paths_handling(self, sample_entities):
        """Test handling of empty path lists."""
        config = {'llm_enabled': False}
        selector = LLMPathSelector(config)

        scored_paths = await selector.select_paths(
            query="Test query",
            starting_entities=sample_entities,
            available_paths=[],
            strategy=TraversalStrategy.BREADTH_FIRST
        )

        assert scored_paths == []

    @pytest.mark.asyncio
    async def test_relevance_threshold_filtering(self, sample_entities, sample_paths, query_context):
        """Test filtering by relevance threshold."""
        config = {
            'llm_enabled': False,
            'min_relevance_threshold': 0.8  # High threshold
        }
        selector = LLMPathSelector(config)

        scored_paths = await selector.select_paths(
            query="Completely unrelated query about cooking",
            starting_entities=[sample_entities[0]],
            available_paths=sample_paths,
            query_context=query_context
        )

        # Should filter out low-relevance paths
        assert all(p.relevance_score >= 0.8 for p in scored_paths)

    @pytest.mark.asyncio
    async def test_path_description(self, sample_entities, sample_relations):
        """Test path description generation."""
        config = {'llm_enabled': False}
        selector = LLMPathSelector(config)

        path = GraphPath(
            entities=[sample_entities[0], sample_entities[1]],
            relations=[sample_relations[0]]
        )

        description = selector._describe_path(path)

        assert "Einstein" in description
        assert "Theory of Relativity" in description
        assert "DEVELOPED" in description
        assert "PERSON" in description
        assert "CONCEPT" in description

    @pytest.mark.asyncio
    async def test_different_strategies(self, sample_entities, sample_paths, query_context):
        """Test different traversal strategies."""
        config = {'llm_enabled': False}
        selector = LLMPathSelector(config)

        strategies = [
            TraversalStrategy.BREADTH_FIRST,
            TraversalStrategy.DEPTH_FIRST,
            TraversalStrategy.RELEVANCE_GUIDED,
            TraversalStrategy.ADAPTIVE
        ]

        for strategy in strategies:
            scored_paths = await selector.select_paths(
                query="What did Einstein develop?",
                starting_entities=[sample_entities[0]],
                available_paths=sample_paths,
                query_context=query_context,
                strategy=strategy
            )

            # Should work with all strategies
            assert isinstance(scored_paths, list)

    @pytest.mark.asyncio
    async def test_error_handling(self, sample_entities):
        """Test error handling in path selection."""
        config = {'llm_enabled': False}
        selector = LLMPathSelector(config)

        # Test with invalid paths
        invalid_path = GraphPath(entities=[], relations=[])

        scored_paths = await selector.select_paths(
            query="Test query",
            starting_entities=sample_entities,
            available_paths=[invalid_path],
            strategy=TraversalStrategy.BREADTH_FIRST
        )

        # Should handle gracefully
        assert isinstance(scored_paths, list)
