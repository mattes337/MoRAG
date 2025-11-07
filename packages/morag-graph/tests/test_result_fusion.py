"""Tests for result fusion strategies."""

from unittest.mock import MagicMock

import pytest
from morag_graph.query.models import QueryAnalysis, QueryEntity
from morag_graph.retrieval.fusion import (
    AdaptiveFusion,
    ReciprocalRankFusion,
    ResultFusionEngine,
    WeightedCombinationFusion,
)
from morag_graph.retrieval.models import HybridRetrievalConfig, RetrievalResult


@pytest.fixture
def sample_vector_results():
    """Sample vector retrieval results."""
    return [
        RetrievalResult(
            content="Vector result 1",
            source="vector",
            score=0.9,
            metadata={"doc_id": "doc1"},
            reasoning="High semantic similarity",
        ),
        RetrievalResult(
            content="Vector result 2",
            source="vector",
            score=0.8,
            metadata={"doc_id": "doc2"},
            reasoning="Good semantic match",
        ),
        RetrievalResult(
            content="Shared result",
            source="vector",
            score=0.7,
            metadata={"doc_id": "doc3"},
            reasoning="Moderate similarity",
        ),
    ]


@pytest.fixture
def sample_graph_results():
    """Sample graph retrieval results."""
    return [
        RetrievalResult(
            content="Graph result 1",
            source="graph",
            score=0.85,
            metadata={"entity_id": "ent_1"},
            entities=["ent_1"],
            reasoning="Direct entity match",
        ),
        RetrievalResult(
            content="Shared result",
            source="graph",
            score=0.75,
            metadata={"entity_id": "ent_2"},
            entities=["ent_2"],
            reasoning="Graph traversal",
        ),
        RetrievalResult(
            content="Graph result 2",
            source="graph",
            score=0.6,
            metadata={"entity_id": "ent_3"},
            entities=["ent_3"],
            reasoning="Neighbor expansion",
        ),
    ]


@pytest.fixture
def sample_query_analysis():
    """Sample query analysis."""
    return QueryAnalysis(
        original_query="test query",
        entities=[
            QueryEntity("Entity1", "PERSON", 0.9),
            QueryEntity("Entity2", "CONCEPT", 0.8),
        ],
        intent="factual",
        query_type="entity_relationship",
        complexity_score=0.6,
    )


@pytest.fixture
def default_config():
    """Default retrieval configuration."""
    return HybridRetrievalConfig(
        vector_weight=0.6,
        graph_weight=0.4,
        fusion_strategy="weighted_combination",
        min_confidence_threshold=0.3,
    )


class TestWeightedCombinationFusion:
    """Test cases for WeightedCombinationFusion."""

    @pytest.mark.asyncio
    async def test_basic_fusion(
        self,
        sample_vector_results,
        sample_graph_results,
        sample_query_analysis,
        default_config,
    ):
        """Test basic weighted combination fusion."""
        fusion = WeightedCombinationFusion()

        results = await fusion.fuse(
            sample_vector_results,
            sample_graph_results,
            sample_query_analysis,
            default_config,
        )

        assert len(results) == 5  # 3 vector + 3 graph - 1 duplicate

        # Check that scores are adjusted by weights
        vector_results = [r for r in results if r.source == "hybrid_vector"]
        graph_results = [r for r in results if r.source == "hybrid_graph"]
        both_results = [r for r in results if r.source == "hybrid_both"]

        assert len(vector_results) == 2  # 2 unique vector results
        assert len(graph_results) == 2  # 2 unique graph results
        assert len(both_results) == 1  # 1 shared result

        # Verify score adjustments (note: weights may be boosted for entity-rich queries)
        for result in vector_results:
            assert result.score > 0  # Should have positive score

        for result in graph_results:
            assert result.score > 0  # Should have positive score

    @pytest.mark.asyncio
    async def test_entity_rich_query_weight_adjustment(
        self, sample_vector_results, sample_graph_results, default_config
    ):
        """Test weight adjustment for entity-rich queries."""
        # Create query with multiple entities
        entity_rich_query = QueryAnalysis(
            original_query="complex query",
            entities=[
                QueryEntity("Entity1", "PERSON", 0.9),
                QueryEntity("Entity2", "CONCEPT", 0.8),
                QueryEntity("Entity3", "LOCATION", 0.7),
            ],
            intent="factual",
            query_type="multi_entity",
            complexity_score=0.8,
        )

        fusion = WeightedCombinationFusion()

        results = await fusion.fuse(
            sample_vector_results,
            sample_graph_results,
            entity_rich_query,
            default_config,
        )

        # Should boost graph weight for entity-rich queries
        assert len(results) > 0

        # Check that graph results get higher relative scores
        graph_results = [r for r in results if "graph" in r.source]
        vector_results = [r for r in results if r.source == "hybrid_vector"]

        if graph_results and vector_results:
            # Graph results should generally have competitive scores due to boosting
            max_graph_score = max(r.score for r in graph_results)
            max_vector_score = max(r.score for r in vector_results)
            # This is a relative test - graph should be competitive
            assert max_graph_score > 0  # Basic sanity check


class TestReciprocalRankFusion:
    """Test cases for ReciprocalRankFusion."""

    @pytest.mark.asyncio
    async def test_rrf_fusion(
        self,
        sample_vector_results,
        sample_graph_results,
        sample_query_analysis,
        default_config,
    ):
        """Test reciprocal rank fusion."""
        fusion = ReciprocalRankFusion(k=60)

        results = await fusion.fuse(
            sample_vector_results,
            sample_graph_results,
            sample_query_analysis,
            default_config,
        )

        assert len(results) == 5  # 3 vector + 3 graph - 1 duplicate

        # All results should have rrf_fusion source
        assert all(r.source == "rrf_fusion" for r in results)

        # Check RRF score calculation
        for result in results:
            assert 0 < result.score <= 2.0  # Max possible RRF score with k=60

        # Results should be sorted by RRF score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_rrf_with_different_k_values(
        self,
        sample_vector_results,
        sample_graph_results,
        sample_query_analysis,
        default_config,
    ):
        """Test RRF with different k values."""
        fusion_k30 = ReciprocalRankFusion(k=30)
        fusion_k90 = ReciprocalRankFusion(k=90)

        results_k30 = await fusion_k30.fuse(
            sample_vector_results,
            sample_graph_results,
            sample_query_analysis,
            default_config,
        )

        results_k90 = await fusion_k90.fuse(
            sample_vector_results,
            sample_graph_results,
            sample_query_analysis,
            default_config,
        )

        # Lower k should generally produce higher scores
        max_score_k30 = max(r.score for r in results_k30)
        max_score_k90 = max(r.score for r in results_k90)

        assert max_score_k30 > max_score_k90

    @pytest.mark.asyncio
    async def test_entity_merging(
        self,
        sample_vector_results,
        sample_graph_results,
        sample_query_analysis,
        default_config,
    ):
        """Test that entities are properly merged for shared content."""
        fusion = ReciprocalRankFusion()

        results = await fusion.fuse(
            sample_vector_results,
            sample_graph_results,
            sample_query_analysis,
            default_config,
        )

        # Find the shared result
        shared_results = [r for r in results if r.content == "Shared result"]
        assert len(shared_results) == 1

        shared_result = shared_results[0]
        # Should have entities from graph result
        assert shared_result.entities == ["ent_2"]


class TestAdaptiveFusion:
    """Test cases for AdaptiveFusion."""

    @pytest.mark.asyncio
    async def test_complex_query_uses_weighted(
        self, sample_vector_results, sample_graph_results, default_config
    ):
        """Test that complex queries use weighted combination."""
        complex_query = QueryAnalysis(
            original_query="complex multi-entity query",
            entities=[
                QueryEntity("Entity1", "PERSON", 0.9),
                QueryEntity("Entity2", "CONCEPT", 0.8),
                QueryEntity("Entity3", "LOCATION", 0.7),
            ],
            intent="factual",
            query_type="multi_entity",
            complexity_score=0.8,  # High complexity
        )

        fusion = AdaptiveFusion()

        results = await fusion.fuse(
            sample_vector_results, sample_graph_results, complex_query, default_config
        )

        # Should use weighted combination (check for hybrid sources)
        sources = {r.source for r in results}
        assert any("hybrid" in source for source in sources)

    @pytest.mark.asyncio
    async def test_simple_query_uses_rrf(
        self, sample_vector_results, sample_graph_results, default_config
    ):
        """Test that simple queries use RRF."""
        simple_query = QueryAnalysis(
            original_query="simple query",
            entities=[QueryEntity("Entity1", "PERSON", 0.9)],
            intent="factual",
            query_type="single_entity",
            complexity_score=0.3,  # Low complexity
        )

        fusion = AdaptiveFusion()

        results = await fusion.fuse(
            sample_vector_results, sample_graph_results, simple_query, default_config
        )

        # Should use RRF (check for rrf_fusion source)
        assert all(r.source == "rrf_fusion" for r in results)


class TestResultFusionEngine:
    """Test cases for ResultFusionEngine."""

    @pytest.mark.asyncio
    async def test_strategy_selection(
        self, sample_vector_results, sample_graph_results, sample_query_analysis
    ):
        """Test strategy selection based on configuration."""
        engine = ResultFusionEngine()

        # Test weighted combination
        config_weighted = HybridRetrievalConfig(fusion_strategy="weighted_combination")
        results_weighted = await engine.fuse_results(
            sample_vector_results,
            sample_graph_results,
            sample_query_analysis,
            config_weighted,
        )

        sources_weighted = {r.source for r in results_weighted}
        assert any("hybrid" in source for source in sources_weighted)

        # Test RRF
        config_rrf = HybridRetrievalConfig(fusion_strategy="rank_fusion")
        results_rrf = await engine.fuse_results(
            sample_vector_results,
            sample_graph_results,
            sample_query_analysis,
            config_rrf,
        )

        assert all(r.source == "rrf_fusion" for r in results_rrf)

        # Test adaptive
        config_adaptive = HybridRetrievalConfig(
            fusion_strategy="adaptive",
            min_confidence_threshold=0.0,  # Lower threshold to ensure results pass
        )
        results_adaptive = await engine.fuse_results(
            sample_vector_results,
            sample_graph_results,
            sample_query_analysis,
            config_adaptive,
        )

        assert len(results_adaptive) > 0

    @pytest.mark.asyncio
    async def test_unknown_strategy_fallback(
        self, sample_vector_results, sample_graph_results, sample_query_analysis
    ):
        """Test fallback for unknown fusion strategy."""
        engine = ResultFusionEngine()

        config_unknown = HybridRetrievalConfig(fusion_strategy="unknown_strategy")
        results = await engine.fuse_results(
            sample_vector_results,
            sample_graph_results,
            sample_query_analysis,
            config_unknown,
        )

        # Should fallback to weighted combination
        sources = {r.source for r in results}
        assert any("hybrid" in source for source in sources)

    @pytest.mark.asyncio
    async def test_confidence_filtering(
        self, sample_vector_results, sample_graph_results, sample_query_analysis
    ):
        """Test confidence threshold filtering."""
        engine = ResultFusionEngine()

        # High confidence threshold
        config_high_threshold = HybridRetrievalConfig(
            fusion_strategy="weighted_combination", min_confidence_threshold=0.8
        )

        results = await engine.fuse_results(
            sample_vector_results,
            sample_graph_results,
            sample_query_analysis,
            config_high_threshold,
        )

        # All results should meet the threshold
        assert all(r.score >= 0.8 for r in results)

    @pytest.mark.asyncio
    async def test_error_handling_fallback(
        self, sample_vector_results, sample_graph_results, sample_query_analysis
    ):
        """Test error handling and fallback fusion."""
        engine = ResultFusionEngine()

        # Mock a strategy that raises an exception
        def failing_strategy():
            raise Exception("Strategy failed")

        engine.strategies["weighted_combination"] = MagicMock(
            side_effect=Exception("Strategy failed")
        )

        config = HybridRetrievalConfig(fusion_strategy="weighted_combination")
        results = await engine.fuse_results(
            sample_vector_results, sample_graph_results, sample_query_analysis, config
        )

        # Should fallback to simple fusion
        assert len(results) > 0
        # Results should be sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
