"""Tests for hybrid retrieval system."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from morag_graph.retrieval import (
    HybridRetrievalCoordinator, ContextExpansionEngine, 
    RetrievalResult, HybridRetrievalConfig, ExpandedContext
)
from morag_graph.retrieval.models import VectorRetriever, DocumentResult
from morag_graph.query import QueryEntityExtractor, QueryAnalysis, QueryEntity
from morag_graph.models import Entity
from morag_graph.operations.traversal import GraphPath
from morag_graph.storage.base import BaseStorage


@pytest.fixture
def mock_vector_retriever():
    """Mock vector retriever."""
    retriever = AsyncMock(spec=VectorRetriever)
    retriever.search.return_value = [
        {
            'content': 'Vector search result 1',
            'score': 0.9,
            'metadata': {'source': 'doc1'}
        },
        {
            'content': 'Vector search result 2', 
            'score': 0.8,
            'metadata': {'source': 'doc2'}
        }
    ]
    return retriever


@pytest.fixture
def mock_context_expansion_engine():
    """Mock context expansion engine."""
    engine = AsyncMock(spec=ContextExpansionEngine)

    # Mock expanded context
    mock_entity = Entity(
        id="ent_test_entity_person",
        name="Test Entity",
        type="PERSON",
        confidence=0.9
    )
    # Add source_doc_id attribute for document retrieval
    mock_entity.source_doc_id = "doc_123"

    engine.expand_context.return_value = ExpandedContext(
        original_entities=["ent_test_entity_person"],
        expanded_entities=[mock_entity],
        expansion_paths=[GraphPath(entities=[mock_entity], relations=[])],
        context_score=0.8,
        expansion_reasoning="Direct neighbors expansion"
    )

    return engine


@pytest.fixture
def mock_query_entity_extractor():
    """Mock query entity extractor."""
    extractor = AsyncMock(spec=QueryEntityExtractor)
    
    # Mock query analysis
    query_entity = QueryEntity(
        text="Test Entity",
        entity_type="PERSON",
        confidence=0.9,
        linked_entity_id="ent_test_entity_person"
    )
    
    extractor.extract_and_link_entities.return_value = QueryAnalysis(
        original_query="test query",
        entities=[query_entity],
        intent="factual",
        query_type="single_entity",
        complexity_score=0.5
    )
    
    return extractor


@pytest.fixture
def hybrid_coordinator(mock_vector_retriever, mock_context_expansion_engine, mock_query_entity_extractor):
    """Hybrid retrieval coordinator instance."""
    return HybridRetrievalCoordinator(
        vector_retriever=mock_vector_retriever,
        context_expansion_engine=mock_context_expansion_engine,
        query_entity_extractor=mock_query_entity_extractor
    )


class TestHybridRetrievalCoordinator:
    """Test cases for HybridRetrievalCoordinator."""
    
    @pytest.mark.asyncio
    async def test_basic_retrieval(self, hybrid_coordinator):
        """Test basic hybrid retrieval functionality."""
        results = await hybrid_coordinator.retrieve("test query", max_results=5)
        
        assert isinstance(results, list)
        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.score > 0 for r in results)
        
        # Results should be sorted by score
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
    
    @pytest.mark.asyncio
    async def test_vector_only_fallback(self, mock_vector_retriever, mock_query_entity_extractor):
        """Test fallback to vector-only retrieval when graph fails."""
        # Create coordinator with failing context expansion
        failing_expansion = AsyncMock(spec=ContextExpansionEngine)
        failing_expansion.expand_context.side_effect = Exception("Graph expansion failed")
        
        coordinator = HybridRetrievalCoordinator(
            vector_retriever=mock_vector_retriever,
            context_expansion_engine=failing_expansion,
            query_entity_extractor=mock_query_entity_extractor
        )
        
        results = await coordinator.retrieve("test query")
        
        # Should still return results from vector retrieval
        assert len(results) > 0
        assert all(r.source in ["vector", "hybrid_vector"] for r in results)
    
    @pytest.mark.asyncio
    async def test_weighted_combination_fusion(self, hybrid_coordinator):
        """Test weighted combination fusion strategy."""
        config = HybridRetrievalConfig(
            fusion_strategy="weighted_combination",
            vector_weight=0.7,
            graph_weight=0.3
        )
        
        hybrid_coordinator.config = config
        results = await hybrid_coordinator.retrieve("test query")
        
        assert len(results) > 0
        # Should have results from both sources or hybrid sources
        sources = {r.source for r in results}
        assert any(source.startswith("hybrid") for source in sources)
    
    @pytest.mark.asyncio
    async def test_rank_fusion_strategy(self, hybrid_coordinator):
        """Test reciprocal rank fusion strategy."""
        config = HybridRetrievalConfig(
            fusion_strategy="rank_fusion",
            min_confidence_threshold=0.0  # Lower threshold to ensure results pass
        )
        hybrid_coordinator.config = config

        results = await hybrid_coordinator.retrieve("test query")

        assert len(results) > 0
        # RRF should produce results with rrf_fusion source
        assert any(r.source == "rrf_fusion" for r in results)
    
    @pytest.mark.asyncio
    async def test_adaptive_fusion_strategy(self, hybrid_coordinator):
        """Test adaptive fusion strategy."""
        config = HybridRetrievalConfig(
            fusion_strategy="adaptive",
            min_confidence_threshold=0.0  # Lower threshold to ensure results pass
        )
        hybrid_coordinator.config = config

        results = await hybrid_coordinator.retrieve("test query")

        assert len(results) > 0
        # Should adapt based on query complexity
    
    @pytest.mark.asyncio
    async def test_empty_query_entities(self, mock_vector_retriever, mock_context_expansion_engine):
        """Test handling of queries with no entities."""
        # Mock extractor that returns no entities
        empty_extractor = AsyncMock(spec=QueryEntityExtractor)
        empty_extractor.extract_and_link_entities.return_value = QueryAnalysis(
            original_query="simple query",
            entities=[],
            intent="general",
            query_type="general",
            complexity_score=0.2
        )
        
        coordinator = HybridRetrievalCoordinator(
            vector_retriever=mock_vector_retriever,
            context_expansion_engine=mock_context_expansion_engine,
            query_entity_extractor=empty_extractor
        )
        
        results = await coordinator.retrieve("simple query")
        
        # Should still return vector results
        assert len(results) > 0
        assert all(r.source in ["vector", "hybrid_vector"] for r in results)
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_filtering(self, hybrid_coordinator):
        """Test filtering by minimum confidence threshold."""
        config = HybridRetrievalConfig(min_confidence_threshold=0.9)
        hybrid_coordinator.config = config
        
        results = await hybrid_coordinator.retrieve("test query")
        
        # All results should meet the confidence threshold
        assert all(r.score >= 0.9 for r in results)
    
    @pytest.mark.asyncio
    async def test_max_results_limit(self, hybrid_coordinator):
        """Test maximum results limit."""
        results = await hybrid_coordinator.retrieve("test query", max_results=2)
        
        assert len(results) <= 2
    
    def test_graph_relevance_score_calculation(self, hybrid_coordinator):
        """Test graph relevance score calculation."""
        entity = Entity(
            id="ent_test_entity_person",
            name="Test Entity",
            type="PERSON",
            confidence=0.9
        )
        
        context = ExpandedContext(
            original_entities=["ent_test_entity_person"],
            expanded_entities=[entity],
            expansion_paths=[],
            context_score=0.8,
            expansion_reasoning="test"
        )
        
        score = hybrid_coordinator._calculate_graph_relevance_score(entity, context)
        
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be boosted for original entity


class TestContextExpansionEngine:
    """Test cases for ContextExpansionEngine."""
    
    @pytest.fixture
    def mock_graph_storage(self):
        """Mock graph storage."""
        storage = AsyncMock(spec=BaseStorage)
        return storage
    
    @pytest.fixture
    def context_engine(self, mock_graph_storage):
        """Context expansion engine instance."""
        return ContextExpansionEngine(mock_graph_storage)
    
    @pytest.mark.asyncio
    async def test_expand_context_with_entities(self, context_engine):
        """Test context expansion with linked entities."""
        query_entity = QueryEntity(
            text="Test Entity",
            entity_type="PERSON",
            confidence=0.9,
            linked_entity_id="ent_test_entity_person"
        )
        
        query_analysis = QueryAnalysis(
            original_query="test query",
            entities=[query_entity],
            intent="factual",
            query_type="single_entity",
            complexity_score=0.5
        )
        
        # Mock the graph traversal to return some neighbors
        context_engine.graph_traversal.find_neighbors = AsyncMock(return_value=[
            Entity(id="ent_neighbor_1", name="Neighbor 1", type="PERSON"),
            Entity(id="ent_neighbor_2", name="Neighbor 2", type="CONCEPT")
        ])
        
        context = await context_engine.expand_context(query_analysis)
        
        assert isinstance(context, ExpandedContext)
        assert len(context.original_entities) > 0
        assert len(context.expanded_entities) > 0
        assert context.context_score > 0
    
    @pytest.mark.asyncio
    async def test_expand_context_no_entities(self, context_engine):
        """Test context expansion with no linked entities."""
        query_analysis = QueryAnalysis(
            original_query="simple query",
            entities=[],
            intent="general",
            query_type="general",
            complexity_score=0.2
        )
        
        context = await context_engine.expand_context(query_analysis)
        
        assert isinstance(context, ExpandedContext)
        assert len(context.original_entities) == 0
        assert len(context.expanded_entities) == 0
        assert context.context_score == 0.0
    
    def test_strategy_selection(self, context_engine):
        """Test expansion strategy selection."""
        # Single entity factual query
        analysis1 = QueryAnalysis(
            original_query="Who is Einstein?",
            entities=[QueryEntity("Einstein", "PERSON", 0.9)],
            intent="factual",
            query_type="single_entity",
            complexity_score=0.3
        )
        
        strategy1 = context_engine._select_expansion_strategy(analysis1)
        assert strategy1 == "direct_neighbors"
        
        # Entity relationship query
        analysis2 = QueryAnalysis(
            original_query="Einstein and physics",
            entities=[
                QueryEntity("Einstein", "PERSON", 0.9),
                QueryEntity("Physics", "CONCEPT", 0.8)
            ],
            intent="factual",
            query_type="entity_relationship",
            complexity_score=0.6
        )
        
        strategy2 = context_engine._select_expansion_strategy(analysis2)
        assert strategy2 == "shortest_path"
    
    def test_context_score_calculation(self, context_engine):
        """Test context score calculation."""
        entities = [
            Entity(id="ent_1", name="Entity 1", type="PERSON"),
            Entity(id="ent_2", name="Entity 2", type="CONCEPT")
        ]
        
        paths = [GraphPath(entities=entities, relations=[])]
        
        query_analysis = QueryAnalysis(
            original_query="test",
            entities=[],
            intent="factual",
            query_type="general",
            complexity_score=0.5
        )
        
        score = context_engine._calculate_context_score(query_analysis, entities, paths)
        
        assert 0.0 <= score <= 1.0
    
    def test_deduplicate_entities(self, context_engine):
        """Test entity deduplication."""
        entity1 = Entity(id="ent_1", name="Entity 1", type="PERSON")
        entity2 = Entity(id="ent_2", name="Entity 2", type="CONCEPT")
        entity1_dup = Entity(id="ent_1", name="Entity 1 Duplicate", type="PERSON")
        
        entities = [entity1, entity2, entity1_dup, entity2]
        unique_entities = context_engine._deduplicate_entities(entities)
        
        assert len(unique_entities) == 2
        assert unique_entities[0].id == "ent_1"
        assert unique_entities[1].id == "ent_2"
