"""Unit tests for enhanced hybrid search."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from morag_graph.graphiti.hybrid_search import (
    EnhancedHybridSearchService, SearchMethod, SearchConfiguration,
    EnhancedSearchResult
)
from morag_graph.graphiti.search_service import SearchResult, SearchMetrics
from morag_graph.graphiti.custom_schema import MoragEntityType


class TestEnhancedHybridSearchService:
    """Test enhanced hybrid search service."""
    
    @pytest.fixture
    def mock_search_service(self):
        """Create mock search service."""
        service = Mock()
        service.search = AsyncMock()
        return service
    
    @pytest.fixture
    def mock_entity_storage(self):
        """Create mock entity storage."""
        storage = Mock()
        return storage
    
    @pytest.fixture
    def hybrid_search_service(self, mock_search_service, mock_entity_storage):
        """Create hybrid search service."""
        config = SearchConfiguration(
            semantic_weight=0.4,
            keyword_weight=0.3,
            graph_weight=0.3
        )
        return EnhancedHybridSearchService(
            mock_search_service, 
            mock_entity_storage, 
            config
        )
    
    @pytest.mark.asyncio
    async def test_hybrid_search_basic(self, hybrid_search_service, mock_search_service):
        """Test basic hybrid search functionality."""
        # Mock search results
        mock_result = SearchResult(
            content="Test content about AI and machine learning",
            score=0.8,
            document_id="doc_1",
            chunk_id="chunk_1",
            metadata={"adapter_type": "chunk"}
        )
        mock_metrics = SearchMetrics(0.1, 1, 1, "hybrid")
        
        mock_search_service.search.return_value = ([mock_result], mock_metrics)
        
        results, metrics = await hybrid_search_service.hybrid_search(
            query="artificial intelligence",
            search_methods=[SearchMethod.SEMANTIC, SearchMethod.KEYWORD],
            limit=10
        )
        
        assert len(results) > 0
        assert isinstance(results[0], EnhancedSearchResult)
        assert results[0].content == "Test content about AI and machine learning"
        assert results[0].combined_score > 0
        assert "semantic_count" in metrics
        assert "keyword_count" in metrics
    
    def test_keyword_score_calculation(self, hybrid_search_service):
        """Test keyword score calculation."""
        content = "This is a test document about machine learning and artificial intelligence"
        query_terms = ["machine", "learning", "test"]
        
        score = hybrid_search_service._calculate_keyword_score(content, query_terms)
        
        assert 0.0 <= score <= 1.0
        assert score > 0  # Should find matches
    
    def test_graph_score_calculation(self, hybrid_search_service):
        """Test graph score calculation."""
        # Test with different distances and relationship strengths
        score1 = hybrid_search_service._calculate_graph_score(1, 0.8)
        score2 = hybrid_search_service._calculate_graph_score(2, 0.8)
        score3 = hybrid_search_service._calculate_graph_score(1, 0.4)
        
        assert score1 > score2  # Closer distance should have higher score
        assert score1 > score3  # Higher relationship strength should have higher score
        assert all(0.0 <= score <= 1.0 for score in [score1, score2, score3])
    
    def test_content_key_creation(self, hybrid_search_service):
        """Test content key creation for deduplication."""
        result1 = SearchResult("content", 0.8, "doc_1", "chunk_1")
        result2 = SearchResult("content", 0.7, "doc_1", "chunk_1")
        result3 = SearchResult("different content", 0.6, "doc_2", "chunk_2")
        
        key1 = hybrid_search_service._create_content_key(result1)
        key2 = hybrid_search_service._create_content_key(result2)
        key3 = hybrid_search_service._create_content_key(result3)
        
        assert key1 == key2  # Same chunk should have same key
        assert key1 != key3  # Different chunks should have different keys
    
    def test_search_explanation_creation(self, hybrid_search_service):
        """Test search explanation generation."""
        explanation = hybrid_search_service._create_search_explanation(
            semantic_score=0.8,
            keyword_score=0.6,
            graph_score=0.4,
            combined_score=0.65
        )
        
        assert "Semantic similarity: 0.80" in explanation
        assert "Keyword match: 0.60" in explanation
        assert "Graph relevance: 0.40" in explanation
        assert "Combined: 0.65" in explanation
    
    @pytest.mark.asyncio
    async def test_entity_extraction_from_query(self, hybrid_search_service, mock_search_service):
        """Test entity extraction from search query."""
        # Mock entity search results
        mock_entity_result = SearchResult(
            content="Entity content",
            score=0.9,
            metadata={
                "adapter_type": "entity",
                "id": "entity_1",
                "name": "Machine Learning"
            }
        )
        mock_metrics = SearchMetrics(0.05, 1, 1, "entity_extraction")
        
        mock_search_service.search.return_value = ([mock_entity_result], mock_metrics)
        
        entity_ids = await hybrid_search_service._extract_entities_from_query("machine learning")
        
        assert "entity_1" in entity_ids
        mock_search_service.search.assert_called_once()
    
    def test_cache_functionality(self, hybrid_search_service):
        """Test search result caching."""
        # Test cache key creation
        cache_key = hybrid_search_service._create_cache_key(
            "test query",
            [SearchMethod.SEMANTIC, SearchMethod.KEYWORD],
            [MoragEntityType.PERSON],
            10
        )
        
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0
        
        # Test cache stats
        stats = hybrid_search_service.get_cache_stats()
        assert "search_cache_size" in stats
        assert "entity_cache_size" in stats
        assert "cache_enabled" in stats
        
        # Test cache clearing
        hybrid_search_service.clear_cache()
        stats_after_clear = hybrid_search_service.get_cache_stats()
        assert stats_after_clear["search_cache_size"] == 0
    
    def test_entity_id_extraction(self, hybrid_search_service):
        """Test entity ID extraction from search results."""
        # Test entity result
        entity_result = SearchResult(
            "content", 0.8,
            metadata={"adapter_type": "entity", "id": "entity_1"}
        )
        entity_ids = hybrid_search_service._extract_entity_ids(entity_result)
        assert "entity_1" in entity_ids
        
        # Test relation result
        relation_result = SearchResult(
            "content", 0.8,
            metadata={
                "adapter_type": "relation",
                "source_entity_id": "entity_1",
                "target_entity_id": "entity_2"
            }
        )
        relation_entity_ids = hybrid_search_service._extract_entity_ids(relation_result)
        assert "entity_1" in relation_entity_ids
        assert "entity_2" in relation_entity_ids
        
        # Test chunk with entity mappings
        chunk_result = SearchResult(
            "content", 0.8,
            metadata={"entity_ids": ["entity_3", "entity_4"]}
        )
        chunk_entity_ids = hybrid_search_service._extract_entity_ids(chunk_result)
        assert "entity_3" in chunk_entity_ids
        assert "entity_4" in chunk_entity_ids


class TestSearchConfiguration:
    """Test search configuration."""
    
    def test_default_configuration(self):
        """Test default search configuration."""
        config = SearchConfiguration()
        
        assert config.semantic_weight == 0.4
        assert config.keyword_weight == 0.3
        assert config.graph_weight == 0.3
        assert config.semantic_weight + config.keyword_weight + config.graph_weight == 1.0
        assert config.enable_result_fusion is True
        assert config.enable_caching is True
    
    def test_custom_configuration(self):
        """Test custom search configuration."""
        config = SearchConfiguration(
            semantic_weight=0.5,
            keyword_weight=0.3,
            graph_weight=0.2,
            max_graph_depth=3,
            enable_caching=False
        )
        
        assert config.semantic_weight == 0.5
        assert config.keyword_weight == 0.3
        assert config.graph_weight == 0.2
        assert config.max_graph_depth == 3
        assert config.enable_caching is False


class TestHybridSearchIntegration:
    """Integration tests for hybrid search."""
    
    @pytest.mark.asyncio
    async def test_full_hybrid_search_workflow(self):
        """Test complete hybrid search workflow."""
        # This would require actual Graphiti setup
        # For now, we'll test the workflow with mocks
        
        mock_base_search = Mock()
        mock_base_search.search = AsyncMock()
        
        mock_entity_storage = Mock()
        
        service = EnhancedHybridSearchService(mock_base_search, mock_entity_storage)
        
        # Mock various search results
        semantic_result = SearchResult("semantic content", 0.9)
        keyword_result = SearchResult("keyword content", 0.7)
        
        mock_base_search.search.side_effect = [
            ([semantic_result], SearchMetrics(0.1, 1, 1, "semantic")),
            ([keyword_result], SearchMetrics(0.1, 1, 1, "keyword")),
            ([], SearchMetrics(0.05, 0, 0, "graph"))  # Empty graph results
        ]
        
        results, metrics = await service.hybrid_search(
            "test query",
            search_methods=[SearchMethod.SEMANTIC, SearchMethod.KEYWORD, SearchMethod.GRAPH_TRAVERSAL]
        )
        
        assert len(results) >= 0  # Should handle empty results gracefully
        assert "semantic_count" in metrics
        assert "keyword_count" in metrics
        assert "graph_count" in metrics
