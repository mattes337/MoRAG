"""Tests for Graphiti search integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from morag_graph.graphiti.search_integration import (
    SearchInterface, GraphitiSearchAdapter, HybridSearchService,
    create_search_adapter, create_hybrid_search_service
)
from morag_graph.graphiti.search_service import GraphitiSearchService, SearchResult, SearchMetrics
from morag_graph.graphiti.config import GraphitiConfig


class MockLegacySearchService(SearchInterface):
    """Mock legacy search service for testing."""
    
    def __init__(self):
        self.search_chunks_results = []
        self.search_entities_results = []
    
    async def search_chunks(self, query: str, limit: int = 10) -> list:
        return self.search_chunks_results[:limit]
    
    async def search_entities(self, entity_names: list, limit: int = 10) -> list:
        return self.search_entities_results[:limit]


class TestGraphitiSearchAdapter:
    """Test Graphiti search adapter."""
    
    def test_init(self):
        """Test adapter initialization."""
        mock_service = MagicMock(spec=GraphitiSearchService)
        adapter = GraphitiSearchAdapter(mock_service)
        
        assert adapter.graphiti_service == mock_service
    
    @pytest.mark.asyncio
    async def test_search_chunks_success(self):
        """Test successful chunk search."""
        # Mock Graphiti service
        mock_service = AsyncMock(spec=GraphitiSearchService)
        search_results = [
            SearchResult(content="Content 1", score=0.9, document_id="doc-1"),
            SearchResult(content="Content 2", score=0.8, document_id="doc-2")
        ]
        metrics = SearchMetrics(0.1, 2, 2, "hybrid")
        mock_service.search = AsyncMock(return_value=(search_results, metrics))
        
        adapter = GraphitiSearchAdapter(mock_service)
        results = await adapter.search_chunks("test query", limit=5)
        
        assert len(results) == 2
        assert results[0]["content"] == "Content 1"
        assert results[0]["score"] == 0.9
        assert results[0]["metadata"]["document_id"] == "doc-1"
        
        # Verify service was called correctly
        mock_service.search.assert_called_once_with("test query", 5)
    
    @pytest.mark.asyncio
    async def test_search_chunks_failure(self):
        """Test chunk search failure handling."""
        # Mock Graphiti service to raise exception
        mock_service = AsyncMock(spec=GraphitiSearchService)
        mock_service.search = AsyncMock(side_effect=Exception("Search failed"))
        
        adapter = GraphitiSearchAdapter(mock_service)
        results = await adapter.search_chunks("test query")
        
        assert results == []
    
    @pytest.mark.asyncio
    async def test_search_entities_success(self):
        """Test successful entity search."""
        # Mock Graphiti service
        mock_service = AsyncMock(spec=GraphitiSearchService)
        search_results = [
            SearchResult(content="Entity content", score=0.95, document_id="doc-1")
        ]
        metrics = SearchMetrics(0.1, 1, 1, "hybrid")
        mock_service.search = AsyncMock(return_value=(search_results, metrics))
        
        adapter = GraphitiSearchAdapter(mock_service)
        results = await adapter.search_entities(["entity1", "entity2"], limit=3)
        
        assert len(results) == 1
        assert results[0]["content"] == "Entity content"
        
        # Verify service was called with entity query
        mock_service.search.assert_called_once_with(
            "entity1 OR entity2", 
            3, 
            filter_options={"entity_search": True}
        )
    
    @pytest.mark.asyncio
    async def test_search_entities_failure(self):
        """Test entity search failure handling."""
        # Mock Graphiti service to raise exception
        mock_service = AsyncMock(spec=GraphitiSearchService)
        mock_service.search = AsyncMock(side_effect=Exception("Entity search failed"))
        
        adapter = GraphitiSearchAdapter(mock_service)
        results = await adapter.search_entities(["entity1"])
        
        assert results == []


class TestHybridSearchService:
    """Test hybrid search service."""
    
    def test_init(self):
        """Test service initialization."""
        mock_graphiti = MagicMock(spec=GraphitiSearchService)
        mock_legacy = MagicMock(spec=SearchInterface)
        
        service = HybridSearchService(mock_graphiti, mock_legacy, prefer_graphiti=False)
        
        assert service.graphiti_service == mock_graphiti
        assert service.legacy_service == mock_legacy
        assert service.prefer_graphiti is False
        assert isinstance(service.graphiti_adapter, GraphitiSearchAdapter)
    
    @pytest.mark.asyncio
    async def test_search_graphiti_success(self):
        """Test successful Graphiti search."""
        # Mock Graphiti service
        mock_graphiti = AsyncMock(spec=GraphitiSearchService)
        search_results = [SearchResult(content="Graphiti result", score=0.9)]
        metrics = SearchMetrics(0.1, 1, 1, "graphiti")
        mock_graphiti.search = AsyncMock(return_value=(search_results, metrics))
        
        service = HybridSearchService(mock_graphiti, prefer_graphiti=True)
        result = await service.search("test query", limit=5)
        
        assert result["method_used"] == "graphiti"
        assert result["fallback_used"] is False
        assert result["error"] is None
        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == "Graphiti result"
        assert result["metrics"]["search_method"] == "graphiti"
    
    @pytest.mark.asyncio
    async def test_search_legacy_success(self):
        """Test successful legacy search."""
        mock_graphiti = MagicMock(spec=GraphitiSearchService)
        mock_legacy = AsyncMock(spec=SearchInterface)
        legacy_results = [{"content": "Legacy result", "score": 0.8}]
        mock_legacy.search_chunks = AsyncMock(return_value=legacy_results)
        
        service = HybridSearchService(mock_graphiti, mock_legacy, prefer_graphiti=False)
        result = await service.search("test query", search_method="legacy")
        
        assert result["method_used"] == "legacy"
        assert result["fallback_used"] is False
        assert result["error"] is None
        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == "Legacy result"
    
    @pytest.mark.asyncio
    async def test_search_with_fallback(self):
        """Test search with fallback when primary method fails."""
        # Mock Graphiti to fail
        mock_graphiti = AsyncMock(spec=GraphitiSearchService)
        mock_graphiti.search = AsyncMock(side_effect=Exception("Graphiti failed"))
        
        # Mock legacy to succeed
        mock_legacy = AsyncMock(spec=SearchInterface)
        legacy_results = [{"content": "Fallback result", "score": 0.7}]
        mock_legacy.search_chunks = AsyncMock(return_value=legacy_results)
        
        service = HybridSearchService(mock_graphiti, mock_legacy, prefer_graphiti=True)
        result = await service.search("test query")
        
        assert result["method_used"] == "legacy"
        assert result["fallback_used"] is True
        assert result["error"] is None
        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == "Fallback result"
    
    @pytest.mark.asyncio
    async def test_search_both_methods_fail(self):
        """Test search when both methods fail."""
        # Mock both services to fail
        mock_graphiti = AsyncMock(spec=GraphitiSearchService)
        mock_graphiti.search = AsyncMock(side_effect=Exception("Graphiti failed"))
        
        mock_legacy = AsyncMock(spec=SearchInterface)
        mock_legacy.search_chunks = AsyncMock(side_effect=Exception("Legacy failed"))
        
        service = HybridSearchService(mock_graphiti, mock_legacy)
        result = await service.search("test query")
        
        assert result["method_used"] is None
        assert result["fallback_used"] is False
        assert result["error"] is not None
        assert "All search methods failed" in result["error"]
        assert len(result["results"]) == 0
    
    @pytest.mark.asyncio
    async def test_search_chunks(self):
        """Test search_chunks method."""
        mock_graphiti = AsyncMock(spec=GraphitiSearchService)
        search_results = [SearchResult(content="Chunk result", score=0.9)]
        metrics = SearchMetrics(0.1, 1, 1, "graphiti")
        mock_graphiti.search = AsyncMock(return_value=(search_results, metrics))
        
        service = HybridSearchService(mock_graphiti)
        
        # Mock the search method
        with patch.object(service, 'search') as mock_search:
            mock_search.return_value = {
                "results": [{"content": "Chunk result", "score": 0.9}],
                "method_used": "graphiti"
            }
            
            results = await service.search_chunks("test query", limit=3)
        
        assert len(results) == 1
        assert results[0]["content"] == "Chunk result"
        mock_search.assert_called_once_with("test query", 3)
    
    @pytest.mark.asyncio
    async def test_search_entities_graphiti_success(self):
        """Test entity search with Graphiti success."""
        mock_graphiti = AsyncMock(spec=GraphitiSearchService)
        service = HybridSearchService(mock_graphiti)
        
        # Mock the adapter
        with patch.object(service.graphiti_adapter, 'search_entities') as mock_search:
            mock_search.return_value = [{"content": "Entity result", "score": 0.9}]
            
            results = await service.search_entities(["entity1"], limit=5)
        
        assert len(results) == 1
        assert results[0]["content"] == "Entity result"
        mock_search.assert_called_once_with(["entity1"], 5)
    
    @pytest.mark.asyncio
    async def test_search_entities_with_fallback(self):
        """Test entity search with fallback to legacy."""
        mock_graphiti = MagicMock(spec=GraphitiSearchService)
        mock_legacy = AsyncMock(spec=SearchInterface)
        legacy_results = [{"content": "Legacy entity result", "score": 0.8}]
        mock_legacy.search_entities = AsyncMock(return_value=legacy_results)
        
        service = HybridSearchService(mock_graphiti, mock_legacy)
        
        # Mock the adapter to fail
        with patch.object(service.graphiti_adapter, 'search_entities') as mock_search:
            mock_search.side_effect = Exception("Graphiti entity search failed")
            
            results = await service.search_entities(["entity1"])
        
        assert len(results) == 1
        assert results[0]["content"] == "Legacy entity result"
        mock_legacy.search_entities.assert_called_once_with(["entity1"], 10)
    
    def test_metrics_to_dict(self):
        """Test converting metrics to dictionary."""
        mock_graphiti = MagicMock(spec=GraphitiSearchService)
        service = HybridSearchService(mock_graphiti)
        
        metrics = SearchMetrics(0.123, 5, 10, "hybrid")
        metrics_dict = service._metrics_to_dict(metrics)
        
        assert metrics_dict["query_time"] == 0.123
        assert metrics_dict["result_count"] == 5
        assert metrics_dict["total_episodes"] == 10
        assert metrics_dict["search_method"] == "hybrid"


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_search_adapter(self):
        """Test creating search adapter."""
        config = GraphitiConfig(openai_api_key="test-key")
        
        with patch('morag_graph.graphiti.search_integration.GraphitiSearchService') as mock_service_class:
            mock_service = MagicMock()
            mock_service_class.return_value = mock_service
            
            adapter = create_search_adapter(config)
        
        assert isinstance(adapter, GraphitiSearchAdapter)
        mock_service_class.assert_called_once_with(config)
    
    def test_create_hybrid_search_service(self):
        """Test creating hybrid search service."""
        config = GraphitiConfig(openai_api_key="test-key")
        mock_legacy = MagicMock(spec=SearchInterface)
        
        with patch('morag_graph.graphiti.search_integration.GraphitiSearchService') as mock_service_class:
            mock_service = MagicMock()
            mock_service_class.return_value = mock_service
            
            hybrid_service = create_hybrid_search_service(
                config, 
                mock_legacy, 
                prefer_graphiti=False
            )
        
        assert isinstance(hybrid_service, HybridSearchService)
        assert hybrid_service.legacy_service == mock_legacy
        assert hybrid_service.prefer_graphiti is False
        mock_service_class.assert_called_once_with(config)
