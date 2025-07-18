"""Tests for Graphiti search service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import time

from morag_graph.graphiti.search_service import (
    GraphitiSearchService, SearchResult, SearchMetrics, SearchResultAdapter, create_search_service
)
from morag_graph.graphiti.config import GraphitiConfig


class TestSearchResult:
    """Test SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test creating a search result."""
        result = SearchResult(
            content="Test content",
            score=0.85,
            document_id="doc-123",
            chunk_id="chunk-456",
            metadata={"key": "value"},
            source_type="graphiti"
        )
        
        assert result.content == "Test content"
        assert result.score == 0.85
        assert result.document_id == "doc-123"
        assert result.chunk_id == "chunk-456"
        assert result.metadata == {"key": "value"}
        assert result.source_type == "graphiti"
    
    def test_search_result_defaults(self):
        """Test search result with default values."""
        result = SearchResult(content="Test", score=0.5)
        
        assert result.content == "Test"
        assert result.score == 0.5
        assert result.document_id is None
        assert result.chunk_id is None
        assert result.metadata is None
        assert result.source_type == "graphiti"


class TestSearchMetrics:
    """Test SearchMetrics dataclass."""
    
    def test_search_metrics_creation(self):
        """Test creating search metrics."""
        metrics = SearchMetrics(
            query_time=0.123,
            result_count=5,
            total_episodes=10,
            search_method="hybrid"
        )
        
        assert metrics.query_time == 0.123
        assert metrics.result_count == 5
        assert metrics.total_episodes == 10
        assert metrics.search_method == "hybrid"


class TestGraphitiSearchService:
    """Test Graphiti search service."""
    
    def test_init(self):
        """Test service initialization."""
        config = GraphitiConfig(openai_api_key="test-key")
        service = GraphitiSearchService(config)
        
        assert service.config == config
        assert service.connection_service is not None
    
    def test_init_without_config(self):
        """Test service initialization without config."""
        service = GraphitiSearchService()
        
        assert service.config is None
        assert service.connection_service is not None
    
    @pytest.mark.asyncio
    async def test_search_success(self):
        """Test successful search."""
        service = GraphitiSearchService()
        
        # Mock connection service
        mock_conn = AsyncMock()
        mock_raw_results = [
            {
                "content": "Test content 1",
                "score": 0.9,
                "metadata": {"morag_document_id": "doc-1", "morag_chunk_id": "chunk-1"}
            },
            {
                "content": "Test content 2", 
                "score": 0.8,
                "metadata": {"morag_document_id": "doc-2"}
            }
        ]
        mock_conn.search_episodes = AsyncMock(return_value=mock_raw_results)
        
        with patch.object(service, 'connection_service') as mock_service:
            mock_service.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_service.__aexit__ = AsyncMock(return_value=None)
            
            results, metrics = await service.search("test query", limit=5)
        
        assert len(results) == 2
        assert isinstance(metrics, SearchMetrics)
        
        # Check first result
        assert results[0].content == "Test content 1"
        assert results[0].score == 0.9
        assert results[0].document_id == "doc-1"
        assert results[0].chunk_id == "chunk-1"
        assert results[0].source_type == "graphiti"
        
        # Check second result
        assert results[1].content == "Test content 2"
        assert results[1].score == 0.8
        assert results[1].document_id == "doc-2"
        assert results[1].chunk_id is None
        
        # Check metrics
        assert metrics.result_count == 2
        assert metrics.search_method == "hybrid"
        assert metrics.query_time > 0
    
    @pytest.mark.asyncio
    async def test_search_with_filters(self):
        """Test search with filters."""
        service = GraphitiSearchService()
        
        # Mock connection service
        mock_conn = AsyncMock()
        mock_raw_results = [
            {
                "content": "Filtered content",
                "score": 0.9,
                "metadata": {"morag_document_id": "doc-1", "category": "test"}
            },
            {
                "content": "Unfiltered content",
                "score": 0.8,
                "metadata": {"morag_document_id": "doc-2", "category": "other"}
            }
        ]
        mock_conn.search_episodes = AsyncMock(return_value=mock_raw_results)
        
        with patch.object(service, 'connection_service') as mock_service:
            mock_service.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_service.__aexit__ = AsyncMock(return_value=None)
            
            results, metrics = await service.search(
                "test query", 
                limit=5,
                filter_options={"category": "test"}
            )
        
        # Should only return the filtered result
        assert len(results) == 1
        assert results[0].content == "Filtered content"
        assert results[0].metadata["category"] == "test"
    
    @pytest.mark.asyncio
    async def test_search_failure(self):
        """Test search failure handling."""
        service = GraphitiSearchService()
        
        # Mock connection service to raise exception
        with patch.object(service, 'connection_service') as mock_service:
            mock_service.__aenter__ = AsyncMock(side_effect=Exception("Connection failed"))
            
            results, metrics = await service.search("test query")
        
        assert len(results) == 0
        assert metrics.result_count == 0
        assert metrics.total_episodes == 0
        assert "error" in metrics.search_method
    
    @pytest.mark.asyncio
    async def test_search_by_document_id(self):
        """Test searching by document ID."""
        service = GraphitiSearchService()
        
        # Mock the search method
        expected_results = [SearchResult(content="Doc content", score=0.9)]
        expected_metrics = SearchMetrics(0.1, 1, 1, "hybrid")
        
        with patch.object(service, 'search', return_value=(expected_results, expected_metrics)) as mock_search:
            results, metrics = await service.search_by_document_id("doc-123", limit=5)
        
        # Verify search was called with correct parameters
        mock_search.assert_called_once_with(
            query="document:doc-123",
            limit=5,
            filter_options={"morag_document_id": "doc-123"}
        )
        
        assert results == expected_results
        assert metrics == expected_metrics
    
    @pytest.mark.asyncio
    async def test_search_by_metadata(self):
        """Test searching by metadata."""
        service = GraphitiSearchService()
        
        # Mock the search method
        expected_results = [SearchResult(content="Meta content", score=0.8)]
        expected_metrics = SearchMetrics(0.1, 1, 1, "hybrid")
        
        with patch.object(service, 'search', return_value=(expected_results, expected_metrics)) as mock_search:
            metadata_filters = {"category": "test", "type": "document"}
            results, metrics = await service.search_by_metadata(metadata_filters, limit=3)
        
        # Verify search was called with correct parameters
        mock_search.assert_called_once_with(
            query="category:test AND type:document",
            limit=3,
            filter_options=metadata_filters
        )
        
        assert results == expected_results
        assert metrics == expected_metrics
    
    def test_format_search_result(self):
        """Test formatting raw search results."""
        service = GraphitiSearchService()
        
        raw_result = {
            "content": "Test content",
            "score": 0.75,
            "metadata": {
                "morag_document_id": "doc-123",
                "morag_chunk_id": "chunk-456",
                "extra_field": "extra_value"
            }
        }
        
        formatted = service._format_search_result(raw_result)
        
        assert isinstance(formatted, SearchResult)
        assert formatted.content == "Test content"
        assert formatted.score == 0.75
        assert formatted.document_id == "doc-123"
        assert formatted.chunk_id == "chunk-456"
        assert formatted.metadata["extra_field"] == "extra_value"
        assert formatted.source_type == "graphiti"
    
    def test_passes_filters(self):
        """Test filter checking."""
        service = GraphitiSearchService()
        
        result = SearchResult(
            content="Test",
            score=0.5,
            metadata={"category": "test", "type": "document"}
        )
        
        # No filters - should pass
        assert service._passes_filters(result, None) is True
        
        # Matching filters - should pass
        assert service._passes_filters(result, {"category": "test"}) is True
        assert service._passes_filters(result, {"category": "test", "type": "document"}) is True
        
        # Non-matching filters - should fail
        assert service._passes_filters(result, {"category": "other"}) is False
        assert service._passes_filters(result, {"category": "test", "type": "other"}) is False
        
        # Result without metadata - should fail with filters
        result_no_meta = SearchResult(content="Test", score=0.5)
        assert service._passes_filters(result_no_meta, {"category": "test"}) is False


class TestSearchResultAdapter:
    """Test SearchResultAdapter."""
    
    def test_to_morag_format(self):
        """Test converting to MoRAG format."""
        results = [
            SearchResult(
                content="Content 1",
                score=0.9,
                document_id="doc-1",
                chunk_id="chunk-1",
                metadata={"extra": "data"},
                source_type="graphiti"
            ),
            SearchResult(
                content="Content 2",
                score=0.8,
                document_id="doc-2"
            )
        ]
        
        morag_results = SearchResultAdapter.to_morag_format(results)
        
        assert len(morag_results) == 2
        
        # Check first result
        assert morag_results[0]["content"] == "Content 1"
        assert morag_results[0]["score"] == 0.9
        assert morag_results[0]["metadata"]["source_type"] == "graphiti"
        assert morag_results[0]["metadata"]["document_id"] == "doc-1"
        assert morag_results[0]["metadata"]["chunk_id"] == "chunk-1"
        assert morag_results[0]["metadata"]["extra"] == "data"
        
        # Check second result
        assert morag_results[1]["content"] == "Content 2"
        assert morag_results[1]["score"] == 0.8
        assert morag_results[1]["metadata"]["document_id"] == "doc-2"
        assert morag_results[1]["metadata"]["chunk_id"] is None
    
    def test_to_dict(self):
        """Test converting single result to dict."""
        result = SearchResult(
            content="Test content",
            score=0.85,
            document_id="doc-123",
            chunk_id="chunk-456",
            metadata={"key": "value"},
            source_type="graphiti"
        )
        
        result_dict = SearchResultAdapter.to_dict(result)
        
        assert result_dict["content"] == "Test content"
        assert result_dict["score"] == 0.85
        assert result_dict["document_id"] == "doc-123"
        assert result_dict["chunk_id"] == "chunk-456"
        assert result_dict["metadata"] == {"key": "value"}
        assert result_dict["source_type"] == "graphiti"


class TestCreateSearchService:
    """Test search service creation function."""
    
    def test_create_service_with_config(self):
        """Test creating service with config."""
        config = GraphitiConfig(openai_api_key="test-key")
        service = create_search_service(config)
        
        assert isinstance(service, GraphitiSearchService)
        assert service.config == config
    
    def test_create_service_without_config(self):
        """Test creating service without config."""
        service = create_search_service()
        
        assert isinstance(service, GraphitiSearchService)
        assert service.config is None
