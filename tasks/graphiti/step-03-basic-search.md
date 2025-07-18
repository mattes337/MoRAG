# Step 3: Basic Search Implementation

**Duration**: 2-3 days  
**Phase**: Proof of Concept  
**Prerequisites**: Steps 1-2 completed, document ingestion working

## Objective

Implement Graphiti-powered search functionality that can replace MoRAG's current chunk-based search, with performance comparison and validation against existing search capabilities.

## Deliverables

1. Graphiti search service with hybrid capabilities
2. Search result formatting and compatibility layer
3. Performance benchmarking tools
4. Search quality validation tests
5. Comparison with existing MoRAG search

## Implementation

### 1. Create Graphiti Search Service

**File**: `packages/morag-graph/src/morag_graph/graphiti/search_service.py`

```python
"""Graphiti-powered search service for MoRAG."""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from graphiti_core import Graphiti

from .config import create_graphiti_instance, GraphitiConfig
from .adapters import EpisodeDocumentReverseAdapter

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Standardized search result format."""
    content: str
    score: float
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    source_type: str = "graphiti"  # "graphiti" or "legacy"


@dataclass
class SearchMetrics:
    """Search performance metrics."""
    query_time: float
    result_count: int
    total_episodes: int
    search_method: str


class GraphitiSearchService:
    """Advanced search service using Graphiti's hybrid capabilities."""
    
    def __init__(self, config: Optional[GraphitiConfig] = None):
        self.config = config
        self.graphiti = create_graphiti_instance(config)
        self.reverse_adapter = EpisodeDocumentReverseAdapter()
        
    async def search(
        self,
        query: str,
        limit: int = 10,
        search_type: str = "hybrid",  # "hybrid", "semantic", "keyword"
        filter_options: Optional[Dict[str, Any]] = None
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """Perform search using Graphiti's capabilities.
        
        Args:
            query: Search query string
            limit: Maximum number of results
            search_type: Type of search to perform
            filter_options: Optional filters for results
            
        Returns:
            Tuple of (search results, metrics)
        """
        start_time = time.time()
        
        try:
            # Perform Graphiti search
            raw_results = await self.graphiti.search(
                query=query,
                limit=limit
            )
            
            # Convert to standardized format
            search_results = []
            for result in raw_results:
                formatted_result = self._format_search_result(result)
                
                # Apply filters if specified
                if filter_options and not self._passes_filters(formatted_result, filter_options):
                    continue
                
                search_results.append(formatted_result)
            
            # Calculate metrics
            query_time = time.time() - start_time
            metrics = SearchMetrics(
                query_time=query_time,
                result_count=len(search_results),
                total_episodes=len(raw_results),
                search_method="graphiti_hybrid"
            )
            
            logger.info(
                f"Graphiti search completed: query='{query}', "
                f"results={len(search_results)}, time={query_time:.3f}s"
            )
            
            return search_results, metrics
            
        except Exception as e:
            logger.error(f"Graphiti search failed: {e}")
            return [], SearchMetrics(
                query_time=time.time() - start_time,
                result_count=0,
                total_episodes=0,
                search_method="graphiti_error"
            )
    
    def _format_search_result(self, raw_result) -> SearchResult:
        """Convert Graphiti search result to standardized format.
        
        Args:
            raw_result: Raw result from Graphiti search
            
        Returns:
            Formatted SearchResult
        """
        # Extract metadata if available
        metadata = getattr(raw_result, 'metadata', {})
        
        # Determine document and chunk IDs from metadata
        document_id = None
        chunk_id = None
        
        if metadata.get('morag_integration'):
            document_id = metadata.get('morag_document_id')
            chunk_id = metadata.get('morag_chunk_id')
        
        return SearchResult(
            content=raw_result.content,
            score=raw_result.score,
            document_id=document_id,
            chunk_id=chunk_id,
            metadata=metadata,
            source_type="graphiti"
        )
    
    def _passes_filters(self, result: SearchResult, filters: Dict[str, Any]) -> bool:
        """Check if search result passes filter criteria.
        
        Args:
            result: Search result to check
            filters: Filter criteria
            
        Returns:
            True if result passes filters
        """
        # Document ID filter
        if 'document_ids' in filters:
            if result.document_id not in filters['document_ids']:
                return False
        
        # Minimum score filter
        if 'min_score' in filters:
            if result.score < filters['min_score']:
                return False
        
        # Content length filter
        if 'min_content_length' in filters:
            if len(result.content) < filters['min_content_length']:
                return False
        
        # MoRAG-only filter
        if filters.get('morag_only', False):
            if not result.metadata or not result.metadata.get('morag_integration'):
                return False
        
        return True
    
    async def search_by_entities(
        self,
        entity_names: List[str],
        limit: int = 10
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """Search for episodes containing specific entities.
        
        Args:
            entity_names: List of entity names to search for
            limit: Maximum number of results
            
        Returns:
            Tuple of (search results, metrics)
        """
        # Create entity-focused query
        query = " OR ".join(entity_names)
        
        return await self.search(
            query=query,
            limit=limit,
            search_type="hybrid"
        )
    
    async def search_by_document(
        self,
        document_id: str,
        query: Optional[str] = None,
        limit: int = 10
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """Search within a specific document.
        
        Args:
            document_id: Document ID to search within
            query: Optional query string
            limit: Maximum number of results
            
        Returns:
            Tuple of (search results, metrics)
        """
        filter_options = {
            'document_ids': [document_id],
            'morag_only': True
        }
        
        search_query = query or "*"  # Search all if no query provided
        
        return await self.search(
            query=search_query,
            limit=limit,
            filter_options=filter_options
        )
    
    async def get_related_content(
        self,
        content: str,
        limit: int = 5
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """Find content related to the given text.
        
        Args:
            content: Content to find related items for
            limit: Maximum number of results
            
        Returns:
            Tuple of (search results, metrics)
        """
        # Extract key terms from content for search
        # This is a simple implementation - could be enhanced with NLP
        words = content.split()
        key_terms = [word for word in words if len(word) > 3][:10]  # Take first 10 significant words
        
        query = " ".join(key_terms)
        
        return await self.search(
            query=query,
            limit=limit,
            search_type="semantic"
        )


class SearchComparison:
    """Service for comparing Graphiti search with legacy MoRAG search."""
    
    def __init__(self, graphiti_service: GraphitiSearchService):
        self.graphiti_service = graphiti_service
    
    async def compare_search_methods(
        self,
        query: str,
        limit: int = 10,
        legacy_search_func: Optional[callable] = None
    ) -> Dict[str, Any]:
        """Compare Graphiti search with legacy search implementation.
        
        Args:
            query: Search query
            limit: Number of results
            legacy_search_func: Optional legacy search function
            
        Returns:
            Comparison results
        """
        comparison = {
            "query": query,
            "limit": limit,
            "graphiti_results": None,
            "legacy_results": None,
            "performance_comparison": {},
            "quality_comparison": {}
        }
        
        # Run Graphiti search
        try:
            graphiti_results, graphiti_metrics = await self.graphiti_service.search(query, limit)
            comparison["graphiti_results"] = {
                "results": [self._result_to_dict(r) for r in graphiti_results],
                "metrics": graphiti_metrics
            }
        except Exception as e:
            comparison["graphiti_results"] = {"error": str(e)}
        
        # Run legacy search if function provided
        if legacy_search_func:
            try:
                start_time = time.time()
                legacy_results = await legacy_search_func(query, limit)
                legacy_time = time.time() - start_time
                
                comparison["legacy_results"] = {
                    "results": legacy_results,
                    "metrics": SearchMetrics(
                        query_time=legacy_time,
                        result_count=len(legacy_results),
                        total_episodes=len(legacy_results),
                        search_method="legacy"
                    )
                }
            except Exception as e:
                comparison["legacy_results"] = {"error": str(e)}
        
        # Calculate performance comparison
        if comparison["graphiti_results"] and comparison["legacy_results"]:
            comparison["performance_comparison"] = self._compare_performance(
                comparison["graphiti_results"]["metrics"],
                comparison["legacy_results"]["metrics"]
            )
        
        return comparison
    
    def _result_to_dict(self, result: SearchResult) -> Dict[str, Any]:
        """Convert SearchResult to dictionary for serialization."""
        return {
            "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
            "score": result.score,
            "document_id": result.document_id,
            "chunk_id": result.chunk_id,
            "source_type": result.source_type
        }
    
    def _compare_performance(self, graphiti_metrics: SearchMetrics, legacy_metrics: SearchMetrics) -> Dict[str, Any]:
        """Compare performance metrics between search methods."""
        return {
            "speed_improvement": {
                "graphiti_time": graphiti_metrics.query_time,
                "legacy_time": legacy_metrics.query_time,
                "improvement_factor": legacy_metrics.query_time / graphiti_metrics.query_time if graphiti_metrics.query_time > 0 else 0
            },
            "result_count": {
                "graphiti": graphiti_metrics.result_count,
                "legacy": legacy_metrics.result_count,
                "difference": graphiti_metrics.result_count - legacy_metrics.result_count
            }
        }


class SearchBenchmark:
    """Benchmarking tools for search performance."""
    
    def __init__(self, search_service: GraphitiSearchService):
        self.search_service = search_service
    
    async def run_benchmark_suite(
        self,
        test_queries: List[str],
        iterations: int = 3
    ) -> Dict[str, Any]:
        """Run comprehensive search benchmarks.
        
        Args:
            test_queries: List of queries to test
            iterations: Number of iterations per query
            
        Returns:
            Benchmark results
        """
        results = {
            "test_queries": test_queries,
            "iterations": iterations,
            "query_results": [],
            "summary": {}
        }
        
        all_times = []
        all_result_counts = []
        
        for query in test_queries:
            query_times = []
            query_result_counts = []
            
            for i in range(iterations):
                search_results, metrics = await self.search_service.search(query)
                query_times.append(metrics.query_time)
                query_result_counts.append(metrics.result_count)
                all_times.append(metrics.query_time)
                all_result_counts.append(metrics.result_count)
            
            results["query_results"].append({
                "query": query,
                "avg_time": sum(query_times) / len(query_times),
                "min_time": min(query_times),
                "max_time": max(query_times),
                "avg_results": sum(query_result_counts) / len(query_result_counts),
                "times": query_times,
                "result_counts": query_result_counts
            })
        
        # Calculate summary statistics
        results["summary"] = {
            "total_queries": len(test_queries) * iterations,
            "avg_query_time": sum(all_times) / len(all_times),
            "min_query_time": min(all_times),
            "max_query_time": max(all_times),
            "avg_results_per_query": sum(all_result_counts) / len(all_result_counts),
            "queries_per_second": len(all_times) / sum(all_times) if sum(all_times) > 0 else 0
        }
        
        return results
```

### 2. Create Search Integration Layer

**File**: `packages/morag-graph/src/morag_graph/graphiti/search_integration.py`

```python
"""Integration layer for Graphiti search with existing MoRAG interfaces."""

import logging
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod

from .search_service import GraphitiSearchService, SearchResult, SearchMetrics

logger = logging.getLogger(__name__)


class SearchInterface(ABC):
    """Abstract interface for search implementations."""
    
    @abstractmethod
    async def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for document chunks."""
        pass
    
    @abstractmethod
    async def search_entities(self, entity_names: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search by entity names."""
        pass


class GraphitiSearchAdapter(SearchInterface):
    """Adapter to make Graphiti search compatible with existing MoRAG interfaces."""
    
    def __init__(self, graphiti_service: GraphitiSearchService):
        self.graphiti_service = graphiti_service
    
    async def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for document chunks using Graphiti.
        
        Args:
            query: Search query
            limit: Maximum results
            
        Returns:
            List of chunk-like results compatible with existing interfaces
        """
        search_results, metrics = await self.graphiti_service.search(query, limit)
        
        # Convert to chunk-compatible format
        chunks = []
        for result in search_results:
            chunk_data = {
                "chunk_id": result.chunk_id or f"graphiti_{hash(result.content)}",
                "document_id": result.document_id,
                "text": result.content,
                "score": result.score,
                "metadata": result.metadata or {},
                "source": "graphiti"
            }
            chunks.append(chunk_data)
        
        logger.info(f"Graphiti chunk search: {len(chunks)} results in {metrics.query_time:.3f}s")
        return chunks
    
    async def search_entities(self, entity_names: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search by entity names using Graphiti.
        
        Args:
            entity_names: List of entity names
            limit: Maximum results
            
        Returns:
            List of entity-related results
        """
        search_results, metrics = await self.graphiti_service.search_by_entities(entity_names, limit)
        
        # Convert to entity-compatible format
        entity_results = []
        for result in search_results:
            entity_data = {
                "content": result.content,
                "score": result.score,
                "document_id": result.document_id,
                "chunk_id": result.chunk_id,
                "entities": entity_names,  # The entities we searched for
                "metadata": result.metadata or {},
                "source": "graphiti"
            }
            entity_results.append(entity_data)
        
        logger.info(f"Graphiti entity search: {len(entity_results)} results in {metrics.query_time:.3f}s")
        return entity_results


class HybridSearchService:
    """Service that can use both Graphiti and legacy search with fallback."""
    
    def __init__(
        self, 
        graphiti_service: GraphitiSearchService,
        legacy_search_service: Optional[SearchInterface] = None,
        prefer_graphiti: bool = True
    ):
        self.graphiti_service = graphiti_service
        self.legacy_service = legacy_search_service
        self.prefer_graphiti = prefer_graphiti
        self.graphiti_adapter = GraphitiSearchAdapter(graphiti_service)
    
    async def search_with_fallback(
        self, 
        query: str, 
        limit: int = 10,
        force_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """Search with automatic fallback between methods.
        
        Args:
            query: Search query
            limit: Maximum results
            force_method: Force specific method ("graphiti" or "legacy")
            
        Returns:
            Search results with method information
        """
        result = {
            "query": query,
            "limit": limit,
            "method_used": None,
            "results": [],
            "metrics": None,
            "fallback_used": False,
            "error": None
        }
        
        # Determine search method
        if force_method == "legacy" and self.legacy_service:
            primary_method = "legacy"
            fallback_method = "graphiti" if self.prefer_graphiti else None
        elif force_method == "graphiti":
            primary_method = "graphiti"
            fallback_method = "legacy" if self.legacy_service else None
        else:
            primary_method = "graphiti" if self.prefer_graphiti else "legacy"
            fallback_method = "legacy" if self.prefer_graphiti else "graphiti"
        
        # Try primary method
        try:
            if primary_method == "graphiti":
                search_results, metrics = await self.graphiti_service.search(query, limit)
                result["results"] = [self._search_result_to_dict(r) for r in search_results]
                result["metrics"] = metrics
                result["method_used"] = "graphiti"
            elif primary_method == "legacy" and self.legacy_service:
                legacy_results = await self.legacy_service.search_chunks(query, limit)
                result["results"] = legacy_results
                result["method_used"] = "legacy"
                # Create mock metrics for legacy
                result["metrics"] = SearchMetrics(
                    query_time=0.0,  # Would need to be measured
                    result_count=len(legacy_results),
                    total_episodes=len(legacy_results),
                    search_method="legacy"
                )
            
        except Exception as e:
            logger.warning(f"Primary search method {primary_method} failed: {e}")
            
            # Try fallback method
            if fallback_method:
                try:
                    if fallback_method == "graphiti":
                        search_results, metrics = await self.graphiti_service.search(query, limit)
                        result["results"] = [self._search_result_to_dict(r) for r in search_results]
                        result["metrics"] = metrics
                        result["method_used"] = "graphiti"
                    elif fallback_method == "legacy" and self.legacy_service:
                        legacy_results = await self.legacy_service.search_chunks(query, limit)
                        result["results"] = legacy_results
                        result["method_used"] = "legacy"
                    
                    result["fallback_used"] = True
                    logger.info(f"Fallback to {fallback_method} successful")
                    
                except Exception as fallback_error:
                    result["error"] = f"Both methods failed. Primary: {e}, Fallback: {fallback_error}"
                    logger.error(result["error"])
            else:
                result["error"] = str(e)
        
        return result
    
    def _search_result_to_dict(self, result: SearchResult) -> Dict[str, Any]:
        """Convert SearchResult to dictionary."""
        return {
            "content": result.content,
            "score": result.score,
            "document_id": result.document_id,
            "chunk_id": result.chunk_id,
            "metadata": result.metadata,
            "source_type": result.source_type
        }
```

## Testing

### Unit Tests

**File**: `packages/morag-graph/tests/test_graphiti_search.py`

```python
"""Unit tests for Graphiti search functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from morag_graph.graphiti.search_service import GraphitiSearchService, SearchResult, SearchMetrics
from morag_graph.graphiti.search_integration import GraphitiSearchAdapter, HybridSearchService


class TestGraphitiSearchService:
    """Test Graphiti search service."""
    
    @pytest.fixture
    def mock_graphiti_service(self):
        """Create mock Graphiti service."""
        service = GraphitiSearchService()
        service.graphiti = Mock()
        return service
    
    @pytest.mark.asyncio
    async def test_basic_search(self, mock_graphiti_service):
        """Test basic search functionality."""
        # Mock Graphiti search results
        mock_result = Mock()
        mock_result.content = "Test content"
        mock_result.score = 0.85
        mock_result.metadata = {"morag_integration": True, "morag_document_id": "doc_1"}
        
        mock_graphiti_service.graphiti.search = AsyncMock(return_value=[mock_result])
        
        results, metrics = await mock_graphiti_service.search("test query")
        
        assert len(results) == 1
        assert results[0].content == "Test content"
        assert results[0].score == 0.85
        assert results[0].document_id == "doc_1"
        assert metrics.result_count == 1
    
    @pytest.mark.asyncio
    async def test_search_filtering(self, mock_graphiti_service):
        """Test search result filtering."""
        # Mock results with different metadata
        mock_result1 = Mock()
        mock_result1.content = "MoRAG content"
        mock_result1.score = 0.9
        mock_result1.metadata = {"morag_integration": True}
        
        mock_result2 = Mock()
        mock_result2.content = "Non-MoRAG content"
        mock_result2.score = 0.8
        mock_result2.metadata = {}
        
        mock_graphiti_service.graphiti.search = AsyncMock(return_value=[mock_result1, mock_result2])
        
        # Test with MoRAG-only filter
        filter_options = {"morag_only": True}
        results, metrics = await mock_graphiti_service.search("test", filter_options=filter_options)
        
        assert len(results) == 1
        assert results[0].content == "MoRAG content"


class TestSearchIntegration:
    """Test search integration components."""
    
    @pytest.fixture
    def mock_graphiti_service(self):
        """Create mock Graphiti service."""
        service = Mock()
        service.search = AsyncMock()
        service.search_by_entities = AsyncMock()
        return service
    
    @pytest.mark.asyncio
    async def test_search_adapter(self, mock_graphiti_service):
        """Test Graphiti search adapter."""
        # Setup mock results
        search_result = SearchResult(
            content="Test chunk content",
            score=0.85,
            document_id="doc_1",
            chunk_id="chunk_1"
        )
        metrics = SearchMetrics(0.1, 1, 1, "graphiti")
        
        mock_graphiti_service.search.return_value = ([search_result], metrics)
        
        adapter = GraphitiSearchAdapter(mock_graphiti_service)
        chunks = await adapter.search_chunks("test query")
        
        assert len(chunks) == 1
        assert chunks[0]["chunk_id"] == "chunk_1"
        assert chunks[0]["document_id"] == "doc_1"
        assert chunks[0]["text"] == "Test chunk content"
        assert chunks[0]["source"] == "graphiti"
    
    @pytest.mark.asyncio
    async def test_hybrid_search_fallback(self, mock_graphiti_service):
        """Test hybrid search with fallback."""
        # Mock Graphiti service to fail
        mock_graphiti_service.search.side_effect = Exception("Graphiti failed")
        
        # Mock legacy service
        mock_legacy_service = Mock()
        mock_legacy_service.search_chunks = AsyncMock(return_value=[{"text": "legacy result"}])
        
        hybrid_service = HybridSearchService(
            mock_graphiti_service, 
            mock_legacy_service, 
            prefer_graphiti=True
        )
        
        result = await hybrid_service.search_with_fallback("test query")
        
        assert result["method_used"] == "legacy"
        assert result["fallback_used"] is True
        assert len(result["results"]) == 1


@pytest.mark.integration
class TestSearchBenchmark:
    """Integration tests for search benchmarking."""
    
    @pytest.mark.asyncio
    async def test_benchmark_suite(self):
        """Test search benchmark suite."""
        from morag_graph.graphiti.search_service import SearchBenchmark
        
        # Mock search service
        mock_service = Mock()
        mock_service.search = AsyncMock()
        
        # Mock consistent results
        search_result = SearchResult("content", 0.8)
        metrics = SearchMetrics(0.1, 1, 1, "test")
        mock_service.search.return_value = ([search_result], metrics)
        
        benchmark = SearchBenchmark(mock_service)
        results = await benchmark.run_benchmark_suite(["query1", "query2"], iterations=2)
        
        assert results["summary"]["total_queries"] == 4
        assert len(results["query_results"]) == 2
        assert "avg_query_time" in results["summary"]
```

## Validation Checklist

- [ ] Graphiti search service works with various query types
- [ ] Search results are properly formatted and compatible
- [ ] Performance metrics are accurately captured
- [ ] Search filtering works correctly
- [ ] Entity-based search functions properly
- [ ] Document-specific search works
- [ ] Search adapter maintains compatibility with existing interfaces
- [ ] Hybrid search with fallback operates correctly
- [ ] Benchmark suite provides meaningful metrics
- [ ] Unit tests cover all major functionality

## Success Criteria

1. **Functional**: Search returns relevant results for various query types
2. **Performance**: Search latency is acceptable (sub-second for most queries)
3. **Compatible**: Results work with existing MoRAG interfaces
4. **Reliable**: Fallback mechanisms work when primary search fails
5. **Measurable**: Comprehensive metrics for performance evaluation

## Next Steps

After completing this step:
1. Run performance benchmarks against existing search
2. Validate search quality with sample queries
3. Document performance characteristics and limitations
4. Proceed to [Step 4: Graphiti Adapter Layer](./step-04-adapter-layer.md)

## Performance Expectations

- **Query Latency**: Target <500ms for typical queries
- **Throughput**: Support 10+ concurrent searches
- **Accuracy**: Maintain or improve search relevance
- **Scalability**: Handle growing episode database efficiently
