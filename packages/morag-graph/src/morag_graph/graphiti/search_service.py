"""Graphiti-powered search service for MoRAG."""

import asyncio
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import structlog

from .config import GraphitiConfig
from .connection import GraphitiConnectionService

logger = structlog.get_logger(__name__)


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
        """Initialize the Graphiti search service.
        
        Args:
            config: Optional Graphiti configuration
        """
        self.config = config
        self.connection_service = GraphitiConnectionService(config)
        
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
            async with self.connection_service as conn:
                # Perform search using Graphiti
                raw_results = await conn.search_episodes(query, limit)
                
                # Convert to standardized format
                search_results = []
                for raw_result in raw_results:
                    formatted_result = self._format_search_result(raw_result)
                    if self._passes_filters(formatted_result, filter_options):
                        search_results.append(formatted_result)
                
                # Create metrics
                query_time = time.time() - start_time
                metrics = SearchMetrics(
                    query_time=query_time,
                    result_count=len(search_results),
                    total_episodes=len(raw_results),
                    search_method=search_type
                )
                
                logger.info(
                    "Graphiti search completed",
                    query=query,
                    result_count=len(search_results),
                    query_time=query_time,
                    search_type=search_type
                )
                
                return search_results, metrics
                
        except Exception as e:
            logger.error(
                "Graphiti search failed",
                query=query,
                error=str(e)
            )
            # Return empty results with error metrics
            query_time = time.time() - start_time
            metrics = SearchMetrics(
                query_time=query_time,
                result_count=0,
                total_episodes=0,
                search_method=f"{search_type}_error"
            )
            return [], metrics
    
    async def search_by_document_id(
        self,
        document_id: str,
        limit: int = 10
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """Search for episodes related to a specific document.
        
        Args:
            document_id: Document ID to search for
            limit: Maximum number of results
            
        Returns:
            Tuple of (search results, metrics)
        """
        # Use document ID as a filter in the search
        filter_options = {"morag_document_id": document_id}
        return await self.search(
            query=f"document:{document_id}",
            limit=limit,
            filter_options=filter_options
        )
    
    async def search_by_metadata(
        self,
        metadata_filters: Dict[str, Any],
        limit: int = 10
    ) -> Tuple[List[SearchResult], SearchMetrics]:
        """Search for episodes by metadata criteria.
        
        Args:
            metadata_filters: Dictionary of metadata key-value pairs to filter by
            limit: Maximum number of results
            
        Returns:
            Tuple of (search results, metrics)
        """
        # Create a query from metadata filters
        query_parts = []
        for key, value in metadata_filters.items():
            query_parts.append(f"{key}:{value}")
        
        query = " AND ".join(query_parts)
        
        return await self.search(
            query=query,
            limit=limit,
            filter_options=metadata_filters
        )
    
    def _format_search_result(self, raw_result: Dict[str, Any]) -> SearchResult:
        """Convert Graphiti search result to standardized format.
        
        Args:
            raw_result: Raw result from Graphiti search
            
        Returns:
            Formatted SearchResult
        """
        # Extract basic information
        content = raw_result.get("content", "")
        score = raw_result.get("score", 0.0)
        
        # Extract metadata if available
        metadata = raw_result.get("metadata", {})
        
        # Determine document and chunk IDs from metadata
        document_id = metadata.get("morag_document_id")
        chunk_id = metadata.get("morag_chunk_id")
        
        return SearchResult(
            content=content,
            score=score,
            document_id=document_id,
            chunk_id=chunk_id,
            metadata=metadata,
            source_type="graphiti"
        )
    
    def _passes_filters(
        self,
        result: SearchResult,
        filter_options: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if a search result passes the given filters.
        
        Args:
            result: Search result to check
            filter_options: Optional filters to apply
            
        Returns:
            True if result passes filters, False otherwise
        """
        if not filter_options:
            return True
        
        if not result.metadata:
            return False
        
        for key, expected_value in filter_options.items():
            actual_value = result.metadata.get(key)
            if actual_value != expected_value:
                return False
        
        return True


class SearchResultAdapter:
    """Adapter to convert Graphiti search results to MoRAG format."""
    
    @staticmethod
    def to_morag_format(results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Convert Graphiti search results to MoRAG API format.
        
        Args:
            results: List of Graphiti search results
            
        Returns:
            List of results in MoRAG API format
        """
        morag_results = []
        
        for result in results:
            morag_result = {
                "content": result.content,
                "score": result.score,
                "metadata": {
                    "source_type": result.source_type,
                    "document_id": result.document_id,
                    "chunk_id": result.chunk_id,
                }
            }
            
            # Add original metadata if available
            if result.metadata:
                morag_result["metadata"].update(result.metadata)
            
            morag_results.append(morag_result)
        
        return morag_results
    
    @staticmethod
    def to_dict(result: SearchResult) -> Dict[str, Any]:
        """Convert a single SearchResult to dictionary format.
        
        Args:
            result: Search result to convert
            
        Returns:
            Dictionary representation of the result
        """
        return {
            "content": result.content,
            "score": result.score,
            "document_id": result.document_id,
            "chunk_id": result.chunk_id,
            "metadata": result.metadata,
            "source_type": result.source_type
        }


# Convenience function for creating search service
def create_search_service(config: Optional[GraphitiConfig] = None) -> GraphitiSearchService:
    """Create a Graphiti search service.
    
    Args:
        config: Optional Graphiti configuration
        
    Returns:
        GraphitiSearchService instance
    """
    return GraphitiSearchService(config)
