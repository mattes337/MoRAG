"""Search integration service for combining Graphiti with existing MoRAG search."""

import asyncio
from typing import Dict, Any, List, Optional, Union
from abc import ABC, abstractmethod
import structlog

from .search_service import GraphitiSearchService, SearchResult, SearchMetrics, SearchResultAdapter
from .config import GraphitiConfig

logger = structlog.get_logger(__name__)


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
        """Initialize the adapter.
        
        Args:
            graphiti_service: Graphiti search service instance
        """
        self.graphiti_service = graphiti_service
    
    async def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for document chunks using Graphiti.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of chunk results in MoRAG format
        """
        try:
            # Perform Graphiti search
            results, metrics = await self.graphiti_service.search(query, limit)
            
            # Convert to MoRAG format
            chunk_results = SearchResultAdapter.to_morag_format(results)
            
            logger.info(
                "Graphiti chunk search completed",
                query=query,
                result_count=len(chunk_results),
                query_time=metrics.query_time
            )
            
            return chunk_results
            
        except Exception as e:
            logger.error(
                "Graphiti chunk search failed",
                query=query,
                error=str(e)
            )
            return []
    
    async def search_entities(self, entity_names: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search by entity names using Graphiti.
        
        Args:
            entity_names: List of entity names to search for
            limit: Maximum number of results
            
        Returns:
            List of entity-related results in MoRAG format
        """
        try:
            # Create query from entity names
            query = " OR ".join(entity_names)
            
            # Search with entity-specific filters
            filter_options = {"entity_search": True}
            results, metrics = await self.graphiti_service.search(
                query, 
                limit, 
                filter_options=filter_options
            )
            
            # Convert to MoRAG format
            entity_results = SearchResultAdapter.to_morag_format(results)
            
            logger.info(
                "Graphiti entity search completed",
                entity_names=entity_names,
                result_count=len(entity_results),
                query_time=metrics.query_time
            )
            
            return entity_results
            
        except Exception as e:
            logger.error(
                "Graphiti entity search failed",
                entity_names=entity_names,
                error=str(e)
            )
            return []


class HybridSearchService:
    """Service that can use both Graphiti and legacy search with fallback."""
    
    def __init__(
        self, 
        graphiti_service: GraphitiSearchService,
        legacy_search_service: Optional[SearchInterface] = None,
        prefer_graphiti: bool = True
    ):
        """Initialize the hybrid search service.
        
        Args:
            graphiti_service: Graphiti search service
            legacy_search_service: Optional legacy search service for fallback
            prefer_graphiti: Whether to prefer Graphiti over legacy search
        """
        self.graphiti_service = graphiti_service
        self.legacy_service = legacy_search_service
        self.prefer_graphiti = prefer_graphiti
        self.graphiti_adapter = GraphitiSearchAdapter(graphiti_service)
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        search_method: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform hybrid search with fallback capability.
        
        Args:
            query: Search query
            limit: Maximum number of results
            search_method: Optional method preference ("graphiti", "legacy", or None for auto)
            
        Returns:
            Dictionary with search results and metadata
        """
        result = {
            "results": [],
            "method_used": None,
            "metrics": None,
            "fallback_used": False,
            "error": None
        }
        
        # Determine search method
        if search_method:
            primary_method = search_method
            fallback_method = "legacy" if search_method == "graphiti" else "graphiti"
        else:
            primary_method = "graphiti" if self.prefer_graphiti else "legacy"
            fallback_method = "legacy" if primary_method == "graphiti" else "graphiti"
        
        # Try primary method
        try:
            if primary_method == "graphiti":
                search_results, metrics = await self.graphiti_service.search(query, limit)
                result["results"] = SearchResultAdapter.to_morag_format(search_results)
                result["metrics"] = self._metrics_to_dict(metrics)
                result["method_used"] = "graphiti"
            elif primary_method == "legacy" and self.legacy_service:
                legacy_results = await self.legacy_service.search_chunks(query, limit)
                result["results"] = legacy_results
                result["method_used"] = "legacy"
                # Create mock metrics for legacy
                result["metrics"] = {
                    "query_time": 0.0,  # Would need to be measured
                    "result_count": len(legacy_results),
                    "total_episodes": len(legacy_results),
                    "search_method": "legacy"
                }
            else:
                raise Exception(f"Primary method '{primary_method}' not available")
                
        except Exception as primary_error:
            logger.warning(
                "Primary search method failed, trying fallback",
                primary_method=primary_method,
                error=str(primary_error)
            )
            
            # Try fallback method
            try:
                if fallback_method == "graphiti":
                    search_results, metrics = await self.graphiti_service.search(query, limit)
                    result["results"] = SearchResultAdapter.to_morag_format(search_results)
                    result["metrics"] = self._metrics_to_dict(metrics)
                    result["method_used"] = "graphiti"
                elif fallback_method == "legacy" and self.legacy_service:
                    legacy_results = await self.legacy_service.search_chunks(query, limit)
                    result["results"] = legacy_results
                    result["method_used"] = "legacy"
                    result["metrics"] = {
                        "query_time": 0.0,
                        "result_count": len(legacy_results),
                        "total_episodes": len(legacy_results),
                        "search_method": "legacy"
                    }
                else:
                    raise Exception(f"Fallback method '{fallback_method}' not available")
                
                result["fallback_used"] = True
                
            except Exception as fallback_error:
                logger.error(
                    "Both primary and fallback search methods failed",
                    primary_method=primary_method,
                    fallback_method=fallback_method,
                    primary_error=str(primary_error),
                    fallback_error=str(fallback_error)
                )
                result["error"] = f"All search methods failed: {str(fallback_error)}"
        
        return result
    
    async def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for chunks using hybrid approach.
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of chunk results
        """
        search_result = await self.search(query, limit)
        return search_result.get("results", [])
    
    async def search_entities(self, entity_names: List[str], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for entities using hybrid approach.
        
        Args:
            entity_names: List of entity names
            limit: Maximum number of results
            
        Returns:
            List of entity-related results
        """
        try:
            # Try Graphiti first
            return await self.graphiti_adapter.search_entities(entity_names, limit)
        except Exception as e:
            logger.warning(
                "Graphiti entity search failed, trying legacy",
                error=str(e)
            )
            # Fallback to legacy if available
            if self.legacy_service:
                return await self.legacy_service.search_entities(entity_names, limit)
            return []
    
    def _metrics_to_dict(self, metrics: SearchMetrics) -> Dict[str, Any]:
        """Convert SearchMetrics to dictionary.
        
        Args:
            metrics: SearchMetrics instance
            
        Returns:
            Dictionary representation of metrics
        """
        return {
            "query_time": metrics.query_time,
            "result_count": metrics.result_count,
            "total_episodes": metrics.total_episodes,
            "search_method": metrics.search_method
        }


# Convenience functions
def create_search_adapter(config: Optional[GraphitiConfig] = None) -> GraphitiSearchAdapter:
    """Create a Graphiti search adapter.
    
    Args:
        config: Optional Graphiti configuration
        
    Returns:
        GraphitiSearchAdapter instance
    """
    graphiti_service = GraphitiSearchService(config)
    return GraphitiSearchAdapter(graphiti_service)


def create_hybrid_search_service(
    config: Optional[GraphitiConfig] = None,
    legacy_service: Optional[SearchInterface] = None,
    prefer_graphiti: bool = True
) -> HybridSearchService:
    """Create a hybrid search service.
    
    Args:
        config: Optional Graphiti configuration
        legacy_service: Optional legacy search service
        prefer_graphiti: Whether to prefer Graphiti over legacy search
        
    Returns:
        HybridSearchService instance
    """
    graphiti_service = GraphitiSearchService(config)
    return HybridSearchService(graphiti_service, legacy_service, prefer_graphiti)
