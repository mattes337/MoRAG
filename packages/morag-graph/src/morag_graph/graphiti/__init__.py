"""Graphiti integration for MoRAG.

This module provides integration with Graphiti, a temporal knowledge graph system
that uses episodes to represent knowledge with built-in deduplication, temporal queries,
and hybrid search capabilities.
"""

from .config import GraphitiConfig, create_graphiti_instance
from .connection import GraphitiConnectionService
from .episode_mapper import DocumentEpisodeMapper, create_episode_mapper
from .search_service import GraphitiSearchService, SearchResult, SearchMetrics, SearchResultAdapter, create_search_service
from .search_integration import SearchInterface, GraphitiSearchAdapter, HybridSearchService, create_search_adapter, create_hybrid_search_service

__all__ = [
    "GraphitiConfig",
    "create_graphiti_instance",
    "GraphitiConnectionService",
    "DocumentEpisodeMapper",
    "create_episode_mapper",
    "GraphitiSearchService",
    "SearchResult",
    "SearchMetrics",
    "SearchResultAdapter",
    "create_search_service",
    "SearchInterface",
    "GraphitiSearchAdapter",
    "HybridSearchService",
    "create_search_adapter",
    "create_hybrid_search_service",
]
