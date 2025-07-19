"""Enhanced hybrid search service combining multiple search methods."""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import structlog

from .search_service import GraphitiSearchService, SearchResult, SearchMetrics
from .entity_storage import GraphitiEntityStorage
from .custom_schema import MoragEntityType, schema_registry

logger = structlog.get_logger(__name__)


class SearchMethod(Enum):
    """Available search methods."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    GRAPH_TRAVERSAL = "graph_traversal"
    HYBRID = "hybrid"


@dataclass
class EnhancedSearchResult:
    """Enhanced search result with multiple scores."""
    content: str
    semantic_score: float
    keyword_score: float
    graph_score: float
    combined_score: float
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    entity_ids: List[str] = None
    metadata: Optional[Dict[str, Any]] = None
    search_method: str = "hybrid"
    explanation: Optional[str] = None

    def __post_init__(self):
        if self.entity_ids is None:
            self.entity_ids = []


@dataclass
class SearchConfiguration:
    """Configuration for hybrid search."""
    semantic_weight: float = 0.4
    keyword_weight: float = 0.3
    graph_weight: float = 0.3
    min_semantic_score: float = 0.1
    min_keyword_score: float = 0.1
    min_graph_score: float = 0.1
    enable_result_fusion: bool = True
    max_graph_depth: int = 2
    enable_caching: bool = True


class EnhancedHybridSearchService:
    """Enhanced hybrid search service with multiple search methods."""
    
    def __init__(
        self, 
        base_search_service: GraphitiSearchService,
        entity_storage: GraphitiEntityStorage,
        config: Optional[SearchConfiguration] = None
    ):
        self.base_search = base_search_service
        self.entity_storage = entity_storage
        self.config = config or SearchConfiguration()
        
        # Cache for search results
        self._search_cache: Dict[str, List[EnhancedSearchResult]] = {}
        self._entity_cache: Dict[str, Dict[str, Any]] = {}
    
    async def hybrid_search(
        self,
        query: str,
        search_methods: List[SearchMethod] = None,
        entity_types: Optional[List[MoragEntityType]] = None,
        limit: int = 20,
        include_explanation: bool = False
    ) -> Tuple[List[EnhancedSearchResult], Dict[str, Any]]:
        """Perform hybrid search combining multiple methods.
        
        Args:
            query: Search query
            search_methods: Methods to use (defaults to all)
            entity_types: Optional entity type filters
            limit: Maximum results
            include_explanation: Whether to include search explanations
            
        Returns:
            Tuple of (enhanced results, search metrics)
        """
        if search_methods is None:
            search_methods = [SearchMethod.SEMANTIC, SearchMethod.KEYWORD, SearchMethod.GRAPH_TRAVERSAL]
        
        # Check cache first
        cache_key = self._create_cache_key(query, search_methods, entity_types, limit)
        if self.config.enable_caching and cache_key in self._search_cache:
            cached_results = self._search_cache[cache_key]
            return cached_results, {"cache_hit": True, "result_count": len(cached_results)}
        
        # Perform searches in parallel
        search_tasks = []
        
        if SearchMethod.SEMANTIC in search_methods:
            search_tasks.append(self._semantic_search(query, entity_types, limit))
        
        if SearchMethod.KEYWORD in search_methods:
            search_tasks.append(self._keyword_search(query, entity_types, limit))
        
        if SearchMethod.GRAPH_TRAVERSAL in search_methods:
            search_tasks.append(self._graph_traversal_search(query, entity_types, limit))
        
        # Execute searches concurrently
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process results
        semantic_results = search_results[0] if SearchMethod.SEMANTIC in search_methods else []
        keyword_results = search_results[1] if SearchMethod.KEYWORD in search_methods else []
        graph_results = search_results[2] if SearchMethod.GRAPH_TRAVERSAL in search_methods else []
        
        # Handle exceptions
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.warning("Search method failed", method_index=i, error=str(result))
        
        # Combine and rank results
        combined_results = await self._combine_and_rank_results(
            semantic_results, keyword_results, graph_results, query, include_explanation
        )
        
        # Apply final filtering and limiting
        final_results = combined_results[:limit]
        
        # Cache results
        if self.config.enable_caching:
            self._search_cache[cache_key] = final_results
        
        # Create metrics
        metrics = {
            "semantic_count": len(semantic_results),
            "keyword_count": len(keyword_results),
            "graph_count": len(graph_results),
            "combined_count": len(combined_results),
            "final_count": len(final_results),
            "search_methods": [method.value for method in search_methods]
        }
        
        return final_results, metrics
    
    async def _semantic_search(
        self,
        query: str,
        entity_types: Optional[List[MoragEntityType]],
        limit: int
    ) -> List[Tuple[SearchResult, float]]:
        """Perform semantic search using embeddings."""
        # Build semantic search query
        search_query = query
        
        # Add entity type filters
        if entity_types:
            type_filters = [f"type:{et.value}" for et in entity_types]
            search_query += f" AND ({' OR '.join(type_filters)})"
        
        # Use Graphiti's semantic search capabilities
        results, _ = await self.base_search.search(
            query=search_query,
            limit=limit * 2,  # Get more results for better ranking
            search_type="semantic"
        )
        
        # Convert to tuples with semantic scores
        semantic_results = []
        for result in results:
            # Graphiti's score is already semantic similarity
            semantic_score = result.score
            if semantic_score >= self.config.min_semantic_score:
                semantic_results.append((result, semantic_score))
        
        return semantic_results
    
    async def _keyword_search(
        self,
        query: str,
        entity_types: Optional[List[MoragEntityType]],
        limit: int
    ) -> List[Tuple[SearchResult, float]]:
        """Perform keyword-based search (BM25-style)."""
        # Build keyword search query
        search_query = query
        
        # Add entity type filters
        if entity_types:
            type_filters = [f"type:{et.value}" for et in entity_types]
            search_query += f" AND ({' OR '.join(type_filters)})"
        
        # Use Graphiti's keyword search capabilities
        results, _ = await self.base_search.search(
            query=search_query,
            limit=limit * 2,
            search_type="keyword"
        )
        
        # Calculate keyword scores based on term frequency and content length
        keyword_results = []
        query_terms = query.lower().split()
        
        for result in results:
            keyword_score = self._calculate_keyword_score(result.content, query_terms)
            if keyword_score >= self.config.min_keyword_score:
                keyword_results.append((result, keyword_score))
        
        return keyword_results
    
    async def _graph_traversal_search(
        self,
        query: str,
        entity_types: Optional[List[MoragEntityType]],
        limit: int
    ) -> List[Tuple[SearchResult, float]]:
        """Perform graph traversal search for relationship discovery."""
        # First, find entities mentioned in the query
        query_entities = await self._extract_entities_from_query(query)
        
        if not query_entities:
            return []
        
        # Perform graph traversal from query entities
        traversal_results = []
        
        for entity_id in query_entities:
            # Get related entities through graph traversal
            related_entities = await self._traverse_entity_graph(
                entity_id, 
                max_depth=self.config.max_graph_depth,
                entity_types=entity_types
            )
            
            # Search for content related to these entities
            for related_entity_id, distance, relationship_strength in related_entities:
                entity_results = await self._search_entity_content(related_entity_id)
                
                for result in entity_results:
                    # Calculate graph score based on distance and relationship strength
                    graph_score = self._calculate_graph_score(distance, relationship_strength)
                    if graph_score >= self.config.min_graph_score:
                        traversal_results.append((result, graph_score))
        
        # Remove duplicates and sort by graph score
        unique_results = self._deduplicate_results(traversal_results)
        unique_results.sort(key=lambda x: x[1], reverse=True)
        
        return unique_results[:limit]

    async def _combine_and_rank_results(
        self,
        semantic_results: List[Tuple[SearchResult, float]],
        keyword_results: List[Tuple[SearchResult, float]],
        graph_results: List[Tuple[SearchResult, float]],
        query: str,
        include_explanation: bool
    ) -> List[EnhancedSearchResult]:
        """Combine and rank results from different search methods."""
        # Create a mapping of content to scores
        result_scores: Dict[str, Dict[str, float]] = {}
        result_objects: Dict[str, SearchResult] = {}

        # Process semantic results
        for result, score in semantic_results:
            content_key = self._create_content_key(result)
            result_scores[content_key] = result_scores.get(content_key, {})
            result_scores[content_key]['semantic'] = score
            result_objects[content_key] = result

        # Process keyword results
        for result, score in keyword_results:
            content_key = self._create_content_key(result)
            result_scores[content_key] = result_scores.get(content_key, {})
            result_scores[content_key]['keyword'] = score
            result_objects[content_key] = result

        # Process graph results
        for result, score in graph_results:
            content_key = self._create_content_key(result)
            result_scores[content_key] = result_scores.get(content_key, {})
            result_scores[content_key]['graph'] = score
            result_objects[content_key] = result

        # Calculate combined scores
        enhanced_results = []

        for content_key, scores in result_scores.items():
            result = result_objects[content_key]

            semantic_score = scores.get('semantic', 0.0)
            keyword_score = scores.get('keyword', 0.0)
            graph_score = scores.get('graph', 0.0)

            # Calculate weighted combined score
            combined_score = (
                semantic_score * self.config.semantic_weight +
                keyword_score * self.config.keyword_weight +
                graph_score * self.config.graph_weight
            )

            # Create explanation if requested
            explanation = None
            if include_explanation:
                explanation = self._create_search_explanation(
                    semantic_score, keyword_score, graph_score, combined_score
                )

            # Extract entity IDs from metadata
            entity_ids = self._extract_entity_ids(result)

            enhanced_result = EnhancedSearchResult(
                content=result.content,
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                graph_score=graph_score,
                combined_score=combined_score,
                document_id=result.document_id,
                chunk_id=result.chunk_id,
                entity_ids=entity_ids,
                metadata=result.metadata,
                search_method="hybrid",
                explanation=explanation
            )

            enhanced_results.append(enhanced_result)

        # Sort by combined score
        enhanced_results.sort(key=lambda x: x.combined_score, reverse=True)

        return enhanced_results

    def _calculate_keyword_score(self, content: str, query_terms: List[str]) -> float:
        """Calculate keyword-based relevance score."""
        content_lower = content.lower()
        content_words = content_lower.split()

        if not content_words:
            return 0.0

        # Calculate term frequency
        term_frequencies = {}
        for term in query_terms:
            term_frequencies[term] = content_lower.count(term)

        # Calculate BM25-like score
        total_terms = sum(term_frequencies.values())
        content_length = len(content_words)

        # Simple BM25 approximation
        score = 0.0
        for term, freq in term_frequencies.items():
            if freq > 0:
                tf = freq / content_length
                score += tf / (tf + 0.5)  # Simplified BM25 term

        return min(1.0, score / len(query_terms))

    def _calculate_graph_score(self, distance: int, relationship_strength: float) -> float:
        """Calculate graph-based relevance score."""
        # Score decreases with distance and increases with relationship strength
        distance_penalty = 1.0 / (distance + 1)
        return distance_penalty * relationship_strength

    async def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entity IDs mentioned in the query."""
        # Search for entities that match query terms
        entity_results, _ = await self.base_search.search(
            query=f"{query} adapter_type:entity",
            limit=10
        )

        entity_ids = []
        for result in entity_results:
            metadata = result.metadata or {}
            entity_id = metadata.get('id')
            if entity_id:
                entity_ids.append(entity_id)

        return entity_ids

    async def _traverse_entity_graph(
        self,
        start_entity_id: str,
        max_depth: int,
        entity_types: Optional[List[MoragEntityType]] = None
    ) -> List[Tuple[str, int, float]]:
        """Traverse entity graph to find related entities."""
        visited = set()
        to_visit = [(start_entity_id, 0, 1.0)]  # (entity_id, depth, strength)
        related_entities = []

        while to_visit and len(related_entities) < 50:  # Limit traversal
            current_entity, depth, strength = to_visit.pop(0)

            if current_entity in visited or depth > max_depth:
                continue

            visited.add(current_entity)

            if depth > 0:  # Don't include the starting entity
                related_entities.append((current_entity, depth, strength))

            # Find relations involving this entity
            relation_results, _ = await self.base_search.search(
                query=f"(source_entity_id:{current_entity} OR target_entity_id:{current_entity}) adapter_type:relation",
                limit=20
            )

            for result in relation_results:
                metadata = result.metadata or {}
                source_id = metadata.get('source_entity_id')
                target_id = metadata.get('target_entity_id')
                confidence = metadata.get('confidence', 0.5)

                # Add connected entities to traversal queue
                next_entity = target_id if source_id == current_entity else source_id
                if next_entity and next_entity not in visited:
                    new_strength = strength * confidence * 0.8  # Decay with distance
                    to_visit.append((next_entity, depth + 1, new_strength))

        return related_entities

    async def _search_entity_content(self, entity_id: str) -> List[SearchResult]:
        """Search for content related to a specific entity."""
        # Search for episodes mentioning this entity
        results, _ = await self.base_search.search(
            query=f"entity_ids:{entity_id} OR id:{entity_id}",
            limit=10
        )

        return results

    def _deduplicate_results(self, results: List[Tuple[SearchResult, float]]) -> List[Tuple[SearchResult, float]]:
        """Remove duplicate results based on content similarity."""
        seen_content = set()
        unique_results = []

        for result, score in results:
            content_key = self._create_content_key(result)
            if content_key not in seen_content:
                seen_content.add(content_key)
                unique_results.append((result, score))

        return unique_results

    def _create_content_key(self, result: SearchResult) -> str:
        """Create a unique key for result content."""
        # Use chunk_id if available, otherwise use content hash
        if result.chunk_id:
            return f"chunk:{result.chunk_id}"
        elif result.document_id:
            return f"doc:{result.document_id}:{hash(result.content[:100])}"
        else:
            return f"content:{hash(result.content)}"

    def _create_cache_key(
        self,
        query: str,
        search_methods: List[SearchMethod],
        entity_types: Optional[List[MoragEntityType]],
        limit: int
    ) -> str:
        """Create cache key for search parameters."""
        methods_str = ",".join(sorted([m.value for m in search_methods]))
        types_str = ",".join(sorted([t.value for t in entity_types])) if entity_types else "all"
        return f"{hash(query)}:{methods_str}:{types_str}:{limit}"

    def _extract_entity_ids(self, result: SearchResult) -> List[str]:
        """Extract entity IDs from search result metadata."""
        metadata = result.metadata or {}

        # Check for entity IDs in various metadata fields
        entity_ids = []

        # Direct entity ID
        if metadata.get('adapter_type') == 'entity':
            entity_id = metadata.get('id')
            if entity_id:
                entity_ids.append(entity_id)

        # Entity IDs from chunk-entity mappings
        if 'entity_ids' in metadata:
            entity_ids.extend(metadata['entity_ids'])

        # Entity IDs from relations
        if metadata.get('adapter_type') == 'relation':
            source_id = metadata.get('source_entity_id')
            target_id = metadata.get('target_entity_id')
            if source_id:
                entity_ids.append(source_id)
            if target_id:
                entity_ids.append(target_id)

        return list(set(entity_ids))  # Remove duplicates

    def _create_search_explanation(
        self,
        semantic_score: float,
        keyword_score: float,
        graph_score: float,
        combined_score: float
    ) -> str:
        """Create human-readable explanation of search scoring."""
        explanations = []

        if semantic_score > 0:
            explanations.append(f"Semantic similarity: {semantic_score:.2f}")

        if keyword_score > 0:
            explanations.append(f"Keyword match: {keyword_score:.2f}")

        if graph_score > 0:
            explanations.append(f"Graph relevance: {graph_score:.2f}")

        explanation = " | ".join(explanations)
        explanation += f" | Combined: {combined_score:.2f}"

        return explanation

    def clear_cache(self):
        """Clear search result cache."""
        self._search_cache.clear()
        self._entity_cache.clear()
        logger.info("Search cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "search_cache_size": len(self._search_cache),
            "entity_cache_size": len(self._entity_cache),
            "cache_enabled": self.config.enable_caching
        }


# Convenience function for creating hybrid search service
def create_hybrid_search_service(
    base_search_service: GraphitiSearchService,
    entity_storage: GraphitiEntityStorage,
    config: Optional[SearchConfiguration] = None
) -> EnhancedHybridSearchService:
    """Create an enhanced hybrid search service.

    Args:
        base_search_service: Base Graphiti search service
        entity_storage: Entity storage service
        config: Optional search configuration

    Returns:
        EnhancedHybridSearchService instance
    """
    return EnhancedHybridSearchService(base_search_service, entity_storage, config)
