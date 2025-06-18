# Task 4.1: Hybrid Retrieval Pipeline

## Overview

Implement a comprehensive hybrid retrieval pipeline that seamlessly combines Qdrant's vector search capabilities with Neo4j's graph traversal to provide contextually rich, semantically accurate, and relationship-aware search results for the RAG system.

## Objectives

- Create unified retrieval interface combining vector and graph search
- Implement intelligent query routing and result fusion
- Enable context-aware result ranking and filtering
- Establish performance optimization and caching strategies
- Provide comprehensive retrieval analytics and monitoring

## Current State Analysis

### Existing Capabilities

**Qdrant**:
- High-performance vector similarity search
- Payload filtering and hybrid search
- Batch operations and streaming
- Limited contextual understanding

**Neo4j**:
- Rich graph traversal and relationship queries
- Entity relationship exploration
- Context-aware filtering
- Limited vector similarity capabilities

**Integration Gaps**:
- No unified retrieval interface
- Separate result sets requiring manual fusion
- No intelligent query routing
- Missing context-aware ranking

## Implementation Plan

### Step 1: Core Hybrid Retrieval Engine

Implement `src/morag_graph/retrieval/hybrid_retriever.py`:

```python
from typing import Dict, List, Optional, Any, Tuple, Union
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest
from neo4j import AsyncSession
import numpy as np

class RetrievalStrategy(Enum):
    VECTOR_ONLY = "vector_only"
    GRAPH_ONLY = "graph_only"
    HYBRID_PARALLEL = "hybrid_parallel"
    HYBRID_SEQUENTIAL = "hybrid_sequential"
    ADAPTIVE = "adaptive"

class QueryType(Enum):
    SEMANTIC_SEARCH = "semantic_search"
    ENTITY_LOOKUP = "entity_lookup"
    RELATIONSHIP_EXPLORATION = "relationship_exploration"
    CONTEXTUAL_SEARCH = "contextual_search"
    MULTI_HOP_REASONING = "multi_hop_reasoning"

@dataclass
class RetrievalQuery:
    """Unified query structure for hybrid retrieval."""
    query_text: str
    query_vector: Optional[List[float]] = None
    query_type: QueryType = QueryType.SEMANTIC_SEARCH
    strategy: RetrievalStrategy = RetrievalStrategy.ADAPTIVE
    
    # Vector search parameters
    vector_limit: int = 20
    vector_threshold: float = 0.7
    vector_filters: Optional[Dict[str, Any]] = None
    
    # Graph search parameters
    graph_limit: int = 20
    entity_types: Optional[List[str]] = None
    relationship_types: Optional[List[str]] = None
    max_depth: int = 3
    
    # Fusion parameters
    fusion_method: str = "rrf"  # reciprocal rank fusion
    vector_weight: float = 0.6
    graph_weight: float = 0.4
    context_boost: float = 0.2
    
    # Result parameters
    final_limit: int = 10
    include_context: bool = True
    include_relationships: bool = True
    
    # Metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RetrievalResult:
    """Unified result structure from hybrid retrieval."""
    result_id: str
    content: str
    score: float
    source_type: str  # 'vector', 'graph', 'hybrid'
    
    # Content metadata
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None
    entity_id: Optional[str] = None
    
    # Scoring details
    vector_score: Optional[float] = None
    graph_score: Optional[float] = None
    context_score: Optional[float] = None
    fusion_score: Optional[float] = None
    
    # Context information
    entities: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    graph_context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    retrieval_path: List[str] = field(default_factory=list)
    processing_time: Optional[float] = None
    confidence: Optional[float] = None

@dataclass
class RetrievalMetrics:
    """Metrics for retrieval performance analysis."""
    total_time: float
    vector_time: Optional[float] = None
    graph_time: Optional[float] = None
    fusion_time: Optional[float] = None
    
    vector_results_count: int = 0
    graph_results_count: int = 0
    final_results_count: int = 0
    
    cache_hits: int = 0
    cache_misses: int = 0
    
    strategy_used: Optional[RetrievalStrategy] = None
    query_complexity: Optional[str] = None

class HybridRetriever:
    """Main hybrid retrieval engine combining vector and graph search."""
    
    def __init__(self, 
                 neo4j_session: AsyncSession, 
                 qdrant_client: QdrantClient,
                 collection_name: str = "morag_vectors"):
        self.neo4j_session = neo4j_session
        self.qdrant_client = qdrant_client
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)
        
        # Performance optimization
        self.result_cache = {}
        self.query_stats = {}
        
        # Strategy selection weights
        self.strategy_weights = {
            "vector_preference": 0.6,
            "graph_preference": 0.4,
            "complexity_threshold": 0.7
        }
    
    async def retrieve(self, query: RetrievalQuery) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """Main retrieval method that orchestrates hybrid search."""
        start_time = datetime.now()
        
        try:
            # Determine optimal strategy if adaptive
            if query.strategy == RetrievalStrategy.ADAPTIVE:
                query.strategy = await self._select_optimal_strategy(query)
            
            # Execute retrieval based on strategy
            if query.strategy == RetrievalStrategy.VECTOR_ONLY:
                results, metrics = await self._vector_only_retrieval(query)
            
            elif query.strategy == RetrievalStrategy.GRAPH_ONLY:
                results, metrics = await self._graph_only_retrieval(query)
            
            elif query.strategy == RetrievalStrategy.HYBRID_PARALLEL:
                results, metrics = await self._hybrid_parallel_retrieval(query)
            
            elif query.strategy == RetrievalStrategy.HYBRID_SEQUENTIAL:
                results, metrics = await self._hybrid_sequential_retrieval(query)
            
            else:
                # Default to parallel hybrid
                results, metrics = await self._hybrid_parallel_retrieval(query)
            
            # Post-process results
            results = await self._post_process_results(results, query)
            
            # Update metrics
            total_time = (datetime.now() - start_time).total_seconds()
            metrics.total_time = total_time
            metrics.strategy_used = query.strategy
            metrics.final_results_count = len(results)
            
            # Update query statistics
            await self._update_query_stats(query, metrics)
            
            return results, metrics
            
        except Exception as e:
            self.logger.error(f"Hybrid retrieval failed: {e}")
            return [], RetrievalMetrics(total_time=0.0)
    
    async def _select_optimal_strategy(self, query: RetrievalQuery) -> RetrievalStrategy:
        """Intelligently select the optimal retrieval strategy."""
        try:
            # Analyze query characteristics
            query_analysis = await self._analyze_query(query)
            
            # Check for entity mentions
            has_entities = query_analysis.get("entity_count", 0) > 0
            
            # Check for relationship keywords
            has_relationships = query_analysis.get("relationship_keywords", 0) > 0
            
            # Check query complexity
            complexity_score = query_analysis.get("complexity_score", 0.0)
            
            # Check for semantic similarity requirements
            is_semantic = query.query_type in [QueryType.SEMANTIC_SEARCH, QueryType.CONTEXTUAL_SEARCH]
            
            # Decision logic
            if query.query_type == QueryType.ENTITY_LOOKUP:
                return RetrievalStrategy.GRAPH_ONLY
            
            elif query.query_type == QueryType.RELATIONSHIP_EXPLORATION:
                return RetrievalStrategy.GRAPH_ONLY
            
            elif query.query_type == QueryType.MULTI_HOP_REASONING:
                return RetrievalStrategy.HYBRID_SEQUENTIAL
            
            elif is_semantic and not has_entities and not has_relationships:
                return RetrievalStrategy.VECTOR_ONLY
            
            elif complexity_score > self.strategy_weights["complexity_threshold"]:
                return RetrievalStrategy.HYBRID_SEQUENTIAL
            
            else:
                return RetrievalStrategy.HYBRID_PARALLEL
            
        except Exception as e:
            self.logger.error(f"Strategy selection failed: {e}")
            return RetrievalStrategy.HYBRID_PARALLEL
    
    async def _vector_only_retrieval(self, query: RetrievalQuery) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """Perform vector-only retrieval."""
        start_time = datetime.now()
        
        try:
            if not query.query_vector:
                self.logger.warning("No query vector provided for vector-only retrieval")
                return [], RetrievalMetrics(total_time=0.0)
            
            # Build Qdrant filter
            qdrant_filter = None
            if query.vector_filters:
                qdrant_filter = self._build_qdrant_filter(query.vector_filters)
            
            # Perform vector search
            search_results = await self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query.query_vector,
                query_filter=qdrant_filter,
                limit=query.vector_limit,
                score_threshold=query.vector_threshold,
                with_payload=True
            )
            
            # Convert to RetrievalResult objects
            results = []
            for result in search_results:
                retrieval_result = RetrievalResult(
                    result_id=str(result.id),
                    content=result.payload.get("text", ""),
                    score=result.score,
                    source_type="vector",
                    document_id=result.payload.get("document_id"),
                    chunk_id=result.payload.get("chunk_id"),
                    entity_id=result.payload.get("entity_id"),
                    vector_score=result.score,
                    retrieval_path=["vector_search"]
                )
                
                # Add context if requested
                if query.include_context:
                    await self._enrich_with_context(retrieval_result)
                
                results.append(retrieval_result)
            
            vector_time = (datetime.now() - start_time).total_seconds()
            metrics = RetrievalMetrics(
                total_time=vector_time,
                vector_time=vector_time,
                vector_results_count=len(results)
            )
            
            return results[:query.final_limit], metrics
            
        except Exception as e:
            self.logger.error(f"Vector-only retrieval failed: {e}")
            return [], RetrievalMetrics(total_time=0.0)
    
    async def _graph_only_retrieval(self, query: RetrievalQuery) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """Perform graph-only retrieval."""
        start_time = datetime.now()
        
        try:
            # Build graph query based on query type
            if query.query_type == QueryType.ENTITY_LOOKUP:
                results = await self._entity_lookup_query(query)
            
            elif query.query_type == QueryType.RELATIONSHIP_EXPLORATION:
                results = await self._relationship_exploration_query(query)
            
            else:
                # General graph search
                results = await self._general_graph_search(query)
            
            graph_time = (datetime.now() - start_time).total_seconds()
            metrics = RetrievalMetrics(
                total_time=graph_time,
                graph_time=graph_time,
                graph_results_count=len(results)
            )
            
            return results[:query.final_limit], metrics
            
        except Exception as e:
            self.logger.error(f"Graph-only retrieval failed: {e}")
            return [], RetrievalMetrics(total_time=0.0)
    
    async def _hybrid_parallel_retrieval(self, query: RetrievalQuery) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """Perform parallel hybrid retrieval."""
        start_time = datetime.now()
        
        try:
            # Execute vector and graph searches in parallel
            vector_task = asyncio.create_task(self._vector_only_retrieval(query))
            graph_task = asyncio.create_task(self._graph_only_retrieval(query))
            
            # Wait for both to complete
            (vector_results, vector_metrics), (graph_results, graph_metrics) = await asyncio.gather(
                vector_task, graph_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(vector_results, Exception):
                self.logger.error(f"Vector search failed: {vector_results}")
                vector_results, vector_metrics = [], RetrievalMetrics(total_time=0.0)
            
            if isinstance(graph_results, Exception):
                self.logger.error(f"Graph search failed: {graph_results}")
                graph_results, graph_metrics = [], RetrievalMetrics(total_time=0.0)
            
            # Fuse results
            fusion_start = datetime.now()
            fused_results = await self._fuse_results(
                vector_results, graph_results, query
            )
            fusion_time = (datetime.now() - fusion_start).total_seconds()
            
            # Combine metrics
            total_time = (datetime.now() - start_time).total_seconds()
            metrics = RetrievalMetrics(
                total_time=total_time,
                vector_time=vector_metrics.vector_time,
                graph_time=graph_metrics.graph_time,
                fusion_time=fusion_time,
                vector_results_count=len(vector_results),
                graph_results_count=len(graph_results)
            )
            
            return fused_results[:query.final_limit], metrics
            
        except Exception as e:
            self.logger.error(f"Parallel hybrid retrieval failed: {e}")
            return [], RetrievalMetrics(total_time=0.0)
    
    async def _hybrid_sequential_retrieval(self, query: RetrievalQuery) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """Perform sequential hybrid retrieval (vector first, then graph expansion)."""
        start_time = datetime.now()
        
        try:
            # Step 1: Vector search
            vector_results, vector_metrics = await self._vector_only_retrieval(query)
            
            # Step 2: Graph expansion of vector results
            graph_start = datetime.now()
            expanded_results = []
            
            for result in vector_results:
                # Expand each result with graph context
                if result.entity_id:
                    graph_expansion = await self._expand_with_graph_context(
                        result, query
                    )
                    expanded_results.extend(graph_expansion)
                else:
                    expanded_results.append(result)
            
            graph_time = (datetime.now() - graph_start).total_seconds()
            
            # Step 3: Re-rank combined results
            rerank_start = datetime.now()
            reranked_results = await self._rerank_results(expanded_results, query)
            rerank_time = (datetime.now() - rerank_start).total_seconds()
            
            # Combine metrics
            total_time = (datetime.now() - start_time).total_seconds()
            metrics = RetrievalMetrics(
                total_time=total_time,
                vector_time=vector_metrics.vector_time,
                graph_time=graph_time,
                fusion_time=rerank_time,
                vector_results_count=len(vector_results),
                graph_results_count=len(expanded_results) - len(vector_results)
            )
            
            return reranked_results[:query.final_limit], metrics
            
        except Exception as e:
            self.logger.error(f"Sequential hybrid retrieval failed: {e}")
            return [], RetrievalMetrics(total_time=0.0)
    
    async def _fuse_results(self, 
                          vector_results: List[RetrievalResult], 
                          graph_results: List[RetrievalResult], 
                          query: RetrievalQuery) -> List[RetrievalResult]:
        """Fuse vector and graph results using specified fusion method."""
        try:
            if query.fusion_method == "rrf":
                return await self._reciprocal_rank_fusion(vector_results, graph_results, query)
            
            elif query.fusion_method == "weighted_sum":
                return await self._weighted_sum_fusion(vector_results, graph_results, query)
            
            elif query.fusion_method == "interleave":
                return await self._interleave_fusion(vector_results, graph_results, query)
            
            else:
                # Default to RRF
                return await self._reciprocal_rank_fusion(vector_results, graph_results, query)
            
        except Exception as e:
            self.logger.error(f"Result fusion failed: {e}")
            return vector_results + graph_results
    
    async def _reciprocal_rank_fusion(self, 
                                    vector_results: List[RetrievalResult], 
                                    graph_results: List[RetrievalResult], 
                                    query: RetrievalQuery) -> List[RetrievalResult]:
        """Fuse results using Reciprocal Rank Fusion (RRF)."""
        k = 60  # RRF parameter
        result_scores = {}
        all_results = {}
        
        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            rrf_score = query.vector_weight / (k + rank)
            result_scores[result.result_id] = result_scores.get(result.result_id, 0) + rrf_score
            all_results[result.result_id] = result
            result.fusion_score = rrf_score
        
        # Process graph results
        for rank, result in enumerate(graph_results, 1):
            rrf_score = query.graph_weight / (k + rank)
            result_scores[result.result_id] = result_scores.get(result.result_id, 0) + rrf_score
            
            if result.result_id in all_results:
                # Merge with existing result
                existing = all_results[result.result_id]
                existing.graph_score = result.score
                existing.source_type = "hybrid"
                existing.retrieval_path.extend(result.retrieval_path)
                existing.fusion_score = result_scores[result.result_id]
            else:
                all_results[result.result_id] = result
                result.fusion_score = rrf_score
        
        # Sort by fused score
        sorted_results = sorted(
            all_results.values(),
            key=lambda x: result_scores[x.result_id],
            reverse=True
        )
        
        # Update final scores
        for result in sorted_results:
            result.score = result_scores[result.result_id]
        
        return sorted_results
    
    async def _weighted_sum_fusion(self, 
                                 vector_results: List[RetrievalResult], 
                                 graph_results: List[RetrievalResult], 
                                 query: RetrievalQuery) -> List[RetrievalResult]:
        """Fuse results using weighted sum of normalized scores."""
        # Normalize vector scores
        if vector_results:
            max_vector_score = max(r.score for r in vector_results)
            for result in vector_results:
                result.vector_score = result.score / max_vector_score if max_vector_score > 0 else 0
        
        # Normalize graph scores
        if graph_results:
            max_graph_score = max(r.score for r in graph_results)
            for result in graph_results:
                result.graph_score = result.score / max_graph_score if max_graph_score > 0 else 0
        
        # Combine results
        all_results = {}
        
        for result in vector_results:
            weighted_score = query.vector_weight * (result.vector_score or 0)
            result.fusion_score = weighted_score
            result.score = weighted_score
            all_results[result.result_id] = result
        
        for result in graph_results:
            weighted_score = query.graph_weight * (result.graph_score or 0)
            
            if result.result_id in all_results:
                existing = all_results[result.result_id]
                existing.graph_score = result.graph_score
                existing.source_type = "hybrid"
                existing.fusion_score = (existing.fusion_score or 0) + weighted_score
                existing.score = existing.fusion_score
                existing.retrieval_path.extend(result.retrieval_path)
            else:
                result.fusion_score = weighted_score
                result.score = weighted_score
                all_results[result.result_id] = result
        
        # Sort by fused score
        return sorted(all_results.values(), key=lambda x: x.score, reverse=True)
    
    async def _interleave_fusion(self, 
                               vector_results: List[RetrievalResult], 
                               graph_results: List[RetrievalResult], 
                               query: RetrievalQuery) -> List[RetrievalResult]:
        """Fuse results by interleaving vector and graph results."""
        fused_results = []
        seen_ids = set()
        
        max_len = max(len(vector_results), len(graph_results))
        
        for i in range(max_len):
            # Add vector result if available and not seen
            if i < len(vector_results):
                result = vector_results[i]
                if result.result_id not in seen_ids:
                    result.source_type = "vector"
                    fused_results.append(result)
                    seen_ids.add(result.result_id)
            
            # Add graph result if available and not seen
            if i < len(graph_results):
                result = graph_results[i]
                if result.result_id not in seen_ids:
                    result.source_type = "graph"
                    fused_results.append(result)
                    seen_ids.add(result.result_id)
        
        return fused_results
    
    # Helper methods for query analysis and processing
    
    async def _analyze_query(self, query: RetrievalQuery) -> Dict[str, Any]:
        """Analyze query characteristics for strategy selection."""
        analysis = {
            "entity_count": 0,
            "relationship_keywords": 0,
            "complexity_score": 0.0
        }
        
        # Simple keyword-based analysis (can be enhanced with NLP)
        text = query.query_text.lower()
        
        # Count potential entity mentions (capitalized words)
        import re
        entities = re.findall(r'\b[A-Z][a-z]+\b', query.query_text)
        analysis["entity_count"] = len(entities)
        
        # Count relationship keywords
        relationship_keywords = [
            "related", "connected", "associated", "linked", "between", 
            "relationship", "connection", "works with", "knows", "part of"
        ]
        
        for keyword in relationship_keywords:
            if keyword in text:
                analysis["relationship_keywords"] += 1
        
        # Calculate complexity score
        word_count = len(text.split())
        analysis["complexity_score"] = min(1.0, (
            0.3 * (word_count / 20) +  # Length factor
            0.4 * (analysis["entity_count"] / 5) +  # Entity factor
            0.3 * (analysis["relationship_keywords"] / 3)  # Relationship factor
        ))
        
        return analysis
    
    async def _enrich_with_context(self, result: RetrievalResult):
        """Enrich result with graph context information."""
        try:
            if result.entity_id:
                # Get entity information
                entity_query = """
                MATCH (e:Entity {id: $entity_id})
                OPTIONAL MATCH (e)-[r]-(connected:Entity)
                RETURN e.name as name, e.type as type,
                       collect({type: type(r), entity: connected.name, entity_type: connected.type}) as relationships
                LIMIT 10
                """
                
                entity_result = await self.neo4j_session.run(entity_query, entity_id=result.entity_id)
                entity_record = await entity_result.single()
                
                if entity_record:
                    result.graph_context = {
                        "entity_name": entity_record["name"],
                        "entity_type": entity_record["type"]
                    }
                    result.relationships = entity_record["relationships"] or []
            
        except Exception as e:
            self.logger.error(f"Failed to enrich result with context: {e}")
    
    def _build_qdrant_filter(self, filters: Dict[str, Any]) -> Filter:
        """Build Qdrant filter from filter dictionary."""
        conditions = []
        
        for key, value in filters.items():
            if isinstance(value, list):
                # Multiple values - use should (OR)
                for v in value:
                    conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=v))
                    )
            else:
                # Single value
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        
        return Filter(must=conditions) if conditions else None
    
    async def _update_query_stats(self, query: RetrievalQuery, metrics: RetrievalMetrics):
        """Update query statistics for performance monitoring."""
        try:
            query_key = f"{query.query_type.value}_{query.strategy.value}"
            
            if query_key not in self.query_stats:
                self.query_stats[query_key] = {
                    "count": 0,
                    "total_time": 0.0,
                    "avg_time": 0.0,
                    "avg_results": 0.0
                }
            
            stats = self.query_stats[query_key]
            stats["count"] += 1
            stats["total_time"] += metrics.total_time
            stats["avg_time"] = stats["total_time"] / stats["count"]
            stats["avg_results"] = (
                (stats["avg_results"] * (stats["count"] - 1) + metrics.final_results_count) / 
                stats["count"]
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update query stats: {e}")
```

### Step 2: Query Coordination and Optimization

Implement `src/morag_graph/retrieval/query_coordinator.py`:

```python
from typing import Dict, List, Optional, Any, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import hashlib
import json

from .hybrid_retriever import HybridRetriever, RetrievalQuery, RetrievalResult, RetrievalMetrics

@dataclass
class CacheEntry:
    """Cache entry for retrieval results."""
    results: List[RetrievalResult]
    metrics: RetrievalMetrics
    timestamp: datetime
    query_hash: str
    ttl_seconds: int = 3600  # 1 hour default TTL
    
    def is_expired(self) -> bool:
        return datetime.now() > self.timestamp + timedelta(seconds=self.ttl_seconds)

class QueryCoordinator:
    """Coordinates and optimizes retrieval queries with caching and batching."""
    
    def __init__(self, hybrid_retriever: HybridRetriever):
        self.hybrid_retriever = hybrid_retriever
        self.logger = logging.getLogger(__name__)
        
        # Caching
        self.result_cache: Dict[str, CacheEntry] = {}
        self.cache_stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0
        }
        
        # Query batching
        self.pending_queries: Dict[str, List[Tuple[RetrievalQuery, asyncio.Future]]] = {}
        self.batch_timeout = 0.1  # 100ms batch window
        
        # Performance monitoring
        self.performance_stats = {
            "total_queries": 0,
            "cache_hit_rate": 0.0,
            "avg_response_time": 0.0,
            "total_response_time": 0.0
        }
    
    async def coordinate_retrieval(self, query: RetrievalQuery) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """Main coordination method with caching and optimization."""
        start_time = datetime.now()
        
        try:
            # Generate query hash for caching
            query_hash = self._generate_query_hash(query)
            
            # Check cache first
            cached_result = self._get_cached_result(query_hash)
            if cached_result:
                self.cache_stats["hits"] += 1
                self._update_performance_stats(start_time, True)
                return cached_result.results, cached_result.metrics
            
            self.cache_stats["misses"] += 1
            
            # Check if we can batch this query
            if self._should_batch_query(query):
                results, metrics = await self._batch_query(query, query_hash)
            else:
                # Execute immediately
                results, metrics = await self.hybrid_retriever.retrieve(query)
            
            # Cache the results
            self._cache_result(query_hash, results, metrics, query)
            
            # Update performance stats
            self._update_performance_stats(start_time, False)
            
            return results, metrics
            
        except Exception as e:
            self.logger.error(f"Query coordination failed: {e}")
            return [], RetrievalMetrics(total_time=0.0)
    
    async def batch_coordinate_retrieval(self, queries: List[RetrievalQuery]) -> List[Tuple[List[RetrievalResult], RetrievalMetrics]]:
        """Coordinate multiple queries with optimized batching."""
        try:
            # Group queries by similarity for potential batching
            query_groups = self._group_similar_queries(queries)
            
            # Execute groups in parallel
            tasks = []
            for group in query_groups:
                if len(group) == 1:
                    # Single query
                    task = self.coordinate_retrieval(group[0])
                else:
                    # Batch execution
                    task = self._execute_query_batch(group)
                tasks.append(task)
            
            # Wait for all groups to complete
            group_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten results
            all_results = []
            for group_result in group_results:
                if isinstance(group_result, Exception):
                    self.logger.error(f"Batch execution failed: {group_result}")
                    all_results.append(([], RetrievalMetrics(total_time=0.0)))
                elif isinstance(group_result, list):
                    all_results.extend(group_result)
                else:
                    all_results.append(group_result)
            
            return all_results
            
        except Exception as e:
            self.logger.error(f"Batch coordination failed: {e}")
            return [([], RetrievalMetrics(total_time=0.0))] * len(queries)
    
    def _generate_query_hash(self, query: RetrievalQuery) -> str:
        """Generate a hash for query caching."""
        # Create a normalized representation of the query
        cache_key_data = {
            "query_text": query.query_text,
            "query_type": query.query_type.value,
            "strategy": query.strategy.value,
            "vector_limit": query.vector_limit,
            "graph_limit": query.graph_limit,
            "final_limit": query.final_limit,
            "vector_threshold": query.vector_threshold,
            "entity_types": sorted(query.entity_types) if query.entity_types else None,
            "relationship_types": sorted(query.relationship_types) if query.relationship_types else None,
            "vector_filters": query.vector_filters
        }
        
        # Convert to JSON string and hash
        cache_key_str = json.dumps(cache_key_data, sort_keys=True)
        return hashlib.md5(cache_key_str.encode()).hexdigest()
    
    def _get_cached_result(self, query_hash: str) -> Optional[CacheEntry]:
        """Get cached result if available and not expired."""
        if query_hash in self.result_cache:
            entry = self.result_cache[query_hash]
            if not entry.is_expired():
                return entry
            else:
                # Remove expired entry
                del self.result_cache[query_hash]
                self.cache_stats["evictions"] += 1
        
        return None
    
    def _cache_result(self, query_hash: str, results: List[RetrievalResult], 
                     metrics: RetrievalMetrics, query: RetrievalQuery):
        """Cache retrieval results."""
        try:
            # Determine TTL based on query type
            ttl = self._calculate_cache_ttl(query)
            
            entry = CacheEntry(
                results=results,
                metrics=metrics,
                timestamp=datetime.now(),
                query_hash=query_hash,
                ttl_seconds=ttl
            )
            
            self.result_cache[query_hash] = entry
            
            # Clean up old entries if cache is getting large
            if len(self.result_cache) > 1000:
                self._cleanup_cache()
            
        except Exception as e:
            self.logger.error(f"Failed to cache result: {e}")
    
    def _calculate_cache_ttl(self, query: RetrievalQuery) -> int:
        """Calculate appropriate TTL for query results."""
        # Different TTLs based on query characteristics
        if query.query_type in ["entity_lookup", "relationship_exploration"]:
            return 7200  # 2 hours for entity/relationship queries
        elif query.query_type == "semantic_search":
            return 3600  # 1 hour for semantic searches
        else:
            return 1800  # 30 minutes for other queries
    
    def _should_batch_query(self, query: RetrievalQuery) -> bool:
        """Determine if query should be batched."""
        # For now, don't batch complex queries
        return query.query_type in ["semantic_search", "entity_lookup"]
    
    async def _batch_query(self, query: RetrievalQuery, query_hash: str) -> Tuple[List[RetrievalResult], RetrievalMetrics]:
        """Handle batched query execution."""
        # Create a future for this query
        future = asyncio.Future()
        
        # Group by query type for batching
        batch_key = f"{query.query_type.value}_{query.strategy.value}"
        
        if batch_key not in self.pending_queries:
            self.pending_queries[batch_key] = []
        
        self.pending_queries[batch_key].append((query, future))
        
        # If this is the first query in the batch, start the timer
        if len(self.pending_queries[batch_key]) == 1:
            asyncio.create_task(self._process_batch_after_timeout(batch_key))
        
        # Wait for the batch to be processed
        return await future
    
    async def _process_batch_after_timeout(self, batch_key: str):
        """Process a batch of queries after timeout."""
        await asyncio.sleep(self.batch_timeout)
        
        if batch_key in self.pending_queries and self.pending_queries[batch_key]:
            batch = self.pending_queries[batch_key]
            del self.pending_queries[batch_key]
            
            # Execute the batch
            await self._execute_query_batch([query for query, _ in batch])
    
    async def _execute_query_batch(self, queries: List[RetrievalQuery]) -> List[Tuple[List[RetrievalResult], RetrievalMetrics]]:
        """Execute a batch of similar queries efficiently."""
        try:
            # For now, execute queries in parallel
            # In the future, this could be optimized for similar vector searches
            tasks = [self.hybrid_retriever.retrieve(query) for query in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    self.logger.error(f"Batch query failed: {result}")
                    processed_results.append(([], RetrievalMetrics(total_time=0.0)))
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            self.logger.error(f"Batch execution failed: {e}")
            return [([], RetrievalMetrics(total_time=0.0))] * len(queries)
    
    def _group_similar_queries(self, queries: List[RetrievalQuery]) -> List[List[RetrievalQuery]]:
        """Group similar queries for batch processing."""
        groups = {}
        
        for query in queries:
            # Group by query type and strategy
            group_key = f"{query.query_type.value}_{query.strategy.value}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(query)
        
        return list(groups.values())
    
    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        try:
            current_time = datetime.now()
            expired_keys = []
            
            for key, entry in self.result_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.result_cache[key]
                self.cache_stats["evictions"] += 1
            
            self.logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
            
        except Exception as e:
            self.logger.error(f"Cache cleanup failed: {e}")
    
    def _update_performance_stats(self, start_time: datetime, cache_hit: bool):
        """Update performance statistics."""
        try:
            response_time = (datetime.now() - start_time).total_seconds()
            
            self.performance_stats["total_queries"] += 1
            self.performance_stats["total_response_time"] += response_time
            self.performance_stats["avg_response_time"] = (
                self.performance_stats["total_response_time"] / 
                self.performance_stats["total_queries"]
            )
            
            total_requests = self.cache_stats["hits"] + self.cache_stats["misses"]
            if total_requests > 0:
                self.performance_stats["cache_hit_rate"] = (
                    self.cache_stats["hits"] / total_requests
                )
            
        except Exception as e:
            self.logger.error(f"Failed to update performance stats: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return {
            "performance": self.performance_stats.copy(),
            "cache": self.cache_stats.copy(),
            "cache_size": len(self.result_cache)
        }
    
    def clear_cache(self):
        """Clear the result cache."""
        self.result_cache.clear()
        self.cache_stats["evictions"] += len(self.result_cache)
        self.logger.info("Result cache cleared")
```

## Testing Strategy

### Unit Tests

Create `tests/test_hybrid_retrieval.py`:

```python
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from morag_graph.retrieval.hybrid_retriever import (
    HybridRetriever, RetrievalQuery, RetrievalResult, RetrievalStrategy, QueryType
)

@pytest.fixture
def mock_neo4j_session():
    return AsyncMock()

@pytest.fixture
def mock_qdrant_client():
    return AsyncMock()

@pytest.fixture
def hybrid_retriever(mock_neo4j_session, mock_qdrant_client):
    return HybridRetriever(mock_neo4j_session, mock_qdrant_client)

@pytest.mark.asyncio
async def test_vector_only_retrieval(hybrid_retriever, mock_qdrant_client):
    # Mock Qdrant search results
    mock_result = MagicMock()
    mock_result.id = "chunk_1"
    mock_result.score = 0.95
    mock_result.payload = {
        "text": "Test content",
        "document_id": "doc_1",
        "chunk_id": "chunk_1"
    }
    
    mock_qdrant_client.search.return_value = [mock_result]
    
    # Create test query
    query = RetrievalQuery(
        query_text="test query",
        query_vector=[0.1] * 384,
        strategy=RetrievalStrategy.VECTOR_ONLY,
        final_limit=5
    )
    
    # Test
    results, metrics = await hybrid_retriever.retrieve(query)
    
    assert len(results) == 1
    assert results[0].content == "Test content"
    assert results[0].score == 0.95
    assert results[0].source_type == "vector"
    assert metrics.vector_results_count == 1

@pytest.mark.asyncio
async def test_hybrid_parallel_retrieval(hybrid_retriever, mock_neo4j_session, mock_qdrant_client):
    # Mock vector search
    mock_vector_result = MagicMock()
    mock_vector_result.id = "chunk_1"
    mock_vector_result.score = 0.95
    mock_vector_result.payload = {
        "text": "Vector content",
        "document_id": "doc_1",
        "chunk_id": "chunk_1"
    }
    mock_qdrant_client.search.return_value = [mock_vector_result]
    
    # Mock graph search
    hybrid_retriever._general_graph_search = AsyncMock(return_value=[
        RetrievalResult(
            result_id="entity_1",
            content="Graph content",
            score=0.8,
            source_type="graph",
            entity_id="entity_1"
        )
    ])
    
    # Create test query
    query = RetrievalQuery(
        query_text="test query",
        query_vector=[0.1] * 384,
        strategy=RetrievalStrategy.HYBRID_PARALLEL,
        final_limit=5
    )
    
    # Test
    results, metrics = await hybrid_retriever.retrieve(query)
    
    assert len(results) >= 1
    assert metrics.vector_results_count >= 0
    assert metrics.graph_results_count >= 0

@pytest.mark.asyncio
async def test_reciprocal_rank_fusion(hybrid_retriever):
    # Create test results
    vector_results = [
        RetrievalResult("v1", "Vector 1", 0.9, "vector"),
        RetrievalResult("v2", "Vector 2", 0.8, "vector")
    ]
    
    graph_results = [
        RetrievalResult("g1", "Graph 1", 0.85, "graph"),
        RetrievalResult("v1", "Vector 1", 0.7, "graph")  # Same as vector result
    ]
    
    query = RetrievalQuery(
        query_text="test",
        vector_weight=0.6,
        graph_weight=0.4
    )
    
    # Test fusion
    fused = await hybrid_retriever._reciprocal_rank_fusion(vector_results, graph_results, query)
    
    assert len(fused) == 3  # v1, v2, g1
    assert any(r.result_id == "v1" and r.source_type == "hybrid" for r in fused)

@pytest.mark.asyncio
async def test_strategy_selection(hybrid_retriever):
    # Test entity lookup query
    entity_query = RetrievalQuery(
        query_text="Find John Doe",
        query_type=QueryType.ENTITY_LOOKUP,
        strategy=RetrievalStrategy.ADAPTIVE
    )
    
    strategy = await hybrid_retriever._select_optimal_strategy(entity_query)
    assert strategy == RetrievalStrategy.GRAPH_ONLY
    
    # Test semantic search query
    semantic_query = RetrievalQuery(
        query_text="machine learning algorithms",
        query_type=QueryType.SEMANTIC_SEARCH,
        strategy=RetrievalStrategy.ADAPTIVE
    )
    
    strategy = await hybrid_retriever._select_optimal_strategy(semantic_query)
    # Should be vector-only or hybrid depending on analysis
    assert strategy in [RetrievalStrategy.VECTOR_ONLY, RetrievalStrategy.HYBRID_PARALLEL]
```

### Integration Tests

Create `tests/integration/test_hybrid_retrieval_integration.py`:

```python
import pytest
import asyncio
from qdrant_client import QdrantClient
from neo4j import AsyncGraphDatabase

from morag_graph.retrieval.hybrid_retriever import (
    HybridRetriever, RetrievalQuery, RetrievalStrategy, QueryType
)
from morag_graph.retrieval.query_coordinator import QueryCoordinator

@pytest.mark.integration
@pytest.mark.asyncio
async def test_end_to_end_hybrid_retrieval():
    # Setup test databases
    qdrant_client = QdrantClient(":memory:")
    neo4j_driver = AsyncGraphDatabase.driver("bolt://localhost:7687")
    
    async with neo4j_driver.session() as session:
        # Create hybrid retriever and coordinator
        hybrid_retriever = HybridRetriever(session, qdrant_client)
        coordinator = QueryCoordinator(hybrid_retriever)
        
        # Create test data
        await session.run("""
            CREATE (p:Entity {id: 'person_1', name: 'Alice Smith', type: 'PERSON'})
            CREATE (c:Entity {id: 'company_1', name: 'TechCorp', type: 'COMPANY'})
            CREATE (d:Document {id: 'doc_1', title: 'AI Research Paper'})
            CREATE (chunk:DocumentChunk {id: 'chunk_1', text: 'Machine learning is transforming industries', document_id: 'doc_1'})
            CREATE (p)-[:WORKS_AT]->(c)
            CREATE (d)-[:HAS_CHUNK]->(chunk)
            CREATE (chunk)-[:MENTIONS]->(p)
        """)
        
        # Test different retrieval strategies
        queries = [
            RetrievalQuery(
                query_text="Find Alice Smith",
                query_type=QueryType.ENTITY_LOOKUP,
                strategy=RetrievalStrategy.ADAPTIVE
            ),
            RetrievalQuery(
                query_text="machine learning research",
                query_type=QueryType.SEMANTIC_SEARCH,
                strategy=RetrievalStrategy.ADAPTIVE,
                query_vector=[0.1] * 384  # Mock vector
            ),
            RetrievalQuery(
                query_text="companies related to Alice",
                query_type=QueryType.RELATIONSHIP_EXPLORATION,
                strategy=RetrievalStrategy.ADAPTIVE
            )
        ]
        
        # Test individual queries
        for query in queries:
            results, metrics = await coordinator.coordinate_retrieval(query)
            assert isinstance(results, list)
            assert isinstance(metrics, object)
            assert metrics.total_time > 0
        
        # Test batch coordination
        batch_results = await coordinator.batch_coordinate_retrieval(queries)
        assert len(batch_results) == len(queries)
        
        # Test caching
        cached_results, cached_metrics = await coordinator.coordinate_retrieval(queries[0])
        stats = coordinator.get_performance_stats()
        assert stats["cache"]["hits"] > 0
        
        # Cleanup
        await session.run("""
            MATCH (n) WHERE n.id IN ['person_1', 'company_1', 'doc_1', 'chunk_1']
            DETACH DELETE n
        """)
    
    await neo4j_driver.close()
```

## Performance Considerations

### Optimization Strategies

1. **Intelligent Caching**: Multi-level caching for queries and results
2. **Query Batching**: Batch similar queries for efficiency
3. **Parallel Execution**: Concurrent vector and graph operations
4. **Result Streaming**: Stream results for large result sets
5. **Adaptive Strategy Selection**: Learn optimal strategies from usage patterns

### Performance Targets

- Single query response time: < 200ms (cached), < 1s (uncached)
- Batch query throughput: > 50 queries/second
- Cache hit rate: > 70%
- Memory usage: < 2GB for 100K cached results
- Concurrent query handling: > 100 simultaneous queries

## Success Criteria

- [ ] Unified hybrid retrieval interface implemented
- [ ] All retrieval strategies working correctly
- [ ] Query coordination with caching operational
- [ ] Result fusion algorithms implemented and tested
- [ ] Performance targets met
- [ ] Comprehensive test coverage (>90%)
- [ ] Integration with existing services complete
- [ ] Monitoring and analytics functional

## Risk Assessment

**Risk Level**: High

**Key Risks**:
- Complex result fusion logic
- Performance bottlenecks with large result sets
- Cache invalidation complexity
- Strategy selection accuracy

**Mitigation Strategies**:
- Implement comprehensive testing for fusion algorithms
- Add performance monitoring and alerting
- Design simple cache invalidation strategies
- Use machine learning for strategy optimization

## Rollback Plan

1. **Immediate Rollback**: Disable hybrid retrieval, use separate systems
2. **Performance Monitoring**: Monitor system performance during rollback
3. **Data Integrity**: Ensure no data loss during rollback
4. **Service Isolation**: Revert to independent vector and graph retrieval

## Next Steps

- **Task 4.2**: [Query Coordination Service](./task-4.2-query-coordination-service.md)
- **Task 4.3**: [Performance Optimization](./task-4.3-performance-optimization.md)

## Dependencies

- **Task 3.1**: Neo4j Vector Storage (must be completed)
- **Task 3.2**: Selective Vector Strategy (must be completed)
- **Task 3.3**: Graph-Aware Vector Operations (must be completed)
- **Task 2.1**: Bidirectional Reference Storage (must be completed)
- **Task 2.2**: Metadata Synchronization (must be completed)

## Estimated Time

**Total**: 8-10 days

- Architecture design: 2 days
- Core implementation: 5 days
- Testing and optimization: 2 days
- Documentation: 1 day

## Status

- [ ] Planning
- [ ] Implementation
- [ ] Testing
- [ ] Documentation
- [ ] Review
- [ ] Deployment